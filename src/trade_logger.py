"""
Trade Logger Module for Auto-Training
======================================
Automatically records all trade data for ML model retraining and SMC optimization.

Features:
- PostgreSQL primary storage with connection pooling
- CSV fallback for offline/disconnected operation
- Trade history with full details (entry, exit, profit, duration)
- Feature snapshots at trade open/close
- SMC signal tracking and outcome analysis
- Market condition logging
- Thread-safe for concurrent access
"""

import os
import csv
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo
import threading

from loguru import logger

# Database imports
try:
    from src.db import (
        get_db,
        init_db,
        TradeRepository,
        SignalRepository,
        MarketSnapshotRepository,
        BotStatusRepository,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database module not available, using CSV fallback")

WIB = ZoneInfo("Asia/Jakarta")


@dataclass
class TradeRecord:
    """Complete trade record for analysis and retraining."""
    # Trade identifiers
    ticket: int
    symbol: str

    # Trade details
    direction: str  # BUY or SELL
    lot_size: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float

    # Results
    profit_usd: float
    profit_pips: float
    duration_seconds: int

    # Timestamps
    open_time: str
    close_time: str

    # Market conditions at ENTRY
    entry_regime: str
    entry_volatility: str
    entry_session: str
    entry_spread: float
    entry_atr: float

    # SMC Analysis at ENTRY
    smc_signal: str  # BUY, SELL, NONE
    smc_confidence: float
    smc_reason: str
    smc_fvg_detected: bool
    smc_ob_detected: bool
    smc_bos_detected: bool
    smc_choch_detected: bool

    # ML Prediction at ENTRY
    ml_signal: str  # BUY, SELL, HOLD
    ml_confidence: float

    # Dynamic threshold at ENTRY
    market_quality: str
    market_score: int
    dynamic_threshold: float

    # Exit details
    exit_reason: str  # TP_HIT, SL_HIT, REVERSAL, MANUAL, etc.
    exit_regime: str
    exit_ml_signal: str
    exit_ml_confidence: float

    # Balance tracking
    balance_before: float
    balance_after: float
    equity_at_entry: float

    # Feature snapshot (JSON string of all features)
    features_at_entry: str = ""
    features_at_exit: str = ""

    # Meta
    bot_version: str = "2.1"
    trade_mode: str = "SMC-ONLY"


@dataclass
class SignalRecord:
    """Record of every signal generated (for analysis)."""
    timestamp: str
    symbol: str
    price: float

    # Signal details
    signal_type: str  # BUY, SELL, NONE
    signal_source: str  # SMC, ML, COMBINED
    confidence: float

    # SMC details
    smc_signal: str
    smc_confidence: float
    smc_fvg: bool
    smc_ob: bool
    smc_bos: bool
    smc_reason: str

    # ML details
    ml_signal: str
    ml_confidence: float

    # Market conditions
    regime: str
    session: str
    volatility: str
    market_score: int

    # Was trade executed?
    trade_executed: bool
    execution_reason: str  # "executed", "below_threshold", "max_positions", etc.


@dataclass
class MarketSnapshot:
    """Periodic market condition snapshot."""
    timestamp: str
    symbol: str
    price: float

    # OHLC
    open: float
    high: float
    low: float
    close: float

    # Indicators
    regime: str
    volatility: str
    session: str
    atr: float
    spread: float

    # ML state
    ml_signal: str
    ml_confidence: float

    # SMC state
    smc_signal: str
    smc_confidence: float

    # Positions
    open_positions: int
    floating_pnl: float

    # Features (JSON)
    features: str = ""


class TradeLogger:
    """
    Automatic trade logger for ML retraining and analysis.

    Primary: PostgreSQL database with connection pooling
    Fallback: CSV files organized by month
    Thread-safe for concurrent access.
    """

    def __init__(self, data_dir: str = "data/trade_logs", use_db: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories for CSV fallback
        self.trades_dir = self.data_dir / "trades"
        self.signals_dir = self.data_dir / "signals"
        self.snapshots_dir = self.data_dir / "snapshots"

        for d in [self.trades_dir, self.signals_dir, self.snapshots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Thread lock for file writes
        self._lock = threading.Lock()

        # Pending trades (open positions waiting for close)
        self._pending_trades: Dict[int, Dict] = {}

        # Stats
        self._trades_logged = 0
        self._signals_logged = 0
        self._db_writes = 0
        self._csv_writes = 0

        # Database setup
        self._use_db = use_db and DB_AVAILABLE
        self._db_connected = False
        self._db = None
        self._trade_repo = None
        self._signal_repo = None
        self._snapshot_repo = None

        if self._use_db:
            self._init_database()

        logger.info(f"TradeLogger initialized: DB={self._db_connected}, CSV={self.data_dir}")

    def _init_database(self):
        """Initialize database connection and repositories."""
        try:
            if init_db():
                self._db = get_db()
                self._trade_repo = TradeRepository(self._db)
                self._signal_repo = SignalRepository(self._db)
                self._snapshot_repo = MarketSnapshotRepository(self._db)
                self._db_connected = True
                logger.info("TradeLogger: Database connected")
            else:
                logger.warning("TradeLogger: Database connection failed, using CSV")
                self._db_connected = False
        except Exception as e:
            logger.error(f"TradeLogger: Database init error: {e}")
            self._db_connected = False

    def _get_monthly_file(self, subdir: Path, prefix: str) -> Path:
        """Get file path for current month."""
        now = datetime.now(WIB)
        filename = f"{prefix}_{now.strftime('%Y_%m')}.csv"
        return subdir / filename

    def _ensure_csv_header(self, filepath: Path, fieldnames: List[str]):
        """Ensure CSV file has header row."""
        if not filepath.exists():
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    # ==================== TRADE LOGGING ====================

    def log_trade_open(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        # Market conditions
        regime: str,
        volatility: str,
        session: str,
        spread: float,
        atr: float,
        # SMC
        smc_signal: str,
        smc_confidence: float,
        smc_reason: str,
        smc_fvg: bool = False,
        smc_ob: bool = False,
        smc_bos: bool = False,
        smc_choch: bool = False,
        # ML
        ml_signal: str = "HOLD",
        ml_confidence: float = 0.5,
        # Dynamic
        market_quality: str = "moderate",
        market_score: int = 50,
        dynamic_threshold: float = 0.7,
        # Balance
        balance: float = 0,
        equity: float = 0,
        # Features
        features: Optional[Dict] = None,
    ):
        """Record trade open - stores to DB and pending dict."""
        now = datetime.now(WIB)

        # Store in pending for close tracking
        trade_data = {
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": now,
            "entry_regime": regime,
            "entry_volatility": volatility,
            "entry_session": session,
            "entry_spread": spread,
            "entry_atr": atr,
            "smc_signal": smc_signal,
            "smc_confidence": smc_confidence,
            "smc_reason": smc_reason,
            "smc_fvg_detected": smc_fvg,
            "smc_ob_detected": smc_ob,
            "smc_bos_detected": smc_bos,
            "smc_choch_detected": smc_choch,
            "ml_signal": ml_signal,
            "ml_confidence": ml_confidence,
            "market_quality": market_quality,
            "market_score": market_score,
            "dynamic_threshold": dynamic_threshold,
            "balance_before": balance,
            "equity_at_entry": equity,
            "features_entry": features or {},
        }

        self._pending_trades[ticket] = trade_data

        # Write to database
        if self._db_connected and self._trade_repo:
            try:
                self._trade_repo.insert_trade(trade_data)
                self._db_writes += 1
                logger.debug(f"TradeLogger: DB recorded open #{ticket}")
            except Exception as e:
                logger.error(f"TradeLogger: DB write failed for open #{ticket}: {e}")

        logger.debug(f"TradeLogger: Recorded open #{ticket}")

    def log_trade_close(
        self,
        ticket: int,
        exit_price: float,
        profit_usd: float,
        profit_pips: float,
        exit_reason: str,
        # Current conditions
        regime: str = "",
        ml_signal: str = "HOLD",
        ml_confidence: float = 0.5,
        balance_after: float = 0,
        # Features at exit
        features: Optional[Dict] = None,
    ):
        """Record trade close - updates DB and writes CSV."""
        now = datetime.now(WIB)

        # Get pending trade data
        pending = self._pending_trades.pop(ticket, None)

        if pending is None:
            # Trade opened before logger started - create minimal record
            logger.warning(f"TradeLogger: No pending data for #{ticket}, creating minimal record")
            pending = {
                "ticket": ticket,
                "symbol": "XAUUSD",
                "direction": "UNKNOWN",
                "lot_size": 0.01,
                "entry_price": exit_price,
                "stop_loss": 0,
                "take_profit": 0,
                "opened_at": now,
                "entry_regime": "",
                "entry_volatility": "",
                "entry_session": "",
                "entry_spread": 0,
                "entry_atr": 0,
                "smc_signal": "",
                "smc_confidence": 0,
                "smc_reason": "",
                "smc_fvg_detected": False,
                "smc_ob_detected": False,
                "smc_bos_detected": False,
                "smc_choch_detected": False,
                "ml_signal": "",
                "ml_confidence": 0,
                "market_quality": "",
                "market_score": 0,
                "dynamic_threshold": 0,
                "balance_before": balance_after - profit_usd,
                "equity_at_entry": 0,
                "features_entry": {},
            }

        # Calculate duration
        try:
            open_time = pending["opened_at"]
            if isinstance(open_time, str):
                open_time = datetime.fromisoformat(open_time)
            duration = int((now - open_time).total_seconds())
        except:
            duration = 0

        # Close data for database
        close_data = {
            "exit_price": exit_price,
            "profit_usd": profit_usd,
            "profit_pips": profit_pips,
            "closed_at": now,
            "duration_seconds": duration,
            "exit_reason": exit_reason,
            "exit_regime": regime,
            "exit_ml_signal": ml_signal,
            "exit_ml_confidence": ml_confidence,
            "balance_after": balance_after,
            "features_exit": features or {},
        }

        # Update database
        if self._db_connected and self._trade_repo:
            try:
                self._trade_repo.update_trade_close(ticket, close_data)
                self._db_writes += 1
                logger.debug(f"TradeLogger: DB recorded close #{ticket}")
            except Exception as e:
                logger.error(f"TradeLogger: DB write failed for close #{ticket}: {e}")

        # Create complete record for CSV
        open_time_str = pending["opened_at"]
        if isinstance(open_time_str, datetime):
            open_time_str = open_time_str.isoformat()

        record = TradeRecord(
            ticket=ticket,
            symbol=pending["symbol"],
            direction=pending["direction"],
            lot_size=pending["lot_size"],
            entry_price=pending["entry_price"],
            exit_price=exit_price,
            stop_loss=pending["stop_loss"],
            take_profit=pending["take_profit"],
            profit_usd=profit_usd,
            profit_pips=profit_pips,
            duration_seconds=duration,
            open_time=open_time_str,
            close_time=now.isoformat(),
            entry_regime=pending["entry_regime"],
            entry_volatility=pending["entry_volatility"],
            entry_session=pending["entry_session"],
            entry_spread=pending["entry_spread"],
            entry_atr=pending["entry_atr"],
            smc_signal=pending["smc_signal"],
            smc_confidence=pending["smc_confidence"],
            smc_reason=pending["smc_reason"],
            smc_fvg_detected=pending["smc_fvg_detected"],
            smc_ob_detected=pending["smc_ob_detected"],
            smc_bos_detected=pending["smc_bos_detected"],
            smc_choch_detected=pending["smc_choch_detected"],
            ml_signal=pending["ml_signal"],
            ml_confidence=pending["ml_confidence"],
            market_quality=pending["market_quality"],
            market_score=pending["market_score"],
            dynamic_threshold=pending["dynamic_threshold"],
            exit_reason=exit_reason,
            exit_regime=regime,
            exit_ml_signal=ml_signal,
            exit_ml_confidence=ml_confidence,
            balance_before=pending["balance_before"],
            balance_after=balance_after,
            equity_at_entry=pending["equity_at_entry"],
            features_at_entry=json.dumps(pending["features_entry"]) if isinstance(pending["features_entry"], dict) else pending["features_entry"],
            features_at_exit=json.dumps(features) if features else "{}",
        )

        # Write to CSV (always, as backup)
        self._write_trade_record(record)
        self._trades_logged += 1

        logger.info(f"TradeLogger: Saved trade #{ticket} | {record.direction} | P/L: ${profit_usd:.2f} | Reason: {exit_reason}")

    def _write_trade_record(self, record: TradeRecord):
        """Write trade record to CSV file."""
        with self._lock:
            filepath = self._get_monthly_file(self.trades_dir, "trades")
            record_dict = asdict(record)
            fieldnames = list(record_dict.keys())

            self._ensure_csv_header(filepath, fieldnames)

            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(record_dict)
            self._csv_writes += 1

    # ==================== SIGNAL LOGGING ====================

    def log_signal(
        self,
        symbol: str,
        price: float,
        signal_type: str,
        signal_source: str,
        confidence: float,
        # SMC
        smc_signal: str,
        smc_confidence: float,
        smc_fvg: bool,
        smc_ob: bool,
        smc_bos: bool,
        smc_reason: str,
        # ML
        ml_signal: str,
        ml_confidence: float,
        # Market
        regime: str,
        session: str,
        volatility: str,
        market_score: int,
        # Execution
        trade_executed: bool,
        execution_reason: str,
        # Optional
        dynamic_threshold: float = 0.7,
        trade_ticket: Optional[int] = None,
    ):
        """Log every signal generated for analysis."""
        now = datetime.now(WIB)

        # Write to database
        if self._db_connected and self._signal_repo:
            try:
                signal_data = {
                    "signal_time": now,
                    "symbol": symbol,
                    "price": price,
                    "signal_type": signal_type,
                    "signal_source": signal_source,
                    "combined_confidence": confidence,
                    "smc_signal": smc_signal,
                    "smc_confidence": smc_confidence,
                    "smc_fvg": smc_fvg,
                    "smc_ob": smc_ob,
                    "smc_bos": smc_bos,
                    "smc_choch": False,
                    "smc_reason": smc_reason,
                    "ml_signal": ml_signal,
                    "ml_confidence": ml_confidence,
                    "regime": regime,
                    "session": session,
                    "volatility": volatility,
                    "market_score": market_score,
                    "dynamic_threshold": dynamic_threshold,
                    "executed": trade_executed,
                    "execution_reason": execution_reason,
                    "trade_ticket": trade_ticket,
                }
                self._signal_repo.insert_signal(signal_data)
                self._db_writes += 1
            except Exception as e:
                logger.error(f"TradeLogger: DB signal write failed: {e}")

        # CSV record
        record = SignalRecord(
            timestamp=now.isoformat(),
            symbol=symbol,
            price=price,
            signal_type=signal_type,
            signal_source=signal_source,
            confidence=confidence,
            smc_signal=smc_signal,
            smc_confidence=smc_confidence,
            smc_fvg=smc_fvg,
            smc_ob=smc_ob,
            smc_bos=smc_bos,
            smc_reason=smc_reason,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            regime=regime,
            session=session,
            volatility=volatility,
            market_score=market_score,
            trade_executed=trade_executed,
            execution_reason=execution_reason,
        )

        self._write_signal_record(record)
        self._signals_logged += 1

    def _write_signal_record(self, record: SignalRecord):
        """Write signal record to CSV."""
        with self._lock:
            filepath = self._get_monthly_file(self.signals_dir, "signals")
            record_dict = asdict(record)
            fieldnames = list(record_dict.keys())

            self._ensure_csv_header(filepath, fieldnames)

            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(record_dict)
            self._csv_writes += 1

    # ==================== MARKET SNAPSHOTS ====================

    def log_market_snapshot(
        self,
        symbol: str,
        price: float,
        ohlc: tuple,  # (open, high, low, close)
        regime: str,
        volatility: str,
        session: str,
        atr: float,
        spread: float,
        ml_signal: str,
        ml_confidence: float,
        smc_signal: str,
        smc_confidence: float,
        open_positions: int,
        floating_pnl: float,
        features: Optional[Dict] = None,
    ):
        """Log periodic market snapshot for analysis."""
        now = datetime.now(WIB)

        # Write to database
        if self._db_connected and self._snapshot_repo:
            try:
                snapshot_data = {
                    "snapshot_time": now,
                    "symbol": symbol,
                    "price": price,
                    "open_price": ohlc[0],
                    "high_price": ohlc[1],
                    "low_price": ohlc[2],
                    "close_price": ohlc[3],
                    "regime": regime,
                    "volatility": volatility,
                    "session": session,
                    "atr": atr,
                    "spread": spread,
                    "ml_signal": ml_signal,
                    "ml_confidence": ml_confidence,
                    "smc_signal": smc_signal,
                    "smc_confidence": smc_confidence,
                    "open_positions": open_positions,
                    "floating_pnl": floating_pnl,
                    "features": features or {},
                }
                self._snapshot_repo.insert_snapshot(snapshot_data)
                self._db_writes += 1
            except Exception as e:
                logger.error(f"TradeLogger: DB snapshot write failed: {e}")

        # CSV record
        record = MarketSnapshot(
            timestamp=now.isoformat(),
            symbol=symbol,
            price=price,
            open=ohlc[0],
            high=ohlc[1],
            low=ohlc[2],
            close=ohlc[3],
            regime=regime,
            volatility=volatility,
            session=session,
            atr=atr,
            spread=spread,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            smc_signal=smc_signal,
            smc_confidence=smc_confidence,
            open_positions=open_positions,
            floating_pnl=floating_pnl,
            features=json.dumps(features) if features else "{}",
        )

        self._write_snapshot_record(record)

    def _write_snapshot_record(self, record: MarketSnapshot):
        """Write snapshot to CSV."""
        with self._lock:
            filepath = self._get_monthly_file(self.snapshots_dir, "snapshots")
            record_dict = asdict(record)
            fieldnames = list(record_dict.keys())

            self._ensure_csv_header(filepath, fieldnames)

            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(record_dict)
            self._csv_writes += 1

    # ==================== ANALYSIS HELPERS ====================

    def get_stats(self) -> Dict:
        """Get logger statistics."""
        return {
            "trades_logged": self._trades_logged,
            "signals_logged": self._signals_logged,
            "pending_trades": len(self._pending_trades),
            "db_connected": self._db_connected,
            "db_writes": self._db_writes,
            "csv_writes": self._csv_writes,
            "data_dir": str(self.data_dir),
        }

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades - from DB if connected, else CSV."""
        # Try database first
        if self._db_connected and self._trade_repo:
            try:
                return self._trade_repo.get_recent_trades(limit)
            except Exception as e:
                logger.error(f"TradeLogger: DB query failed: {e}")

        # Fallback to CSV
        filepath = self._get_monthly_file(self.trades_dir, "trades")

        if not filepath.exists():
            return []

        trades = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)

        return trades[-limit:]

    def get_win_rate(self, days: int = 30) -> Dict:
        """Calculate win rate from logged trades."""
        # Try database first
        if self._db_connected and self._trade_repo:
            try:
                trades = self._trade_repo.get_trades_for_training(days)
                if trades:
                    wins = sum(1 for t in trades if t.get("profit_usd", 0) > 0)
                    losses = sum(1 for t in trades if t.get("profit_usd", 0) < 0)
                    total = wins + losses
                    total_profit = sum(t.get("profit_usd", 0) for t in trades)
                    win_rate = (wins / total * 100) if total > 0 else 0

                    return {
                        "total": total,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": win_rate,
                        "total_profit": total_profit,
                        "source": "database",
                    }
            except Exception as e:
                logger.error(f"TradeLogger: DB win rate query failed: {e}")

        # Fallback to CSV
        filepath = self._get_monthly_file(self.trades_dir, "trades")

        if not filepath.exists():
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "source": "csv"}

        wins = 0
        losses = 0
        total_profit = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profit = float(row.get("profit_usd", 0))
                total_profit += profit
                if profit > 0:
                    wins += 1
                elif profit < 0:
                    losses += 1

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "source": "csv",
        }

    def get_smc_performance(self, days: int = 30) -> Dict:
        """Analyze SMC signal performance."""
        # Try database first
        if self._db_connected and self._trade_repo:
            try:
                return self._trade_repo.get_smc_pattern_stats(days)
            except Exception as e:
                logger.error(f"TradeLogger: DB SMC query failed: {e}")

        # Fallback to CSV
        filepath = self._get_monthly_file(self.trades_dir, "trades")

        if not filepath.exists():
            return {}

        stats = {
            "fvg_trades": {"wins": 0, "losses": 0, "profit": 0},
            "ob_trades": {"wins": 0, "losses": 0, "profit": 0},
            "bos_trades": {"wins": 0, "losses": 0, "profit": 0},
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profit = float(row.get("profit_usd", 0))
                is_win = profit > 0

                if row.get("smc_fvg_detected") == "True":
                    stats["fvg_trades"]["wins" if is_win else "losses"] += 1
                    stats["fvg_trades"]["profit"] += profit

                if row.get("smc_ob_detected") == "True":
                    stats["ob_trades"]["wins" if is_win else "losses"] += 1
                    stats["ob_trades"]["profit"] += profit

                if row.get("smc_bos_detected") == "True":
                    stats["bos_trades"]["wins" if is_win else "losses"] += 1
                    stats["bos_trades"]["profit"] += profit

        return stats

    def get_trades_for_training(self, days: int = 30) -> List[Dict]:
        """Get trades suitable for ML training."""
        if self._db_connected and self._trade_repo:
            try:
                return self._trade_repo.get_trades_for_training(days)
            except Exception as e:
                logger.error(f"TradeLogger: DB training data query failed: {e}")

        # Fallback to CSV - return all trades from file
        return self.get_recent_trades(limit=1000)


# Global instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


if __name__ == "__main__":
    # Test the logger
    tlogger = get_trade_logger()

    print(f"DB Connected: {tlogger._db_connected}")
    print(f"Stats: {tlogger.get_stats()}")

    # Test trade open
    tlogger.log_trade_open(
        ticket=12345,
        symbol="XAUUSD",
        direction="SELL",
        lot_size=0.01,
        entry_price=4850.00,
        stop_loss=4900.00,
        take_profit=4800.00,
        regime="high_volatility",
        volatility="high",
        session="London",
        spread=2.5,
        atr=15.0,
        smc_signal="SELL",
        smc_confidence=0.75,
        smc_reason="Bearish BOS + FVG",
        smc_fvg=True,
        smc_bos=True,
        ml_signal="HOLD",
        ml_confidence=0.55,
        market_quality="good",
        market_score=65,
        dynamic_threshold=0.65,
        balance=5500,
        equity=5500,
        features={"rsi": 45, "macd": -0.5},
    )

    # Test trade close
    tlogger.log_trade_close(
        ticket=12345,
        exit_price=4820.00,
        profit_usd=30.00,
        profit_pips=300,
        exit_reason="TP_HIT",
        regime="high_volatility",
        ml_signal="HOLD",
        ml_confidence=0.52,
        balance_after=5530,
        features={"rsi": 55, "macd": 0.2},
    )

    print(f"Stats: {tlogger.get_stats()}")
    print(f"Win Rate: {tlogger.get_win_rate()}")
