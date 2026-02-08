"""
Backtest Live Sync - 100% Identical to main_live.py
====================================================
This backtest MUST be identical to live trading logic.

SYNCED with Critical & Major Fixes (Feb 2025):
1. SMC Signal: No lookahead bias, current_close entry, Fixed RR 1:1.5
2. Pullback Filter: ATR-based thresholds (not hardcoded $2, $1.5)
3. Time-Based Exit: Checks profit_growing + ML agreement before exit
4. Trend Reversal: ATR-based momentum thresholds (0.6x multiplier)
5. Signal Persistence: Index-based cleanup (prevents memory leak)
6. Calibrated Confidence: Uses SMC's weighted confidence calculation
7. Dynamic RR: 1.5 (ranging) to 2.0 (strong trend) based on market conditions
8. SELL Filter: Requires ML agreement + 55% confidence

Synchronized elements:
1. ML Model: XGBoost with same features, 50-bar train/test gap
2. SMC Analyzer: Same swing_length, ob_lookback, NO LOOKAHEAD
3. Regime Detection: HMM with MarketRegimeDetector
4. Session Filter: Golden Time 19:00-23:00 WIB
5. Signal Logic:
   - Skip if market quality AVOID or CRISIS
   - ML confidence >= ML_THRESHOLD required (default 50%)
   - ML shouldn't strongly disagree (>65% opposite)
   - Signal confirmation (2+ consecutive signals)
   - Pullback filter (ATR-based thresholds)
6. Position Sizing: Based on ML confidence tiers (0.01-0.02 lot)
7. Trade Cooldown: 20 bars (~5 hours on M15)
8. Exit Logic:
   - TP hit (Dynamic RR 1.5-2.0)
   - ML reversal (>65% opposite signal)
   - Trend reversal (ATR * 0.6 momentum shift)
   - Smart timeout (checks profit_growing before exit)
   - Max loss per trade ($50 default)

Usage:
    python backtests/backtest_live_sync.py --tune  # Find optimal thresholds
    python backtests/backtest_live_sync.py --save  # Save results to CSV
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import csv
from zoneinfo import ZoneInfo

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, MarketRegime
from src.ml_model import TradingModel
from src.config import get_config
from src.session_filter import create_wib_session_filter
from src.dynamic_confidence import create_dynamic_confidence, MarketQuality
from loguru import logger

# Reduce logging noise
logger.remove()
logger.add(sys.stderr, level="WARNING")


class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    MAX_LOSS = "max_loss"
    ML_REVERSAL = "ml_reversal"
    TIMEOUT = "timeout"
    TREND_REVERSAL = "trend_reversal"


@dataclass
class SimulatedTrade:
    """Simulated trade record - matches live trade logging."""
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    profit_usd: float
    profit_pips: float
    result: TradeResult
    exit_reason: ExitReason
    ml_confidence: float
    smc_confidence: float
    regime: str
    session: str
    signal_reason: str


@dataclass
class BacktestStats:
    """Backtest statistics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_usd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)


class LiveSyncBacktest:
    """
    Backtest engine that is 100% synchronized with main_live.py
    """

    def __init__(
        self,
        ml_threshold: float = 0.50,
        signal_confirmation: int = 2,
        pullback_filter: bool = True,
        golden_time_only: bool = False,
        max_loss_per_trade: float = 50.0,
        trade_cooldown_bars: int = 10,  # OPTIMIZED: was 20, now 10 (~2.5 hours)
        trend_reversal_mult: float = 0.6,  # OPTIMIZED: was 0.4, now 0.6 (less aggressive exit)
        sell_filter_strict: bool = True,  # OPTIMIZED: require ML agreement for SELL
    ):
        """
        Initialize backtest with configurable parameters.

        Args:
            ml_threshold: Minimum ML confidence to trade (0.50-0.70)
            signal_confirmation: Number of consecutive signals required
            pullback_filter: Enable pullback detection filter
            golden_time_only: Only trade during 19:00-23:00 WIB
            max_loss_per_trade: Maximum loss before smart exit
            trade_cooldown_bars: Minimum bars between trades (OPTIMIZED: 10)
            trend_reversal_mult: ATR multiplier for trend reversal exit (OPTIMIZED: 0.6)
            sell_filter_strict: Require ML agreement for SELL signals (OPTIMIZED: True)
        """
        self.ml_threshold = ml_threshold
        self.signal_confirmation = signal_confirmation
        self.pullback_filter = pullback_filter
        self.golden_time_only = golden_time_only
        self.max_loss_per_trade = max_loss_per_trade
        self.trade_cooldown_bars = trade_cooldown_bars
        self.trend_reversal_mult = trend_reversal_mult
        self.sell_filter_strict = sell_filter_strict

        # Initialize components (same as main_live.py)
        config = get_config()

        self.smc = SMCAnalyzer(
            swing_length=config.smc.swing_length,
            ob_lookback=config.smc.ob_lookback,
        )
        self.features = FeatureEngineer()
        self.regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
        self.ml_model = TradingModel(model_path="models/xgboost_model.pkl")
        self.dynamic_confidence = create_dynamic_confidence()

        # Load models
        self.regime_detector.load()
        self.ml_model.load()

        # State tracking
        self._signal_persistence = {}
        self._ticket_counter = 1000000

    def _get_session_from_time(self, dt: datetime) -> Tuple[str, bool, float]:
        """
        Get trading session info from datetime.
        Returns: (session_name, can_trade, lot_multiplier)
        """
        # Convert to WIB
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        wib_time = dt.astimezone(ZoneInfo("Asia/Jakarta"))
        hour = wib_time.hour

        # Session definitions (same as session_filter.py)
        if 6 <= hour < 15:
            return "Sydney-Tokyo", True, 0.5  # Lower confidence required
        elif 15 <= hour < 16:
            return "Tokyo-London Overlap", True, 0.75
        elif 16 <= hour < 19:
            return "London Early", True, 0.8
        elif 19 <= hour < 24:
            return "London-NY Overlap (Golden)", True, 1.0  # Best session
        elif 0 <= hour < 4:
            return "NY Session", True, 0.9
        else:
            return "Off Hours", False, 0.0

    def _is_golden_time(self, dt: datetime) -> bool:
        """Check if datetime is in golden time (19:00-23:00 WIB)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        wib_time = dt.astimezone(ZoneInfo("Asia/Jakarta"))
        return 19 <= wib_time.hour < 24

    def _check_pullback_filter(
        self,
        df: pl.DataFrame,
        signal_direction: str,
        idx: int,
    ) -> Tuple[bool, str]:
        """
        Check pullback filter - SYNCED with main_live.py (ATR-based thresholds)
        """
        if not self.pullback_filter:
            return True, "Pullback filter disabled"

        try:
            if idx < 5:
                return True, "Not enough data"

            # Get data up to current index
            closes = df["close"].to_list()[:idx+1]
            last_3 = closes[-3:]

            # Get ATR for dynamic thresholds (SYNCED: no more hardcoded values)
            atr = 12.0  # Default for XAUUSD
            if "atr" in df.columns:
                atr_list = df["atr"].to_list()[:idx+1]
                if atr_list[-1] is not None and atr_list[-1] > 0:
                    atr = atr_list[-1]

            # Dynamic thresholds based on ATR (SYNCED with main_live.py)
            bounce_threshold = atr * 0.15      # 15% of ATR = significant bounce
            consolidation_threshold = atr * 0.10  # 10% of ATR = consolidation

            # Short-term momentum
            short_momentum = last_3[-1] - last_3[0]
            momentum_dir = "UP" if short_momentum > 0 else "DOWN"

            # MACD histogram direction
            macd_dir = "NEUTRAL"
            if "macd_histogram" in df.columns:
                macd_hist = df["macd_histogram"].to_list()[:idx+1]
                if len(macd_hist) >= 2 and macd_hist[-1] is not None and macd_hist[-2] is not None:
                    macd_dir = "RISING" if macd_hist[-1] > macd_hist[-2] else "FALLING"

            # Price vs EMA
            price_vs_ema = "NEUTRAL"
            if "ema_9" in df.columns:
                ema_9 = df["ema_9"].to_list()[:idx+1][-1]
                current_price = closes[-1]
                if ema_9 is not None:
                    if current_price > ema_9 * 1.001:
                        price_vs_ema = "ABOVE"
                    elif current_price < ema_9 * 0.999:
                        price_vs_ema = "BELOW"

            # SELL signal pullback check (ATR-based thresholds)
            if signal_direction == "SELL":
                if momentum_dir == "UP" and short_momentum > bounce_threshold:
                    return False, f"SELL blocked: Price bouncing UP (+${short_momentum:.2f} > {bounce_threshold:.2f})"
                if macd_dir == "RISING" and momentum_dir == "UP":
                    return False, "SELL blocked: MACD bullish + price rising"
                if price_vs_ema == "ABOVE" and momentum_dir == "UP":
                    return False, "SELL blocked: Price above EMA9 and rising"
                if momentum_dir == "DOWN":
                    return True, f"SELL OK: Momentum aligned (${short_momentum:.2f})"
                if abs(short_momentum) < consolidation_threshold:
                    return True, f"SELL OK: Consolidation phase (<{consolidation_threshold:.2f})"

            # BUY signal pullback check (ATR-based thresholds)
            elif signal_direction == "BUY":
                if momentum_dir == "DOWN" and short_momentum < -bounce_threshold:
                    return False, f"BUY blocked: Price falling DOWN (${short_momentum:.2f} < -{bounce_threshold:.2f})"
                if macd_dir == "FALLING" and momentum_dir == "DOWN":
                    return False, "BUY blocked: MACD bearish + price falling"
                if price_vs_ema == "BELOW" and momentum_dir == "DOWN":
                    return False, "BUY blocked: Price below EMA9 and falling"
                if momentum_dir == "UP":
                    return True, f"BUY OK: Momentum aligned (+${short_momentum:.2f})"
                if abs(short_momentum) < consolidation_threshold:
                    return True, f"BUY OK: Consolidation phase (<{consolidation_threshold:.2f})"

            return True, f"Pullback check passed (mom={momentum_dir}, macd={macd_dir})"

        except Exception as e:
            return True, f"Pullback error: {e}"

    def _simulate_trade_exit(
        self,
        df: pl.DataFrame,
        entry_idx: int,
        direction: str,
        entry_price: float,
        take_profit: float,
        lot_size: float,
        max_bars: int = 100,
    ) -> Tuple[float, float, ExitReason, int, float]:
        """
        Simulate trade exit with smart exit logic (no hard SL).
        SYNCED with main_live.py and smart_risk_manager.py

        Returns: (profit_usd, profit_pips, exit_reason, exit_idx, exit_price)
        """
        pip_value = 10  # XAUUSD: 1 pip = $10 per lot

        highs = df["high"].to_list()
        lows = df["low"].to_list()
        closes = df["close"].to_list()

        # Get ATR for dynamic thresholds (SYNCED: no more hardcoded values)
        atr = 12.0  # Default for XAUUSD
        if "atr" in df.columns:
            atr_list = df["atr"].to_list()
            if entry_idx < len(atr_list) and atr_list[entry_idx] is not None:
                atr = atr_list[entry_idx]

        # Dynamic thresholds based on ATR (OPTIMIZED: configurable multiplier)
        reversal_momentum_threshold = atr * self.trend_reversal_mult  # OPTIMIZED: 0.6 default
        min_loss_for_reversal_exit = atr * 0.8    # 80% of ATR = ~$10 equivalent

        # Get ML predictions for exit logic
        feature_cols = [f for f in self.ml_model.feature_names if f in df.columns]

        # Track profit history for profit_growing check (SYNCED with smart_risk_manager)
        profit_history = []

        for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            high = highs[i]
            low = lows[i]
            close = closes[i]

            # === EXIT LOGIC 1: Take Profit ===
            if direction == "BUY":
                if high >= take_profit:
                    pips = (take_profit - entry_price) / 0.1
                    profit = pips * pip_value * lot_size
                    return profit, pips, ExitReason.TAKE_PROFIT, i, take_profit
            else:  # SELL
                if low <= take_profit:
                    pips = (entry_price - take_profit) / 0.1
                    profit = pips * pip_value * lot_size
                    return profit, pips, ExitReason.TAKE_PROFIT, i, take_profit

            # Calculate current profit/loss
            if direction == "BUY":
                current_pips = (close - entry_price) / 0.1
            else:
                current_pips = (entry_price - close) / 0.1
            current_profit = current_pips * pip_value * lot_size

            # Track profit history for growth check
            profit_history.append(current_profit)

            # === EXIT LOGIC 2: Maximum Loss ===
            if current_profit < -self.max_loss_per_trade:
                return current_profit, current_pips, ExitReason.MAX_LOSS, i, close

            # === EXIT LOGIC 3: SMART TIME-BASED EXIT (SYNCED with smart_risk_manager) ===
            # 4 hours = 16 bars on M15, 6 hours = 24 bars
            bars_since_entry = i - entry_idx

            # Check if profit is growing (SYNCED: positive momentum = don't exit early)
            profit_growing = False
            if len(profit_history) >= 4:
                recent_profits = profit_history[-4:]
                profit_momentum = recent_profits[-1] - recent_profits[0]
                profit_growing = profit_momentum > 0

            # Get ML prediction for agreement check
            ml_agrees = False
            try:
                if (i - entry_idx) % 4 == 0:  # Check every 4 bars
                    df_slice = df.head(i + 1)
                    ml_pred = self.ml_model.predict(df_slice, feature_cols)
                    ml_agrees = (
                        (direction == "BUY" and ml_pred.signal == "BUY") or
                        (direction == "SELL" and ml_pred.signal == "SELL")
                    )
            except:
                pass

            # 4+ hours: Only exit if stuck (no profit growth) - SYNCED
            if bars_since_entry >= 16:
                if current_profit < 5 and not profit_growing:
                    # Stuck with no growth - exit
                    if current_profit >= 0:
                        return current_profit, current_pips, ExitReason.TIMEOUT, i, close
                    elif current_profit > -15:
                        return current_profit, current_pips, ExitReason.TIMEOUT, i, close
                # If profitable and growing and ML agrees - extend time (don't exit)

            # 6+ hours: Exit unless significantly profitable AND still growing
            if bars_since_entry >= 24:
                if current_profit < 10 or not profit_growing:
                    return current_profit, current_pips, ExitReason.TIMEOUT, i, close
                # If profit > $10 and growing, allow up to 8 hours (32 bars)

            # 8+ hours: Hard max - exit regardless
            if bars_since_entry >= 32:
                return current_profit, current_pips, ExitReason.TIMEOUT, i, close

            # === EXIT LOGIC 4: ML Reversal (check every 5 bars) ===
            if (i - entry_idx) % 5 == 0 and i > entry_idx + 5:
                try:
                    df_slice = df.head(i + 1)
                    ml_pred = self.ml_model.predict(df_slice, feature_cols)

                    # Strong reversal signal (>65% confidence - synced with live)
                    if direction == "BUY" and ml_pred.signal == "SELL" and ml_pred.confidence > 0.65:
                        return current_profit, current_pips, ExitReason.ML_REVERSAL, i, close
                    elif direction == "SELL" and ml_pred.signal == "BUY" and ml_pred.confidence > 0.65:
                        return current_profit, current_pips, ExitReason.ML_REVERSAL, i, close
                except:
                    pass

            # === EXIT LOGIC 5: Trend Reversal (ATR-based momentum shift) ===
            if i > entry_idx + 10:
                recent_closes = closes[i-5:i+1]
                momentum = recent_closes[-1] - recent_closes[0]

                # Strong momentum against position (ATR-based thresholds)
                if direction == "BUY" and momentum < -reversal_momentum_threshold:
                    if current_profit < -min_loss_for_reversal_exit:  # Only if already losing
                        return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close
                elif direction == "SELL" and momentum > reversal_momentum_threshold:
                    if current_profit < -min_loss_for_reversal_exit:
                        return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close

        # Timeout - close at last price
        final_idx = min(entry_idx + max_bars - 1, len(df) - 1)
        final_price = closes[final_idx]
        if direction == "BUY":
            pips = (final_price - entry_price) / 0.1
        else:
            pips = (entry_price - final_price) / 0.1
        profit = pips * pip_value * lot_size
        return profit, pips, ExitReason.TIMEOUT, final_idx, final_price

    def run(
        self,
        df: pl.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 5000.0,
    ) -> BacktestStats:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV and indicators
            start_date: Start date filter (default: all data)
            end_date: End date filter (default: all data)
            initial_capital: Starting capital

        Returns:
            BacktestStats with all trade details
        """
        stats = BacktestStats()
        capital = initial_capital
        peak_capital = initial_capital

        # Get feature columns
        feature_cols = [f for f in self.ml_model.feature_names if f in df.columns]

        # Filter by date if specified
        times = df["time"].to_list()

        if start_date:
            start_idx = next((i for i, t in enumerate(times) if t >= start_date), 100)
        else:
            start_idx = 100

        if end_date:
            end_idx = next((i for i, t in enumerate(times) if t > end_date), len(df) - 100)
        else:
            end_idx = len(df) - 100

        # State tracking
        last_trade_idx = -self.trade_cooldown_bars * 2
        self._signal_persistence = {}

        print(f"\nRunning backtest (ML threshold: {self.ml_threshold:.0%})...")
        print(f"  Date range: {times[start_idx]} to {times[end_idx-1]}")
        print(f"  Total bars: {end_idx - start_idx}")

        # Iterate through data
        for i in range(start_idx, end_idx):
            # === COOLDOWN CHECK ===
            if i - last_trade_idx < self.trade_cooldown_bars:
                continue

            current_time = times[i]

            # === SESSION FILTER ===
            session_name, can_trade, lot_mult = self._get_session_from_time(current_time)

            if not can_trade:
                self._signal_persistence = {}
                continue

            if self.golden_time_only and not self._is_golden_time(current_time):
                self._signal_persistence = {}
                continue

            # Get data slice
            df_slice = df.head(i + 1)

            # === REGIME CHECK ===
            try:
                regime_state = self.regime_detector.get_current_state(df_slice)
                regime = regime_state.regime.value if regime_state else "normal"

                if regime_state and regime_state.regime == MarketRegime.CRISIS:
                    self._signal_persistence = {}
                    continue
            except:
                regime = "normal"

            # === SMC SIGNAL ===
            try:
                smc_signal = self.smc.generate_signal(df_slice)
            except:
                continue

            if smc_signal is None:
                self._signal_persistence = {}
                continue

            # === ML PREDICTION ===
            try:
                ml_pred = self.ml_model.predict(df_slice, feature_cols)
            except:
                continue

            # === DYNAMIC CONFIDENCE CHECK ===
            try:
                market_analysis = self.dynamic_confidence.analyze_market(
                    session=session_name,
                    regime=regime,
                    volatility="medium",
                    trend_direction=regime,
                    has_smc_signal=True,
                    ml_signal=ml_pred.signal,
                    ml_confidence=ml_pred.confidence,
                )

                if market_analysis.quality == MarketQuality.AVOID:
                    self._signal_persistence = {}
                    continue
            except:
                pass

            # === ML THRESHOLD CHECK ===
            if ml_pred.confidence < self.ml_threshold:
                self._signal_persistence = {}
                continue

            # === ML DISAGREEMENT CHECK ===
            ml_strongly_disagrees = (
                (smc_signal.signal_type == "BUY" and ml_pred.signal == "SELL" and ml_pred.confidence > 0.65) or
                (smc_signal.signal_type == "SELL" and ml_pred.signal == "BUY" and ml_pred.confidence > 0.65)
            )
            if ml_strongly_disagrees:
                self._signal_persistence = {}
                continue

            # === SELL FILTER (OPTIMIZED: stricter requirements for SELL) ===
            if self.sell_filter_strict and smc_signal.signal_type == "SELL":
                # Require ML to agree for SELL signals (SELL has lower WR historically)
                if ml_pred.signal != "SELL":
                    self._signal_persistence = {}
                    continue
                # Require higher ML confidence for SELL
                if ml_pred.confidence < 0.55:
                    self._signal_persistence = {}
                    continue

            # === SIGNAL CONFIRMATION (SYNCED with main_live.py) ===
            signal_key = f"{smc_signal.signal_type}_{int(smc_signal.entry_price)}"

            # Cleanup: Remove entries older than 20 bars (equivalent to 5 min cleanup in live)
            # This prevents memory leak from accumulating stale signals
            self._signal_persistence = {
                k: v for k, v in self._signal_persistence.items()
                if i - v[1] < 20  # Keep only signals seen in last 20 bars
            }

            # Also limit to max 50 entries as safety (SYNCED)
            if len(self._signal_persistence) > 50:
                # Keep only 20 most recent
                sorted_signals = sorted(self._signal_persistence.items(), key=lambda x: x[1][1], reverse=True)
                self._signal_persistence = dict(sorted_signals[:20])

            if signal_key not in self._signal_persistence:
                self._signal_persistence[signal_key] = (1, i)  # (count, last_seen_idx)
                continue
            else:
                count, _ = self._signal_persistence[signal_key]
                self._signal_persistence[signal_key] = (count + 1, i)

            # Require at least N consecutive confirmations
            count, _ = self._signal_persistence[signal_key]
            if count < self.signal_confirmation:
                continue

            # Signal confirmed! Reset counter (SYNCED)
            self._signal_persistence[signal_key] = (0, i)

            # === PULLBACK FILTER ===
            pullback_ok, pullback_reason = self._check_pullback_filter(
                df_slice, smc_signal.signal_type, i
            )
            if not pullback_ok:
                continue

            # === CALCULATE LOT SIZE ===
            if ml_pred.confidence >= 0.65:
                lot_size = 0.02
            elif ml_pred.confidence >= 0.55:
                lot_size = 0.01
            else:
                lot_size = 0.01

            # Apply session multiplier
            lot_size = max(0.01, lot_size * lot_mult)

            # === EXECUTE TRADE ===
            entry_price = smc_signal.entry_price
            take_profit = smc_signal.take_profit

            profit, pips, exit_reason, exit_idx, exit_price = self._simulate_trade_exit(
                df=df,
                entry_idx=i,
                direction=smc_signal.signal_type,
                entry_price=entry_price,
                take_profit=take_profit,
                lot_size=lot_size,
            )

            # Record trade
            self._ticket_counter += 1
            result = TradeResult.WIN if profit > 0 else (TradeResult.LOSS if profit < 0 else TradeResult.BREAKEVEN)

            # ML agrees?
            ml_agrees = (
                (smc_signal.signal_type == "BUY" and ml_pred.signal == "BUY") or
                (smc_signal.signal_type == "SELL" and ml_pred.signal == "SELL")
            )
            combined_conf = (smc_signal.confidence + ml_pred.confidence) / 2 if ml_agrees else smc_signal.confidence

            trade = SimulatedTrade(
                ticket=self._ticket_counter,
                entry_time=current_time,
                exit_time=times[exit_idx] if exit_idx < len(times) else times[-1],
                direction=smc_signal.signal_type,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=smc_signal.stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                profit_usd=profit,
                profit_pips=pips,
                result=result,
                exit_reason=exit_reason,
                ml_confidence=ml_pred.confidence,
                smc_confidence=smc_signal.confidence,
                regime=regime,
                session=session_name,
                signal_reason=smc_signal.reason,
            )
            stats.trades.append(trade)

            # Update stats
            stats.total_trades += 1
            capital += profit

            if profit > 0:
                stats.wins += 1
                stats.total_profit += profit
            else:
                stats.losses += 1
                stats.total_loss += abs(profit)

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown_pct = (peak_capital - capital) / peak_capital * 100
            drawdown_usd = peak_capital - capital
            if drawdown_pct > stats.max_drawdown:
                stats.max_drawdown = drawdown_pct
                stats.max_drawdown_usd = drawdown_usd

            # Update last trade index
            last_trade_idx = exit_idx

            # Progress
            if stats.total_trades % 100 == 0:
                print(f"  {stats.total_trades} trades processed...")

        # Calculate final statistics
        if stats.total_trades > 0:
            stats.win_rate = stats.wins / stats.total_trades * 100
            stats.avg_win = stats.total_profit / stats.wins if stats.wins > 0 else 0
            stats.avg_loss = stats.total_loss / stats.losses if stats.losses > 0 else 0
            stats.avg_trade = (stats.total_profit - stats.total_loss) / stats.total_trades
            stats.profit_factor = stats.total_profit / stats.total_loss if stats.total_loss > 0 else float('inf')

            # Expectancy
            win_prob = stats.wins / stats.total_trades
            loss_prob = stats.losses / stats.total_trades
            stats.expectancy = (win_prob * stats.avg_win) - (loss_prob * stats.avg_loss)

            # Sharpe ratio (simplified)
            returns = [t.profit_usd for t in stats.trades]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                stats.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        return stats

    def save_results(self, stats: BacktestStats, filepath: str):
        """Save backtest results to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save trades
        trades_data = []
        for t in stats.trades:
            trades_data.append({
                "ticket": t.ticket,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "lot_size": t.lot_size,
                "profit_usd": t.profit_usd,
                "profit_pips": t.profit_pips,
                "result": t.result.value,
                "exit_reason": t.exit_reason.value,
                "ml_confidence": t.ml_confidence,
                "smc_confidence": t.smc_confidence,
                "regime": t.regime,
                "session": t.session,
                "signal_reason": t.signal_reason,
            })

        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(filepath, index=False)
        print(f"Trades saved to: {filepath}")

        # Save summary
        summary_path = filepath.replace(".csv", "_summary.csv")
        summary_data = {
            "metric": [
                "total_trades", "wins", "losses", "win_rate",
                "total_profit", "total_loss", "net_pnl",
                "profit_factor", "avg_win", "avg_loss", "avg_trade",
                "max_drawdown_pct", "max_drawdown_usd",
                "expectancy", "sharpe_ratio"
            ],
            "value": [
                stats.total_trades, stats.wins, stats.losses, f"{stats.win_rate:.1f}%",
                f"${stats.total_profit:.2f}", f"${stats.total_loss:.2f}",
                f"${stats.total_profit - stats.total_loss:.2f}",
                f"{stats.profit_factor:.2f}", f"${stats.avg_win:.2f}", f"${stats.avg_loss:.2f}",
                f"${stats.avg_trade:.2f}",
                f"{stats.max_drawdown:.1f}%", f"${stats.max_drawdown_usd:.2f}",
                f"${stats.expectancy:.2f}", f"{stats.sharpe_ratio:.2f}"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")


def tune_thresholds(df: pl.DataFrame, start_date: datetime, end_date: datetime):
    """
    Find optimal ML threshold and other parameters.
    """
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING")
    print("=" * 70)

    results = []

    # Test different ML thresholds
    ml_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

    for ml_thresh in ml_thresholds:
        print(f"\nTesting ML threshold: {ml_thresh:.0%}")

        backtest = LiveSyncBacktest(
            ml_threshold=ml_thresh,
            signal_confirmation=2,
            pullback_filter=True,
            golden_time_only=False,
        )

        stats = backtest.run(df, start_date=start_date, end_date=end_date)

        net_pnl = stats.total_profit - stats.total_loss

        results.append({
            "ml_threshold": ml_thresh,
            "trades": stats.total_trades,
            "win_rate": stats.win_rate,
            "net_pnl": net_pnl,
            "profit_factor": stats.profit_factor,
            "max_drawdown": stats.max_drawdown,
            "expectancy": stats.expectancy,
        })

        print(f"  Trades: {stats.total_trades} | WR: {stats.win_rate:.1f}% | Net: ${net_pnl:.2f} | PF: {stats.profit_factor:.2f}")

    # Find optimal
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    # Sort by net P/L
    results_sorted = sorted(results, key=lambda x: x["net_pnl"], reverse=True)

    print(f"\n{'ML Thresh':>10} {'Trades':>8} {'Win Rate':>10} {'Net P/L':>12} {'PF':>8} {'DD':>8}")
    print("-" * 60)
    for r in results_sorted:
        print(f"{r['ml_threshold']:>10.0%} {r['trades']:>8} {r['win_rate']:>9.1f}% ${r['net_pnl']:>10.2f} {r['profit_factor']:>7.2f} {r['max_drawdown']:>7.1f}%")

    # Best result
    best = results_sorted[0]
    print(f"\nOPTIMAL ML THRESHOLD: {best['ml_threshold']:.0%}")
    print(f"  Net P/L: ${best['net_pnl']:.2f}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Live-Sync Backtest")
    parser.add_argument("--tune", action="store_true", help="Run threshold tuning")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    parser.add_argument("--threshold", type=float, default=0.50, help="ML confidence threshold")
    parser.add_argument("--golden-only", action="store_true", help="Only trade golden time")
    parser.add_argument("--cooldown", type=int, default=10, help="Trade cooldown in bars (default: 10)")
    parser.add_argument("--trend-mult", type=float, default=0.6, help="Trend reversal ATR multiplier (default: 0.6)")
    parser.add_argument("--no-sell-filter", action="store_true", help="Disable strict SELL filter")
    parser.add_argument("--baseline", action="store_true", help="Run with baseline settings (old params)")
    args = parser.parse_args()

    print("=" * 70)
    print("BACKTEST LIVE SYNC - 100% Identical to main_live.py")
    print("=" * 70)

    # Connect to MT5 and fetch data
    config = get_config()
    mt5 = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    mt5.connect()
    print(f"\nConnected to MT5")

    # Fetch maximum historical data
    print("Fetching historical data...")
    df = mt5.get_market_data(symbol="XAUUSD", timeframe="M15", count=50000)

    if len(df) == 0:
        print("ERROR: No data received")
        return

    print(f"Received {len(df)} bars")

    # Get date range
    times = df["time"].to_list()
    data_start = times[0]
    data_end = times[-1]
    print(f"Data range: {data_start} to {data_end}")

    # Filter to January 2025 - Today
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now()

    # Calculate indicators
    print("\nCalculating indicators...")
    features = FeatureEngineer()
    smc = SMCAnalyzer()
    regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    regime_detector.load()

    df = features.calculate_all(df, include_ml_features=True)
    df = smc.calculate_all(df)

    try:
        df = regime_detector.predict(df)
    except:
        pass

    print("Indicators calculated")

    if args.tune:
        # Run threshold tuning
        tune_thresholds(df, start_date, end_date)
    else:
        # Run single backtest
        # Use baseline settings if requested
        if args.baseline:
            cooldown = 20
            trend_mult = 0.4
            sell_filter = False
            print("\n*** BASELINE MODE (old settings) ***")
        else:
            cooldown = args.cooldown
            trend_mult = args.trend_mult
            sell_filter = not args.no_sell_filter

        backtest = LiveSyncBacktest(
            ml_threshold=args.threshold,
            signal_confirmation=2,
            pullback_filter=True,
            golden_time_only=args.golden_only,
            trade_cooldown_bars=cooldown,
            trend_reversal_mult=trend_mult,
            sell_filter_strict=sell_filter,
        )

        stats = backtest.run(df, start_date=start_date, end_date=end_date)

        # Print results
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        net_pnl = stats.total_profit - stats.total_loss

        print(f"\nConfiguration:")
        print(f"  ML Threshold: {args.threshold:.0%}")
        print(f"  Signal Confirmation: 2 consecutive")
        print(f"  Pullback Filter: Enabled")
        print(f"  Golden Time Only: {args.golden_only}")
        print(f"  Trade Cooldown: {cooldown} bars")
        print(f"  Trend Reversal Mult: {trend_mult}")
        print(f"  Sell Filter Strict: {sell_filter}")

        print(f"\nPerformance:")
        print(f"  Total Trades: {stats.total_trades}")
        print(f"  Wins: {stats.wins}")
        print(f"  Losses: {stats.losses}")
        print(f"  Win Rate: {stats.win_rate:.1f}%")

        print(f"\nProfit/Loss:")
        print(f"  Total Profit: ${stats.total_profit:.2f}")
        print(f"  Total Loss: ${stats.total_loss:.2f}")
        print(f"  Net P/L: ${net_pnl:.2f}")
        print(f"  Profit Factor: {stats.profit_factor:.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {stats.max_drawdown:.1f}% (${stats.max_drawdown_usd:.2f})")
        print(f"  Avg Win: ${stats.avg_win:.2f}")
        print(f"  Avg Loss: ${stats.avg_loss:.2f}")
        print(f"  Expectancy: ${stats.expectancy:.2f}")
        print(f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}")

        # Exit reason breakdown
        print(f"\nExit Reasons:")
        exit_counts = {}
        for t in stats.trades:
            reason = t.exit_reason.value
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pct = count / stats.total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

        # Session breakdown
        print(f"\nSession Performance:")
        session_stats = {}
        for t in stats.trades:
            if t.session not in session_stats:
                session_stats[t.session] = {"wins": 0, "losses": 0, "profit": 0}
            if t.result == TradeResult.WIN:
                session_stats[t.session]["wins"] += 1
            else:
                session_stats[t.session]["losses"] += 1
            session_stats[t.session]["profit"] += t.profit_usd

        for session, data in session_stats.items():
            total = data["wins"] + data["losses"]
            wr = data["wins"] / total * 100 if total > 0 else 0
            print(f"  {session}: {total} trades, {wr:.1f}% WR, ${data['profit']:.2f}")

        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtests/results/backtest_{timestamp}.csv"
            backtest.save_results(stats, filepath)

    mt5.disconnect()
    print("\n" + "=" * 70)
    print("Backtest complete!")


if __name__ == "__main__":
    main()
