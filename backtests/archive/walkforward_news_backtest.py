"""
Walk-Forward Backtest with News Filter
========================================
Backtest 1 tahun dengan simulasi news filter (NFP, FOMC, CPI).

Fitur:
1. Historical news calendar (actual dates dari 2025)
2. Skip trading saat high-impact news
3. Compare: WITH news filter vs WITHOUT
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")

# ============================================================
# HISTORICAL NEWS CALENDAR 2025
# ============================================================
# Actual high-impact news dates for USD (affects XAUUSD)
# Format: (date, event_name, impact_level)

HISTORICAL_NEWS_2025 = [
    # January 2025
    (date(2025, 1, 3), "NFP", "HIGH"),
    (date(2025, 1, 14), "CPI", "HIGH"),
    (date(2025, 1, 15), "PPI", "MEDIUM"),
    (date(2025, 1, 29), "FOMC", "HIGH"),
    (date(2025, 1, 30), "GDP Q4", "HIGH"),

    # February 2025
    (date(2025, 2, 7), "NFP", "HIGH"),
    (date(2025, 2, 12), "CPI", "HIGH"),
    (date(2025, 2, 13), "PPI", "MEDIUM"),
    (date(2025, 2, 27), "GDP Revision", "MEDIUM"),

    # March 2025
    (date(2025, 3, 7), "NFP", "HIGH"),
    (date(2025, 3, 12), "CPI", "HIGH"),
    (date(2025, 3, 13), "PPI", "MEDIUM"),
    (date(2025, 3, 19), "FOMC", "HIGH"),
    (date(2025, 3, 27), "GDP Final", "MEDIUM"),

    # April 2025
    (date(2025, 4, 4), "NFP", "HIGH"),
    (date(2025, 4, 10), "CPI", "HIGH"),
    (date(2025, 4, 11), "PPI", "MEDIUM"),
    (date(2025, 4, 30), "GDP Q1", "HIGH"),

    # May 2025
    (date(2025, 5, 2), "NFP", "HIGH"),
    (date(2025, 5, 7), "FOMC", "HIGH"),
    (date(2025, 5, 13), "CPI", "HIGH"),
    (date(2025, 5, 14), "PPI", "MEDIUM"),
    (date(2025, 5, 29), "GDP Revision", "MEDIUM"),

    # June 2025
    (date(2025, 6, 6), "NFP", "HIGH"),
    (date(2025, 6, 11), "CPI", "HIGH"),
    (date(2025, 6, 12), "PPI", "MEDIUM"),
    (date(2025, 6, 18), "FOMC", "HIGH"),
    (date(2025, 6, 26), "GDP Final", "MEDIUM"),

    # July 2025
    (date(2025, 7, 3), "NFP", "HIGH"),
    (date(2025, 7, 11), "CPI", "HIGH"),
    (date(2025, 7, 15), "PPI", "MEDIUM"),
    (date(2025, 7, 30), "FOMC", "HIGH"),
    (date(2025, 7, 31), "GDP Q2", "HIGH"),

    # August 2025
    (date(2025, 8, 1), "NFP", "HIGH"),
    (date(2025, 8, 13), "CPI", "HIGH"),
    (date(2025, 8, 14), "PPI", "MEDIUM"),
    (date(2025, 8, 28), "GDP Revision", "MEDIUM"),

    # September 2025
    (date(2025, 9, 5), "NFP", "HIGH"),
    (date(2025, 9, 10), "CPI", "HIGH"),
    (date(2025, 9, 11), "PPI", "MEDIUM"),
    (date(2025, 9, 17), "FOMC", "HIGH"),
    (date(2025, 9, 25), "GDP Final", "MEDIUM"),

    # October 2025
    (date(2025, 10, 3), "NFP", "HIGH"),
    (date(2025, 10, 10), "CPI", "HIGH"),
    (date(2025, 10, 14), "PPI", "MEDIUM"),
    (date(2025, 10, 30), "GDP Q3", "HIGH"),

    # November 2025
    (date(2025, 11, 7), "NFP", "HIGH"),
    (date(2025, 11, 5), "FOMC", "HIGH"),
    (date(2025, 11, 13), "CPI", "HIGH"),
    (date(2025, 11, 14), "PPI", "MEDIUM"),
    (date(2025, 11, 26), "GDP Revision", "MEDIUM"),

    # December 2025
    (date(2025, 12, 5), "NFP", "HIGH"),
    (date(2025, 12, 10), "CPI", "HIGH"),
    (date(2025, 12, 11), "PPI", "MEDIUM"),
    (date(2025, 12, 17), "FOMC", "HIGH"),

    # January 2026
    (date(2026, 1, 10), "NFP", "HIGH"),
    (date(2026, 1, 15), "CPI", "HIGH"),
    (date(2026, 1, 29), "FOMC", "HIGH"),

    # February 2026
    (date(2026, 2, 5), "NFP", "HIGH"),
]


@dataclass
class NewsFilter:
    """News filter untuk backtest."""

    # Buffer hours sebelum dan sesudah news
    high_impact_buffer_hours: int = 2
    medium_impact_buffer_hours: int = 1

    def __post_init__(self):
        # Build lookup dict for fast checking
        self.news_dates = {}
        for news_date, event_name, impact in HISTORICAL_NEWS_2025:
            if news_date not in self.news_dates:
                self.news_dates[news_date] = []
            self.news_dates[news_date].append((event_name, impact))

    def is_news_blocked(self, dt: datetime) -> Tuple[bool, str]:
        """
        Check if trading should be blocked due to news.

        Returns:
            (is_blocked, reason)
        """
        current_date = dt.date()

        # Check current day
        if current_date in self.news_dates:
            for event_name, impact in self.news_dates[current_date]:
                if impact == "HIGH":
                    # Block entire day for HIGH impact news
                    return True, f"{event_name} (HIGH)"
                elif impact == "MEDIUM":
                    # Block around typical release time (14:30-16:00 WIB typical)
                    if 14 <= dt.hour <= 16:
                        return True, f"{event_name} (MEDIUM)"

        # Check day before (for overnight positions)
        prev_date = current_date - timedelta(days=1)
        if prev_date in self.news_dates:
            for event_name, impact in self.news_dates[prev_date]:
                if impact == "HIGH" and dt.hour < 6:
                    return True, f"{event_name} aftermath"

        return False, "Clear"


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    start_date: date = date(2025, 5, 22)  # Adjusted based on available data
    end_date: date = date(2026, 2, 5)
    initial_capital: float = 5000.0
    lot_size: float = 0.02

    # ML thresholds (from previous optimization)
    ml_threshold: float = 0.65
    ml_only_threshold: float = 0.70

    # Risk settings
    max_daily_loss_pct: float = 0.02
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    ml_confidence: float
    news_event: str = ""


@dataclass
class BacktestResult:
    """Backtest result summary."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    trades: List[Trade] = field(default_factory=list)

    # News-specific stats
    trades_blocked_by_news: int = 0
    news_events_avoided: List[str] = field(default_factory=list)


def load_historical_data(symbol: str = "XAUUSD") -> Optional[pl.DataFrame]:
    """Load historical market data."""
    try:
        import MetaTrader5 as mt5
        from src.config import get_config

        config = get_config()

        # Initialize with full config
        if not mt5.initialize(
            path=config.mt5_path,
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return None

        logger.info(f"MT5 connected: {mt5.account_info().server}")

        # Enable symbol
        mt5.symbol_select(symbol, True)
        import time
        time.sleep(0.5)  # Wait for symbol to be ready

        # Get available M5 data (use last N bars instead of date range)
        # MT5 demo accounts typically have limited history
        # Get 60,000 bars (~200 days of M5 data)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60000)

        if rates is None or len(rates) == 0:
            logger.error(f"No data received from MT5: {mt5.last_error()}")
            # Try alternative method with smaller batch
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, datetime.now(), 50000)
            if rates is None or len(rates) == 0:
                logger.error(f"Still no data: {mt5.last_error()}")
                mt5.shutdown()
                return None

        logger.info(f"Received {len(rates)} bars")

        df = pl.DataFrame({
            "time": [datetime.fromtimestamp(r[0]) for r in rates],
            "open": [r[1] for r in rates],
            "high": [r[2] for r in rates],
            "low": [r[3] for r in rates],
            "close": [r[4] for r in rates],
            "volume": [r[5] for r in rates],
        })

        logger.info(f"Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def calculate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate technical features for ML prediction."""
    # ATR
    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("tr1"),
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr2"),
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr3"),
    ])
    df = df.with_columns([
        pl.max_horizontal("tr1", "tr2", "tr3").alias("tr")
    ])
    df = df.with_columns([
        pl.col("tr").rolling_mean(window_size=14).alias("atr_14")
    ])

    # RSI
    df = df.with_columns([
        (pl.col("close") - pl.col("close").shift(1)).alias("change")
    ])
    df = df.with_columns([
        pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0).alias("gain"),
        pl.when(pl.col("change") < 0).then(pl.col("change").abs()).otherwise(0).alias("loss"),
    ])
    df = df.with_columns([
        pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14).alias("avg_loss"),
    ])
    df = df.with_columns([
        (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-10)))).alias("rsi_14")
    ])

    # Moving Averages
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close").ewm_mean(span=12).alias("ema_12"),
        pl.col("close").ewm_mean(span=26).alias("ema_26"),
    ])

    # MACD
    df = df.with_columns([
        (pl.col("ema_12") - pl.col("ema_26")).alias("macd")
    ])
    df = df.with_columns([
        pl.col("macd").ewm_mean(span=9).alias("macd_signal")
    ])

    # Bollinger Bands
    df = df.with_columns([
        pl.col("close").rolling_std(window_size=20).alias("bb_std")
    ])
    df = df.with_columns([
        (pl.col("sma_20") + 2 * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("sma_20") - 2 * pl.col("bb_std")).alias("bb_lower"),
    ])

    # Momentum features
    df = df.with_columns([
        ((pl.col("close") - pl.col("close").shift(5)) / pl.col("close").shift(5) * 100).alias("momentum_5"),
        ((pl.col("close") - pl.col("close").shift(10)) / pl.col("close").shift(10) * 100).alias("momentum_10"),
        ((pl.col("close") - pl.col("sma_20")) / pl.col("sma_20") * 100).alias("price_to_sma"),
    ])

    # Volatility
    df = df.with_columns([
        (pl.col("atr_14") / pl.col("close") * 100).alias("volatility_pct")
    ])

    # Hour and day features
    df = df.with_columns([
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.weekday().alias("dayofweek"),
    ])

    return df.drop_nulls()


def simulate_ml_prediction(df: pl.DataFrame, idx: int) -> Tuple[str, float]:
    """
    Simulate ML prediction based on technical indicators.
    Returns (signal, confidence).
    """
    row = df.row(idx, named=True)

    # Score based on multiple factors
    score = 0.5  # Neutral base

    # RSI
    rsi = row.get("rsi_14", 50)
    if rsi < 30:
        score += 0.15  # Oversold - bullish
    elif rsi > 70:
        score -= 0.15  # Overbought - bearish

    # MACD
    macd = row.get("macd", 0)
    macd_signal = row.get("macd_signal", 0)
    if macd > macd_signal:
        score += 0.1
    else:
        score -= 0.1

    # Price vs SMA
    close = row.get("close", 0)
    sma_20 = row.get("sma_20", close)
    sma_50 = row.get("sma_50", close)

    if close > sma_20 > sma_50:
        score += 0.1  # Bullish trend
    elif close < sma_20 < sma_50:
        score -= 0.1  # Bearish trend

    # Bollinger Bands
    bb_upper = row.get("bb_upper", close + 10)
    bb_lower = row.get("bb_lower", close - 10)

    if close < bb_lower:
        score += 0.1  # Oversold
    elif close > bb_upper:
        score -= 0.1  # Overbought

    # Momentum
    momentum = row.get("momentum_5", 0)
    if momentum > 0.5:
        score += 0.05
    elif momentum < -0.5:
        score -= 0.05

    # Add some randomness to simulate real ML variance
    noise = np.random.normal(0, 0.1)
    score = max(0, min(1, score + noise))

    # Determine signal and confidence
    if score > 0.5:
        signal = "BUY"
        confidence = 0.5 + (score - 0.5) * 0.8  # Scale to 0.5-0.9
    else:
        signal = "SELL"
        confidence = 0.5 + (0.5 - score) * 0.8

    return signal, confidence


def run_backtest(
    df: pl.DataFrame,
    config: BacktestConfig,
    use_news_filter: bool = True,
) -> BacktestResult:
    """
    Run backtest with or without news filter.
    """
    news_filter = NewsFilter() if use_news_filter else None

    trades: List[Trade] = []
    trades_blocked = 0
    news_avoided = []

    capital = config.initial_capital
    daily_pnl = 0.0
    current_date = None

    position = None  # {"direction": str, "entry_price": float, "entry_time": datetime, "sl": float, "tp": float, "confidence": float}

    logger.info(f"Starting backtest ({'WITH' if use_news_filter else 'WITHOUT'} news filter)")
    logger.info(f"Period: {config.start_date} to {config.end_date}")

    for idx in range(100, len(df)):  # Start after warmup
        row = df.row(idx, named=True)
        current_time = row["time"]

        # Filter by date range
        if current_time.date() < config.start_date:
            continue
        if current_time.date() > config.end_date:
            break

        # Daily reset
        if current_date != current_time.date():
            current_date = current_time.date()
            daily_pnl = 0.0

        # Check daily loss limit
        if daily_pnl < -config.max_daily_loss_pct * capital:
            continue

        # Get current price
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row.get("atr_14", close * 0.003)

        # Manage existing position
        if position is not None:
            # Check SL/TP
            if position["direction"] == "BUY":
                if low <= position["sl"]:
                    # Stop loss hit
                    pnl = (position["sl"] - position["entry_price"]) * config.lot_size * 100
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="BUY",
                        entry_price=position["entry_price"],
                        exit_price=position["sl"],
                        lot_size=config.lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                    ))
                    daily_pnl += pnl
                    capital += pnl
                    position = None
                elif high >= position["tp"]:
                    # Take profit hit
                    pnl = (position["tp"] - position["entry_price"]) * config.lot_size * 100
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="BUY",
                        entry_price=position["entry_price"],
                        exit_price=position["tp"],
                        lot_size=config.lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                    ))
                    daily_pnl += pnl
                    capital += pnl
                    position = None
            else:  # SELL
                if high >= position["sl"]:
                    # Stop loss hit
                    pnl = (position["entry_price"] - position["sl"]) * config.lot_size * 100
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="SELL",
                        entry_price=position["entry_price"],
                        exit_price=position["sl"],
                        lot_size=config.lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                    ))
                    daily_pnl += pnl
                    capital += pnl
                    position = None
                elif low <= position["tp"]:
                    # Take profit hit
                    pnl = (position["entry_price"] - position["tp"]) * config.lot_size * 100
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="SELL",
                        entry_price=position["entry_price"],
                        exit_price=position["tp"],
                        lot_size=config.lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                    ))
                    daily_pnl += pnl
                    capital += pnl
                    position = None

        # Skip if already in position
        if position is not None:
            continue

        # NEWS FILTER CHECK
        if news_filter is not None:
            is_blocked, news_reason = news_filter.is_news_blocked(current_time)
            if is_blocked:
                trades_blocked += 1
                if news_reason not in news_avoided:
                    news_avoided.append(news_reason)
                continue

        # Session filter (simplified - only trade during London/NY)
        hour = current_time.hour
        if hour < 14 or hour > 23:  # WIB timezone
            continue

        # Get ML prediction
        signal, confidence = simulate_ml_prediction(df, idx)

        # Check confidence threshold
        if confidence < config.ml_only_threshold:
            continue

        # Entry signal
        if signal == "BUY":
            sl = close - (atr * config.sl_atr_mult)
            tp = close + (atr * config.tp_atr_mult)
            position = {
                "direction": "BUY",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }
        else:
            sl = close + (atr * config.sl_atr_mult)
            tp = close - (atr * config.tp_atr_mult)
            position = {
                "direction": "SELL",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }

    # Close any remaining position
    if position is not None and len(df) > 0:
        last_row = df.row(-1, named=True)
        last_close = last_row["close"]
        if position["direction"] == "BUY":
            pnl = (last_close - position["entry_price"]) * config.lot_size * 100
        else:
            pnl = (position["entry_price"] - last_close) * config.lot_size * 100
        trades.append(Trade(
            entry_time=position["entry_time"],
            exit_time=last_row["time"],
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=last_close,
            lot_size=config.lot_size,
            pnl=pnl,
            ml_confidence=position["confidence"],
        ))

    # Calculate results
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl <= 0)

    total_pnl = sum(t.pnl for t in trades)

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in trades if t.pnl <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    total_wins = sum(wins) if wins else 0
    total_losses = sum(losses) if losses else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Calculate max drawdown
    equity_curve = [config.initial_capital]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t.pnl)

    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return BacktestResult(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / total_trades * 100 if total_trades > 0 else 0,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        trades=trades,
        trades_blocked_by_news=trades_blocked,
        news_events_avoided=news_avoided,
    )


def main():
    """Run comparison backtest."""
    print("=" * 70)
    print("WALK-FORWARD BACKTEST WITH NEWS FILTER")
    print("=" * 70)
    print()

    # Load data
    logger.info("Loading historical data...")
    df = load_historical_data()

    if df is None:
        logger.error("Failed to load data")
        return

    # Calculate features
    logger.info("Calculating features...")
    df = calculate_features(df)
    logger.info(f"Data ready: {len(df)} bars with features")

    # Configuration
    config = BacktestConfig(
        start_date=date(2025, 5, 22),  # Based on available MT5 data
        end_date=date(2026, 2, 5),
        initial_capital=5000.0,
        lot_size=0.02,
        ml_threshold=0.65,
        ml_only_threshold=0.70,
    )

    print()
    print("=" * 70)
    print("BACKTEST 1: WITHOUT NEWS FILTER")
    print("=" * 70)

    result_no_news = run_backtest(df, config, use_news_filter=False)

    print(f"""
Results WITHOUT News Filter:
-----------------------------
Total Trades   : {result_no_news.total_trades}
Win Rate       : {result_no_news.win_rate:.1f}%
Total P/L      : ${result_no_news.total_pnl:,.2f}
Avg Win        : ${result_no_news.avg_win:.2f}
Avg Loss       : ${result_no_news.avg_loss:.2f}
Profit Factor  : {result_no_news.profit_factor:.2f}
Max Drawdown   : {result_no_news.max_drawdown:.1f}%
""")

    print()
    print("=" * 70)
    print("BACKTEST 2: WITH NEWS FILTER")
    print("=" * 70)

    result_with_news = run_backtest(df, config, use_news_filter=True)

    print(f"""
Results WITH News Filter:
-----------------------------
Total Trades   : {result_with_news.total_trades}
Win Rate       : {result_with_news.win_rate:.1f}%
Total P/L      : ${result_with_news.total_pnl:,.2f}
Avg Win        : ${result_with_news.avg_win:.2f}
Avg Loss       : ${result_with_news.avg_loss:.2f}
Profit Factor  : {result_with_news.profit_factor:.2f}
Max Drawdown   : {result_with_news.max_drawdown:.1f}%

News Filter Stats:
-----------------------------
Trades Blocked : {result_with_news.trades_blocked_by_news}
Events Avoided : {len(result_with_news.news_events_avoided)}
""")

    # Print avoided events
    if result_with_news.news_events_avoided:
        print("News Events Avoided:")
        for event in result_with_news.news_events_avoided[:20]:
            print(f"  - {event}")

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Calculate improvement
    if result_no_news.total_pnl != 0:
        pnl_improvement = ((result_with_news.total_pnl - result_no_news.total_pnl) / abs(result_no_news.total_pnl)) * 100
    else:
        pnl_improvement = 0

    wr_improvement = result_with_news.win_rate - result_no_news.win_rate
    dd_improvement = result_no_news.max_drawdown - result_with_news.max_drawdown

    print(f"""
                    Without News    With News     Improvement
                    ------------    ---------     -----------
Total Trades        {result_no_news.total_trades:<15} {result_with_news.total_trades:<13} {result_with_news.total_trades - result_no_news.total_trades:+d}
Win Rate            {result_no_news.win_rate:<15.1f} {result_with_news.win_rate:<13.1f} {wr_improvement:+.1f}%
Total P/L           ${result_no_news.total_pnl:<14,.2f} ${result_with_news.total_pnl:<12,.2f} {pnl_improvement:+.1f}%
Profit Factor       {result_no_news.profit_factor:<15.2f} {result_with_news.profit_factor:<13.2f}
Max Drawdown        {result_no_news.max_drawdown:<15.1f}% {result_with_news.max_drawdown:<12.1f}% {dd_improvement:+.1f}%
""")

    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if result_with_news.win_rate > result_no_news.win_rate and result_with_news.total_pnl > result_no_news.total_pnl:
        print("""
✅ NEWS FILTER RECOMMENDED

Alasan:
1. Win Rate meningkat
2. Total Profit meningkat
3. Menghindari volatilitas tinggi saat high-impact news

Dengan menghindari trading saat NFP, FOMC, CPI, bot menghindari
pergerakan tidak terduga yang sering merugikan.
""")
    elif result_with_news.win_rate > result_no_news.win_rate:
        print("""
⚠️ NEWS FILTER BERGUNA untuk Win Rate

Alasan:
- Win Rate meningkat (lebih sedikit loss dari news spike)
- Tapi total trades berkurang signifikan
- Pertimbangkan risk tolerance Anda
""")
    elif result_with_news.max_drawdown < result_no_news.max_drawdown:
        print("""
[!] NEWS FILTER BERGUNA untuk Risk Management

Alasan:
- Max Drawdown berkurang
- Menghindari loss besar saat news
- Trade lebih aman walau profit mungkin berkurang
""")
    else:
        print("""
❌ NEWS FILTER KURANG BERDAMPAK dalam backtest ini

Catatan:
- Backtest menggunakan simulated ML, bukan model asli
- Real-world impact mungkin berbeda
- High-impact news tetap berisiko tinggi
""")

    print()
    print("=" * 70)
    print("Backtest completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
