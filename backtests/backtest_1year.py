"""
Backtest 1 Year: 2025 - Today
=============================
Comprehensive backtest comparing old vs new filter logic.

Tests:
1. Old Logic: SMC-only with ML weak filter
2. New Logic: ML threshold (55%) + Signal Confirmation + Pullback Filter
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, MarketRegime
from src.ml_model import TradingModel, get_default_feature_columns
from src.config import get_config
from loguru import logger

# Reduce logging noise
logger.remove()
logger.add(sys.stderr, level="WARNING")


class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


@dataclass
class SimulatedTrade:
    """A simulated trade."""
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
    exit_reason: str
    ml_confidence: float
    smc_confidence: float
    regime: str
    filter_version: str  # "old" or "new"


@dataclass
class BacktestStats:
    """Statistics for a backtest run."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)


def check_pullback_filter(df: pl.DataFrame, signal_direction: str, idx: int) -> Tuple[bool, str]:
    """Check if pullback filter would block at given index."""
    try:
        if idx < 5:
            return False, "OK"

        # Get data up to current index
        closes = df["close"].to_list()[:idx+1]
        last_3 = closes[-3:]

        short_momentum = last_3[-1] - last_3[0]
        momentum_dir = "UP" if short_momentum > 0 else "DOWN"

        # MACD histogram
        macd_dir = "NEUTRAL"
        if "macd_histogram" in df.columns:
            macd_hist = df["macd_histogram"].to_list()[:idx+1]
            if len(macd_hist) >= 2 and macd_hist[-1] is not None and macd_hist[-2] is not None:
                macd_dir = "RISING" if macd_hist[-1] > macd_hist[-2] else "FALLING"

        # Pullback logic
        if signal_direction == "SELL":
            if momentum_dir == "UP" and short_momentum > 2:
                return True, f"Price bouncing UP (+${short_momentum:.2f})"
            if macd_dir == "RISING" and momentum_dir == "UP":
                return True, "MACD bullish + price rising"
        elif signal_direction == "BUY":
            if momentum_dir == "DOWN" and short_momentum < -2:
                return True, f"Price falling DOWN (${short_momentum:.2f})"
            if macd_dir == "FALLING" and momentum_dir == "DOWN":
                return True, "MACD bearish + price falling"

        return False, "OK"
    except:
        return False, "OK"


def simulate_trade_outcome(
    df: pl.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    lot_size: float = 0.01,
    max_bars: int = 100,  # Max bars to hold position
) -> Tuple[float, float, str, int]:
    """
    Simulate trade outcome by walking forward through price data.

    Returns: (profit_usd, profit_pips, exit_reason, exit_idx)
    """
    pip_value = 10  # For XAUUSD, 1 pip = $10 per lot

    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()

    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = highs[i]
        low = lows[i]
        close = closes[i]

        if direction == "BUY":
            # Check stop loss
            if low <= stop_loss:
                pips = (stop_loss - entry_price) / 0.1
                profit = pips * pip_value * lot_size
                return profit, pips, "stop_loss", i

            # Check take profit
            if high >= take_profit:
                pips = (take_profit - entry_price) / 0.1
                profit = pips * pip_value * lot_size
                return profit, pips, "take_profit", i

        else:  # SELL
            # Check stop loss
            if high >= stop_loss:
                pips = (entry_price - stop_loss) / 0.1
                profit = pips * pip_value * lot_size
                return profit, pips, "stop_loss", i

            # Check take profit
            if low <= take_profit:
                pips = (entry_price - take_profit) / 0.1
                profit = pips * pip_value * lot_size
                return profit, pips, "take_profit", i

    # Position still open after max_bars - close at current price
    final_price = closes[min(entry_idx + max_bars - 1, len(df) - 1)]
    if direction == "BUY":
        pips = (final_price - entry_price) / 0.1
    else:
        pips = (entry_price - final_price) / 0.1

    profit = pips * pip_value * lot_size
    return profit, pips, "timeout", min(entry_idx + max_bars - 1, len(df) - 1)


def run_backtest(
    df: pl.DataFrame,
    smc: SMCAnalyzer,
    ml_model: TradingModel,
    regime_detector: MarketRegimeDetector,
    filter_version: str = "old",
    initial_capital: float = 5000.0,
) -> BacktestStats:
    """
    Run backtest with specified filter version.

    Args:
        df: Full DataFrame with all indicators
        smc: SMC analyzer
        ml_model: ML model for predictions
        regime_detector: Regime detector
        filter_version: "old" or "new"
        initial_capital: Starting capital
    """
    stats = BacktestStats()
    capital = initial_capital
    peak_capital = initial_capital

    # Get feature columns
    feature_cols = [f for f in ml_model.feature_names if f in df.columns]

    # Track for signal confirmation (new filter)
    signal_persistence = {}
    last_trade_idx = -100  # Cooldown tracking
    cooldown_bars = 20  # ~5 hours on M15

    # Iterate through data
    print(f"\nRunning backtest with {filter_version.upper()} filters...")

    for i in range(100, len(df) - 100):  # Leave margin for lookback and forward simulation
        # Cooldown check
        if i - last_trade_idx < cooldown_bars:
            continue

        # Get data slice up to current bar
        df_slice = df.head(i + 1)

        # Generate SMC signal
        try:
            smc_signal = smc.generate_signal(df_slice)
        except:
            continue

        if smc_signal is None:
            # Reset signal persistence
            signal_persistence = {}
            continue

        # Get ML prediction
        try:
            ml_pred = ml_model.predict(df_slice, feature_cols)
        except:
            continue

        # Get regime
        try:
            regime_state = regime_detector.get_current_state(df_slice)
            regime = regime_state.regime.value if regime_state else "normal"
        except:
            regime = "normal"

        # Skip if CRISIS regime
        if regime == "crisis":
            continue

        # === FILTER LOGIC ===
        should_trade = False

        if filter_version == "old":
            # OLD LOGIC: SMC signal with weak ML filter
            # Only block if ML strongly disagrees (>65% opposite)
            ml_strongly_disagrees = (
                (smc_signal.signal_type == "BUY" and ml_pred.signal == "SELL" and ml_pred.confidence > 0.65) or
                (smc_signal.signal_type == "SELL" and ml_pred.signal == "BUY" and ml_pred.confidence > 0.65)
            )
            should_trade = not ml_strongly_disagrees

        else:  # "new"
            # NEW LOGIC: ML threshold + confirmation + pullback filter

            # Filter 1: ML confidence threshold (>= 55%)
            if ml_pred.confidence < 0.55:
                signal_persistence = {}
                continue

            # Filter 2: ML shouldn't strongly disagree
            ml_strongly_disagrees = (
                (smc_signal.signal_type == "BUY" and ml_pred.signal == "SELL" and ml_pred.confidence > 0.65) or
                (smc_signal.signal_type == "SELL" and ml_pred.signal == "BUY" and ml_pred.confidence > 0.65)
            )
            if ml_strongly_disagrees:
                signal_persistence = {}
                continue

            # Filter 3: Signal confirmation
            signal_key = f"{smc_signal.signal_type}_{int(smc_signal.entry_price)}"
            if signal_key not in signal_persistence:
                signal_persistence[signal_key] = 1
                continue  # Wait for confirmation
            else:
                signal_persistence[signal_key] += 1

            if signal_persistence[signal_key] < 2:
                continue

            # Reset persistence
            signal_persistence = {}

            # Filter 4: Pullback filter
            pullback_blocked, _ = check_pullback_filter(df_slice, smc_signal.signal_type, i)
            if pullback_blocked:
                continue

            should_trade = True

        if not should_trade:
            continue

        # === EXECUTE TRADE ===
        # Determine lot size based on ML confidence (new) or fixed (old)
        if filter_version == "new":
            if ml_pred.confidence >= 0.65:
                lot_size = 0.02
            elif ml_pred.confidence >= 0.55:
                lot_size = 0.01
            else:
                lot_size = 0.01
        else:
            lot_size = 0.01

        # Simulate trade
        entry_price = smc_signal.entry_price
        stop_loss = smc_signal.stop_loss
        take_profit = smc_signal.take_profit

        profit, pips, exit_reason, exit_idx = simulate_trade_outcome(
            df=df,
            entry_idx=i,
            direction=smc_signal.signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
        )

        # Record trade
        entry_time = df["time"].to_list()[i]
        exit_time = df["time"].to_list()[exit_idx]

        result = TradeResult.WIN if profit > 0 else (TradeResult.LOSS if profit < 0 else TradeResult.BREAKEVEN)

        trade = SimulatedTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=smc_signal.signal_type,
            entry_price=entry_price,
            exit_price=df["close"].to_list()[exit_idx],
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            profit_usd=profit,
            profit_pips=pips,
            result=result,
            exit_reason=exit_reason,
            ml_confidence=ml_pred.confidence,
            smc_confidence=smc_signal.confidence,
            regime=regime,
            filter_version=filter_version,
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
        drawdown = (peak_capital - capital) / peak_capital * 100
        if drawdown > stats.max_drawdown:
            stats.max_drawdown = drawdown

        # Update last trade index for cooldown
        last_trade_idx = exit_idx

        # Progress
        if stats.total_trades % 50 == 0:
            print(f"  {stats.total_trades} trades processed...")

    # Calculate final stats
    if stats.total_trades > 0:
        stats.win_rate = stats.wins / stats.total_trades * 100
        stats.avg_win = stats.total_profit / stats.wins if stats.wins > 0 else 0
        stats.avg_loss = stats.total_loss / stats.losses if stats.losses > 0 else 0
        stats.profit_factor = stats.total_profit / stats.total_loss if stats.total_loss > 0 else float('inf')

    return stats


def main():
    """Run 1-year backtest."""
    print("=" * 70)
    print("BACKTEST: 1 Year (2025 - Today)")
    print("=" * 70)

    # Initialize
    config = get_config()

    mt5 = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    mt5.connect()
    print(f"\nConnected to MT5")

    # Initialize components
    smc = SMCAnalyzer()
    features = FeatureEngineer()

    regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    regime_detector.load()

    ml_model = TradingModel(model_path="models/xgboost_model.pkl")
    ml_model.load()
    print(f"Models loaded")

    # Fetch historical data
    # MT5 typically allows ~10000 bars, which is about 3-4 months on M15
    # For 1 year, we need to fetch in chunks or use a larger timeframe
    print(f"\nFetching historical data...")

    # Try to get maximum available data
    df = mt5.get_market_data(
        symbol="XAUUSD",
        timeframe="M15",
        count=50000,  # Request max, MT5 will return what's available
    )

    if len(df) == 0:
        print("ERROR: No data received")
        return

    print(f"Received {len(df)} bars")

    # Get date range
    times = df["time"].to_list()
    start_date = times[0]
    end_date = times[-1]
    print(f"Date range: {start_date} to {end_date}")

    # Calculate indicators
    print(f"\nCalculating indicators...")
    df = features.calculate_all(df, include_ml_features=True)
    df = smc.calculate_all(df)

    try:
        df = regime_detector.predict(df)
    except:
        pass

    print(f"Indicators calculated")

    # Run backtests
    print("\n" + "=" * 70)

    # OLD filters
    old_stats = run_backtest(
        df=df,
        smc=smc,
        ml_model=ml_model,
        regime_detector=regime_detector,
        filter_version="old",
    )

    # NEW filters
    new_stats = run_backtest(
        df=df,
        smc=smc,
        ml_model=ml_model,
        regime_detector=regime_detector,
        filter_version="new",
    )

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 70)

    print(f"\nData Period: {start_date} to {end_date}")
    print(f"Total Bars: {len(df)}")

    print(f"\n{'Metric':<25} {'OLD Filters':>15} {'NEW Filters':>15} {'Diff':>15}")
    print("-" * 70)

    metrics = [
        ("Total Trades", old_stats.total_trades, new_stats.total_trades),
        ("Wins", old_stats.wins, new_stats.wins),
        ("Losses", old_stats.losses, new_stats.losses),
        ("Win Rate (%)", f"{old_stats.win_rate:.1f}", f"{new_stats.win_rate:.1f}"),
        ("Total Profit ($)", f"{old_stats.total_profit:.2f}", f"{new_stats.total_profit:.2f}"),
        ("Total Loss ($)", f"{old_stats.total_loss:.2f}", f"{new_stats.total_loss:.2f}"),
        ("Net P/L ($)", f"{old_stats.total_profit - old_stats.total_loss:.2f}",
         f"{new_stats.total_profit - new_stats.total_loss:.2f}"),
        ("Profit Factor", f"{old_stats.profit_factor:.2f}" if old_stats.profit_factor != float('inf') else "∞",
         f"{new_stats.profit_factor:.2f}" if new_stats.profit_factor != float('inf') else "∞"),
        ("Avg Win ($)", f"{old_stats.avg_win:.2f}", f"{new_stats.avg_win:.2f}"),
        ("Avg Loss ($)", f"{old_stats.avg_loss:.2f}", f"{new_stats.avg_loss:.2f}"),
        ("Max Drawdown (%)", f"{old_stats.max_drawdown:.1f}", f"{new_stats.max_drawdown:.1f}"),
    ]

    for name, old_val, new_val in metrics:
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            diff = new_val - old_val
            diff_str = f"{diff:+.2f}" if isinstance(diff, float) else f"{diff:+d}"
        else:
            diff_str = "-"
        print(f"{name:<25} {str(old_val):>15} {str(new_val):>15} {diff_str:>15}")

    # Net P/L comparison
    old_net = old_stats.total_profit - old_stats.total_loss
    new_net = new_stats.total_profit - new_stats.total_loss
    improvement = new_net - old_net

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOLD Filters Net P/L: ${old_net:.2f}")
    print(f"NEW Filters Net P/L: ${new_net:.2f}")
    print(f"IMPROVEMENT: ${improvement:.2f} ({improvement/abs(old_net)*100 if old_net != 0 else 0:.1f}%)")

    if new_stats.win_rate > old_stats.win_rate:
        print(f"\nWin Rate improved: {old_stats.win_rate:.1f}% -> {new_stats.win_rate:.1f}%")

    if new_stats.max_drawdown < old_stats.max_drawdown:
        print(f"Max Drawdown reduced: {old_stats.max_drawdown:.1f}% -> {new_stats.max_drawdown:.1f}%")

    # Trade distribution by ML confidence (NEW)
    if new_stats.trades:
        print(f"\n--- NEW Filter Trade Analysis ---")
        high_conf = [t for t in new_stats.trades if t.ml_confidence >= 0.65]
        med_conf = [t for t in new_stats.trades if 0.55 <= t.ml_confidence < 0.65]

        if high_conf:
            high_wr = len([t for t in high_conf if t.result == TradeResult.WIN]) / len(high_conf) * 100
            high_pnl = sum(t.profit_usd for t in high_conf)
            print(f"High Confidence (>=65%): {len(high_conf)} trades, {high_wr:.1f}% WR, ${high_pnl:.2f}")

        if med_conf:
            med_wr = len([t for t in med_conf if t.result == TradeResult.WIN]) / len(med_conf) * 100
            med_pnl = sum(t.profit_usd for t in med_conf)
            print(f"Med Confidence (55-65%): {len(med_conf)} trades, {med_wr:.1f}% WR, ${med_pnl:.2f}")

    mt5.disconnect()
    print("\n" + "=" * 70)
    print("Backtest complete!")


if __name__ == "__main__":
    main()
