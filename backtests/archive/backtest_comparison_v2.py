"""
BACKTEST COMPARISON v2 - SMC-only vs ML+SMC
===========================================
Compare different signal strategies:
- System A: SMC-only (original profitable backtest)
- System B: ML+SMC during Golden Time (new conservative)
- System C: Tighter Smart Hold (50% cut vs 80% cut)
"""

import polars as pl
import numpy as np
import pickle
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")


def get_session(dt: datetime) -> Tuple[str, bool]:
    """Get trading session and if it's golden time."""
    hour = dt.hour

    if 19 <= hour <= 23:
        return "London-NY Overlap", True  # GOLDEN TIME
    elif 14 <= hour < 19:
        return "London", False
    elif 5 <= hour < 14:
        return "Sydney/Tokyo", False
    else:
        return "Off-hours", False


def hours_to_golden(dt: datetime) -> float:
    """Calculate hours until golden time (19:00 WIB)."""
    current_hour = dt.hour + dt.minute / 60
    golden_start = 19.0

    if 19 <= current_hour <= 23:
        return 0  # Already in golden time
    elif current_hour < 19:
        return golden_start - current_hour
    else:  # After 23:00
        return (24 - current_hour) + golden_start


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    hold_time_hours: float
    session: str
    is_golden: bool


class MLSimulator:
    """Simulate ML predictions based on loaded model."""

    def __init__(self, model_path: str = "models/xgboost_model.pkl"):
        self.model = None
        self.features = None
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.features = data.get("features", [])
                else:
                    self.model = data
            logger.info(f"ML model loaded for backtest")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def predict(self, df: pl.DataFrame, idx: int) -> Tuple[str, float]:
        """Predict signal and confidence at given index."""
        if self.model is None:
            return "HOLD", 0.50

        try:
            # Get features for this row
            row = df.row(idx, named=True)

            # Simple momentum-based prediction for simulation
            # (Real model would use actual features)
            close = row.get("close", 0)
            sma_20 = row.get("sma_20", close)
            rsi = row.get("rsi", 50)

            # Simulate prediction
            if close > sma_20 and rsi < 70:
                return "BUY", 0.55 + (70 - rsi) / 200
            elif close < sma_20 and rsi > 30:
                return "SELL", 0.55 + (rsi - 30) / 200
            else:
                return "HOLD", 0.50

        except Exception:
            return "HOLD", 0.50


def run_comparison():
    """Run comprehensive comparison backtest."""

    print("=" * 80)
    print("BACKTEST COMPARISON v2: SMC-only vs ML+SMC")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    import MetaTrader5 as mt5
    from src.feature_eng import FeatureEngineer
    from src.smc_polars import SMCAnalyzer

    if not mt5.initialize():
        print("MT5 init failed")
        return

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 40000)
    mt5.shutdown()

    if rates is None:
        print("Failed to get data")
        return

    df = pl.DataFrame({
        "time": [datetime.fromtimestamp(r[0]) for r in rates],
        "open": [r[1] for r in rates],
        "high": [r[2] for r in rates],
        "low": [r[3] for r in rates],
        "close": [r[4] for r in rates],
        "volume": [r[5] for r in rates],
    })

    print(f"    Loaded {len(df)} bars")
    print(f"    Range: {df['time'][0]} to {df['time'][-1]}")

    # Calculate features
    print("\n[2] Calculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_all(df)

    # Parameters
    lot_size = 0.02
    initial_capital = 5000.0
    max_loss_per_trade = 50.0
    confidence_threshold = 0.70
    min_bars_between_trades = 4

    # Initialize ML simulator
    ml_sim = MLSimulator()

    print("\n[3] Running backtests...")
    print(f"    Lot size: {lot_size}")
    print(f"    Initial capital: ${initial_capital}")
    print(f"    Max loss per trade: ${max_loss_per_trade}")

    # ========================================
    # SYSTEM A: SMC-ONLY (Original Backtest)
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM A: SMC-ONLY (No ML requirement)")
    print("  - Trade on SMC signal only")
    print("  - Cut loss at 80% of max")
    print("=" * 80)

    trades_a = run_system(
        df, lot_size, initial_capital, max_loss_per_trade,
        confidence_threshold, min_bars_between_trades,
        ml_sim, system_type="SMC_ONLY", cut_loss_pct=0.80
    )

    # ========================================
    # SYSTEM B: ML+SMC during Golden Time
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM B: ML+SMC during Golden Time")
    print("  - Golden Time (19:00-23:00): Require ML+SMC alignment")
    print("  - Other times: SMC-only")
    print("  - Cut loss at 80% of max")
    print("=" * 80)

    trades_b = run_system(
        df, lot_size, initial_capital, max_loss_per_trade,
        confidence_threshold, min_bars_between_trades,
        ml_sim, system_type="ML_SMC_GOLDEN", cut_loss_pct=0.80
    )

    # ========================================
    # SYSTEM C: Tighter Smart Hold
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM C: Tighter Smart Hold")
    print("  - SMC-only mode")
    print("  - Cut loss at 50% of max (tighter)")
    print("=" * 80)

    trades_c = run_system(
        df, lot_size, initial_capital, max_loss_per_trade,
        confidence_threshold, min_bars_between_trades,
        ml_sim, system_type="SMC_ONLY", cut_loss_pct=0.50
    )

    # ========================================
    # SYSTEM D: ML+SMC + Tighter Hold
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM D: ML+SMC + Tighter Hold (NEW LIVE SYSTEM)")
    print("  - Golden Time: Require ML+SMC alignment")
    print("  - Cut loss at 50% of max")
    print("=" * 80)

    trades_d = run_system(
        df, lot_size, initial_capital, max_loss_per_trade,
        confidence_threshold, min_bars_between_trades,
        ml_sim, system_type="ML_SMC_GOLDEN", cut_loss_pct=0.50
    )

    # ========================================
    # COMPARISON RESULTS
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    results = []
    for name, trades in [
        ("A: SMC-only (80% cut)", trades_a),
        ("B: ML+SMC Golden (80% cut)", trades_b),
        ("C: SMC-only (50% cut)", trades_c),
        ("D: ML+SMC + 50% cut (NEW)", trades_d),
    ]:
        stats = calc_stats(trades, name, initial_capital)
        results.append(stats)
        print_stats(stats)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'System':<30} {'Trades':>8} {'Win%':>8} {'P/L':>12} {'PF':>8} {'MaxDD':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} {r['trades']:>8} {r['win_rate']:>7.1f}% ${r['total_pnl']:>10.2f} {r['profit_factor']:>7.2f} {r['max_drawdown']:>9.2f}%")

    # Golden Time breakdown
    print("\n" + "=" * 80)
    print("GOLDEN TIME BREAKDOWN")
    print("=" * 80)

    for name, trades in [
        ("A: SMC-only (80%)", trades_a),
        ("D: ML+SMC + 50% (NEW)", trades_d),
    ]:
        golden_trades = [t for t in trades if t.is_golden]
        non_golden_trades = [t for t in trades if not t.is_golden]

        print(f"\n{name}:")
        if golden_trades:
            golden_pnl = sum(t.pnl for t in golden_trades)
            golden_wins = len([t for t in golden_trades if t.pnl > 0])
            print(f"  Golden Time: {len(golden_trades)} trades, {golden_wins}/{len(golden_trades)} wins ({100*golden_wins/len(golden_trades):.1f}%), P/L: ${golden_pnl:.2f}")
        if non_golden_trades:
            ng_pnl = sum(t.pnl for t in non_golden_trades)
            ng_wins = len([t for t in non_golden_trades if t.pnl > 0])
            print(f"  Non-Golden:  {len(non_golden_trades)} trades, {ng_wins}/{len(non_golden_trades)} wins ({100*ng_wins/len(non_golden_trades):.1f}%), P/L: ${ng_pnl:.2f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best = max(results, key=lambda x: x['total_pnl'])
    safest = min(results, key=lambda x: x['max_drawdown'])

    print(f"  Most Profitable: {best['name']} (${best['total_pnl']:.2f})")
    print(f"  Lowest Drawdown: {safest['name']} ({safest['max_drawdown']:.2f}%)")

    if best['name'] == safest['name']:
        print(f"\n  ✓ RECOMMENDED: {best['name']}")
    else:
        print(f"\n  Trade-off detected:")
        print(f"    - For max profit: {best['name']}")
        print(f"    - For safety: {safest['name']}")


def run_system(
    df: pl.DataFrame,
    lot_size: float,
    initial_capital: float,
    max_loss_per_trade: float,
    confidence_threshold: float,
    min_bars_between_trades: int,
    ml_sim: MLSimulator,
    system_type: str,  # "SMC_ONLY" or "ML_SMC_GOLDEN"
    cut_loss_pct: float,  # 0.80 or 0.50
) -> List[Trade]:
    """Run backtest for a specific system configuration."""

    from src.smc_polars import SMCAnalyzer

    trades: List[Trade] = []
    position = None
    capital = initial_capital
    last_trade_idx = -min_bars_between_trades

    for idx in range(200, len(df) - 1):
        row = df.row(idx, named=True)
        current_time = row["time"]

        if current_time.date() < date(2025, 6, 1):
            continue
        if current_time.date() > date(2026, 2, 5):
            break

        close = row["close"]
        high = row["high"]
        low = row["low"]

        session, is_golden = get_session(current_time)
        hrs_to_golden = hours_to_golden(current_time)

        # Manage position
        if position is not None:
            exit_reason = None
            exit_price = None

            # Calculate current P/L
            if position["direction"] == "BUY":
                current_pnl = (close - position["entry"]) * lot_size * 100
                if high >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP_HIT"
            else:
                current_pnl = (position["entry"] - close) * lot_size * 100
                if low <= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP_HIT"

            # Smart Hold Logic
            if exit_reason is None:
                loss_percent = abs(current_pnl) / max_loss_per_trade if current_pnl < 0 else 0

                if current_pnl < 0:
                    # Max loss - use cut_loss_pct parameter
                    if loss_percent >= cut_loss_pct:
                        exit_price = close
                        exit_reason = f"CUT_LOSS_{int(cut_loss_pct*100)}PCT"

                    # Smart Hold - only if loss < 30% and golden near
                    elif loss_percent < 0.30 and hrs_to_golden <= 3:
                        pass  # HOLD

                    # Medium loss, not near golden - cut
                    elif loss_percent >= 0.30 and hrs_to_golden > 3:
                        exit_price = close
                        exit_reason = "CUT_LOSS_NO_GOLDEN"

                # Check reversal
                df_slice = df.slice(max(0, idx - 200), min(201, idx + 1))
                smc_temp = SMCAnalyzer()
                df_slice = smc_temp.calculate_all(df_slice)
                signal = smc_temp.generate_signal(df_slice)

                if signal and signal.confidence >= 0.75:
                    if position["direction"] == "BUY" and signal.signal_type == "SELL":
                        exit_price = close
                        exit_reason = "REVERSAL"
                    elif position["direction"] == "SELL" and signal.signal_type == "BUY":
                        exit_price = close
                        exit_reason = "REVERSAL"

            # Execute exit
            if exit_reason:
                if position["direction"] == "BUY":
                    pnl = (exit_price - position["entry"]) * lot_size * 100
                else:
                    pnl = (position["entry"] - exit_price) * lot_size * 100

                capital += pnl
                hold_hours = (current_time - position["time"]).total_seconds() / 3600

                trades.append(Trade(
                    entry_time=position["time"],
                    exit_time=current_time,
                    direction=position["direction"],
                    entry_price=position["entry"],
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    hold_time_hours=hold_hours,
                    session=position["session"],
                    is_golden=position["is_golden"],
                ))
                position = None

        # Check for new signal
        if position is None and (idx - last_trade_idx) >= min_bars_between_trades:
            df_slice = df.slice(max(0, idx - 200), min(201, idx + 1))
            smc_temp = SMCAnalyzer()
            df_slice = smc_temp.calculate_all(df_slice)
            signal = smc_temp.generate_signal(df_slice)

            if signal and signal.signal_type in ["BUY", "SELL"] and signal.confidence >= confidence_threshold:
                # Get ML prediction
                ml_signal, ml_conf = ml_sim.predict(df, idx)

                should_trade = False

                if system_type == "SMC_ONLY":
                    # SMC-only: always trade on SMC signal
                    should_trade = True

                elif system_type == "ML_SMC_GOLDEN":
                    if is_golden:
                        # Golden Time: require ML+SMC alignment
                        ml_agrees = (
                            (signal.signal_type == "BUY" and ml_signal == "BUY") or
                            (signal.signal_type == "SELL" and ml_signal == "SELL")
                        )
                        should_trade = ml_agrees and ml_conf >= 0.50
                    else:
                        # Non-golden: SMC-only with ML weak filter
                        ml_strongly_disagrees = (
                            (signal.signal_type == "BUY" and ml_signal == "SELL" and ml_conf > 0.65) or
                            (signal.signal_type == "SELL" and ml_signal == "BUY" and ml_conf > 0.65)
                        )
                        should_trade = not ml_strongly_disagrees

                if should_trade:
                    position = {
                        "time": current_time,
                        "direction": signal.signal_type,
                        "entry": signal.entry_price,
                        "tp": signal.take_profit,
                        "conf": signal.confidence,
                        "session": session,
                        "is_golden": is_golden,
                    }
                    last_trade_idx = idx

        # Progress
        if idx % 10000 == 0:
            print(f"    Processing bar {idx}/{len(df)}...")

    return trades


def calc_stats(trades: List[Trade], name: str, initial_capital: float) -> Dict:
    """Calculate statistics for trades."""
    if not trades:
        return {
            "name": name, "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "max_drawdown": 0, "avg_hold_hours": 0,
            "final_capital": initial_capital,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_wins = sum(t.pnl for t in wins) if wins else 0
    total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

    # Calculate drawdown
    capital = initial_capital
    peak = capital
    max_dd = 0
    for t in trades:
        capital += t.pnl
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "name": name,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": 100 * len(wins) / len(trades) if trades else 0,
        "total_pnl": sum(t.pnl for t in trades),
        "avg_win": total_wins / len(wins) if wins else 0,
        "avg_loss": total_losses / len(losses) if losses else 0,
        "profit_factor": total_wins / total_losses if total_losses > 0 else float('inf'),
        "max_drawdown": max_dd,
        "avg_hold_hours": sum(t.hold_time_hours for t in trades) / len(trades) if trades else 0,
        "final_capital": initial_capital + sum(t.pnl for t in trades),
    }


def print_stats(stats: Dict):
    """Print statistics for a system."""
    print(f"\n  {stats['name']}:")
    print(f"    Total Trades: {stats['trades']}")
    print(f"    Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}/{stats['losses']})")
    print(f"    Total P/L: ${stats['total_pnl']:.2f}")
    print(f"    Avg Win: ${stats['avg_win']:.2f}")
    print(f"    Avg Loss: ${stats['avg_loss']:.2f}")
    print(f"    Profit Factor: {stats['profit_factor']:.2f}")
    print(f"    Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"    Avg Hold Time: {stats['avg_hold_hours']:.1f}h")
    print(f"    Final Capital: ${stats['final_capital']:.2f}")


if __name__ == "__main__":
    run_comparison()
