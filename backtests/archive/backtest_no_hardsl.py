"""
BACKTEST NO HARD STOP LOSS - Match Live System
===============================================
Simulates the actual live trading system:
- NO hard stop loss
- Smart Hold logic (hold if loss < 50% max and near golden time)
- Exit on: TP hit, ML reversal, or max loss threshold
- Compare with traditional SL/TP system
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date, time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")


def get_session(dt: datetime) -> Tuple[str, bool]:
    """Get trading session and if it's golden time."""
    hour = dt.hour

    if 19 <= hour < 23:
        return "London-NY Overlap", True  # GOLDEN TIME
    elif 14 <= hour < 19:
        return "London", False
    elif 5 <= hour < 14:
        return "Sydney/Tokyo", False
    else:
        return "Off-hours", False

    return session, is_golden


def hours_to_golden(dt: datetime) -> float:
    """Calculate hours until golden time (19:00 WIB)."""
    current_hour = dt.hour + dt.minute / 60
    golden_start = 19.0

    if 19 <= current_hour < 23:
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


def run_comparison_backtest():
    """Run backtest comparing Hard SL vs No Hard SL systems."""

    print("=" * 80)
    print("BACKTEST COMPARISON: HARD SL vs NO HARD SL (LIVE SYSTEM)")
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

    smc = SMCAnalyzer()
    df = smc.calculate_all(df)

    # Parameters
    lot_size = 0.02
    initial_capital = 5000.0
    max_loss_per_trade = 50.0  # $50 max loss per trade (1% of $5000)
    confidence_threshold = 0.70
    min_bars_between_trades = 4  # Minimum bars between trades

    print("\n[3] Running backtests...")
    print(f"    Lot size: {lot_size}")
    print(f"    Initial capital: ${initial_capital}")
    print(f"    Max loss per trade: ${max_loss_per_trade}")
    print(f"    Confidence threshold: {confidence_threshold*100}%")

    # ========================================
    # SYSTEM A: Traditional Hard SL/TP
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM A: TRADITIONAL (Hard SL from SMC, TP from SMC)")
    print("=" * 80)

    trades_a: List[Trade] = []
    position_a = None
    capital_a = initial_capital
    last_trade_idx_a = -min_bars_between_trades

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

        # Manage position
        if position_a is not None:
            exit_reason = None
            exit_price = None

            if position_a["direction"] == "BUY":
                if low <= position_a["sl"]:
                    exit_price = position_a["sl"]
                    exit_reason = "SL_HIT"
                elif high >= position_a["tp"]:
                    exit_price = position_a["tp"]
                    exit_reason = "TP_HIT"
            else:
                if high >= position_a["sl"]:
                    exit_price = position_a["sl"]
                    exit_reason = "SL_HIT"
                elif low <= position_a["tp"]:
                    exit_price = position_a["tp"]
                    exit_reason = "TP_HIT"

            if exit_reason:
                if position_a["direction"] == "BUY":
                    pnl = (exit_price - position_a["entry"]) * lot_size * 100
                else:
                    pnl = (position_a["entry"] - exit_price) * lot_size * 100

                capital_a += pnl
                hold_hours = (current_time - position_a["time"]).total_seconds() / 3600

                trades_a.append(Trade(
                    entry_time=position_a["time"],
                    exit_time=current_time,
                    direction=position_a["direction"],
                    entry_price=position_a["entry"],
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    hold_time_hours=hold_hours,
                ))
                position_a = None

        # Check for new signal
        if position_a is None and (idx - last_trade_idx_a) >= min_bars_between_trades:
            # Get SMC signal
            df_slice = df.slice(max(0, idx - 200), min(201, idx + 1))
            smc_temp = SMCAnalyzer()
            df_slice = smc_temp.calculate_all(df_slice)
            signal = smc_temp.generate_signal(df_slice)

            if signal and signal.signal_type in ["BUY", "SELL"] and signal.confidence >= confidence_threshold:
                position_a = {
                    "time": current_time,
                    "direction": signal.signal_type,
                    "entry": signal.entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "conf": signal.confidence,
                }
                last_trade_idx_a = idx

    # ========================================
    # SYSTEM B: No Hard SL (Live System)
    # ========================================
    print("\n" + "=" * 80)
    print("SYSTEM B: NO HARD SL (Smart Hold + Max Loss)")
    print("=" * 80)

    trades_b: List[Trade] = []
    position_b = None
    capital_b = initial_capital
    last_trade_idx_b = -min_bars_between_trades

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

        # Manage position - NO HARD SL
        if position_b is not None:
            exit_reason = None
            exit_price = None

            # Calculate current P/L
            if position_b["direction"] == "BUY":
                current_pnl = (close - position_b["entry"]) * lot_size * 100
                # Check TP
                if high >= position_b["tp"]:
                    exit_price = position_b["tp"]
                    exit_reason = "TP_HIT"
            else:
                current_pnl = (position_b["entry"] - close) * lot_size * 100
                # Check TP
                if low <= position_b["tp"]:
                    exit_price = position_b["tp"]
                    exit_reason = "TP_HIT"

            # Smart Hold Logic (if not TP hit)
            if exit_reason is None:
                loss_percent = abs(current_pnl) / max_loss_per_trade if current_pnl < 0 else 0

                # Exit conditions for losing position
                if current_pnl < 0:
                    # 1. Max loss exceeded
                    if abs(current_pnl) >= max_loss_per_trade:
                        exit_price = close
                        exit_reason = "MAX_LOSS"

                    # 2. Smart Hold - keep if loss < 50% and golden time near
                    elif loss_percent < 0.5 and hrs_to_golden <= 4:
                        pass  # HOLD - Smart Hold active

                    # 3. Loss > 50% and not near golden time - cut loss
                    elif loss_percent >= 0.5 and hrs_to_golden > 4:
                        exit_price = close
                        exit_reason = "CUT_LOSS_NO_GOLDEN"

                    # 4. Loss > 80% - cut regardless
                    elif loss_percent >= 0.8:
                        exit_price = close
                        exit_reason = "CUT_LOSS_80PCT"

                # Check for reversal signal
                df_slice = df.slice(max(0, idx - 200), min(201, idx + 1))
                smc_temp = SMCAnalyzer()
                df_slice = smc_temp.calculate_all(df_slice)
                signal = smc_temp.generate_signal(df_slice)

                if signal and signal.confidence >= 0.75:
                    if position_b["direction"] == "BUY" and signal.signal_type == "SELL":
                        exit_price = close
                        exit_reason = "REVERSAL_SIGNAL"
                    elif position_b["direction"] == "SELL" and signal.signal_type == "BUY":
                        exit_price = close
                        exit_reason = "REVERSAL_SIGNAL"

            # Execute exit
            if exit_reason:
                if position_b["direction"] == "BUY":
                    pnl = (exit_price - position_b["entry"]) * lot_size * 100
                else:
                    pnl = (position_b["entry"] - exit_price) * lot_size * 100

                capital_b += pnl
                hold_hours = (current_time - position_b["time"]).total_seconds() / 3600

                trades_b.append(Trade(
                    entry_time=position_b["time"],
                    exit_time=current_time,
                    direction=position_b["direction"],
                    entry_price=position_b["entry"],
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    hold_time_hours=hold_hours,
                ))
                position_b = None

        # Check for new signal
        if position_b is None and (idx - last_trade_idx_b) >= min_bars_between_trades:
            df_slice = df.slice(max(0, idx - 200), min(201, idx + 1))
            smc_temp = SMCAnalyzer()
            df_slice = smc_temp.calculate_all(df_slice)
            signal = smc_temp.generate_signal(df_slice)

            if signal and signal.signal_type in ["BUY", "SELL"] and signal.confidence >= confidence_threshold:
                position_b = {
                    "time": current_time,
                    "direction": signal.signal_type,
                    "entry": signal.entry_price,
                    "tp": signal.take_profit,
                    "conf": signal.confidence,
                }
                last_trade_idx_b = idx

        # Progress
        if idx % 5000 == 0:
            print(f"    Processing bar {idx}/{len(df)}...")

    # ========================================
    # RESULTS COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    def calc_stats(trades: List[Trade], name: str):
        if not trades:
            return {
                "name": name, "trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "max_drawdown": 0, "avg_hold_hours": 0,
                "final_capital": initial_capital,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 0
        max_drawdown = 0
        peak = initial_capital
        running = initial_capital
        for t in trades:
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

        avg_hold = sum(t.hold_time_hours for t in trades) / len(trades) if trades else 0

        return {
            "name": name,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "avg_hold_hours": avg_hold,
            "final_capital": initial_capital + total_pnl,
        }

    stats_a = calc_stats(trades_a, "HARD SL (Traditional)")
    stats_b = calc_stats(trades_b, "NO HARD SL (Live System)")

    # Print comparison table
    print(f"\n{'Metric':<25} {'HARD SL':<20} {'NO HARD SL':<20} {'Diff':<15}")
    print("-" * 80)
    print(f"{'Total Trades':<25} {stats_a['trades']:<20} {stats_b['trades']:<20} {stats_b['trades'] - stats_a['trades']:+}")
    print(f"{'Wins':<25} {stats_a['wins']:<20} {stats_b['wins']:<20} {stats_b['wins'] - stats_a['wins']:+}")
    print(f"{'Losses':<25} {stats_a['losses']:<20} {stats_b['losses']:<20} {stats_b['losses'] - stats_a['losses']:+}")
    print(f"{'Win Rate':<25} {stats_a['win_rate']:.1f}%{'':<17} {stats_b['win_rate']:.1f}%{'':<17} {stats_b['win_rate'] - stats_a['win_rate']:+.1f}%")
    print(f"{'Total P/L':<25} ${stats_a['total_pnl']:,.2f}{'':<13} ${stats_b['total_pnl']:,.2f}{'':<13} ${stats_b['total_pnl'] - stats_a['total_pnl']:+,.2f}")
    print(f"{'Avg Win':<25} ${stats_a['avg_win']:.2f}{'':<15} ${stats_b['avg_win']:.2f}{'':<15}")
    print(f"{'Avg Loss':<25} ${stats_a['avg_loss']:.2f}{'':<14} ${stats_b['avg_loss']:.2f}{'':<14}")
    print(f"{'Profit Factor':<25} {stats_a['profit_factor']:.2f}{'':<18} {stats_b['profit_factor']:.2f}{'':<18}")
    print(f"{'Max Drawdown':<25} {stats_a['max_drawdown']:.1f}%{'':<17} {stats_b['max_drawdown']:.1f}%{'':<17}")
    print(f"{'Avg Hold (hours)':<25} {stats_a['avg_hold_hours']:.1f}{'':<19} {stats_b['avg_hold_hours']:.1f}{'':<19}")
    print(f"{'Final Capital':<25} ${stats_a['final_capital']:,.2f}{'':<11} ${stats_b['final_capital']:,.2f}{'':<11}")

    # Exit reason breakdown for both systems
    print("\n" + "=" * 80)
    print("EXIT REASONS BREAKDOWN")
    print("=" * 80)

    for trades, name in [(trades_a, "HARD SL"), (trades_b, "NO HARD SL")]:
        print(f"\n{name}:")
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0, "wins": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += t.pnl
            if t.pnl > 0:
                exit_reasons[reason]["wins"] += 1

        print(f"{'Exit Reason':<25} {'Count':<10} {'Wins':<10} {'Win%':<10} {'Total P/L':<15}")
        print("-" * 70)
        for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
            win_pct = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
            print(f"{reason:<25} {data['count']:<10} {data['wins']:<10} {win_pct:.1f}%{'':<6} ${data['pnl']:+,.2f}")

    # Verdict
    print("\n" + "=" * 80)
    diff_pnl = stats_b['total_pnl'] - stats_a['total_pnl']
    diff_wr = stats_b['win_rate'] - stats_a['win_rate']
    if diff_pnl > 0:
        print(f"VERDICT: NO HARD SL BETTER (+${diff_pnl:,.2f}, {diff_wr:+.1f}% win rate)")
    else:
        print(f"VERDICT: HARD SL BETTER (+${-diff_pnl:,.2f}, {-diff_wr:+.1f}% win rate)")
    print("=" * 80)

    return stats_a, stats_b


if __name__ == "__main__":
    run_comparison_backtest()
