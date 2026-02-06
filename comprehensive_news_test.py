"""
COMPREHENSIVE NEWS FILTER VERIFICATION
=======================================
Multiple test scenarios to verify news filter effectiveness.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import time
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")

# Complete news calendar with exact dates
HISTORICAL_NEWS = [
    # NFP (Non-Farm Payrolls) - First Friday each month at 19:30 WIB
    (date(2025, 5, 2), 19, "NFP", "HIGH"),
    (date(2025, 6, 6), 19, "NFP", "HIGH"),
    (date(2025, 7, 3), 19, "NFP", "HIGH"),
    (date(2025, 8, 1), 19, "NFP", "HIGH"),
    (date(2025, 9, 5), 19, "NFP", "HIGH"),
    (date(2025, 10, 3), 19, "NFP", "HIGH"),
    (date(2025, 11, 7), 19, "NFP", "HIGH"),
    (date(2025, 12, 5), 19, "NFP", "HIGH"),
    (date(2026, 1, 10), 20, "NFP", "HIGH"),
    (date(2026, 2, 7), 20, "NFP", "HIGH"),
    # FOMC (Federal Reserve)
    (date(2025, 5, 7), 1, "FOMC", "HIGH"),
    (date(2025, 6, 18), 1, "FOMC", "HIGH"),
    (date(2025, 7, 30), 1, "FOMC", "HIGH"),
    (date(2025, 9, 17), 1, "FOMC", "HIGH"),
    (date(2025, 11, 5), 1, "FOMC", "HIGH"),
    (date(2025, 12, 17), 1, "FOMC", "HIGH"),
    (date(2026, 1, 29), 2, "FOMC", "HIGH"),
    # CPI (Consumer Price Index)
    (date(2025, 5, 13), 19, "CPI", "HIGH"),
    (date(2025, 6, 11), 19, "CPI", "HIGH"),
    (date(2025, 7, 10), 19, "CPI", "HIGH"),
    (date(2025, 8, 13), 19, "CPI", "HIGH"),
    (date(2025, 9, 10), 19, "CPI", "HIGH"),
    (date(2025, 10, 10), 19, "CPI", "HIGH"),
    (date(2025, 11, 13), 20, "CPI", "HIGH"),
    (date(2025, 12, 11), 20, "CPI", "HIGH"),
    (date(2026, 1, 15), 20, "CPI", "HIGH"),
]


def is_news_window(dt: datetime, buffer_hours: int = 1) -> Tuple[bool, str]:
    """Check if within buffer hours of HIGH impact news."""
    current_date = dt.date()
    current_hour = dt.hour

    for news_date, news_hour, name, impact in HISTORICAL_NEWS:
        if news_date == current_date and impact == "HIGH":
            if abs(current_hour - news_hour) <= buffer_hours:
                return True, name
    return False, ""


def get_news_on_date(dt: date) -> List[Tuple[int, str]]:
    """Get all news events on a specific date."""
    events = []
    for news_date, news_hour, name, impact in HISTORICAL_NEWS:
        if news_date == dt:
            events.append((news_hour, name))
    return events


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    confidence: float
    exit_reason: str
    news_blocked: bool = False
    news_name: str = ""


def run_comprehensive_test():
    """Run multiple test scenarios."""
    print("=" * 80)
    print("COMPREHENSIVE NEWS FILTER VERIFICATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data and models...")
    import MetaTrader5 as mt5
    from src.config import get_config
    from src.feature_eng import FeatureEngineer
    from src.smc_polars import SMCAnalyzer
    from src.regime_detector import MarketRegimeDetector
    from src.ml_model import TradingModel

    config = get_config()
    mt5.initialize(path=config.mt5_path, login=config.mt5_login,
                   password=config.mt5_password, server=config.mt5_server)
    mt5.symbol_select("XAUUSD", True)
    time.sleep(0.5)

    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 60000)
    mt5.shutdown()

    df = pl.DataFrame({
        "time": [datetime.fromtimestamp(r[0]) for r in rates],
        "open": [r[1] for r in rates],
        "high": [r[2] for r in rates],
        "low": [r[3] for r in rates],
        "close": [r[4] for r in rates],
        "volume": [float(r[5]) for r in rates],
    })

    print(f"    Loaded {len(df)} bars")
    print(f"    Range: {df['time'].min()} to {df['time'].max()}")

    # Calculate features
    print("\n[2] Calculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=True)

    smc = SMCAnalyzer()
    df = smc.calculate_all(df)

    regime = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    regime.load()
    df = regime.predict(df)

    # Load ML model
    print("\n[3] Loading ML model...")
    ml_model = TradingModel(model_path="models/xgboost_model.pkl")
    ml_model.load()

    available_features = [f for f in ml_model.feature_names if f in df.columns]
    print(f"    Features: {len(available_features)}/{len(ml_model.feature_names)}")

    # ========================================================================
    # TEST 1: Analyze trades blocked by news filter
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: ANALYZING BLOCKED TRADES DURING NEWS WINDOWS")
    print("=" * 80)

    lot_size = 0.02
    sl_atr_mult = 1.5
    tp_atr_mult = 3.0

    blocked_trades: List[Trade] = []

    for idx in range(200, len(df) - 1):
        row = df.row(idx, named=True)
        current_time = row["time"]

        if current_time.date() < date(2025, 5, 22):
            continue
        if current_time.date() > date(2026, 2, 5):
            break

        # Session filter
        hour = current_time.hour
        if hour < 14 or hour > 23:
            continue

        # Check if in news window
        in_news, news_name = is_news_window(current_time, buffer_hours=1)
        if not in_news:
            continue

        close = row["close"]
        atr = row.get("atr", close * 0.003)
        if atr is None or atr <= 0:
            atr = close * 0.003

        # Get ML prediction
        try:
            df_slice = df.slice(max(0, idx - 100), 101)
            pred = ml_model.predict(df_slice, available_features)

            if pred.confidence < 0.70:
                continue

            signal = pred.signal
            confidence = pred.confidence

        except Exception:
            continue

        if signal not in ["BUY", "SELL"]:
            continue

        # Simulate what would have happened if we traded
        entry_price = close
        if signal == "BUY":
            sl = close - (atr * sl_atr_mult)
            tp = close + (atr * tp_atr_mult)
        else:
            sl = close + (atr * sl_atr_mult)
            tp = close - (atr * tp_atr_mult)

        # Look forward to find exit
        exit_price = None
        exit_time = None
        exit_reason = None

        for future_idx in range(idx + 1, min(idx + 200, len(df))):
            future_row = df.row(future_idx, named=True)
            future_high = future_row["high"]
            future_low = future_row["low"]

            if signal == "BUY":
                if future_low <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_time = future_row["time"]
                    break
                elif future_high >= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_time = future_row["time"]
                    break
            else:
                if future_high >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_time = future_row["time"]
                    break
                elif future_low <= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_time = future_row["time"]
                    break

        if exit_price is None:
            continue

        # Calculate P/L
        if signal == "BUY":
            pnl = (exit_price - entry_price) * lot_size * 100
        else:
            pnl = (entry_price - exit_price) * lot_size * 100

        blocked_trades.append(Trade(
            entry_time=current_time,
            exit_time=exit_time,
            direction=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            confidence=confidence,
            exit_reason=exit_reason,
            news_blocked=True,
            news_name=news_name,
        ))

    print(f"\nTrades that WOULD have happened during news windows: {len(blocked_trades)}")

    if blocked_trades:
        print("\n--- BLOCKED TRADE DETAILS ---")
        for i, t in enumerate(blocked_trades):
            win = "WIN" if t.pnl > 0 else "LOSS"
            print(f"{i+1:3}. {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.news_name:6} | {t.direction:4} | "
                  f"Entry: {t.entry_price:.2f} | Exit: {t.exit_price:.2f} | "
                  f"{t.exit_reason} | P/L: ${t.pnl:+.2f} | {win}")

        wins = [t for t in blocked_trades if t.pnl > 0]
        losses = [t for t in blocked_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in blocked_trades)
        win_rate = len(wins) / len(blocked_trades) * 100

        print(f"\n--- BLOCKED TRADES SUMMARY ---")
        print(f"Total: {len(blocked_trades)} trades")
        print(f"Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P/L if traded: ${total_pnl:+.2f}")

        if total_pnl < 0:
            print("\n>>> NEWS FILTER PROTECTED US FROM ${:.2f} LOSS <<<".format(abs(total_pnl)))
        else:
            print("\n>>> NEWS FILTER COST US ${:.2f} PROFIT <<<".format(total_pnl))

    # ========================================================================
    # TEST 2: Different buffer periods
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: COMPARING DIFFERENT BUFFER PERIODS")
    print("=" * 80)

    buffer_results = {}

    for buffer_hours in [0, 1, 2, 3]:
        trades: List[Trade] = []
        position = None

        for idx in range(200, len(df) - 1):
            row = df.row(idx, named=True)
            current_time = row["time"]

            if current_time.date() < date(2025, 5, 22):
                continue
            if current_time.date() > date(2026, 2, 5):
                break

            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr = row.get("atr", close * 0.003)
            if atr is None or atr <= 0:
                atr = close * 0.003

            # Manage position
            if position is not None:
                exit_reason = None
                exit_price = None

                if position["direction"] == "BUY":
                    if low <= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "SL"
                    elif high >= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "TP"
                else:
                    if high >= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "SL"
                    elif low <= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "TP"

                if exit_reason:
                    if position["direction"] == "BUY":
                        pnl = (exit_price - position["entry_price"]) * lot_size * 100
                    else:
                        pnl = (position["entry_price"] - exit_price) * lot_size * 100

                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        pnl=pnl,
                        confidence=position["confidence"],
                        exit_reason=exit_reason,
                    ))
                    position = None

            if position is not None:
                continue

            # Session filter
            hour = current_time.hour
            if hour < 14 or hour > 23:
                continue

            # News filter (if buffer > 0)
            if buffer_hours > 0:
                in_news, _ = is_news_window(current_time, buffer_hours=buffer_hours)
                if in_news:
                    continue

            # ML Prediction
            try:
                df_slice = df.slice(max(0, idx - 100), 101)
                pred = ml_model.predict(df_slice, available_features)

                if pred.confidence < 0.70:
                    continue

                signal = pred.signal
                confidence = pred.confidence

            except Exception:
                continue

            # Entry
            if signal == "BUY":
                sl = close - (atr * sl_atr_mult)
                tp = close + (atr * tp_atr_mult)
                position = {
                    "direction": "BUY",
                    "entry_price": close,
                    "entry_time": current_time,
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                }
            elif signal == "SELL":
                sl = close + (atr * sl_atr_mult)
                tp = close - (atr * tp_atr_mult)
                position = {
                    "direction": "SELL",
                    "entry_price": close,
                    "entry_time": current_time,
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                }

        wins = [t for t in trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        buffer_results[buffer_hours] = {
            "trades": len(trades),
            "wins": len(wins),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        }

    print("\n--- BUFFER COMPARISON ---")
    print(f"{'Buffer':>10} | {'Trades':>8} | {'Wins':>6} | {'Win Rate':>10} | {'Total P/L':>12}")
    print("-" * 60)

    for buffer_hours, result in buffer_results.items():
        label = "No Filter" if buffer_hours == 0 else f"+/-{buffer_hours}h"
        print(f"{label:>10} | {result['trades']:>8} | {result['wins']:>6} | "
              f"{result['win_rate']:>9.1f}% | ${result['total_pnl']:>11,.2f}")

    # ========================================================================
    # TEST 3: Monthly breakdown
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: MONTHLY PERFORMANCE COMPARISON")
    print("=" * 80)

    # Run full backtest and track by month
    monthly_results: Dict[str, Dict[str, Dict]] = {}

    for filter_mode in ["NO_FILTER", "WITH_FILTER"]:
        trades: List[Trade] = []
        position = None

        for idx in range(200, len(df) - 1):
            row = df.row(idx, named=True)
            current_time = row["time"]

            if current_time.date() < date(2025, 5, 22):
                continue
            if current_time.date() > date(2026, 2, 5):
                break

            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr = row.get("atr", close * 0.003)
            if atr is None or atr <= 0:
                atr = close * 0.003

            # Manage position
            if position is not None:
                exit_reason = None
                exit_price = None

                if position["direction"] == "BUY":
                    if low <= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "SL"
                    elif high >= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "TP"
                else:
                    if high >= position["sl"]:
                        exit_price = position["sl"]
                        exit_reason = "SL"
                    elif low <= position["tp"]:
                        exit_price = position["tp"]
                        exit_reason = "TP"

                if exit_reason:
                    if position["direction"] == "BUY":
                        pnl = (exit_price - position["entry_price"]) * lot_size * 100
                    else:
                        pnl = (position["entry_price"] - exit_price) * lot_size * 100

                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        pnl=pnl,
                        confidence=position["confidence"],
                        exit_reason=exit_reason,
                    ))
                    position = None

            if position is not None:
                continue

            # Session filter
            hour = current_time.hour
            if hour < 14 or hour > 23:
                continue

            # News filter (only for WITH_FILTER)
            if filter_mode == "WITH_FILTER":
                in_news, _ = is_news_window(current_time, buffer_hours=1)
                if in_news:
                    continue

            # ML Prediction
            try:
                df_slice = df.slice(max(0, idx - 100), 101)
                pred = ml_model.predict(df_slice, available_features)

                if pred.confidence < 0.70:
                    continue

                signal = pred.signal
                confidence = pred.confidence

            except Exception:
                continue

            # Entry
            if signal == "BUY":
                sl = close - (atr * sl_atr_mult)
                tp = close + (atr * tp_atr_mult)
                position = {
                    "direction": "BUY",
                    "entry_price": close,
                    "entry_time": current_time,
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                }
            elif signal == "SELL":
                sl = close + (atr * sl_atr_mult)
                tp = close - (atr * tp_atr_mult)
                position = {
                    "direction": "SELL",
                    "entry_price": close,
                    "entry_time": current_time,
                    "sl": sl,
                    "tp": tp,
                    "confidence": confidence,
                }

        # Group by month
        for trade in trades:
            month_key = trade.entry_time.strftime("%Y-%m")
            if month_key not in monthly_results:
                monthly_results[month_key] = {"NO_FILTER": [], "WITH_FILTER": []}
            monthly_results[month_key][filter_mode].append(trade)

    print("\n--- MONTHLY BREAKDOWN ---")
    print(f"{'Month':<10} | {'NO FILTER':^25} | {'WITH FILTER':^25} | {'Diff':>10}")
    print(f"{'':10} | {'Trades':>8} {'WR':>7} {'P/L':>9} | {'Trades':>8} {'WR':>7} {'P/L':>9} | {'':>10}")
    print("-" * 85)

    total_diff = 0
    for month in sorted(monthly_results.keys()):
        no_filter = monthly_results[month]["NO_FILTER"]
        with_filter = monthly_results[month]["WITH_FILTER"]

        nf_trades = len(no_filter)
        nf_wins = len([t for t in no_filter if t.pnl > 0])
        nf_wr = nf_wins / nf_trades * 100 if nf_trades > 0 else 0
        nf_pnl = sum(t.pnl for t in no_filter)

        wf_trades = len(with_filter)
        wf_wins = len([t for t in with_filter if t.pnl > 0])
        wf_wr = wf_wins / wf_trades * 100 if wf_trades > 0 else 0
        wf_pnl = sum(t.pnl for t in with_filter)

        diff = wf_pnl - nf_pnl
        total_diff += diff

        print(f"{month:<10} | {nf_trades:>8} {nf_wr:>6.1f}% ${nf_pnl:>7.0f} | "
              f"{wf_trades:>8} {wf_wr:>6.1f}% ${wf_pnl:>7.0f} | ${diff:>+9.0f}")

    print("-" * 85)
    print(f"{'TOTAL':>10} | {' ' * 25} | {' ' * 25} | ${total_diff:>+9.0f}")

    # ========================================================================
    # TEST 4: Analyze trades around specific news events
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: TRADES AROUND SPECIFIC NEWS EVENTS")
    print("=" * 80)

    # Get all trades without filter
    all_trades: List[Trade] = []
    position = None

    for idx in range(200, len(df) - 1):
        row = df.row(idx, named=True)
        current_time = row["time"]

        if current_time.date() < date(2025, 5, 22):
            continue
        if current_time.date() > date(2026, 2, 5):
            break

        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row.get("atr", close * 0.003)
        if atr is None or atr <= 0:
            atr = close * 0.003

        # Manage position
        if position is not None:
            exit_reason = None
            exit_price = None

            if position["direction"] == "BUY":
                if low <= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "SL"
                elif high >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP"
            else:
                if high >= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "SL"
                elif low <= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TP"

            if exit_reason:
                if position["direction"] == "BUY":
                    pnl = (exit_price - position["entry_price"]) * lot_size * 100
                else:
                    pnl = (position["entry_price"] - exit_price) * lot_size * 100

                # Check if this trade was in a news window
                in_news, news_name = is_news_window(position["entry_time"], buffer_hours=1)

                all_trades.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=current_time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    pnl=pnl,
                    confidence=position["confidence"],
                    exit_reason=exit_reason,
                    news_blocked=in_news,
                    news_name=news_name if in_news else "",
                ))
                position = None

        if position is not None:
            continue

        # Session filter
        hour = current_time.hour
        if hour < 14 or hour > 23:
            continue

        # ML Prediction (no news filter)
        try:
            df_slice = df.slice(max(0, idx - 100), 101)
            pred = ml_model.predict(df_slice, available_features)

            if pred.confidence < 0.70:
                continue

            signal = pred.signal
            confidence = pred.confidence

        except Exception:
            continue

        # Entry
        if signal == "BUY":
            sl = close - (atr * sl_atr_mult)
            tp = close + (atr * tp_atr_mult)
            position = {
                "direction": "BUY",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }
        elif signal == "SELL":
            sl = close + (atr * sl_atr_mult)
            tp = close - (atr * tp_atr_mult)
            position = {
                "direction": "SELL",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }

    # Analyze by news type
    news_trades = [t for t in all_trades if t.news_blocked]

    if news_trades:
        print("\n--- TRADES DURING NEWS WINDOWS (By Event Type) ---")

        by_event: Dict[str, List[Trade]] = {}
        for t in news_trades:
            if t.news_name not in by_event:
                by_event[t.news_name] = []
            by_event[t.news_name].append(t)

        for event_name, event_trades in sorted(by_event.items()):
            wins = len([t for t in event_trades if t.pnl > 0])
            total_pnl = sum(t.pnl for t in event_trades)
            wr = wins / len(event_trades) * 100

            print(f"\n{event_name}:")
            print(f"  Trades: {len(event_trades)}, Wins: {wins}, Win Rate: {wr:.1f}%")
            print(f"  Total P/L: ${total_pnl:+.2f}")

            for t in event_trades:
                result = "WIN" if t.pnl > 0 else "LOSS"
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.direction} | "
                      f"${t.pnl:+.2f} | {result}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPREHENSIVE SUMMARY")
    print("=" * 80)

    baseline = buffer_results[0]
    filtered = buffer_results[1]

    print(f"""
    BASELINE (No Filter):
        Total Trades: {baseline['trades']}
        Win Rate: {baseline['win_rate']:.1f}%
        Total P/L: ${baseline['total_pnl']:,.2f}

    WITH NEWS FILTER (+/-1h):
        Total Trades: {filtered['trades']}
        Win Rate: {filtered['win_rate']:.1f}%
        Total P/L: ${filtered['total_pnl']:,.2f}

    IMPACT ANALYSIS:
        Trades Blocked: {baseline['trades'] - filtered['trades']}
        Win Rate Change: {filtered['win_rate'] - baseline['win_rate']:+.1f}%
        P/L Change: ${filtered['total_pnl'] - baseline['total_pnl']:+,.2f}
    """)

    # Verdict
    pnl_diff = filtered['total_pnl'] - baseline['total_pnl']
    wr_diff = filtered['win_rate'] - baseline['win_rate']

    print("=" * 80)
    if pnl_diff > 50:  # Significant positive impact
        print("VERDICT: NEWS FILTER IS BENEFICIAL")
        print(f"         Improved P/L by ${pnl_diff:+.2f}")
    elif pnl_diff < -50:  # Significant negative impact
        print("VERDICT: NEWS FILTER IS NOT BENEFICIAL")
        print(f"         Reduced P/L by ${abs(pnl_diff):.2f}")
    else:  # Minimal impact
        print("VERDICT: NEWS FILTER HAS MINIMAL IMPACT")
        print(f"         P/L difference: ${pnl_diff:+.2f} (negligible)")
        if wr_diff > 0:
            print(f"         However, win rate improved by {wr_diff:.1f}%")
            print("         RECOMMENDATION: Keep filter for risk management")
        else:
            print("         RECOMMENDATION: Filter provides no significant benefit")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_test()
