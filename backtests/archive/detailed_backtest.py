"""
DETAILED BACKTEST WITH TRADE-BY-TRADE OUTPUT
=============================================
Verifikasi backtest dengan menampilkan setiap trade.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")

# News events
HISTORICAL_NEWS = [
    (date(2025, 5, 2), 19, "NFP", "HIGH"),
    (date(2025, 6, 6), 19, "NFP", "HIGH"),
    (date(2025, 7, 3), 19, "NFP", "HIGH"),
    (date(2025, 8, 1), 19, "NFP", "HIGH"),
    (date(2025, 9, 5), 19, "NFP", "HIGH"),
    (date(2025, 10, 3), 19, "NFP", "HIGH"),
    (date(2025, 11, 7), 19, "NFP", "HIGH"),
    (date(2025, 12, 5), 19, "NFP", "HIGH"),
    (date(2026, 1, 10), 20, "NFP", "HIGH"),
    (date(2026, 2, 5), 20, "NFP", "HIGH"),
    # FOMC
    (date(2025, 5, 7), 1, "FOMC", "HIGH"),
    (date(2025, 6, 18), 1, "FOMC", "HIGH"),
    (date(2025, 7, 30), 1, "FOMC", "HIGH"),
    (date(2025, 9, 17), 1, "FOMC", "HIGH"),
    (date(2025, 11, 5), 1, "FOMC", "HIGH"),
    (date(2025, 12, 17), 1, "FOMC", "HIGH"),
    (date(2026, 1, 29), 2, "FOMC", "HIGH"),
]


def is_news_blocked(dt: datetime) -> Tuple[bool, str]:
    """Check if within +/-1h of HIGH impact news."""
    current_date = dt.date()
    current_hour = dt.hour

    for news_date, news_hour, name, impact in HISTORICAL_NEWS:
        if news_date == current_date and impact == "HIGH":
            if abs(current_hour - news_hour) <= 1:
                return True, name
    return False, ""


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


def run_detailed_backtest():
    """Run backtest with detailed output."""
    print("=" * 80)
    print("DETAILED BACKTEST - TRADE BY TRADE VERIFICATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
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
    print(f"    Total columns: {len(df.columns)}")

    # Load ML model
    print("\n[3] Loading ML model...")
    ml_model = TradingModel(model_path="models/xgboost_model.pkl")
    ml_model.load()

    available_features = [f for f in ml_model.feature_names if f in df.columns]
    print(f"    Features: {len(available_features)}/{len(ml_model.feature_names)}")

    # Backtest parameters
    lot_size = 0.02
    initial_capital = 5000.0
    sl_atr_mult = 1.5
    tp_atr_mult = 3.0

    print("\n[4] Running backtest...")
    print(f"    Lot size: {lot_size}")
    print(f"    Initial capital: ${initial_capital}")
    print(f"    SL: {sl_atr_mult}x ATR, TP: {tp_atr_mult}x ATR")

    # === BACKTEST WITHOUT NEWS FILTER ===
    print("\n" + "=" * 80)
    print("SCENARIO A: WITHOUT NEWS FILTER")
    print("=" * 80)

    trades_no_filter: List[Trade] = []
    position = None
    capital = initial_capital
    signals_checked = 0
    signals_valid = 0

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

                trades_no_filter.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=current_time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    pnl=pnl,
                    confidence=position["confidence"],
                    exit_reason=exit_reason,
                ))
                capital += pnl
                position = None

        if position is not None:
            continue

        # Session filter (14:00-23:00 WIB only)
        hour = current_time.hour
        if hour < 14 or hour > 23:
            continue

        signals_checked += 1

        # ML Prediction
        try:
            df_slice = df.slice(max(0, idx - 100), 101)
            pred = ml_model.predict(df_slice, available_features)

            if pred.confidence < 0.70:
                continue

            signals_valid += 1
            signal = pred.signal
            confidence = pred.confidence

        except Exception as e:
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

    # Print trades
    print(f"\nSignals checked: {signals_checked}")
    print(f"Valid signals (>=70%): {signals_valid}")
    print(f"Total trades: {len(trades_no_filter)}")

    if trades_no_filter:
        print("\n--- TRADE LIST (first 20) ---")
        for i, t in enumerate(trades_no_filter[:20]):
            win = "WIN" if t.pnl > 0 else "LOSS"
            print(f"{i+1:3}. {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.direction:4} | "
                  f"Entry: {t.entry_price:.2f} | Exit: {t.exit_price:.2f} | "
                  f"{t.exit_reason} | P/L: ${t.pnl:+.2f} | {win}")

        if len(trades_no_filter) > 20:
            print(f"... and {len(trades_no_filter) - 20} more trades ...")

    # Calculate stats
    wins = [t for t in trades_no_filter if t.pnl > 0]
    losses = [t for t in trades_no_filter if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades_no_filter)
    win_rate = len(wins) / len(trades_no_filter) * 100 if trades_no_filter else 0

    print(f"\n--- SUMMARY (NO FILTER) ---")
    print(f"Total Trades: {len(trades_no_filter)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P/L: ${total_pnl:,.2f}")
    print(f"Final Capital: ${initial_capital + total_pnl:,.2f}")

    # === BACKTEST WITH NEWS FILTER ===
    print("\n" + "=" * 80)
    print("SCENARIO B: WITH NEWS FILTER (+/-1h HIGH impact)")
    print("=" * 80)

    trades_with_filter: List[Trade] = []
    position = None
    capital = initial_capital
    news_blocked = 0

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

        # Manage position (same as before)
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

                trades_with_filter.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=current_time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    pnl=pnl,
                    confidence=position["confidence"],
                    exit_reason=exit_reason,
                ))
                capital += pnl
                position = None

        if position is not None:
            continue

        # Session filter
        hour = current_time.hour
        if hour < 14 or hour > 23:
            continue

        # NEWS FILTER
        blocked, news_name = is_news_blocked(current_time)
        if blocked:
            news_blocked += 1
            continue

        # ML Prediction
        try:
            df_slice = df.slice(max(0, idx - 100), 101)
            pred = ml_model.predict(df_slice, available_features)

            if pred.confidence < 0.70:
                continue

            signal = pred.signal
            confidence = pred.confidence

        except Exception as e:
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

    print(f"\nNews blocked entries: {news_blocked}")
    print(f"Total trades: {len(trades_with_filter)}")

    # Calculate stats
    wins2 = [t for t in trades_with_filter if t.pnl > 0]
    losses2 = [t for t in trades_with_filter if t.pnl <= 0]
    total_pnl2 = sum(t.pnl for t in trades_with_filter)
    win_rate2 = len(wins2) / len(trades_with_filter) * 100 if trades_with_filter else 0

    print(f"\n--- SUMMARY (WITH FILTER) ---")
    print(f"Total Trades: {len(trades_with_filter)}")
    print(f"Wins: {len(wins2)} | Losses: {len(losses2)}")
    print(f"Win Rate: {win_rate2:.1f}%")
    print(f"Total P/L: ${total_pnl2:,.2f}")
    print(f"Final Capital: ${initial_capital + total_pnl2:,.2f}")

    # === COMPARISON ===
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"""
                        NO FILTER       WITH FILTER     DIFFERENCE
    -----------------------------------------------------------------
    Total Trades        {len(trades_no_filter):<15} {len(trades_with_filter):<15} {len(trades_with_filter) - len(trades_no_filter):+d}
    Win Rate            {win_rate:<14.1f}% {win_rate2:<14.1f}% {win_rate2 - win_rate:+.1f}%
    Total P/L           ${total_pnl:<13,.2f} ${total_pnl2:<13,.2f} ${total_pnl2 - total_pnl:+,.2f}
    Final Capital       ${initial_capital + total_pnl:<13,.2f} ${initial_capital + total_pnl2:<13,.2f}
    """)

    # Verdict
    print("=" * 80)
    if total_pnl2 > total_pnl:
        print("VERDICT: NEWS FILTER BENEFICIAL (+${:.2f})".format(total_pnl2 - total_pnl))
    elif total_pnl2 < total_pnl:
        print("VERDICT: NEWS FILTER NOT BENEFICIAL (-${:.2f})".format(total_pnl - total_pnl2))
    else:
        print("VERDICT: NEWS FILTER HAS NO IMPACT")
    print("=" * 80)


if __name__ == "__main__":
    run_detailed_backtest()
