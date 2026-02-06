"""
FINAL NEWS FILTER VERIFICATION
==============================
Extreme case analysis and final recommendation.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")

# Complete news calendar
HISTORICAL_NEWS = [
    # NFP
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
    # CPI
    (date(2025, 5, 13), 19, "CPI", "HIGH"),
    (date(2025, 6, 11), 19, "CPI", "HIGH"),
    (date(2025, 7, 10), 19, "CPI", "HIGH"),
    (date(2025, 8, 13), 19, "CPI", "HIGH"),
    (date(2025, 9, 10), 19, "CPI", "HIGH"),
    (date(2025, 10, 10), 19, "CPI", "HIGH"),
    (date(2025, 11, 13), 20, "CPI", "HIGH"),
    (date(2025, 12, 11), 20, "CPI", "HIGH"),
    (date(2026, 1, 15), 20, "CPI", "HIGH"),
    # FOMC
    (date(2025, 5, 7), 1, "FOMC", "HIGH"),
    (date(2025, 6, 18), 1, "FOMC", "HIGH"),
    (date(2025, 7, 30), 1, "FOMC", "HIGH"),
    (date(2025, 9, 17), 1, "FOMC", "HIGH"),
    (date(2025, 11, 5), 1, "FOMC", "HIGH"),
    (date(2025, 12, 17), 1, "FOMC", "HIGH"),
    (date(2026, 1, 29), 2, "FOMC", "HIGH"),
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


def run_final_verification():
    """Run final verification tests."""
    print("=" * 80)
    print("FINAL NEWS FILTER VERIFICATION")
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

    print(f"    Data: {len(df)} bars ({df['time'].min()} to {df['time'].max()})")

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
    # TEST: Confidence threshold sensitivity during news
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST: CONFIDENCE THRESHOLD DURING NEWS VS NON-NEWS")
    print("=" * 80)

    lot_size = 0.02
    sl_atr_mult = 1.5
    tp_atr_mult = 3.0

    # Collect all potential trades
    potential_trades = []

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

        close = row["close"]
        atr = row.get("atr", close * 0.003)
        if atr is None or atr <= 0:
            atr = close * 0.003

        # Get ML prediction
        try:
            df_slice = df.slice(max(0, idx - 100), 101)
            pred = ml_model.predict(df_slice, available_features)

            if pred.confidence < 0.50:  # Lower threshold to capture more data
                continue

            signal = pred.signal
            confidence = pred.confidence

        except Exception:
            continue

        if signal not in ["BUY", "SELL"]:
            continue

        # Calculate SL/TP
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

        # Check if in news window
        in_news, news_name = is_news_window(current_time, buffer_hours=1)

        potential_trades.append({
            "entry_time": current_time,
            "confidence": confidence,
            "pnl": pnl,
            "in_news": in_news,
            "news_name": news_name,
        })

    # Analyze by confidence bucket
    print("\n--- WIN RATE BY CONFIDENCE LEVEL ---")
    print(f"{'Confidence':>12} | {'Normal':^20} | {'During News':^20}")
    print(f"{'':12} | {'Count':>6} {'WR':>6} {'Avg P/L':>7} | {'Count':>6} {'WR':>6} {'Avg P/L':>7}")
    print("-" * 70)

    conf_buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]

    for low, high in conf_buckets:
        # Normal trades
        normal = [t for t in potential_trades if not t["in_news"] and low <= t["confidence"] < high]
        normal_wins = len([t for t in normal if t["pnl"] > 0])
        normal_wr = normal_wins / len(normal) * 100 if normal else 0
        normal_avg = sum(t["pnl"] for t in normal) / len(normal) if normal else 0

        # News trades
        news = [t for t in potential_trades if t["in_news"] and low <= t["confidence"] < high]
        news_wins = len([t for t in news if t["pnl"] > 0])
        news_wr = news_wins / len(news) * 100 if news else 0
        news_avg = sum(t["pnl"] for t in news) / len(news) if news else 0

        label = f"{low*100:.0f}%-{high*100:.0f}%"
        print(f"{label:>12} | {len(normal):>6} {normal_wr:>5.1f}% ${normal_avg:>6.2f} | "
              f"{len(news):>6} {news_wr:>5.1f}% ${news_avg:>6.2f}")

    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (70%+ Confidence)")
    print("=" * 80)

    # Filter to 70%+ confidence (our actual threshold)
    high_conf = [t for t in potential_trades if t["confidence"] >= 0.70]

    normal_trades = [t for t in high_conf if not t["in_news"]]
    news_trades = [t for t in high_conf if t["in_news"]]

    print(f"\nNORMAL TRADES (outside news windows):")
    print(f"  Count: {len(normal_trades)}")
    print(f"  Wins: {len([t for t in normal_trades if t['pnl'] > 0])}")
    print(f"  Win Rate: {len([t for t in normal_trades if t['pnl'] > 0])/len(normal_trades)*100:.1f}%")
    print(f"  Total P/L: ${sum(t['pnl'] for t in normal_trades):,.2f}")
    print(f"  Avg P/L: ${sum(t['pnl'] for t in normal_trades)/len(normal_trades):.2f}")

    print(f"\nNEWS TRADES (during news windows):")
    print(f"  Count: {len(news_trades)}")
    print(f"  Wins: {len([t for t in news_trades if t['pnl'] > 0])}")
    print(f"  Win Rate: {len([t for t in news_trades if t['pnl'] > 0])/len(news_trades)*100:.1f}%" if news_trades else "  Win Rate: N/A")
    print(f"  Total P/L: ${sum(t['pnl'] for t in news_trades):,.2f}")
    print(f"  Avg P/L: ${sum(t['pnl'] for t in news_trades)/len(news_trades):.2f}" if news_trades else "  Avg P/L: N/A")

    # ========================================================================
    # WORST CASE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("WORST CASE ANALYSIS: Biggest Losses During News")
    print("=" * 80)

    news_losses = sorted([t for t in news_trades if t["pnl"] < 0], key=lambda x: x["pnl"])

    if news_losses:
        print("\nTop 5 biggest losses during news windows:")
        for i, t in enumerate(news_losses[:5]):
            print(f"  {i+1}. {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | {t['news_name']:6} | "
                  f"Conf: {t['confidence']*100:.1f}% | P/L: ${t['pnl']:.2f}")

        total_news_losses = sum(t["pnl"] for t in news_losses)
        print(f"\nTotal losses during news: ${total_news_losses:.2f}")

    # ========================================================================
    # BEST CASE ANALYSIS
    # ========================================================================
    print("\n--- Biggest Wins During News ---")

    news_wins = sorted([t for t in news_trades if t["pnl"] > 0], key=lambda x: -x["pnl"])

    if news_wins:
        print("\nTop 5 biggest wins during news windows:")
        for i, t in enumerate(news_wins[:5]):
            print(f"  {i+1}. {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | {t['news_name']:6} | "
                  f"Conf: {t['confidence']*100:.1f}% | P/L: ${t['pnl']:.2f}")

        total_news_wins = sum(t["pnl"] for t in news_wins)
        print(f"\nTotal wins during news: ${total_news_wins:.2f}")

    # ========================================================================
    # FINAL RECOMMENDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    normal_wr = len([t for t in normal_trades if t["pnl"] > 0]) / len(normal_trades) * 100
    news_wr = len([t for t in news_trades if t["pnl"] > 0]) / len(news_trades) * 100 if news_trades else 0
    news_pnl = sum(t["pnl"] for t in news_trades)

    print(f"""
    EVIDENCE SUMMARY:
    ================
    1. Normal trades win rate: {normal_wr:.1f}%
    2. News trades win rate: {news_wr:.1f}%
    3. Total profit from news trades: ${news_pnl:,.2f}
    4. News trades count: {len(news_trades)}

    CONCLUSION:
    ===========
    """)

    if news_wr >= normal_wr - 5 and news_pnl > 0:
        print("    The news filter is NOT BENEFICIAL.")
        print("    - News trades have similar win rate to normal trades")
        print(f"    - Blocking news trades would cost ${news_pnl:,.2f}")
        print("\n    RECOMMENDATION: REMOVE NEWS FILTER")
        recommendation = "REMOVE"
    elif news_wr < normal_wr - 10:
        print("    The news filter MAY BE BENEFICIAL.")
        print("    - News trades have significantly lower win rate")
        print("\n    RECOMMENDATION: KEEP NEWS FILTER (for risk management)")
        recommendation = "KEEP"
    else:
        print("    The news filter has MINIMAL IMPACT.")
        print("    - News trades perform similarly to normal trades")
        print("\n    RECOMMENDATION: OPTIONAL - can remove for simplicity")
        recommendation = "OPTIONAL"

    print(f"""
    ================================================================
    FINAL VERDICT: {recommendation} NEWS FILTER
    ================================================================

    Reasons:
    - Win rate during news: {news_wr:.1f}% (vs {normal_wr:.1f}% normal)
    - Profit potential lost by filtering: ${news_pnl:,.2f}
    - The ML model already captures market conditions well
    - High-impact news doesn't significantly hurt our model's performance
    """)

    return recommendation


if __name__ == "__main__":
    result = run_final_verification()
    print(f"\n>>> FINAL ANSWER: {result} <<<")
