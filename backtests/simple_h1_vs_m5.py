"""
Simple Backtest: H1 Bias vs M5 Confirmation
============================================
Simplified comparison focusing on confirmation logic only.
Uses SMC signals without ML to make it faster and clearer.

Author: Claude Opus 4.6
Date: 2026-02-09
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import polars as pl
import numpy as np
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.feature_eng import FeatureEngineer
from src.m5_confirmation import M5ConfirmationAnalyzer

load_dotenv()


def main():
    """Run simple H1 vs M5 backtest."""
    logger.info("="*60)
    logger.info("SIMPLE BACKTEST: H1 Bias vs M5 Confirmation")
    logger.info("="*60)

    # Parameters
    days = 14
    initial_capital = 5000
    lot_size = 0.02
    rr_ratio = 1.5

    # Initialize
    features = FeatureEngineer()
    smc = SMCAnalyzer()
    m5_analyzer = M5ConfirmationAnalyzer(smc, features)

    # Connect MT5
    mt5 = MT5Connector(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH")
    )
    mt5.connect()

    # Fetch data
    logger.info(f"Fetching {days} days of data...")
    bars_m15 = days * 24 * 4
    bars_m5 = days * 24 * 12

    df_m15 = mt5.get_market_data("XAUUSD", "M15", bars_m15)
    df_m5 = mt5.get_market_data("XAUUSD", "M5", bars_m5)
    mt5.disconnect()

    logger.info(f"M15 bars: {len(df_m15)}, M5 bars: {len(df_m5)}")

    # Prepare data
    logger.info("Calculating features and SMC...")
    df_m15 = features.calculate_all(df_m15, include_ml_features=False)
    df_m15 = smc.calculate_all(df_m15)

    df_m5 = features.calculate_all(df_m5, include_ml_features=False)
    df_m5 = smc.calculate_all(df_m5)

    # Create H1 from M15
    df_h1 = df_m15.group_by_dynamic(
        "time",
        every="1h",
        period="1h",
    ).agg([
        pl.first("open").alias("open"),
        pl.max("high").alias("high"),
        pl.min("low").alias("low"),
        pl.last("close").alias("close"),
    ])

    logger.info(f"H1 bars: {len(df_h1)}")

    # --- BACKTEST 1: H1 BIAS ---
    logger.info("\n" + "="*60)
    logger.info("BACKTEST 1: H1 BIAS")
    logger.info("="*60)

    trades_h1 = []
    for i in range(100, len(df_m15)):
        # Update H1 bias every 4 candles
        h1_bias = "NEUTRAL"
        if i % 4 == 0:
            h1_idx = i // 4
            if h1_idx < len(df_h1):
                closes = df_h1["close"][:h1_idx+1].to_list()
                if len(closes) >= 20:
                    price = closes[-1]
                    ema = np.mean(closes[-20:])
                    for c in closes[-19:]:
                        ema = (c - ema) * (2/21) + ema

                    if price > ema * 1.001:
                        h1_bias = "BULLISH"
                    elif price < ema * 0.999:
                        h1_bias = "BEARISH"

        # Get SMC signal
        row = df_m15.row(i, named=True)

        # Simple SMC signal detection
        has_bull_ob = row.get("bullish_ob", False)
        has_bear_ob = row.get("bearish_ob", False)
        bos_bull = row.get("bos_bullish", False)
        bos_bear = row.get("bos_bearish", False)

        signal = None
        if (has_bull_ob or bos_bull) and not (has_bear_ob or bos_bear):
            signal = "BUY"
        elif (has_bear_ob or bos_bear) and not (has_bull_ob or bos_bull):
            signal = "SELL"

        if not signal:
            continue

        # H1 FILTER
        if h1_bias != "NEUTRAL":
            if (signal == "BUY" and h1_bias != "BULLISH") or \
               (signal == "SELL" and h1_bias != "BEARISH"):
                continue  # Blocked

        # Execute trade
        entry = row["close"]
        atr = row.get("atr", 15)
        sl_dist = atr * 1.5
        tp_dist = sl_dist * rr_ratio

        if signal == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
            direction = 1
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
            direction = -1

        # Find exit
        exit_price = None
        exit_reason = None
        for j in range(i+1, min(i+100, len(df_m15))):
            c = df_m15.row(j, named=True)
            if direction == 1:
                if c["low"] <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    break
                elif c["high"] >= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break
            else:
                if c["high"] >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    break
                elif c["low"] <= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break

        if not exit_price:
            exit_price = df_m15["close"][min(i+100, len(df_m15)-1)]
            exit_reason = "TIME"

        pnl = (exit_price - entry) * direction * lot_size * 100

        trades_h1.append({
            "signal": signal,
            "entry": entry,
            "exit": exit_price,
            "reason": exit_reason,
            "pnl": pnl
        })

    # --- BACKTEST 2: M5 CONFIRMATION ---
    logger.info("\n" + "="*60)
    logger.info("BACKTEST 2: M5 CONFIRMATION")
    logger.info("="*60)

    trades_m5 = []
    for i in range(100, len(df_m15)):
        # Get SMC signal
        row = df_m15.row(i, named=True)

        has_bull_ob = row.get("bullish_ob", False)
        has_bear_ob = row.get("bearish_ob", False)
        bos_bull = row.get("bos_bullish", False)
        bos_bear = row.get("bos_bearish", False)

        signal = None
        if (has_bull_ob or bos_bull) and not (has_bear_ob or bos_bear):
            signal = "BUY"
        elif (has_bear_ob or bos_bear) and not (has_bull_ob or bos_bull):
            signal = "SELL"

        if not signal:
            continue

        # M5 CONFIRMATION
        m5_idx = i * 3
        if m5_idx >= len(df_m5):
            continue

        df_m5_slice = df_m5[:m5_idx+1].tail(100)
        m5_conf = m5_analyzer.analyze(df_m5_slice, signal, 0.7)

        if m5_conf.signal == "NEUTRAL":
            continue  # Blocked by M5

        # Execute trade
        entry = row["close"]
        atr = row.get("atr", 15)
        sl_dist = atr * 1.5
        tp_dist = sl_dist * rr_ratio

        if signal == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
            direction = 1
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
            direction = -1

        # Find exit
        exit_price = None
        exit_reason = None
        for j in range(i+1, min(i+100, len(df_m15))):
            c = df_m15.row(j, named=True)
            if direction == 1:
                if c["low"] <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    break
                elif c["high"] >= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break
            else:
                if c["high"] >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    break
                elif c["low"] <= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break

        if not exit_price:
            exit_price = df_m15["close"][min(i+100, len(df_m15)-1)]
            exit_reason = "TIME"

        pnl = (exit_price - entry) * direction * lot_size * 100

        trades_m5.append({
            "signal": signal,
            "entry": entry,
            "exit": exit_price,
            "reason": exit_reason,
            "pnl": pnl
        })

    # --- RESULTS ---
    logger.info("\n" + "="*60)
    logger.info("RESULTS COMPARISON")
    logger.info("="*60)

    def calc_metrics(trades):
        if not trades:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "wr": 0,
                "pnl": 0,
                "avg_win": 0,
                "avg_loss": 0
            }

        total = len(trades)
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]

        return {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "wr": len(wins)/total * 100 if total > 0 else 0,
            "pnl": sum(t["pnl"] for t in trades),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        }

    m_h1 = calc_metrics(trades_h1)
    m_m5 = calc_metrics(trades_m5)

    print("\n{:<20} {:<15} {:<15} {:<15}".format("Metric", "H1 Bias", "M5 Confirm", "Improvement"))
    print("-"*65)
    print(f"{'Total Trades':<20} {m_h1['total']:<15} {m_m5['total']:<15} {m_m5['total']-m_h1['total']:+.0f}")
    print(f"{'Wins':<20} {m_h1['wins']:<15} {m_m5['wins']:<15} {m_m5['wins']-m_h1['wins']:+.0f}")
    print(f"{'Losses':<20} {m_h1['losses']:<15} {m_m5['losses']:<15} {m_m5['losses']-m_h1['losses']:+.0f}")
    print(f"{'Win Rate':<20} {m_h1['wr']:.1f}%{'':<10} {m_m5['wr']:.1f}%{'':<10} {m_m5['wr']-m_h1['wr']:+.1f}%")
    print(f"{'Total P/L':<20} ${m_h1['pnl']:.2f}{'':<9} ${m_m5['pnl']:.2f}{'':<9} ${m_m5['pnl']-m_h1['pnl']:+.2f}")
    print(f"{'Avg Win':<20} ${m_h1['avg_win']:.2f}{'':<9} ${m_m5['avg_win']:.2f}{'':<9} ${m_m5['avg_win']-m_h1['avg_win']:+.2f}")
    print(f"{'Avg Loss':<20} ${m_h1['avg_loss']:.2f}{'':<9} ${m_m5['avg_loss']:.2f}{'':<9} ${m_m5['avg_loss']-m_h1['avg_loss']:+.2f}")
    print(f"{'Profit Factor':<20} {m_h1['profit_factor']:.2f}{'':<12} {m_m5['profit_factor']:.2f}{'':<12} {m_m5['profit_factor']-m_h1['profit_factor']:+.2f}")
    print("="*65)

    # Save
    output_dir = Path("backtests/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    output_file = output_dir / f"h1_vs_m5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "h1_bias": m_h1,
            "m5_confirmation": m_m5,
            "trades_h1": trades_h1,
            "trades_m5": trades_m5
        }, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\n✅ BACKTEST COMPLETE!")

    return m_h1, m_m5


if __name__ == "__main__":
    main()
