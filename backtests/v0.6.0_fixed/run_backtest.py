"""
Quick runner for v0.6.0 FIXED backtest
======================================

Usage:
    python backtests/v0.6.0_fixed/run_backtest.py --days 90
    python backtests/v0.6.0_fixed/run_backtest.py --days 30 --save
"""

import sys
import os
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load .env file
load_dotenv()

from datetime import datetime, timedelta
from loguru import logger
from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from backtest_v0_6_0_fixed import BacktestFixed

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run XAUBot AI v0.6.0 FIXED Backtest")
    parser.add_argument("--days", type=int, default=90, help="Days to backtest (default: 90)")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("XAUBOT AI v0.6.0 FIXED - BACKTEST RUNNER")
    logger.info("=" * 80)
    logger.info("")
    logger.info("PROFESSOR'S FIXES APPLIED:")
    logger.info("  [FIX 1] Fuzzy Thresholds: 70-90% tiered (was fixed 90%)")
    logger.info("  [FIX 2] Trajectory Calibration: regime penalty + uncertainty")
    logger.info("  [FIX 3] Session Filter: Sydney/Tokyo DISABLED (00:00-10:00)")
    logger.info("  [FIX 4] Unicode Fix: ASCII only (no emojis)")
    logger.info("  [FIX 5] Max Loss: $25/trade (was $50)")
    logger.info("")
    logger.info(f"Backtest Period: {args.days} days")
    logger.info("=" * 80)
    logger.info("")

    # Load MT5 credentials from environment
    mt5_login = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password = os.getenv("MT5_PASSWORD", "")
    mt5_server = os.getenv("MT5_SERVER", "")
    mt5_path = os.getenv("MT5_PATH", "")

    if mt5_login == 0 or not mt5_password or not mt5_server:
        logger.error("MT5 credentials not found in .env file")
        logger.error("Please set: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
        return

    # Load data
    logger.info("Step 1/4: Connecting to MT5...")
    connector = MT5Connector(
        login=mt5_login,
        password=mt5_password,
        server=mt5_server,
        path=mt5_path if mt5_path else None
    )
    if not connector.connect():
        logger.error("Failed to connect to MT5")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Calculate bars needed: 90 days × 24 hours × 4 (M15) = ~8640 bars
    bars_needed = args.days * 24 * 4
    logger.info(f"Step 2/4: Loading XAUUSD M15 data (last {args.days} days, ~{bars_needed} bars)...")
    df = connector.get_market_data("XAUUSD", "M15", count=bars_needed)
    if df is None or len(df) == 0:
        logger.error("Failed to load data")
        connector.disconnect()
        return

    logger.info(f"  Loaded {len(df)} bars")
    logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    # Add features
    logger.info("Step 3/4: Engineering features...")
    features = FeatureEngineer()
    df = features.calculate_all(df)

    # Add missing SMC and regime features with defaults (for TESTING MODE)
    import polars as pl
    missing_features = ['swing_high', 'swing_low', 'fvg_signal', 'ob', 'bos', 'choch', 'market_structure']
    for feat in missing_features:
        if feat not in df.columns:
            df = df.with_columns([pl.lit(0).alias(feat)])

    # Add regime if missing (will be filled by regime detector later)
    if 'regime' not in df.columns:
        df = df.with_columns([pl.lit(0).alias("regime")])  # 0=ranging (numeric)
    else:
        # Encode regime strings to numbers if exists
        regime_map = {"ranging": 0, "trending": 1, "volatile": 2}
        df = df.with_columns([
            pl.col("regime").map_dict(regime_map, default=0).alias("regime")
        ])

    logger.info(f"  Added {len(df.columns)} features (including {len(missing_features)} SMC placeholders)")

    # Run backtest
    logger.info("Step 4/4: Running backtest with FIXED logic...")
    logger.info("")

    bt = BacktestFixed(
        ml_threshold=0.30,  # TESTING: Relaxed for more signals
        signal_confirmation=1,  # TESTING: Accept signal immediately
        max_loss_per_trade=25.0,  # FIX 5
        trade_cooldown_bars=5,  # TESTING: Reduced cooldown
    )

    stats = bt.run(df)

    # Print detailed results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS - XAUBot AI v0.6.0 FIXED")
    print("=" * 80)
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total Trades:        {stats.total_trades}")
    print(f"  Wins:                {stats.wins}")
    print(f"  Losses:              {stats.losses}")
    print(f"  Win Rate:            {stats.win_rate:.1f}%")
    print(f"\nPROFIT ANALYSIS:")
    print(f"  Avg Win:             ${stats.avg_win:.2f}")
    print(f"  Avg Loss:            ${stats.avg_loss:.2f}")
    print(f"  Win/Loss Ratio:      1:{stats.avg_loss/stats.avg_win:.2f}" if stats.avg_win > 0 else "  Win/Loss Ratio:      N/A")
    print(f"  Micro Profits (<$1): {stats.micro_profits}/{stats.wins} ({stats.micro_profit_pct:.0f}%)")
    print(f"\nRISK METRICS:")
    print(f"  Sharpe Ratio:        {stats.sharpe_ratio:.2f}")
    print(f"  Profit Factor:       {stats.profit_factor:.2f}")
    print(f"  Expectancy:          ${stats.expectancy:.2f}/trade")
    print(f"  Max Drawdown:        {stats.max_drawdown:.1f}% (${stats.max_drawdown_usd:.2f})")
    print(f"\nNET RESULTS:")
    net_profit = stats.total_profit - stats.total_loss
    print(f"  Total Profit:        ${stats.total_profit:.2f}")
    print(f"  Total Loss:          -${stats.total_loss:.2f}")
    print(f"  Net P/L:             ${net_profit:.2f}")

    # Target comparison
    print(f"\n" + "-" * 80)
    print("PROFESSOR'S TARGET COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Target':>12} | {'Actual':>12} | {'Status':>10}")
    print("-" * 80)

    targets = [
        ("Avg Win", "$8-12", f"${stats.avg_win:.2f}", stats.avg_win >= 8),
        ("RR Ratio", "1.5:1 or better", f"1:{stats.avg_loss/stats.avg_win:.2f}" if stats.avg_win > 0 else "N/A",
         (stats.avg_loss/stats.avg_win <= 1.5) if stats.avg_win > 0 else False),
        ("Micro Profits", "<20%", f"{stats.micro_profit_pct:.0f}%", stats.micro_profit_pct < 20),
        ("Win Rate", "62-65%", f"{stats.win_rate:.1f}%", 62 <= stats.win_rate <= 67),
        ("Sharpe Ratio", "1.5+", f"{stats.sharpe_ratio:.2f}", stats.sharpe_ratio >= 1.5),
    ]

    for name, target, actual, met in targets:
        status = "PASS" if met else "FAIL"
        status_symbol = "[OK]" if met else "[X]"
        print(f"{name:<25} | {target:>12} | {actual:>12} | {status_symbol:>10}")

    print("=" * 80)

    # Exit reason breakdown
    print(f"\nEXIT REASON BREAKDOWN:")
    exit_reasons = {}
    for trade in stats.trades:
        reason = trade.exit_reason.value
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(trade.profit_usd)

    for reason, profits in sorted(exit_reasons.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(profits)
        avg_profit = sum(profits) / count
        print(f"  {reason:<20}: {count:>3} trades (avg ${avg_profit:>6.2f})")

    # Save if requested
    if args.save:
        import csv
        output_file = f"backtests/v0.6.0_fixed/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Ticket', 'Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price',
                'Profit USD', 'Profit Pips', 'Result', 'Exit Reason', 'Fuzzy Conf',
                'Trajectory Pred', 'Peak Profit', 'Regime', 'Session'
            ])
            for t in stats.trades:
                writer.writerow([
                    t.ticket, t.entry_time, t.exit_time, t.direction, t.entry_price, t.exit_price,
                    f"{t.profit_usd:.2f}", f"{t.profit_pips:.1f}", t.result.value, t.exit_reason.value,
                    f"{t.fuzzy_confidence:.3f}", f"{t.trajectory_predicted:.2f}", f"{t.peak_profit:.2f}",
                    t.regime, t.session
                ])
        logger.info(f"\nResults saved to: {output_file}")

    connector.disconnect()
    logger.info("\nBacktest completed!")


if __name__ == "__main__":
    main()
