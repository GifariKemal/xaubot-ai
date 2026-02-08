"""
Backtest #38 — Model Comparison: Live (V1) vs ML V2
====================================================
Compare old live model vs new ML V2 model on same data.

Models compared:
- Model V1 (Live): models/xgboost_model.pkl (37 features)
- Model V2 (New): backtests/36_ml_v2_results/model_d.pkl (76 features)

Same data, same trading logic, different models only.

Usage:
    python backtests/backtest_38_model_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from loguru import logger
import polars as pl

from src.mt5_connector import MT5Connector
from src.config import get_config
from backtests.backtest_37_ml_v2_test import (
    prepare_data_with_v2_features,
    run_backtest,
    calculate_metrics,
    BacktestMetrics,
)

# For V1 model
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.regime_detector import MarketRegimeDetector
from src.ml_model import TradingModel

logger.remove()
logger.add(sys.stderr, level="INFO")


def prepare_data_v1(df_m15: pl.DataFrame) -> tuple:
    """Prepare M15 data with V1 features (37 features only)."""
    logger.info("Preparing data with V1 features (37 base)...")

    # Base features
    features = FeatureEngineer()
    df_m15 = features.calculate_all(df_m15, include_ml_features=True)

    # SMC
    config = get_config()
    smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
    df_m15 = smc.calculate_all(df_m15)

    # Regime
    regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    try:
        regime_detector.load()
        df_m15 = regime_detector.predict(df_m15)
        logger.info("  HMM regime loaded")
    except Exception as e:
        logger.warning(f"  HMM regime not available: {e}")
        df_m15 = df_m15.with_columns([
            pl.lit(1).alias("regime"),
            pl.lit("medium_volatility").alias("regime_name"),
        ])

    logger.info(f"  Data prepared: {len(df_m15)} M15 bars, {len(df_m15.columns)} columns")

    # Load V1 model
    logger.info(f"Loading V1 model from models/xgboost_model.pkl...")
    model = TradingModel(model_path="models/xgboost_model.pkl")
    model.load()
    logger.info(f"  Model loaded: {len(model.feature_names)} features")
    logger.info(f"  Model fitted: {model.fitted}")

    # Override confidence threshold to match V2
    logger.info(f"  Original confidence threshold: {model.confidence_threshold}")
    model.confidence_threshold = 0.50
    logger.info(f"  Overridden to: {model.confidence_threshold}")

    # Test prediction
    test_pred = model.predict(df_m15.tail(1))
    logger.info(f"  Test prediction: {test_pred.signal}, confidence: {test_pred.confidence:.4f}")

    return df_m15, model


def print_comparison(metrics_v1: BacktestMetrics, metrics_v2: BacktestMetrics):
    """Print side-by-side comparison."""
    print(f"\n{'='*90}")
    print(f"MODEL COMPARISON: V1 (Live) vs V2 (ML V2)")
    print(f"{'='*90}")

    print(f"\n{'Metric':<25} {'V1 (Live)':<25} {'V2 (ML V2)':<25} {'Improvement':<15}")
    print(f"{'-'*90}")

    # Model info
    print(f"{'Model File':<25} {'xgboost_model.pkl':<25} {'model_d.pkl':<25} {'':<15}")
    print(f"{'Features':<25} {f'{metrics_v1.num_features} (base only)':<25} {f'{metrics_v2.num_features} (base+V2)':<25} {f'+{metrics_v2.num_features - metrics_v1.num_features}':<15}")
    print(f"{'Test AUC':<25} {f'{metrics_v1.test_auc:.4f}':<25} {f'{metrics_v2.test_auc:.4f}':<25} {f'+{(metrics_v2.test_auc - metrics_v1.test_auc):.4f}':<15}")

    print(f"\n{'Trading Performance':<25} {'':<25} {'':<25} {'':<15}")
    print(f"{'-'*90}")

    # Trades
    print(f"{'Total Trades':<25} {f'{metrics_v1.total_trades}':<25} {f'{metrics_v2.total_trades}':<25} {f'{metrics_v2.total_trades - metrics_v1.total_trades:+d}':<15}")

    # Win Rate
    wr_diff = metrics_v2.win_rate - metrics_v1.win_rate
    wr_mark = "[BETTER]" if wr_diff > 0 else "[WORSE]"
    print(f"{'Win Rate':<25} {f'{metrics_v1.win_rate:.1f}%':<25} {f'{metrics_v2.win_rate:.1f}%':<25} {f'{wr_diff:+.1f}% {wr_mark}':<15}")

    # Net PnL
    pnl_diff = metrics_v2.net_pnl - metrics_v1.net_pnl
    pnl_mark = "[BETTER]" if pnl_diff > 0 else "[WORSE]"
    print(f"{'Net P&L':<25} {f'${metrics_v1.net_pnl:,.2f}':<25} {f'${metrics_v2.net_pnl:,.2f}':<25} {f'${pnl_diff:+,.2f} {pnl_mark}':<15}")

    # Profit Factor
    pf_diff = metrics_v2.profit_factor - metrics_v1.profit_factor
    pf_mark = "[BETTER]" if pf_diff > 0 else "[WORSE]"
    print(f"{'Profit Factor':<25} {f'{metrics_v1.profit_factor:.2f}':<25} {f'{metrics_v2.profit_factor:.2f}':<25} {f'{pf_diff:+.2f} {pf_mark}':<15}")

    # Avg Win/Loss
    print(f"{'Avg Win':<25} {f'${metrics_v1.avg_win:.2f}':<25} {f'${metrics_v2.avg_win:.2f}':<25} {f'${metrics_v2.avg_win - metrics_v1.avg_win:+.2f}':<15}")
    print(f"{'Avg Loss':<25} {f'${metrics_v1.avg_loss:.2f}':<25} {f'${metrics_v2.avg_loss:.2f}':<25} {f'${metrics_v2.avg_loss - metrics_v1.avg_loss:+.2f}':<15}")

    # Max DD
    dd_diff = metrics_v2.max_drawdown - metrics_v1.max_drawdown
    dd_mark = "[BETTER]" if dd_diff < 0 else "[WORSE]"  # Lower is better
    print(f"{'Max Drawdown':<25} {f'{metrics_v1.max_drawdown:.2f}%':<25} {f'{metrics_v2.max_drawdown:.2f}%':<25} {f'{dd_diff:+.2f}% {dd_mark}':<15}")

    # Sharpe
    sharpe_diff = metrics_v2.sharpe_ratio - metrics_v1.sharpe_ratio
    sharpe_mark = "[BETTER]" if sharpe_diff > 0 else "[WORSE]"
    print(f"{'Sharpe Ratio':<25} {f'{metrics_v1.sharpe_ratio:.2f}':<25} {f'{metrics_v2.sharpe_ratio:.2f}':<25} {f'{sharpe_diff:+.2f} {sharpe_mark}':<15}")

    print(f"\n{'='*90}")

    # Summary
    improvements = sum([
        1 if wr_diff > 0 else 0,
        1 if pnl_diff > 0 else 0,
        1 if pf_diff > 0 else 0,
        1 if dd_diff < 0 else 0,
        1 if sharpe_diff > 0 else 0,
    ])

    print(f"\nSUMMARY:")
    print(f"  V2 wins in {improvements}/5 key metrics")

    if improvements >= 4:
        print(f"  >> RECOMMENDATION: V2 (ML V2) significantly better!")
    elif improvements >= 3:
        print(f"  >> RECOMMENDATION: V2 (ML V2) moderately better")
    else:
        print(f"  >> RECOMMENDATION: Keep V1 (Live)")

    print(f"{'='*90}\n")


def main():
    print(f"{'='*90}")
    print(f"XAUBOT AI — Backtest #38: Model Comparison")
    print(f"V1 (Live) vs V2 (ML V2)")
    print(f"{'='*90}\n")

    # Connect to MT5
    config = get_config()
    mt5_conn = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    mt5_conn.connect()
    logger.info("Connected to MT5\n")

    # Fetch data
    bars = 10000
    logger.info(f"Fetching XAUUSD data ({bars} M15 bars + H1)...")
    df_m15 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="M15", count=bars)
    df_h1 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="H1", count=bars // 4)
    logger.info(f"  Fetched: {len(df_m15)} M15 bars, {len(df_h1)} H1 bars\n")

    # Make a copy for V1 (so V2 doesn't affect it)
    df_m15_v1 = df_m15.clone()

    # ========== V1 Model ==========
    print(f"\n{'='*90}")
    print(f"TESTING MODEL V1 (LIVE)")
    print(f"{'='*90}\n")

    df_v1, model_v1 = prepare_data_v1(df_m15_v1)
    trades_v1, metrics_v1 = run_backtest(df_v1, model_v1, ml_threshold=0.50)

    logger.info(f"\nV1 Results: {metrics_v1.total_trades} trades, WR {metrics_v1.win_rate:.1f}%, PnL ${metrics_v1.net_pnl:.2f}")

    # ========== V2 Model ==========
    print(f"\n{'='*90}")
    print(f"TESTING MODEL V2 (ML V2)")
    print(f"{'='*90}\n")

    model_path = "backtests/36_ml_v2_results/model_d.pkl"
    df_v2, model_v2 = prepare_data_with_v2_features(df_m15, df_h1, model_path)
    trades_v2, metrics_v2 = run_backtest(df_v2, model_v2, ml_threshold=0.50)

    logger.info(f"\nV2 Results: {metrics_v2.total_trades} trades, WR {metrics_v2.win_rate:.1f}%, PnL ${metrics_v2.net_pnl:.2f}")

    # ========== Comparison ==========
    print_comparison(metrics_v1, metrics_v2)

    # Save comparison report
    output_dir = Path("backtests/38_model_comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    report_file = output_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*90 + "\n\n")
        f.write(f"V1 (Live): models/xgboost_model.pkl\n")
        f.write(f"  Features: {metrics_v1.num_features}\n")
        f.write(f"  Trades: {metrics_v1.total_trades}\n")
        f.write(f"  Win Rate: {metrics_v1.win_rate:.1f}%\n")
        f.write(f"  Net P&L: ${metrics_v1.net_pnl:.2f}\n")
        f.write(f"  Profit Factor: {metrics_v1.profit_factor:.2f}\n")
        f.write(f"  Sharpe: {metrics_v1.sharpe_ratio:.2f}\n\n")

        f.write(f"V2 (ML V2): backtests/36_ml_v2_results/model_d.pkl\n")
        f.write(f"  Features: {metrics_v2.num_features}\n")
        f.write(f"  Trades: {metrics_v2.total_trades}\n")
        f.write(f"  Win Rate: {metrics_v2.win_rate:.1f}%\n")
        f.write(f"  Net P&L: ${metrics_v2.net_pnl:.2f}\n")
        f.write(f"  Profit Factor: {metrics_v2.profit_factor:.2f}\n")
        f.write(f"  Sharpe: {metrics_v2.sharpe_ratio:.2f}\n\n")

        f.write(f"IMPROVEMENTS (V2 vs V1):\n")
        f.write(f"  Win Rate: {metrics_v2.win_rate - metrics_v1.win_rate:+.1f}%\n")
        f.write(f"  Net P&L: ${metrics_v2.net_pnl - metrics_v1.net_pnl:+.2f}\n")
        f.write(f"  Profit Factor: {metrics_v2.profit_factor - metrics_v1.profit_factor:+.2f}\n")
        f.write(f"  Sharpe: {metrics_v2.sharpe_ratio - metrics_v1.sharpe_ratio:+.2f}\n")

    logger.info(f"\nComparison report saved: {report_file}")

    mt5_conn.disconnect()

    print(f"\nComparison complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
