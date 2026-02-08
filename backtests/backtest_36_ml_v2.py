"""
Backtest #36 — ML V2 Full Overhaul
===================================
Tests 6 configurations to measure impact of each ML improvement:

Baseline: 1-bar target + 37 base features + XGBoost (V1 reproduction)
A: 3-bar + ATR threshold target + 37 base features
B: Config A + 8 H1 MTF features (45 total)
C: Config B + 7 continuous SMC features (52 total)
D: Config C + 8 regime/PA features (60 total)
E: Config D + ensemble (XGBoost + LightGBM)

Base: #34A (best time filter config)
Modified: ML model only (entry/exit logic stays same)

Usage:
    python backtests/backtest_36_ml_v2.py
"""

import polars as pl
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.regime_detector import MarketRegimeDetector
from src.config import get_config
from loguru import logger

# ML V2 imports
from backtests.ml_v2.ml_v2_target import TargetBuilder
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from backtests.ml_v2.ml_v2_model import TradingModelV2, ModelType
from backtests.ml_v2.ml_v2_train import (
    MLV2Trainer,
    get_baseline_config,
    get_config_a,
    get_config_b,
    get_config_c,
    get_config_d,
    get_config_e,
)

# Suppress debug logs
logger.remove()
logger.add(sys.stderr, level="INFO")


def prepare_data(df_m15, df_h1):
    """
    Prepare M15 and H1 data with all indicators and features.

    Returns:
        df_m15 with all base + V2 features and all targets
    """
    logger.info("Preparing M15 data...")

    # Base features (37)
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
    except Exception:
        logger.warning("  HMM regime not available, using defaults")
        df_m15 = df_m15.with_columns([
            pl.lit(1).alias("regime"),
            pl.lit("medium_volatility").alias("regime_name"),
        ])

    logger.info("Preparing H1 data...")
    if df_h1 is not None:
        df_h1 = features.calculate_all(df_h1, include_ml_features=False)
        df_h1 = smc.calculate_all(df_h1)

    # V2 Features (23)
    logger.info("Adding V2 features...")
    fe_v2 = MLV2FeatureEngineer()
    df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)

    # Create all targets
    logger.info("Creating targets...")
    target_builder = TargetBuilder()
    df_m15 = target_builder.create_all_targets(df_m15, lookahead=3, threshold_atr_mult=0.3)

    logger.info(f"Data prepared: {len(df_m15)} M15 bars, {len(df_m15.columns)} columns")
    return df_m15


def get_base_feature_list(df: pl.DataFrame) -> list:
    """Get list of base 37 features from V1."""
    # Use V1 logic from src/feature_eng.py::get_feature_columns
    exclude_cols = {
        "time", "open", "high", "low", "close", "volume",
        "spread", "real_volume",
        # Targets
        "target", "target_return", "baseline_target", "multi_bar_target", "target_3class",
        # SMC level columns (not features)
        "swing_high_level", "swing_low_level",
        "fvg_top", "fvg_bottom", "fvg_mid",
        "ob_top", "ob_bottom",
        "bos_level", "choch_level",
        "bsl_level", "ssl_level",
        "last_swing_high", "last_swing_low",
        # Regime labels
        "regime_name",
        # V2 features (will be added separately)
    }

    v2_feature_names = MLV2FeatureEngineer().get_v2_feature_columns()
    exclude_cols.update(v2_feature_names)

    base_features = [
        col for col in df.columns
        if col not in exclude_cols and not col.startswith("_")
    ]

    return base_features


def main():
    print("=" * 70)
    print("XAUBOT AI — #36 ML V2 Full Overhaul")
    print("Comparing Baseline + A/B/C/D/E configurations")
    print("=" * 70)

    # Connect to MT5
    config = get_config()
    mt5_conn = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    mt5_conn.connect()
    logger.info("Connected to MT5")

    # Fetch data
    logger.info("Fetching XAUUSD data...")
    df_m15 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="M15", count=50000)
    df_h1 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="H1", count=15000)
    logger.info(f"  M15: {len(df_m15)} bars, H1: {len(df_h1)} bars")

    # Prepare data
    df_m15 = prepare_data(df_m15, df_h1)

    # Get feature lists
    base_features = get_base_feature_list(df_m15)
    v2_fe = MLV2FeatureEngineer()
    v2_features = v2_fe.get_v2_feature_columns()

    # Split into categories
    h1_features = [f for f in v2_features if f.startswith("h1_")]
    # SMC features: exclude h1_ features to avoid duplicates (e.g., h1_swing_proximity)
    smc_features = [f for f in v2_features if not f.startswith("h1_") and any(x in f for x in ["fvg_", "ob_", "bos_", "confluence", "swing_"])]
    regime_features = [f for f in v2_features if "regime" in f or "volatility" in f or "crisis" in f]
    pa_features = [f for f in v2_features if f in ["wick_ratio", "body_ratio", "gap_from_prev_close", "consecutive_direction"]]

    logger.info(f"Feature counts: Base={len(base_features)}, H1={len(h1_features)}, "
               f"SMC={len(smc_features)}, Regime={len(regime_features)}, PA={len(pa_features)}")

    # Create experiment configs
    configs = [
        ("Baseline", get_baseline_config(base_features)),
        ("A", get_config_a(base_features)),
        ("B", get_config_b(base_features, h1_features)),
        ("C", get_config_c(base_features, h1_features, smc_features)),
        ("D", get_config_d(base_features, h1_features, smc_features, regime_features, pa_features)),
        ("E", get_config_e(base_features, h1_features, smc_features, regime_features, pa_features)),
    ]

    # Train all configs
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ALL CONFIGURATIONS")
    logger.info("=" * 60)

    output_dir = Path("backtests/36_ml_v2_results")
    output_dir.mkdir(exist_ok=True)

    trainer = MLV2Trainer(
        train_size=5000,
        test_size=1000,
        gap_size=50,
        n_folds=5,
    )

    all_results = []

    for cfg_id, cfg in configs:
        model_path = output_dir / f"model_{cfg_id.lower()}.pkl"

        model, cv_results = trainer.train_experiment(
            cfg,
            df_m15,
            save_path=str(model_path),
            run_cv=False,  # Skip CV for faster testing
        )

        all_results.append((cfg_id, cfg.name, model, cv_results))

    # Print comparison table
    print("\n" + "=" * 70)
    print("ML V2 — ALL CONFIGURATIONS COMPARISON")
    print("=" * 70)

    print(f"\n{'Config':<10} {'Name':<25} {'Feats':>6} {'Train AUC':>10} {'Test AUC':>10} {'Overfit':>8}")
    print("-" * 70)

    for cfg_id, cfg_name, model, cv_results in all_results:
        if cv_results:
            train_auc = cv_results.get("mean_train_auc", 0.0)
            test_auc = cv_results.get("mean_test_auc", 0.0)
            overfit = cv_results.get("overfitting_ratio", 0.0)
        else:
            train_auc = model._train_metrics.get("xgb_train_score", 0.0)
            test_auc = model._train_metrics.get("xgb_test_score", 0.0)
            overfit = train_auc / test_auc if test_auc > 0 else 999.0

        n_feats = len(model.feature_names)

        print(f"{cfg_id:<10} {cfg_name:<25} {n_feats:>6} {train_auc:>10.4f} {test_auc:>10.4f} {overfit:>8.2f}")

    # Find best config
    best_cfg = max(all_results, key=lambda x: x[3].get("mean_test_auc", 0.0) if x[3] else 0.0)
    best_id, best_name, best_model, best_cv = best_cfg

    print(f"\nBest Config: {best_id} ({best_name})")
    print(f"  Test AUC: {best_cv.get('mean_test_auc', 0.0):.4f} ± {best_cv.get('std_test_auc', 0.0):.4f}")
    print(f"  Overfitting Ratio: {best_cv.get('overfitting_ratio', 0.0):.2f}")

    # Save summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"ml_v2_summary_{timestamp}.txt"

    with open(log_path, "w") as f:
        f.write(f"ML V2 Full Overhaul — Training Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Dataset: {len(df_m15)} M15 bars\n\n")

        f.write(f"=== FEATURE COUNTS ===\n")
        f.write(f"Base features (V1): {len(base_features)}\n")
        f.write(f"H1 MTF features: {len(h1_features)}\n")
        f.write(f"Continuous SMC features: {len(smc_features)}\n")
        f.write(f"Regime features: {len(regime_features)}\n")
        f.write(f"Price action features: {len(pa_features)}\n")
        f.write(f"Total V2 features: {len(v2_features)}\n\n")

        f.write(f"=== EXPERIMENT RESULTS ===\n")
        f.write(f"{'Config':<10} {'Name':<25} {'Feats':>6} {'Train AUC':>10} {'Test AUC':>10} {'Overfit':>8}\n")
        f.write("-" * 70 + "\n")

        for cfg_id, cfg_name, model, cv_results in all_results:
            if cv_results:
                train_auc = cv_results.get("mean_train_auc", 0.0)
                test_auc = cv_results.get("mean_test_auc", 0.0)
                overfit = cv_results.get("overfitting_ratio", 0.0)
            else:
                train_auc = 0.0
                test_auc = 0.0
                overfit = 0.0

            n_feats = len(model.feature_names)
            f.write(f"{cfg_id:<10} {cfg_name:<25} {n_feats:>6} {train_auc:>10.4f} {test_auc:>10.4f} {overfit:>8.2f}\n")

        f.write(f"\nBest Config: {best_id} ({best_name})\n")
        f.write(f"  Test AUC: {best_cv.get('mean_test_auc', 0.0):.4f}\n")

    logger.info(f"\nSummary saved: {log_path}")

    # Feature importance (best model)
    print(f"\n=== Top 20 Features ({best_id}) ===")
    importance = best_model._feature_importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (feat, score) in enumerate(sorted_importance, 1):
        print(f"  {i:2d}. {feat:<30} {score:>10.2f}")

    mt5_conn.disconnect()

    print(f"\n{'=' * 70}")
    print(f"ML V2 training complete!")
    print(f"Output directory: {output_dir}")
    print(f"  Summary: {log_path.name}")
    print(f"  Models: model_*.pkl (6 files)")
    print(f"\nNext steps:")
    print(f"  1. Review AUC improvements: Baseline -> A -> B -> C -> D -> E")
    print(f"  2. Check overfitting ratio (target < 1.2)")
    print(f"  3. If improvement found, integrate best model into backtests/")
    print("=" * 70)


if __name__ == "__main__":
    main()
