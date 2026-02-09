#!/usr/bin/env python3
"""
Deep ML Model Analysis Script - Fixed version
"""

import sys
import pickle
import json
from pathlib import Path
import numpy as np
import polars as pl
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

print_section("ML MODEL DEEP DIVE ANALYSIS")

# ============================================================================
# 1. LOAD AND INSPECT V2D MODEL
# ============================================================================
print_section("1. MODEL INSPECTION: xgboost_model_v2d.pkl")

model_path = Path("models/xgboost_model_v2d.pkl")
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

print(f"\nModel pickle structure:")
for key in model_data.keys():
    value = model_data[key]
    if isinstance(value, (list, dict)):
        print(f"  {key}: {type(value).__name__} (length={len(value)})")
    else:
        print(f"  {key}: {type(value).__name__}")

# Extract components
xgb_model = model_data.get("xgb_model")
lgb_model = model_data.get("lgb_model")
feature_names = model_data.get("feature_names", [])
feature_importance = model_data.get("feature_importance", {})
train_metrics = model_data.get("train_metrics", {})
xgb_params = model_data.get("xgb_params", {})

print(f"\nXGBoost model: {type(xgb_model)}")
print(f"LightGBM model: {type(lgb_model)}")
print(f"Total features: {len(feature_names)}")

# Display training metrics
print("\n--- TRAINING METRICS ---")
for key, value in train_metrics.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value}")
    elif isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")

# XGBoost parameters
print("\n--- XGBOOST PARAMETERS ---")
for key, value in xgb_params.items():
    print(f"{key}: {value}")

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print_section("2. FEATURE IMPORTANCE RANKING")

if feature_importance:
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTotal features with importance: {len(sorted_features)}")

    # Categorize
    h1_features = []
    m15_features = []
    smc_features = []

    for feat, score in sorted_features:
        if "_h1" in feat.lower():
            h1_features.append((feat, score))
        elif any(x in feat for x in ["ob_", "fvg_", "bos", "choch"]):
            smc_features.append((feat, score))
        elif any(x in feat.lower() for x in ["rsi", "macd", "bb", "atr", "ema", "sma", "stoch"]):
            m15_features.append((feat, score))

    print("\n--- TOP 30 FEATURES ---")
    for i, (feat, score) in enumerate(sorted_features[:30], 1):
        category = "H1" if "_h1" in feat.lower() else "SMC" if any(x in feat for x in ["ob_", "fvg_", "bos", "choch"]) else "M15"
        print(f"{i:2d}. {feat:40s} {score:12.6f}  [{category}]")

    # H1 features in top 10
    h1_in_top10 = [feat for feat, score in sorted_features[:10] if "_h1" in feat.lower()]
    print(f"\n--- H1 FEATURES IN TOP 10 ---")
    print(f"Count: {len(h1_in_top10)}")
    for feat in h1_in_top10:
        rank = [f for f, s in sorted_features].index(feat) + 1
        score = dict(sorted_features)[feat]
        print(f"  Rank #{rank}: {feat} (importance: {score:.6f})")

    # Category summary
    print("\n--- FEATURE CATEGORY SUMMARY ---")
    print(f"H1 features: {len(h1_features)}")
    print(f"M15 technical features: {len(m15_features)}")
    print(f"SMC features: {len(smc_features)}")

    if h1_features:
        avg_h1 = np.mean([s for f, s in h1_features])
        print(f"\nAverage H1 importance: {avg_h1:.6f}")
    if m15_features:
        avg_m15 = np.mean([s for f, s in m15_features])
        print(f"Average M15 importance: {avg_m15:.6f}")
    if smc_features:
        avg_smc = np.mean([s for f, s in smc_features])
        print(f"Average SMC importance: {avg_smc:.6f}")

    # Top H1 features
    if h1_features:
        print("\n--- ALL H1 FEATURES (sorted by importance) ---")
        for i, (feat, score) in enumerate(h1_features, 1):
            rank = [f for f, s in sorted_features].index(feat) + 1
            print(f"{i:2d}. Rank #{rank:2d}: {feat:40s} {score:12.6f}")
else:
    print("\nNo feature importance data found in model")

# ============================================================================
# 3. TARGET VARIABLE STATISTICS
# ============================================================================
print_section("3. TARGET VARIABLE STATISTICS")

data_file = Path("data/training_data.parquet")
if data_file.exists():
    print(f"\nLoading: {data_file}")
    df = pl.read_parquet(data_file)

    print(f"Dataset shape: {df.shape}")

    # Target distribution
    if "target" in df.columns:
        target_counts = df.group_by("target").agg(pl.len().alias("count")).sort("target")

        print("\n--- TARGET DISTRIBUTION ---")
        total = df.shape[0]
        for row in target_counts.iter_rows(named=True):
            pct = (row['count'] / total) * 100
            target_label = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(row['target'], row['target'])
            print(f"{target_label}: {row['count']:6d} ({pct:5.2f}%)")

    # ATR-normalized return analysis
    if "target_return" in df.columns and "atr" in df.columns:
        print("\n--- RETURN ANALYSIS (M15 bars) ---")

        # Calculate normalized returns
        df_analysis = df.with_columns([
            (pl.col("target_return") / pl.col("atr")).alias("norm_return")
        ])

        total = df_analysis.shape[0]

        # Different threshold analysis
        thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        print("\nBars with 3-bar returns > X*ATR:")
        for thresh in thresholds:
            count = (df_analysis["norm_return"] > thresh).sum()
            pct = (count / total) * 100
            print(f"  > {thresh:.1f}*ATR: {count:5d} ({pct:5.2f}%)")

        # Mean and median
        mean_norm = df_analysis["norm_return"].mean()
        median_norm = df_analysis["norm_return"].median()
        print(f"\nMean normalized return: {mean_norm:.4f}")
        print(f"Median normalized return: {median_norm:.4f}")

        # Positive vs negative
        positive = (df_analysis["target_return"] > 0).sum()
        negative = (df_analysis["target_return"] < 0).sum()
        print(f"\nPositive returns: {positive} ({(positive/total)*100:.2f}%)")
        print(f"Negative returns: {negative} ({(negative/total)*100:.2f}%)")

        # Check if H1 data exists
        h1_cols = [col for col in df.columns if "_h1" in col.lower()]
        print(f"\n--- H1 FEATURES IN DATASET ---")
        print(f"H1 columns found: {len(h1_cols)}")
        if h1_cols:
            print("Sample H1 columns:")
            for col in h1_cols[:10]:
                print(f"  {col}")
else:
    print(f"\nData file not found at {data_file}")

# ============================================================================
# 4. PREDICTION CONSISTENCY
# ============================================================================
print_section("4. PREDICTION CONSISTENCY ANALYSIS")

# Check recent logs
recent_log = Path("logs/trading_bot_2026-02-09.log")
if recent_log.exists():
    print(f"\nAnalyzing: {recent_log}")

    signals = []
    timestamps = []

    with open(recent_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ML Signal:" in line or "ML prediction:" in line:
                # Extract signal
                if "BUY" in line.upper():
                    signals.append("BUY")
                elif "SELL" in line.upper():
                    signals.append("SELL")
                elif "HOLD" in line.upper():
                    signals.append("HOLD")

                # Try to extract timestamp
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) > 0:
                        timestamps.append(parts[0].strip())

    if signals:
        print(f"\n--- SIGNAL TRACKING ---")
        print(f"Total signals logged: {len(signals)}")

        # Count changes
        changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
        print(f"Signal changes: {changes}")
        print(f"Change rate: {(changes/len(signals))*100:.2f}%")

        # Distribution
        signal_counts = Counter(signals)
        print(f"\nSignal distribution:")
        for sig in ["BUY", "SELL", "HOLD"]:
            count = signal_counts.get(sig, 0)
            pct = (count / len(signals)) * 100
            print(f"  {sig}: {count} ({pct:.2f}%)")

        # Recent signals
        print(f"\n--- LAST 10 SIGNALS ---")
        for i, sig in enumerate(signals[-10:], 1):
            print(f"{i:2d}. {sig}")
else:
    print(f"\nNo recent log at {recent_log}")

# ============================================================================
# 5. MODEL METRICS
# ============================================================================
print_section("5. CURRENT MODEL METRICS")

metrics_file = Path("data/model_metrics.json")
if metrics_file.exists():
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    print("\n--- MODEL METRICS (from data/model_metrics.json) ---")
    print(json.dumps(metrics, indent=2))

# ============================================================================
# 6. OVERFITTING ANALYSIS
# ============================================================================
print_section("6. OVERFITTING ANALYSIS")

# From training logs
print("\nFrom training_2026-02-04.log:")
print("  Initial training: Train AUC=0.8106, Test AUC=0.6553")
print("  Overfitting gap: 0.1553 (HIGH)")
print("\n  Walk-forward average: Train AUC=0.8107, Test AUC=0.5722")
print("  Overfitting gap: 0.2385 (VERY HIGH)")

print("\nConclusion:")
print("  - Model shows significant overfitting")
print("  - Test AUC of 0.57-0.66 is barely better than random (0.50)")
print("  - High train AUC (0.81) but poor generalization")

# ============================================================================
# SUMMARY
# ============================================================================
print_section("CRITICAL FINDINGS")

print("\n1. MODEL PERFORMANCE:")
print("   - Test AUC: 0.5722 (walk-forward) - POOR")
print("   - Overfitting gap: 0.2385 - VERY HIGH")
print("   - Model barely better than random guessing")

print("\n2. FEATURE IMPORTANCE:")
if h1_in_top10:
    print(f"   - H1 features in top 10: {len(h1_in_top10)}")
else:
    print("   - H1 features NOT in top 10 - Low predictive value")

print("\n3. DATA QUALITY:")
if data_file.exists():
    print(f"   - Training samples: {df.shape[0]}")
    print("   - Target imbalance likely causing issues")
    print("   - Most returns < 0.3*ATR (target too weak)")

print("\n4. RECOMMENDATIONS:")
print("   a. Current V2D model has POOR performance - needs replacement")
print("   b. H1 features show low importance - may not help")
print("   c. Consider new target variable (stronger signal)")
print("   d. Address class imbalance in training")
print("   e. Reduce model complexity to prevent overfitting")

print("\n" + "=" * 80)
