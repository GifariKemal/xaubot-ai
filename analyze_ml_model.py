#!/usr/bin/env python3
"""
Deep ML Model Analysis Script
Analyzes xgboost_model_v2d.pkl for performance, feature importance, and limitations
"""

import sys
import pickle
import json
from pathlib import Path
import numpy as np
import polars as pl
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("ML MODEL DEEP DIVE ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD AND INSPECT V2D MODEL
# ============================================================================
print("\n" + "=" * 80)
print("1. MODEL INSPECTION: xgboost_model_v2d.pkl")
print("=" * 80)

model_path = Path("models/xgboost_model_v2d.pkl")
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

print(f"\nModel pickle keys: {list(model_data.keys())}")

# Extract model and metadata
model = model_data.get("model")
metadata = model_data.get("metadata", {})
feature_names = model_data.get("feature_names", [])

print(f"\nModel type: {type(model)}")
print(f"Number of features: {len(feature_names)}")
print(f"\nMetadata keys: {list(metadata.keys())}")

# Display all metadata
print("\n--- MODEL METADATA ---")
for key, value in metadata.items():
    if isinstance(value, (int, float, str, bool)):
        print(f"{key}: {value}")
    elif isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    elif isinstance(value, (list, tuple)) and len(value) < 10:
        print(f"{key}: {value}")
    else:
        print(f"{key}: {type(value)} (length={len(value) if hasattr(value, '__len__') else 'N/A'})")

# Extract key metrics
train_auc = metadata.get("train_auc", "N/A")
test_auc = metadata.get("test_auc", "N/A")
train_samples = metadata.get("train_samples", "N/A")
test_samples = metadata.get("test_samples", "N/A")
class_distribution = metadata.get("class_distribution", {})

print("\n--- KEY METRICS ---")
print(f"Training AUC: {train_auc}")
print(f"Test AUC: {test_auc}")
if isinstance(train_auc, float) and isinstance(test_auc, float):
    overfitting_gap = train_auc - test_auc
    print(f"Overfitting gap: {overfitting_gap:.4f} ({'HIGH' if overfitting_gap > 0.05 else 'NORMAL'})")

print(f"\nTraining samples: {train_samples}")
print(f"Test samples: {test_samples}")

print("\n--- CLASS DISTRIBUTION ---")
for class_name, count in class_distribution.items():
    print(f"{class_name}: {count}")

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. FEATURE IMPORTANCE RANKING (ALL FEATURES)")
print("=" * 80)

# Get feature importance from XGBoost
if hasattr(model, 'feature_importances_'):
    importance_scores = model.feature_importances_
elif hasattr(model, 'get_score'):
    # For XGBoost Booster
    importance_dict = model.get_score(importance_type='gain')
    importance_scores = [importance_dict.get(f"f{i}", 0) for i in range(len(feature_names))]
else:
    print("WARNING: Could not extract feature importance from model")
    importance_scores = [0] * len(feature_names)

# Create ranking
feature_importance = list(zip(feature_names, importance_scores))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\nTotal features: {len(feature_importance)}")

# Categorize features
h1_features = []
m15_features = []
smc_features = []
other_features = []

for feat, score in feature_importance:
    if "_h1" in feat.lower():
        h1_features.append((feat, score))
    elif "ob_" in feat or "fvg_" in feat or "bos" in feat or "choch" in feat:
        smc_features.append((feat, score))
    elif any(x in feat.lower() for x in ["rsi", "macd", "bb", "atr", "ema", "sma", "stoch"]):
        m15_features.append((feat, score))
    else:
        other_features.append((feat, score))

print("\n--- TOP 20 FEATURES (BY IMPORTANCE) ---")
for i, (feat, score) in enumerate(feature_importance[:20], 1):
    category = "H1" if "_h1" in feat.lower() else "SMC" if any(x in feat for x in ["ob_", "fvg_", "bos", "choch"]) else "M15"
    print(f"{i:2d}. {feat:40s} {score:10.4f}  [{category}]")

print("\n--- H1 FEATURES IN TOP 10 ---")
h1_in_top10 = [feat for feat, score in feature_importance[:10] if "_h1" in feat.lower()]
print(f"Count: {len(h1_in_top10)}")
for feat in h1_in_top10:
    rank = [f for f, s in feature_importance].index(feat) + 1
    score = [s for f, s in feature_importance if f == feat][0]
    print(f"  Rank #{rank}: {feat} (importance: {score:.4f})")

print("\n--- FEATURE CATEGORY SUMMARY ---")
print(f"H1 features: {len(h1_features)} total")
print(f"M15 technical features: {len(m15_features)} total")
print(f"SMC features: {len(smc_features)} total")
print(f"Other features: {len(other_features)} total")

# Calculate average importance by category
if h1_features:
    avg_h1 = np.mean([s for f, s in h1_features])
    print(f"\nAverage H1 importance: {avg_h1:.4f}")
if m15_features:
    avg_m15 = np.mean([s for f, s in m15_features])
    print(f"Average M15 importance: {avg_m15:.4f}")
if smc_features:
    avg_smc = np.mean([s for f, s in smc_features])
    print(f"Average SMC importance: {avg_smc:.4f}")

# ============================================================================
# 3. TRAINING LOGS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. TRAINING LOGS ANALYSIS")
print("=" * 80)

log_file = Path("logs/training_2026-02-04.log")
if log_file.exists():
    print(f"\nReading: {log_file}")
    with open(log_file, "r") as f:
        log_content = f.read()

    # Extract key training info
    lines = log_content.split("\n")

    # Look for training metrics
    print("\n--- TRAINING METRICS FROM LOG ---")
    for line in lines:
        if any(kw in line.lower() for kw in ["auc", "accuracy", "precision", "recall", "f1", "samples", "features", "hyperparameter"]):
            print(line.strip())
else:
    print(f"\nNo training log found at {log_file}")

# ============================================================================
# 4. TARGET VARIABLE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("4. TARGET VARIABLE STATISTICS")
print("=" * 80)

data_file = Path("data/training_data.parquet")
if data_file.exists():
    print(f"\nLoading training data from {data_file}...")
    df = pl.read_parquet(data_file)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns}")

    # Check if we have necessary columns
    has_target = "target_signal" in df.columns or "target" in df.columns
    has_atr = "atr" in df.columns
    has_returns = any("return" in col.lower() for col in df.columns)

    print(f"\nHas target column: {has_target}")
    print(f"Has ATR column: {has_atr}")
    print(f"Has return columns: {has_returns}")

    if has_target:
        target_col = "target_signal" if "target_signal" in df.columns else "target"
        target_dist = df.group_by(target_col).agg(pl.count()).sort(target_col)
        print(f"\n--- TARGET DISTRIBUTION ---")
        print(target_dist)

        # Calculate percentages
        total = df.shape[0]
        for row in target_dist.iter_rows(named=True):
            pct = (row['count'] / total) * 100
            print(f"{row[target_col]}: {row['count']} ({pct:.2f}%)")

    # Analyze returns if available
    return_cols = [col for col in df.columns if "return" in col.lower()]
    if return_cols and has_atr:
        print(f"\n--- RETURN ANALYSIS ---")
        print(f"Available return columns: {return_cols}")

        for ret_col in return_cols[:5]:  # First 5 return columns
            if ret_col in df.columns:
                # Calculate stats
                ret_mean = df[ret_col].mean()
                ret_std = df[ret_col].std()
                ret_positive = (df[ret_col] > 0).sum()
                ret_negative = (df[ret_col] < 0).sum()

                print(f"\n{ret_col}:")
                print(f"  Mean: {ret_mean:.6f}")
                print(f"  Std: {ret_std:.6f}")
                print(f"  Positive: {ret_positive} ({(ret_positive/total)*100:.2f}%)")
                print(f"  Negative: {ret_negative} ({(ret_negative/total)*100:.2f}%)")

                # Calculate ATR-normalized returns
                if "atr" in df.columns:
                    atr_mean = df["atr"].mean()
                    print(f"  Mean ATR: {atr_mean:.4f}")

                    # Check different thresholds
                    thresh_03 = ((df[ret_col] / df["atr"]) > 0.3).sum()
                    thresh_05 = ((df[ret_col] / df["atr"]) > 0.5).sum()
                    thresh_10 = ((df[ret_col] / df["atr"]) > 1.0).sum()

                    print(f"  Returns > 0.3×ATR: {thresh_03} ({(thresh_03/total)*100:.2f}%)")
                    print(f"  Returns > 0.5×ATR: {thresh_05} ({(thresh_05/total)*100:.2f}%)")
                    print(f"  Returns > 1.0×ATR: {thresh_10} ({(thresh_10/total)*100:.2f}%)")
else:
    print(f"\nNo training data found at {data_file}")

# ============================================================================
# 5. FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. FEATURE CORRELATION ANALYSIS")
print("=" * 80)

if data_file.exists() and df is not None:
    # Extract H1 and M15 features
    h1_cols = [col for col in df.columns if "_h1" in col.lower()]
    m15_cols = [col for col in df.columns if any(ind in col.lower() for ind in ["rsi", "macd", "bb", "atr", "ema", "sma", "stoch"])]

    print(f"\nH1 columns found: {len(h1_cols)}")
    print(f"M15 columns found: {len(m15_cols)}")

    if h1_cols and m15_cols:
        # Select numeric columns only
        numeric_h1 = [col for col in h1_cols if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        numeric_m15 = [col for col in m15_cols if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        print(f"Numeric H1 columns: {len(numeric_h1)}")
        print(f"Numeric M15 columns: {len(numeric_m15)}")

        if numeric_h1 and numeric_m15:
            # Calculate correlations between H1 and M15 features
            print("\n--- HIGH CORRELATIONS BETWEEN H1 AND M15 FEATURES ---")
            print("(Correlation > 0.7 suggests redundancy)")

            high_corr_count = 0
            for h1_col in numeric_h1[:10]:  # Check first 10 H1 features
                for m15_col in numeric_m15[:10]:  # Against first 10 M15 features
                    try:
                        corr_df = df.select([h1_col, m15_col]).drop_nulls()
                        if corr_df.shape[0] > 0:
                            corr = corr_df.corr()[h1_col, m15_col]
                            if abs(corr) > 0.7:
                                print(f"  {h1_col} <-> {m15_col}: {corr:.3f}")
                                high_corr_count += 1
                    except:
                        pass

            if high_corr_count == 0:
                print("  No high correlations found (good - features are independent)")
            else:
                print(f"\n  Total high correlations: {high_corr_count}")

        # Check H1 feature autocorrelation
        print("\n--- H1 FEATURE INTERNAL CORRELATIONS ---")
        if len(numeric_h1) >= 2:
            high_h1_corr = 0
            for i, col1 in enumerate(numeric_h1[:10]):
                for col2 in numeric_h1[i+1:10]:
                    try:
                        corr_df = df.select([col1, col2]).drop_nulls()
                        if corr_df.shape[0] > 0:
                            corr = corr_df.corr()[col1, col2]
                            if abs(corr) > 0.8:
                                print(f"  {col1} <-> {col2}: {corr:.3f}")
                                high_h1_corr += 1
                    except:
                        pass

            if high_h1_corr == 0:
                print("  No high internal correlations (good)")
else:
    print("\nCannot perform correlation analysis - data not available")

# ============================================================================
# 6. PREDICTION CONSISTENCY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("6. PREDICTION CONSISTENCY ANALYSIS")
print("=" * 80)

persistence_file = Path("data/signal_persistence.json")
if persistence_file.exists():
    print(f"\nReading: {persistence_file}")
    with open(persistence_file, "r") as f:
        persistence_data = json.load(f)

    print(f"Persistence data: {json.dumps(persistence_data, indent=2)}")
else:
    print(f"\nNo persistence data found at {persistence_file}")

# Analyze recent logs for signal flipping
recent_log = Path("logs/trading_bot_2026-02-09.log")
if recent_log.exists():
    print(f"\n--- ANALYZING RECENT SIGNALS FROM LOG ---")
    print(f"Reading: {recent_log}")

    signal_history = []
    with open(recent_log, "r") as f:
        for line in f:
            if "ML Signal:" in line or "prediction:" in line.lower() or "signal=" in line.lower():
                signal_history.append(line.strip())

    print(f"\nFound {len(signal_history)} signal-related log entries")

    if signal_history:
        print("\n--- RECENT SIGNAL SAMPLES (Last 20) ---")
        for entry in signal_history[-20:]:
            print(f"  {entry}")

        # Count signal changes
        signals = []
        for entry in signal_history:
            if "BUY" in entry.upper():
                signals.append("BUY")
            elif "SELL" in entry.upper():
                signals.append("SELL")
            elif "HOLD" in entry.upper():
                signals.append("HOLD")

        if len(signals) > 1:
            changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
            print(f"\n--- SIGNAL STABILITY ---")
            print(f"Total signals tracked: {len(signals)}")
            print(f"Signal changes: {changes}")
            print(f"Change rate: {(changes/len(signals))*100:.2f}%")

            # Count by type
            from collections import Counter
            signal_counts = Counter(signals)
            print(f"\nSignal distribution:")
            for sig, count in signal_counts.items():
                print(f"  {sig}: {count} ({(count/len(signals))*100:.2f}%)")
else:
    print(f"\nNo recent log found at {recent_log}")

# ============================================================================
# 7. MODEL METRICS FROM DATA
# ============================================================================
print("\n" + "=" * 80)
print("7. MODEL METRICS (from data/model_metrics.json)")
print("=" * 80)

metrics_file = Path("data/model_metrics.json")
if metrics_file.exists():
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    print(json.dumps(metrics, indent=2))
else:
    print(f"\nNo metrics file found at {metrics_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print("\n1. MODEL PERFORMANCE:")
print(f"   - Test AUC: {test_auc}")
print(f"   - Overfitting: {'YES' if isinstance(train_auc, float) and isinstance(test_auc, float) and (train_auc - test_auc) > 0.05 else 'NO'}")

print("\n2. FEATURE IMPORTANCE:")
print(f"   - H1 features in top 10: {len(h1_in_top10)}")
print(f"   - Total H1 features: {len(h1_features)}")

if h1_in_top10:
    print(f"   - Highest ranked H1: {h1_in_top10[0]} (rank #{[f for f, s in feature_importance].index(h1_in_top10[0]) + 1})")
else:
    print("   - No H1 features in top 10")

print("\n3. DATA QUALITY:")
if data_file.exists() and has_target:
    print(f"   - Training samples: {total}")
    print(f"   - Target balance: See distribution above")
else:
    print("   - Could not analyze training data")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
