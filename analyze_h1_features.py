#!/usr/bin/env python3
"""
H1 Feature Analysis - Check if H1 features actually exist and their correlation
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("H1 FEATURE DEEP ANALYSIS")
print("=" * 80)

# Load model
model_path = Path("models/xgboost_model_v2d.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

feature_names = model_data.get("feature_names", [])
feature_importance = model_data.get("feature_importance", {})

print(f"\nTotal features in model: {len(feature_names)}")

# Find all H1-related features
h1_features = [f for f in feature_names if "h1" in f.lower() or "H1" in f]
print(f"\nH1 features found: {len(h1_features)}")

if h1_features:
    print("\n--- ALL H1 FEATURES ---")
    for feat in sorted(h1_features):
        importance = feature_importance.get(feat, 0)
        # Find rank
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        rank = [f for f, s in sorted_features].index(feat) + 1 if feat in dict(sorted_features) else 999
        print(f"  Rank #{rank:2d}: {feat:40s} importance={importance:10.4f}")

    # Top H1 features
    h1_with_importance = [(f, feature_importance.get(f, 0)) for f in h1_features]
    h1_with_importance.sort(key=lambda x: x[1], reverse=True)

    print("\n--- TOP 10 H1 FEATURES (by importance) ---")
    for i, (feat, imp) in enumerate(h1_with_importance[:10], 1):
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        rank = [f for f, s in sorted_features].index(feat) + 1
        print(f"{i:2d}. Rank #{rank:3d}: {feat:40s} {imp:10.4f}")

    # Summary stats
    importances = [imp for f, imp in h1_with_importance]
    print(f"\n--- H1 FEATURE STATISTICS ---")
    print(f"Total H1 features: {len(h1_features)}")
    print(f"Mean importance: {np.mean(importances):.4f}")
    print(f"Median importance: {np.median(importances):.4f}")
    print(f"Max importance: {np.max(importances):.4f}")
    print(f"Min importance: {np.min(importances):.4f}")

    # Check how many in top N
    sorted_all = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top10_features = [f for f, s in sorted_all[:10]]
    top20_features = [f for f, s in sorted_all[:20]]
    top30_features = [f for f, s in sorted_all[:30]]

    h1_in_top10 = [f for f in top10_features if "h1" in f.lower()]
    h1_in_top20 = [f for f in top20_features if "h1" in f.lower()]
    h1_in_top30 = [f for f in top30_features if "h1" in f.lower()]

    print(f"\nH1 features in top 10: {len(h1_in_top10)}")
    print(f"H1 features in top 20: {len(h1_in_top20)}")
    print(f"H1 features in top 30: {len(h1_in_top30)}")

else:
    print("\nNO H1 FEATURES FOUND IN MODEL!")

# Check all feature names
print("\n" + "=" * 80)
print("ALL FEATURE NAMES IN MODEL")
print("=" * 80)

for i, feat in enumerate(feature_names, 1):
    importance = feature_importance.get(feat, 0)
    print(f"{i:2d}. {feat:50s} {importance:10.4f}")

# Load training data and check for H1 columns
print("\n" + "=" * 80)
print("CHECKING TRAINING DATA FOR H1 FEATURES")
print("=" * 80)

data_file = Path("data/training_data.parquet")
if data_file.exists():
    df = pl.read_parquet(data_file)
    print(f"\nDataset columns: {len(df.columns)}")

    # Find H1 columns
    h1_cols = [col for col in df.columns if "h1" in col.lower() or "H1" in col]
    print(f"H1 columns in dataset: {len(h1_cols)}")

    if h1_cols:
        print("\n--- H1 COLUMNS IN DATASET ---")
        for col in sorted(h1_cols):
            # Check if in model features
            in_model = "YES" if col in feature_names else "NO"
            print(f"  {col:50s} in_model={in_model}")
    else:
        print("\nNO H1 COLUMNS IN TRAINING DATA!")

        # Check if there are any columns that might be H1-related
        print("\nLooking for potential H1-related columns:")
        potential = [col for col in df.columns if any(x in col.lower() for x in ["hour", "h4", "d1", "timeframe"])]
        if potential:
            for col in potential:
                print(f"  {col}")
        else:
            print("  None found")

# Check if feature engineering creates H1 features
print("\n" + "=" * 80)
print("CHECKING FEATURE ENGINEERING CODE")
print("=" * 80)

feature_eng_file = Path("src/feature_eng.py")
if feature_eng_file.exists():
    with open(feature_eng_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Search for H1 references
    if "h1" in content.lower() or "H1" in content:
        print("\nH1 references found in feature_eng.py:")
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "h1" in line.lower() or "H1" in line:
                print(f"  Line {i}: {line.strip()}")
    else:
        print("\nNO H1 references found in feature_eng.py")

    # Check for multi-timeframe
    if "timeframe" in content.lower() or "TIMEFRAME_H1" in content or "mt5.TIMEFRAME_H1" in content:
        print("\nMulti-timeframe references found:")
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "timeframe" in line.lower():
                print(f"  Line {i}: {line.strip()}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if h1_features:
    print(f"\n✓ Model HAS {len(h1_features)} H1 features")
    print(f"✓ Highest ranked H1 feature: {h1_with_importance[0][0]} at rank #{[f for f, s in sorted_all].index(h1_with_importance[0][0]) + 1}")
    print(f"✓ Average H1 importance: {np.mean(importances):.4f}")
else:
    print("\n✗ Model has NO H1 features!")
    print("✗ The 'V2D' model does not include H1 timeframe data")
    print("✗ Need to retrain with H1 features to test hypothesis")
