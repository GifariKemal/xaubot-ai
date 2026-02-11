"""
Test ML V3 Binary Model Integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtests.ml_v2.ml_v2_model import TradingModelV2
from src.config import TradingConfig
from src.mt5_connector import MT5Connector
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer

print("=" * 60)
print("ML V3 BINARY MODEL - INTEGRATION TEST")
print("=" * 60)

# 1. Load model
print("\n[1/4] Loading ML V3 Binary Model...")
model = TradingModelV2(
    confidence_threshold=0.60,
    model_path="backtests/ml_v3/xgboost_model_v3.pkl",
)
model.load()

print(f"   Model type: {model.model_type.value}")
print(f"   Features: {len(model.feature_names)}")
print(f"   Confidence threshold: {model.confidence_threshold}")
print(f"   Train accuracy: {model._train_metrics.get('train_accuracy', 0):.4f}")
print(f"   Test accuracy: {model._train_metrics.get('test_accuracy', 0):.4f}")

# 2. Connect to MT5 and fetch data
print("\n[2/4] Fetching market data...")
config = TradingConfig()
mt5 = MT5Connector(
    login=config.mt5_login,
    password=config.mt5_password,
    server=config.mt5_server,
    path=config.mt5_path
)
mt5.connect()

df_m15 = mt5.get_market_data(symbol="XAUUSD", timeframe="M15", count=500)
df_h1 = mt5.get_market_data(symbol="XAUUSD", timeframe="H1", count=100)
print(f"   Fetched {len(df_m15)} M15 bars, {len(df_h1)} H1 bars")

# 3. Calculate features
print("\n[3/4] Calculating features...")
fe = FeatureEngineer()
df_m15 = fe.calculate_all(df_m15, include_ml_features=True)

smc = SMCAnalyzer()
df_m15 = smc.calculate_all(df_m15)

fe_v2 = MLV2FeatureEngineer()
df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)

print(f"   Total features calculated: {len(df_m15.columns)}")

# 4. Make prediction
print("\n[4/4] Making prediction...")
prediction = model.predict(df_m15, feature_cols=model.feature_names)

print(f"\n   Signal: {prediction.signal}")
print(f"   Confidence: {prediction.confidence:.2%}")
print(f"   Probability (BUY): {prediction.probability:.2%}")
print(f"   Probability (SELL): {1-prediction.probability:.2%}")

print("\n" + "=" * 60)
print("INTEGRATION TEST PASSED!")
print("=" * 60)
print(f"\nModel ready for deployment in main_live.py")
print(f"Path: backtests/ml_v3/xgboost_model_v3.pkl")
