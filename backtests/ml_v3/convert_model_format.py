"""
Convert ML V3 model to TradingModelV2 compatible format.
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtests.ml_v2.ml_v2_model import ModelType

# Load old format
old_path = Path("backtests/ml_v3/xgboost_model_v3.pkl")
with open(old_path, 'rb') as f:
    old_data = pickle.load(f)

print(f"Loaded model from: {old_path}")
print(f"Old keys: {list(old_data.keys())}")

# Convert to TradingModelV2 format
new_data = {
    'xgb_model': old_data['model'],  # XGBoost Booster object
    'lgb_model': None,
    'model_type': ModelType.XGBOOST_BINARY,
    'feature_names': old_data['feature_cols'],
    'confidence_threshold': 0.60,
    'xgb_params': old_data['metadata'].get('hyperparameters', {}),
    'lgb_params': {},
    'feature_importance': {},
    'train_metrics': {
        'train_accuracy': old_data['metadata']['train_accuracy'],
        'test_accuracy': old_data['metadata']['test_accuracy'],
    },
    'fitted': True,
    'metadata': old_data['metadata'],
    'version': '3.0_binary',
    'trained_at': old_data['trained_at'],
    'symbol': old_data['symbol'],
    'timeframe': old_data['timeframe']
}

# Save new format
with open(old_path, 'wb') as f:
    pickle.dump(new_data, f)

print(f"\n✅ Model converted to TradingModelV2 format!")
print(f"   Model type: {new_data['model_type'].value}")
print(f"   Features: {len(new_data['feature_names'])}")
print(f"   Train accuracy: {new_data['train_metrics']['train_accuracy']:.4f}")
print(f"   Test accuracy: {new_data['train_metrics']['test_accuracy']:.4f}")
