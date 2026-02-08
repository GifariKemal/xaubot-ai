"""
ML V2 Package
==============
Full ML overhaul with better target variables, enhanced features, and ensemble models.

Components:
- ml_v2_target.py: Improved target variables (multi-bar + ATR threshold)
- ml_v2_feature_eng.py: 23 new features (H1 MTF, continuous SMC, regime, price action)
- ml_v2_model.py: Multi-model support (XGBoost, LightGBM, ensemble)
- ml_v2_train.py: Training pipeline with purged walk-forward CV
- backtest_36_ml_v2.py: Main backtest (configs A/B/C/D/E)
"""

from .ml_v2_target import TargetBuilder
from .ml_v2_feature_eng import MLV2FeatureEngineer
from .ml_v2_model import TradingModelV2, ModelType
from .ml_v2_train import ExperimentConfig, MLV2Trainer

__all__ = [
    "TargetBuilder",
    "MLV2FeatureEngineer",
    "TradingModelV2",
    "ModelType",
    "ExperimentConfig",
    "MLV2Trainer",
]
