"""
Smart Automatic Trading BOT + AI
================================
Hybrid AI Forex Trading System for XAUUSD

Tech Stack:
- Polars (Rust-based DataFrame engine)
- MetaTrader5 (Broker connection)
- XGBoost (ML predictions)
- HMM (Regime detection)
- Native SMC implementation (FVG, Order Blocks, BOS)

Author: Smart Trading Bot Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Smart Trading Bot Team"

from .config import TradingConfig, CapitalMode
from .mt5_connector import MT5Connector
from .smc_polars import SMCAnalyzer
from .feature_eng import FeatureEngineer
from .regime_detector import MarketRegimeDetector
from .risk_engine import RiskEngine
from .ml_model import TradingModel

__all__ = [
    "TradingConfig",
    "CapitalMode",
    "MT5Connector",
    "SMCAnalyzer",
    "FeatureEngineer",
    "MarketRegimeDetector",
    "RiskEngine",
    "TradingModel",
]
