"""
Main Live Trading Orchestrator
==============================
Asynchronous event-driven trading system.

Pipeline:
1. Load trained models (.pkl)
2. Fetch Data -> Convert to Polars
3. Apply SMC & Feature Engineering
4. Detect Market Regime (HMM)
5. Get AI Signal (XGBoost)
6. Check Risk & Position Size
7. Execute Trade

Target: < 0.05 seconds per loop
"""

import asyncio
import time
import os
import json
from collections import deque
from datetime import datetime, date, timedelta
from types import SimpleNamespace
from typing import Optional, Dict, Tuple
from zoneinfo import ZoneInfo
from pathlib import Path
import polars as pl
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "logs/trading_bot_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",  # v0.2.2: Fix Unicode encoding errors (Professor AI Fix #5)
)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import modules
from src.config import TradingConfig, get_config
from src.mt5_connector import MT5Connector, MT5SimulationConnector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, FlashCrashDetector, MarketRegime, RegimeState
from src.risk_engine import RiskEngine
from backtests.ml_v2.ml_v2_model import TradingModelV2
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from src.ml_model import get_default_feature_columns  # keep for fallback
from src.position_manager import SmartPositionManager
from src.session_filter import SessionFilter, create_wib_session_filter
from src.auto_trainer import AutoTrainer, create_auto_trainer
from src.telegram_notifier import TelegramNotifier, create_telegram_notifier
from src.telegram_notifications import TelegramNotifications
from src.smart_risk_manager import SmartRiskManager, create_smart_risk_manager
from src.dynamic_confidence import DynamicConfidenceManager, create_dynamic_confidence
# from src.news_agent import NewsAgent, create_news_agent, MarketCondition  # DISABLED
from src.trade_logger import TradeLogger, get_trade_logger
from src.filter_config import FilterConfigManager


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components in an asynchronous event loop.
    """
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        simulation: bool = False,
    ):
        """
        Initialize trading bot.
        
        Args:
            config: Trading configuration (auto-detect if None)
            simulation: Run in simulation mode (no real trades)
        """
        self.config = config or get_config()
        self.simulation = simulation
        
        # Initialize MT5 connector
        if simulation:
            self.mt5 = MT5SimulationConnector()
        else:
            self.mt5 = MT5Connector(
                login=self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server,
                path=self.config.mt5_path,
            )
        
        # Initialize SMC analyzer
        self.smc = SMCAnalyzer(
            swing_length=self.config.smc.swing_length,
            ob_lookback=self.config.smc.ob_lookback,
        )
        
        # Initialize feature engineer
        self.features = FeatureEngineer()
        
        # Initialize regime detector (will load model)
        self.regime_detector = MarketRegimeDetector(
            n_regimes=self.config.regime.n_regimes,
            lookback_periods=self.config.regime.lookback_periods,
            retrain_frequency=self.config.regime.retrain_frequency,
            model_path="models/hmm_regime.pkl",
        )
        
        # Initialize flash crash detector
        self.flash_crash = FlashCrashDetector(
            threshold_percent=self.config.flash_crash_threshold,
        )
        
        # Initialize risk engine
        self.risk_engine = RiskEngine(self.config)

        # Initialize filter config manager
        self.filter_config = FilterConfigManager("data/filter_config.json")

        # Initialize ML Model (unified path — auto-trainer saves here after retrain)
        self.ml_model = TradingModelV2(
            confidence_threshold=0.60,  # Binary confidence threshold (adjustable 0.55-0.65)
            model_path="models/xgboost_model.pkl",
        )
        self.fe_v2 = MLV2FeatureEngineer()
        self._h1_df_cached = None  # Cache H1 DataFrame with indicators for V2 features

        # Initialize Smart Position Manager — EXIT STRATEGY v4 "Patient Recovery"
        # Philosophy: Let trades BREATHE. Don't cut winners at $3-4.
        # Regime danger only at $8+. Give losers room to recover.
        self.position_manager = SmartPositionManager(
            breakeven_pips=20.0,       # Fallback if ATR unavailable
            trail_start_pips=35.0,     # Fallback if ATR unavailable
            trail_step_pips=20.0,      # Fallback if ATR unavailable
            atr_be_mult=2.0,           # v4: BE at 2x ATR (from 1.0) — don't lock too early
            atr_trail_start_mult=3.0,  # v4: Trail at 3x ATR (from 2.0) — let profit run
            atr_trail_step_mult=2.0,   # v4: Trail step 2x ATR (from 1.5)
            min_profit_to_protect=8.0, # v4: Regime/signal exit only at $8+ (from $3)
            max_drawdown_from_peak=40.0,  # v4: Allow 40% drawdown (from 25%)
            # Smart Market Close Handler
            enable_market_close_handler=True,
            min_profit_before_close=3.0,   # v4: from $2
            max_loss_to_hold=10.0,     # v4: Max loss $10 per position (from $5)
        )

        # Initialize Session Filter (WIB timezone for Batam)
        self.session_filter = create_wib_session_filter(aggressive=True)

        # Initialize Auto Trainer - learns from market every day
        self.auto_trainer = create_auto_trainer()

        # Initialize Smart Risk Manager - ULTRA SAFE MODE
        self.smart_risk = create_smart_risk_manager(capital=self.config.capital)

        # Initialize Dynamic Confidence - threshold berdasarkan kondisi market
        self.dynamic_confidence = create_dynamic_confidence()

        # Initialize Telegram Notifier - smart notifications
        self.telegram = create_telegram_notifier()

        # Initialize Telegram Notifications helper (extracts notification logic)
        self.notifications = TelegramNotifications(self)

        # News Agent DISABLED - backtest proved it costs $178 profit
        # ML model already handles volatility well
        self.news_agent = None

        # Initialize Trade Logger - for ML auto-training
        self.trade_logger = get_trade_logger()

        # State tracking
        self._running = False
        self._loop_count = 0
        self._h1_bias_cache = "NEUTRAL"
        self._h1_bias_loop = 0
        self._h1_bias_score = 0.0
        self._h1_bias_strength = "weak"
        self._h1_bias_signals = {}
        self._h1_bias_regime_weights = "unknown"
        self._last_signal: Optional[SMCSignal] = None
        self._last_retrain_check: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None
        self._execution_times: list = []
        self._current_date = date.today()
        self._models_loaded = False
        self._trade_cooldown_seconds = 150  # OPTIMIZED: 2.5 min (~10 bars on M15) - was 300
        self._start_time = datetime.now()
        self._daily_start_balance: float = 0
        self._total_session_profit: float = 0
        self._total_session_trades: int = 0
        self._total_session_wins: int = 0
        self._last_market_update_time: Optional[datetime] = None
        self._last_hourly_report_time: Optional[datetime] = None
        self._open_trade_info: Dict = {}  # Track trade info for close notification
        self._last_news_alert_reason: Optional[str] = None  # Track news alert to avoid duplicates
        self._current_session_multiplier: float = 1.0  # Session lot multiplier
        self._is_sydney_session: bool = False  # Sydney session flag (needs higher confidence)
        self._last_candle_time: Optional[datetime] = None  # Track last processed candle
        self._pyramid_done_tickets: set = set()  # Tickets that already triggered a pyramid
        self._last_pyramid_time: Optional[datetime] = None  # Cooldown between pyramids
        self._position_check_interval: int = 5  # Check positions every N seconds between candles (more data points for velocity)

        # Entry filter tracking for dashboard
        self._last_filter_results: list = []

        # H1 EMA cache for dashboard
        self._h1_ema20_value: float = 0.0
        self._h1_current_price: float = 0.0

        # Dashboard status bridge (written to JSON for Docker API)
        self._dash_price_history: deque = deque(maxlen=120)
        self._dash_equity_history: deque = deque(maxlen=120)
        self._dash_balance_history: deque = deque(maxlen=120)
        self._dash_logs: deque = deque(maxlen=50)
        self._dash_last_price: float = 0.0
        self._dash_status_file = Path("data/bot_status.json")

        # Restore dashboard state from previous session
        self._restore_dashboard_state()

    def _restore_dashboard_state(self):
        """Restore dashboard histories from bot_status.json so restart doesn't lose data."""
        try:
            if not self._dash_status_file.exists():
                return
            import json
            with open(self._dash_status_file, "r") as f:
                prev = json.load(f)

            # Restore price/equity/balance histories
            for val in prev.get("priceHistory", []):
                self._dash_price_history.append(val)
            for val in prev.get("equityHistory", []):
                self._dash_equity_history.append(val)
            for val in prev.get("balanceHistory", []):
                self._dash_balance_history.append(val)

            # Restore logs
            for log in prev.get("logs", []):
                self._dash_logs.append(log)

            # Restore last price
            self._dash_last_price = prev.get("price", 0.0)

            # Restore signal caches so dashboard doesn't show empty
            smc = prev.get("smc", {})
            if smc.get("signal"):
                self._last_raw_smc_signal = smc["signal"]
                self._last_raw_smc_confidence = smc.get("confidence", 0.0)
                self._last_raw_smc_reason = smc.get("reason", "")
                self._last_raw_smc_updated = smc.get("updatedAt", "")

            ml = prev.get("ml", {})
            if ml.get("signal"):
                self._last_ml_signal = ml["signal"]
                self._last_ml_confidence = ml.get("confidence", 0.0)
                self._last_ml_probability = ml.get("buyProb", ml.get("confidence", 0.0))
                self._last_ml_updated = ml.get("updatedAt", "")

            regime = prev.get("regime", {})
            if regime.get("name"):
                from src.regime_detector import MarketRegime
                regime_val = regime["name"].lower().replace(" ", "_")
                try:
                    self._last_regime = MarketRegime(regime_val)
                except ValueError:
                    pass
                self._last_regime_volatility = regime.get("volatility", 0.0)
                self._last_regime_confidence = regime.get("confidence", 0.0)
                self._last_regime_updated = regime.get("updatedAt", "")

            # Restore performance stats
            perf = prev.get("performance", {})
            self._loop_count = perf.get("loopCount", 0)
            self._total_session_trades = perf.get("totalSessionTrades", 0)
            self._total_session_wins = perf.get("totalSessionWins", 0)
            self._total_session_profit = perf.get("totalSessionProfit", 0.0)
            # Restore uptime: shift start_time back by previous uptime
            prev_uptime_h = perf.get("uptimeHours", 0)
            if prev_uptime_h > 0:
                self._start_time = datetime.now() - timedelta(hours=prev_uptime_h)

            # H1 bias: restore values but force recalc on first loop
            self._h1_ema20_value = prev.get("h1BiasDetails", {}).get("ema20", 0.0)
            self._h1_current_price = prev.get("h1BiasDetails", {}).get("price", 0.0)
            # DON'T restore _h1_bias_cache — let it recalculate fresh from MT5
            self._h1_bias_loop = -999  # Force recalc on first iteration

            logger.info(f"Dashboard state restored: {len(self._dash_price_history)} prices, {len(self._dash_logs)} logs, loops={self._loop_count}, uptime={prev_uptime_h}h")
        except Exception as e:
            logger.warning(f"Could not restore dashboard state: {e}")

    def _load_models(self) -> bool:
        """Load pre-trained models."""
        logger.info("Loading trained models...")
        
        models_ok = True
        
        # Load HMM model
        try:
            self.regime_detector.load()
            if self.regime_detector.fitted:
                logger.info("HMM Regime model loaded successfully")
            else:
                logger.warning("HMM model not found or not fitted")
                models_ok = False
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            models_ok = False
        
        # Load ML V2 Model D
        try:
            self.ml_model.load()
            if self.ml_model.fitted:
                logger.info("ML V2 Model D loaded successfully")
                logger.info(f"  Features: {len(self.ml_model.feature_names)}")
                logger.info(f"  Type: {self.ml_model.model_type.value}")
            else:
                logger.warning("ML V2 Model D not found or not fitted")
                models_ok = False
        except Exception as e:
            logger.error(f"Failed to load ML V2 Model D: {e}")
            models_ok = False
        
        self._models_loaded = models_ok

        # Write model metrics for dashboard
        if models_ok:
            self._write_model_metrics()

        return models_ok

    def _dash_log(self, level: str, message: str):
        """Add log entry to dashboard buffer."""
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
        self._dash_logs.append({
            "time": now.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        })

    def _write_model_metrics(self, retrain_results: dict = None):
        """Write model metrics JSON for dashboard Model Insights feature."""
        try:
            import json as _json
            metrics = {
                "featureImportance": [],
                "trainAuc": 0,
                "testAuc": 0,
                "sampleCount": 0,
                "updatedAt": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
            }

            # Extract feature importance from XGBoost model (V2: xgb_model, V1: model)
            booster = getattr(self.ml_model, 'xgb_model', None) or getattr(self.ml_model, 'model', None)
            if self.ml_model.fitted and booster is not None:
                try:
                    importance = booster.get_score(importance_type='gain') if hasattr(booster, 'get_score') else {}
                    # Map f0/f1/... back to feature names if needed
                    if importance and self.ml_model.feature_names:
                        mapped = {}
                        for key, val in importance.items():
                            if key.startswith('f') and key[1:].isdigit():
                                idx = int(key[1:])
                                if idx < len(self.ml_model.feature_names):
                                    mapped[self.ml_model.feature_names[idx]] = val
                                else:
                                    mapped[key] = val
                            else:
                                mapped[key] = val
                        importance = mapped
                    if not importance and hasattr(booster, 'feature_importances_'):
                        names = self.ml_model.feature_names if hasattr(self.ml_model, 'feature_names') else []
                        importance = dict(zip(names, booster.feature_importances_))

                    total = sum(importance.values()) if importance else 1
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    metrics["featureImportance"] = [
                        {"name": name, "importance": round(val / total, 4)}
                        for name, val in sorted_features[:20]
                    ]
                except Exception:
                    pass

            # Use retrain results if available, then model's stored metrics, then auto_trainer
            if retrain_results:
                metrics["trainAuc"] = retrain_results.get("xgb_train_auc", 0)
                metrics["testAuc"] = retrain_results.get("xgb_test_auc", 0)
                metrics["sampleCount"] = retrain_results.get("sample_count", 0)
            elif hasattr(self.ml_model, '_train_metrics') and self.ml_model._train_metrics:
                # Use metrics stored in the model pickle (loaded on startup)
                # V1: train_auc/test_auc, V2: xgb_train_score/xgb_test_score, V3: train_accuracy/test_accuracy
                tm = self.ml_model._train_metrics
                metrics["trainAuc"] = tm.get("train_auc", 0) or tm.get("xgb_train_score", 0) or tm.get("train_accuracy", 0)
                metrics["testAuc"] = tm.get("test_auc", 0) or tm.get("xgb_test_score", 0) or tm.get("test_accuracy", 0)
                metrics["sampleCount"] = tm.get("train_samples", 0) + tm.get("test_samples", 0)
            elif hasattr(self, 'auto_trainer') and hasattr(self.auto_trainer, 'last_auc'):
                metrics["testAuc"] = self.auto_trainer.last_auc or 0

            # Also use model's stored feature importance if booster extraction failed
            if not metrics["featureImportance"] and hasattr(self.ml_model, '_feature_importance') and self.ml_model._feature_importance:
                fi = self.ml_model._feature_importance
                total = sum(fi.values()) if fi else 1
                sorted_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                metrics["featureImportance"] = [
                    {"name": name, "importance": round(val / total, 4)}
                    for name, val in sorted_features[:20] if val > 0
                ]

            metrics_file = Path("data/model_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text(_json.dumps(metrics, indent=2))
        except Exception as e:
            logger.debug(f"Failed to write model metrics: {e}")

    def _write_dashboard_status(self):
        """Write current bot state to JSON file for Docker dashboard API."""
        try:
            wib = ZoneInfo("Asia/Jakarta")
            now = datetime.now(wib)

            # Gather price data
            tick = self.mt5.get_tick(self.config.symbol)
            price = 0.0
            spread = 0.0
            price_change = 0.0
            if tick:
                price = (tick.bid + tick.ask) / 2
                spread = (tick.ask - tick.bid) * 100
                price_change = price - self._dash_last_price if self._dash_last_price > 0 else 0
                self._dash_last_price = price
                self._dash_price_history.append(price)

            # Account data
            balance = self.mt5.account_balance or 0
            equity = self.mt5.account_equity or 0
            profit = equity - balance
            self._dash_equity_history.append(equity)
            self._dash_balance_history.append(balance)

            # Session
            session_name = "Unknown"
            can_trade = False
            try:
                session_info = self.session_filter.get_status_report()
                if session_info:
                    session_name = session_info.get("current_session", "Unknown")
                can_trade, _, _ = self.session_filter.can_trade()
            except Exception:
                pass

            is_golden_time = 19 <= now.hour < 23

            # Risk state
            daily_loss = 0.0
            daily_profit = 0.0
            consecutive_losses = 0
            risk_percent = 0.0
            risk_file = Path("data/risk_state.txt")
            if risk_file.exists():
                try:
                    content = risk_file.read_text()
                    for line in content.strip().split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            if key == "daily_loss":
                                daily_loss = float(value)
                            elif key == "daily_profit":
                                daily_profit = float(value)
                            elif key == "consecutive_losses":
                                consecutive_losses = int(value)
                except Exception:
                    pass
            max_loss = self.config.capital * (self.config.risk.max_daily_loss / 100)
            if max_loss > 0:
                risk_percent = (daily_loss / max_loss) * 100

            # Signals — use raw cached values (before filtering)
            smc_data = {
                "signal": getattr(self, "_last_raw_smc_signal", ""),
                "confidence": getattr(self, "_last_raw_smc_confidence", 0.0),
                "reason": getattr(self, "_last_raw_smc_reason", ""),
                "updatedAt": getattr(self, "_last_raw_smc_updated", ""),
            }

            ml_signal = getattr(self, "_last_ml_signal", "")
            ml_conf = getattr(self, "_last_ml_confidence", 0.0)
            ml_prob = getattr(self, "_last_ml_probability", ml_conf)
            ml_data = {
                "signal": ml_signal,
                "confidence": ml_conf,
                "buyProb": ml_prob,           # ml_prob = probability of BUY (always)
                "sellProb": 1.0 - ml_prob,    # complement = probability of SELL
                "updatedAt": getattr(self, "_last_ml_updated", ""),
            }

            regime_data = {"name": "", "volatility": 0.0, "confidence": 0.0, "updatedAt": ""}
            if hasattr(self, "_last_regime") and self._last_regime:
                regime_data = {
                    "name": self._last_regime.value.replace("_", " ").title(),
                    "volatility": getattr(self, "_last_regime_volatility", 0.0),
                    "confidence": getattr(self, "_last_regime_confidence", 0.0),
                    "updatedAt": getattr(self, "_last_regime_updated", ""),
                }

            # Positions
            positions_list = []
            try:
                positions = self.mt5.get_open_positions(self.config.symbol)
                if positions is not None and not positions.is_empty():
                    for row in positions.iter_rows(named=True):
                        positions_list.append({
                            "ticket": row.get("ticket", 0),
                            "type": "BUY" if row.get("type", 0) == 0 else "SELL",
                            "volume": row.get("volume", 0),
                            "priceOpen": row.get("price_open", 0),
                            "profit": row.get("profit", 0),
                        })
            except Exception:
                pass

            status = {
                "timestamp": now.strftime("%H:%M:%S"),
                "connected": True,
                "price": price,
                "spread": spread,
                "priceChange": price_change,
                "priceHistory": list(self._dash_price_history),
                "balance": balance,
                "equity": equity,
                "profit": profit,
                "equityHistory": list(self._dash_equity_history),
                "balanceHistory": list(self._dash_balance_history),
                "session": session_name,
                "isGoldenTime": is_golden_time,
                "canTrade": can_trade,
                "dailyLoss": daily_loss,
                "dailyProfit": daily_profit,
                "consecutiveLosses": consecutive_losses,
                "riskPercent": risk_percent,
                "smc": smc_data,
                "ml": ml_data,
                "regime": regime_data,
                "positions": positions_list,
                "logs": list(self._dash_logs),
                "settings": {
                    "capitalMode": self.config.capital_mode.value,
                    "capital": self.smart_risk.capital,
                    "riskPerTrade": self.smart_risk.max_loss_per_trade_percent,
                    "maxDailyLoss": self.smart_risk.max_daily_loss_percent,
                    "maxPositions": self.smart_risk.max_concurrent_positions,
                    "maxLotSize": self.smart_risk.max_lot_size,
                    "leverage": self.config.risk.max_leverage,
                    "executionTF": self.config.execution_timeframe,
                    "trendTF": self.config.trend_timeframe,
                    "minRR": 1.5,
                    "mlConfidence": self.config.ml.confidence_threshold,
                    "cooldownSeconds": self.config.thresholds.trade_cooldown_seconds,
                    "symbol": self.config.symbol,
                },
                "h1Bias": getattr(self, "_h1_bias_cache", "NEUTRAL"),
                "dynamicThreshold": getattr(self, "_last_dynamic_threshold", self.config.ml.confidence_threshold),
                "marketQuality": getattr(self, "_last_market_quality", "unknown"),
                "marketScore": getattr(self, "_last_market_score", 0),

                # === NEW: Entry Filter Pipeline ===
                "entryFilters": getattr(self, "_last_filter_results", []),

                # === NEW: Risk Mode ===
                "riskMode": self._get_risk_mode_status(),

                # === NEW: Cooldown ===
                "cooldown": self._get_cooldown_status(),

                # === NEW: Time Filter ===
                "timeFilter": self._get_time_filter_status(),

                # === NEW: Session extras ===
                "sessionMultiplier": getattr(self, "_current_session_multiplier", 1.0),

                # === NEW: Position Details ===
                "positionDetails": self._get_position_details(),

                # === NEW: Auto Trainer ===
                "autoTrainer": self._get_auto_trainer_status(),

                # === NEW: Performance ===
                "performance": self._get_performance_status(),

                # === NEW: Market Close ===
                "marketClose": self._get_market_close_status(),

                # === NEW: H1 Bias Details ===
                "h1BiasDetails": {
                    "bias": getattr(self, "_h1_bias_cache", "NEUTRAL"),
                    "score": getattr(self, "_h1_bias_score", 0.0),
                    "strength": getattr(self, "_h1_bias_strength", "weak"),
                    "indicators": getattr(self, "_h1_bias_signals", {}),
                    "regimeWeights": getattr(self, "_h1_bias_regime_weights", "unknown"),
                    "ema20": getattr(self, "_h1_ema20_value", 0.0),
                    "price": getattr(self, "_h1_current_price", 0.0),
                },
            }

            # Direct write with retry (Windows-friendly)
            json_data = json.dumps(status, default=str)
            status_path = str(self._dash_status_file)
            written = False
            for attempt in range(3):
                try:
                    with open(status_path, "w", encoding="utf-8") as f:
                        f.write(json_data)
                    written = True
                    break
                except (PermissionError, OSError) as e:
                    if attempt < 2:
                        import time as _time
                        _time.sleep(0.05)
                    else:
                        logger.debug(f"Dashboard write failed after 3 attempts: {e}")

        except Exception as e:
            logger.debug(f"Dashboard status write error: {e}")

    def _get_risk_mode_status(self) -> dict:
        """Get risk mode info for dashboard."""
        try:
            rec = self.smart_risk.get_trading_recommendation()
            return {
                "mode": rec.get("mode", "normal"),
                "reason": rec.get("reason", ""),
                "recommendedLot": rec.get("recommended_lot", 0.01),
                "maxAllowedLot": rec.get("max_lot", 0.03),
                "totalLoss": rec.get("total_loss", 0.0),
                "maxTotalLoss": self.smart_risk.max_total_loss_usd,
                "remainingDailyRisk": rec.get("remaining_daily_risk", 0.0),
            }
        except Exception:
            return {"mode": "unknown", "reason": "", "recommendedLot": 0.01, "maxAllowedLot": 0.03, "totalLoss": 0.0, "maxTotalLoss": 0.0, "remainingDailyRisk": 0.0}

    def _get_cooldown_status(self) -> dict:
        """Get trade cooldown info for dashboard."""
        try:
            if self._last_trade_time:
                elapsed = (datetime.now() - self._last_trade_time).total_seconds()
                remaining = max(0, self._trade_cooldown_seconds - elapsed)
                return {
                    "active": remaining > 0,
                    "secondsRemaining": round(remaining),
                    "totalSeconds": self._trade_cooldown_seconds,
                }
            return {"active": False, "secondsRemaining": 0, "totalSeconds": self._trade_cooldown_seconds}
        except Exception:
            return {"active": False, "secondsRemaining": 0, "totalSeconds": 150}

    def _get_time_filter_status(self) -> dict:
        """Get time filter (#34A) status for dashboard."""
        try:
            wib_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
            blocked_hours = []  # All hours enabled
            return {
                "wibHour": wib_hour,
                "isBlocked": wib_hour in blocked_hours,
                "blockedHours": blocked_hours,
            }
        except Exception:
            return {"wibHour": 0, "isBlocked": False, "blockedHours": [9, 21]}

    def _get_position_details(self) -> list:
        """Get detailed position info from SmartRiskManager guards."""
        details = []
        try:
            for ticket, guard in self.smart_risk._position_guards.items():
                trade_hours = (datetime.now(ZoneInfo("Asia/Jakarta")) - guard.entry_time).total_seconds() / 3600
                drawdown_pct = 0.0
                if guard.peak_profit > 0:
                    drawdown_pct = ((guard.peak_profit - guard.current_profit) / guard.peak_profit) * 100

                details.append({
                    "ticket": ticket,
                    "peakProfit": guard.peak_profit,
                    "drawdownFromPeak": round(drawdown_pct, 1),
                    "momentum": round(guard.momentum_score, 1),
                    "tpProbability": round(guard.get_tp_probability(), 1),
                    "reversalWarnings": guard.reversal_warnings,
                    "stalls": guard.stall_count,
                    "tradeHours": round(trade_hours, 1),
                })
        except Exception:
            pass
        return details

    def _get_auto_trainer_status(self) -> dict:
        """Get auto trainer status for dashboard."""
        try:
            hours_since = 0.0
            if self.auto_trainer._last_retrain_time:
                hours_since = (datetime.now(ZoneInfo("Asia/Jakarta")) - self.auto_trainer._last_retrain_time).total_seconds() / 3600

            # Get AUC: prefer auto_trainer's cached value, fallback to model's stored metrics
            current_auc = self.auto_trainer._current_auc
            if current_auc is None and hasattr(self.ml_model, '_train_metrics') and self.ml_model._train_metrics:
                tm = self.ml_model._train_metrics
                current_auc = tm.get("test_auc") or tm.get("xgb_test_score") or tm.get("test_accuracy")

            # Sanitize NaN values for JSON compliance
            if current_auc is not None:
                import math
                if math.isnan(current_auc) or math.isinf(current_auc):
                    current_auc = None

            return {
                "lastRetrain": self.auto_trainer._last_retrain_time.strftime("%Y-%m-%d %H:%M") if self.auto_trainer._last_retrain_time else None,
                "currentAuc": current_auc,
                "minAucThreshold": self.auto_trainer.min_auc_threshold,
                "hoursSinceRetrain": round(hours_since, 1),
                "nextRetrainHour": self.auto_trainer.daily_retrain_hour,
                "modelsFitted": self.ml_model.fitted and self.regime_detector.fitted,
            }
        except Exception:
            return {"lastRetrain": None, "currentAuc": None, "minAucThreshold": 0.65, "hoursSinceRetrain": 0, "nextRetrainHour": 5, "modelsFitted": False}

    def _get_performance_status(self) -> dict:
        """Get bot performance stats for dashboard."""
        try:
            uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
            avg_ms = 0.0
            if self._execution_times:
                recent = self._execution_times[-20:]
                avg_ms = (sum(recent) / len(recent)) * 1000

            return {
                "loopCount": self._loop_count,
                "avgExecutionMs": round(avg_ms, 1),
                "uptimeHours": round(uptime_hours, 1),
                "totalSessionTrades": self._total_session_trades,
                "totalSessionWins": self._total_session_wins,
                "totalSessionProfit": round(self._total_session_profit, 2),
                "winRate": round(self._total_session_wins / self._total_session_trades * 100, 1) if self._total_session_trades > 0 else 0,
            }
        except Exception:
            return {"loopCount": 0, "avgExecutionMs": 0, "uptimeHours": 0, "totalSessionTrades": 0, "totalSessionWins": 0, "totalSessionProfit": 0, "winRate": 0}

    def _get_market_close_status(self) -> dict:
        """Get market close timing info for dashboard."""
        try:
            now = datetime.now(ZoneInfo("Asia/Jakarta"))
            # Daily close: ~05:00 WIB (rollover)
            daily_close_hour = 5
            if now.hour >= daily_close_hour:
                hours_to_daily = (24 - now.hour + daily_close_hour) + (0 - now.minute) / 60
            else:
                hours_to_daily = (daily_close_hour - now.hour) + (0 - now.minute) / 60

            # Weekend close: Friday ~04:00 WIB (Saturday)
            weekday = now.weekday()  # 0=Mon
            if weekday < 4:  # Mon-Thu
                days_to_fri = 4 - weekday
                hours_to_weekend = days_to_fri * 24 + (daily_close_hour - now.hour)
            elif weekday == 4:  # Friday
                hours_to_weekend = max(0, (24 + daily_close_hour - now.hour))
            else:  # Sat-Sun
                hours_to_weekend = 0

            # Market open: Mon-Fri 06:00-05:00 WIB (next day)
            market_open = weekday < 5 and (now.hour >= 6 or now.hour < 4)

            return {
                "hoursToDailyClose": round(max(0, hours_to_daily), 1),
                "hoursToWeekendClose": round(max(0, hours_to_weekend), 1),
                "nearWeekend": weekday == 4 and now.hour >= 20,
                "marketOpen": market_open,
            }
        except Exception:
            return {"hoursToDailyClose": 0, "hoursToWeekendClose": 0, "nearWeekend": False, "marketOpen": False}

    async def start(self):
        """Start the trading bot."""
        # Import version info
        try:
            from src.version import get_detailed_version, __exit_strategy__
            version_str = get_detailed_version()
            exit_str = __exit_strategy__
        except ImportError:
            version_str = "v0.0.0 (Core)"
            exit_str = "Exit v5.0"

        logger.info("=" * 60)
        logger.info(f"XAUBOT AI {version_str}")
        logger.info(f"Strategy: {exit_str}")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Capital: ${self.config.capital:,.2f}")
        logger.info(f"Mode: {self.config.capital_mode.value}")
        logger.info(f"Simulation: {self.simulation}")
        logger.info("=" * 60)
        
        # Load trained models
        if not self._load_models():
            logger.error("Models not loaded. Please run train_models.py first!")
            logger.info("Run: python train_models.py")
            return
        
        # Connect to MT5
        try:
            self.mt5.connect()
            logger.info("MT5 connected successfully!")
            
            # Show account info
            balance = self.mt5.account_balance
            equity = self.mt5.account_equity
            logger.info(f"Account Balance: ${balance:,.2f}")
            logger.info(f"Account Equity: ${equity:,.2f}")

            # Show session status
            session_status = self.session_filter.get_status_report()
            logger.info(f"Session: {session_status['current_session']} ({session_status['volatility']} vol)")
            logger.info(f"Can Trade: {session_status['can_trade']} - {session_status['reason']}")

            # Track daily start balance
            self._daily_start_balance = balance
            self._start_time = datetime.now()
            self.telegram.set_daily_start_balance(balance)

            # Send Telegram startup notification
            await self.notifications.send_startup()

        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            if not self.simulation:
                return

        # Register Telegram commands
        self._register_telegram_commands()

        # Sync position guards with MT5 (cleanup stale guards from previous restarts)
        self._sync_position_guards()

        # Start main loop
        self._running = True
        self._dash_log("info", "Bot started - trading loop active")
        logger.info("Starting main trading loop...")
        await self._main_loop()
    
    async def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False

        # Send Telegram shutdown notification
        await self.notifications.send_shutdown()
        try:
            await self.telegram.close()
        except Exception as e:
            logger.error(f"Failed to close telegram session: {e}")

        self.mt5.disconnect()
        self._log_summary()
    
    def _sync_position_guards(self):
        """Sync position guards with actual MT5 positions — remove stale guards from previous restarts."""
        try:
            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            mt5_tickets = set()
            if open_positions is not None and not open_positions.is_empty():
                mt5_tickets = set(open_positions["ticket"].to_list())

            stale_guards = set(self.smart_risk._position_guards.keys()) - mt5_tickets
            for ticket in stale_guards:
                self.smart_risk.unregister_position(ticket)

            if stale_guards:
                logger.info(f"Cleaned up {len(stale_guards)} stale position guards: {stale_guards}")
            logger.info(f"Position guards synced: {len(self.smart_risk._position_guards)} active (MT5 has {len(mt5_tickets)} positions)")
        except Exception as e:
            logger.warning(f"Position guard sync failed: {e}")

    def _get_available_features(self, df: pl.DataFrame) -> list:
        """Get feature columns that exist in DataFrame."""
        if self.ml_model.fitted and self.ml_model.feature_names:
            return [f for f in self.ml_model.feature_names if f in df.columns]

        default_features = get_default_feature_columns()
        return [f for f in default_features if f in df.columns]

    # --- Signal persistence file helpers (Fix 3) ---
    _SIGNAL_PERSISTENCE_FILE = "data/signal_persistence.json"

    def _load_signal_persistence(self) -> dict:
        """Load signal persistence state from file (survives restarts)."""
        import json, os
        try:
            if os.path.exists(self._SIGNAL_PERSISTENCE_FILE):
                with open(self._SIGNAL_PERSISTENCE_FILE, "r") as f:
                    raw = json.load(f)
                # Convert lists back to tuples
                result = {k: (v[0], v[1]) for k, v in raw.items()}
                logger.info(f"Loaded signal persistence: {result}")
                return result
        except Exception as e:
            logger.debug(f"Could not load signal persistence: {e}")
        return {}

    def _save_signal_persistence(self):
        """Save signal persistence state to file."""
        import json, os
        try:
            os.makedirs(os.path.dirname(self._SIGNAL_PERSISTENCE_FILE), exist_ok=True)
            with open(self._SIGNAL_PERSISTENCE_FILE, "w") as f:
                json.dump(self._signal_persistence, f)
        except Exception as e:
            logger.debug(f"Could not save signal persistence: {e}")

    # --- H1 Multi-Timeframe Bias (Fix 5) ---
    def _get_h1_bias(self) -> str:
        """
        Dynamic H1 higher-timeframe bias using multi-indicator scoring + regime-based weights.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"

        Uses 5 indicators with regime-adaptive weights:
        1. EMA Trend (price vs EMA21)
        2. EMA Cross (EMA9 vs EMA21)
        3. RSI Zone (>55 bull, <45 bear)
        4. MACD Histogram
        5. Candle Structure (last 5 candles)

        Weights adjust based on HMM regime (trending/ranging/volatile).
        Score range: -1.0 (max bearish) to +1.0 (max bullish).
        Threshold: ±0.3 (30% agreement needed).
        """
        try:
            # Cache H1 bias — only update every 4 candles (1 hour) since H1 changes slowly
            if hasattr(self, '_h1_bias_cache') and hasattr(self, '_h1_bias_loop'):
                if self._loop_count - self._h1_bias_loop < 4:
                    return self._h1_bias_cache

            df_h1 = self.mt5.get_market_data(
                symbol=self.config.symbol,
                timeframe="H1",
                count=100,
            )

            if len(df_h1) < 30:
                return "NEUTRAL"

            # Calculate indicators + SMC on H1 and cache for V2 features
            df_h1 = self.features.calculate_all(df_h1, include_ml_features=False)
            df_h1 = self.smc.calculate_all(df_h1)
            self._h1_df_cached = df_h1  # Cache for V2 features

            # Extract latest values
            last = df_h1.row(-1, named=True)
            price = last["close"]
            ema_9 = last["ema_9"]
            ema_21 = last["ema_21"]
            rsi = last["rsi"]
            macd_hist = last["macd_histogram"]

            # === 5 Indicator Signals (+1, -1, 0) ===
            signals = {
                "ema_trend": 1 if price > ema_21 else (-1 if price < ema_21 else 0),
                "ema_cross": 1 if ema_9 > ema_21 else (-1 if ema_9 < ema_21 else 0),
                "rsi": 1 if rsi > 55 else (-1 if rsi < 45 else 0),
                "macd": 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0),
                "candles": self._count_candle_bias(df_h1),
            }

            # === Regime-Based Weights ===
            weights = self._get_regime_weights()

            # === Weighted Score ===
            score = sum(signals[k] * weights[k] for k in signals)

            # === Dynamic Threshold ===
            if score >= 0.3:
                bias = "BULLISH"
            elif score <= -0.3:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            # === Determine Strength ===
            abs_score = abs(score)
            if abs_score >= 0.7:
                strength = "strong"
            elif abs_score >= 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            # === Cache Results ===
            self._h1_bias_cache = bias
            self._h1_bias_loop = self._loop_count
            self._h1_bias_score = float(score)
            self._h1_bias_strength = strength
            self._h1_bias_signals = signals.copy()
            _regime_str = self._last_regime.value if hasattr(self, '_last_regime') and self._last_regime else "unknown"
            self._h1_bias_regime_weights = _regime_str
            self._h1_current_price = float(price)
            # Keep EMA20 for backward compatibility (use EMA21 as proxy)
            self._h1_ema20_value = float(ema_21)

            if self._loop_count % 4 == 0:
                logger.info(
                    f"H1 Bias: {bias} ({strength}, score={score:.2f}) | "
                    f"Signals: EMA_trend={signals['ema_trend']:+d}, EMA_cross={signals['ema_cross']:+d}, "
                    f"RSI={signals['rsi']:+d}, MACD={signals['macd']:+d}, Candles={signals['candles']:+d} | "
                    f"Regime: {_regime_str}"
                )

            return bias

        except Exception as e:
            logger.debug(f"H1 dynamic bias error: {e}")
            return "NEUTRAL"

    def _count_candle_bias(self, df_h1) -> int:
        """
        Count bullish/bearish candles in last 5 H1 candles.
        Returns: +1 if majority bullish (≥3), -1 if majority bearish (≥3), 0 otherwise.
        """
        try:
            last_5 = df_h1.tail(5)
            bullish = sum(1 for row in last_5.iter_rows(named=True) if row["close"] > row["open"])
            bearish = 5 - bullish

            if bullish >= 3:
                return 1
            elif bearish >= 3:
                return -1
            else:
                return 0
        except Exception:
            return 0

    def _get_regime_weights(self) -> dict:
        """
        Get indicator weights based on current HMM regime.

        Regimes:
        - Low volatility (ranging): RSI/MACD dominate (mean-reversion)
        - Medium volatility: Balanced
        - High volatility (trending): EMA trend/cross dominate

        Returns: dict with keys matching signals (ema_trend, ema_cross, rsi, macd, candles)
        """
        regime = (self._last_regime.value if hasattr(self, '_last_regime') and self._last_regime else "medium_volatility").lower()

        if "low" in regime or "ranging" in regime:
            # Low volatility / ranging — RSI and MACD more useful
            return {
                "ema_trend": 0.15,
                "ema_cross": 0.15,
                "rsi": 0.30,
                "macd": 0.25,
                "candles": 0.15,
            }
        elif "high" in regime or "trending" in regime:
            # High volatility / trending — EMA trend dominates
            return {
                "ema_trend": 0.30,
                "ema_cross": 0.25,
                "rsi": 0.10,
                "macd": 0.25,
                "candles": 0.10,
            }
        else:
            # Medium volatility — balanced weights
            return {
                "ema_trend": 0.25,
                "ema_cross": 0.20,
                "rsi": 0.20,
                "macd": 0.20,
                "candles": 0.15,
            }

    def _is_filter_enabled(self, filter_key: str) -> bool:
        """
        Check if a filter is enabled via filter_config.json.

        Args:
            filter_key: Filter key (e.g., "h1_bias", "ml_confidence")

        Returns:
            True if enabled, False if disabled
        """
        return self.filter_config.is_enabled(filter_key)

    def _register_telegram_commands(self):
        """Register Telegram command handlers from separate module."""
        from src.telegram_commands import register_commands
        register_commands(self)

    async def _main_loop(self):
        """Main trading loop - CANDLE-BASED (not time-based)."""
        last_position_check = time.time()

        while self._running:
            loop_start = time.perf_counter()

            try:
                # Check for new day
                if date.today() != self._current_date:
                    self._on_new_day()

                # Ensure MT5 connection is alive (auto-reconnect if needed)
                if not self.mt5.ensure_connected():
                    logger.warning("MT5 disconnected, attempting reconnection...")
                    await asyncio.sleep(10)  # Wait before retrying
                    continue

                # Get current candle time to check if new candle formed
                df_check = self.mt5.get_market_data(
                    symbol=self.config.symbol,
                    timeframe=self.config.execution_timeframe,
                    count=2,
                )

                if len(df_check) == 0:
                    logger.warning("No data received from MT5")
                    await asyncio.sleep(5)
                    continue

                current_candle_time = df_check["time"].tail(1).item()

                # Check if new candle formed
                is_new_candle = (
                    self._last_candle_time is None or
                    current_candle_time > self._last_candle_time
                )

                if is_new_candle:
                    # NEW CANDLE: Run full analysis
                    self._last_candle_time = current_candle_time
                    await self._trading_iteration()
                    self._loop_count += 1

                    # Log on new candle
                    if self._loop_count % 4 == 0:  # Every 4 candles (1 hour on M15)
                        avg_time = sum(self._execution_times[-4:]) / min(4, len(self._execution_times)) if self._execution_times else 0
                        logger.info(f"Candle #{self._loop_count} | Avg execution: {avg_time*1000:.1f}ms")

                    # AUTO-RETRAINING CHECK - every 20 candles (5 hours on M15)
                    if self._loop_count % 20 == 0:
                        await self._check_auto_retrain()
                else:
                    # SAME CANDLE: Only check positions (every 10 seconds)
                    if time.time() - last_position_check >= self._position_check_interval:
                        await self._position_check_only()
                        last_position_check = time.time()

            except Exception as e:
                logger.error(f"Loop error: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            # Track execution time
            execution_time = time.perf_counter() - loop_start
            self._execution_times.append(execution_time)

            # Write dashboard status file (for Docker API)
            self._write_dashboard_status()

            # Poll Telegram commands (non-blocking, every loop)
            try:
                await self.telegram.poll_commands()
            except Exception:
                pass

            # Wait before next check (5 seconds between candle checks)
            await asyncio.sleep(5)

    async def _position_check_only(self):
        """Quick position check between candles — uses cached ML/features, adds flash crash detection."""
        try:
            # Get live tick price (cheap call)
            tick = self.mt5.get_tick(self.config.symbol)
            if not tick:
                return
            current_price = tick.bid

            # --- FLASH CRASH DETECTION (Fix 2) ---
            # Fetch minimal bars for flash crash check
            df_mini = self.mt5.get_market_data(
                symbol=self.config.symbol,
                timeframe=self.config.execution_timeframe,
                count=5,
            )
            if len(df_mini) > 0:
                is_flash, move_pct = self.flash_crash.detect(df_mini)
                if is_flash:
                    logger.warning(f"FLASH CRASH detected between candles: {move_pct:.2f}% move!")
                    try:
                        await self._emergency_close_all()
                    except Exception as e:
                        logger.critical(f"CRITICAL: Emergency close failed: {e}")
                        await self.notifications.send_flash_crash_critical(move_pct, e)
                    return

            # --- POSITION MANAGEMENT (uses cached data — Fix 4) ---
            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )

            if len(open_positions) > 0 and not self.simulation:
                # Use cached ML prediction and DataFrame from last candle (Fix 4)
                # No need to recalculate 37 features every 5 seconds
                cached_ml = getattr(self, '_cached_ml_prediction', None)
                cached_df = getattr(self, '_cached_df', None)
                cached_regime = None
                if hasattr(self, '_last_regime') and self._last_regime:
                    # Build a simple regime state from cached values
                    cached_regime = RegimeState(
                        regime=self._last_regime,
                        volatility=getattr(self, '_last_regime_volatility', 0.0),
                        confidence=getattr(self, '_last_regime_confidence', 0.0),
                        probabilities={},
                        recommendation="TRADE",
                    )

                if cached_ml and cached_df is not None and len(cached_df) > 0:
                    await self._smart_position_management(
                        open_positions=open_positions,
                        df=cached_df,
                        regime_state=cached_regime,
                        ml_prediction=cached_ml,
                        current_price=current_price,
                    )
                else:
                    # Fallback: first iteration before any candle processed
                    df = self.mt5.get_market_data(
                        symbol=self.config.symbol,
                        timeframe=self.config.execution_timeframe,
                        count=50,
                    )
                    if len(df) == 0:
                        return
                    df = self.features.calculate_all(df, include_ml_features=True)
                    df = self.smc.calculate_all(df)
                    df = self.fe_v2.add_all_v2_features(df, self._h1_df_cached)
                    # Ensure regime columns exist for ML model
                    if "regime" not in df.columns:
                        df = df.with_columns(pl.lit(1).alias("regime"))
                    if "regime_confidence" not in df.columns:
                        df = df.with_columns(pl.lit(1.0).alias("regime_confidence"))
                    feature_cols = self._get_available_features(df)
                    ml_prediction = self.ml_model.predict(df, feature_cols)
                    await self._smart_position_management(
                        open_positions=open_positions,
                        df=df,
                        regime_state=cached_regime,
                        ml_prediction=ml_prediction,
                        current_price=current_price,
                    )
            # --- PYRAMID CHECK: Add to Winner when trade 1 is in profit ---
            if len(open_positions) > 0 and not self.simulation:
                await self._check_pyramid_opportunity(open_positions, current_price)

        except Exception as e:
            logger.debug(f"Position check error: {e}")

    async def _check_pyramid_opportunity(self, open_positions, current_price: float):
        """
        Add to Winner (Pyramiding): Buka trade ke-2 saat trade pertama sudah profit.

        Rules:
        1. Trade pertama harus profit >= $8 (ATR-scaled)
        2. Ticket belum pernah trigger pyramid sebelumnya
        3. SMC signal >= 75% sama arah
        4. ML prediction setuju sama arah
        5. Session harus London atau New York (high liquidity)
        6. Max 2 posisi concurrent
        7. Cooldown 30 detik antar pyramid
        8. Lot size sama dengan trade pertama
        """
        try:
            # Cooldown check: minimal 30 detik antar pyramid
            if self._last_pyramid_time:
                seconds_since = (datetime.now() - self._last_pyramid_time).total_seconds()
                if seconds_since < 30:
                    return

            # Position limit check
            can_open, limit_reason = self.smart_risk.can_open_position()
            if not can_open:
                return

            # Session check: only London and New York (high liquidity for pyramiding)
            session_info = self.session_filter.get_status_report()
            session_name = session_info.get("current_session", "Unknown")
            if session_name not in ("London", "New York", "London-NY Overlap"):
                return

            # Get cached signals
            cached_smc_signal = getattr(self, '_last_raw_smc_signal', '')
            cached_smc_conf = getattr(self, '_last_raw_smc_confidence', 0.0)
            cached_ml = getattr(self, '_cached_ml_prediction', None)

            if not cached_smc_signal or not cached_ml:
                return

            # ATR scaling for profit threshold
            _current_atr = 0.0
            _baseline_atr = 0.0
            cached_df = getattr(self, '_cached_df', None)
            if cached_df is not None and "atr" in cached_df.columns:
                atr_series = cached_df["atr"].drop_nulls()
                if len(atr_series) > 0:
                    _current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:
                    _baseline_atr = atr_series.tail(96).mean()
                elif len(atr_series) >= 20:
                    _baseline_atr = atr_series.mean()

            # Check each open position for pyramid opportunity
            for row in open_positions.iter_rows(named=True):
                ticket = row["ticket"]
                profit = row.get("profit", 0)
                position_type = row.get("type", 0)  # 0=BUY, 1=SELL
                direction = "BUY" if position_type == 0 else "SELL"
                lot_size = row.get("volume", 0.01)

                # Skip if already triggered pyramid
                if ticket in self._pyramid_done_tickets:
                    continue

                # ATR-based profit threshold (per-position, adapts to lot size)
                atr_dollars = _current_atr * lot_size * 100 if _current_atr > 0 else 0
                sm = max(0.3, min(1.5, _current_atr / _baseline_atr)) if _baseline_atr > 0 else 1.0
                atr_unit = atr_dollars if atr_dollars > 0 else 10 * sm
                min_profit_for_pyramid = 0.5 * atr_unit  # 0.5 ATR — same as tp_min

                # Trade must be profitable enough
                if profit < min_profit_for_pyramid:
                    continue

                # Check velocity is positive (trade still moving in our favor)
                guard = self.smart_risk._position_guards.get(ticket)
                if guard and guard.velocity <= 0:
                    continue  # Don't pyramid into a stalling trade

                # SMC signal must match direction with >= 75% confidence
                if cached_smc_signal != direction or cached_smc_conf < 0.75:
                    continue

                # ML must agree with direction
                if cached_ml.signal != direction:
                    continue

                # All conditions passed — execute pyramid trade
                logger.info(f"[PYRAMID] Conditions met for #{ticket}: profit=${profit:.2f}, "
                           f"SMC={cached_smc_signal}({cached_smc_conf:.0%}), ML={cached_ml.signal}({cached_ml.confidence:.0%})")

                # Build signal from cached data
                last_signal = getattr(self, '_last_signal', None)
                if not last_signal:
                    logger.debug("[PYRAMID] No cached signal available")
                    continue

                # Create fresh SMC signal for pyramid entry
                tick = self.mt5.get_tick(self.config.symbol)
                if not tick:
                    continue

                entry_price = tick.ask if direction == "BUY" else tick.bid

                # Use cached signal's SL/TP structure but adjust entry to current price
                pyramid_signal = SMCSignal(
                    signal_type=direction,
                    entry_price=entry_price,
                    stop_loss=last_signal.stop_loss,
                    take_profit=last_signal.take_profit,
                    confidence=cached_smc_conf,
                    reason=f"PYRAMID: Add to winner #{ticket} (profit=${profit:.2f})",
                )

                # Use same lot size as original trade
                sl_distance = abs(entry_price - pyramid_signal.stop_loss)
                risk_amount = lot_size * sl_distance * 10
                account_balance = self.mt5.account_balance or self.config.capital
                risk_percent = (risk_amount / account_balance) * 100

                pyramid_pos = SimpleNamespace(
                    lot_size=lot_size,  # Same lot as original
                    risk_amount=risk_amount,
                    risk_percent=risk_percent,
                )

                # Get regime state for execution
                cached_regime = None
                if hasattr(self, '_last_regime') and self._last_regime:
                    cached_regime = RegimeState(
                        regime=self._last_regime,
                        volatility=getattr(self, '_last_regime_volatility', 0.0),
                        confidence=getattr(self, '_last_regime_confidence', 0.0),
                        probabilities={},
                        recommendation="TRADE",
                    )

                # Execute pyramid trade
                logger.info(f"[PYRAMID] Opening {direction} {lot_size} lot @ {entry_price:.2f} "
                           f"(adding to winner #{ticket})")

                trade_time_before = self._last_trade_time
                await self._execute_trade_safe(pyramid_signal, pyramid_pos, cached_regime)

                # Only mark as done if trade was actually executed (trade_time updates on success)
                if self._last_trade_time != trade_time_before:
                    self._pyramid_done_tickets.add(ticket)
                    self._last_pyramid_time = datetime.now()
                    self._dash_log("trade", f"PYRAMID: {direction} {lot_size} lot (adding to #{ticket}, profit=${profit:.2f})")
                else:
                    logger.warning(f"[PYRAMID] Trade execution failed for #{ticket}, will retry next cycle")

                # Only one pyramid per check cycle
                break

        except Exception as e:
            logger.debug(f"Pyramid check error: {e}")

    async def _trading_iteration(self):
        """Single trading iteration."""
        # Reset filter tracking for dashboard
        self._last_filter_results = []

        # Reload filter config (lightweight JSON read, allows live updates from dashboard)
        self.filter_config.load()

        # 1. Fetch fresh data
        df = self.mt5.get_market_data(
            symbol=self.config.symbol,
            timeframe=self.config.execution_timeframe,
            count=200,
        )
        
        if len(df) == 0:
            logger.warning("No data received")
            return
        
        # 2. Apply feature engineering
        df = self.features.calculate_all(df, include_ml_features=True)
        
        # 3. Apply SMC analysis
        df = self.smc.calculate_all(df)

        # 3a. Ensure H1 data is cached BEFORE V2 features (fixes "No H1 data" warning)
        if self._h1_df_cached is None:
            self._get_h1_bias()

        # 3b. Add V2 features for Model D (23 extra features)
        df = self.fe_v2.add_all_v2_features(df, self._h1_df_cached)

        # 4. Detect regime
        try:
            df = self.regime_detector.predict(df)
            regime_state = self.regime_detector.get_current_state(df)
            
            # Log regime change
            if hasattr(self, '_last_regime') and self._last_regime != regime_state.regime:
                logger.info(f"Regime changed: {self._last_regime.value} -> {regime_state.regime.value}")
            self._last_regime = regime_state.regime
            self._last_regime_volatility = regime_state.volatility
            self._last_regime_confidence = regime_state.confidence
            self._last_regime_updated = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%H:%M:%S")
            
        except Exception as e:
            logger.warning(f"Regime detection error: {e}")
            regime_state = None

        # Ensure regime columns exist for ML model (even if regime detection failed)
        if "regime" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("regime"))
        if "regime_confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("regime_confidence"))

        # 5. Check flash crash
        is_flash, move_pct = self.flash_crash.detect(df.tail(5))
        flash_enabled = self._is_filter_enabled("flash_crash_guard")
        flash_blocked = is_flash and flash_enabled
        self._last_filter_results.append({
            "name": "Flash Crash Guard",
            "passed": not flash_blocked,
            "detail": f"{move_pct:.2f}% move" if is_flash else "OK" + (" [DISABLED]" if not flash_enabled else "")
        })
        if flash_blocked:
            logger.warning(f"Flash crash detected: {move_pct:.2f}% move")
            try:
                await self._emergency_close_all()
            except Exception as e:
                logger.critical(f"CRITICAL: Emergency close failed completely: {e}")
                await self.notifications.send_flash_crash_critical(move_pct, e)
            return

        # 6. Check if trading is allowed
        account_balance = self.mt5.account_balance or self.config.capital
        account_equity = self.mt5.account_equity or self.config.capital
        open_positions = self.mt5.get_open_positions(
            symbol=self.config.symbol,
            magic=self.config.magic_number,
        )

        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if tick else df["close"].tail(1).item()

        # Get ML prediction early for position management
        feature_cols = self._get_available_features(df)
        ml_prediction = self.ml_model.predict(df, feature_cols)

        # Store for trade logging + dashboard
        self._last_ml_signal = ml_prediction.signal
        self._last_ml_confidence = ml_prediction.confidence
        self._last_ml_probability = ml_prediction.probability
        self._last_ml_updated = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%H:%M:%S")

        # Cache ML prediction and DataFrame for inter-candle position checks (Fix 4)
        self._cached_ml_prediction = ml_prediction
        self._cached_df = df

        # Cache SMC signal for dashboard (runs before filters so dashboard always updates)
        smc_signal = self.smc.generate_signal(df)
        _wib_now = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%H:%M:%S")
        if smc_signal:
            self._last_raw_smc_signal = smc_signal.signal_type
            self._last_raw_smc_confidence = smc_signal.confidence
            self._last_raw_smc_reason = smc_signal.reason
            self._last_raw_smc_updated = _wib_now
            self._dash_log("trade", f"SMC: {smc_signal.signal_type} ({smc_signal.confidence:.0%}) - {smc_signal.reason}")
        else:
            self._last_raw_smc_signal = ""
            self._last_raw_smc_confidence = 0.0
            self._last_raw_smc_reason = ""
            self._last_raw_smc_updated = _wib_now

        # H1 Multi-Timeframe Bias (runs before filters so dashboard always updates)
        h1_bias = self._get_h1_bias()

        # 6.5 SMART POSITION MANAGEMENT - NO HARD STOP LOSS
        # Hanya close jika: TP tercapai, ML reversal kuat, atau max loss
        if len(open_positions) > 0:
            if not self.simulation:
                await self._smart_position_management(
                    open_positions=open_positions,
                    df=df,
                    regime_state=regime_state,
                    ml_prediction=ml_prediction,
                    current_price=current_price,
                )

            # Log position summary periodically
            if self._loop_count % 60 == 0:
                total_profit = 0
                for row in open_positions.iter_rows(named=True):
                    total_profit += row.get("profit", 0)
                logger.info(f"Positions: {len(open_positions)} | Total P/L: ${total_profit:.2f}")

        # Send hourly analysis report to Telegram (every 1 hour)
        # Placed here to ensure it's sent regardless of trading conditions
        await self.notifications.send_hourly_analysis_if_due(
            df=df,
            regime_state=regime_state,
            ml_prediction=ml_prediction,
            open_positions=open_positions,
            current_price=current_price,
        )

        risk_metrics = self.risk_engine.check_risk(
            account_balance=account_balance,
            account_equity=account_equity,
            open_positions=open_positions,
            current_price=current_price,
        )
        
        # 7. Check regime allows trading
        regime_sleep = regime_state and regime_state.recommendation == "SLEEP"
        regime_enabled = self._is_filter_enabled("regime_filter")
        regime_blocked = regime_sleep and regime_enabled
        self._last_filter_results.append({
            "name": "Regime Filter",
            "passed": not regime_blocked,
            "detail": (regime_state.regime.value if regime_state else "N/A") + (" [DISABLED]" if not regime_enabled else "")
        })
        if regime_blocked:
            logger.debug(f"Regime SLEEP: {regime_state.regime.value}")
            return

        risk_enabled = self._is_filter_enabled("risk_check")
        risk_blocked = not risk_metrics.can_trade and risk_enabled
        self._last_filter_results.append({
            "name": "Risk Check",
            "passed": not risk_blocked,
            "detail": (risk_metrics.reason if not risk_metrics.can_trade else "OK") + (" [DISABLED]" if not risk_enabled else "")
        })
        if risk_blocked:
            logger.debug(f"Risk blocked: {risk_metrics.reason}")
            return

        # 7.5 Check trading session (WIB timezone)
        session_ok, session_reason, session_multiplier = self.session_filter.can_trade()
        session_enabled = self._is_filter_enabled("session_filter")
        session_blocked = not session_ok and session_enabled
        self._last_filter_results.append({
            "name": "Session Filter",
            "passed": not session_blocked,
            "detail": session_reason + (" [DISABLED]" if not session_enabled else "")
        })
        if session_blocked:
            if self._loop_count % 300 == 0:  # Log every 5 minutes
                logger.info(f"Session filter: {session_reason}")
                next_window = self.session_filter.get_next_trading_window()
                logger.info(f"Next trading window: {next_window['session']} in {next_window['hours_until']} hours")
            return

        # Store session info for later use (Sydney needs higher confidence)
        self._current_session_multiplier = session_multiplier
        self._is_sydney_session = "Sydney" in session_reason or session_multiplier == 0.5

        # 7.6 NEWS AGENT - DISABLED (backtest: costs $178 profit, ML handles volatility)

        # 7.7 H1 bias already calculated above (before filters, for dashboard)

        # 8. SMC signal already generated above (before filters, for dashboard)

        # 9. ML prediction already done above for position management

        # Log signal status every 4 loops (~1 hour on M15)
        if self._loop_count % 4 == 0:
            price = df["close"].tail(1).item()
            h1_tag = f" | H1: {h1_bias}" if h1_bias != "NEUTRAL" else ""
            logger.info(f"Price: {price:.2f} | Regime: {regime_state.regime.value if regime_state else 'N/A'} | SMC: {smc_signal.signal_type if smc_signal else 'NONE'} | ML: {ml_prediction.signal}({ml_prediction.confidence:.0%}){h1_tag}")

        # Market update disabled from auto-send (available via command)
        # if self._loop_count > 0 and self._loop_count % 30 == 0:
        #     await self._send_market_update(df, regime_state, ml_prediction)

        # Track SMC signal for filter pipeline
        self._last_filter_results.append({"name": "SMC Signal", "passed": smc_signal is not None, "detail": f"{smc_signal.signal_type} ({smc_signal.confidence:.0%})" if smc_signal else "No signal"})

        # 10. Combine signals
        final_signal = self._combine_signals(smc_signal, ml_prediction, regime_state)
        signal_enabled = self._is_filter_enabled("signal_combination")
        signal_blocked = final_signal is None and signal_enabled
        self._last_filter_results.append({
            "name": "Signal Combination",
            "passed": not signal_blocked,
            "detail": (f"{final_signal.signal_type} ({final_signal.confidence:.0%})" if final_signal else "Filtered out") + (" [DISABLED]" if not signal_enabled else "")
        })

        if signal_blocked:
            return

        # 10.1 H1 Multi-Timeframe Filter (#31B: Price vs EMA20 — backtest +$343)
        # BUY only when H1 is BULLISH, SELL only when H1 is BEARISH
        h1_enabled = self._is_filter_enabled("h1_bias")
        h1_passed = True
        h1_detail = f"H1={h1_bias}"

        if h1_enabled:
            h1_opposed = False
            if h1_bias == "NEUTRAL":
                # NEUTRAL = no opinion → allow trade (don't block)
                logger.debug(f"H1 Filter: NEUTRAL — no H1 opinion, allowing {final_signal.signal_type}")
            elif (final_signal.signal_type == "BUY" and h1_bias == "BEARISH") or \
                 (final_signal.signal_type == "SELL" and h1_bias == "BULLISH"):
                # Actively opposed — block unless strong override
                h1_opposed = True
            else:
                logger.info(f"H1 Filter: {final_signal.signal_type} aligned with H1={h1_bias}")

            if h1_opposed:
                # Strong signal override: if SMC >= 80% AND ML agrees >= 65%, bypass H1
                smc_strong = smc_signal and smc_signal.confidence >= 0.80
                ml_agrees = ml_prediction and ml_prediction.signal == final_signal.signal_type
                ml_strong = ml_prediction and ml_prediction.confidence >= 0.65

                if smc_strong and ml_agrees and ml_strong:
                    h1_passed = True
                    h1_detail = f"OVERRIDE: {final_signal.signal_type} vs H1={h1_bias} (SMC={smc_signal.confidence:.0%}+ML={ml_prediction.confidence:.0%})"
                    logger.info(f"H1 Filter: OVERRIDE — {final_signal.signal_type} allowed despite H1={h1_bias} (SMC={smc_signal.confidence:.0%}, ML={ml_prediction.signal} {ml_prediction.confidence:.0%})")
                    self._last_filter_results.append({"name": "H1 Bias (#31B)", "passed": True, "detail": h1_detail})
                else:
                    h1_passed = False
                    h1_detail = f"{final_signal.signal_type} vs H1={h1_bias}"
                    self._last_filter_results.append({"name": "H1 Bias (#31B)", "passed": False, "detail": h1_detail})
                    logger.info(f"H1 Filter: {final_signal.signal_type} blocked (H1={h1_bias})")
                    return
            else:
                self._last_filter_results.append({"name": "H1 Bias (#31B)", "passed": True, "detail": f"Aligned {h1_bias}"})
        else:
            self._last_filter_results.append({"name": "H1 Bias (#31B)", "passed": True, "detail": f"H1={h1_bias} [DISABLED]"})

        # 10.2 Time-of-Hour Filter (#34A: skip WIB hours 9 and 21 — backtest +$356)
        # Hour 9 WIB (02:00 UTC) = end of NY session, low liquidity
        # Hour 21 WIB (14:00 UTC) = London-NY transition, whipsaw prone
        wib_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
        time_blocked = False  # All hours enabled — risk managed by ATR scaling + lot multiplier
        time_enabled = self._is_filter_enabled("time_filter")
        time_filter_blocked = time_blocked and time_enabled

        # v6.1: NIGHT SAFETY - Spread filter for late night hours (22:00-05:59 WIB)
        is_night_hours = wib_hour >= 22 or wib_hour <= 5
        night_spread_ok = True
        night_spread_msg = ""
        if is_night_hours:
            # Get current spread
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                current_spread_points = (tick.ask - tick.bid) / 0.01  # Spread in points (0.01 = 1 pip for gold)
                # Normal max spread: 30 points ($0.30)
                # Night max spread: 50 points ($0.50) - allow wider spread but still filter extremes
                night_max_spread = 50
                if current_spread_points > night_max_spread:
                    night_spread_ok = False
                    night_spread_msg = f"spread {current_spread_points:.1f}p > {night_max_spread}p"
                else:
                    night_spread_msg = f"spread {current_spread_points:.1f}p OK (night limit {night_max_spread}p)"

        self._last_filter_results.append({
            "name": "Time Filter (#34A)",
            "passed": not time_filter_blocked and night_spread_ok,
            "detail": f"WIB {wib_hour}" + (" BLOCKED" if time_blocked else "") + (" [DISABLED]" if not time_enabled else "") + (f" NIGHT: {night_spread_msg}" if is_night_hours else "")
        })
        if time_filter_blocked:
            logger.info(f"Time Filter: {final_signal.signal_type} blocked (WIB hour {wib_hour} is skip hour)")
            return
        if not night_spread_ok:
            logger.warning(f"Night Safety: {final_signal.signal_type} blocked - {night_spread_msg} (WIB {wib_hour})")
            return

        # 10.5 Check trade cooldown
        cooldown_blocked = False
        cooldown_remaining = 0
        if self._last_trade_time:
            time_since_last = (datetime.now() - self._last_trade_time).total_seconds()
            cooldown_remaining = self._trade_cooldown_seconds - time_since_last
            if cooldown_remaining > 0:
                cooldown_blocked = True
        cooldown_enabled = self._is_filter_enabled("cooldown")
        cooldown_filter_blocked = cooldown_blocked and cooldown_enabled
        self._last_filter_results.append({
            "name": "Trade Cooldown",
            "passed": not cooldown_filter_blocked,
            "detail": (f"{cooldown_remaining:.0f}s left" if cooldown_blocked else "OK") + (" [DISABLED]" if not cooldown_enabled else "")
        })
        if cooldown_filter_blocked:
            logger.info(f"Trade cooldown: {cooldown_remaining:.0f}s remaining")
            return

        # 10.6 PULLBACK FILTER - DISABLED (SMC-only mode)
        # SMC structure already validates entry zones

        # 11. SMART RISK CHECK - Ultra safe mode
        self.smart_risk.check_new_day()
        risk_rec = self.smart_risk.get_trading_recommendation()
        self._last_filter_results.append({"name": "Smart Risk Gate", "passed": risk_rec["can_trade"], "detail": risk_rec.get("reason", risk_rec["mode"])})

        if not risk_rec["can_trade"]:
            logger.warning(f"Smart Risk: Trading blocked - {risk_rec['reason']}")
            return

        # 12. Calculate SAFE lot size (0.01-0.02 max) with ML confidence
        regime_name = regime_state.regime.value if regime_state else "normal"
        safe_lot = self.smart_risk.calculate_lot_size(
            entry_price=final_signal.entry_price,
            confidence=final_signal.confidence,
            regime=regime_name,
            ml_confidence=ml_prediction.confidence,  # IMPROVEMENT 3: Pass ML confidence
        )

        # Apply session multiplier (Sydney = 0.5x for safety)
        session_mult = getattr(self, '_current_session_multiplier', 1.0)
        if session_mult < 1.0:
            original_lot = safe_lot
            safe_lot = max(0.01, round(safe_lot * session_mult, 2))  # Minimum 0.01, rounded to 0.01 step
            sydney_mode = getattr(self, '_is_sydney_session', False)
            if sydney_mode:
                logger.info(f"Sydney SAFE MODE: Lot {original_lot:.2f} -> {safe_lot:.2f} (0.5x)")

        # v6.1: NIGHT SAFETY - Lot reduction for late night hours (22:00-05:59 WIB)
        # Night trading has lower win rate (14%) and higher loss risk
        # Reduce lot by 50% to minimize damage from night volatility
        wib_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
        is_night_hours = wib_hour >= 22 or wib_hour <= 5
        if is_night_hours:
            original_lot = safe_lot
            safe_lot = max(0.01, round(safe_lot * 0.5, 2))  # 50% reduction
            logger.warning(f"NIGHT SAFETY MODE: Lot {original_lot:.2f} -> {safe_lot:.2f} (0.5x) - WIB {wib_hour}:xx (high risk hours)")

        if safe_lot <= 0:
            logger.debug("Smart Risk: Lot size is 0 - skipping trade")
            return

        # Create position result with safe lot
        from dataclasses import dataclass

        @dataclass
        class SafePosition:
            lot_size: float
            risk_amount: float
            risk_percent: float

        # Calculate risk amount (with our tiny lot, risk is minimal)
        sl_distance = abs(final_signal.entry_price - final_signal.stop_loss)
        risk_amount = safe_lot * sl_distance * 10  # Approximate for gold
        risk_percent = (risk_amount / account_balance) * 100

        position_result = SafePosition(
            lot_size=safe_lot,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
        )

        logger.info(f"Smart Risk: Lot={safe_lot}, Risk=${risk_amount:.2f} ({risk_percent:.2f}%), Mode={risk_rec['mode']}")

        # 13. Check position limit (max 2 concurrent positions)
        can_open, limit_reason = self.smart_risk.can_open_position()
        self._last_filter_results.append({"name": "Position Limit", "passed": can_open, "detail": limit_reason if not can_open else "OK"})
        if not can_open:
            logger.warning(f"Position limit: {limit_reason} - skipping trade")
            return

        # 14. Execute trade (with Emergency Broker SL)
        await self._execute_trade_safe(final_signal, position_result, regime_state)
    
    def _combine_signals(
        self,
        smc_signal: Optional[SMCSignal],
        ml_prediction,
        regime_state,
    ) -> Optional[SMCSignal]:
        """Combine SMC and ML signals with DYNAMIC confidence threshold."""
        # Get current price for ML-only signals
        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if tick else 0

        # Get session info for dynamic analysis
        session_status = self.session_filter.get_status_report()
        session_name = session_status.get("current_session", "Unknown")
        volatility = session_status.get("volatility", "medium")

        # Determine trend direction
        trend_direction = "NEUTRAL"
        if hasattr(self, '_last_regime') and regime_state:
            trend_direction = regime_state.regime.value

        # DYNAMIC CONFIDENCE ANALYSIS
        market_analysis = self.dynamic_confidence.analyze_market(
            session=session_name,
            regime=regime_state.regime.value if regime_state else "unknown",
            volatility=volatility,
            trend_direction=trend_direction,
            has_smc_signal=(smc_signal is not None),
            ml_signal=ml_prediction.signal,
            ml_confidence=ml_prediction.confidence,
        )

        # Get dynamic threshold
        dynamic_threshold = market_analysis.confidence_threshold
        self._last_dynamic_threshold = dynamic_threshold
        self._last_market_quality = market_analysis.quality.value
        self._last_market_score = market_analysis.score

        # Log dynamic analysis periodically
        if self._loop_count % 60 == 0:
            logger.info(f"Dynamic: {market_analysis.quality.value} (score={market_analysis.score}) -> threshold={dynamic_threshold:.0%}")

        # ============================================================
        # IMPROVED SIGNAL LOGIC v2 (ML+SMC Required for Golden Time)
        # ============================================================
        # Golden Time (19:00-23:00 WIB): Require ML+SMC alignment
        # Other Sessions: SMC-only with ML weak filter

        # Check if in golden time (London-NY Overlap, 19:00-23:00 WIB)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        current_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
        is_golden_time = 19 <= current_hour <= 23  # Fixed detection

        # 1. JANGAN trade jika market quality AVOID atau CRISIS
        if market_analysis.quality.value == "avoid":
            if self._loop_count % 120 == 0:
                logger.info(f"Skip: Market quality AVOID - tidak entry")
            return None

        if regime_state and regime_state.regime == MarketRegime.CRISIS:
            if self._loop_count % 120 == 0:
                logger.info(f"Skip: CRISIS regime - tidak entry")
            return None

        # ============================================================
        # v0.2.2 FIX #3: FALSE BREAKOUT FILTER (Professor AI)
        # ============================================================
        # London session + low ATR = potential whipsaw → require HIGHER ML confidence
        session_info = self.session_filter.get_status_report()
        session_name = session_info.get("current_session", "Unknown")
        is_london = session_name == "London"

        # Calculate ATR ratio
        atr_ratio = 1.0
        if "atr" in df.columns:
            atr_series = df["atr"].drop_nulls()
            if len(atr_series) > 0:
                current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:
                    baseline_atr = atr_series.tail(96).mean()
                    atr_ratio = current_atr / baseline_atr if baseline_atr > 0 else 1.0

        # Filter false breakouts
        if is_london and atr_ratio < 1.2:
            # London + low volatility = whipsaw risk
            # Require ML confidence >= 0.70 (instead of 0.60)
            if ml_prediction.confidence < 0.70:
                if self._loop_count % 60 == 0:
                    logger.info(
                        f"[FALSE BREAKOUT RISK] London + low ATR ({atr_ratio:.2f}x) → "
                        f"ML conf {ml_prediction.confidence:.0%} < 70% required"
                    )
                return None

        # ============================================================
        # SIGNAL LOGIC v4 - SMC-Only (ML DISABLED)
        # ============================================================
        golden_marker = "[GOLDEN] " if is_golden_time else ""
        if smc_signal is not None:
            # ML filters DISABLED — trading based on SMC only
            # Signal persistence DISABLED — SMC signal = immediate trade

            # SMC-Only: Use SMC signal with confidence adjustment
            ml_agrees = (
                (smc_signal.signal_type == "BUY" and ml_prediction.signal == "BUY") or
                (smc_signal.signal_type == "SELL" and ml_prediction.signal == "SELL")
            )

            # SELL-SPECIFIC CONFIDENCE FILTER (Step 4: Improve 41.2% win rate)
            # Require ML confidence >= 0.75 for SELL signals to filter weak trades
            if smc_signal.signal_type == "SELL":
                if ml_prediction.signal != "SELL" or ml_prediction.confidence < 0.75:
                    if self._loop_count % 60 == 0:
                        logger.info(f"SELL blocked: ML confidence too low ({ml_prediction.signal} {ml_prediction.confidence:.0%}, need SELL >=75%)")
                    return None

            if ml_agrees:
                combined_confidence = (smc_signal.confidence + ml_prediction.confidence) / 2
                reason_suffix = f" | ML AGREES: {ml_prediction.signal} ({ml_prediction.confidence:.0%})"
            else:
                combined_confidence = smc_signal.confidence
                reason_suffix = f" | ML: {ml_prediction.signal} ({ml_prediction.confidence:.0%})"

            # Apply regime adjustment for high volatility
            if regime_state and regime_state.regime == MarketRegime.HIGH_VOLATILITY:
                combined_confidence *= 0.9

            logger.info(f"{golden_marker}SMC Signal: {smc_signal.signal_type} @ {smc_signal.entry_price:.2f} (SMC={smc_signal.confidence:.0%}, ML={ml_prediction.signal} {ml_prediction.confidence:.0%})")

            return SMCSignal(
                signal_type=smc_signal.signal_type,
                entry_price=smc_signal.entry_price,
                stop_loss=smc_signal.stop_loss,
                take_profit=smc_signal.take_profit,
                confidence=combined_confidence,
                reason=f"SMC-CONFIRMED: {smc_signal.reason}{reason_suffix}",
            )

        # No valid signal
        return None

    def _check_pullback_filter(
        self,
        df: pl.DataFrame,
        signal_direction: str,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        Check if price is in a pullback/retrace against signal direction.

        PREVENTS entry during temporary bounces that cause early losses.

        Logic:
        - For SELL: Skip if price momentum is UP (bouncing)
        - For BUY: Skip if price momentum is DOWN (falling)

        Uses multiple confirmations:
        1. Short-term momentum (last 3 candles)
        2. MACD histogram direction
        3. Price vs EMA relationship

        Returns:
            Tuple[bool, str]: (can_trade, reason)
        """
        try:
            # Get recent data (last 10 candles)
            recent = df.tail(10)

            if len(recent) < 5:
                return True, "Not enough data for pullback check"

            # Get ATR for dynamic thresholds (no more hardcoded $2, $1.5)
            atr = 12.0  # Default for XAUUSD
            if "atr" in df.columns:
                atr_val = recent["atr"].to_list()[-1]
                if atr_val is not None and atr_val > 0:
                    atr = atr_val

            # Dynamic thresholds based on ATR
            bounce_threshold = atr * 0.15      # 15% of ATR = significant bounce
            consolidation_threshold = atr * 0.10  # 10% of ATR = consolidation

            # === 1. SHORT-TERM MOMENTUM (Last 3 candles) ===
            closes = recent["close"].to_list()
            last_3_closes = closes[-3:]

            # Calculate short momentum: positive = rising, negative = falling
            short_momentum = last_3_closes[-1] - last_3_closes[0]
            momentum_direction = "UP" if short_momentum > 0 else "DOWN"

            # === 2. MACD HISTOGRAM DIRECTION ===
            macd_hist_direction = "NEUTRAL"
            if "macd_histogram" in df.columns:
                macd_hist = recent["macd_histogram"].to_list()
                last_hist = macd_hist[-1] if macd_hist[-1] is not None else 0
                prev_hist = macd_hist[-2] if macd_hist[-2] is not None else 0

                # MACD histogram rising = bullish momentum, falling = bearish
                if last_hist > prev_hist:
                    macd_hist_direction = "RISING"  # Bullish momentum increasing
                else:
                    macd_hist_direction = "FALLING"  # Bearish momentum increasing

            # === 3. PRICE VS SHORT EMA ===
            price_vs_ema = "NEUTRAL"
            if "ema_9" in df.columns:
                ema_9 = recent["ema_9"].to_list()[-1]
                if ema_9 is not None:
                    if current_price > ema_9 * 1.001:  # Above EMA by 0.1%
                        price_vs_ema = "ABOVE"
                    elif current_price < ema_9 * 0.999:  # Below EMA by 0.1%
                        price_vs_ema = "BELOW"

            # === 4. RSI EXTREME CHECK ===
            rsi_extreme = False
            rsi_value = 50
            if "rsi" in df.columns:
                rsi_value = recent["rsi"].to_list()[-1]
                if rsi_value is not None:
                    # RSI extreme = potential reversal zone
                    rsi_extreme = rsi_value > 75 or rsi_value < 25

            # === PULLBACK DETECTION LOGIC ===

            if signal_direction == "SELL":
                # For SELL signal, we want:
                # - Price momentum DOWN (not bouncing up)
                # - MACD histogram FALLING (bearish momentum)
                # - Price BELOW or AT EMA (not extended above)

                # BLOCK if price is bouncing UP (ATR-based threshold)
                if momentum_direction == "UP" and short_momentum > bounce_threshold:
                    return False, f"SELL blocked: Price bouncing UP (+${short_momentum:.2f} > {bounce_threshold:.2f})"

                # BLOCK if MACD showing bullish momentum increasing
                if macd_hist_direction == "RISING" and momentum_direction == "UP":
                    return False, f"SELL blocked: MACD bullish + price rising"

                # BLOCK if price extended above EMA (overbought bounce)
                if price_vs_ema == "ABOVE" and momentum_direction == "UP":
                    return False, f"SELL blocked: Price above EMA9 and rising"

                # ALLOW if momentum aligned with signal
                if momentum_direction == "DOWN":
                    return True, f"SELL OK: Momentum aligned (${short_momentum:.2f})"

                # ALLOW if price in consolidation (ATR-based threshold)
                if abs(short_momentum) < consolidation_threshold:
                    return True, f"SELL OK: Consolidation phase (<{consolidation_threshold:.2f})"

            elif signal_direction == "BUY":
                # For BUY signal, we want:
                # - Price momentum UP (not falling down)
                # - MACD histogram RISING (bullish momentum)
                # - Price ABOVE or AT EMA (not falling below)

                # BLOCK if price is falling DOWN (ATR-based threshold)
                if momentum_direction == "DOWN" and short_momentum < -bounce_threshold:
                    return False, f"BUY blocked: Price falling DOWN (${short_momentum:.2f} < -{bounce_threshold:.2f})"

                # BLOCK if MACD showing bearish momentum increasing
                if macd_hist_direction == "FALLING" and momentum_direction == "DOWN":
                    return False, f"BUY blocked: MACD bearish + price falling"

                # BLOCK if price extended below EMA (oversold drop)
                if price_vs_ema == "BELOW" and momentum_direction == "DOWN":
                    return False, f"BUY blocked: Price below EMA9 and falling"

                # ALLOW if momentum aligned with signal
                if momentum_direction == "UP":
                    return True, f"BUY OK: Momentum aligned (+${short_momentum:.2f})"

                # ALLOW if price in consolidation (ATR-based threshold)
                if abs(short_momentum) < consolidation_threshold:
                    return True, f"BUY OK: Consolidation phase (<{consolidation_threshold:.2f})"

            # Default: allow trade if no strong pullback detected
            return True, f"Pullback check passed (mom={momentum_direction}, macd={macd_hist_direction})"

        except Exception as e:
            logger.warning(f"Pullback filter error: {e}")
            return True, f"Pullback check error: {e}"

    async def _execute_trade(self, signal: SMCSignal, position):
        """Execute trade order."""
        logger.info("=" * 50)
        logger.info(f"TRADE SIGNAL: {signal.signal_type}")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  SL: {signal.stop_loss:.2f}")
        logger.info(f"  TP: {signal.take_profit:.2f}")
        logger.info(f"  Lot: {position.lot_size}")
        logger.info(f"  Risk: ${position.risk_amount:.2f} ({position.risk_percent:.2f}%)")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
        logger.info(f"  Reason: {signal.reason}")
        logger.info("=" * 50)
        
        if self.simulation:
            logger.info("[SIMULATION] Trade not executed")
            self._last_signal = signal
            self._last_trade_time = datetime.now()
            return
        
        # Send order
        result = self.mt5.send_order(
            symbol=self.config.symbol,
            order_type=signal.signal_type,
            volume=position.lot_size,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            magic=self.config.magic_number,
            comment="AI Bot",
        )
        
        if result.success:
            logger.info(f"ORDER EXECUTED! ID: {result.order_id}")
            self._last_signal = signal
            self._last_trade_time = datetime.now()

            # Get current regime and volatility for notification
            regime = self._last_regime.value if hasattr(self, '_last_regime') else "unknown"
            session_status = self.session_filter.get_status_report()
            volatility = session_status.get("volatility", "unknown")

            # Send Telegram notification (stores trade info + builds context internally)
            await self.notifications.notify_trade_open(
                result=result,
                signal=signal,
                position=position,
                regime=regime,
                volatility=volatility,
                session_status=session_status,
            )
        else:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")

    async def _execute_trade_safe(self, signal: SMCSignal, position, regime_state):
        """
        Execute trade dengan mode ULTRA SAFE v2.

        PRINSIP:
        1. Lot size SANGAT KECIL (0.01-0.03)
        2. Emergency broker SL sebagai safety net (2% = ~$100)
        3. Software S/L lebih ketat (1% = ~$50)
        4. Smart management untuk exit (ML reversal detection)
        """
        # Calculate emergency broker SL (safety net)
        emergency_sl = self.smart_risk.calculate_emergency_sl(
            entry_price=signal.entry_price,
            direction=signal.signal_type,
            lot_size=position.lot_size,
            symbol=self.config.symbol,
        )

        logger.info("=" * 50)
        logger.info("SAFE TRADE MODE v2 - SMART S/L")
        logger.info("=" * 50)
        logger.info(f"TRADE SIGNAL: {signal.signal_type}")
        logger.info(f"  Entry: {signal.entry_price:.2f}")
        logger.info(f"  TP: {signal.take_profit:.2f}")
        logger.info(f"  Emergency SL: {emergency_sl:.2f} (broker safety net)")
        logger.info(f"  Software S/L: ${self.smart_risk.max_loss_per_trade:.2f} (smart management)")
        logger.info(f"  Lot: {position.lot_size} (Ultra Safe)")
        logger.info(f"  Confidence: {signal.confidence:.2%}")
        logger.info(f"  Reason: {signal.reason}")
        logger.info("=" * 50)

        if self.simulation:
            logger.info("[SIMULATION] Trade not executed")
            self._last_signal = signal
            self._last_trade_time = datetime.now()
            return

        # === FIX: Use broker-level SL for protection ===
        # SMC signal now has ATR-based SL (minimum 1.5 ATR distance)
        # Use this as primary SL, with emergency backup
        broker_sl = signal.stop_loss

        # Validate SL is far enough from current price (min 10 pips for XAUUSD)
        tick = self.mt5.get_tick(self.config.symbol)
        current_price = tick.bid if signal.signal_type == "SELL" else tick.ask

        min_sl_distance = 1.0  # Minimum $1 distance (10 pips for XAUUSD)
        if signal.signal_type == "BUY":
            if current_price - broker_sl < min_sl_distance:
                broker_sl = current_price - (min_sl_distance * 2)  # Force wider SL
        else:  # SELL
            if broker_sl - current_price < min_sl_distance:
                broker_sl = current_price + (min_sl_distance * 2)  # Force wider SL

        logger.info(f"  Broker SL: {broker_sl:.2f} (ATR-based protection)")

        # Send order WITH broker SL
        result = self.mt5.send_order(
            symbol=self.config.symbol,
            order_type=signal.signal_type,
            volume=position.lot_size,
            sl=broker_sl,  # BROKER-LEVEL PROTECTION (ATR-based)
            tp=signal.take_profit,
            magic=self.config.magic_number,
            comment="AI Safe v3",
        )

        # Fallback: If SL rejected, try without SL (software will manage)
        if not result.success and result.retcode == 10016:
            logger.warning(f"Broker SL rejected, trying without SL...")
            result = self.mt5.send_order(
                symbol=self.config.symbol,
                order_type=signal.signal_type,
                volume=position.lot_size,
                sl=0,  # Fallback to software SL
                tp=signal.take_profit,
                magic=self.config.magic_number,
                comment="AI Safe v3 NoSL",
            )

        if result.success:
            logger.info(f"SAFE ORDER EXECUTED! ID: {result.order_id}")
            self._last_signal = signal
            self._last_trade_time = datetime.now()

            # === SLIPPAGE VALIDATION ===
            expected_price = signal.entry_price
            actual_price = result.price if result.price > 0 else expected_price
            slippage = abs(actual_price - expected_price)
            slippage_pips = slippage * 10  # For XAUUSD, $1 = 10 pips

            # Max acceptable slippage: 0.15% or $7 for XAUUSD
            max_slippage = expected_price * 0.0015  # 0.15% of price

            if slippage > max_slippage:
                logger.warning(f"HIGH SLIPPAGE: Expected {expected_price:.2f}, Got {actual_price:.2f} (slip: ${slippage:.2f} / {slippage_pips:.1f} pips)")
            elif slippage > 0:
                logger.info(f"Slippage OK: ${slippage:.2f} ({slippage_pips:.1f} pips)")

            # === PARTIAL FILL CHECK ===
            requested_volume = position.lot_size
            filled_volume = result.volume if result.volume > 0 else requested_volume

            if filled_volume < requested_volume:
                fill_ratio = filled_volume / requested_volume * 100
                logger.warning(f"PARTIAL FILL: Requested {requested_volume}, Got {filled_volume} ({fill_ratio:.1f}%)")
                # Update position with actual filled volume
                position.lot_size = filled_volume
            elif filled_volume > 0:
                logger.debug(f"Full fill: {filled_volume} lots")

            # Use actual price and volume for registration
            entry_price_actual = actual_price if actual_price > 0 else signal.entry_price
            lot_size_actual = filled_volume

            # Register with smart risk manager (use actual values)
            self.smart_risk.register_position(
                ticket=result.order_id,
                entry_price=entry_price_actual,  # Actual entry price
                lot_size=lot_size_actual,        # Actual filled volume
                direction=signal.signal_type,
            )

            # Get current regime and volatility for notification
            regime = self._last_regime.value if hasattr(self, '_last_regime') else "unknown"
            session_status = self.session_filter.get_status_report()
            volatility = session_status.get("volatility", "unknown")

            # Store trade info for close notification (use actual values)
            self._open_trade_info[result.order_id] = {
                "entry_price": entry_price_actual,  # Actual price
                "expected_price": signal.entry_price,
                "slippage": slippage,
                "lot_size": lot_size_actual,        # Actual filled volume
                "requested_lot_size": requested_volume,
                "open_time": datetime.now(),
                "balance_before": self.mt5.account_balance,
                "ml_confidence": signal.confidence,
                "regime": regime,
                "volatility": volatility,
                "direction": signal.signal_type,
            }

            # Log trade for auto-training
            try:
                # Get SMC details
                smc_fvg = "FVG" in signal.reason.upper()
                smc_ob = "OB" in signal.reason.upper() or "ORDER BLOCK" in signal.reason.upper()
                smc_bos = "BOS" in signal.reason.upper()
                smc_choch = "CHOCH" in signal.reason.upper()

                # Get dynamic confidence info
                market_quality = self.dynamic_confidence._last_quality if hasattr(self.dynamic_confidence, '_last_quality') else "moderate"
                market_score = self.dynamic_confidence._last_score if hasattr(self.dynamic_confidence, '_last_score') else 50
                dynamic_threshold = self.dynamic_confidence._last_threshold if hasattr(self.dynamic_confidence, '_last_threshold') else 0.7

                self.trade_logger.log_trade_open(
                    ticket=result.order_id,
                    symbol=self.config.symbol,
                    direction=signal.signal_type,
                    lot_size=position.lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=0,
                    take_profit=signal.take_profit,
                    regime=regime,
                    volatility=volatility,
                    session=session_status.get("session", "unknown"),
                    spread=self.mt5.get_symbol_info(self.config.symbol).get("spread", 0) if hasattr(self.mt5, 'get_symbol_info') else 0,
                    atr=0,  # ATR calculated in main loop, not available here
                    smc_signal=signal.signal_type,
                    smc_confidence=signal.confidence,
                    smc_reason=signal.reason,
                    smc_fvg=smc_fvg,
                    smc_ob=smc_ob,
                    smc_bos=smc_bos,
                    smc_choch=smc_choch,
                    ml_signal=self._last_ml_signal if hasattr(self, '_last_ml_signal') else "HOLD",
                    ml_confidence=self._last_ml_confidence if hasattr(self, '_last_ml_confidence') else 0.5,
                    market_quality=str(market_quality),
                    market_score=int(market_score) if market_score else 50,
                    dynamic_threshold=float(dynamic_threshold) if dynamic_threshold else 0.7,
                    balance=self.mt5.account_balance,
                    equity=self.mt5.account_equity,
                )
            except Exception as e:
                logger.warning(f"Failed to log trade open: {e}")

            # Send Telegram notification (stores trade info + builds context internally)
            await self.notifications.notify_trade_open(
                result=result,
                signal=signal,
                position=position,
                regime=regime,
                volatility=volatility,
                session_status=session_status,
                safe_mode=True,
                smc_fvg=smc_fvg,
                smc_ob=smc_ob,
                smc_bos=smc_bos,
                smc_choch=smc_choch,
                dynamic_threshold=dynamic_threshold,
                market_quality=market_quality,
                market_score=market_score,
            )
        else:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")

    async def _smart_position_management(self, open_positions, df, regime_state, ml_prediction, current_price):
        """
        Smart position management with dual evaluation:
        1. SmartRiskManager: TP, ML reversal, max loss, daily limit
        2. SmartPositionManager: Trailing SL, breakeven, market close, drawdown protection
        """
        # Sync guards with MT5 — remove guards for positions that no longer exist
        # IMPORTANT: Use FRESH MT5 call, not stale open_positions parameter
        # (open_positions may not include positions opened during this loop iteration)
        try:
            fresh_mt5 = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            mt5_tickets = set()
            if fresh_mt5 is not None and not fresh_mt5.is_empty():
                mt5_tickets = set(fresh_mt5["ticket"].to_list())
            stale = set(self.smart_risk._position_guards.keys()) - mt5_tickets
            for ticket in stale:
                self.smart_risk.unregister_position(ticket)
                logger.debug(f"Cleaned stale guard #{ticket}")
        except Exception as e:
            logger.debug(f"Guard sync error: {e}")

        # --- SmartPositionManager: trailing SL, breakeven, market close ---
        if df is not None and len(df) > 0:
            pm_actions = self.position_manager.analyze_positions(
                positions=open_positions,
                df_market=df,
                regime_state=regime_state,
                ml_prediction=ml_prediction,
                current_price=current_price,
            )
            for action in pm_actions:
                if action.action == "TRAIL_SL":
                    result = self.position_manager._modify_sl(action.ticket, action.new_sl)
                    if result["success"]:
                        logger.info(f"Trailing SL #{action.ticket} -> {action.new_sl:.2f}: {action.reason}")
                    else:
                        logger.debug(f"Trail SL failed #{action.ticket}: {result['message']}")
                elif action.action == "CLOSE":
                    logger.info(f"PositionManager Close #{action.ticket}: {action.reason}")
                    result = self.mt5.close_position(action.ticket)
                    if result.success:
                        profit = 0
                        for row in open_positions.iter_rows(named=True):
                            if row["ticket"] == action.ticket:
                                profit = row.get("profit", 0)
                                break
                        risk_result = self.smart_risk.record_trade_result(profit)
                        self.smart_risk.unregister_position(action.ticket)
                        self.position_manager._peak_profits.pop(action.ticket, None)
                        self._pyramid_done_tickets.discard(action.ticket)  # Cleanup pyramid tracking
                        await self.notifications.notify_trade_close_smart(action.ticket, profit, current_price, action.reason)
                        logger.info(f"CLOSED #{action.ticket}: {action.reason}")
                        continue  # Skip SmartRiskManager eval for this ticket

        # --- SmartRiskManager: TP, ML reversal, max loss, daily limit ---
        for row in open_positions.iter_rows(named=True):
            ticket = row["ticket"]
            profit = row.get("profit", 0)
            entry_price = row.get("price_open", current_price)
            lot_size = row.get("volume", 0.01)
            position_type = row.get("type", 0)  # 0=BUY, 1=SELL
            direction = "BUY" if position_type == 0 else "SELL"

            # Skip if already closed by PositionManager above
            current_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            still_open = any(
                r["ticket"] == ticket
                for r in current_positions.iter_rows(named=True)
            ) if len(current_positions) > 0 else False
            if not still_open:
                continue

            # AUTO-REGISTER posisi yang belum terdaftar (dari sebelum bot start)
            if not self.smart_risk.is_position_registered(ticket):
                self.smart_risk.auto_register_existing_position(
                    ticket=ticket,
                    entry_price=entry_price,
                    lot_size=lot_size,
                    direction=direction,
                    current_profit=profit,
                )

            # Calculate ATR for dynamic threshold scaling
            _current_atr = 0.0
            _baseline_atr = 0.0
            if df is not None and "atr" in df.columns:
                atr_series = df["atr"].drop_nulls()
                if len(atr_series) > 0:
                    _current_atr = atr_series.tail(1).item() or 0
                if len(atr_series) >= 96:  # ~24h of M15 data
                    _baseline_atr = atr_series.tail(96).mean()
                elif len(atr_series) >= 20:
                    _baseline_atr = atr_series.mean()

            # Build market context for dynamic exit intelligence
            _market_ctx = None
            if df is not None:
                _market_ctx = {}
                for col in ("rsi", "stoch_k", "adx", "histogram"):
                    if col in df.columns:
                        vals = df[col].drop_nulls()
                        _market_ctx[col if col != "histogram" else "macd_hist"] = (
                            vals.tail(1).item() if len(vals) > 0 else None
                        )

            # Evaluate with smart risk manager (dynamic thresholds v5)
            should_close, reason, message = self.smart_risk.evaluate_position(
                ticket=ticket,
                current_price=current_price,
                current_profit=profit,
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                regime=regime_state.regime.value if regime_state else "normal",
                current_atr=_current_atr,
                baseline_atr=_baseline_atr,
                market_context=_market_ctx,
            )

            # Per-ticket momentum log (~every 30 seconds)
            guard = self.smart_risk._position_guards.get(ticket)
            if guard and len(guard.profit_timestamps) >= 2:
                now_ts = time.time()
                if now_ts - guard.last_momentum_log_time >= 30:
                    guard.last_momentum_log_time = now_ts
                    vel_summary = guard.get_velocity_summary()
                    atr_ratio = _current_atr / _baseline_atr if _baseline_atr > 0 else 1.0
                    logger.info(
                        f"[MOMENTUM] #{ticket} profit=${profit:+.2f} | "
                        f"vel={vel_summary['velocity']:.4f}$/s | "
                        f"accel={vel_summary['acceleration']:.4f} | "
                        f"stag={vel_summary['stagnation_s']:.0f}s | "
                        f"ATR={_current_atr:.1f}({atr_ratio:.2f}x) | "
                        f"samples={vel_summary['samples']}"
                    )

            if should_close:
                logger.info(f"Smart Close #{ticket}: {reason.value if reason else 'unknown'} - {message}")

                # Close position
                result = self.mt5.close_position(ticket)
                if result.success:
                    logger.info(f"CLOSED #{ticket}: {message}")

                    # Record result and check for limit violations
                    risk_result = self.smart_risk.record_trade_result(profit)
                    self.smart_risk.unregister_position(ticket)
                    self._pyramid_done_tickets.discard(ticket)  # Cleanup pyramid tracking

                    # Log trade close for auto-training
                    try:
                        trade_info = self._open_trade_info.get(ticket, {})
                        entry_price = trade_info.get("entry_price", current_price)
                        lot_size = trade_info.get("lot_size", 0.01)

                        # Calculate pips
                        pips = abs(current_price - entry_price) * 100
                        if profit < 0:
                            pips = -pips

                        self.trade_logger.log_trade_close(
                            ticket=ticket,
                            exit_price=current_price,
                            profit_usd=profit,
                            profit_pips=pips,
                            exit_reason=reason.value if reason else message[:30],
                            regime=regime_state.regime.value if regime_state else "normal",
                            ml_signal=ml_prediction.signal if ml_prediction else "HOLD",
                            ml_confidence=ml_prediction.confidence if ml_prediction else 0.5,
                            balance_after=self.mt5.account_balance or 0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log trade close: {e}")

                    # Send notification
                    await self.notifications.notify_trade_close_smart(ticket, profit, current_price, message)

                    # Check for critical limit violations and send alerts
                    if risk_result.get("total_limit_hit"):
                        await self.notifications.send_critical_limit_alert(
                            "TOTAL LOSS LIMIT",
                            risk_result.get("total_loss", 0),
                            self.smart_risk.max_total_loss_usd,
                            self.smart_risk.max_total_loss_percent
                        )
                    elif risk_result.get("daily_limit_hit"):
                        await self.notifications.send_critical_limit_alert(
                            "DAILY LOSS LIMIT",
                            risk_result.get("daily_loss", 0),
                            self.smart_risk.max_daily_loss_usd,
                            self.smart_risk.max_daily_loss_percent
                        )
                else:
                    logger.error(f"Failed to close #{ticket}: {result.comment}")
            else:
                # Just log status periodically
                if self._loop_count % 60 == 0:
                    logger.info(f"Position #{ticket}: {message}")

    async def _emergency_close_all(self, max_retries: int = 3):
        """
        Emergency close all positions with retry logic and error handling.

        CRITICAL: This function must be robust as it's called during flash crashes.
        """
        logger.warning("=" * 50)
        logger.warning("EMERGENCY: Closing all positions!")
        logger.warning("=" * 50)

        if self.simulation:
            return

        failed_tickets = []
        closed_count = 0

        for attempt in range(max_retries):
            try:
                positions = self.mt5.get_open_positions(magic=self.config.magic_number)

                if positions is None or len(positions) == 0:
                    logger.info("No positions to close")
                    break

                for row in positions.iter_rows(named=True):
                    ticket = row["ticket"]
                    try:
                        result = self.mt5.close_position(ticket)
                        if result.success:
                            logger.info(f"Closed position {ticket}")
                            closed_count += 1
                            self._pyramid_done_tickets.discard(ticket)  # Cleanup pyramid tracking
                            self.smart_risk.unregister_position(ticket)  # Cleanup risk tracking
                            # Remove from failed list if was there
                            if ticket in failed_tickets:
                                failed_tickets.remove(ticket)
                        else:
                            logger.error(f"Failed to close {ticket}: {result.comment}")
                            if ticket not in failed_tickets:
                                failed_tickets.append(ticket)
                    except Exception as e:
                        logger.error(f"Exception closing {ticket}: {e}")
                        if ticket not in failed_tickets:
                            failed_tickets.append(ticket)

                # Check if all closed
                remaining = self.mt5.get_open_positions(magic=self.config.magic_number)
                if remaining is None or len(remaining) == 0:
                    logger.info(f"Emergency close complete: {closed_count} positions closed")
                    break

                # If still have positions, wait and retry
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 2}/{max_retries} - {len(remaining)} positions still open")
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Emergency close attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        # Send critical alert
        await self.notifications.send_emergency_close_result(closed_count, failed_tickets)
    
    def _on_new_day(self):
        """Handle new trading day."""
        logger.info("=" * 60)
        logger.info(f"NEW TRADING DAY: {date.today()}")
        logger.info("=" * 60)

        # Daily summary disabled from auto-send (available via command)
        # try:
        #     import asyncio
        #     asyncio.create_task(self._send_daily_summary())
        # except Exception as e:
        #     logger.warning(f"Could not send daily summary: {e}")

        self._current_date = date.today()
        self.risk_engine.reset_daily_stats()

        # Reset daily tracking
        self._daily_start_balance = self.mt5.account_balance or self.config.capital
        self.telegram.set_daily_start_balance(self._daily_start_balance)

        self._log_summary()
    
    def _log_summary(self):
        """Log session summary."""
        if not self._execution_times:
            return
        
        avg_time = sum(self._execution_times) / len(self._execution_times)
        max_time = max(self._execution_times)
        min_time = min(self._execution_times)
        
        logger.info("=" * 40)
        logger.info("SESSION SUMMARY")
        logger.info(f"Total loops: {self._loop_count}")
        logger.info(f"Avg execution: {avg_time*1000:.2f}ms")
        logger.info(f"Min execution: {min_time*1000:.2f}ms")
        logger.info(f"Max execution: {max_time*1000:.2f}ms")
        
        daily = self.risk_engine.get_daily_summary()
        logger.info(f"Trades today: {daily['trades']}")
        logger.info("=" * 40)

    async def _check_auto_retrain(self):
        """
        Check if auto-retraining should happen and execute if needed.
        Called every 5 minutes (300 loops) during main loop.
        """
        try:
            should_train, reason = self.auto_trainer.should_retrain()

            if not should_train:
                logger.debug(f"Auto-retrain check: {reason}")
                return

            logger.info("=" * 50)
            logger.info(f"AUTO-RETRAIN TRIGGERED: {reason}")
            logger.info("=" * 50)

            # Check if market is closed (safe to retrain)
            session_status = self.session_filter.get_status_report()
            if session_status.get("can_trade", True):
                # Market is open - skip training, wait for close
                logger.info("Market still open - will retrain when closed")
                return

            # Close any open positions before retraining
            open_positions = self.mt5.get_open_positions(
                symbol=self.config.symbol,
                magic=self.config.magic_number,
            )
            if len(open_positions) > 0:
                logger.warning(f"Skipping retrain - {len(open_positions)} open positions")
                return

            # Perform retraining
            is_weekend = self.auto_trainer.should_retrain()[1] == "Weekend deep training time"

            results = self.auto_trainer.retrain(
                connector=self.mt5,
                symbol=self.config.symbol,
                timeframe=self.config.execution_timeframe,
                is_weekend=is_weekend,
            )

            if results["success"]:
                logger.info("Retraining successful! Reloading models...")

                # Reload the newly trained models
                self.regime_detector.load()
                self.ml_model.load()

                logger.info(f"  HMM: {'OK' if self.regime_detector.fitted else 'FAILED'}")
                logger.info(f"  XGBoost: {'OK' if self.ml_model.fitted else 'FAILED'}")
                logger.info(f"  Features: {len(self.ml_model.feature_names) if self.ml_model.feature_names else 0}")
                logger.info(f"  Train AUC: {results.get('xgb_train_auc', 0):.4f}")
                logger.info(f"  Test AUC: {results.get('xgb_test_auc', 0):.4f}")

                # Update auto_trainer's cached AUC for dashboard
                self.auto_trainer._current_auc = results.get("xgb_test_auc", 0)

                # Write updated model metrics for dashboard
                self._write_model_metrics(retrain_results=results)

                # Check if new model is worse - rollback if needed
                if results.get("xgb_test_auc", 0) < 0.60:
                    logger.warning("New model AUC too low - rolling back!")
                    self.auto_trainer.rollback_models()
                    self.regime_detector.load()
                    self.ml_model.load()
                    logger.info("Rollback complete")
            else:
                logger.error(f"Retraining failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Auto-retrain error: {e}")
            import traceback
            logger.debug(traceback.format_exc())


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart AI Trading Bot")
    parser.add_argument("--simulation", "-s", action="store_true", help="Run in simulation mode")
    parser.add_argument("--capital", "-c", type=float, help="Trading capital (override)")
    parser.add_argument("--symbol", type=str, help="Trading symbol (override)")
    args = parser.parse_args()
    
    # Load config from .env
    config = get_config()
    
    # Override if provided
    if args.capital:
        config = TradingConfig(capital=args.capital, symbol=config.symbol)
    if args.symbol:
        config.symbol = args.symbol
    
    # Create and run bot
    bot = TradingBot(config=config, simulation=args.simulation)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.stop()


def _acquire_lock():
    """Prevent duplicate bot instances via PID lockfile."""
    lockfile = Path("data/bot.lock")
    lockfile.parent.mkdir(exist_ok=True)

    if lockfile.exists():
        try:
            old_pid = int(lockfile.read_text().strip())
            # Check if old process is still alive (Windows)
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            if f"{old_pid}" in result.stdout and "python" in result.stdout.lower():
                logger.error(f"ANOTHER BOT INSTANCE IS RUNNING (PID {old_pid})!")
                logger.error("Kill it first: taskkill /F /PID " + str(old_pid))
                sys.exit(1)
            else:
                logger.info(f"Stale lockfile found (PID {old_pid} not running), removing...")
        except (ValueError, Exception) as e:
            logger.warning(f"Could not check lockfile: {e}, removing...")

    # Write our PID
    lockfile.write_text(str(os.getpid()))
    logger.info(f"Bot lockfile acquired: PID {os.getpid()}")
    return lockfile


def _release_lock():
    """Release PID lockfile."""
    lockfile = Path("data/bot.lock")
    try:
        if lockfile.exists():
            stored_pid = int(lockfile.read_text().strip())
            if stored_pid == os.getpid():
                lockfile.unlink()
                logger.info("Bot lockfile released")
    except Exception:
        pass


if __name__ == "__main__":
    lockfile = _acquire_lock()
    try:
        asyncio.run(main())
    finally:
        _release_lock()
