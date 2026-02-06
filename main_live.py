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
from datetime import datetime, date
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
from src.ml_model import TradingModel, get_default_feature_columns
from src.position_manager import SmartPositionManager
from src.session_filter import SessionFilter, create_wib_session_filter
from src.auto_trainer import AutoTrainer, create_auto_trainer
from src.telegram_notifier import TelegramNotifier, create_telegram_notifier
from src.smart_risk_manager import SmartRiskManager, create_smart_risk_manager
from src.dynamic_confidence import DynamicConfidenceManager, create_dynamic_confidence
from src.news_agent import NewsAgent, create_news_agent, MarketCondition
from src.trade_logger import TradeLogger, get_trade_logger


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
        
        # Initialize ML model (will load model)
        self.ml_model = TradingModel(
            confidence_threshold=self.config.ml.confidence_threshold,
            model_path="models/xgboost_model.pkl",
        )

        # Initialize Smart Position Manager - ULTRA SAFE MODE
        self.position_manager = SmartPositionManager(
            breakeven_pips=5.0,        # Move to breakeven after 5 pips profit
            trail_start_pips=10.0,     # Start trailing after 10 pips
            trail_step_pips=5.0,       # Trail by 5 pips
            min_profit_to_protect=5.0,   # Protect profits > $5
            max_drawdown_from_peak=50.0,  # Allow 50% drawdown (we use tiny lots)
            # Smart Market Close Handler
            enable_market_close_handler=True,
            min_profit_before_close=5.0,   # Take profit >= $5 before market close
            max_loss_to_hold=30.0,     # Max loss $30 per position
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

        # Initialize News Agent - economic calendar monitoring (NO BLOCKING)
        # Based on comprehensive backtest (29 trades, 62.1% WR, $178 profit):
        # - News filter COSTS us $178.15 profit
        # - ML model already handles market volatility well
        # - Keep monitoring for logging but DO NOT block trades
        self.news_agent = create_news_agent(
            news_buffer_minutes=0,  # No blocking
            high_impact_buffer_minutes=0,  # No blocking - ML model handles volatility
        )

        # Initialize Trade Logger - for ML auto-training
        self.trade_logger = get_trade_logger()

        # State tracking
        self._running = False
        self._loop_count = 0
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
        self._last_market_update_time: Optional[datetime] = None
        self._last_hourly_report_time: Optional[datetime] = None
        self._open_trade_info: Dict = {}  # Track trade info for close notification
        self._last_news_alert_reason: Optional[str] = None  # Track news alert to avoid duplicates
        self._current_session_multiplier: float = 1.0  # Session lot multiplier
        self._is_sydney_session: bool = False  # Sydney session flag (needs higher confidence)
        self._last_candle_time: Optional[datetime] = None  # Track last processed candle
        self._position_check_interval: int = 10  # Check positions every N seconds between candles

        # Dashboard status bridge (written to JSON for Docker API)
        self._dash_price_history: deque = deque(maxlen=120)
        self._dash_equity_history: deque = deque(maxlen=120)
        self._dash_balance_history: deque = deque(maxlen=120)
        self._dash_logs: deque = deque(maxlen=50)
        self._dash_last_price: float = 0.0
        self._dash_status_file = Path("data/bot_status.json")
    
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
        
        # Load XGBoost model
        try:
            self.ml_model.load()
            if self.ml_model.fitted:
                logger.info("XGBoost model loaded successfully")
                logger.info(f"  Features: {len(self.ml_model.feature_names)}")
            else:
                logger.warning("XGBoost model not found or not fitted")
                models_ok = False
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            models_ok = False
        
        self._models_loaded = models_ok
        return models_ok

    def _dash_log(self, level: str, message: str):
        """Add log entry to dashboard buffer."""
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
        self._dash_logs.append({
            "time": now.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        })

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
                "buyProb": ml_prob if ml_signal == "BUY" else (1.0 - ml_prob),
                "sellProb": ml_prob if ml_signal == "SELL" else (1.0 - ml_prob),
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
                    "capital": self.config.capital,
                    "riskPerTrade": self.config.risk.risk_per_trade,
                    "maxDailyLoss": self.config.risk.max_daily_loss,
                    "maxPositions": self.config.risk.max_positions,
                    "maxLotSize": self.config.risk.max_lot_size,
                    "leverage": self.config.risk.max_leverage,
                    "executionTF": self.config.execution_timeframe,
                    "trendTF": self.config.trend_timeframe,
                    "minRR": 1.5,
                    "mlConfidence": self.config.ml.confidence_threshold,
                    "cooldownSeconds": self.config.thresholds.trade_cooldown_seconds,
                    "symbol": self.config.symbol,
                },
            }

            # Atomic write (write to temp then rename)
            tmp_file = self._dash_status_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(status, default=str))
            tmp_file.replace(self._dash_status_file)

        except Exception as e:
            logger.debug(f"Dashboard status write error: {e}")

    async def start(self):
        """Start the trading bot."""
        logger.info("=" * 60)
        logger.info("SMART AUTOMATIC TRADING BOT + AI")
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

            # Show news agent status
            news_can_trade, news_reason, _ = self.news_agent.should_trade()
            logger.info(f"News Agent: {'SAFE' if news_can_trade else 'BLOCKED'} - {news_reason}")

            # Track daily start balance
            self._daily_start_balance = balance
            self._start_time = datetime.now()
            self.telegram.set_daily_start_balance(balance)

            # Send Telegram startup notification
            ml_status = f"Loaded ({len(self.ml_model.feature_names)} features)" if self.ml_model.fitted else "Not loaded"
            news_status = "SAFE" if news_can_trade else "BLOCKED"
            await self.telegram.send_startup_message(
                symbol=self.config.symbol,
                capital=self.config.capital,
                balance=balance,
                mode=self.config.capital_mode.value,
                ml_model_status=ml_status,
                news_status=news_status,
            )

        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            if not self.simulation:
                return

        # Start main loop
        self._running = True
        self._dash_log("info", "Bot started - trading loop active")
        logger.info("Starting main trading loop...")
        await self._main_loop()
    
    async def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False

        # Calculate uptime
        uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600

        # Send Telegram shutdown notification
        try:
            balance = self.mt5.account_balance or self.config.capital
            await self.telegram.send_shutdown_message(
                balance=balance,
                total_trades=self._total_session_trades,
                total_profit=self._total_session_profit,
                uptime_hours=uptime_hours,
            )
            await self.telegram.close()
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")

        self.mt5.disconnect()
        self._log_summary()
    
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
        Determine H1 higher-timeframe bias using SMC structure.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"

        Logic:
        - Fetch H1 data (100 bars)
        - Run SMC analysis (BOS, CHoCH, OB, FVG)
        - Last BOS/CHoCH direction = H1 bias
        - If H1 has bullish OB near price → BULLISH zone
        - If H1 has bearish OB near price → BEARISH zone
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

            if len(df_h1) < 20:
                return "NEUTRAL"

            # Run SMC on H1 data
            from src.smc_polars import SMCAnalyzer
            h1_smc = SMCAnalyzer(swing_length=5, fvg_min_gap_pips=5.0, ob_lookback=10)
            df_h1 = h1_smc.calculate_all(df_h1)

            current_price = df_h1["close"].tail(1).item()
            bias = "NEUTRAL"

            # 1. Check last BOS direction on H1
            bos_col = df_h1["bos"].to_list()
            last_bos = 0
            for v in reversed(bos_col[-20:]):
                if v != 0:
                    last_bos = v
                    break

            # 2. Check last CHoCH direction on H1
            choch_col = df_h1["choch"].to_list()
            last_choch = 0
            for v in reversed(choch_col[-20:]):
                if v != 0:
                    last_choch = v
                    break

            # 3. Check if price is near H1 Order Block
            ob_col = df_h1["ob"].to_list()
            highs = df_h1["high"].to_list()
            lows = df_h1["low"].to_list()
            near_bullish_ob = False
            near_bearish_ob = False

            for i in range(-10, 0):  # Last 10 H1 candles
                idx = len(ob_col) + i
                if idx < 0:
                    continue
                ob_val = ob_col[idx]
                if ob_val == 1:  # Bullish OB
                    # Price within OB zone (low to high of that candle)
                    if lows[idx] <= current_price <= highs[idx] * 1.002:
                        near_bullish_ob = True
                elif ob_val == -1:  # Bearish OB
                    if lows[idx] * 0.998 <= current_price <= highs[idx]:
                        near_bearish_ob = True

            # Determine bias: BOS > CHoCH > OB proximity
            if last_bos == 1:
                bias = "BULLISH"
            elif last_bos == -1:
                bias = "BEARISH"
            elif last_choch == 1:
                bias = "BULLISH"
            elif last_choch == -1:
                bias = "BEARISH"

            # OB proximity can override if no clear structure
            if bias == "NEUTRAL":
                if near_bullish_ob:
                    bias = "BULLISH"
                elif near_bearish_ob:
                    bias = "BEARISH"

            # Cache result
            self._h1_bias_cache = bias
            self._h1_bias_loop = self._loop_count

            if self._loop_count % 4 == 0:
                logger.info(f"H1 Bias: {bias} (BOS={last_bos}, CHoCH={last_choch}, near_bull_OB={near_bullish_ob}, near_bear_OB={near_bearish_ob})")

            return bias

        except Exception as e:
            logger.debug(f"H1 bias error: {e}")
            return "NEUTRAL"

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
                        try:
                            await self.telegram.send_message(
                                f"CRITICAL: Flash crash {move_pct:.2f}% but emergency close FAILED!\n"
                                f"Error: {e}\nMANUAL INTERVENTION REQUIRED!"
                            )
                        except:
                            pass
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
                    feature_cols = self._get_available_features(df)
                    ml_prediction = self.ml_model.predict(df, feature_cols)
                    await self._smart_position_management(
                        open_positions=open_positions,
                        df=df,
                        regime_state=cached_regime,
                        ml_prediction=ml_prediction,
                        current_price=current_price,
                    )
        except Exception as e:
            logger.debug(f"Position check error: {e}")
    
    async def _trading_iteration(self):
        """Single trading iteration."""
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
            logger.debug(f"Regime detection error: {e}")
            regime_state = None
        
        # 5. Check flash crash
        is_flash, move_pct = self.flash_crash.detect(df.tail(5))
        if is_flash:
            logger.warning(f"Flash crash detected: {move_pct:.2f}% move")
            try:
                await self._emergency_close_all()
            except Exception as e:
                logger.critical(f"CRITICAL: Emergency close failed completely: {e}")
                # Try to send alert even if close failed
                try:
                    await self.telegram.send_message(
                        f"🚨🚨 CRITICAL ERROR 🚨🚨\n\n"
                        f"Flash crash detected but emergency close FAILED!\n"
                        f"Error: {e}\n\n"
                        f"MANUAL INTERVENTION REQUIRED!"
                    )
                except:
                    pass
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
        await self._send_hourly_analysis_if_due(
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
        if regime_state and regime_state.recommendation == "SLEEP":
            logger.debug(f"Regime SLEEP: {regime_state.regime.value}")
            return

        if not risk_metrics.can_trade:
            logger.debug(f"Risk blocked: {risk_metrics.reason}")
            return

        # 7.5 Check trading session (WIB timezone)
        session_ok, session_reason, session_multiplier = self.session_filter.can_trade()
        if not session_ok:
            if self._loop_count % 300 == 0:  # Log every 5 minutes
                logger.info(f"Session filter: {session_reason}")
                next_window = self.session_filter.get_next_trading_window()
                logger.info(f"Next trading window: {next_window['session']} in {next_window['hours_until']} hours")
            return

        # Store session info for later use (Sydney needs higher confidence)
        self._current_session_multiplier = session_multiplier
        self._is_sydney_session = "Sydney" in session_reason or session_multiplier == 0.5

        # 7.6 NEWS AGENT MONITORING (NO BLOCKING)
        # Based on backtest analysis: News filter COSTS $178 profit
        # ML model already handles volatility well - no need to block
        can_trade_news, news_reason, news_lot_mult = self.news_agent.should_trade()

        # Log news status for monitoring but DO NOT block trades
        if not can_trade_news and self._loop_count % 300 == 0:
            logger.info(f"News Agent: HIGH IMPACT NEWS - {news_reason} (trading allowed)")

        # Note: We no longer block trades during news events
        # Backtest showed trades during news have 62.1% win rate (vs 64.9% normal)
        # The $178 profit opportunity outweighs the minimal risk difference

        # 7.7 H1 Multi-Timeframe Bias (Fix 5)
        # Fetch H1 data and determine higher-TF bias for M15 signal filtering
        h1_bias = self._get_h1_bias()

        # 8. Get SMC signal
        smc_signal = self.smc.generate_signal(df)

        # Cache raw SMC for dashboard (before filtering)
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

        # 9. ML prediction already done above for position management

        # Log signal status every 4 loops (~1 hour on M15)
        if self._loop_count % 4 == 0:
            price = df["close"].tail(1).item()
            h1_tag = f" | H1: {h1_bias}" if h1_bias != "NEUTRAL" else ""
            logger.info(f"Price: {price:.2f} | Regime: {regime_state.regime.value if regime_state else 'N/A'} | SMC: {smc_signal.signal_type if smc_signal else 'NONE'} | ML: {ml_prediction.signal}({ml_prediction.confidence:.0%}){h1_tag}")

        # Send market update to Telegram (every 30 minutes) - only after first loop
        if self._loop_count > 0 and self._loop_count % 30 == 0:
            await self._send_market_update(df, regime_state, ml_prediction)

        # 10. Combine signals
        final_signal = self._combine_signals(smc_signal, ml_prediction, regime_state)

        if final_signal is None:
            return

        # 10.1 H1 Multi-Timeframe Filter (Fix 5)
        # Block M15 signal if it contradicts H1 bias
        if h1_bias != "NEUTRAL":
            if final_signal.signal_type == "BUY" and h1_bias == "BEARISH":
                logger.info(f"Skip BUY: H1 bias is BEARISH (counter-trend)")
                return
            if final_signal.signal_type == "SELL" and h1_bias == "BULLISH":
                logger.info(f"Skip SELL: H1 bias is BULLISH (counter-trend)")
                return
            # Boost confidence when aligned
            if (final_signal.signal_type == "BUY" and h1_bias == "BULLISH") or \
               (final_signal.signal_type == "SELL" and h1_bias == "BEARISH"):
                final_signal = SMCSignal(
                    signal_type=final_signal.signal_type,
                    entry_price=final_signal.entry_price,
                    stop_loss=final_signal.stop_loss,
                    take_profit=final_signal.take_profit,
                    confidence=min(final_signal.confidence * 1.1, 0.95),
                    reason=f"{final_signal.reason} | H1-ALIGNED",
                )

        # 10.5 Check trade cooldown
        if self._last_trade_time:
            time_since_last = (datetime.now() - self._last_trade_time).total_seconds()
            if time_since_last < self._trade_cooldown_seconds:
                logger.debug(f"Trade cooldown: {self._trade_cooldown_seconds - time_since_last:.0f}s remaining")
                return

        # 10.6 PULLBACK FILTER - Prevent entry during temporary retracements
        pullback_ok, pullback_reason = self._check_pullback_filter(
            df=df,
            signal_direction=final_signal.signal_type,
            current_price=current_price,
        )
        if not pullback_ok:
            if self._loop_count % 30 == 0:  # Log every 30 loops
                logger.info(f"Pullback Filter: {pullback_reason}")
            return

        # 11. SMART RISK CHECK - Ultra safe mode
        self.smart_risk.check_new_day()
        risk_rec = self.smart_risk.get_trading_recommendation()

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
            safe_lot = max(0.01, safe_lot * session_mult)  # Minimum 0.01
            sydney_mode = getattr(self, '_is_sydney_session', False)
            if sydney_mode:
                logger.info(f"Sydney SAFE MODE: Lot {original_lot:.2f} -> {safe_lot:.2f} (0.5x)")

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
        # IMPROVED SIGNAL LOGIC v3 - With ML Threshold & Confirmation
        # ============================================================
        golden_marker = "[GOLDEN] " if is_golden_time else ""
        if smc_signal is not None:
            # === IMPROVEMENT 1: ML Confidence Threshold ===
            # Based on backtest tuning (Jan 2025 - Feb 2026):
            # - 50% threshold: 485 trades, 61.6% WR, $3120 profit, PF 2.02
            # - 55% threshold: 306 trades, 59.5% WR, $1443 profit, PF 1.74
            # OPTIMAL: 50% threshold (more trades, higher WR, better profit)
            ml_min_threshold = 0.50
            if ml_prediction.confidence < ml_min_threshold:
                if self._loop_count % 60 == 0:
                    logger.info(f"Skip: ML uncertain ({ml_prediction.confidence:.0%} < {ml_min_threshold:.0%}) - waiting for clearer signal")
                return None

            # Check if ML strongly disagrees (>65% opposite)
            ml_strongly_disagrees = (
                (smc_signal.signal_type == "BUY" and ml_prediction.signal == "SELL" and ml_prediction.confidence > 0.65) or
                (smc_signal.signal_type == "SELL" and ml_prediction.signal == "BUY" and ml_prediction.confidence > 0.65)
            )

            if ml_strongly_disagrees:
                if self._loop_count % 60 == 0:
                    logger.info(f"Skip: ML strongly disagrees ({ml_prediction.signal} {ml_prediction.confidence:.0%}) vs SMC {smc_signal.signal_type}")
                return None

            # === IMPROVEMENT 1.5: SELL Filter (OPTIMIZED) ===
            # SELL signals historically have lower win rate than BUY
            # Require ML agreement and higher confidence for SELL
            if smc_signal.signal_type == "SELL":
                if ml_prediction.signal != "SELL":
                    if self._loop_count % 60 == 0:
                        logger.info(f"Skip SELL: ML does not agree ({ml_prediction.signal} {ml_prediction.confidence:.0%})")
                    return None
                if ml_prediction.confidence < 0.55:
                    if self._loop_count % 60 == 0:
                        logger.info(f"Skip SELL: ML confidence too low ({ml_prediction.confidence:.0%} < 55%)")
                    return None

            # === IMPROVEMENT 2: Signal Confirmation (Entry Delay) ===
            # Track signal persistence - only entry if signal consistent for 2+ candles
            # Fix 3: Use direction-only key (not exact price) and persist to file
            signal_key = smc_signal.signal_type  # "BUY" or "SELL" — direction matters, not exact price
            current_time = time.time()

            if not hasattr(self, '_signal_persistence'):
                self._signal_persistence = self._load_signal_persistence()

            # Cleanup: Remove entries older than 30 minutes (1800 seconds)
            self._signal_persistence = {
                k: v for k, v in self._signal_persistence.items()
                if current_time - v[1] < 1800
            }

            if signal_key not in self._signal_persistence:
                self._signal_persistence[signal_key] = (1, current_time)
                self._save_signal_persistence()
                logger.debug(f"Signal confirmation: {signal_key} seen 1st time - waiting")
                return None  # Wait for confirmation
            else:
                count, _ = self._signal_persistence[signal_key]
                self._signal_persistence[signal_key] = (count + 1, current_time)

            # Require at least 2 consecutive confirmations (2 candles)
            count, _ = self._signal_persistence[signal_key]
            if count < 2:
                self._save_signal_persistence()
                logger.debug(f"Signal confirmation: {signal_key} count={count} - waiting")
                return None

            # Signal confirmed! Reset counter
            logger.info(f"Signal CONFIRMED: {signal_key} after {count} checks")
            self._signal_persistence[signal_key] = (0, current_time)
            self._save_signal_persistence()

            # SMC-Only: Use SMC signal with confidence adjustment
            ml_agrees = (
                (smc_signal.signal_type == "BUY" and ml_prediction.signal == "BUY") or
                (smc_signal.signal_type == "SELL" and ml_prediction.signal == "SELL")
            )

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

            # Store trade info for close notification
            self._open_trade_info[result.order_id] = {
                "entry_price": signal.entry_price,
                "open_time": datetime.now(),
                "balance_before": self.mt5.account_balance,
                "ml_confidence": signal.confidence,
                "regime": regime,
                "volatility": volatility,
            }

            # Send Telegram notification
            try:
                await self.telegram.notify_trade_open(
                    ticket=result.order_id,
                    symbol=self.config.symbol,
                    order_type=signal.signal_type,
                    lot_size=position.lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    ml_confidence=signal.confidence,
                    signal_reason=signal.reason,
                    regime=regime,
                    volatility=volatility,
                )
            except Exception as e:
                logger.warning(f"Failed to send trade open notification: {e}")
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

            # Send Telegram notification
            try:
                await self.telegram.notify_trade_open(
                    ticket=result.order_id,
                    symbol=self.config.symbol,
                    order_type=signal.signal_type,
                    lot_size=position.lot_size,
                    entry_price=signal.entry_price,
                    stop_loss=0,  # No SL
                    take_profit=signal.take_profit,
                    ml_confidence=signal.confidence,
                    signal_reason=f"SAFE MODE: {signal.reason}",
                    regime=regime,
                    volatility=volatility,
                )
            except Exception as e:
                logger.warning(f"Failed to send trade open notification: {e}")
        else:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")

    async def _smart_position_management(self, open_positions, df, regime_state, ml_prediction, current_price):
        """
        Smart position management with dual evaluation:
        1. SmartRiskManager: TP, ML reversal, max loss, daily limit
        2. SmartPositionManager: Trailing SL, breakeven, market close, drawdown protection
        """
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
                        await self._notify_trade_close_smart(action.ticket, profit, current_price, action.reason)
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

            # Evaluate with smart risk manager
            should_close, reason, message = self.smart_risk.evaluate_position(
                ticket=ticket,
                current_price=current_price,
                current_profit=profit,
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                regime=regime_state.regime.value if regime_state else "normal",
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
                    await self._notify_trade_close_smart(ticket, profit, current_price, message)

                    # Check for critical limit violations and send alerts
                    if risk_result.get("total_limit_hit"):
                        await self._send_critical_limit_alert(
                            "TOTAL LOSS LIMIT",
                            risk_result.get("total_loss", 0),
                            self.smart_risk.max_total_loss_usd,
                            self.smart_risk.max_total_loss_percent
                        )
                    elif risk_result.get("daily_limit_hit"):
                        await self._send_critical_limit_alert(
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

    async def _notify_trade_close_smart(self, ticket: int, profit: float, current_price: float, reason: str):
        """Send notification for smart close."""
        try:
            trade_info = self._open_trade_info.pop(ticket, {})

            balance_before = trade_info.get("balance_before", 0)
            balance_after = self.mt5.account_balance or 0
            entry_price = trade_info.get("entry_price", current_price)
            duration = int((datetime.now() - trade_info.get("open_time", datetime.now())).total_seconds())

            # Track stats
            self._total_session_profit += profit
            self._total_session_trades += 1

            await self.telegram.notify_trade_close(
                ticket=ticket,
                symbol=self.config.symbol,
                order_type=trade_info.get("direction", "BUY"),
                lot_size=trade_info.get("lot_size", 0.01),
                entry_price=entry_price,
                close_price=current_price,
                profit=profit,
                profit_pips=(current_price - entry_price) / 0.1,
                balance_before=balance_before,
                balance_after=balance_after,
                duration_seconds=duration,
                ml_confidence=trade_info.get("ml_confidence", 0),
                regime=trade_info.get("regime", "unknown"),
                volatility=trade_info.get("volatility", "unknown"),
            )
        except Exception as e:
            logger.warning(f"Failed to send close notification: {e}")

    async def _send_critical_limit_alert(
        self,
        limit_type: str,
        current_loss: float,
        max_loss: float,
        max_percent: float
    ):
        """
        Send critical alert when loss limits are reached.

        Args:
            limit_type: "DAILY LOSS LIMIT" or "TOTAL LOSS LIMIT"
            current_loss: Current loss amount
            max_loss: Maximum allowed loss
            max_percent: Maximum loss percentage
        """
        logger.critical("=" * 60)
        logger.critical(f"CRITICAL: {limit_type} REACHED!")
        logger.critical(f"Loss: ${current_loss:.2f} / ${max_loss:.2f} ({max_percent}%)")
        logger.critical("TRADING HAS BEEN STOPPED!")
        logger.critical("=" * 60)

        try:
            if limit_type == "TOTAL LOSS LIMIT":
                message = (
                    f"🚨🚨 CRITICAL: TOTAL LOSS LIMIT REACHED 🚨🚨\n\n"
                    f"Total Loss: ${current_loss:.2f}\n"
                    f"Limit: ${max_loss:.2f} ({max_percent}%)\n\n"
                    f"⛔ TRADING STOPPED PERMANENTLY\n"
                    f"Manual reset required to resume trading.\n\n"
                    f"Please review your trading strategy."
                )
            else:
                message = (
                    f"🚨 DAILY LOSS LIMIT REACHED 🚨\n\n"
                    f"Daily Loss: ${current_loss:.2f}\n"
                    f"Limit: ${max_loss:.2f} ({max_percent}%)\n\n"
                    f"⛔ TRADING STOPPED FOR TODAY\n"
                    f"Will resume tomorrow automatically."
                )

            await self.telegram.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")

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

        # Send critical alert if any failed
        if failed_tickets:
            alert_msg = f"CRITICAL: Failed to close {len(failed_tickets)} positions: {failed_tickets}"
            logger.error(alert_msg)
            try:
                await self.telegram.send_message(
                    f"🚨 EMERGENCY CLOSE FAILED!\n\n"
                    f"Failed tickets: {failed_tickets}\n"
                    f"Please close manually!"
                )
            except:
                pass  # Don't let telegram failure stop us
        else:
            try:
                await self.telegram.send_message(
                    f"🚨 EMERGENCY CLOSE COMPLETE\n\n"
                    f"Closed {closed_count} positions due to flash crash detection"
                )
            except:
                pass
    
    async def _notify_trade_close(self, action, current_price: float):
        """Send Telegram notification for trade close."""
        try:
            ticket = action.ticket

            # Get trade info from our stored data
            trade_info = self._open_trade_info.pop(ticket, {})
            entry_price = trade_info.get("entry_price", current_price)
            open_time = trade_info.get("open_time", datetime.now())
            balance_before = trade_info.get("balance_before", self._daily_start_balance)
            ml_confidence = trade_info.get("ml_confidence", 0)
            regime = trade_info.get("regime", "unknown")
            volatility = trade_info.get("volatility", "unknown")

            # Get current balance (after close)
            balance_after = self.mt5.account_balance or self.config.capital

            # Calculate profit from action
            profit = action.profit if hasattr(action, 'profit') else 0
            if profit == 0:
                # Try to calculate from price difference (rough estimate)
                profit = balance_after - balance_before

            # Calculate duration
            duration_seconds = int((datetime.now() - open_time).total_seconds())

            # Calculate pips (for XAUUSD, 1 pip = 0.1)
            price_diff = current_price - entry_price
            profit_pips = price_diff / 0.1 if "XAU" in self.config.symbol else price_diff / 0.0001

            # Get order type from action
            order_type = "BUY"  # Default, will be extracted from action if available

            # Track session stats
            self._total_session_profit += profit
            self._total_session_trades += 1

            await self.telegram.notify_trade_close(
                ticket=ticket,
                symbol=self.config.symbol,
                order_type=order_type,
                lot_size=0.2,  # Will be extracted from actual position if available
                entry_price=entry_price,
                close_price=current_price,
                profit=profit,
                profit_pips=profit_pips,
                balance_before=balance_before,
                balance_after=balance_after,
                duration_seconds=duration_seconds,
                ml_confidence=ml_confidence,
                regime=regime,
                volatility=volatility,
            )
        except Exception as e:
            logger.warning(f"Failed to send trade close notification: {e}")

    async def _send_market_update(self, df, regime_state, ml_prediction):
        """Send periodic market update to Telegram."""
        try:
            now = datetime.now()

            # Only send market update every 30 minutes
            if self._last_market_update_time:
                time_since = (now - self._last_market_update_time).total_seconds()
                if time_since < 1800:  # 30 minutes
                    return

            session_status = self.session_filter.get_status_report()

            # Get ATR and spread
            atr = df["atr"].tail(1).item() if "atr" in df.columns else 0
            tick = self.mt5.get_tick(self.config.symbol)
            spread = (tick.ask - tick.bid) if tick else 0

            # Determine trend direction
            if "ema_9" in df.columns and "ema_21" in df.columns:
                ema_9 = df["ema_9"].tail(1).item()
                ema_21 = df["ema_21"].tail(1).item()
                trend_direction = "UPTREND" if ema_9 > ema_21 else "DOWNTREND"
            else:
                trend_direction = "NEUTRAL"

            await self.telegram.notify_market_update(
                symbol=self.config.symbol,
                price=df["close"].tail(1).item(),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "unknown"),
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                trend_direction=trend_direction,
                session=session_status.get("current_session", "Unknown"),
                can_trade=session_status.get("can_trade", True),
                atr=atr,
                spread=spread,
            )

            self._last_market_update_time = now
            logger.info("Telegram: Market update sent")

        except Exception as e:
            logger.warning(f"Failed to send market update: {e}")

    async def _send_daily_summary(self):
        """Send daily trading summary to Telegram."""
        try:
            balance = self.mt5.account_balance or self.config.capital
            await self.telegram.send_daily_summary(
                start_balance=self._daily_start_balance,
                end_balance=balance,
            )
            logger.info("Telegram: Daily summary sent")
        except Exception as e:
            logger.warning(f"Failed to send daily summary: {e}")

    async def _send_hourly_analysis_if_due(
        self,
        df,
        regime_state,
        ml_prediction,
        open_positions,
        current_price: float,
    ):
        """
        Send comprehensive hourly analysis report to Telegram.
        Interval: Every 1 hour
        """
        now = datetime.now()

        # Check if 1 hour has passed since last report
        if self._last_hourly_report_time:
            time_since = (now - self._last_hourly_report_time).total_seconds()
            if time_since < 3600:  # 1 hour = 3600 seconds
                return

        try:
            # Gather all data for report
            balance = self.mt5.account_balance or self.config.capital
            equity = self.mt5.account_equity or self.config.capital
            floating_pnl = equity - balance

            # Position details with Smart Risk data
            position_details = []
            for row in open_positions.iter_rows(named=True):
                ticket = row["ticket"]
                profit = row.get("profit", 0)
                position_type = row.get("type", 0)
                direction = "BUY" if position_type == 0 else "SELL"

                # Get guard data if available
                guard = self.smart_risk._position_guards.get(ticket)
                momentum = guard.momentum_score if guard else 0
                tp_prob = guard.get_tp_probability() if guard else 50

                position_details.append({
                    "ticket": ticket,
                    "direction": direction,
                    "profit": profit,
                    "momentum": momentum,
                    "tp_probability": tp_prob,
                })

            # Session info
            session_status = self.session_filter.get_status_report()

            # Dynamic confidence data
            market_analysis = self.dynamic_confidence.analyze_market(
                session=session_status.get("current_session", "Unknown"),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "medium"),
                trend_direction=regime_state.regime.value if regime_state else "neutral",
                has_smc_signal=False,
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
            )

            # Risk state
            risk_rec = self.smart_risk.get_trading_recommendation()

            # News Agent status
            news_can_trade, news_reason, _ = self.news_agent.should_trade()
            news_status = "SAFE" if news_can_trade else "BLOCKED"

            # Execution stats
            avg_exec = (sum(self._execution_times) / len(self._execution_times) * 1000) if self._execution_times else 0
            uptime = (now - self._start_time).total_seconds() / 3600  # hours

            # Send the report
            await self.telegram.send_hourly_analysis(
                # Account
                balance=balance,
                equity=equity,
                floating_pnl=floating_pnl,
                # Positions
                open_positions=len(open_positions),
                position_details=position_details,
                # Market
                symbol=self.config.symbol,
                current_price=current_price,
                session=session_status.get("current_session", "Unknown"),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "unknown"),
                # AI/ML
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                dynamic_threshold=market_analysis.confidence_threshold,
                market_quality=market_analysis.quality.value,
                market_score=market_analysis.score,
                # Risk
                daily_pnl=self._total_session_profit,
                daily_trades=self._total_session_trades,
                risk_mode=risk_rec.get("mode", "normal"),
                max_daily_loss=self.smart_risk.max_daily_loss_usd,
                # Bot
                uptime_hours=uptime,
                total_loops=self._loop_count,
                avg_execution_ms=avg_exec,
                # News
                news_status=news_status,
                news_reason=news_reason,
            )

            self._last_hourly_report_time = now
            logger.info("Telegram: Hourly analysis report sent")

        except Exception as e:
            logger.warning(f"Failed to send hourly analysis: {e}")

    def _on_new_day(self):
        """Handle new trading day."""
        logger.info("=" * 60)
        logger.info(f"NEW TRADING DAY: {date.today()}")
        logger.info("=" * 60)

        # Send daily summary before resetting (run synchronously)
        try:
            import asyncio
            asyncio.create_task(self._send_daily_summary())
        except Exception as e:
            logger.warning(f"Could not send daily summary: {e}")

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
                logger.info(f"  Train AUC: {results.get('xgb_train_auc', 0):.4f}")
                logger.info(f"  Test AUC: {results.get('xgb_test_auc', 0):.4f}")

                # Check if new model is worse - rollback if needed
                # FIX: Increased minimum AUC from 0.52 to 0.60 (0.52 is barely better than random)
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


if __name__ == "__main__":
    asyncio.run(main())
