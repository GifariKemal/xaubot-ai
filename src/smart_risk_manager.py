"""
Smart Risk Manager v2.0
========================
Sistem risk management cerdas untuk mencegah kerugian besar.

FILOSOFI: "Slow but Steady - Mental Health First"
- Lot size SANGAT KECIL (0.01-0.03)
- TANPA hard stop loss (menggunakan soft management)
- Hanya close jika trend BENAR-BENAR berbalik
- Recovery mode setelah loss
- Maximum loss per hari dibatasi ketat

Author: AI Assistant
"""

import os
import time
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo
from loguru import logger
import polars as pl

WIB = ZoneInfo("Asia/Jakarta")

# Feature flags
_KALMAN_ENABLED = os.environ.get("KALMAN_ENABLED", "1") == "1"
_ADVANCED_EXITS_ENABLED = os.environ.get("ADVANCED_EXITS_ENABLED", "1") == "1"
_PREDICTIVE_ENABLED = os.environ.get("PREDICTIVE_ENABLED", "1") == "1"  # v6.3 Predictive features


class TradingMode(Enum):
    """Mode trading berdasarkan kondisi."""
    NORMAL = "normal"           # Trading normal dengan lot kecil
    RECOVERY = "recovery"       # Setelah loss, lot lebih kecil lagi
    PROTECTED = "protected"     # Mendekati daily loss limit
    STOPPED = "stopped"         # Stop trading hari ini


class ExitReason(Enum):
    """Alasan untuk exit position."""
    TAKE_PROFIT = "take_profit"
    TREND_REVERSAL = "trend_reversal"      # ML signal berbalik KUAT
    DAILY_LIMIT = "daily_limit"            # Mencapai daily loss limit
    POSITION_LIMIT = "position_limit"      # Mencapai max loss per trade (S/L)
    TOTAL_LIMIT = "total_limit"            # Mencapai total loss limit
    WEEKEND_CLOSE = "weekend_close"        # Menjelang weekend
    MANUAL = "manual"


@dataclass
class RiskState:
    """Current risk state."""
    mode: TradingMode = TradingMode.NORMAL
    daily_profit: float = 0
    daily_loss: float = 0
    daily_trades: int = 0
    consecutive_losses: int = 0
    last_loss_amount: float = 0
    can_trade: bool = True
    reason: str = ""
    recommended_lot: float = 0.01
    max_allowed_lot: float = 0.03


@dataclass
class PositionGuard:
    """Guard untuk setiap position - menentukan kapan harus close."""
    ticket: int
    entry_price: float
    entry_time: datetime
    lot_size: float
    direction: str  # BUY or SELL

    # Soft stops (hanya warning, tidak auto close)
    soft_stop_price: float = 0
    soft_stop_triggered: bool = False

    # Hard protection (hanya close jika ini tercapai)
    max_loss_usd: float = 50.0  # Maximum loss $50 per position

    # Profit tracking
    peak_profit: float = 0
    current_profit: float = 0

    # Exit conditions met
    should_close: bool = False
    close_reason: Optional[ExitReason] = None

    # === SMART DYNAMIC TP TRACKING ===
    # Target tracking
    target_tp_price: float = 0  # Original TP target
    target_tp_profit: float = 0  # Expected profit at TP

    # Momentum tracking (untuk prediksi)
    price_history: List[float] = field(default_factory=list)  # Last N prices
    profit_history: List[float] = field(default_factory=list)  # Last N profits
    ml_confidence_history: List[float] = field(default_factory=list)  # ML confidence trend

    # Smart analysis
    momentum_score: float = 0  # -100 to +100, positive = moving towards TP
    stall_count: int = 0  # Berapa kali harga stall/sideways
    reversal_warnings: int = 0  # Jumlah warning ML reversal
    profit_capture_count: int = 0  # Consecutive intervals with profit >= tp_min + velocity <= 0

    # === VELOCITY & ACCELERATION TRACKING ===
    profit_timestamps: List[float] = field(default_factory=list)  # time.time() per entry
    velocity: float = 0.0             # $/second (profit change rate)
    acceleration: float = 0.0         # $/s² (velocity change rate)
    prev_velocity: float = 0.0       # previous velocity for acceleration calc
    stagnation_seconds: float = 0.0  # how long velocity near zero
    last_significant_move_time: float = 0.0  # last time velocity exceeded threshold
    last_momentum_log_time: float = 0.0  # throttle logging per ticket

    # === RECOVERY TRACKING (v4) ===
    min_profit_seen: float = 0.0     # Lowest profit ever seen for this trade
    recovery_count: int = 0           # How many times trade bounced from loss to profit
    has_recovered: bool = False       # True if trade recovered from significant loss to positive
    was_positive: bool = False        # True if trade was ever meaningfully positive

    # === SMART PROFIT DETECTION (v5b) ===
    peak_update_time: float = 0.0          # time.time() when peak was last updated
    failed_peak_attempts: int = 0          # Times price approached but failed to exceed peak
    velocity_was_positive: bool = False    # Velocity was positive in recent past
    velocity_sign_flips: int = 0           # Consecutive vel positive->negative transitions
    decel_at_profit_count: int = 0         # Consecutive readings with negative accel while in profit
    profit_stall_start_time: float = 0.0   # time.time() when profit stall began
    profit_stall_anchor: float = 0.0       # Profit level when stall started
    rsi_extreme_count: int = 0             # Consecutive readings at RSI/Stoch extreme

    # === KALMAN FILTER (v6) ===
    kalman: object = None                  # ProfitKalmanFilter instance (lazy init)
    kalman_velocity: float = 0.0           # Kalman-filtered velocity ($/s)
    kalman_acceleration: float = 0.0       # Kalman-filtered acceleration ($/s^2)

    # === ADVANCED EXIT SYSTEMS (v7) ===
    ekf: object = None                     # ExtendedKalmanFilter instance (lazy init)
    ekf_velocity: float = 0.0              # EKF velocity (3D state)
    ekf_acceleration: float = 0.0          # EKF acceleration (3D state)
    pid_controller: object = None          # PIDExitController instance

    # === v6.3 PREDICTIVE INTELLIGENCE ===
    velocity_history: List[float] = field(default_factory=list)      # Historical velocity values
    acceleration_history: List[float] = field(default_factory=list)  # Historical acceleration values
    peak_loss: float = 0.0  # Most negative profit ever reached (for recovery detection)
    last_profit_for_derivative: float = 0.0  # For velocity derivative calculation
    peak_hold_active: bool = False  # v0.2.2: Suppress exits when approaching peak

    def update_history(self, price: float, profit: float, ml_confidence: float, max_history: int = 20):
        """Update price/profit history untuk analisis momentum."""
        now = time.time()
        self.price_history.append(price)
        self.profit_history.append(profit)
        self.ml_confidence_history.append(ml_confidence)
        self.profit_timestamps.append(now)

        # Keep only last N entries
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.profit_history = self.profit_history[-max_history:]
            self.ml_confidence_history = self.ml_confidence_history[-max_history:]
            self.profit_timestamps = self.profit_timestamps[-max_history:]

        # Update velocity, acceleration, and stagnation
        self._calculate_velocity_acceleration()
        self._update_stagnation(now)

        # === Kalman filter update (v6) ===
        # V7: Use EKF if advanced exits enabled, otherwise use basic Kalman
        if _KALMAN_ENABLED:
            # Use basic 2D Kalman filter
            if self.kalman is None:
                try:
                    from src.kalman_filter import ProfitKalmanFilter
                    self.kalman = ProfitKalmanFilter()
                except ImportError:
                    pass
            if self.kalman is not None:
                _, self.kalman_velocity, self.kalman_acceleration = (
                    self.kalman.update(profit, now)
                )

        # === v6.3 PREDICTIVE: Track velocity/acceleration history ===
        if _PREDICTIVE_ENABLED:
            self.velocity_history.append(self.velocity)
            self.acceleration_history.append(self.acceleration)
            # Keep last N samples only
            if len(self.velocity_history) > max_history:
                self.velocity_history = self.velocity_history[-max_history:]
            if len(self.acceleration_history) > max_history:
                self.acceleration_history = self.acceleration_history[-max_history:]

        # === Recovery tracking (v4) ===
        if profit < self.min_profit_seen:
            self.min_profit_seen = profit
        # v6.3: Track peak loss for recovery detection
        if profit < self.peak_loss:
            self.peak_loss = profit
        if profit >= 1.0:
            self.was_positive = True
        # Detect recovery: trade was at significant loss (<-$2) and now positive
        if self.min_profit_seen < -2.0 and profit > 0 and not self.has_recovered:
            self.has_recovered = True
            self.recovery_count += 1

        # === Smart Profit Detection tracking (v5b) ===
        # Track peak freshness
        if profit >= self.peak_profit and profit > 0:
            self.peak_update_time = now
            self.failed_peak_attempts = 0  # Reset: new peak achieved
        elif profit > 0 and self.peak_profit > 0 and profit >= self.peak_profit * 0.85:
            # Approached peak (within 85%) but didn't break it
            self.failed_peak_attempts += 1

        # Track velocity sign transitions (positive -> negative)
        if self.velocity < -0.01 and self.velocity_was_positive:
            self.velocity_sign_flips += 1
        elif self.velocity > 0.01:
            self.velocity_was_positive = True
            self.velocity_sign_flips = 0  # Reset: back to positive

        # Track deceleration while in profit
        if profit > 0 and self.acceleration < -0.001 and self.velocity < self.prev_velocity:
            self.decel_at_profit_count += 1
        elif profit <= 0 or self.acceleration >= 0:
            self.decel_at_profit_count = 0

        # Track profit stall (profit in narrow range)
        if profit > 0 and len(self.profit_history) >= 3:
            recent_3 = self.profit_history[-3:]
            stall_range = max(recent_3) - min(recent_3)
            if stall_range < 1.0:  # Profit barely moving ($1 range)
                if self.profit_stall_start_time == 0:
                    self.profit_stall_start_time = now
                    self.profit_stall_anchor = profit
            else:
                self.profit_stall_start_time = 0  # Reset: profit is moving

    def calculate_momentum(self) -> float:
        """
        Hitung momentum score -100 to +100.
        Positive = bergerak ke arah TP (bagus)
        Negative = bergerak menjauhi TP (bahaya)
        """
        if len(self.profit_history) < 3:
            return 0

        # Recent profit change
        recent = self.profit_history[-5:] if len(self.profit_history) >= 5 else self.profit_history
        profit_change = recent[-1] - recent[0]

        # Normalize: $10 change = 50 points
        momentum = (profit_change / 10) * 50
        momentum = max(-100, min(100, momentum))

        self.momentum_score = momentum
        return momentum

    def get_tp_probability(self) -> float:
        """
        Estimasi probabilitas mencapai TP (0-100%).

        Faktor:
        1. Jarak ke TP vs jarak sudah ditempuh
        2. Momentum saat ini
        3. ML confidence trend
        4. Waktu sudah berjalan
        """
        if self.target_tp_profit <= 0:
            return 50  # Unknown TP

        # Factor 1: Progress to TP (0-40 points)
        progress = (self.current_profit / self.target_tp_profit) * 100 if self.target_tp_profit > 0 else 0
        progress_score = min(40, max(0, progress * 0.4))

        # Factor 2: Momentum (0-30 points)
        momentum = self.calculate_momentum()
        momentum_score = ((momentum + 100) / 200) * 30  # Convert -100..100 to 0..30

        # Factor 3: ML confidence trend (0-20 points)
        if len(self.ml_confidence_history) >= 3:
            recent_conf = self.ml_confidence_history[-3:]
            conf_trend = recent_conf[-1] - recent_conf[0]
            conf_score = ((conf_trend + 0.3) / 0.6) * 20  # -0.3 to +0.3 -> 0 to 20
            conf_score = max(0, min(20, conf_score))
        else:
            conf_score = 10

        # Factor 4: Time penalty (0-10 points lost)
        time_elapsed = (datetime.now(WIB) - self.entry_time).total_seconds() / 3600  # hours
        time_penalty = min(10, time_elapsed * 2)  # Lose 2 points per hour

        probability = progress_score + momentum_score + conf_score - time_penalty
        return max(0, min(100, probability))

    def _calculate_velocity_acceleration(self):
        """Calculate velocity ($/s) from last 5 samples and acceleration ($/s²) from split-half."""
        if len(self.profit_timestamps) < 2:
            return

        # Velocity from last 5 samples (or all if < 5)
        n = min(5, len(self.profit_timestamps))
        recent_times = self.profit_timestamps[-n:]
        recent_profits = self.profit_history[-n:]
        dt = recent_times[-1] - recent_times[0]
        if dt > 0:
            self.prev_velocity = self.velocity
            self.velocity = (recent_profits[-1] - recent_profits[0]) / dt
        else:
            self.velocity = 0.0

        # Acceleration from split-half comparison (need >= 6 samples)
        if len(self.profit_timestamps) >= 6:
            mid = len(self.profit_timestamps) // 2

            t1 = self.profit_timestamps[:mid]
            p1 = self.profit_history[:mid]
            dt1 = t1[-1] - t1[0]
            v1 = (p1[-1] - p1[0]) / dt1 if dt1 > 0 else 0.0

            t2 = self.profit_timestamps[mid:]
            p2 = self.profit_history[mid:]
            dt2 = t2[-1] - t2[0]
            v2 = (p2[-1] - p2[0]) / dt2 if dt2 > 0 else 0.0

            dt_total = self.profit_timestamps[-1] - self.profit_timestamps[0]
            self.acceleration = (v2 - v1) / dt_total if dt_total > 0 else 0.0

    def _update_stagnation(self, now: float):
        """Track how long velocity stays near zero (< 0.05 $/s)."""
        if abs(self.velocity) < 0.05:
            # Stagnating — accumulate time since last update
            if len(self.profit_timestamps) >= 2:
                dt = self.profit_timestamps[-1] - self.profit_timestamps[-2]
                self.stagnation_seconds += dt
        else:
            # Moving — reset stagnation and record significant move
            self.stagnation_seconds = 0.0
            self.last_significant_move_time = now

    def get_velocity_summary(self) -> Dict:
        """Return dict with velocity metrics for logging."""
        return {
            "velocity": round(self.velocity, 4),
            "acceleration": round(self.acceleration, 4),
            "stagnation_s": round(self.stagnation_seconds, 1),
            "samples": len(self.profit_timestamps),
        }


class SmartRiskManager:
    """
    Smart Risk Manager - Sistem manajemen risiko cerdas.

    PRINSIP UTAMA:
    1. Lot size SANGAT KECIL (0.01-0.03 max)
    2. TIDAK menggunakan hard stop loss
    3. Hanya close jika trend BENAR-BENAR berbalik (ML confidence tinggi)
    4. Maximum loss per hari: 5% of capital
    5. Maximum total loss: 10% of capital (stop trading)
    6. S/L 1% per trade
    7. Recovery mode setelah loss besar
    """

    def __init__(
        self,
        capital: float = 5000.0,
        max_daily_loss_percent: float = 5.0,      # Max 5% daily loss
        max_total_loss_percent: float = 10.0,     # Max 10% total loss (stop trading)
        max_loss_per_trade_percent: float = 0.5,  # Max 0.5% per trade (FIX 5: was 1.0%, ~$25 for $5k capital)
        emergency_sl_percent: float = 2.0,        # Emergency broker S/L 2% per trade
        base_lot_size: float = 0.01,              # Lot dasar sangat kecil
        max_lot_size: float = 0.03,               # Maximum lot
        recovery_lot_size: float = 0.01,          # Lot saat recovery
        trend_reversal_threshold: float = 0.75,   # ML confidence untuk close
        max_concurrent_positions: int = 2,        # Max posisi bersamaan
    ):
        self.capital = capital
        self.max_daily_loss_percent = max_daily_loss_percent
        self.max_daily_loss_usd = capital * (max_daily_loss_percent / 100)
        self.max_total_loss_percent = max_total_loss_percent
        self.max_total_loss_usd = capital * (max_total_loss_percent / 100)
        self.max_loss_per_trade_percent = max_loss_per_trade_percent
        self.max_loss_per_trade = capital * (max_loss_per_trade_percent / 100)  # Software S/L in USD
        self.emergency_sl_percent = emergency_sl_percent
        self.emergency_sl_usd = capital * (emergency_sl_percent / 100)  # Broker S/L in USD
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.recovery_lot_size = recovery_lot_size
        self.trend_reversal_threshold = trend_reversal_threshold
        self.max_concurrent_positions = max_concurrent_positions

        # Total loss tracking (across all days)
        self._total_loss: float = 0.0

        # State tracking
        self._state = RiskState()
        self._position_guards: Dict[int, PositionGuard] = {}
        self._daily_pnl: List[float] = []
        self._current_date = date.today()

        # Load state
        self._load_daily_state()

        # === Advanced Exit Systems (v7) ===
        self._init_advanced_exit_systems()

        logger.info("=" * 50)
        # Get version from centralized version manager
        try:
            from src.version import get_version, __exit_strategy__
            version_str = f"v{get_version()} ({__exit_strategy__})"
        except ImportError:
            if _PREDICTIVE_ENABLED and _ADVANCED_EXITS_ENABLED:
                version_str = "v0.6.0 (Exit v6.3 Predictive Intelligence)"
            elif _ADVANCED_EXITS_ENABLED:
                version_str = "v0.3.0 (Exit v6.2 Advanced)"
            else:
                version_str = "v0.1.0 (Exit v6.0 Kalman)"
        logger.info(f"SMART RISK MANAGER {version_str} INITIALIZED")
        logger.info(f"  Capital: ${capital:,.2f}")
        logger.info(f"  Max Daily Loss: {max_daily_loss_percent}% (${self.max_daily_loss_usd:.2f})")
        logger.info(f"  Max Total Loss: {max_total_loss_percent}% (${self.max_total_loss_usd:.2f})")
        logger.info(f"  Software S/L: {max_loss_per_trade_percent}% (${self.max_loss_per_trade:.2f})")
        logger.info(f"  Emergency Broker S/L: {emergency_sl_percent}% (${self.emergency_sl_usd:.2f})")
        logger.info(f"  Max Positions: {max_concurrent_positions}")
        logger.info(f"  Base Lot: {base_lot_size}")
        logger.info(f"  Max Lot: {max_lot_size}")
        logger.info("  Mode: SMART S/L (software + broker safety net)")
        if _ADVANCED_EXITS_ENABLED:
            if _PREDICTIVE_ENABLED:
                logger.info("  Advanced Exits: ENABLED (Kalman + Fuzzy + Kelly + Predictive)")
            else:
                logger.info("  Advanced Exits: ENABLED (Kalman + Fuzzy + Kelly)")
        logger.info("=" * 50)

    def _init_advanced_exit_systems(self):
        """Initialize advanced exit systems (v7) - Kalman + Fuzzy + Kelly + Predictive (v6.3)."""
        if not _ADVANCED_EXITS_ENABLED:
            self.fuzzy_controller = None
            self.kelly_scaler = None
            self.trajectory_predictor = None
            self.momentum_persistence = None
            self.recovery_detector = None
            return

        try:
            # Fuzzy Logic Controller
            from src.fuzzy_exit_logic import FuzzyExitController
            self.fuzzy_controller = FuzzyExitController()
            logger.info("  [OK] Fuzzy Exit Controller initialized")
        except Exception as e:
            logger.warning(f"Could not initialize FuzzyExitController: {e}")
            self.fuzzy_controller = None

        try:
            # Kelly Position Scaler
            from src.kelly_position_scaler import KellyPositionScaler
            self.kelly_scaler = KellyPositionScaler(
                base_win_rate=0.55,
                avg_win=8.0,
                avg_loss=4.0,
                kelly_fraction=0.5,
            )
            logger.info("  [OK] Kelly Position Scaler initialized")
        except Exception as e:
            logger.warning(f"Could not initialize KellyPositionScaler: {e}")
            self.kelly_scaler = None

        # === v6.3 PREDICTIVE INTELLIGENCE ===
        if _PREDICTIVE_ENABLED:
            try:
                # Trajectory Predictor - Forecast profit 1-5 minutes ahead
                from src.trajectory_predictor import TrajectoryPredictor
                self.trajectory_predictor = TrajectoryPredictor()
                logger.info("  [OK] Trajectory Predictor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize TrajectoryPredictor: {e}")
                self.trajectory_predictor = None

            try:
                # Momentum Persistence - Detect if momentum will continue
                from src.momentum_persistence import MomentumPersistence
                self.momentum_persistence = MomentumPersistence(lookback_periods=5)
                logger.info("  [OK] Momentum Persistence initialized")
            except Exception as e:
                logger.warning(f"Could not initialize MomentumPersistence: {e}")
                self.momentum_persistence = None

            try:
                # Recovery Detector - Analyze recovery strength from losses
                from src.recovery_detector import RecoveryDetector
                self.recovery_detector = RecoveryDetector()
                logger.info("  [OK] Recovery Detector initialized")
            except Exception as e:
                logger.warning(f"Could not initialize RecoveryDetector: {e}")
                self.recovery_detector = None
        else:
            self.trajectory_predictor = None
            self.momentum_persistence = None
            self.recovery_detector = None

    def _load_daily_state(self):
        """Load daily state from file."""
        state_file = "data/risk_state.txt"
        backup_file = "data/risk_state.bak"

        def load_from_file(filepath):
            """Load state from a specific file."""
            with open(filepath, "r") as f:
                lines = f.readlines()
                saved_date = None
                for line in lines:
                    if line.startswith("date:"):
                        saved_date = line.split(":")[1].strip()
                    # Always load total_loss (persists across days)
                    if line.startswith("total_loss:"):
                        self._total_loss = float(line.split(":")[1].strip())
                        logger.info(f"Loaded total loss: ${self._total_loss:.2f}")

                if saved_date == str(date.today()):
                    # Load today's state
                    for l in lines:
                        if l.startswith("daily_loss:"):
                            self._state.daily_loss = float(l.split(":")[1].strip())
                        elif l.startswith("daily_profit:"):
                            self._state.daily_profit = float(l.split(":")[1].strip())
                        elif l.startswith("consecutive_losses:"):
                            self._state.consecutive_losses = int(l.split(":")[1].strip())
                    logger.info(f"Loaded today's state: loss=${self._state.daily_loss:.2f}, profit=${self._state.daily_profit:.2f}")
                return True

        try:
            # Try main state file first
            if os.path.exists(state_file):
                load_from_file(state_file)
            # If main file missing/corrupt, try backup
            elif os.path.exists(backup_file):
                logger.warning("Main state file missing, loading from backup...")
                load_from_file(backup_file)
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
            # Try backup if main file failed
            try:
                if os.path.exists(backup_file):
                    load_from_file(backup_file)
            except:
                logger.error("Could not load risk state from backup either")

    def _save_daily_state(self):
        """Save daily state to file with atomic write (crash-safe)."""
        os.makedirs("data", exist_ok=True)
        state_file = "data/risk_state.txt"
        temp_file = "data/risk_state.tmp"
        backup_file = "data/risk_state.bak"

        try:
            # Write to temp file first (atomic write pattern)
            content = (
                f"date:{date.today()}\n"
                f"daily_loss:{self._state.daily_loss}\n"
                f"daily_profit:{self._state.daily_profit}\n"
                f"consecutive_losses:{self._state.consecutive_losses}\n"
                f"total_loss:{self._total_loss}\n"
                f"saved_at:{datetime.now(WIB).isoformat()}\n"
            )

            with open(temp_file, "w") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Backup existing file
            if os.path.exists(state_file):
                try:
                    import shutil
                    shutil.copy2(state_file, backup_file)
                except:
                    pass

            # Atomic rename (crash-safe)
            os.replace(temp_file, state_file)

        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")
            # Try to restore from backup if main file corrupted
            if os.path.exists(backup_file) and not os.path.exists(state_file):
                try:
                    import shutil
                    shutil.copy2(backup_file, state_file)
                except:
                    pass

    def check_new_day(self):
        """Check if it's a new day and reset state."""
        if date.today() != self._current_date:
            logger.info("=" * 40)
            logger.info(f"NEW DAY - Resetting risk state")
            logger.info(f"Yesterday P/L: ${self._state.daily_profit - self._state.daily_loss:.2f}")
            logger.info("=" * 40)

            self._current_date = date.today()
            self._state = RiskState()
            self._state.mode = TradingMode.NORMAL
            self._daily_pnl = []

    def update_capital(self, new_capital: float):
        """Update capital and recalculate ALL limits."""
        self.capital = new_capital
        self.max_daily_loss_usd = new_capital * (self.max_daily_loss_percent / 100)
        self.max_total_loss_usd = new_capital * (self.max_total_loss_percent / 100)
        self.max_loss_per_trade = new_capital * (self.max_loss_per_trade_percent / 100)
        self.emergency_sl_usd = new_capital * (self.emergency_sl_percent / 100)
        logger.info(f"Capital updated: ${new_capital:.2f}")
        logger.info(f"  Daily loss limit: {self.max_daily_loss_percent}% = ${self.max_daily_loss_usd:.2f}")
        logger.info(f"  Total loss limit: {self.max_total_loss_percent}% = ${self.max_total_loss_usd:.2f}")
        logger.info(f"  Software S/L: {self.max_loss_per_trade_percent}% = ${self.max_loss_per_trade:.2f}")
        logger.info(f"  Emergency Broker S/L: {self.emergency_sl_percent}% = ${self.emergency_sl_usd:.2f}")

    def calculate_emergency_sl(
        self,
        entry_price: float,
        direction: str,
        lot_size: float,
        symbol: str = "XAUUSD",
    ) -> float:
        """
        Calculate emergency stop loss price (broker level).

        This is the LAST LINE OF DEFENSE if software fails.
        Set at 2% of capital (~$100) as max loss per trade.

        Args:
            entry_price: Entry price of the trade
            direction: "BUY" or "SELL"
            lot_size: Position size
            symbol: Trading symbol

        Returns:
            Emergency SL price
        """
        # For XAUUSD: 1 lot = $1 per 0.01 price movement (1 pip = $0.10 for 0.01 lot)
        # pip_value = lot_size * 10 (for XAUUSD)
        pip_value = lot_size * 10  # $1 per pip for 0.1 lot, $0.10 per pip for 0.01 lot

        # Calculate how many pips = emergency_sl_usd
        if pip_value > 0:
            emergency_pips = self.emergency_sl_usd / pip_value
        else:
            emergency_pips = 1000  # Default fallback

        # Convert pips to price movement (XAUUSD: 1 pip = 0.01)
        price_distance = emergency_pips * 0.01

        if direction.upper() == "BUY":
            sl_price = entry_price - price_distance
        else:
            sl_price = entry_price + price_distance

        logger.info(f"Emergency SL calculated: {sl_price:.2f} (${self.emergency_sl_usd:.2f} max loss)")
        return round(sl_price, 2)

    def can_open_position(self) -> Tuple[bool, str]:
        """
        Check if we can open a new position.

        Returns:
            (can_open, reason)
        """
        self._update_state()

        # Check if trading is allowed
        if not self._state.can_trade:
            return False, f"Trading stopped: {self._state.reason}"

        # Check max concurrent positions
        active_positions = len(self._position_guards)
        if active_positions >= self.max_concurrent_positions:
            return False, f"Max positions reached ({active_positions}/{self.max_concurrent_positions})"

        return True, f"Can open ({active_positions}/{self.max_concurrent_positions} positions)"

    def get_state(self) -> RiskState:
        """Get current risk state."""
        self._update_state()
        return self._state

    def _update_state(self):
        """Update risk state based on daily and total performance."""
        net_pnl = self._state.daily_profit - self._state.daily_loss

        # Check TOTAL loss limit (10%) - highest priority
        if self._total_loss >= self.max_total_loss_usd:
            self._state.mode = TradingMode.STOPPED
            self._state.can_trade = False
            self._state.reason = f"TOTAL LOSS LIMIT reached ({self.max_total_loss_percent}% = ${self._total_loss:.2f}) - TRADING STOPPED"
            return

        # Check daily loss limit (5%)
        if self._state.daily_loss >= self.max_daily_loss_usd:
            self._state.mode = TradingMode.STOPPED
            self._state.can_trade = False
            self._state.reason = f"Daily loss limit reached ({self.max_daily_loss_percent}% = ${self._state.daily_loss:.2f})"
            return

        # Check if approaching TOTAL limit (80%)
        if self._total_loss >= self.max_total_loss_usd * 0.8:
            self._state.mode = TradingMode.PROTECTED
            self._state.recommended_lot = self.recovery_lot_size
            self._state.max_allowed_lot = self.recovery_lot_size
            self._state.reason = f"Approaching TOTAL loss limit ({self._total_loss:.2f}/${self.max_total_loss_usd:.2f}) - protected mode"
            self._state.can_trade = True
            return

        # Check if approaching daily limit (80%)
        if self._state.daily_loss >= self.max_daily_loss_usd * 0.8:
            self._state.mode = TradingMode.PROTECTED
            self._state.recommended_lot = self.recovery_lot_size
            self._state.max_allowed_lot = self.recovery_lot_size
            self._state.reason = "Approaching daily loss limit - protected mode"
            self._state.can_trade = True
            return

        # Check consecutive losses
        if self._state.consecutive_losses >= 3:
            self._state.mode = TradingMode.RECOVERY
            self._state.recommended_lot = self.recovery_lot_size
            self._state.max_allowed_lot = self.base_lot_size
            self._state.reason = f"{self._state.consecutive_losses} consecutive losses - recovery mode"
            self._state.can_trade = True
            return

        # Normal mode
        self._state.mode = TradingMode.NORMAL
        self._state.recommended_lot = self.base_lot_size
        self._state.max_allowed_lot = self.max_lot_size
        self._state.can_trade = True
        self._state.reason = "Normal trading mode"

    def calculate_lot_size(
        self,
        entry_price: float,
        confidence: float = 0.5,
        regime: str = "normal",
        ml_confidence: float = 0.5,  # NEW: ML-specific confidence
    ) -> float:
        """
        Calculate safe lot size with ML confidence adjustment.

        PRINSIP: Lot size SANGAT KECIL
        - Base: 0.01
        - Max: 0.02 (reduced from 0.03)

        IMPROVEMENT 3: ML Confidence-based sizing
        - ML 50-55%: 0.01 lot (minimum) - uncertain
        - ML 55-65%: 0.01 lot (base)
        - ML >65%: 0.02 lot (max) - high confidence
        """
        self._update_state()

        if not self._state.can_trade:
            return 0

        # Start with base lot
        lot = self.base_lot_size

        # Adjust based on mode
        if self._state.mode == TradingMode.RECOVERY:
            lot = self.recovery_lot_size
        elif self._state.mode == TradingMode.PROTECTED:
            lot = self.recovery_lot_size

        # === IMPROVEMENT 3: ML Confidence-based lot sizing ===
        # Use the more conservative of confidence or ml_confidence
        effective_confidence = min(confidence, ml_confidence)

        if effective_confidence >= 0.65:
            # High confidence: allow max lot
            lot = self.max_lot_size
            confidence_tier = "HIGH"
        elif effective_confidence >= 0.55:
            # Medium confidence: base lot
            lot = self.base_lot_size
            confidence_tier = "MEDIUM"
        else:
            # Low confidence: minimum lot
            lot = self.recovery_lot_size
            confidence_tier = "LOW"

        # Adjust based on regime (override if risky)
        if regime.lower() in ["high_volatility", "crisis"]:
            lot = self.recovery_lot_size
            confidence_tier = "VOLATILE"

        # Cap at maximum
        lot = min(lot, self._state.max_allowed_lot)

        # Round to 0.01
        lot = round(lot, 2)

        logger.info(f"Calculated lot: {lot} (mode={self._state.mode.value}, ML={ml_confidence:.0%}, tier={confidence_tier})")

        return lot

    def register_position(
        self,
        ticket: int,
        entry_price: float,
        lot_size: float,
        direction: str,
    ) -> PositionGuard:
        """
        Register a new position for monitoring.

        TIDAK menggunakan hard stop loss.
        Menggunakan soft management berdasarkan:
        - Maximum loss per position ($30-50)
        - Trend reversal (ML confidence tinggi berlawanan)
        """
        guard = PositionGuard(
            ticket=ticket,
            entry_price=entry_price,
            entry_time=datetime.now(WIB),
            lot_size=lot_size,
            direction=direction,
            max_loss_usd=self.max_loss_per_trade,
        )

        self._position_guards[ticket] = guard
        logger.info(f"Position #{ticket} registered - NO HARD SL, max loss ${self.max_loss_per_trade}")

        return guard

    def auto_register_existing_position(
        self,
        ticket: int,
        entry_price: float,
        lot_size: float,
        direction: str,
        current_profit: float = 0,
    ) -> PositionGuard:
        """
        Auto-register posisi yang sudah ada (dari sebelum bot start).

        Penting untuk memastikan SEMUA posisi terlindungi oleh:
        - Max loss $50 per trade
        - ML reversal detection
        - Daily loss tracking
        """
        # Skip jika sudah registered
        if ticket in self._position_guards:
            return self._position_guards[ticket]

        guard = PositionGuard(
            ticket=ticket,
            entry_price=entry_price,
            entry_time=datetime.now(WIB),  # Approximate, tidak tahu exact time
            lot_size=lot_size,
            direction=direction,
            max_loss_usd=self.max_loss_per_trade,
            current_profit=current_profit,
            peak_profit=max(0, current_profit),  # Track peak dari sekarang
        )

        self._position_guards[ticket] = guard
        logger.info(f"Position #{ticket} AUTO-REGISTERED (existing) - Protected with max loss ${self.max_loss_per_trade}")

        return guard

    def is_position_registered(self, ticket: int) -> bool:
        """Check if position is registered."""
        return ticket in self._position_guards

    # Baseline ATR for XAUUSD M15 (long-term average, updated periodically)
    _BASELINE_ATR: float = 18.0  # Conservative default

    def _classify_trade_state(self, guard) -> str:
        """
        Classify the trade's velocity pattern into a state.
        Used for dynamic threshold adjustments.
        """
        vel = guard.velocity
        accel = guard.acceleration
        if vel > 0.05 and accel > 0:
            return "accelerating_profit"   # Best case: profit growing faster
        elif vel > 0.02:
            return "steady_profit"         # Profit still growing
        elif abs(vel) <= 0.02:
            return "stalling"              # Not moving much
        elif vel < -0.05 and accel < -0.001:
            return "crashing"              # Fast loss, getting worse
        elif vel < -0.02:
            return "declining"             # Losing but may stabilize
        return "neutral"

    def _calculate_dynamic_multipliers(
        self, guard, regime: str, ml_signal: str, ml_confidence: float,
        market_context: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        """
        Calculate dynamic multipliers for profit targets and loss tolerance.

        Returns: (profit_mult, loss_mult)
        - profit_mult > 1 = let profits run further
        - loss_mult > 1 = give more room before cutting
        """
        profit_mult = 1.0
        loss_mult = 1.0

        # === 1. REGIME ADJUSTMENT ===
        if regime == "trending":
            profit_mult *= 1.5   # Trending: big moves expected, let profit run
            loss_mult *= 0.7     # Trending: if against us, cut faster
        elif regime in ("ranging", "mean_reverting"):
            profit_mult *= 0.6   # Ranging: take what you can, price will bounce
            loss_mult *= 1.3     # Ranging: give room, will likely bounce back
        elif regime in ("high_volatility", "volatile", "crisis"):
            profit_mult *= 1.3   # Volatile: big moves possible
            loss_mult *= 1.5     # Volatile: swings are normal, give room

        # === 2. ML AGREEMENT ===
        ml_agrees = (
            (guard.direction == "BUY" and ml_signal == "BUY") or
            (guard.direction == "SELL" and ml_signal == "SELL")
        )
        ml_disagrees = (
            (guard.direction == "BUY" and ml_signal == "SELL") or
            (guard.direction == "SELL" and ml_signal == "BUY")
        )
        if ml_agrees and ml_confidence >= 0.60:
            conf_bonus = min(0.3, (ml_confidence - 0.60) * 1.5)  # 0-0.3 bonus
            profit_mult *= (1.2 + conf_bonus)  # ML agrees: let it run
            loss_mult *= (1.2 + conf_bonus)    # ML agrees: give room
        elif ml_disagrees and ml_confidence >= 0.65:
            conf_penalty = min(0.3, (ml_confidence - 0.65) * 1.5)
            profit_mult *= (0.7 - conf_penalty)      # ML disagrees: take profit sooner
            loss_mult *= (1.0 + conf_penalty * 0.5)  # v6 FIX: WIDEN loss tolerance (ML 56% accuracy)

        # === 3. VELOCITY PATTERN ===
        trade_state = self._classify_trade_state(guard)
        if trade_state == "accelerating_profit":
            profit_mult *= 1.3   # Momentum strong: let it run
        elif trade_state == "crashing":
            profit_mult *= 0.5   # Crashing: take any profit you can
            loss_mult *= 0.7     # Crashing: cut losses faster
        elif trade_state == "declining":
            profit_mult *= 0.8
            loss_mult *= 0.9

        # === 4. RECOVERY BONUS ===
        if guard.has_recovered:
            loss_mult *= 1.5     # Trade proved it can bounce back

        # === 5. MARKET CONTEXT (RSI, ADX, Stochastic) ===
        if market_context:
            rsi = market_context.get("rsi", 50)
            adx = market_context.get("adx", 25)
            stoch_k = market_context.get("stoch_k", 50)

            # ADX: trend strength
            if adx > 30:
                profit_mult *= 1.2  # Strong trend: let profits run
                loss_mult *= 1.1    # Strong trend: slightly more room
            elif adx < 15:
                profit_mult *= 0.7  # No trend: take profits sooner
                loss_mult *= 1.2    # No trend: ranging = give room

            # RSI extremes: reversal likely
            if guard.direction == "BUY" and rsi > 75:
                profit_mult *= 0.7   # Overbought: take profits for BUY
            elif guard.direction == "SELL" and rsi < 25:
                profit_mult *= 0.7   # Oversold: take profits for SELL
            elif guard.direction == "BUY" and rsi < 30:
                loss_mult *= 1.3    # Oversold: BUY should recover
            elif guard.direction == "SELL" and rsi > 70:
                loss_mult *= 1.3    # Overbought: SELL should recover

            # Stochastic extreme crossover
            if guard.direction == "SELL" and stoch_k < 20:
                profit_mult *= 0.8  # Oversold: SELL may reverse
            elif guard.direction == "BUY" and stoch_k > 80:
                profit_mult *= 0.8  # Overbought: BUY may reverse

        # Clamp multipliers to reasonable ranges
        # v5c: loss_mult minimum raised 0.3->0.5 (give trades more breathing room)
        profit_mult = max(0.3, min(2.5, profit_mult))
        loss_mult = max(0.5, min(2.5, loss_mult))

        return profit_mult, loss_mult

    def _calculate_fuzzy_exit_threshold(self, current_profit: float) -> float:
        """
        FIX 1 (v0.1.1): Tiered fuzzy exit thresholds based on profit magnitude.

        BEFORE: Fixed 90% threshold for all profits
        AFTER: Dynamic thresholds:
          - Micro (<$1): 70% -> exit early
          - Small ($1-$3): 75% -> protection
          - Medium ($3-$8): 85% -> hold longer
          - Large (>$8): 90% -> maximize

        Returns threshold value 0.0-1.0
        """
        if current_profit < 1.0:
            return 0.70  # Micro: exit early
        elif current_profit < 3.0:
            return 0.75  # Small: protect
        elif current_profit < 8.0:
            return 0.85  # Medium: hold
        else:
            return 0.90  # Large: maximize

    def _predict_trajectory_calibrated(
        self,
        current_profit: float,
        velocity: float,
        acceleration: float,
        regime: str,
        horizon_seconds: int = 60
    ) -> float:
        """
        FIX 2 (v0.1.1): Calibrated trajectory prediction with regime penalty + uncertainty.

        BEFORE: Optimistic parabolic prediction (95% error rate)
        AFTER: Conservative with:
          - Regime penalty (ranging 0.4x, volatile 0.6x, trending 0.9x)
          - Uncertainty bounds (95% CI lower bound)

        Returns predicted profit in dollars
        """
        # Parabolic motion: p(t) = p₀ + v*t + 0.5*a*t²
        raw_prediction = current_profit + velocity * horizon_seconds + 0.5 * acceleration * (horizon_seconds ** 2)

        # Apply regime penalty
        regime_penalties = {
            "ranging": 0.4,
            "mean_reverting": 0.4,
            "volatile": 0.6,
            "high_volatility": 0.6,
            "crisis": 0.5,
            "trending": 0.9,
            "normal": 0.6,
        }
        penalty = regime_penalties.get(regime, 0.6)
        calibrated_prediction = raw_prediction * penalty

        # Add uncertainty (95% confidence interval lower bound)
        # Uncertainty grows with acceleration magnitude
        prediction_std = abs(acceleration) * horizon_seconds * 5
        conservative_prediction = calibrated_prediction - 1.96 * prediction_std

        # Floor at current profit (can't predict below current)
        return max(current_profit, conservative_prediction)

    def evaluate_position(
        self,
        ticket: int,
        current_price: float,
        current_profit: float,
        ml_signal: str,
        ml_confidence: float,
        regime: str = "normal",
        current_atr: float = 0,
        baseline_atr: float = 0,
        market_context: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[ExitReason], str]:
        """
        SMART DYNAMIC TP v6.4 - Evaluate if position should be closed (Professor AI Validated Fixes).

        Uses ATR-based dynamic scaling + regime/ML/velocity multipliers:
          - current_atr: ATR(14) in price points from latest M15 data
          - baseline_atr: 24h average ATR for normalization
          - All dollar thresholds scale with atr_ratio AND dynamic multipliers
          - market_context: dict with rsi, stoch_k, adx, macd_hist for smart exits
          - Low ATR (quiet market) = tighter exits, smaller losses
          - High ATR (volatile market) = wider thresholds

        Returns: (should_close, reason, message)
        """
        guard = self._position_guards.get(ticket)
        if not guard:
            return False, None, "Position not registered"

        # === ATR-BASED DYNAMIC SCALING ===
        # ATR ratio: data-driven volatility multiplier (replaces fixed session multiplier)
        base = baseline_atr if baseline_atr > 0 else self._BASELINE_ATR
        if current_atr > 0:
            sm = max(0.3, min(current_atr / base, 1.5))  # Clamp 0.3-1.5
        else:
            sm = 1.0  # Fallback: no scaling if ATR unavailable

        # ATR in dollars for this position (XAUUSD: 1 point = $1 per 0.01 lot)
        atr_dollars = current_atr * guard.lot_size * 100 if current_atr > 0 else 0

        effective_max_loss = self.max_loss_per_trade * sm

        # === ATR-BASED THRESHOLDS — "Detak Jantung Market" ===
        # All thresholds use ATR as the base unit, making them SYMMETRIC and adaptive:
        # - London (high vol) -> wider stops, bigger targets
        # - Sydney (low vol) -> tighter stops, smaller targets
        # - Big lot -> wider in dollars, same in ATR terms
        # atr_unit = how many $ of P/L per 1 ATR move for THIS position
        atr_unit = atr_dollars if atr_dollars > 0 else 10 * sm  # Fallback if ATR unavailable

        # === EXIT STRATEGY v5 — "Dynamic Intelligence" ===
        # Philosophy: EVERY threshold adapts to regime, ML, velocity, RSI/ADX.
        # No more fixed numbers — the market tells us when to hold and when to cut.

        # Calculate dynamic multipliers based on ALL available signals
        profit_mult, loss_mult = self._calculate_dynamic_multipliers(
            guard, regime, ml_signal, ml_confidence, market_context
        )
        trade_state = self._classify_trade_state(guard)

        # BASE thresholds (ATR multiples) — these get MULTIPLIED by dynamic factors
        # Profit thresholds: base * profit_mult * atr_unit
        tp_min = 0.35 * profit_mult * atr_unit         # Dynamic min TP
        tp_secure = 0.60 * profit_mult * atr_unit      # Dynamic secure TP
        tp_hard = 1.20 * profit_mult * atr_unit        # Dynamic hard TP
        tp_peak_trigger = 0.60 * profit_mult * atr_unit # Dynamic peak trigger
        tp_prob = 0.50 * profit_mult * atr_unit        # Dynamic TP probability
        tp_decel = 0.50 * profit_mult * atr_unit       # Dynamic decel check
        tp_small_min = 0.20 * profit_mult * atr_unit   # Dynamic small min
        tp_small_max = 0.30 * profit_mult * atr_unit   # Dynamic small max
        tp_early_min = 0.15 * profit_mult * atr_unit   # Dynamic early exit min

        # Loss thresholds: base * loss_mult * atr_unit
        max_atr_loss = 0.60 * loss_mult * atr_unit     # Dynamic hard stop
        stall_loss = -0.35 * loss_mult * atr_unit      # Dynamic stall
        reversal_loss = -0.20 * loss_mult * atr_unit   # Dynamic reversal
        warn_loss = -0.30 * loss_mult * atr_unit       # Dynamic warning
        timeout_loss = -0.35 * loss_mult * atr_unit    # Dynamic timeout
        stagnant_loss = 0.25 * loss_mult * atr_unit    # Dynamic stagnation

        # === v6: KALMAN VELOCITY ALIASES (moved here for dynamic grace) ===
        # Use Kalman-filtered velocity/acceleration for exit decisions (smoother).
        # Raw velocity still used for counter logic (sign flips, was_positive).
        _vel = guard.kalman_velocity if guard.kalman else (guard.velocity if hasattr(guard, 'velocity') else 0.0)
        _accel = guard.kalman_acceleration if guard.kalman else (guard.acceleration if hasattr(guard, 'acceleration') else 0.0)

        # === DYNAMIC GRACE PERIOD (3-12 minutes based on loss velocity) ===
        # v6.1: Grace adapts to how fast the trade is losing money
        # Fast crash -> short grace (3-4 min)
        # Slow loss/recovery -> long grace (10-12 min)

        if current_profit >= 0:
            # In profit: full grace (regime-based)
            if regime in ("ranging", "mean_reverting"):
                grace_minutes = 12
            elif regime in ("high_volatility", "volatile", "crisis"):
                grace_minutes = 10
            elif regime == "trending":
                grace_minutes = 6
            else:
                grace_minutes = 8
        else:
            # In loss: dynamic grace based on velocity
            loss_velocity = abs(_vel) if _vel < 0 else 0  # Only count negative velocity

            if loss_velocity >= 0.30:
                # VERY FAST crash (>$0.30/sec = $18/min)
                grace_minutes = 3  # Cut fast!
            elif loss_velocity >= 0.15:
                # Fast loss ($0.15/sec = $9/min)
                grace_minutes = 4
            elif loss_velocity >= 0.08:
                # Moderate loss ($0.08/sec = $4.80/min)
                grace_minutes = 5
            elif loss_velocity >= 0.03:
                # Slow loss ($0.03/sec = $1.80/min)
                grace_minutes = 7
            else:
                # Very slow loss or recovering (velocity positive/near zero)
                # Use regime-based grace but reduced 50%
                if regime in ("ranging", "mean_reverting"):
                    grace_minutes = 8  # 12 -> 8
                elif regime in ("high_volatility", "volatile", "crisis"):
                    grace_minutes = 6  # 10 -> 6
                else:
                    grace_minutes = 5  # 8 -> 5

        # Log dynamic multipliers periodically (every 60s)
        if len(guard.profit_timestamps) > 0:
            now_ts = time.time()
            if not hasattr(guard, '_last_dynamic_log') or now_ts - guard._last_dynamic_log >= 60:
                guard._last_dynamic_log = now_ts
                logger.info(
                    f"[DYNAMIC] #{ticket} regime={regime} state={trade_state} "
                    f"P×{profit_mult:.2f} L×{loss_mult:.2f} | "
                    f"tp_min=${tp_min:.1f} max_loss=${max_atr_loss:.1f} "
                    f"grace={grace_minutes}m"
                )

        # === UPDATE TRACKING DATA ===
        guard.current_profit = current_profit
        if current_profit > guard.peak_profit:
            guard.peak_profit = current_profit

        # Update history untuk analisis momentum
        guard.update_history(current_price, current_profit, ml_confidence)

        # Calculate momentum dan TP probability
        momentum = guard.calculate_momentum()
        tp_probability = guard.get_tp_probability()

        # Pre-calculate trade age (used by multiple checks)
        now = datetime.now(WIB)
        current_hour = now.hour
        trade_age_seconds = (now - guard.entry_time).total_seconds()
        trade_age_minutes = trade_age_seconds / 60

        # === v6: ADVANCED EXIT SYSTEMS (Fuzzy + Kelly) ===
        if _ADVANCED_EXITS_ENABLED:
            # === FUZZY LOGIC EXIT CONFIDENCE ===
            if self.fuzzy_controller is not None:
                # Calculate profit retention
                # FIX v0.1.2: Small loss after small profit = micro swing, bukan collapse
                if current_profit < 0 and 0 < guard.peak_profit < 300:  # Peak <$3
                    # Small loss after small profit: treat as medium retention (0.5)
                    # Prevents false "collapsed" trigger (retention < 0.3 -> 95% exit)
                    profit_retention = 0.50
                else:
                    profit_retention = current_profit / guard.peak_profit if guard.peak_profit > 0 else 1.0

                # Calculate profit level (vs target)
                profit_level = current_profit / tp_hard if tp_hard > 0 else 0.5

                # Get RSI from market context
                rsi = market_context.get('rsi', 50) if market_context else 50

                # Evaluate fuzzy exit confidence
                exit_confidence = self.fuzzy_controller.evaluate(
                    velocity=_vel,
                    acceleration=_accel,
                    profit_retention=profit_retention,
                    rsi=rsi,
                    time_in_trade=trade_age_minutes,
                    profit_level=profit_level,
                )

                # === v6.3 PREDICTIVE INTELLIGENCE ===
                # Override or adjust exits based on future predictions

                if _PREDICTIVE_ENABLED:
                    # 1. TRAJECTORY PREDICTION: Check if future profit exceeds targets
                    if self.trajectory_predictor is not None and len(guard.velocity_history) >= 3:
                        # === DEBUG v0.2.0: Trajectory input validation ===
                        logger.info(f"[TRAJ-IN] profit=${current_profit:.4f} | vel={_vel:.6f} | accel={_accel:.6f} | regime={regime}")

                        should_hold, pred_reason, predictions = self.trajectory_predictor.should_hold_position(
                            current_profit=current_profit,
                            velocity=_vel,
                            acceleration=_accel,
                            min_target=tp_min,
                            velocity_history=guard.velocity_history,
                            acceleration_history=guard.acceleration_history,
                            regime=regime  # v0.2.0: Pass regime for dampening
                        )

                        # === v0.2.2: Trajectory prediction output ===
                        pred_1m = predictions.get('pred_1m', 0)
                        logger.info(f"[TRAJ-OUT] pred_1m=${pred_1m:.2f} | conf={predictions['confidence']:.0%}")

                        if should_hold:
                            # Predicted high profit - DON'T EXIT yet
                            logger.info(
                                f"[TRAJECTORY HOLD] {pred_reason} | "
                                f"Predictions: 1m=${predictions['pred_1m']:.2f}, "
                                f"3m=${predictions['pred_3m']:.2f} (conf={predictions['confidence']:.0%})"
                            )
                            # Skip fuzzy exit check - continue holding
                            # (will still be subject to other safety checks below)
                            pass  # Don't return yet, continue to other checks

                    # 2. MOMENTUM PERSISTENCE: Adjust fuzzy threshold based on momentum strength
                    if self.momentum_persistence is not None and len(guard.velocity_history) >= 3:
                        should_raise, new_threshold, momentum_reason = self.momentum_persistence.should_raise_exit_threshold(
                            velocity_history=guard.velocity_history,
                            acceleration_history=guard.acceleration_history,
                            current_profit=current_profit,
                            base_threshold=0.85  # Base threshold (will be adjusted by profit tier below)
                        )

                        if should_raise:
                            logger.info(f"[MOMENTUM PERSIST] {momentum_reason}")
                            # We'll apply this threshold adjustment below in profit-tier logic

                    # 3. RECOVERY STRENGTH: Special handling for recovering positions
                    if self.recovery_detector is not None and guard.peak_loss < -3.0:
                        # Position had significant loss (< -$3), check recovery strength
                        is_strong_recovery, recovery_metrics = self.recovery_detector.analyze_recovery_strength(
                            profit_history=guard.profit_history,
                            peak_loss=guard.peak_loss,
                            velocity_history=guard.velocity_history
                        )

                        if is_strong_recovery:
                            recovery_action, recovery_threshold, recovery_reason = self.recovery_detector.get_recovery_recommendation(
                                profit_history=guard.profit_history,
                                peak_loss=guard.peak_loss,
                                velocity_history=guard.velocity_history,
                                current_exit_threshold=0.85
                            )

                            if recovery_action in ["HOLD_STRONG", "HOLD_WEAK"]:
                                logger.info(
                                    f"[RECOVERY {recovery_action}] {recovery_reason} | "
                                    f"Recovery: {recovery_metrics['recovery_pct']:.0%} from ${guard.peak_loss:.2f}, "
                                    f"vel={recovery_metrics['avg_recovery_vel']:.4f}$/s"
                                )
                                # Apply recovery-adjusted threshold below

                # === PROFIT-AWARE EXIT STRATEGY (v6.2 improvement) ===
                # Different thresholds for profit vs loss to prevent early profit exits

                if current_profit > 0:
                    # === PROFIT TRADES: Hold longer for better gains ===

                    # FIX v0.1.3: Use tiered fuzzy threshold function (FIX 1 v0.1.1 finally active!)
                    fuzzy_threshold = self._calculate_fuzzy_exit_threshold(current_profit)

                    # Determine tier for logging
                    if current_profit < 1.0:
                        tier = "MICRO"
                    elif current_profit < 3.0:
                        tier = "SMALL"
                    elif current_profit < 8.0:
                        tier = "MEDIUM"
                    else:
                        tier = "LARGE"

                    # === v6.3 PREDICTIVE ADJUSTMENTS ===
                    adjustments = []

                    # v0.2.1 FIX 1: LOWER threshold when crash predicted (exit faster!)
                    if (_PREDICTIVE_ENABLED and self.trajectory_predictor is not None and
                        len(guard.velocity_history) >= 3):
                        # Get trajectory prediction (already calculated above)
                        pred_1m = predictions.get('pred_1m', 0) if 'predictions' in locals() else None
                        if pred_1m is not None and pred_1m < 0:
                            # Crash predicted! Lower threshold by 10% (exit faster)
                            crash_penalty = 0.10
                            old_threshold = fuzzy_threshold
                            fuzzy_threshold = max(fuzzy_threshold - crash_penalty, 0.60)  # Floor at 60%
                            if fuzzy_threshold < old_threshold:
                                adjustments.append(f"crash-{crash_penalty:.0%}")
                                logger.warning(
                                    f"[CRASH DETECTED] Trajectory pred ${pred_1m:.2f} < 0 -> "
                                    f"Lowering fuzzy threshold {old_threshold:.0%} -> {fuzzy_threshold:.0%}"
                                )

                    # Apply momentum persistence adjustment
                    if (_PREDICTIVE_ENABLED and self.momentum_persistence is not None and
                        len(guard.velocity_history) >= 3):
                        should_raise, adjusted_threshold, momentum_reason = (
                            self.momentum_persistence.should_raise_exit_threshold(
                                velocity_history=guard.velocity_history,
                                acceleration_history=guard.acceleration_history,
                                current_profit=current_profit,
                                base_threshold=fuzzy_threshold
                            )
                        )
                        if should_raise and adjusted_threshold > fuzzy_threshold:
                            delta = adjusted_threshold - fuzzy_threshold
                            fuzzy_threshold = adjusted_threshold
                            adjustments.append(f"momentum+{delta:.0%}")

                    # Apply recovery strength adjustment (if recovering from loss)
                    if (_PREDICTIVE_ENABLED and self.recovery_detector is not None and
                        guard.peak_loss < -3.0 and len(guard.profit_history) >= 5):
                        is_strong, metrics = self.recovery_detector.analyze_recovery_strength(
                            guard.profit_history, guard.peak_loss, guard.velocity_history
                        )
                        if is_strong and metrics.get('recovery_pct', 0) > 0.8:
                            # Strong recovery - raise threshold by 10%
                            old_threshold = fuzzy_threshold
                            fuzzy_threshold = min(fuzzy_threshold + 0.10, 0.98)
                            if fuzzy_threshold > old_threshold:
                                adjustments.append(f"recovery+{fuzzy_threshold-old_threshold:.0%}")

                    # Check trajectory prediction to prevent premature exit
                    trajectory_override = False
                    if (_PREDICTIVE_ENABLED and self.trajectory_predictor is not None and
                        len(guard.velocity_history) >= 3):
                        should_hold, pred_reason, predictions = (
                            self.trajectory_predictor.should_hold_position(
                                current_profit, _vel, _accel, tp_min,
                                guard.velocity_history, guard.acceleration_history,
                                regime=regime  # v0.2.0: Pass regime for dampening
                            )
                        )
                        if should_hold and predictions.get('pred_1m', 0) > current_profit * 2:
                            # Predicted profit 2x higher in 1 minute - strong hold signal
                            trajectory_override = True
                            logger.warning(
                                f"[TRAJECTORY OVERRIDE] Predicted ${predictions['pred_1m']:.2f} in 1min "
                                f"(current: ${current_profit:.2f}, conf={predictions['confidence']:.0%})"
                            )

                    # Build adjustment string for logging
                    adj_str = f" [{'+'.join(adjustments)}]" if adjustments else ""

                    # High confidence exit (unless trajectory override or peak hold)
                    peak_suppression = getattr(guard, 'peak_hold_active', False)
                    if exit_confidence > fuzzy_threshold and not trajectory_override and not peak_suppression:
                        return True, ExitReason.TAKE_PROFIT, (
                            f"[FUZZY HIGH] Exit confidence: {exit_confidence:.2%} "
                            f"(profit=${current_profit:.2f}, tier={tier}, threshold={fuzzy_threshold:.0%}{adj_str})"
                        )
                    elif trajectory_override:
                        # Log but don't exit - trajectory prediction says hold
                        logger.info(
                            f"[FUZZY SUPPRESSED] Exit confidence {exit_confidence:.2%} > {fuzzy_threshold:.0%} "
                            f"but trajectory override active (pred 1m=${predictions['pred_1m']:.2f})"
                        )
                    elif peak_suppression:
                        # Log but don't exit - approaching peak
                        logger.info(
                            f"[FUZZY SUPPRESSED] Exit confidence {exit_confidence:.2%} > {fuzzy_threshold:.0%} "
                            f"but peak hold active (approaching peak)"
                        )

                    # v0.2.2 Professor AI Enhancement: Partial Exit Strategy
                    # Exit 50% at tp_target * 0.5, hold 50% for peak capture
                    if self.kelly_scaler is not None and current_profit >= tp_min * 0.5:
                        should_exit, close_fraction, kelly_msg = self.kelly_scaler.get_exit_action(
                            exit_confidence, current_profit, tp_hard
                        )

                        # Partial exit: Kelly suggests 30-70% close
                        if should_exit and 0.3 <= close_fraction < 1.0:
                            logger.warning(
                                f"[KELLY PARTIAL] Recommendation: {kelly_msg} "
                                f"(profit=${current_profit:.2f}, fuzzy={exit_confidence:.2%}) "
                                f"[NOTE: Partial close not yet implemented - recommend manual close {close_fraction:.0%}]"
                            )
                            # TODO: Implement actual partial close via mt5.close_position(ticket, volume=lot*close_fraction)
                            # For now, continue to full exit logic below

                        # Full exit: Kelly suggests >70% close
                        elif should_exit and close_fraction >= 0.70:
                            return True, ExitReason.TAKE_PROFIT, (
                                f"[KELLY FULL EXIT] {kelly_msg} (fuzzy={exit_confidence:.2%})"
                            )

                else:
                    # === LOSS TRADES: Exit faster to minimize damage ===

                    # FIX v0.1.2: Grace period untuk loss trades - cegah early exit pada micro swings
                    grace_period_sec = {
                        "ranging": 120,
                        "mean_reverting": 120,
                        "volatile": 90,
                        "high_volatility": 90,
                        "crisis": 60,
                        "trending": 60,
                        "normal": 90,
                    }.get(regime, 90)

                    time_since_entry = time.time() - guard.entry_time.timestamp()
                    in_grace_period = time_since_entry < grace_period_sec

                    # Lower threshold for losses (75%)
                    if exit_confidence > 0.75:
                        # Suppress exit during grace period for small losses (<$2)
                        if in_grace_period and abs(current_profit) < 200:  # $2.00
                            logger.info(
                                f"[GRACE PERIOD] Loss fuzzy={exit_confidence:.2%} suppressed "
                                f"(t={time_since_entry:.0f}s < {grace_period_sec}s, loss=${current_profit:.2f})"
                            )
                        else:
                            return True, ExitReason.POSITION_LIMIT, (
                                f"[FUZZY HIGH LOSS] Exit confidence: {exit_confidence:.2%} "
                                f"(loss=${current_profit:.2f}, cut early)"
                            )

                    # Kelly active for losses (help cut faster)
                    # Also respect grace period for small losses
                    if self.kelly_scaler is not None and exit_confidence > 0.60:
                        should_exit, close_fraction, kelly_msg = self.kelly_scaler.get_exit_action(
                            exit_confidence, current_profit, tp_hard
                        )
                        if should_exit and close_fraction > 0.3:
                            # Suppress kelly exit during grace period for small losses
                            if in_grace_period and abs(current_profit) < 200:  # $2.00
                                logger.info(
                                    f"[GRACE PERIOD] Kelly loss exit suppressed "
                                    f"(t={time_since_entry:.0f}s < {grace_period_sec}s)"
                                )
                            else:
                                return True, ExitReason.POSITION_LIMIT, (
                                    f"[KELLY LOSS] {kelly_msg} (fuzzy={exit_confidence:.2%})"
                                )

        # === PRIORITY 0: EMERGENCY SAFETY CHECKS ===

        # CHECK -1: NO RECOVERY ZONE ($15 threshold)
        # If loss >= $15, exit immediately - no point waiting for recovery
        NO_RECOVERY_THRESHOLD = 1500  # $15.00 per 0.01 lot
        if current_profit <= -NO_RECOVERY_THRESHOLD:
            return True, ExitReason.POSITION_LIMIT, (
                f"[NO RECOVERY] Loss ${abs(current_profit):.2f} too deep "
                f"(threshold ${NO_RECOVERY_THRESHOLD/100:.2f}) - cut immediately"
            )

        # CHECK 0: EMERGENCY CAP ($20 per 0.01 lot)
        # Absolute maximum loss cap - last resort protection
        EMERGENCY_MAX_LOSS = 2000  # $20.00 per 0.01 lot
        if current_profit <= -EMERGENCY_MAX_LOSS:
            return True, ExitReason.POSITION_LIMIT, (
                f"[EMERGENCY CAP] Max loss ${abs(current_profit):.2f} exceeded "
                f"${EMERGENCY_MAX_LOSS/100:.2f} limit - emergency exit!"
            )

        # === CHECK 0A: BREAKEVEN SHIELD (percentage-based, dynamic) ===
        # v5: Protect ANY meaningful profit from becoming a loss.
        # Uses percentage drawdown from peak (not fixed ATR threshold).
        # Peak $3+ -> protect if drops below $1.50
        # Peak $6+ -> protect if drops 70%+ from peak
        # Peak $10+ -> protect if drops 60%+ from peak
        # v5c: min peak raised $3->$5, min age raised 5->8 min (patient protection)
        if atr_unit > 0 and trade_age_minutes >= 8 and guard.peak_profit >= 5.0:
            if guard.peak_profit >= 10.0:
                max_drawdown_pct = 0.60  # Peak $10+: protect at 60% drawdown
            elif guard.peak_profit >= 6.0:
                max_drawdown_pct = 0.70  # Peak $6+: protect at 70% drawdown
            else:
                max_drawdown_pct = 0.80  # Peak $5+: protect at 80% drawdown

            profit_floor = guard.peak_profit * (1 - max_drawdown_pct)
            # Floor must be at least $1.50 to avoid micro-profit exits
            profit_floor = max(profit_floor, 1.50)

            if current_profit <= profit_floor and guard.peak_profit > profit_floor:
                return True, ExitReason.TAKE_PROFIT, (
                    f"[BE-SHIELD] Was +${guard.peak_profit:.2f}, now ${current_profit:+.2f} "
                    f"— {max_drawdown_pct:.0%} drawdown protection (floor=${profit_floor:.1f})"
                )

        # === CHECK 0A.5: DEAD ZONE PROTECTION (v6) ===
        # Protect trades with peak $3-5 that have no other protection.
        # BE-SHIELD kicks in at $5+, so this covers the gap below.
        if trade_age_minutes >= 5 and guard.peak_profit >= 3.0 and guard.peak_profit < 5.0:
            deadzone_floor = max(0.50, guard.peak_profit * 0.33)
            if current_profit <= deadzone_floor:
                return True, ExitReason.TAKE_PROFIT, (
                    f"[DEADZONE] Securing ${current_profit:.2f} — "
                    f"peak ${guard.peak_profit:.2f} floor ${deadzone_floor:.2f} "
                    f"(age {trade_age_minutes:.1f}m)"
                )

        # === CHECK 0A.3: VELOCITY CRASH OVERRIDE (v0.2.1 FIX 3) ===
        # Emergency exit when velocity FLIPS from strong positive to negative
        # This catches extreme momentum crashes that fuzzy logic might delay
        if current_profit > 0 and _vel < -0.05:
            # Check if velocity was previously positive (crash!)
            if len(guard.velocity_history) >= 2:
                prev_velocity = guard.velocity_history[-2] if len(guard.velocity_history) > 1 else 0
                if prev_velocity > 0.10:
                    # EXTREME velocity flip: +0.10 -> -0.05 = crash!
                    velocity_drop = prev_velocity - _vel
                    if velocity_drop > 0.15:  # Change > 0.15 $/s
                        return True, ExitReason.TAKE_PROFIT, (
                            f"[VELOCITY CRASH] Emergency exit! "
                            f"Velocity crashed {prev_velocity:.3f} -> {_vel:.3f} (Δ{velocity_drop:.3f}), "
                            f"profit=${current_profit:.2f}"
                        )

        # === CHECK 0A.4: PEAK DETECTION (v0.2.2 Professor AI Fix #2) ===
        # Hold position if approaching peak (velocity > 0, acceleration < 0)
        # Prevents early exit when profit still rising but decelerating
        if current_profit >= tp_min and _vel > 0.02 and _accel < -0.001:
            # Approaching peak: velocity positive but decelerating
            time_to_peak = -_vel / _accel  # Time when velocity reaches 0 (peak)
            if 0 < time_to_peak <= 30:  # Peak within next 30 seconds
                peak_profit_estimate = current_profit + _vel * time_to_peak + 0.5 * _accel * time_to_peak**2
                # Only hold if estimated peak is significant
                if peak_profit_estimate > current_profit * 1.15:  # At least 15% more
                    logger.info(
                        f"[PEAK HOLD] Approaching peak in {time_to_peak:.0f}s "
                        f"(current=${current_profit:.2f}, est_peak=${peak_profit_estimate:.2f}, "
                        f"vel={_vel:.3f}, accel={_accel:.4f})"
                    )
                    # Don't exit yet - suppress fuzzy logic for this cycle
                    guard.peak_hold_active = True
        else:
            guard.peak_hold_active = False

        # === CHECK 0B: ATR TRAILING (v6 multi-factor + stochastic floor) ===
        # Trail distance = BASE × REGIME × PROFIT_LEVEL × VELOCITY_QUALITY
        # Stochastic floor: profit_floor = max(atr_floor, alpha × peak_profit)
        if atr_unit > 0 and trade_age_minutes >= 8:
            trail_trigger = 0.60 * profit_mult * atr_unit  # Dynamic trigger
            if guard.peak_profit >= trail_trigger:
                # BASE factor (trade state)
                if trade_state == "accelerating_profit":
                    trail_base = 0.40
                elif trade_state in ("steady_profit", "neutral"):
                    trail_base = 0.28
                else:
                    trail_base = 0.18

                # REGIME factor
                if regime == "trending":
                    regime_factor = 1.2
                elif regime in ("ranging", "mean_reverting"):
                    regime_factor = 0.85
                elif regime in ("high_volatility", "volatile", "crisis"):
                    regime_factor = 1.3
                else:
                    regime_factor = 1.0

                # PROFIT LEVEL factor (how close to target)
                if tp_hard > 0 and guard.peak_profit >= tp_hard * 0.75:
                    profit_level_factor = 0.75   # Near target: tighten
                elif tp_hard > 0 and guard.peak_profit >= tp_hard * 0.50:
                    profit_level_factor = 0.90   # Mid range
                else:
                    profit_level_factor = 1.15   # Early: wider trail

                # VELOCITY QUALITY factor (using Kalman-filtered velocity)
                if _vel > 0.05:
                    vel_factor = 1.1    # Positive velocity: wider trail
                elif _vel < -0.03:
                    vel_factor = 0.85   # Negative velocity: tighter trail
                else:
                    vel_factor = 1.0

                trail_atr = trail_base * regime_factor * profit_level_factor * vel_factor
                trail_atr = max(0.12, min(0.50, trail_atr))  # Clamp [0.12, 0.50]

                atr_floor = guard.peak_profit - trail_atr * atr_unit

                # Stochastic floor: alpha × peak_profit (Gemini research)
                if guard.peak_profit >= tp_secure:
                    alpha = 0.60
                elif guard.peak_profit >= tp_min:
                    alpha = 0.50
                else:
                    alpha = 0.40
                stoch_floor = alpha * guard.peak_profit

                profit_floor = max(atr_floor, stoch_floor)
                floor_type = "STOCH" if stoch_floor > atr_floor else "ATR"

                if current_profit < profit_floor and profit_floor > 0:
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[ATR-TRAIL] Profit ${current_profit:.2f} < floor ${profit_floor:.2f} "
                        f"(peak ${guard.peak_profit:.2f}, trail {trail_atr:.2f}*ATR=${trail_atr*atr_unit:.1f}, "
                        f"floor={floor_type}, state={trade_state})"
                    )

        # === CHECK 0C: PROFIT MOMENTUM FADE ===
        # Detect when profit velocity transitions from positive to negative.
        # This catches the exact moment momentum fades — before big drawdown.
        # Example: Trade peaked $7.58, velocity was +0.05, now -0.03 -> fading
        if current_profit >= tp_min and trade_age_minutes >= 3:
            # Velocity was positive and now turned negative (momentum fading)
            # v6: uses Kalman-filtered velocity for trigger, raw for counter tracking
            if guard.velocity_was_positive and _vel < -0.01:
                # Require multiple deceleration readings to avoid false triggers
                if guard.decel_at_profit_count >= 3 or guard.velocity_sign_flips >= 2:
                    fade_strength = "strong" if _vel < -0.05 else "moderate"
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[MOM-FADE] Securing ${current_profit:.2f} — momentum fading ({fade_strength}) "
                        f"vel={_vel:.3f} decel={guard.decel_at_profit_count}x "
                        f"flips={guard.velocity_sign_flips} peak=${guard.peak_profit:.2f}"
                    )

        # === CHECK 0D: CAN'T MAKE NEW HIGHS ===
        # Detect when trade has profit but can't push to new peaks.
        # Pattern: price approaches peak multiple times but fails -> resistance.
        # Example: Peak $6.35, tried 4x to break, profit now $5.20 -> take it
        if current_profit >= tp_min and trade_age_minutes >= 5 and guard.peak_update_time > 0:
            peak_age = time.time() - guard.peak_update_time
            if peak_age >= 60 and guard.failed_peak_attempts >= 3:
                # Peak is stale (60s+) and multiple failed attempts
                peak_retention = current_profit / guard.peak_profit if guard.peak_profit > 0 else 1
                if peak_retention < 0.90:  # Lost 10%+ from peak
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[NO-NEW-HIGH] Securing ${current_profit:.2f} — "
                        f"peak ${guard.peak_profit:.2f} stale {peak_age:.0f}s, "
                        f"{guard.failed_peak_attempts} failed attempts, "
                        f"retention {peak_retention:.0%}"
                    )

        # === CHECK 0E: RSI/STOCH REVERSAL AT PROFIT ===
        # Use market indicators to detect imminent reversal while in profit.
        # When RSI/Stoch reaches extreme, mean reversion is likely.
        # SELL + oversold -> price will bounce up (against us)
        # BUY + overbought -> price will drop (against us)
        if current_profit >= tp_min and market_context and trade_age_minutes >= 3:
            rsi = market_context.get("rsi")
            stoch_k = market_context.get("stoch_k")

            reversal_signal = False
            reversal_detail = ""

            if rsi is not None and stoch_k is not None:
                if guard.direction == "SELL":
                    # Oversold = price about to bounce UP (bad for SELL)
                    if rsi < 25 and stoch_k < 20:
                        reversal_signal = True
                        reversal_detail = f"RSI={rsi:.0f} Stoch={stoch_k:.0f} (double oversold)"
                    elif rsi < 20 or stoch_k < 10:
                        reversal_signal = True
                        reversal_detail = f"RSI={rsi:.0f} Stoch={stoch_k:.0f} (extreme oversold)"
                elif guard.direction == "BUY":
                    # Overbought = price about to drop (bad for BUY)
                    if rsi > 75 and stoch_k > 80:
                        reversal_signal = True
                        reversal_detail = f"RSI={rsi:.0f} Stoch={stoch_k:.0f} (double overbought)"
                    elif rsi > 80 or stoch_k > 90:
                        reversal_signal = True
                        reversal_detail = f"RSI={rsi:.0f} Stoch={stoch_k:.0f} (extreme overbought)"

            if reversal_signal:
                guard.rsi_extreme_count += 1
                # Require 2+ consecutive extreme readings to avoid whipsaw
                if guard.rsi_extreme_count >= 2:
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[RSI-EXIT] Securing ${current_profit:.2f} — "
                        f"{reversal_detail} for {guard.rsi_extreme_count} readings "
                        f"(peak=${guard.peak_profit:.2f})"
                    )
            else:
                guard.rsi_extreme_count = 0  # Reset: not at extreme

        # === CHECK 0F: TIME-WEIGHTED PROFIT STALL ===
        # Detect profit stuck at the same level for too long.
        # If profitable but not growing, market lost momentum — take it.
        # Higher profit = more patience, lower profit = exit sooner.
        if current_profit >= tp_min and trade_age_minutes >= 5 and guard.profit_stall_start_time > 0:
            stall_duration = time.time() - guard.profit_stall_start_time
            # Dynamic stall patience based on profit level
            if current_profit >= tp_secure:
                stall_patience = 120  # $6+ profit: wait 120s before declaring stall
            elif current_profit >= tp_min:
                stall_patience = 90   # $3+ profit: wait 90s
            else:
                stall_patience = 60   # Small profit: exit at 60s stall

            if stall_duration >= stall_patience:
                drift = current_profit - guard.profit_stall_anchor
                return True, ExitReason.TAKE_PROFIT, (
                    f"[PROFIT-STALL] Securing ${current_profit:.2f} — "
                    f"stalled {stall_duration:.0f}s (patience={stall_patience}s) "
                    f"drift=${drift:+.2f} peak=${guard.peak_profit:.2f}"
                )

        # === CHECK 1: SMART TAKE PROFIT ===
        if current_profit >= tp_min:  # Profit >= scaled threshold
            # A. Hard TP - profit sangat bagus
            if current_profit >= tp_hard:
                return True, ExitReason.TAKE_PROFIT, f"[TP] Target profit reached: ${current_profit:.2f}"

            # B. Momentum-based TP - profit bagus tapi momentum turun
            if current_profit >= tp_secure and momentum < -30:
                return True, ExitReason.TAKE_PROFIT, f"[SECURE] Securing ${current_profit:.2f} (momentum dropping: {momentum:.0f})"

            # C. Peak protection - profit turun dari peak
            # v5d: only LOCK at substantial peaks (tp_secure, ~$6+) not small ones (~$4)
            # Small peaks ($3-5) are noise — let trade develop to full potential
            if guard.peak_profit > tp_secure and current_profit < guard.peak_profit * 0.6:
                return True, ExitReason.TAKE_PROFIT, f"[LOCK] Securing ${current_profit:.2f} (was ${guard.peak_profit:.2f} peak)"

            # D. Low TP probability - kemungkinan TP rendah
            if tp_probability < 25 and current_profit >= tp_prob:
                return True, ExitReason.TAKE_PROFIT, f"[PROB] Taking profit ${current_profit:.2f} (TP prob: {tp_probability:.0f}%)"

            # F. Velocity reversal — profit >= tp_min but velocity turning strongly negative
            # v4: only at substantial profit AND strong reversal
            # v6: uses Kalman-filtered velocity
            if _vel < -0.25 and trade_age_minutes >= 5 and current_profit >= tp_secure:
                return True, ExitReason.TAKE_PROFIT, f"[VEL-EXIT] Securing ${current_profit:.2f} (velocity: {_vel:.3f} $/s, momentum: {momentum:+.0f})"

            # G. Deceleration — profit >= tp_decel, growth slowing significantly
            # v6: uses Kalman-filtered velocity/acceleration
            if current_profit >= tp_decel and _accel < -0.05 and _vel < 0.1:
                return True, ExitReason.TAKE_PROFIT, f"[DECEL] Securing ${current_profit:.2f} (accel: {_accel:.4f}, vel: {_vel:.3f})"

            # H. PROFIT CAPTURE — velocity stall/reversal at good profit level
            # Fills the gap: profit is good (>= tp_min) but below tp_secure/tp_hard,
            # and the move is stalling. Captures profit BEFORE big drawdown happens.
            # v6: uses Kalman-filtered velocity
            if _vel <= 0:
                guard.profit_capture_count += 1
                # Immediate capture: velocity clearly negative at GOOD profit (>= tp_secure)
                if _vel < -0.25 and guard.profit_capture_count >= 3 and current_profit >= tp_secure:
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[CAPTURE] Securing ${current_profit:.2f} — velocity reversing "
                        f"(vel={_vel:.3f}, peak=${guard.peak_profit:.2f}, "
                        f"stall={guard.profit_capture_count}x)"
                    )
                # Stall capture: velocity near zero for many intervals at good profit
                if guard.profit_capture_count >= 6 and current_profit >= tp_secure:
                    return True, ExitReason.TAKE_PROFIT, (
                        f"[CAPTURE] Securing ${current_profit:.2f} — profit stalling "
                        f"(vel={_vel:.3f}, peak=${guard.peak_profit:.2f}, "
                        f"stall={guard.profit_capture_count}x)"
                    )
            else:
                guard.profit_capture_count = 0  # Reset: velocity positive, profit growing

            # E. Masih bagus, let it run
            if momentum >= 0:
                return False, None, f"Profit ${current_profit:.2f} [GOOD] (momentum: {momentum:+.0f}, TP prob: {tp_probability:.0f}%)"

        # === CHECK 1.5: FAST REVERSAL (small profit, ATR-scaled) ===
        # v4: DISABLED — small profit exits killed winning trades in v3/v3b
        # Let trades run through small-profit zone without panic exits
        # The BE-SHIELD and ATR-TRAIL handle protection at higher profit levels

        # === CHECK 2: SMART EARLY EXIT (small profit, scaled) ===
        # v4: DISABLED — taking small profits prevents reaching $10+ targets
        # Only the ML reversal + high confidence check remains, with higher bar
        if tp_early_min <= current_profit < tp_small_max:
            # Only exit small profit if ML is VERY confident about reversal AND momentum very negative
            if momentum < -70 and ml_confidence >= 0.75 and trade_age_minutes >= 10:
                is_reversal = (
                    (guard.direction == "BUY" and ml_signal == "SELL") or
                    (guard.direction == "SELL" and ml_signal == "BUY")
                )
                if is_reversal:
                    return True, ExitReason.TAKE_PROFIT, f"[WARN] Early exit ${current_profit:.2f} (reversal signal: {ml_signal} {ml_confidence:.0%})"

        # === CHECK 3: SMART HOLD FOR GOLDEN TIME (TIGHTENED v2) ===
        # FIX: REMOVED SMART HOLD MARTINGALE BEHAVIOR
        # Holding losing positions waiting for "golden time" is DANGEROUS
        # It encourages holding losers hoping they'll recover
        # PROPER RISK MANAGEMENT: Follow SL rules, don't hope for recovery

        # === ATR HARD STOP — dynamic min age based on regime ===
        # v5c: Max loss is DYNAMIC (0.60 ATR * loss_mult).
        # Min age for hard stop = grace_minutes * 0.75 (at least 5 min).
        # Raised from max(3, grace/2) -> max(5, grace*0.75) for more breathing room.
        hard_stop_min_age = max(5.0, grace_minutes * 0.75)
        if current_profit < 0 and abs(current_profit) >= max_atr_loss and trade_age_minutes >= hard_stop_min_age:
            return True, ExitReason.POSITION_LIMIT, (
                f"[ATR-STOP] Loss ${abs(current_profit):.2f} >= ${max_atr_loss:.2f} "
                f"(0.60×{loss_mult:.1f}×ATR) after {trade_age_minutes:.1f}m "
                f"[{regime}|{trade_state}]"
            )

        if current_profit < 0:
            loss_in_atr = abs(current_profit) / atr_unit if atr_unit > 0 else 0

            # v5: Early cut thresholds ADAPT to trade state
            # In crashing state: cut sooner. In recovering state: give more room.
            mom_threshold = -60 if trade_state != "crashing" else -40
            loss_threshold = 0.30 * loss_mult if trade_state != "crashing" else 0.20 * loss_mult

            momentum_trigger = momentum < mom_threshold and loss_in_atr >= loss_threshold
            velocity_trigger = _vel < -0.30 and loss_in_atr >= 0.20 * loss_mult  # v6: Kalman

            # VELOCITY EMERGENCY EXIT — only bypass grace for EXTREME drops
            # v6: uses Kalman-filtered velocity/acceleration
            velocity_emergency = (
                _vel < -0.40
                and loss_in_atr >= 0.40 * loss_mult
                and _accel < -0.005
                and len(guard.profit_history) >= 6
            )

            if velocity_emergency:
                logger.info(
                    f"[VELOCITY EXIT] Loss ${abs(current_profit):.2f} ({loss_in_atr:.2f} ATR) "
                    f"vel={_vel:.3f} accel={_accel:.4f} — EMERGENCY CUT"
                )
                return True, ExitReason.TREND_REVERSAL, (
                    f"[VELOCITY EXIT] Loss ${abs(current_profit):.2f} ({loss_in_atr:.2f} ATR) "
                    f"vel={_vel:.3f} accel={_accel:.4f} — fast drop detected"
                )

            if momentum_trigger or velocity_trigger:
                if trade_age_minutes < grace_minutes:
                    logger.info(f"[GRACE] Loss ${abs(current_profit):.2f} ({loss_in_atr:.2f} ATR) + momentum ({momentum:.0f}) vel({_vel:.3f}) — holding {trade_age_minutes:.1f}m/{grace_minutes}m grace")
                else:
                    trigger_type = "momentum" if momentum_trigger else "velocity"
                    logger.info(f"[EARLY CUT] Loss ${abs(current_profit):.2f} ({loss_in_atr:.2f} ATR) + weak {trigger_type} — CUTTING")
                    return True, ExitReason.TREND_REVERSAL, f"[EARLY CUT] Loss ${abs(current_profit):.2f} ({loss_in_atr:.2f} ATR) + {trigger_type} — cutting"

            # Time-aware stagnation: stagnant for 120s+ with loss > 0.25 ATR
            # v4: much more patient — 120s (from 45s), min age 5 min (from 2)
            if guard.stagnation_seconds >= 120 and abs(current_profit) > stagnant_loss and trade_age_minutes >= 5:
                return True, ExitReason.TREND_REVERSAL, f"[STAGNANT] Loss ${abs(current_profit):.2f} stagnant {guard.stagnation_seconds:.0f}s — cutting"

        # === CHECK 4: TREND REVERSAL (ATR-based) ===
        # Close lebih cepat jika ML reversal + loss > 0.2 ATR
        is_reversal = False
        if guard.direction == "BUY" and ml_signal == "SELL" and ml_confidence >= self.trend_reversal_threshold:
            is_reversal = True
            guard.reversal_warnings += 1
        elif guard.direction == "SELL" and ml_signal == "BUY" and ml_confidence >= self.trend_reversal_threshold:
            is_reversal = True
            guard.reversal_warnings += 1

        # ML reversal + loss > 0.2 ATR -> cut (shorter grace: 10 min)
        if is_reversal and current_profit < reversal_loss:
            if trade_age_minutes < grace_minutes:
                logger.info(f"[GRACE] Reversal ({ml_signal} {ml_confidence:.0%}) loss ${current_profit:.2f} — holding {trade_age_minutes:.1f}m/{grace_minutes}m grace")
            else:
                return True, ExitReason.TREND_REVERSAL, f"[REVERSAL] {ml_signal} ({ml_confidence:.0%}) - Loss: ${current_profit:.2f}"

        # 3x reversal warnings + loss > 0.3 ATR -> cut
        if guard.reversal_warnings >= 3 and current_profit < warn_loss:
            if trade_age_minutes < grace_minutes:
                logger.info(f"[GRACE] {guard.reversal_warnings}x reversal warnings, loss ${current_profit:.2f} — holding {trade_age_minutes:.1f}m/{grace_minutes}m grace")
            else:
                return True, ExitReason.TREND_REVERSAL, f"[WARN] {guard.reversal_warnings}x reversal warnings - Loss: ${current_profit:.2f}"

        # === CHECK 5: ABSOLUTE BACKUP STOP (dynamic safety net) ===
        # v5d: BACKUP-SL now respects grace period (was firing at 1-2 min!)
        # Also uses loss_mult floor of 0.8 so ML disagreement can't crush threshold
        # to $4-5 (which fires on normal gold noise within seconds).
        backup_loss_mult = max(0.7, loss_mult)  # v6: relaxed 0.8->0.7 (ML fix makes band-aid unnecessary)
        backup_pct = min(0.30, 0.20 * backup_loss_mult)  # Cap at 30% of max_loss
        if trade_age_minutes >= grace_minutes and current_profit <= -(effective_max_loss * backup_pct):
            return True, ExitReason.POSITION_LIMIT, (
                f"[BACKUP-SL] Loss ${abs(current_profit):.2f} ({backup_pct:.0%} of "
                f"${effective_max_loss:.2f}) — safety net [{regime}|L×{backup_loss_mult:.1f}]"
            )

        # === CHECK 5b: STALL DETECTION (ATR-scaled) ===
        # v4: more patient stall detection — 10 samples, 8 count threshold
        stall_range_threshold = 0.10 * atr_unit  # 10% of ATR unit
        if len(guard.profit_history) >= 10 and trade_age_minutes >= 8:
            recent_range = max(guard.profit_history[-10:]) - min(guard.profit_history[-10:])
            if recent_range < stall_range_threshold and current_profit < stall_loss:
                guard.stall_count += 1
                if guard.stall_count >= 8:  # v4: from 4 to 8
                    return True, ExitReason.TREND_REVERSAL, f"[STALL] Loss ${current_profit:.2f} stalled (range ${recent_range:.1f} < ${stall_range_threshold:.1f}) — cutting"

        # === CHECK 6: DAILY LOSS LIMIT ===
        potential_daily_loss = self._state.daily_loss + abs(min(0, current_profit))
        if potential_daily_loss >= self.max_daily_loss_usd:
            return True, ExitReason.DAILY_LIMIT, f"[LIMIT] Would exceed daily loss limit"

        # === CHECK 7: WEEKEND CLOSE ===
        # Market closes Saturday 05:00 WIB
        # Friday 22:30+ WIB = approaching weekend (reduce exposure)
        # Saturday 04:30-05:00 WIB = last 30 min before close
        is_friday_late = now.weekday() == 4 and now.hour >= 22 and now.minute >= 30
        is_saturday_close = now.weekday() == 5 and now.hour >= 4 and now.minute >= 30 and now.hour < 5
        near_weekend_close = is_friday_late or is_saturday_close
        if near_weekend_close:
            if current_profit > 0:
                return True, ExitReason.WEEKEND_CLOSE, f"[WEEKEND] Weekend close - profit ${current_profit:.2f}"
            elif current_profit > warn_loss:
                return True, ExitReason.WEEKEND_CLOSE, f"[WEEKEND] Weekend close - small loss ${current_profit:.2f}"

        # === CHECK 8: SMART TIME-BASED EXIT (session-scaled) ===
        # Don't cut winners short - check profit growth and trend
        trade_duration_hours = (now - guard.entry_time).total_seconds() / 3600

        # Check if profit is growing (positive momentum AND positive velocity)
        # v6: uses Kalman-filtered velocity
        profit_growing = momentum > 0 and _vel > 0
        ml_agrees = (
            (guard.direction == "BUY" and ml_signal == "BUY") or
            (guard.direction == "SELL" and ml_signal == "SELL")
        )

        # v4: PATIENT time exits — gold trends can take hours to develop
        # 4+ hours: Only exit if stuck (no profit growth)
        if trade_duration_hours >= 4:
            if current_profit < tp_early_min and not profit_growing:
                # Stuck with no growth - exit
                if current_profit >= 0:
                    return True, ExitReason.TAKE_PROFIT, f"[TIMEOUT] Breakeven + no growth after {trade_duration_hours:.1f}h"
                elif current_profit > timeout_loss:
                    return True, ExitReason.TREND_REVERSAL, f"[TIMEOUT] Small loss ${current_profit:.2f} + no growth after {trade_duration_hours:.1f}h"
            elif current_profit >= tp_early_min and profit_growing and ml_agrees:
                # Profitable and growing - extend time (log only)
                logger.debug(f"[TIME OK] Profit growing +${current_profit:.2f}, extending time (was {trade_duration_hours:.1f}h)")

        # 6+ hours: Exit unless significantly profitable AND still growing
        if trade_duration_hours >= 6:
            if current_profit < tp_min or not profit_growing:
                return True, ExitReason.TREND_REVERSAL, f"[MAX TIME] {trade_duration_hours:.1f}h - profit ${current_profit:.2f}"
            elif trade_duration_hours >= 8:
                return True, ExitReason.TAKE_PROFIT, f"[MAX TIME] Taking profit ${current_profit:.2f} after {trade_duration_hours:.1f}h"

        # === DEFAULT: HOLD ===
        status = f"+${current_profit:.2f}" if current_profit > 0 else f"-${abs(current_profit):.2f}"
        return False, None, f"HOLD {status} | Mom: {momentum:+.0f} | TP%: {tp_probability:.0f} | ML: {ml_signal}({ml_confidence:.0%})"

    def record_trade_result(self, profit: float) -> Dict:
        """
        Record trade result for daily and total tracking.

        Returns:
            Dict with status info including any limit violations
        """
        self._daily_pnl.append(profit)

        result = {
            "profit": profit,
            "daily_loss": 0,
            "total_loss": 0,
            "daily_limit_hit": False,
            "total_limit_hit": False,
            "can_trade": True,
        }

        if profit >= 0:
            self._state.daily_profit += profit
            self._state.consecutive_losses = 0
            # Reduce total loss with profit (recovery)
            self._total_loss = max(0, self._total_loss - profit)
            logger.info(f"PROFIT recorded: +${profit:.2f} | Daily: +${self._state.daily_profit:.2f} | Total Loss: ${self._total_loss:.2f}")
        else:
            loss_amount = abs(profit)
            self._state.daily_loss += loss_amount
            self._total_loss += loss_amount  # Add to total loss
            self._state.consecutive_losses += 1
            self._state.last_loss_amount = loss_amount
            logger.warning(f"LOSS recorded: -${loss_amount:.2f} | Daily loss: ${self._state.daily_loss:.2f} | Total Loss: ${self._total_loss:.2f}")

            # Check if we should stop - TOTAL loss limit
            if self._total_loss >= self.max_total_loss_usd:
                self._state.mode = TradingMode.STOPPED
                self._state.can_trade = False
                result["total_limit_hit"] = True
                result["can_trade"] = False
                logger.error(f"TOTAL LOSS LIMIT REACHED ({self.max_total_loss_percent}%) - TRADING STOPPED PERMANENTLY")

            # Check if we should stop - daily loss limit
            elif self._state.daily_loss >= self.max_daily_loss_usd:
                self._state.mode = TradingMode.STOPPED
                self._state.can_trade = False
                result["daily_limit_hit"] = True
                result["can_trade"] = False
                logger.error(f"DAILY LOSS LIMIT REACHED ({self.max_daily_loss_percent}%) - STOPPING TRADING TODAY")

        result["daily_loss"] = self._state.daily_loss
        result["total_loss"] = self._total_loss

        self._save_daily_state()
        self._update_state()

        return result

    def unregister_position(self, ticket: int):
        """Remove position from monitoring."""
        if ticket in self._position_guards:
            del self._position_guards[ticket]

    def get_trading_recommendation(self) -> Dict:
        """Get trading recommendation based on current state."""
        self._update_state()

        return {
            "can_trade": self._state.can_trade,
            "mode": self._state.mode.value,
            "reason": self._state.reason,
            "recommended_lot": self._state.recommended_lot,
            "max_lot": self._state.max_allowed_lot,
            "daily_profit": self._state.daily_profit,
            "daily_loss": self._state.daily_loss,
            "daily_net": self._state.daily_profit - self._state.daily_loss,
            "remaining_daily_risk": max(0, self.max_daily_loss_usd - self._state.daily_loss),
            "total_loss": self._total_loss,
            "remaining_total_risk": max(0, self.max_total_loss_usd - self._total_loss),
            "max_loss_per_trade": self.max_loss_per_trade,
            "consecutive_losses": self._state.consecutive_losses,
        }

    def should_use_stop_loss(self) -> Tuple[bool, str]:
        """
        Determine if we should use stop loss.

        REKOMENDASI: TIDAK menggunakan hard stop loss.
        Alasan:
        1. Market sering "sweep" stop loss sebelum reversal
        2. Dengan lot kecil, bisa hold lebih lama
        3. ML akan mendeteksi trend reversal yang sebenarnya
        """
        return False, "Smart management tanpa hard SL - lot kecil, hold through volatility"

    def reset_total_loss(self):
        """Reset total loss counter (admin function - use with caution)."""
        old_total = self._total_loss
        self._total_loss = 0.0
        self._save_daily_state()
        logger.warning(f"TOTAL LOSS RESET: ${old_total:.2f} -> $0.00")
        self._update_state()

    def get_risk_summary(self) -> str:
        """Get human-readable risk summary."""
        self._update_state()
        lines = [
            "=" * 40,
            "RISK MANAGEMENT SUMMARY",
            "=" * 40,
            f"Capital: ${self.capital:.2f}",
            f"",
            f"Daily Loss: ${self._state.daily_loss:.2f} / ${self.max_daily_loss_usd:.2f} ({self.max_daily_loss_percent}%)",
            f"Total Loss: ${self._total_loss:.2f} / ${self.max_total_loss_usd:.2f} ({self.max_total_loss_percent}%)",
            f"S/L Per Trade: ${self.max_loss_per_trade:.2f} ({self.max_loss_per_trade_percent}%)",
            f"",
            f"Mode: {self._state.mode.value}",
            f"Can Trade: {self._state.can_trade}",
            f"Reason: {self._state.reason}",
            "=" * 40,
        ]
        return "\n".join(lines)


def create_smart_risk_manager(capital: float = 5000.0) -> SmartRiskManager:
    """Create smart risk manager instance with NEW settings."""
    return SmartRiskManager(
        capital=capital,
        max_daily_loss_percent=5.0,         # Max 5% daily loss
        max_total_loss_percent=10.0,        # Max 10% total loss (stop trading)
        max_loss_per_trade_percent=0.5,     # FIX 5 v0.1.1: S/L 0.5% per trade (~$25 for $5k)
        emergency_sl_percent=2.0,           # Emergency broker SL 2% per trade
        base_lot_size=0.01,                 # Base lot 0.01 (minimum)
        max_lot_size=0.02,                  # Maximum 0.02 (sangat kecil)
        recovery_lot_size=0.01,             # Saat recovery tetap 0.01
        trend_reversal_threshold=0.65,      # Close jika ML 65%+ yakin (lebih sensitif)
        max_concurrent_positions=2,         # Max 2 posisi bersamaan
    )


if __name__ == "__main__":
    # Test dengan modal $50
    print("=" * 50)
    print("TESTING DENGAN MODAL $50")
    print("=" * 50)
    manager = create_smart_risk_manager(50)

    print("\n=== Risk Settings ===")
    print(f"Capital: ${manager.capital:.2f}")
    print(f"Daily Loss Limit: {manager.max_daily_loss_percent}% = ${manager.max_daily_loss_usd:.2f}")
    print(f"Total Loss Limit: {manager.max_total_loss_percent}% = ${manager.max_total_loss_usd:.2f}")
    print(f"S/L Per Trade: {manager.max_loss_per_trade_percent}% = ${manager.max_loss_per_trade:.2f}")

    print("\n=== Risk State ===")
    state = manager.get_state()
    print(f"Mode: {state.mode.value}")
    print(f"Can Trade: {state.can_trade}")
    print(f"Recommended Lot: {state.recommended_lot}")

    print("\n=== Lot Calculation ===")
    lot = manager.calculate_lot_size(4950, confidence=0.70)
    print(f"Calculated Lot: {lot}")

    print("\n=== Trading Recommendation ===")
    rec = manager.get_trading_recommendation()
    for k, v in rec.items():
        print(f"  {k}: {v}")

    print("\n=== Stop Loss Recommendation ===")
    use_sl, reason = manager.should_use_stop_loss()
    print(f"Use Stop Loss: {use_sl}")
    print(f"Reason: {reason}")
