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
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo
from loguru import logger
import polars as pl

WIB = ZoneInfo("Asia/Jakarta")


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

    def update_history(self, price: float, profit: float, ml_confidence: float, max_history: int = 20):
        """Update price/profit history untuk analisis momentum."""
        self.price_history.append(price)
        self.profit_history.append(profit)
        self.ml_confidence_history.append(ml_confidence)

        # Keep only last N entries
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.profit_history = self.profit_history[-max_history:]
            self.ml_confidence_history = self.ml_confidence_history[-max_history:]

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
            conf_score = ((conf_trend + 0.3) / 0.6) * 20  # -0.3 to +0.3 → 0 to 20
            conf_score = max(0, min(20, conf_score))
        else:
            conf_score = 10

        # Factor 4: Time penalty (0-10 points lost)
        time_elapsed = (datetime.now(WIB) - self.entry_time).total_seconds() / 3600  # hours
        time_penalty = min(10, time_elapsed * 2)  # Lose 2 points per hour

        probability = progress_score + momentum_score + conf_score - time_penalty
        return max(0, min(100, probability))


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
        max_loss_per_trade_percent: float = 1.0,  # Max 1% per trade (software S/L)
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

        logger.info("=" * 50)
        logger.info("SMART RISK MANAGER v2.2 INITIALIZED")
        logger.info(f"  Capital: ${capital:,.2f}")
        logger.info(f"  Max Daily Loss: {max_daily_loss_percent}% (${self.max_daily_loss_usd:.2f})")
        logger.info(f"  Max Total Loss: {max_total_loss_percent}% (${self.max_total_loss_usd:.2f})")
        logger.info(f"  Software S/L: {max_loss_per_trade_percent}% (${self.max_loss_per_trade:.2f})")
        logger.info(f"  Emergency Broker S/L: {emergency_sl_percent}% (${self.emergency_sl_usd:.2f})")
        logger.info(f"  Max Positions: {max_concurrent_positions}")
        logger.info(f"  Base Lot: {base_lot_size}")
        logger.info(f"  Max Lot: {max_lot_size}")
        logger.info("  Mode: SMART S/L (software + broker safety net)")
        logger.info("=" * 50)

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

    def evaluate_position(
        self,
        ticket: int,
        current_price: float,
        current_profit: float,
        ml_signal: str,
        ml_confidence: float,
        regime: str = "normal",
    ) -> Tuple[bool, Optional[ExitReason], str]:
        """
        SMART DYNAMIC TP - Evaluate if position should be closed.

        TIDAK hanya menunggu TP tercapai, tapi juga:
        1. Analisis momentum - apakah harga bergerak ke arah TP?
        2. Probabilitas TP - masih mungkin tercapai?
        3. ML confidence trend - apakah trend masih kuat?
        4. Early exit jika probabilitas TP rendah

        Returns: (should_close, reason, message)
        """
        guard = self._position_guards.get(ticket)
        if not guard:
            return False, None, "Position not registered"

        # === UPDATE TRACKING DATA ===
        guard.current_profit = current_profit
        if current_profit > guard.peak_profit:
            guard.peak_profit = current_profit

        # Update history untuk analisis momentum
        guard.update_history(current_price, current_profit, ml_confidence)

        # Calculate momentum dan TP probability
        momentum = guard.calculate_momentum()
        tp_probability = guard.get_tp_probability()

        # === CHECK 1: SMART TAKE PROFIT ===
        if current_profit >= 15:  # Profit $15+
            # A. Hard TP - profit sangat bagus
            if current_profit >= 40:
                return True, ExitReason.TAKE_PROFIT, f"[TP] Target profit reached: ${current_profit:.2f}"

            # B. Momentum-based TP - profit bagus tapi momentum turun
            if current_profit >= 25 and momentum < -30:
                return True, ExitReason.TAKE_PROFIT, f"[SECURE] Securing ${current_profit:.2f} (momentum dropping: {momentum:.0f})"

            # C. Peak protection - profit turun dari peak
            if guard.peak_profit > 30 and current_profit < guard.peak_profit * 0.6:
                return True, ExitReason.TAKE_PROFIT, f"[LOCK] Securing ${current_profit:.2f} (was ${guard.peak_profit:.2f} peak)"

            # D. Low TP probability - kemungkinan TP rendah
            if tp_probability < 25 and current_profit >= 20:
                return True, ExitReason.TAKE_PROFIT, f"[PROB] Taking profit ${current_profit:.2f} (TP prob: {tp_probability:.0f}%)"

            # E. Masih bagus, let it run
            if momentum >= 0:
                return False, None, f"Profit ${current_profit:.2f} [GOOD] (momentum: {momentum:+.0f}, TP prob: {tp_probability:.0f}%)"

        # === CHECK 2: SMART EARLY EXIT (small profit) ===
        if 5 <= current_profit < 15:
            # Ambil profit kecil jika momentum sangat negatif
            if momentum < -50 and ml_confidence >= 0.65:
                # ML yakin trend berbalik
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

        now = datetime.now(WIB)
        current_hour = now.hour

        # Early cut: If loss > 30% of max and momentum negative, cut early
        if current_profit < 0:
            loss_percent_of_max = abs(current_profit) / self.max_loss_per_trade * 100

            # Cut early if momentum is against us AND loss is significant
            if momentum < -30 and loss_percent_of_max >= 30:
                logger.info(f"[EARLY CUT] Loss ${abs(current_profit):.2f} ({loss_percent_of_max:.0f}%) + weak momentum ({momentum:.0f}) - CUTTING EARLY")
                return True, ExitReason.TREND_REVERSAL, f"[EARLY CUT] Loss ${abs(current_profit):.2f} + momentum {momentum:.0f} - cutting to preserve daily limit"

            # NOTE: Smart Hold REMOVED - no more holding losers hoping for golden time
            # If SL is hit, close the trade immediately

        # === CHECK 4: TREND REVERSAL (LEBIH SENSITIF) ===
        # Close lebih cepat jika ada reversal signal - tidak perlu tunggu loss besar
        is_reversal = False
        if guard.direction == "BUY" and ml_signal == "SELL" and ml_confidence >= self.trend_reversal_threshold:
            is_reversal = True
            guard.reversal_warnings += 1
        elif guard.direction == "SELL" and ml_signal == "BUY" and ml_confidence >= self.trend_reversal_threshold:
            is_reversal = True
            guard.reversal_warnings += 1

        # LEBIH KETAT: Close pada reversal jika loss > 40% dari max (sebelumnya 60%)
        loss_moderate = abs(current_profit) > (self.max_loss_per_trade * 0.4)
        if is_reversal and current_profit < -8 and loss_moderate:
            return True, ExitReason.TREND_REVERSAL, f"[REVERSAL] Reversal signal ({ml_signal} {ml_confidence:.0%}) - Loss: ${current_profit:.2f}"

        # Close jika sudah 3x warning reversal (sebelumnya 5x)
        if guard.reversal_warnings >= 3 and current_profit < -10:
            return True, ExitReason.TREND_REVERSAL, f"[WARN] Multiple reversal warnings ({guard.reversal_warnings}x) - Loss: ${current_profit:.2f}"

        # === CHECK 5: MAXIMUM LOSS PER TRADE (LEBIH KETAT) ===
        # Close jika loss sudah 50%+ dari max (sebelumnya 80%)
        if current_profit <= -(self.max_loss_per_trade * 0.50):
            # Hanya hold jika golden time SANGAT dekat (1 jam) dan momentum tidak terlalu buruk
            if hours_to_golden <= 1 and hours_to_golden > 0 and momentum > -40:
                return False, None, f"LAST CHANCE HOLD: Loss ${abs(current_profit):.2f} | Golden in {hours_to_golden}h - waiting for recovery"
            return True, ExitReason.POSITION_LIMIT, f"[S/L] Position loss limit: ${current_profit:.2f} (50% of ${self.max_loss_per_trade:.2f})"

        # === CHECK 5: STALL DETECTION ===
        # Jika harga tidak bergerak (stall) terlalu lama dengan loss
        if len(guard.profit_history) >= 10:
            recent_range = max(guard.profit_history[-10:]) - min(guard.profit_history[-10:])
            if recent_range < 3 and current_profit < -15:  # Stall dengan loss
                guard.stall_count += 1
                if guard.stall_count >= 5:
                    return True, ExitReason.TREND_REVERSAL, f"[STALL] Stalled with loss ${current_profit:.2f} - cutting"

        # === CHECK 6: DAILY LOSS LIMIT ===
        potential_daily_loss = self._state.daily_loss + abs(min(0, current_profit))
        if potential_daily_loss >= self.max_daily_loss_usd:
            return True, ExitReason.DAILY_LIMIT, f"[LIMIT] Would exceed daily loss limit"

        # === CHECK 7: WEEKEND CLOSE ===
        now = datetime.now(WIB)
        if now.weekday() == 4 and now.hour >= 4:  # Friday after 4 AM WIB
            if current_profit > 0:
                return True, ExitReason.WEEKEND_CLOSE, f"[WEEKEND] Weekend close - profit ${current_profit:.2f}"
            elif current_profit > -10:
                return True, ExitReason.WEEKEND_CLOSE, f"[WEEKEND] Weekend close - small loss ${current_profit:.2f}"

        # === CHECK 8: TIME-BASED EXIT (NEW) ===
        # Close trades yang stuck terlalu lama tanpa progress
        trade_duration_hours = (now - guard.entry_time).total_seconds() / 3600

        # 4+ jam tanpa profit berarti = exit
        if trade_duration_hours >= 4 and current_profit < 5:
            if current_profit >= 0:
                return True, ExitReason.TAKE_PROFIT, f"[TIMEOUT] Closing breakeven/small profit after {trade_duration_hours:.1f}h"
            elif current_profit > -15:
                return True, ExitReason.TREND_REVERSAL, f"[TIMEOUT] Closing small loss ${current_profit:.2f} after {trade_duration_hours:.1f}h"

        # Maximum 6 jam untuk any trade
        if trade_duration_hours >= 6:
            return True, ExitReason.TREND_REVERSAL, f"[MAX TIME] Position open {trade_duration_hours:.1f}h - forcing close"

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
        max_loss_per_trade_percent=1.0,     # S/L 1% per trade (software)
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
