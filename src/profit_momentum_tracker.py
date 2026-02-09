"""
Profit Momentum Tracker
========================
Monitors real-time profit movements to detect optimal exit timing.

NOTE: Velocity/acceleration logic has been ported to PositionGuard in
smart_risk_manager.py (Feb 2026). PositionGuard now tracks velocity,
acceleration, and stagnation inline with its existing momentum scoring.
This module is kept available for potential future sub-second monitoring
use cases but is NOT actively used by the live trading loop.

Features:
- Track profit velocity (rate of change)
- Detect profit acceleration/deceleration
- Identify momentum reversals
- Prevent early exits while protecting from losses
- Smart exit timing based on profit patterns

Usage:
    tracker = ProfitMomentumTracker()

    # In trading loop (every 500ms):
    tracker.update(ticket, current_profit, current_price)

    # Check exit signal:
    should_exit, reason = tracker.should_exit(ticket)
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from loguru import logger


@dataclass
class ProfitSnapshot:
    """Single profit measurement at a point in time."""
    timestamp: float
    profit: float
    price: float


@dataclass
class MomentumMetrics:
    """Calculated momentum metrics for a position."""
    velocity: float  # $/second (profit change rate)
    acceleration: float  # $/s² (velocity change rate)
    peak_profit: float  # Maximum profit achieved
    drawdown_from_peak: float  # % drawdown from peak
    drawdown_amount: float  # $ amount of drawdown
    stagnation_count: int  # Consecutive samples with low velocity
    momentum_direction: str  # "INCREASING", "STABLE", "DECREASING"
    time_in_profit: float  # Seconds since first profitable
    sample_count: int  # Number of samples collected


@dataclass
class PositionMomentum:
    """Track momentum for a single position."""
    ticket: int
    entry_time: float = field(default_factory=time.time)
    first_profit_time: Optional[float] = None
    history: deque = field(default_factory=lambda: deque(maxlen=40))  # ~20 seconds at 500ms
    peak_profit: float = 0.0
    peak_profit_time: float = 0.0
    total_samples: int = 0


class ProfitMomentumTracker:
    """
    Tracks profit momentum for all open positions.

    Analyzes profit patterns to determine optimal exit timing:
    - Exit when momentum is reversing (profit turning to loss)
    - Exit when deceleration is significant (growth slowing)
    - Protect profits from reversal
    - Avoid premature exits during healthy momentum
    """

    def __init__(
        self,
        # Velocity thresholds
        velocity_reversal_threshold: float = -0.5,  # Exit if velocity < -0.5 $/s
        deceleration_threshold: float = -1.0,  # Exit if acceleration < -1.0 $/s²
        stagnation_threshold: float = 0.1,  # Velocity < 0.1 $/s = stagnant
        stagnation_count_max: int = 8,  # Exit after 8 consecutive stagnant samples (4s)

        # Drawdown protection
        peak_drawdown_threshold: float = 40.0,  # Exit if drawdown > 40% from peak
        min_peak_to_protect: float = 10.0,  # Only protect peaks > $10

        # Anti-early-exit protection
        min_profit_for_momentum_exit: float = 5.0,  # Don't exit on momentum if profit < $5
        grace_period_seconds: float = 10.0,  # Minimum 10s in profit before momentum exit
        min_samples_required: int = 6,  # Minimum 6 samples (3s) before analyzing

        # Logging
        enable_logging: bool = True,
    ):
        self.velocity_reversal_threshold = velocity_reversal_threshold
        self.deceleration_threshold = deceleration_threshold
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_count_max = stagnation_count_max
        self.peak_drawdown_threshold = peak_drawdown_threshold
        self.min_peak_to_protect = min_peak_to_protect
        self.min_profit_for_momentum_exit = min_profit_for_momentum_exit
        self.grace_period_seconds = grace_period_seconds
        self.min_samples_required = min_samples_required
        self.enable_logging = enable_logging

        # Track positions
        self.positions: Dict[int, PositionMomentum] = {}

    def update(self, ticket: int, profit: float, price: float) -> None:
        """
        Update profit tracking for a position.

        Args:
            ticket: MT5 ticket number
            profit: Current profit in $
            price: Current market price
        """
        now = time.time()

        # Initialize position tracking if new
        if ticket not in self.positions:
            self.positions[ticket] = PositionMomentum(
                ticket=ticket,
                entry_time=now,
            )

        pos = self.positions[ticket]

        # Track first time in profit
        if profit > 0 and pos.first_profit_time is None:
            pos.first_profit_time = now

        # Update peak profit
        if profit > pos.peak_profit:
            pos.peak_profit = profit
            pos.peak_profit_time = now

        # Add snapshot to history
        snapshot = ProfitSnapshot(
            timestamp=now,
            profit=profit,
            price=price,
        )
        pos.history.append(snapshot)
        pos.total_samples += 1

    def calculate_metrics(self, ticket: int) -> Optional[MomentumMetrics]:
        """
        Calculate momentum metrics for a position.

        Args:
            ticket: MT5 ticket number

        Returns:
            MomentumMetrics or None if insufficient data
        """
        if ticket not in self.positions:
            return None

        pos = self.positions[ticket]

        # Need at least 2 samples to calculate velocity
        if len(pos.history) < 2:
            return None

        # Convert history to arrays
        history = list(pos.history)
        times = np.array([s.timestamp for s in history])
        profits = np.array([s.profit for s in history])

        # Calculate velocity (profit change rate)
        # Use recent samples for velocity (last 5 samples = 2.5s)
        if len(history) >= 5:
            recent_times = times[-5:]
            recent_profits = profits[-5:]
            dt = recent_times[-1] - recent_times[0]
            if dt > 0:
                velocity = (recent_profits[-1] - recent_profits[0]) / dt
            else:
                velocity = 0.0
        else:
            dt = times[-1] - times[0]
            velocity = (profits[-1] - profits[0]) / dt if dt > 0 else 0.0

        # Calculate acceleration (velocity change rate)
        # Need at least 10 samples for acceleration
        acceleration = 0.0
        if len(history) >= 10:
            # Split into two halves and compare velocities
            mid = len(history) // 2

            # First half velocity
            t1 = times[:mid]
            p1 = profits[:mid]
            dt1 = t1[-1] - t1[0]
            v1 = (p1[-1] - p1[0]) / dt1 if dt1 > 0 else 0.0

            # Second half velocity
            t2 = times[mid:]
            p2 = profits[mid:]
            dt2 = t2[-1] - t2[0]
            v2 = (p2[-1] - p2[0]) / dt2 if dt2 > 0 else 0.0

            # Acceleration = change in velocity
            dt_total = times[-1] - times[0]
            acceleration = (v2 - v1) / dt_total if dt_total > 0 else 0.0

        # Determine momentum direction
        if velocity > self.stagnation_threshold:
            momentum_direction = "INCREASING"
        elif velocity < -self.stagnation_threshold:
            momentum_direction = "DECREASING"
        else:
            momentum_direction = "STABLE"

        # Count stagnation (consecutive samples with low velocity)
        stagnation_count = 0
        if len(history) >= 4:
            for i in range(len(history) - 1, max(len(history) - 9, 0), -1):
                if i > 0:
                    dt = times[i] - times[i-1]
                    dp = profits[i] - profits[i-1]
                    v = dp / dt if dt > 0 else 0.0
                    if abs(v) < self.stagnation_threshold:
                        stagnation_count += 1
                    else:
                        break

        # Calculate drawdown from peak
        current_profit = profits[-1]
        drawdown_amount = pos.peak_profit - current_profit
        drawdown_pct = (drawdown_amount / pos.peak_profit * 100) if pos.peak_profit > 0 else 0.0

        # Time in profit
        time_in_profit = 0.0
        if pos.first_profit_time is not None:
            time_in_profit = time.time() - pos.first_profit_time

        return MomentumMetrics(
            velocity=velocity,
            acceleration=acceleration,
            peak_profit=pos.peak_profit,
            drawdown_from_peak=drawdown_pct,
            drawdown_amount=drawdown_amount,
            stagnation_count=stagnation_count,
            momentum_direction=momentum_direction,
            time_in_profit=time_in_profit,
            sample_count=len(pos.history),
        )

    def should_exit(self, ticket: int, current_profit: float) -> Tuple[bool, Optional[str]]:
        """
        Determine if position should exit based on momentum analysis.

        Args:
            ticket: MT5 ticket number
            current_profit: Current profit in $

        Returns:
            (should_exit: bool, reason: str or None)
        """
        metrics = self.calculate_metrics(ticket)

        if metrics is None:
            return False, None

        # Not enough samples yet
        if metrics.sample_count < self.min_samples_required:
            return False, None

        pos = self.positions[ticket]

        # === EXIT CONDITIONS ===

        # 1. VELOCITY REVERSAL - Profit momentum turning negative
        if metrics.velocity < self.velocity_reversal_threshold:
            # Anti-early-exit: only if profit is significant or past grace period
            if current_profit >= self.min_profit_for_momentum_exit or \
               metrics.time_in_profit >= self.grace_period_seconds:
                reason = (
                    f"Momentum reversal detected (velocity: {metrics.velocity:.2f} $/s, "
                    f"profit: ${current_profit:.2f})"
                )
                if self.enable_logging:
                    logger.warning(f"#{ticket} {reason}")
                return True, reason

        # 2. STRONG DECELERATION - Profit growth slowing significantly
        if metrics.acceleration < self.deceleration_threshold:
            # Only exit if already in decent profit
            if current_profit >= self.min_profit_for_momentum_exit:
                reason = (
                    f"Strong deceleration (accel: {metrics.acceleration:.2f} $/s², "
                    f"velocity: {metrics.velocity:.2f} $/s)"
                )
                if self.enable_logging:
                    logger.warning(f"#{ticket} {reason}")
                return True, reason

        # 3. PEAK DRAWDOWN - Profit pulled back significantly from peak
        if metrics.peak_profit >= self.min_peak_to_protect:
            if metrics.drawdown_from_peak >= self.peak_drawdown_threshold:
                reason = (
                    f"Peak drawdown exceeded (peak: ${metrics.peak_profit:.2f}, "
                    f"current: ${current_profit:.2f}, drawdown: {metrics.drawdown_from_peak:.1f}%)"
                )
                if self.enable_logging:
                    logger.warning(f"#{ticket} {reason}")
                return True, reason

        # 4. STAGNATION - Profit flat for too long (might reverse soon)
        if metrics.stagnation_count >= self.stagnation_count_max:
            # Only exit if in profit and past grace period
            if current_profit >= self.min_profit_for_momentum_exit and \
               metrics.time_in_profit >= self.grace_period_seconds:
                reason = (
                    f"Profit stagnation ({metrics.stagnation_count} samples, "
                    f"${current_profit:.2f} profit)"
                )
                if self.enable_logging:
                    logger.info(f"#{ticket} {reason}")
                return True, reason

        # No exit signal
        return False, None

    def get_position_summary(self, ticket: int) -> Optional[Dict]:
        """
        Get detailed summary for a position.

        Args:
            ticket: MT5 ticket number

        Returns:
            Dictionary with position metrics or None
        """
        metrics = self.calculate_metrics(ticket)

        if metrics is None:
            return None

        pos = self.positions[ticket]
        history = list(pos.history)

        return {
            "ticket": ticket,
            "samples": metrics.sample_count,
            "time_in_profit": metrics.time_in_profit,
            "current_profit": history[-1].profit if history else 0.0,
            "peak_profit": metrics.peak_profit,
            "velocity": metrics.velocity,
            "acceleration": metrics.acceleration,
            "momentum": metrics.momentum_direction,
            "stagnation_count": metrics.stagnation_count,
            "drawdown_pct": metrics.drawdown_from_peak,
            "drawdown_amount": metrics.drawdown_amount,
        }

    def cleanup_position(self, ticket: int) -> None:
        """
        Remove position tracking when closed.

        Args:
            ticket: MT5 ticket number
        """
        if ticket in self.positions:
            if self.enable_logging:
                summary = self.get_position_summary(ticket)
                if summary:
                    logger.info(
                        f"Cleanup #{ticket} | "
                        f"Peak: ${summary['peak_profit']:.2f} | "
                        f"Samples: {summary['samples']} | "
                        f"Time in profit: {summary['time_in_profit']:.1f}s"
                    )
            del self.positions[ticket]

    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all tracked positions."""
        summaries = []
        for ticket in self.positions:
            summary = self.get_position_summary(ticket)
            if summary:
                summaries.append(summary)
        return summaries
