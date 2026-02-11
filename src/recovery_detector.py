"""
Recovery Strength Detector - Deteksi kekuatan recovery dari loss
Khusus untuk trade yang recovering dari drawdown
"""

import numpy as np
from typing import List, Tuple, Dict
from loguru import logger


class RecoveryDetector:
    """
    Analisis kekuatan recovery dari loss positions.

    Scenario: Trade went to -$6.39, now at $0.05
    Question: Apakah recovery akan continue ke profit besar, atau stop di sini?

    Strong recovery indicators:
    1. High recovery percentage (>80% from peak loss)
    2. Fast recovery velocity (>0.05 $/s average)
    3. Sustained recovery (not just spike)
    4. Accelerating recovery (getting faster)
    """

    def __init__(self):
        self.strong_recovery_threshold = 0.8  # 80% recovery from loss
        self.fast_recovery_velocity = 0.05    # $/second
        self.min_recovery_samples = 5         # Min data points untuk validate

    def analyze_recovery_strength(
        self,
        profit_history: List[float],
        peak_loss: float,
        velocity_history: List[float] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Analisis apakah recovery dari loss cukup kuat untuk continue.

        Args:
            profit_history: Recent profit values
            peak_loss: Peak (worst) loss achieved (negative value)
            velocity_history: Optional velocity history for trend analysis

        Returns:
            (is_strong_recovery, metrics_dict)

        Example:
            >>> detector = RecoveryDetector()
            >>> is_strong, metrics = detector.analyze_recovery_strength(
            ...     profit_history=[-6.39, -5.20, -4.35, -2.99, 0.05],
            ...     peak_loss=-6.39
            ... )
            >>> print(f"Strong: {is_strong}, Recovery: {metrics['recovery_pct']:.0%}")
            Strong: True, Recovery: 101%
        """
        if not profit_history or len(profit_history) < 2:
            return False, {"reason": "Insufficient data"}

        current_profit = profit_history[-1]

        # Can't analyze recovery if never was in loss
        if peak_loss >= 0:
            return False, {"reason": "No loss to recover from"}

        # 1. Recovery Percentage
        # From peak_loss (-6.39) to current (0.05) = 6.44 improvement
        # Recovery % = 6.44 / 6.39 = 100.78%
        recovery_amount = current_profit - peak_loss
        recovery_pct = recovery_amount / abs(peak_loss)

        # 2. Recovery Velocity (average over last N samples)
        recovery_samples = []
        for i in range(len(profit_history) - 1, 0, -1):
            if profit_history[i] > peak_loss:
                recovery_samples.append(profit_history[i])
            else:
                break  # Stop when we hit the loss zone

        if len(recovery_samples) < self.min_recovery_samples:
            return False, {
                "reason": "Recovery too brief",
                "samples": len(recovery_samples),
                "recovery_pct": recovery_pct
            }

        # Calculate average recovery velocity
        recovery_deltas = [
            recovery_samples[i] - recovery_samples[i-1]
            for i in range(1, len(recovery_samples))
        ]
        avg_recovery_vel = np.mean(recovery_deltas) if recovery_deltas else 0.0

        # 3. Recovery Acceleration (is it speeding up?)
        # Compare first half vs second half velocity
        if len(recovery_deltas) >= 4:
            mid = len(recovery_deltas) // 2
            first_half_vel = np.mean(recovery_deltas[:mid])
            second_half_vel = np.mean(recovery_deltas[mid:])
            is_accelerating = second_half_vel > first_half_vel
        else:
            is_accelerating = False

        # 4. Recovery Consistency (not erratic)
        recovery_std = np.std(recovery_deltas) if len(recovery_deltas) > 1 else 0.0
        is_consistent = recovery_std < 0.5  # Low variance

        # === DECISION LOGIC ===
        is_strong = False

        # Strong recovery criteria:
        if (
            recovery_pct > self.strong_recovery_threshold and  # >80% recovered
            avg_recovery_vel > self.fast_recovery_velocity and  # Fast recovery
            len(recovery_samples) >= self.min_recovery_samples  # Sustained
        ):
            is_strong = True

        # OR: Accelerating recovery even if not 80% yet
        elif (
            recovery_pct > 0.5 and       # At least 50% recovered
            is_accelerating and          # Getting faster
            avg_recovery_vel > 0.03      # Reasonable speed
        ):
            is_strong = True

        # Metrics
        metrics = {
            "recovery_pct": recovery_pct,
            "recovery_amount": recovery_amount,
            "avg_recovery_vel": avg_recovery_vel,
            "recovery_samples": len(recovery_samples),
            "is_accelerating": is_accelerating,
            "is_consistent": is_consistent,
            "recovery_std": recovery_std
        }

        return is_strong, metrics

    def should_extend_grace_period(
        self,
        profit_history: List[float],
        peak_loss: float,
        current_grace_seconds: int,
        max_grace_seconds: int = 720  # 12 minutes
    ) -> Tuple[bool, int, str]:
        """
        Tentukan apakah grace period harus diperpanjang untuk recovery.

        Args:
            profit_history: Recent profit values
            peak_loss: Peak loss value
            current_grace_seconds: Current grace period
            max_grace_seconds: Maximum allowed grace

        Returns:
            (should_extend, new_grace_seconds, reason)
        """
        is_strong, metrics = self.analyze_recovery_strength(
            profit_history, peak_loss
        )

        if not is_strong:
            return False, current_grace_seconds, "Weak recovery, no extension"

        # Calculate extension based on recovery strength
        recovery_pct = metrics.get("recovery_pct", 0)
        recovery_vel = metrics.get("avg_recovery_vel", 0)

        # Strong recovery = extend grace significantly
        if recovery_pct > 0.8 and recovery_vel > 0.08:
            extension = 180  # +3 minutes
            reason = f"Very strong recovery ({recovery_pct:.0%} at {recovery_vel:.4f}$/s)"
        elif recovery_pct > 0.6 and recovery_vel > 0.05:
            extension = 120  # +2 minutes
            reason = f"Strong recovery ({recovery_pct:.0%})"
        else:
            extension = 60   # +1 minute
            reason = f"Moderate recovery ({recovery_pct:.0%})"

        new_grace = min(current_grace_seconds + extension, max_grace_seconds)

        return True, new_grace, reason

    def predict_breakeven_time(
        self,
        profit_history: List[float],
        velocity_history: List[float]
    ) -> Tuple[int, float]:
        """
        Estimasi berapa lama lagi untuk mencapai breakeven.

        Args:
            profit_history: Recent profit values
            velocity_history: Recent velocity values

        Returns:
            (seconds_to_breakeven, confidence)
        """
        if not profit_history or not velocity_history:
            return -1, 0.0

        current_profit = profit_history[-1]

        # Already at breakeven or profit
        if current_profit >= 0:
            return 0, 1.0

        # Calculate average velocity during recovery
        avg_vel = np.mean(velocity_history[-10:])  # Last 10 samples

        # Not recovering (velocity negative or near zero)
        if avg_vel <= 0.01:
            return -1, 0.0  # Can't predict

        # Time to breakeven = distance / velocity
        distance_to_be = abs(current_profit)
        time_to_be = distance_to_be / avg_vel

        # Confidence based on velocity stability
        vel_std = np.std(velocity_history[-10:])
        confidence = max(0, 1.0 - vel_std * 10)  # Lower std = higher confidence

        return int(time_to_be), confidence

    def get_recovery_recommendation(
        self,
        profit_history: List[float],
        peak_loss: float,
        velocity_history: List[float] = None,
        current_exit_threshold: float = 0.85
    ) -> Tuple[str, float, str]:
        """
        Rekomendasi lengkap untuk recovering position.

        Returns:
            (action, adjusted_threshold, reason)
            action: "HOLD_STRONG", "HOLD_WEAK", "EXIT"
        """
        is_strong, metrics = self.analyze_recovery_strength(
            profit_history, peak_loss, velocity_history
        )

        current_profit = profit_history[-1]
        recovery_pct = metrics.get("recovery_pct", 0)
        recovery_vel = metrics.get("avg_recovery_vel", 0)

        # HOLD_STRONG: Very strong recovery, raise threshold
        if is_strong and current_profit >= 0:
            # Recovered to profit - strong signal
            adjusted_threshold = min(current_exit_threshold + 0.15, 0.98)
            action = "HOLD_STRONG"
            reason = (
                f"Strong recovery to profit ({recovery_pct:.0%} from ${peak_loss:.2f}, "
                f"vel={recovery_vel:.4f}$/s)"
            )

        elif is_strong and current_profit < 0:
            # Still in loss but strong recovery - give more time
            adjusted_threshold = min(current_exit_threshold + 0.10, 0.95)
            action = "HOLD_STRONG"
            reason = (
                f"Strong recovery in progress ({recovery_pct:.0%}, "
                f"vel={recovery_vel:.4f}$/s)"
            )

        # HOLD_WEAK: Moderate recovery
        elif recovery_pct > 0.5 and recovery_vel > 0.03:
            adjusted_threshold = min(current_exit_threshold + 0.05, 0.90)
            action = "HOLD_WEAK"
            reason = f"Moderate recovery ({recovery_pct:.0%})"

        # EXIT: Weak or stalled recovery
        else:
            adjusted_threshold = max(current_exit_threshold - 0.05, 0.70)
            action = "EXIT"
            reason = f"Weak recovery ({recovery_pct:.0%}, vel={recovery_vel:.4f}$/s)"

        return action, adjusted_threshold, reason


if __name__ == "__main__":
    # Test cases
    detector = RecoveryDetector()

    # Test 1: Strong recovery (Trade #161613468)
    print("=== Test 1: Strong Recovery from -$6.39 to $0.05 ===")
    profit_history = [-6.39, -5.67, -5.20, -4.35, -2.99, -0.50, 0.05]
    peak_loss = -6.39

    is_strong, metrics = detector.analyze_recovery_strength(profit_history, peak_loss)
    print(f"Is Strong Recovery: {is_strong}")
    print(f"Recovery %: {metrics['recovery_pct']:.0%}")
    print(f"Recovery Velocity: {metrics['avg_recovery_vel']:.4f} $/s")
    print(f"Samples: {metrics['recovery_samples']}")
    print(f"Accelerating: {metrics['is_accelerating']}\n")

    # Test 2: Recovery recommendation
    print("=== Test 2: Recovery Recommendation ===")
    velocity_history = [0.0058, 0.0250, 0.0827, 0.0411, 0.1335]

    action, adj_threshold, reason = detector.get_recovery_recommendation(
        profit_history, peak_loss, velocity_history, current_exit_threshold=0.90
    )
    print(f"Action: {action}")
    print(f"Adjusted Threshold: {adj_threshold:.0%}")
    print(f"Reason: {reason}\n")

    # Test 3: Breakeven prediction
    print("=== Test 3: Breakeven Time Prediction ===")
    profit_history_loss = [-3.0, -2.5, -2.0, -1.5, -1.0]
    velocity_history_loss = [0.05, 0.05, 0.05, 0.05, 0.05]

    time_to_be, confidence = detector.predict_breakeven_time(
        profit_history_loss, velocity_history_loss
    )
    print(f"Time to Breakeven: {time_to_be}s ({time_to_be//60}m {time_to_be%60}s)")
    print(f"Confidence: {confidence:.0%}")

    # Test 4: Weak recovery
    print("\n=== Test 4: Weak/Stalled Recovery ===")
    profit_history_weak = [-5.0, -4.8, -4.7, -4.6, -4.5]  # Slow
    peak_loss_weak = -5.0

    is_strong_weak, metrics_weak = detector.analyze_recovery_strength(
        profit_history_weak, peak_loss_weak
    )
    print(f"Is Strong Recovery: {is_strong_weak}")
    print(f"Recovery %: {metrics_weak['recovery_pct']:.0%}")
    print(f"Recovery Velocity: {metrics_weak['avg_recovery_vel']:.4f} $/s")
