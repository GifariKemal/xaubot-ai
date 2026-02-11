"""
Momentum Persistence Detector - Deteksi apakah momentum akan continue atau reverse
Menggunakan velocity/acceleration history untuk predict persistence
"""

import numpy as np
from typing import List, Tuple, Dict
from loguru import logger


class MomentumPersistence:
    """
    Analisis persistence (kekuatan berkelanjutan) dari momentum trading.

    Skor tinggi (>0.7) = Momentum kuat, likely continue -> HOLD position
    Skor rendah (<0.3) = Momentum lemah, likely reverse -> EXIT position

    Features analyzed:
    1. Velocity trend consistency (all positive/negative)
    2. Velocity increasing/decreasing pattern
    3. Acceleration stability (low variance = stable momentum)
    4. Momentum duration (how long momentum has persisted)
    """

    def __init__(self, lookback_periods: int = 5):
        """
        Args:
            lookback_periods: Number of recent samples to analyze (default: 5 = 30 seconds)
        """
        self.lookback = lookback_periods
        self.high_threshold = 0.7  # Persistence > 0.7 = strong, HOLD
        self.low_threshold = 0.3   # Persistence < 0.3 = weak, EXIT

    def calculate_persistence_score(
        self,
        velocity_history: List[float],
        acceleration_history: List[float],
        profit_history: List[float] = None
    ) -> float:
        """
        Hitung momentum persistence score (0-1).

        Args:
            velocity_history: Recent velocity values ($/second)
            acceleration_history: Recent acceleration values ($/second²)
            profit_history: Recent profit values (optional, for trend analysis)

        Returns:
            Persistence score 0.0-1.0
            - 1.0 = Very persistent (strong momentum, HOLD)
            - 0.5 = Neutral
            - 0.0 = Reversing (EXIT)

        Example:
            >>> persistence = MomentumPersistence()
            >>> score = persistence.calculate_persistence_score(
            ...     velocity_history=[0.08, 0.09, 0.10, 0.12, 0.13],  # Increasing!
            ...     acceleration_history=[0.001, 0.001, 0.001, 0.001, 0.001]  # Stable
            ... )
            >>> print(f"Persistence: {score:.2f}")  # Should be high (~0.9)
        """
        if len(velocity_history) < 3 or len(acceleration_history) < 3:
            return 0.5  # Neutral if insufficient data

        # Get recent samples
        recent_vels = velocity_history[-self.lookback:]
        recent_accels = acceleration_history[-self.lookback:]

        score = 0.0

        # === COMPONENT 1: Velocity Direction Consistency (40%) ===
        # All positive or all negative = consistent
        all_positive = all(v > 0 for v in recent_vels)
        all_negative = all(v < 0 for v in recent_vels)

        if all_positive or all_negative:
            score += 0.4
        else:
            # Mixed signs = weak momentum
            positive_ratio = sum(1 for v in recent_vels if v > 0) / len(recent_vels)
            score += abs(positive_ratio - 0.5) * 0.8  # Max 0.4 if all one sign

        # === COMPONENT 2: Velocity Trend (30%) ===
        # Increasing velocity = strengthening momentum
        # Decreasing velocity = weakening momentum

        # Check if velocity magnitude is increasing
        vel_magnitudes = [abs(v) for v in recent_vels]
        increasing_count = sum(
            1 for i in range(1, len(vel_magnitudes))
            if vel_magnitudes[i] > vel_magnitudes[i-1]
        )
        increasing_ratio = increasing_count / (len(vel_magnitudes) - 1)

        if increasing_ratio > 0.6:  # Mostly increasing
            score += 0.3
        elif increasing_ratio > 0.4:  # Mixed
            score += 0.15
        # else: decreasing, no points

        # === COMPONENT 3: Acceleration Stability (30%) ===
        # Low variance = stable momentum (predictable)
        # High variance = erratic movement (unpredictable)
        accel_std = np.std(recent_accels)

        if accel_std < 0.001:  # Very stable
            score += 0.3
        elif accel_std < 0.003:  # Moderately stable
            score += 0.2
        elif accel_std < 0.005:  # Slightly unstable
            score += 0.1
        # else: very unstable, no points

        # Normalize to 0-1
        return min(max(score, 0.0), 1.0)

    def analyze_momentum_quality(
        self,
        velocity_history: List[float],
        acceleration_history: List[float],
        current_profit: float
    ) -> Dict[str, any]:
        """
        Analisis komprehensif kualitas momentum.

        Returns dict dengan:
        - persistence_score: Overall score (0-1)
        - trend: "strengthening", "weakening", "stable", "reversing"
        - recommendation: "HOLD", "CONSIDER_EXIT", "EXIT"
        - components: Breakdown of score components
        """
        persistence = self.calculate_persistence_score(
            velocity_history, acceleration_history
        )

        # Determine trend
        if len(velocity_history) >= 3:
            recent_vels = velocity_history[-3:]
            if all(abs(recent_vels[i]) > abs(recent_vels[i-1]) for i in range(1, len(recent_vels))):
                trend = "strengthening"
            elif all(abs(recent_vels[i]) < abs(recent_vels[i-1]) for i in range(1, len(recent_vels))):
                trend = "weakening"
            elif len(velocity_history) >= 2 and \
                 (recent_vels[-1] * recent_vels[-2]) < 0:  # Sign flip
                trend = "reversing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        # Recommendation based on persistence + trend
        if persistence > self.high_threshold and trend in ["strengthening", "stable"]:
            recommendation = "HOLD"
        elif persistence < self.low_threshold or trend == "reversing":
            recommendation = "EXIT"
        else:
            recommendation = "CONSIDER_EXIT"

        # Component breakdown
        recent_vels = velocity_history[-self.lookback:]
        recent_accels = acceleration_history[-self.lookback:]

        components = {
            "direction_consistency": 1.0 if all(v > 0 for v in recent_vels) or all(v < 0 for v in recent_vels) else 0.5,
            "trend_strength": abs(np.mean(recent_vels)),
            "acceleration_stability": 1.0 / (1.0 + np.std(recent_accels) * 100),  # Inverse of std
            "sample_count": len(velocity_history)
        }

        return {
            "persistence_score": persistence,
            "trend": trend,
            "recommendation": recommendation,
            "components": components
        }

    def should_raise_exit_threshold(
        self,
        velocity_history: List[float],
        acceleration_history: List[float],
        current_profit: float,
        base_threshold: float = 0.85
    ) -> Tuple[bool, float, str]:
        """
        Tentukan apakah exit threshold harus dinaikkan karena momentum kuat.

        Args:
            velocity_history: Recent velocity values
            acceleration_history: Recent acceleration values
            current_profit: Current profit ($)
            base_threshold: Base fuzzy exit threshold

        Returns:
            (should_raise, new_threshold, reason)

        Example:
            >>> persistence = MomentumPersistence()
            >>> should_raise, new_threshold, reason = persistence.should_raise_exit_threshold(
            ...     velocity_history=[0.10, 0.11, 0.12, 0.13, 0.14],  # Strong increasing
            ...     acceleration_history=[0.001] * 5,  # Stable
            ...     current_profit=2.0,
            ...     base_threshold=0.85
            ... )
            >>> print(f"Raise: {should_raise}, New: {new_threshold:.0%}")
            Raise: True, New: 95%
        """
        analysis = self.analyze_momentum_quality(
            velocity_history, acceleration_history, current_profit
        )

        persistence = analysis["persistence_score"]
        trend = analysis["trend"]

        should_raise = False
        new_threshold = base_threshold
        reason = ""

        # HIGH PERSISTENCE + STRENGTHENING = Raise threshold significantly
        if persistence > 0.8 and trend == "strengthening":
            should_raise = True
            new_threshold = min(base_threshold + 0.10, 0.98)
            reason = f"Very strong momentum (persistence={persistence:.0%}, {trend})"

        # MODERATE PERSISTENCE + STABLE = Raise threshold slightly
        elif persistence > 0.7 and trend in ["strengthening", "stable"]:
            should_raise = True
            new_threshold = min(base_threshold + 0.05, 0.95)
            reason = f"Strong momentum (persistence={persistence:.0%}, {trend})"

        # LOW PERSISTENCE or REVERSING = Keep or lower threshold
        elif persistence < 0.3 or trend == "reversing":
            should_raise = False
            new_threshold = max(base_threshold - 0.05, 0.70)
            reason = f"Weak/reversing momentum (persistence={persistence:.0%}, {trend})"

        else:
            reason = f"Neutral momentum (persistence={persistence:.0%})"

        return should_raise, new_threshold, reason

    def detect_momentum_reversal(
        self,
        velocity_history: List[float],
        min_samples: int = 3
    ) -> Tuple[bool, str]:
        """
        Deteksi reversal cepat dalam momentum (danger signal).

        Returns:
            (is_reversing, reason)

        Example momentum reversal patterns:
        - Velocity sign flip: [+0.05, +0.03, -0.02] -> reversing!
        - Rapid deceleration: [+0.10, +0.08, +0.03, +0.01] -> reversing!
        """
        if len(velocity_history) < min_samples:
            return False, "Insufficient data"

        recent = velocity_history[-min_samples:]

        # Pattern 1: Sign flip (positive -> negative or vice versa)
        if len(recent) >= 2:
            signs = [1 if v > 0 else -1 if v < 0 else 0 for v in recent]
            if signs[-1] != signs[0] and signs[-1] != 0 and signs[0] != 0:
                return True, f"Momentum sign flip: {signs[0]} -> {signs[-1]}"

        # Pattern 2: Rapid deceleration (magnitude dropping >50% in 3 samples)
        if len(recent) >= 3:
            magnitudes = [abs(v) for v in recent]
            if magnitudes[0] > 0.05:  # Only if initial velocity significant
                decel_ratio = magnitudes[-1] / magnitudes[0]
                if decel_ratio < 0.5:
                    return True, f"Rapid deceleration: {decel_ratio:.0%} of initial velocity"

        # Pattern 3: Consistent deceleration (all decreasing)
        if len(recent) >= 3:
            magnitudes = [abs(v) for v in recent]
            if all(magnitudes[i] < magnitudes[i-1] for i in range(1, len(magnitudes))):
                return True, "Consistent deceleration trend"

        return False, "No reversal detected"


if __name__ == "__main__":
    # Test cases
    persistence = MomentumPersistence()

    # Test 1: Strong persistent momentum (Trade #161613468 at exit)
    print("=== Test 1: Strong Persistent Momentum ===")
    vel_history = [0.0827, 0.0411, 0.0273, 0.0433, 0.1335]  # Increasing
    accel_history = [0.0004, 0.0004, 0.0001, 0.0005, 0.0017]  # Accelerating

    score = persistence.calculate_persistence_score(vel_history, accel_history)
    print(f"Persistence Score: {score:.2f}")

    analysis = persistence.analyze_momentum_quality(vel_history, accel_history, 0.05)
    print(f"Trend: {analysis['trend']}")
    print(f"Recommendation: {analysis['recommendation']}")

    should_raise, new_thresh, reason = persistence.should_raise_exit_threshold(
        vel_history, accel_history, 0.05, base_threshold=0.90
    )
    print(f"Raise Threshold: {should_raise} -> {new_thresh:.0%}")
    print(f"Reason: {reason}\n")

    # Test 2: Reversing momentum
    print("=== Test 2: Reversing Momentum ===")
    vel_history_rev = [0.08, 0.05, 0.02, -0.01, -0.03]  # Sign flip!
    accel_history_rev = [0.001, 0.0005, 0.0, -0.0005, -0.001]

    score_rev = persistence.calculate_persistence_score(vel_history_rev, accel_history_rev)
    print(f"Persistence Score: {score_rev:.2f}")

    is_reversing, reason = persistence.detect_momentum_reversal(vel_history_rev)
    print(f"Reversing: {is_reversing}")
    print(f"Reason: {reason}\n")

    # Test 3: Stable momentum
    print("=== Test 3: Stable Momentum ===")
    vel_history_stable = [0.05, 0.05, 0.05, 0.05, 0.05]
    accel_history_stable = [0.0, 0.0, 0.0, 0.0, 0.0]

    score_stable = persistence.calculate_persistence_score(vel_history_stable, accel_history_stable)
    print(f"Persistence Score: {score_stable:.2f}")

    should_raise, new_thresh, reason = persistence.should_raise_exit_threshold(
        vel_history_stable, accel_history_stable, 3.0, base_threshold=0.85
    )
    print(f"Raise Threshold: {should_raise} -> {new_thresh:.0%}")
    print(f"Reason: {reason}")
