"""
Trajectory Predictor - Prediksi pergerakan profit masa depan
Menggunakan parabolic motion model untuk forecast profit 1-5 menit ke depan
"""

import numpy as np
from typing import List, Tuple, Dict
from loguru import logger


class TrajectoryPredictor:
    """
    Prediksi trajectory profit menggunakan kinematic equations.

    Model: profit(t) = profit₀ + velocity*t + 0.5*acceleration*t²

    Cocok untuk:
    - Deteksi early exit (jangan close jika prediksi profit tinggi)
    - Validasi exit timing (exit jika prediksi profit turun)
    - Recovery continuation (prediksi apakah recovery akan lanjut)
    """

    def __init__(self):
        self.default_horizons = [60, 180, 300]  # 1m, 3m, 5m (seconds)
        self.confidence_threshold = 0.7  # Minimum confidence untuk pakai prediksi

        # v0.2.0: Regime-based dampening factors (validated from live trades)
        # Trade #161778984: avg over-prediction 7.5x -> need 85% reduction
        # Trade #161850770: predicted profit from loss -> need 70% reduction
        self.dampening_factors = {
            "ranging": 0.20,      # 80% reduction (most conservative)
            "volatile": 0.30,     # 70% reduction (validated: 3-17x over -> 1-5x)
            "medium_volatility": 0.30,  # Same as volatile
            "trending": 0.50,     # 50% reduction (momentum likely continues)
            "normal": 0.30        # Default fallback
        }

    def predict_future_profit(
        self,
        current_profit: float,
        velocity: float,
        acceleration: float,
        horizons: List[int] = None,
        regime: str = "normal"
    ) -> List[float]:
        """
        Prediksi profit di masa depan menggunakan parabolic motion dengan regime dampening.

        Args:
            current_profit: Profit saat ini ($)
            velocity: Profit velocity ($/second)
            acceleration: Profit acceleration ($/second²)
            horizons: List of time horizons dalam seconds (default: [60, 180, 300])
            regime: Market regime for dampening ("ranging"/"volatile"/"trending")

        Returns:
            List of predicted profits untuk setiap horizon (damped)

        Example:
            >>> predictor = TrajectoryPredictor()
            >>> pred_1m, pred_3m, pred_5m = predictor.predict_future_profit(
            ...     current_profit=0.05,
            ...     velocity=0.1335,
            ...     acceleration=0.0017,
            ...     regime="volatile"
            ... )
            >>> print(f"1min: ${pred_1m:.2f}, 3min: ${pred_3m:.2f}")
            1min: $3.34, 3min: $8.22  (damped by 0.30x)
        """
        if horizons is None:
            horizons = self.default_horizons

        # v0.2.0: Get dampening factor based on regime
        dampening = self.dampening_factors.get(regime, 0.30)

        predictions = []
        for dt in horizons:
            # Kinematic equation: s = s₀ + v*t + 0.5*a*t²
            term1 = current_profit
            term2 = velocity * dt
            term3 = 0.5 * acceleration * dt**2
            predicted_profit_raw = term1 + term2 + term3

            # v0.2.1: ASYMMETRIC dampening - only dampen positive growth (optimism)
            # Keep negative growth RAW (crash warnings must stay urgent!)
            growth = term2 + term3
            if growth > 0:
                # Positive growth = over-optimism -> dampen it
                growth_damped = growth * dampening
                dampen_applied = True
            else:
                # Negative growth = crash warning -> keep RAW (urgent!)
                growth_damped = growth
                dampen_applied = False

            predicted_profit = term1 + growth_damped

            # DEBUG v0.2.1: Log calculation with asymmetric dampening (only for 60s)
            if dt == 60:
                dampen_str = f"× {dampening:.2f}" if dampen_applied else "× 1.00 (crash!)"
                logger.info(
                    f"[TRAJ-CALC] {term1:.2f} + ({term2:.2f} + {term3:.2f}) {dampen_str} = "
                    f"{predicted_profit:.2f} (raw: {predicted_profit_raw:.2f}, regime: {regime})"
                )

            predictions.append(predicted_profit)

        return predictions

    def calculate_prediction_confidence(
        self,
        velocity_history: List[float],
        acceleration_history: List[float]
    ) -> float:
        """
        Hitung confidence level prediksi (0-1).

        High confidence jika:
        - Velocity stable (low variance)
        - Acceleration consistent
        - Sufficient data points

        Args:
            velocity_history: List of recent velocity values
            acceleration_history: List of recent acceleration values

        Returns:
            Confidence score 0.0-1.0
        """
        if len(velocity_history) < 3 or len(acceleration_history) < 3:
            return 0.3  # Low confidence if insufficient data

        # 1. Velocity stability (lower std = higher confidence)
        vel_std = np.std(velocity_history[-5:])
        vel_score = max(0, 1.0 - vel_std * 10)  # Normalize

        # 2. Acceleration consistency
        accel_std = np.std(acceleration_history[-5:])
        accel_score = max(0, 1.0 - accel_std * 100)

        # 3. Data sufficiency bonus
        data_score = min(len(velocity_history) / 20, 1.0)  # Max at 20 samples

        # Weighted average
        confidence = vel_score * 0.4 + accel_score * 0.4 + data_score * 0.2
        return min(max(confidence, 0.0), 1.0)

    def should_hold_position(
        self,
        current_profit: float,
        velocity: float,
        acceleration: float,
        min_target: float,
        velocity_history: List[float] = None,
        acceleration_history: List[float] = None,
        regime: str = "normal"
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Rekomendasi apakah HOLD position berdasarkan prediksi (dengan regime dampening).

        Args:
            current_profit: Current profit ($)
            velocity: Current velocity ($/s)
            acceleration: Current acceleration ($/s²)
            min_target: Minimum profit target ($)
            velocity_history: Recent velocity values (optional)
            acceleration_history: Recent acceleration values (optional)
            regime: Market regime for dampening (v0.2.0)

        Returns:
            (should_hold, reason, predictions_dict)

        Example:
            >>> should_hold, reason, preds = predictor.should_hold_position(
            ...     current_profit=0.05,
            ...     velocity=0.1335,
            ...     acceleration=0.0017,
            ...     min_target=3.0,
            ...     regime="volatile"
            ... )
            >>> print(f"Hold: {should_hold}, Reason: {reason}")
            Hold: True, Reason: Predicted $3.34 in 1min (target: $3.00)
        """
        # v0.2.0: Predict 1m, 3m, 5m ahead with regime dampening
        pred_1m, pred_3m, pred_5m = self.predict_future_profit(
            current_profit, velocity, acceleration, regime=regime
        )

        # Calculate confidence (if history provided)
        confidence = 1.0
        if velocity_history and acceleration_history:
            confidence = self.calculate_prediction_confidence(
                velocity_history, acceleration_history
            )

        predictions = {
            'pred_1m': pred_1m,
            'pred_3m': pred_3m,
            'pred_5m': pred_5m,
            'confidence': confidence
        }

        # Decision logic
        should_hold = False
        reason = ""

        # Check if low confidence - don't rely on predictions
        if confidence < self.confidence_threshold:
            reason = f"Low prediction confidence ({confidence:.0%}), use standard logic"
            return False, reason, predictions

        # HOLD if 1-minute prediction exceeds target significantly
        if pred_1m > min_target * 2 and acceleration > 0:
            should_hold = True
            reason = f"Predicted ${pred_1m:.2f} in 1min (target: ${min_target:.2f}, conf: {confidence:.0%})"

        # HOLD if strong acceleration even if current profit low
        elif acceleration > 0.001 and velocity > 0.05 and pred_1m > min_target:
            should_hold = True
            reason = f"Strong acceleration ({acceleration:.4f}), pred ${pred_1m:.2f} > target"

        # HOLD if recovering strongly (negative to positive trajectory)
        elif current_profit < 0 and pred_1m > abs(current_profit) * 0.5:
            should_hold = True
            reason = f"Strong recovery trajectory: ${current_profit:.2f} -> ${pred_1m:.2f}"

        # EXIT if prediction shows decline
        elif pred_1m < current_profit * 0.8 and velocity < 0:
            should_hold = False
            reason = f"Declining trajectory: ${current_profit:.2f} -> ${pred_1m:.2f}"

        else:
            reason = f"Neutral prediction (1m: ${pred_1m:.2f})"

        return should_hold, reason, predictions

    def get_optimal_exit_time(
        self,
        current_profit: float,
        velocity: float,
        acceleration: float,
        tp_target: float
    ) -> Tuple[float, int]:
        """
        Estimasi waktu optimal untuk exit berdasarkan trajectory.

        Args:
            current_profit: Current profit
            velocity: Current velocity
            acceleration: Current acceleration
            tp_target: Take profit target

        Returns:
            (peak_profit, time_to_peak_seconds)

        Example:
            >>> peak, time_to_peak = predictor.get_optimal_exit_time(
            ...     current_profit=5.0,
            ...     velocity=0.08,
            ...     acceleration=-0.002,  # Decelerating
            ...     tp_target=10.0
            ... )
            >>> print(f"Peak at ${peak:.2f} in {time_to_peak}s")
        """
        # For parabolic motion with deceleration:
        # Profit reaches peak when velocity = 0
        # velocity(t) = v₀ + a*t = 0  ->  t = -v₀/a

        if acceleration >= 0:
            # Still accelerating - no peak in near future
            # Estimate based on reaching TP
            if velocity > 0:
                time_to_tp = (tp_target - current_profit) / velocity
                return tp_target, int(time_to_tp)
            else:
                return current_profit, 0

        # Decelerating (acceleration < 0)
        time_to_peak = -velocity / acceleration  # When velocity reaches 0

        # Clamp to reasonable range (0-600 seconds = 10 minutes)
        time_to_peak = max(0, min(time_to_peak, 600))

        # Calculate peak profit
        peak_profit = current_profit + velocity * time_to_peak + 0.5 * acceleration * time_to_peak**2

        return peak_profit, int(time_to_peak)


if __name__ == "__main__":
    # Test cases
    predictor = TrajectoryPredictor()

    # Test 1: Strong upward momentum (Trade #161613468 case)
    print("=== Test 1: Strong Upward Momentum ===")
    should_hold, reason, preds = predictor.should_hold_position(
        current_profit=0.05,
        velocity=0.1335,
        acceleration=0.0017,
        min_target=3.0
    )
    print(f"Should Hold: {should_hold}")
    print(f"Reason: {reason}")
    print(f"Predictions: 1m=${preds['pred_1m']:.2f}, 3m=${preds['pred_3m']:.2f}, 5m=${preds['pred_5m']:.2f}\n")

    # Test 2: Declining trajectory
    print("=== Test 2: Declining Trajectory ===")
    should_hold, reason, preds = predictor.should_hold_position(
        current_profit=5.0,
        velocity=-0.05,
        acceleration=-0.001,
        min_target=3.0
    )
    print(f"Should Hold: {should_hold}")
    print(f"Reason: {reason}")
    print(f"Predictions: 1m=${preds['pred_1m']:.2f}\n")

    # Test 3: Optimal exit time
    print("=== Test 3: Optimal Exit Time ===")
    peak, time_to_peak = predictor.get_optimal_exit_time(
        current_profit=5.0,
        velocity=0.08,
        acceleration=-0.002,
        tp_target=10.0
    )
    print(f"Peak Profit: ${peak:.2f}")
    print(f"Time to Peak: {time_to_peak}s ({time_to_peak//60}m {time_to_peak%60}s)")
