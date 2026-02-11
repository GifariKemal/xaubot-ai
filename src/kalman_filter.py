"""
Kalman Filter for Profit Velocity Smoothing
============================================
Constant-velocity Kalman filter that smooths noisy profit readings
and produces filtered velocity + acceleration estimates.

State vector: [profit, velocity]
Observation:  [profit] (direct measurement)

Tuning:
  - process_noise_velocity=0.01: smooth velocity strongly (suppress single-sample spikes)
  - measurement_noise=0.25: XAUUSD bid/ask noise for 0.01 lot (~$0.25 per tick)
  - Responds to genuine reversal within 2-3 samples (10-15s) while ignoring noise

Author: AI Assistant
"""

import time

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_continuous_white_noise
    import numpy as np
    _FILTERPY_AVAILABLE = True
except ImportError:
    _FILTERPY_AVAILABLE = False


class ProfitKalmanFilter:
    """
    Kalman filter for profit time series.

    Tracks [profit, velocity] state with constant-velocity dynamics.
    F matrix updated per call with actual time delta for accuracy.
    """

    def __init__(
        self,
        process_noise_velocity: float = 0.01,
        measurement_noise: float = 0.25,
    ):
        if not _FILTERPY_AVAILABLE:
            raise ImportError(
                "filterpy not installed. Install with: pip install filterpy"
            )

        self._process_noise_vel = process_noise_velocity
        self._measurement_noise = measurement_noise
        self._last_time: float = 0.0
        self._initialized: bool = False
        self._prev_velocity: float = 0.0

        # Create 2D Kalman filter: state = [profit, velocity]
        self._kf = KalmanFilter(dim_x=2, dim_z=1)

        # Observation matrix: we observe profit directly
        self._kf.H = np.array([[1.0, 0.0]])

        # Measurement noise
        self._kf.R = np.array([[self._measurement_noise]])

        # Initial state covariance (high uncertainty)
        self._kf.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

    def update(self, profit: float, timestamp: float = 0.0) -> tuple:
        """
        Feed a new profit observation and get filtered estimates.

        Args:
            profit: Current profit in USD
            timestamp: time.time() value (0 = use current time)

        Returns:
            (filtered_profit, filtered_velocity, acceleration)
        """
        now = timestamp if timestamp > 0 else time.time()

        if not self._initialized:
            # First observation: initialize state
            self._kf.x = np.array([[profit], [0.0]])
            self._last_time = now
            self._initialized = True
            return profit, 0.0, 0.0

        # Time delta since last update
        dt = now - self._last_time
        if dt <= 0:
            dt = 1.0  # Fallback: assume 1 second
        self._last_time = now

        # Update F matrix (state transition) with actual dt
        self._kf.F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        # Update Q matrix (process noise) scaled by dt
        self._kf.Q = Q_continuous_white_noise(
            dim=2, dt=dt, spectral_density=self._process_noise_vel
        )

        # Predict + update
        self._kf.predict()
        self._kf.update(np.array([[profit]]))

        # Extract filtered state
        filtered_profit = float(self._kf.x[0, 0])
        filtered_velocity = float(self._kf.x[1, 0])

        # Calculate acceleration from velocity change
        acceleration = (filtered_velocity - self._prev_velocity) / dt if dt > 0 else 0.0
        self._prev_velocity = filtered_velocity

        return filtered_profit, filtered_velocity, acceleration

    def reset(self):
        """Reset filter state (e.g., for new trade)."""
        self._initialized = False
        self._last_time = 0.0
        self._prev_velocity = 0.0
        self._kf.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
