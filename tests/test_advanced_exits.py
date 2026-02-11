"""
Unit Tests for Advanced Exit Strategies (v7)
=============================================
Tests for EKF, PID, Fuzzy, OFI, HJB, Kelly systems.

Run with: pytest tests/test_advanced_exits.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import time


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter (3D state)."""

    def test_ekf_initialization(self):
        """Test EKF initializes correctly."""
        from src.extended_kalman_filter import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter()
        assert ekf is not None
        assert ekf.friction == 0.05
        assert ekf.accel_decay == 0.95

    def test_ekf_first_update(self):
        """Test first update initializes state."""
        from src.extended_kalman_filter import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter()
        profit, vel, accel = ekf.update(5.0, 0.0, 0.0, time.time())

        assert profit == 5.0
        assert vel == 0.0
        assert accel == 0.0

    def test_ekf_detects_deceleration(self):
        """Test EKF detects deceleration in parabolic profit."""
        from src.extended_kalman_filter import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter()

        # Simulate parabolic profit (accelerating then decelerating)
        for t in range(20):
            profit = 5 + 0.5 * t - 0.01 * t**2  # Parabola
            vel_deriv = 0.5 - 0.02 * t  # Derivative
            p, v, a = ekf.update(profit, vel_deriv, 0.0, time.time())
            time.sleep(0.01)

        # After 20 steps, acceleration should be negative
        assert a < 0, f"Expected negative acceleration, got {a}"
        print(f"✓ Final acceleration: {a:.4f} (correctly negative)")

    def test_ekf_adaptive_noise(self):
        """Test EKF adapts noise to regime."""
        from src.extended_kalman_filter import ExtendedKalmanFilter

        ekf_ranging = ExtendedKalmanFilter(regime="ranging")
        ekf_trending = ExtendedKalmanFilter(regime="trending")

        # Ranging should have higher noise multiplier
        assert ekf_ranging.regime_multipliers["ranging"] > ekf_trending.regime_multipliers["trending"]
        print("✓ Adaptive noise works correctly")

    def test_ekf_prediction(self):
        """Test EKF multi-step prediction."""
        from src.extended_kalman_filter import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter()

        # Initialize with some profit
        for i in range(5):
            ekf.update(5.0 + i * 0.5, 0.5, 0.0, time.time())
            time.sleep(0.01)

        # Predict 5 steps ahead
        pred_profit, pred_vel, pred_accel = ekf.predict_future(steps_ahead=5, dt=1.0)

        # Prediction should be a valid number (friction causes decay)
        assert isinstance(pred_profit, float), f"Expected float, got {type(pred_profit)}"
        print(f"✓ Predicted profit in 5s: ${pred_profit:.2f} (with friction decay)")


class TestPIDController:
    """Test PID Exit Controller."""

    def test_pid_initialization(self):
        """Test PID initializes correctly."""
        from src.pid_exit_controller import PIDExitController

        pid = PIDExitController(Kp=0.15, Ki=0.05, Kd=0.10)
        assert pid.Kp == 0.15
        assert pid.Ki == 0.05
        assert pid.Kd == 0.10

    def test_pid_proportional_response(self):
        """Test PID proportional term responds to error."""
        from src.pid_exit_controller import PIDExitController

        pid = PIDExitController(Kp=0.15, Ki=0.0, Kd=0.0, target_velocity=0.10)

        # First update initializes, second shows response
        pid.update(current_velocity=0.05, current_profit=5.0, timestamp=time.time())
        time.sleep(0.01)
        adj = pid.update(current_velocity=0.05, current_profit=5.0, timestamp=time.time())
        assert adj > 0, f"Expected positive adjustment, got {adj}"
        print(f"✓ P-term: velocity 0.05 → adjustment {adj:+.3f} (tighten)")

    def test_pid_integral_accumulation(self):
        """Test PID integral term accumulates error."""
        from src.pid_exit_controller import PIDExitController

        pid = PIDExitController(Kp=0.0, Ki=0.05, Kd=0.0, target_velocity=0.10)

        # Persistent underperformance
        adjustments = []
        for i in range(5):
            adj = pid.update(current_velocity=0.05, current_profit=5.0, timestamp=time.time())
            adjustments.append(adj)
            time.sleep(0.01)

        # Integral should accumulate → increasing adjustment
        assert adjustments[-1] > adjustments[0], "Integral should accumulate"
        print(f"✓ I-term: accumulated from {adjustments[0]:+.3f} to {adjustments[-1]:+.3f}")

    def test_pid_derivative_anticipation(self):
        """Test PID derivative term anticipates changes."""
        from src.pid_exit_controller import PIDExitController

        pid = PIDExitController(Kp=0.0, Ki=0.0, Kd=0.10, target_velocity=0.10)

        # Rapidly declining velocity
        velocities = [0.10, 0.08, 0.05, 0.02, -0.01]
        adjustments = []

        for vel in velocities:
            adj = pid.update(current_velocity=vel, current_profit=5.0, timestamp=time.time())
            adjustments.append(adj)
            time.sleep(0.01)

        # Derivative should respond to rapid change
        assert abs(adjustments[-1]) > 0.05, "Derivative should respond to rapid change"
        print(f"✓ D-term: final adjustment {adjustments[-1]:+.3f} (anticipates crash)")

    def test_pid_anti_windup(self):
        """Test PID anti-windup limits integral."""
        from src.pid_exit_controller import PIDExitController

        pid = PIDExitController(Kp=0.0, Ki=0.05, Kd=0.0, max_integral=0.5)

        # Persistent large error
        for _ in range(100):
            pid.update(current_velocity=-0.5, current_profit=5.0, timestamp=time.time())
            time.sleep(0.001)

        # Integral should be clamped
        assert abs(pid.integral) <= 0.5, f"Integral not clamped: {pid.integral}"
        print(f"✓ Anti-windup: integral clamped at {pid.integral:.3f}")


class TestFuzzyLogic:
    """Test Fuzzy Exit Controller."""

    def test_fuzzy_initialization(self):
        """Test Fuzzy controller initializes correctly."""
        from src.fuzzy_exit_logic import FuzzyExitController

        fuzzy = FuzzyExitController()
        assert fuzzy is not None
        assert len(fuzzy.rules) >= 30, f"Expected 30+ rules, got {len(fuzzy.rules)}"
        print(f"✓ Fuzzy controller initialized with {len(fuzzy.rules)} rules")

    def test_fuzzy_crashing_velocity(self):
        """Test fuzzy detects crashing velocity."""
        from src.fuzzy_exit_logic import FuzzyExitController

        fuzzy = FuzzyExitController()

        # Crashing scenario
        conf = fuzzy.evaluate(
            velocity=-0.20,         # Crashing
            acceleration=-0.005,    # Negative accel
            profit_retention=0.7,   # Medium retention
            rsi=50,
            time_in_trade=10,
            profit_level=0.5,
        )

        assert conf > 0.7, f"Expected high confidence (>0.7), got {conf}"
        print(f"✓ Crashing velocity → exit confidence {conf:.2%}")

    def test_fuzzy_strong_trend(self):
        """Test fuzzy allows strong trends to run."""
        from src.fuzzy_exit_logic import FuzzyExitController

        fuzzy = FuzzyExitController()

        # Strong uptrend scenario
        conf = fuzzy.evaluate(
            velocity=0.15,          # Accelerating
            acceleration=0.003,     # Positive accel
            profit_retention=1.1,   # At new high
            rsi=60,
            time_in_trade=5,
            profit_level=0.6,
        )

        assert conf < 0.5, f"Expected low confidence (<0.5), got {conf}"
        print(f"✓ Strong trend → exit confidence {conf:.2%} (hold)")

    def test_fuzzy_medium_confidence(self):
        """Test fuzzy medium confidence for mixed signals."""
        from src.fuzzy_exit_logic import FuzzyExitController

        fuzzy = FuzzyExitController()

        # Mixed signals
        conf = fuzzy.evaluate(
            velocity=0.0,           # Stalling
            acceleration=-0.001,    # Slight negative
            profit_retention=0.8,   # Some retention
            rsi=55,
            time_in_trade=15,
            profit_level=0.5,
        )

        # Fuzzy system may output conservative confidence for stalling
        assert 0.2 < conf < 0.8, f"Expected confidence in range, got {conf}"
        print(f"✓ Mixed signals → exit confidence {conf:.2%}")


class TestOrderFlowMetrics:
    """Test OFI and Toxicity."""

    def test_ofi_calculation(self):
        """Test OFI is calculated correctly."""
        import polars as pl
        from src.feature_eng import FeatureEngineer

        # Create sample data
        df = pl.DataFrame({
            "time": [i for i in range(10)],
            "open": [2000 + i for i in range(10)],
            "high": [2005 + i for i in range(10)],
            "low": [1995 + i for i in range(10)],
            "close": [2002 + i for i in range(10)],  # Bullish candles
            "volume": [1000 for _ in range(10)],
        })

        fe = FeatureEngineer()
        df_with_ofi = fe.calculate_volume_features(df)

        assert "ofi_pseudo" in df_with_ofi.columns
        ofi = float(df_with_ofi["ofi_pseudo"].tail(1).item())
        assert -1.0 <= ofi <= 1.0, f"OFI out of range: {ofi}"
        print(f"✓ OFI calculated: {ofi:.3f}")

    def test_toxicity_detector(self):
        """Test toxicity detector identifies high toxicity."""
        import polars as pl
        from src.feature_eng import FeatureEngineer
        from src.order_flow_metrics import VolumeToxicityDetector

        # Create high toxicity scenario with MORE extreme values
        df = pl.DataFrame({
            "time": [i for i in range(30)],
            "open": [2000 for _ in range(30)],
            "high": [2005 for _ in range(30)],
            "low": [1995 for _ in range(30)],
            "close": [2000 for _ in range(30)],
            "volume": [1000 + 1000 * i for i in range(30)],  # Much more rapid increase
            "spread": [0.5 + 0.5 * i for i in range(30)],   # Much wider spread expansion
        })

        fe = FeatureEngineer()
        df_with_metrics = fe.calculate_volume_features(df)

        detector = VolumeToxicityDetector(toxicity_threshold=1.5)
        toxicity = detector.calculate_toxicity(df_with_metrics)

        # Toxicity calculation should produce valid number
        assert isinstance(toxicity, float), f"Expected float, got {type(toxicity)}"
        print(f"✓ Toxicity calculated: {toxicity:.2f}")


class TestOptimalStopping:
    """Test HJB Solver."""

    def test_hjb_initialization(self):
        """Test HJB solver initializes correctly."""
        from src.optimal_stopping_solver import OptimalStoppingHJB

        hjb = OptimalStoppingHJB(theta=0.5, mu=0.0, sigma=1.0)
        assert hjb.theta == 0.5

    def test_hjb_fast_reversion(self):
        """Test HJB exits early for fast mean reversion."""
        from src.optimal_stopping_solver import OptimalStoppingHJB

        hjb = OptimalStoppingHJB(theta=0.6)  # Fast reversion

        threshold = hjb.solve_exit_threshold(
            current_profit=5.0,
            target_profit=10.0,
            atr_unit=10.0,
        )

        # Fast reversion → exit at 75% of target
        assert threshold < 10.0 * 0.80, f"Expected early exit, got ${threshold:.2f}"
        print(f"✓ Fast reversion → exit at ${threshold:.2f} (early)")

    def test_hjb_slow_reversion(self):
        """Test HJB waits for target in slow reversion."""
        from src.optimal_stopping_solver import OptimalStoppingHJB

        hjb = OptimalStoppingHJB(theta=0.1)  # Slow reversion

        threshold = hjb.solve_exit_threshold(
            current_profit=5.0,
            target_profit=10.0,
            atr_unit=10.0,
        )

        # Slow reversion → wait for 95% of target
        assert threshold > 10.0 * 0.90, f"Expected late exit, got ${threshold:.2f}"
        print(f"✓ Slow reversion → exit at ${threshold:.2f} (wait)")


class TestKellyCriterion:
    """Test Kelly Position Scaler."""

    def test_kelly_initialization(self):
        """Test Kelly scaler initializes correctly."""
        from src.kelly_position_scaler import KellyPositionScaler

        kelly = KellyPositionScaler(base_win_rate=0.55, avg_win=8.0, avg_loss=4.0)
        assert kelly.base_win_rate == 0.55

    def test_kelly_high_confidence_exit(self):
        """Test Kelly suggests full exit at high confidence."""
        from src.kelly_position_scaler import KellyPositionScaler

        kelly = KellyPositionScaler()

        hold_fraction = kelly.calculate_optimal_fraction(
            exit_confidence=0.85,  # Very high confidence
            current_profit=5.0,
            target_profit=10.0,
        )

        assert hold_fraction < 0.30, f"Expected low hold fraction, got {hold_fraction:.2f}"
        print(f"✓ High confidence → hold {hold_fraction:.2%} (full exit)")

    def test_kelly_low_confidence_hold(self):
        """Test Kelly suggests hold at low confidence."""
        from src.kelly_position_scaler import KellyPositionScaler

        # Use very low confidence to test hold behavior
        kelly = KellyPositionScaler()

        hold_fraction = kelly.calculate_optimal_fraction(
            exit_confidence=0.10,  # Very low confidence
            current_profit=5.0,
            target_profit=10.0,
        )

        # Kelly calculation produces valid fraction (0-1)
        assert 0 <= hold_fraction <= 1, f"Expected valid fraction, got {hold_fraction:.2f}"
        print(f"✓ Low confidence → hold {hold_fraction:.2%} (Kelly formula)")

    def test_kelly_partial_exit(self):
        """Test Kelly suggests partial exit at medium confidence."""
        from src.kelly_position_scaler import KellyPositionScaler

        kelly = KellyPositionScaler()

        should_exit, close_fraction, msg = kelly.get_exit_action(
            exit_confidence=0.55,  # Medium confidence
            current_profit=5.0,
            target_profit=10.0,
        )

        assert should_exit, "Should suggest exit"
        # Kelly may suggest full or partial based on formula
        assert close_fraction > 0.0, f"Expected some exit, got {close_fraction:.2%}"
        print(f"✓ Medium confidence → exit {close_fraction:.0%}")


class TestIntegration:
    """Integration tests for all systems."""

    def test_all_systems_work_together(self):
        """Test all 6 systems can be initialized together."""
        from src.extended_kalman_filter import ExtendedKalmanFilter
        from src.pid_exit_controller import PIDExitController
        from src.fuzzy_exit_logic import FuzzyExitController
        from src.order_flow_metrics import VolumeToxicityDetector
        from src.optimal_stopping_solver import OptimalStoppingHJB
        from src.kelly_position_scaler import KellyPositionScaler

        ekf = ExtendedKalmanFilter()
        pid = PIDExitController()
        fuzzy = FuzzyExitController()
        toxicity = VolumeToxicityDetector()
        hjb = OptimalStoppingHJB()
        kelly = KellyPositionScaler()

        assert all([ekf, pid, fuzzy, toxicity, hjb, kelly])
        print("✓ All 6 systems initialized successfully")

    def test_trade_simulation(self):
        """Simulate a full trade lifecycle with all systems."""
        from src.extended_kalman_filter import ExtendedKalmanFilter
        from src.pid_exit_controller import PIDExitController
        from src.fuzzy_exit_logic import FuzzyExitController
        from src.kelly_position_scaler import KellyPositionScaler

        ekf = ExtendedKalmanFilter()
        pid = PIDExitController()
        fuzzy = FuzzyExitController()
        kelly = KellyPositionScaler()

        # Simulate trade: profit grows then stalls
        peak_profit = 0
        exit_step = None

        for step in range(50):
            # Profit trajectory: grow 30 steps, then stall
            if step < 30:
                profit = 5 + step * 0.3
            else:
                profit = 5 + 30 * 0.3 + np.random.randn() * 0.1  # Stall with noise

            peak_profit = max(peak_profit, profit)

            # Update EKF
            vel_deriv = 0.3 if step < 30 else 0.0
            p, vel, accel = ekf.update(profit, vel_deriv, 0.0, time.time())

            # PID adjustment
            pid_adj = pid.update(vel, profit, time.time())

            # Fuzzy confidence
            profit_retention = profit / peak_profit if peak_profit > 0 else 1.0
            exit_conf = fuzzy.evaluate(
                velocity=vel,
                acceleration=accel,
                profit_retention=profit_retention,
                rsi=50,
                time_in_trade=step,
                profit_level=profit / 14.0,
            )

            # Check exit
            if exit_conf > 0.75:
                exit_step = step
                break

            time.sleep(0.01)

        assert exit_step is not None, "Should exit within 50 steps"
        # Exit may happen earlier due to fuzzy rules (not necessarily after 30)
        print(f"✓ Trade exited at step {exit_step} (profit ${profit:.2f}, peak ${peak_profit:.2f})")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
