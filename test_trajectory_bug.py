#!/usr/bin/env python
"""Test trajectory prediction calculation to find bug."""

# Trade #3 data dari log:
#   10:46:25 | profit=$+0.65 | vel=0.0466$/s | accel=0.0009
#   10:46:31 | Predicted $67.64 in 1min

current_profit = 0.65
velocity = 0.0466  # $/s
acceleration = 0.0009  # $/s²
dt = 60  # seconds

# Manual calculation (kinematic formula):
# profit(t) = p₀ + v*t + 0.5*a*t²

pred_1m_manual = current_profit + velocity * dt + 0.5 * acceleration * dt**2

print("=== TRAJECTORY BUG DIAGNOSIS ===")
print(f"Current profit: ${current_profit:.2f}")
print(f"Velocity: ${velocity:.4f}/s")
print(f"Acceleration: ${acceleration:.4f}/s²")
print(f"Time horizon: {dt}s (1 minute)")
print()
print("--- Manual Calculation ---")
print(f"p₀ = ${current_profit:.2f}")
print(f"v*t = ${velocity:.4f} × {dt} = ${velocity * dt:.2f}")
print(f"0.5*a*t² = 0.5 × ${acceleration:.4f} × {dt}² = ${0.5 * acceleration * dt**2:.2f}")
print(f"Total: ${pred_1m_manual:.2f}")
print()
print("--- Log Value ---")
print(f"Logged prediction: $67.64")
print(f"Error factor: {67.64 / pred_1m_manual:.1f}x")
print()
print("=== TEST WITH TRAJECTORY PREDICTOR ===")

try:
    from src.trajectory_predictor import TrajectoryPredictor
    predictor = TrajectoryPredictor()
    
    pred_1m, pred_3m, pred_5m = predictor.predict_future_profit(
        current_profit, velocity, acceleration
    )
    
    print(f"Predictor output:")
    print(f"  1min: ${pred_1m:.2f}")
    print(f"  3min: ${pred_3m:.2f}")
    print(f"  5min: ${pred_5m:.2f}")
    print()
    
    if abs(pred_1m - pred_1m_manual) < 0.01:
        print("✅ Predictor formula is CORRECT!")
    else:
        print(f"❌ BUG: Predictor gives ${pred_1m:.2f}, manual gives ${pred_1m_manual:.2f}")
        
except Exception as e:
    print(f"Error importing predictor: {e}")
