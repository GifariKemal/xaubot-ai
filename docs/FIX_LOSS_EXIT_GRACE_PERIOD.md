# FIX: Loss Exit Grace Period (v0.1.2)

## Problem
Trade #161699163 exit terlalu cepat (18 detik) meskipun exit decision ternyata correct.
User concern: Sistem tidak memberikan kesempatan recovery untuk micro swings.

## Root Cause
1. **No grace period for loss trades** - langsung fuzzy check setelah entry
2. **Profit retention bug** - loss setelah profit kecil dianggap "collapsed" (trigger 95% exit)

## Proposed Fix

### FIX 1: Grace Period untuk Loss Trades
```python
# Line ~1397 smart_risk_manager.py
# BEFORE:
if exit_confidence > 0.75:
    return True, ExitReason.POSITION_LIMIT, ...

# AFTER:
# Grace period: 60-120s tergantung regime
grace_period_sec = {
    "ranging": 120,
    "volatile": 90,
    "trending": 60
}.get(regime, 90)

time_since_entry = time.time() - guard.entry_time
if time_since_entry < grace_period_sec:
    # Suppress fuzzy exit during grace period
    logger.info(f"[GRACE PERIOD] Loss fuzzy={exit_confidence:.2%} suppressed (t={time_since_entry:.0f}s < {grace_period_sec}s)")
else:
    if exit_confidence > 0.75:
        return True, ExitReason.POSITION_LIMIT, ...
```

### FIX 2: Profit Retention Fix untuk Small Loss After Small Profit
```python
# fuzzy_exit_logic.py - evaluate() method
# BEFORE:
profit_retention_val = current_profit / peak_profit

# AFTER:
if current_profit < 0 and 0 < peak_profit < 3.0:
    # Small loss after small profit = micro swing, bukan collapse
    profit_retention_val = 0.50  # Medium retention (bukan collapsed)
else:
    profit_retention_val = current_profit / peak_profit
```

## Expected Impact
- Avg trade duration: 18s → 60-120s (lebih reasonable)
- False early exits: -30% (grace period filtering)
- Recovery opportunities: Lebih banyak micro swings yang bisa recovery

## Testing
- Backtest with grace period enabled
- Monitor next 10 trades: avg duration harus >60s

## Version
- Bump to v0.1.2 (PATCH - bug fix)
