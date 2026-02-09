# Profit Momentum Tracker Integration Guide

## 📋 Overview

Profit Momentum Tracker adalah sistem monitoring real-time yang menganalisa pergerakan profit per 500ms untuk mendeteksi timing exit yang optimal. Sistem ini mencegah early exit sambil melindungi profit dari reversal.

## 🎯 Problem yang Diselesaikan

1. **Early Cut** - Bot sering exit terlalu cepat ketika profit masih bisa grow
2. **Late Exit** - Bot terlambat exit ketika profit sudah mulai reverse
3. **Tidak Ada Visibility** - Tidak ada tracking real-time profit pattern per ticket
4. **Exit Decision Tidak Optimal** - Exit hanya based on fixed levels (TP/SL), tidak adaptive

## 🔧 How It Works

### 1. Profit Tracking (500ms interval)
```python
tracker.update(ticket, current_profit, current_price)
```
- Track profit history dalam deque (max 40 samples = 20 detik)
- Calculate peak profit
- Track time in profit

### 2. Momentum Metrics Calculation
```python
metrics = tracker.calculate_metrics(ticket)
```

Metrics yang dihitung:
- **Velocity** - Rate of profit change ($/s)
- **Acceleration** - Rate of velocity change ($/s²)
- **Peak Profit** - Maximum profit achieved
- **Drawdown from Peak** - % dan $ amount
- **Momentum Direction** - INCREASING/STABLE/DECREASING
- **Stagnation Count** - Consecutive low-velocity samples

### 3. Exit Conditions

#### A. Velocity Reversal
```
Trigger: velocity < -0.5 $/s
Protection: Only if profit >= $5 OR time_in_profit >= 10s
```
Detect ketika profit mulai turn negative (momentum reversal).

#### B. Strong Deceleration
```
Trigger: acceleration < -1.0 $/s²
Protection: Only if profit >= $5
```
Detect ketika profit growth slowing down significantly.

#### C. Peak Drawdown
```
Trigger: drawdown > 40% from peak
Protection: Only if peak >= $10
```
Exit ketika profit pulled back signifikan dari peak.

#### D. Stagnation
```
Trigger: 8 consecutive samples with velocity < 0.1 $/s
Protection: Only if profit >= $5 AND time_in_profit >= 10s
```
Exit ketika profit flat terlalu lama (might reverse soon).

## 🚀 Integration Steps

### Step 1: Import & Initialize in `main_live.py`

```python
from src.profit_momentum_tracker import ProfitMomentumTracker

class TradingBot:
    def __init__(self, ...):
        # ... existing init code ...

        # Initialize Profit Momentum Tracker (NEW)
        self.momentum_tracker = ProfitMomentumTracker(
            velocity_reversal_threshold=-0.5,  # Exit if velocity < -0.5 $/s
            deceleration_threshold=-1.0,       # Exit if accel < -1.0 $/s²
            stagnation_threshold=0.1,          # Velocity < 0.1 $/s = stagnant
            stagnation_count_max=8,            # 8 samples = 4 seconds
            peak_drawdown_threshold=40.0,      # Exit if 40% drawdown from peak
            min_peak_to_protect=10.0,          # Protect peaks > $10
            min_profit_for_momentum_exit=5.0,  # Don't exit on momentum if < $5
            grace_period_seconds=10.0,         # Min 10s in profit before momentum exit
            enable_logging=True,
        )

        # Pass tracker to Position Manager
        self.position_manager = SmartPositionManager(
            # ... existing params ...
            momentum_tracker=self.momentum_tracker,  # NEW
            enable_momentum_exit=True,               # NEW
        )
```

### Step 2: Add Monitoring Loop in Trading Loop

Tambahkan async task untuk monitor profit setiap 500ms:

```python
async def _monitor_positions_momentum(self):
    """
    Monitor open positions momentum every 500ms.
    Updates profit tracker for real-time analysis.
    """
    while self.running:
        try:
            # Get open positions
            positions_df = self.mt5.get_positions()

            if len(positions_df) > 0:
                # Update momentum tracker for each position
                for row in positions_df.iter_rows(named=True):
                    ticket = row["ticket"]
                    profit = row.get("profit", 0.0)
                    current_price = row.get("price_current", 0.0)

                    # Update tracker
                    self.momentum_tracker.update(ticket, profit, current_price)

                    # Optional: Log metrics every 2 seconds
                    if self._should_log_momentum(ticket):
                        summary = self.momentum_tracker.get_position_summary(ticket)
                        if summary:
                            logger.debug(
                                f"#{ticket} | Profit: ${summary['current_profit']:.2f} | "
                                f"Peak: ${summary['peak_profit']:.2f} | "
                                f"Velocity: {summary['velocity']:.2f} $/s | "
                                f"Momentum: {summary['momentum']}"
                            )

            # Wait 500ms before next update
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Momentum monitoring error: {e}")
            await asyncio.sleep(0.5)


def _should_log_momentum(self, ticket: int) -> bool:
    """Throttle logging to every 2 seconds per ticket."""
    if not hasattr(self, "_last_momentum_log"):
        self._last_momentum_log = {}

    now = time.time()
    last_log = self._last_momentum_log.get(ticket, 0)

    if now - last_log >= 2.0:  # Log every 2 seconds
        self._last_momentum_log[ticket] = now
        return True
    return False
```

### Step 3: Start Monitoring Task in Main Loop

```python
async def run(self):
    """Main trading loop."""
    self.running = True

    # Start background tasks
    tasks = [
        asyncio.create_task(self._trading_loop()),
        asyncio.create_task(self._monitor_positions_momentum()),  # NEW
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Trading error: {e}")
    finally:
        self.running = False
```

### Step 4: Cleanup on Position Close

Already handled automatically in `SmartPositionManager`:

```python
# In position_manager.py - execute_actions()
if close_result["success"]:
    logger.info(f"CLOSED #{action.ticket}: {action.reason}")
    self._peak_profits.pop(action.ticket, None)
    # Clean up momentum tracker
    if self.momentum_tracker:
        self.momentum_tracker.cleanup_position(action.ticket)  # ✅ Auto cleanup
```

## 📊 Usage Examples

### Example 1: Check Exit Signal Manually
```python
should_exit, reason = tracker.should_exit(ticket, current_profit)
if should_exit:
    logger.warning(f"Exit signal for #{ticket}: {reason}")
    # Close position
```

### Example 2: Get Position Summary
```python
summary = tracker.get_position_summary(ticket)
print(f"Ticket: {summary['ticket']}")
print(f"Current Profit: ${summary['current_profit']:.2f}")
print(f"Peak Profit: ${summary['peak_profit']:.2f}")
print(f"Velocity: {summary['velocity']:.2f} $/s")
print(f"Momentum: {summary['momentum']}")
print(f"Drawdown: {summary['drawdown_pct']:.1f}%")
```

### Example 3: Get All Summaries
```python
all_summaries = tracker.get_all_summaries()
for summary in all_summaries:
    logger.info(
        f"#{summary['ticket']}: ${summary['current_profit']:.2f} | "
        f"Peak: ${summary['peak_profit']:.2f} | "
        f"Vel: {summary['velocity']:.2f} $/s"
    )
```

## 🧪 Testing

Run test simulations:

```bash
python tests/test_profit_momentum.py
```

Test scenarios:
1. **Pattern 1**: Steady growth → reversal (should exit at ~90-94% of peak)
2. **Pattern 2**: Quick spike → sharp reversal (should exit fast on velocity reversal)
3. **Pattern 3**: Healthy trend (should NOT exit, maintain position)

## ⚙️ Tuning Parameters

### Conservative (Protect Profit Aggressively)
```python
ProfitMomentumTracker(
    velocity_reversal_threshold=-0.3,  # Exit sooner
    peak_drawdown_threshold=30.0,      # Exit on smaller drawdown
    grace_period_seconds=5.0,          # Shorter grace period
)
```

### Aggressive (Let Profit Run)
```python
ProfitMomentumTracker(
    velocity_reversal_threshold=-1.0,  # Exit later
    peak_drawdown_threshold=50.0,      # Allow larger drawdown
    grace_period_seconds=15.0,         # Longer grace period
)
```

### Balanced (Default - Recommended)
```python
ProfitMomentumTracker(
    velocity_reversal_threshold=-0.5,
    deceleration_threshold=-1.0,
    peak_drawdown_threshold=40.0,
    grace_period_seconds=10.0,
)
```

## 📈 Expected Benefits

1. **Better Exit Timing** - Exit based on momentum analysis, not just fixed levels
2. **Avoid Early Cuts** - Grace period & minimum profit protection
3. **Protect from Reversals** - Detect momentum changes before profit turns to loss
4. **Real-time Visibility** - Log profit patterns per ticket
5. **Adaptive Exits** - Respond to actual market movement, not just static TP/SL

## 🔍 Monitoring & Logging

Enable detailed logging:
```python
tracker = ProfitMomentumTracker(enable_logging=True)
```

Log output examples:
```
14:32:10 | WARNING | #123456 Momentum reversal detected (velocity: -0.8 $/s, profit: $45.20)
14:32:10 | WARNING | 🚨 EXIT SIGNAL at $45.20: Momentum Exit: Momentum reversal detected
14:32:10 | SUCCESS | ✅ Exit Summary: Peak $50.00 → Exit $45.20 (9.6% from peak)
```

## 🎯 Integration Checklist

- [ ] Import `ProfitMomentumTracker` in `main_live.py`
- [ ] Initialize tracker with tuned parameters
- [ ] Pass tracker to `SmartPositionManager`
- [ ] Add `_monitor_positions_momentum()` method
- [ ] Start monitoring task in `run()` method
- [ ] Test with `test_profit_momentum.py`
- [ ] Monitor logs during live trading
- [ ] Tune parameters based on results

## 📝 Notes

- Tracker menggunakan **deque with maxlen=40** (20 detik history)
- **Minimal 6 samples** (3 detik) required untuk analisis
- **Grace period** mencegah exit terlalu cepat di awal profit
- **Peak drawdown** hanya aktif jika peak >= threshold
- **Velocity & acceleration** calculated from recent samples untuk responsiveness

## 🚨 Important Warnings

1. **Jangan disable grace period** - Bisa cause excessive early exits
2. **Jangan set threshold terlalu ketat** - Bisa exit di normal volatility
3. **Monitor backtest results** - Tune parameters based on historical performance
4. **Test di simulation dulu** - Jangan langsung live trading

## 📚 Related Files

- `src/profit_momentum_tracker.py` - Main tracker implementation
- `src/position_manager.py` - Integration with exit logic
- `tests/test_profit_momentum.py` - Simulation tests
- `main_live.py` - Main integration point
