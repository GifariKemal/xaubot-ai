# XAUBot AI v0.6.0 FIXED - Backtest

Backtest dengan implementasi rekomendasi dari **Profesor AI & Ilmuwan Algoritma Trading**.

## 📋 Fixes Implemented

### PRIORITY 1: Tiered Fuzzy Exit Thresholds
**Problem:** 75% wins adalah micro profits (<$1) karena fuzzy threshold fixed 90%
**Solution:** Dynamic thresholds based on profit tier

```python
Micro  (<$1):  70% threshold  # Exit early (was 90%)
Small  ($1-3): 75% threshold  # Small protection (was 85%)
Medium ($3-8): 85% threshold  # Hold for more (was 85%)
Large  (>$8):  90% threshold  # Maximize (was 80%)
```

**Expected Impact:** Micro profits 75% → <20% (-73%)

### PRIORITY 2: Trajectory Confidence Calibration
**Problem:** Predictions $10-66 but actual $0.28-0.58 (error 95%+)
**Solution:** Conservative predictions with regime penalty

```python
# Regime penalties
ranging:  0.4  # 60% discount (low predictability)
volatile: 0.6  # 40% discount (high noise)
trending: 0.9  # 10% discount (best predictability)

# Add 95% CI uncertainty
prediction_std = abs(acceleration) * horizon * 5
conservative = calibrated - 1.96 * prediction_std
```

**Expected Impact:** Prediction error 95% → <40% (-58%)

### PRIORITY 3: Session Filter
**Problem:** Sydney/Tokyo (00:00-10:00) generated micro profits only
**Solution:** DISABLE low-volatility sessions

```python
Sydney/Tokyo (00:00-10:00): BLOCKED
Late NY      (22:00-01:00): BLOCKED
London       (14:00-20:00): ALLOWED (best performance)
```

**Expected Impact:** Avg profit/trade +50%+ (filtering bad trades)

### PRIORITY 4: Unicode Fix
**Problem:** Log corruption from emojis (⏳, →, ✓)
**Solution:** ASCII-only logging

**Impact:** Stable logs, easier debugging

### PRIORITY 5: Tighter Stop-Loss
**Problem:** Max loss -$34.70 (17x avg win)
**Solution:** Reduce max loss per trade

```python
Max Loss: $50 → $25
```

**Expected Impact:** Avg loss $20.91 → $8-12 (-60%)

---

## 🎯 Expected Results

| Metric | Before | Target | Improvement |
|--------|--------|--------|-------------|
| Avg Win | $4.07 | $8-12 | +100-200% |
| RR Ratio | 1:5 | 1.5:1 | +650% |
| Micro Profits | 75% | <20% | -73% |
| Win Rate | 57% | 62-65% | +8% |
| Sharpe Ratio | 0.8 | 1.5+ | +87% |

---

## 🚀 Usage

### Quick Run (90 days)
```bash
cd "C:/Users/Administrator/Videos/Smart Automatic Trading BOT + AI/backtests/v0.6.0_fixed"
python run_backtest.py
```

### Custom Period
```bash
python run_backtest.py --days 30
python run_backtest.py --days 180
```

### Save Results to CSV
```bash
python run_backtest.py --days 90 --save
```

---

## 📁 Files

- `backtest_v0_6_0_fixed.py` - Main backtest engine with fixes
- `run_backtest.py` - Quick runner script
- `README.md` - This file
- `results_*.csv` - Backtest results (when using --save)

---

## 📊 Understanding Results

### PASS Criteria
- ✅ Avg Win ≥ $8
- ✅ RR Ratio ≤ 1.5:1 (avg loss ≤ 1.5x avg win)
- ✅ Micro Profits < 20%
- ✅ Win Rate 62-65%
- ✅ Sharpe Ratio ≥ 1.5

### What to Look For
1. **Micro Profit %** - Should be dramatically lower (<20% vs 75%)
2. **RR Ratio** - Should be balanced (1.5:1 or better)
3. **Sharpe Ratio** - Should exceed 1.5 (risk-adjusted returns)
4. **Exit Reasons** - Fuzzy exits should dominate (not trajectory overrides)

---

## 🔬 Technical Details

### Exit Logic Flow
```
1. Take Profit Hit        → Exit (ideal)
2. Max Loss ($25)         → Exit (protection)
3. Fuzzy Confidence > X%  → Exit (tiered threshold)
   - <$1:  70% threshold
   - $1-3: 75% threshold
   - $3-8: 85% threshold
   - >$8:  90% threshold
4. ML Reversal (>65%)     → Exit (signal change)
5. Timeout (8 hours)      → Exit (stuck trade)
```

### Fuzzy Confidence Calculation
```python
Components (0.0-1.0):
- Velocity (40%):   crashing=-0.10 → conf +0.40
- Retention (30%):  <70% from peak → conf +0.30
- Acceleration (20%): <-0.002 → conf +0.20
- Time (10%):       >6h → conf +0.10
```

---

## 🎓 Next Steps

### If Results PASS (meet targets):
1. Apply fixes to `main_live.py`
2. Update `smart_risk_manager.py` with new thresholds
3. Demo account testing (2 weeks)
4. Go live if Sharpe >1.2

### If Results FAIL (below targets):
1. Analyze exit reason distribution
2. Adjust fuzzy thresholds (try 65-85%)
3. Test different session windows
4. Re-run with different parameters

---

**Author:** Profesor AI & Ilmuwan Algoritma Trading
**Date:** 2026-02-11
**Version:** v0.6.0 FIXED
