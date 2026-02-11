# 🚨 Regime Detection Stuck on "Low Volatility"

**Date:** 2026-02-09 19:20 WIB
**Issue:** HMM Regime Detector always shows "Low Volatility"
**Status:** 🔴 MODEL CALIBRATION ISSUE

---

## 📊 THE PROBLEM

Dashboard always shows:
```
Regime: Low Volatility
Volatility: 0.27
Confidence: 100%
```

**Observation:** Regime **NEVER** changes from "Low Volatility" despite market conditions changing.

---

## 🔍 ROOT CAUSE ANALYSIS

### HMM Model Thresholds (dari `models/hmm_regime.pkl`):

```python
State 0 (Low Vol):    0.001039  # Volatility 20-period std
State 1 (Medium Vol): 0.001350  # +0.000311 difference
State 2 (High Vol):   0.001621  # +0.000271 difference
```

**Masalah:**
1. **Threshold terlalu sempit!** Difference antara Low dan High cuma **0.00058** (0.058%)
2. **Gold lebih volatile** dari thresholds ini → selalu fall into "Low" bucket
3. Model di-train dengan data yang **terlalu low volatility** atau old data

### Perbandingan dengan Real Market:

**Gold (XAUUSD) Typical Volatility:**
- **Quiet market:** 0.0005 - 0.0015 (0.05% - 0.15%)
- **Normal market:** 0.0015 - 0.0030 (0.15% - 0.30%)
- **Volatile market:** 0.0030 - 0.0060+ (0.30% - 0.60%+)

**Current HMM bands:**
- Low: < 0.001350 (< 0.135%)
- Medium: 0.001350 - 0.001621 (0.135% - 0.162%)
- High: > 0.001621 (> 0.162%)

**Problem:**
- Band "Medium" dan "High" terlalu sempit (only 0.027% range!)
- Most Gold trading happens in 0.15% - 0.40% range
- Current thresholds: 0.104% - 0.162% (MISALIGNED!)

---

## 📈 EVIDENCE

### From Bot Logs:

```
19:14:08 | Session: London (high volatility)   ← Session filter
19:15:03 | Regime: low_volatility             ← HMM detector
```

**Contradiction:**
- Session filter (based on session time) says "high volatility"
- HMM detector (based on price action) says "low volatility"

**Both can be correct IF:**
- London session = typically high volatility hours
- BUT actual price action RIGHT NOW = low volatility movement

**However,** the issue is HMM **NEVER** changes. Meaning thresholds are miscalibrated.

### From HMM Model Analysis:

```python
Regime Mapping: {
  0: LOW_VOLATILITY (mean: 0.001039),
  1: MEDIUM_VOLATILITY (mean: 0.001350),
  2: HIGH_VOLATILITY (mean: 0.001621)
}

Samples: 1888 (training data)
Log Likelihood: 33039.09
```

**Training Data Issue:**
- Model trained on 1888 samples (probably old M15 data)
- If data was from low volatility period → thresholds too low
- If data included mix → thresholds compressed

---

## 🎯 WHY THIS IS A PROBLEM

### 1. **H1 Bias Weights Misaligned**

Dynamic H1 Bias menggunakan regime untuk adjust weights:

```python
if regime == "Low Volatility":  # RANGING
    weights = {
        "rsi": 0.30,      # RSI prioritas tinggi
        "macd": 0.25,
        "ema_trend": 0.15  # EMA trend kurang penting
    }
elif regime == "High Volatility":  # TRENDING
    weights = {
        "ema_trend": 0.30,  # EMA trend prioritas
        "ema_cross": 0.25,
        "rsi": 0.10        # RSI kurang reliable
    }
```

**Problem:**
- Jika regime stuck on "Low Vol" → weights selalu set untuk ranging
- Padahal market bisa trending → weights jadi **suboptimal**

### 2. **Risk Management Suboptimal**

Risk manager bisa adjust based on regime:
- Low vol → bisa increase position size (safe)
- High vol → reduce position size (dangerous)

**Stuck on Low Vol:**
- Risk manager thinks market always safe
- Might be taking too much risk saat actually volatile

### 3. **Filter Decisions Wrong**

Entry filters might check regime:
- "Don't trade in extreme volatility"
- "Increase confidence threshold in choppy low vol"

**If regime wrong:**
- Filters make wrong decisions
- Miss good trades or take bad trades

---

## 🔧 SOLUTIONS

### Option 1: **Retrain HMM Model** (RECOMMENDED)

Retrain dengan data yang include diverse market conditions:

```bash
python train_models.py --retrain-hmm --data-period 90  # Last 90 days
```

**Steps:**
1. Fetch 90 days of M15 Gold data (include volatile + quiet periods)
2. Calculate 8 features (log returns, vol 20, vol 100, ATR, etc.)
3. Train HMM with 3-4 states
4. Map states based on actual volatility distribution

**Expected new thresholds:**
```python
Low Vol:    < 0.002 (< 0.20%)      # Quiet market
Medium Vol: 0.002 - 0.004 (0.20% - 0.40%)  # Normal trading
High Vol:   > 0.004 (> 0.40%)      # Volatile/news events
```

---

### Option 2: **Manual Threshold Adjustment**

Edit `src/regime_detector.py` to use rule-based regime:

```python
def get_current_state_simple(self, df: pl.DataFrame) -> RegimeState:
    """Simple rule-based regime (fallback if HMM stuck)."""

    # Calculate 20-period volatility
    log_returns = (df["close"] / df["close"].shift(1)).log()
    vol_20 = log_returns.rolling_std(window_size=20).tail(1).item()

    # Adjusted thresholds for Gold
    if vol_20 < 0.0020:
        regime = MarketRegime.LOW_VOLATILITY
        recommendation = "TRADE"
    elif vol_20 < 0.0040:
        regime = MarketRegime.MEDIUM_VOLATILITY
        recommendation = "TRADE"
    else:
        regime = MarketRegime.HIGH_VOLATILITY
        recommendation = "REDUCE"

    # Calculate confidence based on distance from thresholds
    if regime == MarketRegime.LOW_VOLATILITY:
        confidence = 1.0 - (vol_20 / 0.0020)
    elif regime == MarketRegime.MEDIUM_VOLATILITY:
        confidence = min(
            1.0 - abs(vol_20 - 0.0030) / 0.0010,
            0.9
        )
    else:
        confidence = min((vol_20 - 0.0040) / 0.0020, 1.0)

    return RegimeState(
        regime=regime,
        confidence=max(0.5, min(confidence, 1.0)),
        probabilities={r.value: 0.33 for r in MarketRegime},
        volatility=vol_20 * 100,  # Convert to percentage
        recommendation=recommendation
    )
```

---

### Option 3: **Use ATR % Instead**

Replace HMM with simple ATR-based regime:

```python
def get_regime_from_atr(df: pl.DataFrame) -> str:
    """Simple ATR-based regime detection."""

    atr_pct = df["atr_percent"].tail(1).item()

    if atr_pct < 0.25:
        return "low_volatility"
    elif atr_pct < 0.50:
        return "medium_volatility"
    else:
        return "high_volatility"
```

**Thresholds based on ATR %:**
- Low: < 0.25% ATR (quiet)
- Medium: 0.25% - 0.50% (normal)
- High: > 0.50% (volatile)

---

## 📊 EXPECTED IMPACT AFTER FIX

### Before (Current - Stuck):
```
Regime Distribution (Last 100 candles):
  Low: 100 (100%)    ❌ STUCK
  Medium: 0 (0%)
  High: 0 (0%)

H1 Bias Weights: ALWAYS "ranging mode"
Risk Management: ALWAYS "safe mode"
```

### After (Fixed):
```
Regime Distribution (Last 100 candles):
  Low: 45 (45%)      ✓ Quiet periods
  Medium: 40 (40%)   ✓ Normal trading
  High: 15 (15%)     ✓ Volatile spikes

H1 Bias Weights: ADAPTIVE (changes with market)
Risk Management: DYNAMIC (responds to volatility)
```

---

## 🚀 RECOMMENDED ACTION

**PRIORITY: HIGH** (affects all adaptive systems)

**Quick Fix (5 minutes):**
1. Use Option 3 (ATR-based) as temporary replacement
2. Modify `src/regime_detector.py` to add fallback logic
3. Restart bot

**Permanent Fix (30 minutes):**
1. Retrain HMM with 90 days data
2. Verify new thresholds make sense
3. Backtest to ensure regime changes appropriately
4. Deploy new model

**Verification:**
After fix, regime should change 10-20 times per day (not stuck on one!)

---

## 📝 FILES TO MODIFY

### Quick Fix:
- `src/regime_detector.py` - Add fallback ATR-based regime

### Permanent Fix:
- `train_models.py` - Add HMM retraining with better data
- `models/hmm_regime.pkl` - Replace with new model

---

**Next Step:** User decides which solution to implement.

**Expected improvement:**
- More accurate regime detection
- Better H1 bias weight selection
- Improved risk management decisions
- Higher overall profitability
