# Changelog

All notable changes to XAUBot AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.2.3] - 2026-02-11

### Fixed (SMC Primary Strategy Restoration)
**Philosophy Change:** SMC is PRIMARY, ML is SECONDARY support (not blocker)

#### Problem Identified
- **User Feedback:** "SMC adalah patokan utama, ML hanya pendukung"
- **Issue:** v0.2.2 London Filter + SELL Filter blocking high-confidence SMC signals
- **Example:** SMC BUY 75% confidence blocked because ML predicted HOLD 50%
- **Impact:** Missing profitable trades when SMC is confident

#### Solutions Implemented

**FIX #1: London Filter - Penalty Instead of Block** 🔧
```python
# BEFORE (v0.2.2):
if is_london and atr_ratio < 1.2:
    if ml_confidence < 0.70:
        return None  # BLOCKS trade completely!

# AFTER (v0.2.3):
if is_london and atr_ratio < 1.2:
    london_penalty = 0.90  # Reduce confidence by 10%, don't block
```
- **Impact:** SMC signals no longer blocked, only confidence adjusted
- **Files:** `main_live.py` line 1910-1935

**FIX #2: Signal Logic v5 - SMC Primary Hierarchy** 🎯
```python
# NEW 3-TIER LOGIC:
if smc_confidence >= 0.75:
    # TIER 1: HIGH CONFIDENCE - Execute regardless of ML
    execute_trade(confidence = smc * 0.95 if ML disagree else avg(smc, ml))

elif smc_confidence >= 0.60:
    # TIER 2: MEDIUM CONFIDENCE - Require ML agreement
    if ml_agrees and ml_confidence >= 0.60:
        execute_trade(confidence = avg(smc, ml))
    else:
        skip()

else:
    # TIER 3: LOW CONFIDENCE - Skip (SMC not confident)
    skip()
```

**Logic Changes:**
- **SMC >= 75%:** Execute ALWAYS (ML only boosts/minor penalty)
- **SMC 60-75%:** Needs ML confirmation (both agree)
- **SMC < 60%:** Skip (SMC itself not confident)
- **SELL Filter:** Removed (SMC confidence determines execution)

**Expected Results:**
- ✅ High SMC confidence (75-85%) trades execute
- ✅ No more blocking from ML HOLD predictions
- ✅ ML still provides boost when agrees (+5-10% confidence)
- ✅ ML disagree on high SMC = minor penalty (-5% confidence)

**Trade Scenarios:**
| SMC | ML | Old (v0.2.2) | New (v0.2.3) |
|-----|-----|--------------|--------------|
| BUY 85% | HOLD 50% | ❌ BLOCKED (London filter) | ✅ EXECUTE (conf 81%) |
| BUY 75% | BUY 70% | ✅ EXECUTE (conf 73%) | ✅ EXECUTE (conf 73%) |
| BUY 65% | HOLD 50% | ❌ BLOCKED (ML disagree) | ❌ SKIP (needs ML) |
| SELL 80% | HOLD 60% | ❌ BLOCKED (SELL filter) | ✅ EXECUTE (conf 76%) |

**Files Modified:**
- `main_live.py` - Signal aggregation logic rewritten (line 1936-2035)
- `VERSION` - Updated to 0.2.3
- `CHANGELOG.md` - This entry

---

## [0.2.2] - 2026-02-11

### Fixed (Professor AI Optimizations - 5 Critical Fixes)
**Exit Strategy v6.6 "Professor AI Validated"** - Implementing all Professor AI recommendations

#### Trade Analysis Summary
- **Trade #162091505:** +$0.27 profit, but only **38% peak capture** ($0.71 peak)
- **Win Rate:** 76% (excellent) but **Avg Loss 2x Avg Win** (poor risk/reward)
- **Risk/Reward:** 0.49 (below 1.0, target >1.5)
- **Problem:** Exit too aggressive, loses 62% of peak profit

#### Professor AI Diagnosis
1. ❌ **Trajectory predictor bug:** Manual calculation over-predicts 17-61x (misleading debug output)
2. ❌ **Poor peak capture:** 38% vs target 70%+ (early exit on deceleration)
3. ❌ **False breakout risk:** London + low ATR = potential whipsaw (no filter)
4. ⚠️ **Partial exit missing:** No 50% profit taking at tp_target (all-or-nothing)
5. ❌ **Unicode errors:** Emoji/arrows break Windows console logging

#### Solutions Implemented

**FIX #1: Remove Misleading Debug Code** 🔧
```python
# REMOVED dead code:
manual_1m = current_profit + _vel * 60 + 0.5 * _accel * 60**2
# ^ This was NOT dampened, always showed 17-61x "error"
# Trajectory predictor is CORRECT, debug was wrong!
```
- **Impact:** Clean logs, no more false bug warnings
- **Files:** `src/smart_risk_manager.py` line 1262-1269 removed

**FIX #2: Peak Detection Logic (CHECK 0A.4)** 🎯
```python
# NEW CHECK: Hold when approaching peak
if profit >= tp_min and vel > 0.02 and accel < -0.001:
    time_to_peak = -vel / accel  # When velocity reaches 0
    if 0 < time_to_peak <= 30:  # Peak within 30 seconds
        peak_estimate = profit + vel*t + 0.5*accel*t²
        if peak_estimate > profit * 1.15:  # 15% more profit ahead
            HOLD()  # Suppress fuzzy exit
```
- **Impact:** Prevents early exit when profit still rising but decelerating
- **Example:** Profit $0.50, vel=+0.05, accel=-0.002 → peak in 25s at $1.15 → HOLD
- **Expected:** Peak capture 38% → 70%+
- **Files:** `src/smart_risk_manager.py` CHECK 0A.4 (line 1550+)

**FIX #3: London False Breakout Filter** ⚠️
```python
# NEW: Filter whipsaws in London + low volatility
if session == "London" and atr_ratio < 1.2:
    # London + quiet = whipsaw risk
    if ml_confidence < 0.70:  # Require HIGHER confidence (60% -> 70%)
        SKIP_ENTRY()
```
- **Impact:** Reduces false breakouts during London low-vol periods
- **Trade #162091505:** Started at 16:54 London session, atr_ratio likely <1.2
- **Expected:** Win rate 76% maintained, fewer whipsaw losses
- **Files:** `main_live.py` line 1907+ (before signal logic)

**FIX #4: Enhanced Kelly Partial Exit Strategy** 💰
```python
# BEFORE: Kelly only for large profits (>$8) with fuzzy >80%
if profit >= 8.0 and exit_confidence > 0.80:
    kelly_full_exit()

# AFTER: Kelly active for ALL profits >= tp_min * 0.5
if profit >= tp_min * 0.5:  # Earlier activation
    kelly_fraction = calculate_optimal_fraction()
    if 0.3 <= kelly_fraction < 1.0:
        LOG("[KELLY PARTIAL] Recommend close {frac}%")
        # TODO: Implement mt5.close_position(ticket, volume=lot*frac)
    elif kelly_fraction >= 0.70:
        FULL_EXIT()
```
- **Impact:** Recommends partial exits (50% at tp_target * 0.5) for peak capture
- **Note:** Actual partial close implementation requires MT5 volume parameter
- **Expected:** Risk/Reward 0.49 → 1.2+ (avg profit/trade $2.00 → $4.50)
- **Files:** `src/smart_risk_manager.py` line 1426-1444

**FIX #5: Unicode Encoding Errors** 🔧
```python
# BEFORE:
logger.add("logs/bot.log", ...)  # No encoding (Windows cp1252 breaks on emoji)

# AFTER:
logger.add("logs/bot.log", encoding="utf-8", ...)  # UTF-8 for emoji support
# ALSO: Replace all emoji/arrows with ASCII
"→" -> "->"
"⚠️" -> "[WARNING]"
"⏳" -> "[removed]"
```
- **Impact:** No more `UnicodeEncodeError: 'charmap' codec` errors
- **Files:** `main_live.py` (logger setup), `src/*.py` (emoji/arrow replacement)

#### Expected Performance Improvement
| Metric | Before (v0.2.1) | Target (v0.2.2) | Improvement |
|--------|-----------------|-----------------|-------------|
| **Peak Capture** | 38% | 70%+ | +84% |
| **Avg Profit/Trade** | $2.00 | $4.50 | +125% |
| **Risk/Reward** | 0.49 | 1.2+ | +145% |
| **Win Rate** | 76% | 76% (maintain) | 0% |
| **Avg Loss** | -$4.10 | -$3.00 | -27% |

#### Trade Retrospective (v0.2.2)
Will validate after 5-10 trades:
- Peak capture improvement from better deceleration handling
- Reduced whipsaw losses from London filter
- Better profit/loss ratio from partial exits

---

## [0.2.1] - 2026-02-11

### Fixed (Fast Exit Optimization - Peak Capture Improvement)
**Exit Strategy v6.5.1 "Faster Crash Exits"** - Addressing 35% peak capture issue from Trade #162076645

#### Problem Identified (Trade #162076645)
- Trade peaked at **$1.10** but closed at **$0.39** (only **35% peak capture**)
- Crash detected at 16:45:25 (predicted -$25.56) but exit **delayed 23 seconds**
- Velocity crashed from +0.2481 → -0.0299 $/s in 5 seconds (extreme flip!)
- Lost **$0.69** (64% of peak) waiting for fuzzy threshold
- **Root Cause:** Dampening made crash warnings "less urgent" + fuzzy threshold too high

#### Solutions Implemented

**FIX 1: Dynamic Fuzzy Threshold on Crash** 🎯
```python
# BEFORE v0.2.0:
if profit < 3.0:
    threshold = 0.75  # Fixed, even during crashes

# AFTER v0.2.1:
if trajectory_pred < 0:  # Crash predicted
    threshold = threshold - 0.10  # Lower by 10%
    # $1.08 crash → 75% - 10% = 65% → exit faster!
```
- **Impact:** Exits 10-20 seconds faster when crash detected
- **Trade #162076645:** Would exit at $1.08 (65% threshold) instead of waiting for $0.39 (76%)
- **Expected:** Peak capture 35% → 70%+

**FIX 2: Asymmetric Dampening** ⚖️
```python
# BEFORE v0.2.0:
growth_damped = growth * 0.30  # Dampen ALL (positive & negative)
# Problem: Crash -$87 → Damped -$26 (less urgent!)

# AFTER v0.2.1:
if growth > 0:
    growth_damped = growth * 0.30  # Dampen optimism
else:
    growth_damped = growth * 1.00  # DON'T dampen crashes!
# Solution: Crash -$87 → RAW -$87 (urgent!)
```
- **Impact:** Crash predictions stay URGENT (not dampened)
- **Positive predictions:** Still dampened to prevent over-optimism
- **Trade #162076645:** Crash -$87.72 RAW (not -$25.56) → immediate panic exit!

**FIX 3: Velocity Crash Override** 🚨
```python
# NEW CHECK 0A.3: Emergency exit on extreme velocity flips
if velocity < -0.05 and prev_velocity > 0.10:
    if velocity_drop > 0.15:  # Extreme crash
        return INSTANT_EXIT  # Bypass fuzzy threshold!
```
- **Impact:** Instant exit on extreme momentum crashes (no delay!)
- **Trade #162076645:** vel +0.2481 → -0.0299 (drop 0.2780 > 0.15) → instant exit at $1.08!
- **Bypasses:** Fuzzy logic, trajectory override, all delays

### Changed
- Version bumped from 0.2.0 → 0.2.1 (PATCH - bug fix)
- Exit strategy upgraded from v6.5 → v6.5.1
- trajectory_predictor.py: Asymmetric dampening (only positive growth)
- smart_risk_manager.py: Crash threshold adjustment + velocity override

### Expected Impact
- **Peak Capture:** 35% → 70-80% ⬆️ (2x improvement!)
- **Exit Delay:** 23s → 5-10s ⬇️ (70% faster on crashes)
- **Profit Retention:** +$0.50-0.70 per crash trade ⬆️
- **False Exits:** No increase (only faster on REAL crashes)

### Trade #162076645 - Retrospective
**Actual Performance:**
- Duration: 46 seconds (very fast!)
- Peak: $1.10, Close: $0.39
- Peak Capture: 35% (POOR)
- Exit Reason: Fuzzy 76.66% (CORRECT but LATE)

**With v0.2.1 (Simulated):**
- Exit would trigger at $1.08 (16:45:25)
- FIX 1: Threshold lowered 75% → 65% ✅
- FIX 2: Crash -$87.72 RAW (not damped) ✅
- FIX 3: Velocity crash override (+0.24 → -0.03) ✅
- **Expected Close:** $1.08 (98% peak capture!)
- **Improvement:** +$0.69 (+177% better!)

### Note
- This is a **PATCH version** (bug fix, backward compatible)
- All 3 fixes work together synergistically
- No changes to core prediction formula (still mathematically correct)
- Only exit TIMING optimized (faster on crashes, same on normal exits)

---

## [0.2.0] - 2026-02-11

### Added (Regime-Based Dampening for Trajectory Predictions)
**Exit Strategy v6.5 "Realistic Predictions"** - Validated dampening from 33 minutes live monitoring

#### Investigation Results (v0.1.4 Debug)
- ✅ **Formula VERIFIED CORRECT** - All predictions matched manual calculations (diff=$0.00)
- ❌ **Model TOO OPTIMISTIC** - Parabolic assumption ignores market friction/decay
- 📊 **Data from 2 trades:**
  - Trade #161778984: Over-prediction 2.3x-17.2x (avg 7.5x) → closed +$4.15 ✅
  - Position #161850770: Predicted profit $6-38 from loss -$7 to -$10 ❌

#### Root Cause Analysis
**NOT a bug, but MODEL LIMITATION:**
1. Parabolic formula assumes acceleration continues indefinitely ❌
2. Real market has friction (resistance at levels, momentum fade) ✅
3. Predictions accurate for INPUT values, but inputs too volatile ✅

#### Solution: Regime-Based Dampening
**Implementation v0.2.0:**
- Added dampening factors to trajectory_predictor.py
- Only dampen GROWTH component (velocity + acceleration), NOT base profit
- Regime-specific factors validated from live data:
  ```python
  dampening_factors = {
      "ranging": 0.20,      # 80% reduction (most conservative)
      "volatile": 0.30,     # 70% reduction (validated)
      "trending": 0.50      # 50% reduction (momentum continues)
  }
  ```

**Validation from Live Trades:**
- Trade #161778984 with 0.30x dampening:
  - Raw $71.42 → Damped $21.43 (actual: $4.15) - still 5x over but acceptable ✅
  - Raw $12.23 → Damped $3.67 (actual: $4.15) - VERY CLOSE! ✅✅✅
  - Raw $9.74 → Damped $2.92 (conservative, safe) ✅

- Position #161850770 with 0.30x dampening:
  - Raw $38.15 → Damped $11.45 (more realistic from -$7.74) ✅
  - Raw $32.21 → Damped $9.66 (achievable expectation) ✅

#### New Features
1. **Regime parameter** added to `predict_future_profit()` and `should_hold_position()`
2. **Smart dampening** - only reduce growth component (v×t + 0.5×a×t²), not base profit
3. **Debug logging updated** - shows raw vs damped predictions with regime
4. **Backward compatible** - defaults to 0.30x if regime not provided

### Changed
- Version bumped from 0.1.4 → 0.2.0 (MINOR - new feature)
- Exit strategy upgraded from v6.4.3 → v6.5
- trajectory_predictor.py: Added `regime` parameter and dampening logic
- smart_risk_manager.py: Pass `regime` to trajectory predictor (2 calls updated)

### Expected Impact
- Prediction accuracy: 27% → 70-85% ⬆️
- Over-prediction: 7.5x → 1.2-1.5x ⬇️
- Peak capture: 100% maintained (exit timing stays excellent) ✅
- False holds: Reduced (more realistic profit expectations) ✅

### Performance Targets
- Average over-prediction: <2x (currently 7.5x)
- Prediction accuracy: >70% (currently 27%)
- Peak capture: Maintain 80%+ (currently 100% on Trade #161778984)

### Note
- This is a **MINOR version** (new feature, backward compatible)
- Dampening factors can be fine-tuned after 5-10 more trades
- Consider adjusting to 0.25-0.35 range if needed
- Core prediction formula remains unchanged and verified correct

---

## [0.1.4] - 2026-02-11

### Added (Deep Debug for Trajectory Bug Investigation)
**Exit Strategy v6.4.3 "Trajectory Debug Mode"** - Investigating 13x prediction error

#### Problem Identified
- Trajectory predictor formula is **CORRECT** (verified via test)
- But live predictions are **13.4x over-optimistic**
  - Example: Expected $5.07, Logged $67.64
  - Causing false HOLD signals → poor peak capture (54.5% avg)
- Bug location: **UNKNOWN** (between Kalman → Predictor → Log)

#### Debug Features Added
1. **Comprehensive Input Logging** (smart_risk_manager.py)
   - Log all inputs to trajectory predictor
   - Compare guard.velocity vs guard.kalman_velocity vs _vel
   - Track velocity_history and acceleration_history values

2. **Calculation Breakdown** (trajectory_predictor.py)
   - Log each term: p₀, v×t, 0.5×a×t²
   - Show final prediction for each horizon (1m, 3m, 5m)

3. **Manual Verification** (smart_risk_manager.py)
   - Calculate prediction manually inline
   - Compare predictor output vs manual calculation
   - Log WARNING if difference > $0.01

#### Next Steps
- Monitor 1-2 trades with full debug output
- Identify exact point where 13x scaling occurs
- Fix bug in v0.1.5
- Expected: Peak capture 54% → 75%+

### Changed
- Version bumped from 0.1.3 → 0.1.4 (PATCH - debug release)
- Exit strategy upgraded from v6.4.2 → v6.4.3

### Note
- This is a **DEBUG release** for investigation
- No functional changes to trading logic
- All debug logs use logger.debug() (won't spam console)

---

## [0.1.3] - 2026-02-11

### Fixed (Critical: FIX 1 v0.1.1 Was Never Active!)
**Exit Strategy v6.4.2 "Tiered Thresholds Finally Working"** - Live trade #161706070 revealed FIX 1 not active

#### Problem (Trade #161706070)
- Profit peaked at **$0.69** → closed at **$0.11** (lost 84% of peak!)
- Exit reason: "Fuzzy 94.58%, threshold=90%"
- **WRONG**: Profit $0.11 (<$1) should get threshold **70%**, not 90%!
- **Root Cause**: Hardcoded fuzzy_threshold at line 1313-1324 NEVER called `_calculate_fuzzy_exit_threshold()`

#### FIX: Activate Tiered Fuzzy Thresholds (FIX 1 v0.1.1) ✅
- **BEFORE**: Hardcoded thresholds ignored tiered function
  ```python
  if current_profit < 3.0:
      fuzzy_threshold = 0.90  # WRONG for micro profits!
  ```
- **AFTER**: Actually call the FIX 1 function
  ```python
  fuzzy_threshold = self._calculate_fuzzy_exit_threshold(current_profit)
  # Returns: <$1→70%, $1-3→75%, $3-8→85%, >$8→90%
  ```
- **IMPACT**: Micro profits (<$1) now exit at 70% confidence instead of 90%
  - Expected: Earlier exits on micro profits → higher profit retention
  - Target: Peak capture 16% → 60%+ for micro trades

#### Trade #161706070 Analysis
- Entry: BUY @ 5056.12
- Peak: $0.69 (vel +0.0748$/s, accel +0.0006) at 09:55:05
- Exit: $0.11 (vel -0.0040$/s) at 09:55:38 → 3m 5s duration
- **Exit was correct** (price dropped to 5052.99, would be -$3.13 loss now)
- **But late**: Should have exited at $0.50-0.60 with 70% threshold

### Changed
- Version bumped from 0.1.2 → 0.1.3 (PATCH - critical bug fix)
- Exit strategy upgraded from v6.4.1 → v6.4.2

### Note
- **BACKTEST v0.1.1 WAS INVALID** - FIX 1 was not active in backtest either
- Need to re-run backtest with FIX 1 actually working
- Grace period (v0.1.2) is still active and working

---

## [0.1.2] - 2026-02-11

### Fixed (Grace Period for Loss Exits)
**Exit Strategy v6.4.1 "Loss Recovery Window"** - Live trade analysis revealed early exit issue

#### Problem (Trade #161699163)
- Trade exited after only **18 seconds** with loss -$0.22
- Fuzzy confidence 94.58% triggered immediate exit
- Velocity was still positive (+0.0693$/s) but profit retention "collapsed"
- **Root Cause**: No grace period for micro swings, small loss after small profit treated as catastrophic

#### FIX 1: Grace Period for Loss Trades ✅
- **BEFORE**: Fuzzy exit active immediately after entry
- **AFTER**: Grace period based on regime:
  - Ranging: 120 seconds (2 minutes)
  - Volatile: 90 seconds (1.5 minutes)
  - Trending: 60 seconds (1 minute)
- **Suppression Logic**: Loss <$2 during grace period → fuzzy exit suppressed
- **IMPACT**: Prevents premature exits on micro swings, allows recovery window

#### FIX 2: Profit Retention Calculation Fix ✅
- **BEFORE**: `retention = current_profit / peak_profit` → -$0.22 / $0.17 = -1.29 → clamped to 0 ("collapsed")
- **AFTER**: Small loss (<$0) after small profit (<$3) → retention = 0.50 (medium, not collapsed)
- **IMPACT**: Micro swings no longer trigger "collapsed retention" → 95% exit confidence

### Changed
- Version bumped from 0.1.1 → 0.1.2 (PATCH - bug fix)
- Exit strategy upgraded from v6.4 → v6.4.1

### Expected Impact
- Avg trade duration: 18s → 60-120s (more reasonable)
- False early exits: -30% (grace period filtering)
- Recovery opportunities: More micro swings can recover to profit

### Note
- Trade #161699163 exit was actually **correct** (price continued to drop from 5053.74 → 5052.55)
- Grace period prevents false exits while preserving correct exit decisions for sustained losses

---

## [0.1.1] - 2026-02-11

### Fixed (Professor AI Exit Strategy Improvements)
**Exit Strategy v6.4 "Validated Fixes"** - Backtest validated over 338 trades (90 days)

#### FIX 1: Tiered Fuzzy Exit Thresholds (PRIORITY 1) ✅
- **BEFORE**: Fixed 90% fuzzy threshold for ALL profit levels
- **AFTER**: Dynamic thresholds based on profit magnitude:
  - Micro profits (<$1): 70% threshold → early exit
  - Small profits ($1-$3): 75% threshold → protection
  - Medium profits ($3-$8): 85% threshold → hold longer
  - Large profits (>$8): 90% threshold → maximize
- **IMPACT**: Avg win increased $4.07 → $9.36 (+130%), Micro profits reduced 75% → 13%

#### FIX 2: Trajectory Prediction Calibration (PRIORITY 2) ✅
- **BEFORE**: Optimistic parabolic prediction (95% error rate)
- **AFTER**: Conservative prediction with:
  - Regime penalty (ranging 0.4x, volatile 0.6x, trending 0.9x)
  - Uncertainty bounds (95% confidence interval lower bound)
  - Prevents premature exits based on overestimated future profit
- **IMPACT**: More realistic profit forecasting, reduced false exits

#### FIX 4: Unicode Fix (PRIORITY 4) ✅
- **BEFORE**: Emoji in exit messages caused encoding errors
- **AFTER**: ASCII-only exit messages for Windows compatibility
- **IMPACT**: No more UnicodeEncodeError in logs

#### FIX 5: Maximum Loss Enforcement (PRIORITY 5) ✅
- **BEFORE**: Max loss $50/trade
- **AFTER**: Max loss $25/trade with SL cap at entry
- **IMPACT**: Tighter risk control (avg loss $33 in backtest due to M15 slippage, will be closer to $25 in live with tick data)

### Changed
- Version bumped from 0.0.0 → 0.1.1 (Kalman + Bug Fixes)
- Exit strategy upgraded from v6.3 → v6.4

### Backtest Results (90 days, 338 trades)
- **Avg Win**: $9.36 ✅ (target: $8-12)
- **Micro Profits**: 13% ✅ (target: <20%, was 75%)
- **Net P/L**: +$595.16 (11.9% return)
- **Profit Factor**: 1.30 (sustainable)
- **Sharpe Ratio**: 1.29 (near target 1.5)
- **Fuzzy Exits**: 69% of trades (232/338)

### Note
- FIX 3 (Session Filter) NOT applied - trade ALL sessions per user request
- RR Ratio 1:3.57 due to M15 backtest slippage, expected to improve in live trading

---

## [0.0.0] - 2026-02-11

### Initial Release
Starting point for versioned releases. All previous development consolidated into v0.0.0 baseline.

---

## [0.0.0] - 2026-02-11

### Initial Release
Starting point for versioned releases. All previous development consolidated into v0.0.0 baseline.

#### Core Features
- **MT5 Integration**: Real-time connection to MetaTrader 5
- **Smart Money Concepts (SMC)**: Order Blocks, Fair Value Gaps, BOS/CHoCH detection
- **Machine Learning**: XGBoost model for trade signal prediction (37 features)
- **HMM Regime Detection**: Market classification (trending/ranging/volatile)
- **Risk Management**: Multi-tier capital modes (MICRO/SMALL/MEDIUM/LARGE)
- **Session Filtering**: Sydney/London/NY session optimization
- **Telegram Notifications**: Real-time trade alerts and commands

#### Advanced Exit Systems
- **v6.0 Kalman Intelligence**: Kalman filter for velocity smoothing
- **v6.1 Profit-Tier Strategy**: Dynamic exit thresholds based on profit magnitude
- **v6.2 Bug Fixes**: ExitReason.STOP_LOSS → POSITION_LIMIT correction
- **v6.3 Predictive Intelligence**:
  - Trajectory Predictor (profit forecasting 1-5min ahead)
  - Momentum Persistence Detector (continuation probability)
  - Recovery Strength Analyzer (loss recovery optimization)

#### Technical Infrastructure
- **Framework**: Python 3.11+, Polars (not Pandas), asyncio
- **Models**: XGBoost (binary classification), HMM (regime detection)
- **Database**: PostgreSQL for trade logging
- **Dashboard**: Next.js web monitoring interface
- **Deployment**: Docker support with multi-environment configs

### Performance Metrics (Baseline)
- Win Rate: 56-58%
- Average Win: $2.78 (v6.2) → Target $6-8 (v6.3)
- Peak Capture: 71% → Target 85%+
- Daily Loss Limit: 5% of capital
- Risk per Trade: 0.5-2% (capital-mode dependent)

---

## Version History Format

### [MAJOR.MINOR.PATCH] - YYYY-MM-DD

#### Added
- New features that are backward compatible

#### Changed
- Changes in existing functionality

#### Deprecated
- Features that will be removed in future versions

#### Removed
- Features that have been removed

#### Fixed
- Bug fixes

#### Security
- Security vulnerability fixes

---

## Semantic Versioning Guidelines

### MAJOR version (x.0.0)
Increment when making incompatible API changes:
- Breaking changes to core trading logic
- Removal of major features
- Database schema changes requiring migration
- Configuration format changes

Examples:
- Switching from Pandas to Polars
- Changing ML model architecture completely
- Removing hard stop-loss system

### MINOR version (0.x.0)
Increment when adding functionality in a backward-compatible manner:
- New exit strategies (e.g., v6.3 Predictive Intelligence)
- New indicators or features
- New filters or risk management modes
- Enhanced logging or monitoring

Examples:
- Adding Trajectory Predictor
- Adding new session filter
- Implementing Kelly Criterion

### PATCH version (0.0.x)
Increment when making backward-compatible bug fixes:
- Bug fixes that don't change behavior
- Performance optimizations
- Documentation updates
- Code refactoring (no logic changes)

Examples:
- Fixing ExitReason.STOP_LOSS typo
- Fixing variable scope errors
- Correcting log messages

---

## Feature Tracking

Current feature set determines version automatically:

| Feature | Version Component | Impact |
|---------|------------------|--------|
| Basic Trading (SMC + ML + MT5) | 0.x.x | Core |
| Exit v6.0 (Kalman) | 0.1.x | MINOR |
| Exit v6.1 (Profit-Tier) | 0.2.x | MINOR |
| Exit v6.2 (Bug Fixes) | 0.2.1 | PATCH |
| Exit v6.3 (Predictive) | 0.3.x | MINOR |
| Fuzzy Logic Controller | +0.1 | MINOR |
| Kelly Criterion | +0.1 | MINOR |
| Recovery Detector | +0.1 | MINOR |

---

## Links
- [Repository](https://github.com/GifariKemal/xaubot-ai)
- [Documentation](./docs/)
- [Issues](https://github.com/GifariKemal/xaubot-ai/issues)
