# ML V2 — Full ML Overhaul

**Problem:** Current ML model has AUC ~0.696 (barely better than random). Too noisy, limited features, single model.

**Solution:** 3-phase improvement:
1. Better target (multi-bar + ATR threshold)
2. 23 new features (H1, continuous SMC, regime, price action)
3. Ensemble models (XGBoost + LightGBM)

---

## File Structure

```
backtests/ml_v2/
├── __init__.py                 # Package init
├── ml_v2_target.py             # Better target variables (Step 1)
├── ml_v2_feature_eng.py        # 23 new features (Step 2)
├── ml_v2_model.py              # Multi-model support (Step 3)
├── ml_v2_train.py              # Training pipeline + walk-forward CV
└── README.md                   # This file

backtests/backtest_36_ml_v2.py  # Main backtest script
backtests/36_ml_v2_results/     # Output directory
```

---

## Components

### 1. `ml_v2_target.py` — Better Targets (Highest Impact)

**Problem:** Current target predicts 1-bar ahead with threshold=0 → captures noise.

**Solutions:**
- **Multi-bar target** (primary): Look 3 bars ahead, filter moves < 0.3 * ATR (~$3.6)
- **3-class target**: BUY/SELL/HOLD explicit classes
- **Baseline target**: V1 reproduction for comparison

**Expected impact:** AUC +0.05 to +0.10 (biggest single improvement)

---

### 2. `ml_v2_feature_eng.py` — 23 New Features

Adds 23 features on top of base 37:

**H1 Multi-Timeframe (8 features):**
- `h1_market_structure`: H1 trend direction
- `h1_ema20_distance`: Price vs H1 EMA20 / ATR
- `h1_trend_strength`: H1 BOS count
- `h1_swing_proximity`: Distance to H1 swing / ATR
- `h1_fvg_active`: Inside H1 FVG zone?
- `h1_ob_proximity`: Distance to H1 OB / ATR
- `h1_atr_ratio`: H1 ATR / M15 ATR
- `h1_rsi`: H1 RSI value

**Continuous SMC (7 features):**
- `fvg_gap_size_atr`: FVG gap / ATR (bigger = more reliable)
- `fvg_age_bars`: Bars since last FVG
- `ob_width_atr`: OB width / ATR
- `ob_distance_atr`: Distance to OB / ATR
- `bos_recency`: Bars since last BOS
- `confluence_score`: Count SMC signals in last 10 bars
- `swing_distance_atr`: Distance to swing / ATR

**Regime Conditioning (4 features):**
- `regime_duration_bars`: Consecutive bars in regime
- `regime_transition_prob`: 1 / duration
- `volatility_zscore`: (ATR - mean) / std
- `crisis_proximity`: ATR / (mean * 2.5)

**Price Action (4 features):**
- `wick_ratio`: (upper + lower wick) / range
- `body_ratio`: |close - open| / range
- `gap_from_prev_close`: Gap / ATR
- `consecutive_direction`: # candles same direction

**Total:** 37 (base) + 23 (new) = **60 features**

---

### 3. `ml_v2_model.py` — Multi-Model Support

**Model types:**
- `XGBOOST_BINARY`: Binary classification (UP/DOWN)
- `XGBOOST_3CLASS`: 3-class (BUY/SELL/HOLD)
- `LIGHTGBM_BINARY`: LightGBM binary
- `ENSEMBLE`: Average XGBoost + LightGBM probabilities

**Features:**
- Backward compatible with V1 TradingModel
- Same anti-overfitting philosophy (depth 3, heavy regularization)
- Saves/loads as `.pkl`
- Can load V1 models via `load_legacy_v1()`

---

### 4. `ml_v2_train.py` — Training Pipeline

**Purged Walk-Forward CV:**
- 5 folds
- 5000 train / 1000 test / 50 gap per fold
- Gap prevents temporal leakage
- Reports mean ± std AUC, overfitting ratio

**Experiment Configs:**
| Config | Target | Features | Model |
|--------|--------|----------|-------|
| Baseline | 1-bar (V1) | 37 base | XGBoost |
| **A** | 3-bar + ATR | 37 base | XGBoost |
| **B** | 3-bar + ATR | 37 + 8 H1 = 45 | XGBoost |
| **C** | 3-bar + ATR | 45 + 7 SMC = 52 | XGBoost |
| **D** | 3-bar + ATR | 52 + 8 regime/PA = 60 | XGBoost |
| **E** | 3-bar + ATR | 60 | XGB + LGBM ensemble |

---

## Usage

### Run Main Backtest

```bash
python backtests/backtest_36_ml_v2.py
```

This will:
1. Fetch XAUUSD M15 + H1 data from MT5
2. Calculate all features (base + V2)
3. Create all targets (baseline, multi-bar, 3-class)
4. Train all 6 configs (Baseline, A, B, C, D, E)
5. Run 5-fold purged walk-forward CV for each
6. Print comparison table
7. Save models to `backtests/36_ml_v2_results/model_*.pkl`

**Expected runtime:** 10-20 minutes (depends on CV depth)

---

### Standalone Usage

```python
from backtests.ml_v2 import TargetBuilder, MLV2FeatureEngineer, TradingModelV2, ModelType

# 1. Create better targets
builder = TargetBuilder()
df = builder.create_multi_bar_target(df, lookahead=3, threshold_atr_mult=0.3)

# 2. Add V2 features
fe_v2 = MLV2FeatureEngineer()
df = fe_v2.add_all_v2_features(df_m15, df_h1)

# 3. Train model
model = TradingModelV2(model_type=ModelType.XGBOOST_BINARY)
model.fit(df, feature_cols, target_col="multi_bar_target")

# 4. Predict
pred = model.predict(df)
print(f"Signal: {pred.signal}, Confidence: {pred.confidence}")
```

---

## Expected Results

**Baseline (V1):**
- Train AUC: ~0.75, Test AUC: ~0.70 (overfitting)
- Actual: ~0.696 (from live model)

**Config A (Better Target):**
- Expected: Test AUC +0.05 to +0.10 vs Baseline
- Why: Filters noise, focuses on tradeable moves

**Config B (+H1 Features):**
- Expected: Test AUC +0.02 to +0.05 vs A
- Why: Higher timeframe context

**Config C (+Continuous SMC):**
- Expected: Test AUC +0.01 to +0.03 vs B
- Why: SMC strength (gap size, confluence)

**Config D (+All Features):**
- Expected: Test AUC +0.01 to +0.02 vs C
- Why: Regime transitions, price action patterns

**Config E (Ensemble):**
- Expected: Test AUC +0.00 to +0.02 vs D
- Why: Ensemble reduces variance

**Target:** Test AUC > 0.75 (from 0.696)

---

## Validation Checklist

After running backtest, check:

1. **AUC Improvement**: Each config should improve or maintain test AUC
2. **Overfitting Ratio**: Train AUC / Test AUC < 1.2 (acceptable)
3. **Feature Importance**: Check if new features are used (not ignored)
4. **Nulls**: Verify no excessive nulls in V2 features
5. **Baseline Match**: Baseline config should reproduce V1 results (~0.70 AUC)

---

## Integration Plan (If Successful)

If Config D or E shows significant improvement (test AUC > 0.75):

1. **Copy best model** to `models/xgboost_model_v2.pkl`
2. **Update `src/ml_model.py`** to load V2 by default
3. **Modify `main_live.py`** to:
   - Add V2 feature calculation (H1 data fetch required)
   - Use V2 model for predictions
4. **Run forward test** on demo account for 1 week
5. **Compare metrics** vs V1 (WR, PnL, Sharpe)

---

## Dependencies

All dependencies already in `requirements.txt`:
- `xgboost>=2.0.0` (core)
- `polars>=0.20.0` (data processing)
- `scikit-learn>=1.3.0` (metrics)
- `lightgbm>=4.0.0` (optional, for ensemble Config E)

If `lightgbm` not installed, ensemble will fall back to XGBoost-only.

---

## Notes

- **No live code changes**: All files in `backtests/ml_v2/` (isolated)
- **Backward compatible**: Can load V1 models via `load_legacy_v1()`
- **Windows compatible**: Tested on Windows 11, Python 3.11+
- **Polars-first**: All data processing uses Polars (not Pandas)
- **Anti-overfitting**: Same regularization philosophy as V1

---

## File Sizes

- `ml_v2_target.py`: ~9 KB (target builder)
- `ml_v2_feature_eng.py`: ~22 KB (23 features)
- `ml_v2_model.py`: ~19 KB (multi-model support)
- `ml_v2_train.py`: ~9 KB (training pipeline)
- `backtest_36_ml_v2.py`: ~11 KB (main backtest)

**Total package**: ~70 KB (5 files)

---

## Troubleshooting

**Import Error:**
```bash
# Ensure you're in project root
cd "C:/Users/Administrator/Videos/Smart Automatic Trading BOT + AI"
python backtests/backtest_36_ml_v2.py
```

**LightGBM Not Found:**
- Config E will skip LightGBM and use XGBoost-only ensemble
- Optional: `pip install lightgbm>=4.0.0`

**MT5 Connection Failed:**
- Check `.env` credentials
- Ensure MT5 terminal is running

**Low AUC (<0.65):**
- Check feature nulls: `df[feature_cols].null_count()`
- Verify target distribution: `df['multi_bar_target'].value_counts()`
- Inspect feature importance: Are new features used?

---

## References

- **Plan**: See plan mode transcript (`1a09c953-bc49-4062-b130-8dd676f7eb1f.jsonl`)
- **V1 Model**: `src/ml_model.py`
- **Base Features**: `src/feature_eng.py`
- **SMC**: `src/smc_polars.py`
- **Regime**: `src/regime_detector.py`
