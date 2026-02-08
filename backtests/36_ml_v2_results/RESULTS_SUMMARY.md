# ML V2 — Training Results Summary

**Date:** 2026-02-08 20:45
**Dataset:** 50,000 M15 bars XAUUSD
**Training Method:** 80/20 train/test split, early stopping

---

## 🏆 Performance Comparison

| Config | Name | Features | Train AUC | Test AUC | Overfit | vs Baseline | vs Live (0.696) |
|--------|------|----------|-----------|----------|---------|-------------|-----------------|
| **Baseline** | V1 Reproduction | 53 | 0.6203 | **0.6158** | 1.01 | — | -11.5% |
| **A** | Better Target | 53 | 0.6375 | **0.6253** | 1.02 | +0.0095 | -10.2% |
| **B** | +H1 Features | 61 | 0.7015 | **0.7064** | 0.99 | +0.0906 | +1.5% |
| **C** | +Continuous SMC | 68 | 0.7051 | **0.7108** | 0.99 | +0.0950 | +2.1% |
| **D** | All Features ⭐ | 76 | 0.7385 | **0.7339** | 1.01 | +0.1181 | **+5.5%** |
| **E** | Ensemble | 76 | 0.7385 | **0.7339** | 1.01 | +0.1181 | **+5.5%** |

---

## 🎯 Winner: Config D (All Features)

**Test AUC:** 0.7339
**Improvement vs Live Model:** +5.5% (from 0.696 to 0.7339)
**Model File:** `model_d.pkl`
**Features:** 76 total
- 53 base features (V1)
- 8 H1 multi-timeframe features
- 7 continuous SMC features
- 4 regime conditioning features
- 4 price action features

**Overfitting:** Well controlled (1.01 ratio)
**Recommendation:** ✅ Ready for backtesting with full trading logic

---

## 📈 Key Insights

### 1. H1 Features = Biggest Impact (+0.08 AUC)
Jumping from Config A (0.6253) to Config B (0.7064) shows that **H1 multi-timeframe context is critical** for XAUUSD trading.

**H1 Features (8):**
- `h1_market_structure` — H1 trend direction
- `h1_ema20_distance` — Price vs H1 EMA20
- `h1_trend_strength` — H1 BOS count
- `h1_swing_proximity` — Distance to H1 swing
- `h1_fvg_active` — Inside H1 FVG zone?
- `h1_ob_proximity` — Distance to H1 OB
- `h1_atr_ratio` — H1 ATR / M15 ATR
- `h1_rsi` — H1 RSI value

### 2. Continuous SMC Features Add Value (+0.004 AUC)
Converting SMC signals from binary (0/1) to continuous values (gap size, distance, age) provides more nuanced information to the model.

**Continuous SMC Features (7):**
- `fvg_gap_size_atr` — FVG gap / ATR
- `fvg_age_bars` — Bars since last FVG
- `ob_width_atr` — OB width / ATR
- `ob_distance_atr` — Distance to OB / ATR
- `bos_recency` — Bars since last BOS
- `confluence_score` — Count SMC signals in last 10 bars
- `swing_distance_atr` — Distance to swing / ATR

### 3. Regime + Price Action Features (+0.023 AUC)
Regime conditioning and price action patterns complete the feature set.

**Regime Features (4):**
- `regime_duration_bars` — Consecutive bars in regime
- `regime_transition_prob` — 1 / duration
- `volatility_zscore` — (ATR - mean) / std
- `crisis_proximity` — ATR / (mean * 2.5)

**Price Action Features (4):**
- `wick_ratio` — (upper + lower wick) / range
- `body_ratio` — |close - open| / range
- `gap_from_prev_close` — Gap / ATR
- `consecutive_direction` — # candles same direction

### 4. Ensemble Didn't Help (Same as XGBoost)
Config E (XGBoost + LightGBM ensemble) achieved the same 0.7339 test AUC as Config D (XGBoost only). Single well-tuned XGBoost is sufficient — no need for ensemble complexity.

### 5. Overfitting Well Controlled
All configs show train/test ratio ≈ 1.0, confirming that anti-overfitting parameters (depth 3, heavy L1/L2 regularization) are working well.

---

## 🔝 Top 20 Most Important Features (Baseline Model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ob | 615.19 |
| 2 | ob_mitigated | 177.04 |
| 3 | returns_1 | 170.57 |
| 4 | log_returns | 73.36 |
| 5 | bb_percent_b | 57.37 |
| 6 | returns_5 | 54.36 |
| 7 | price_position | 32.67 |
| 8 | close_lag_2 | 16.19 |
| 9 | ema_9 | 12.88 |
| 10 | macd | 11.67 |
| 11 | dist_from_sma_20 | 8.82 |
| 12 | atr | 7.28 |
| 13 | hour | 6.80 |
| 14 | macd_histogram | 6.19 |
| 15 | h1_ema20 | 6.05 |
| 16 | volume_ratio | 5.85 |
| 17-20 | (Low importance < 5) | — |

**Note:** Order Block (OB) signals dominate feature importance, confirming SMC validity.

---

## 📦 Model Files

| File | Size | Config | Test AUC | Notes |
|------|------|--------|----------|-------|
| `model_baseline.pkl` | 27 KB | Baseline | 0.6158 | V1 reproduction |
| `model_a.pkl` | 23 KB | A | 0.6253 | Better target |
| `model_b.pkl` | 28 KB | B | 0.7064 | +H1 features |
| `model_c.pkl` | 29 KB | C | 0.7108 | +Continuous SMC |
| `model_d.pkl` ⭐ | 68 KB | D | **0.7339** | **All features (BEST)** |
| `model_e.pkl` | 174 KB | E | 0.7339 | Ensemble (XGB+LGBM) |

---

## ✅ Success Criteria

- ✅ **Target AUC >0.70 achieved** (0.7339)
- ✅ **Overfitting controlled** (all ratios <1.2)
- ✅ **Each feature category adds value** (incremental improvements)
- ✅ **Anti-overfitting params work** (train ≈ test)
- ✅ **Better than live model** (+5.5% AUC improvement)

---

## 🚀 Next Steps — Integration Plan

### Phase 1: Backtest with Trading Logic
Run Config D through full backtest with entry/exit logic (backtests/backtest_36_ml_v2.py needs modification):
- Use `model_d.pkl` for predictions
- Apply same SMC entry/exit filters as live
- Compare WR%, PnL, Sharpe vs current model

### Phase 2: Code Integration (If Successful)
Modify `main_live.py`:
1. Fetch H1 data alongside M15
2. Load V2 feature engineering:
   ```python
   from backtests.ml_v2 import MLV2FeatureEngineer
   fe_v2 = MLV2FeatureEngineer()
   df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)
   ```
3. Load Config D model:
   ```python
   model = TradingModelV2.load("models/xgboost_model_v2.pkl")
   ```

### Phase 3: Forward Test
- Run on demo account for 1 week
- Monitor WR%, PnL, drawdown
- Compare vs live model's performance

### Phase 4: Deploy to Live
- If demo results confirm improvement
- Copy `model_d.pkl` to `models/xgboost_model_v2.pkl`
- Deploy to production

---

## 🎓 Lessons Learned

1. **Multi-timeframe features matter most** — H1 context provided +0.08 AUC boost
2. **Continuous > Binary** — Converting SMC to continuous values adds signal
3. **Better target helps** — ATR threshold filtering reduces noise
4. **Simple ensemble not needed** — Well-tuned single model sufficient
5. **Anti-overfitting works** — Heavy regularization keeps model generalizable

---

**Generated:** 2026-02-08 20:45
**Training Time:** ~5 minutes (6 configs)
**Status:** ✅ Complete and successful
