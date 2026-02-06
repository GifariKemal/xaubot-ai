# XGBoost — Signal Predictor

> **File:** `src/ml_model.py`
> **Model:** `models/xgboost_model.pkl`
> **Library:** `xgboost`

---

## Apa Itu XGBoost?

XGBoost (eXtreme Gradient Boosting) adalah algoritma machine learning berbasis **ensemble decision tree**. Model ini belajar dari puluhan fitur teknikal untuk **memprediksi arah harga** di bar berikutnya.

**Analogi:** XGBoost adalah **navigator AI** — menentukan apakah harga akan naik atau turun.

---

## Fungsi Utama

XGBoost bertugas **memprediksi probabilitas harga naik atau turun** di bar M15 berikutnya, lalu menghasilkan signal BUY, SELL, atau HOLD.

```
prob_up > 0.65   ->  BUY
prob_down > 0.65 ->  SELL
lainnya          ->  HOLD (tidak cukup yakin)
```

---

## Arsitektur Model

```python
params = {
    "objective": "binary:logistic",   # Klasifikasi biner (naik/turun)
    "eval_metric": "auc",             # Area Under Curve
    "max_depth": 3,                   # Kedalaman tree (anti-overfitting)
    "learning_rate": 0.05,            # Lambat & stabil
    "min_child_weight": 10,           # Minimum sampel per leaf
    "subsample": 0.7,                 # 70% data per round
    "colsample_bytree": 0.6,          # 60% fitur per tree
    "reg_alpha": 1.0,                 # L1 regularization
    "reg_lambda": 5.0,                # L2 regularization (kuat)
    "gamma": 1.0,                     # Min loss reduction per split
}
```

**Anti-Overfitting:**
- Tree dangkal (depth 3, bukan 6)
- Early stopping setelah 5 round tanpa improvement
- Feature subsampling 60%
- Regularisasi L2 kuat (lambda=5.0)

---

## Input (24 Fitur)

### Indikator Teknikal
| Fitur | Sumber | Fungsi |
|-------|--------|--------|
| `rsi` | Feature Eng | Overbought/oversold |
| `atr`, `atr_percent` | Feature Eng | Volatilitas |
| `macd`, `macd_signal`, `macd_histogram` | Feature Eng | Momentum tren |
| `bb_percent_b`, `bb_width` | Feature Eng | Posisi dalam Bollinger Band |
| `ema_9`, `ema_21` | Feature Eng | Tren jangka pendek |

### Returns & Momentum
| Fitur | Formula | Fungsi |
|-------|---------|--------|
| `returns_1` | `close[t]/close[t-1] - 1` | Return 1 bar |
| `returns_5` | `close[t]/close[t-5] - 1` | Return 5 bar |
| `returns_20` | `close[t]/close[t-20] - 1` | Return 20 bar |
| `log_returns` | `ln(close[t]/close[t-1])` | Log return |

### Volatilitas & Posisi Harga
| Fitur | Fungsi |
|-------|--------|
| `volatility_20` | Realized volatility 20 bar |
| `normalized_range` | (High-Low)/Close |
| `avg_normalized_range` | Rata-rata range 14 bar |
| `price_position` | Posisi 0-1 dalam range |
| `dist_from_sma_20` | Jarak dari SMA 20 |

### Smart Money Concepts (SMC)
| Fitur | Fungsi |
|-------|--------|
| `swing_high`, `swing_low` | Fractal structure |
| `fvg_signal` | Fair Value Gap (1/-1/0) |
| `ob` | Order Block (1/-1/0) |
| `bos`, `choch` | Break of Structure, Change of Character |
| `market_structure` | Bullish/Bearish (1/-1/0) |

### Waktu & Regime
| Fitur | Fungsi |
|-------|--------|
| `hour`, `weekday` | Pola jam & hari |
| `london_session`, `ny_session` | Flag sesi trading |
| `regime` | HMM regime state (0/1/2) |

---

## Cara Kerja

### Proses Prediksi (Setiap Loop)

```
DataFrame lengkap (200 bar + semua fitur)
        |
        v
Ambil baris terakhir (1 bar)
        |
        v
Pilih 24 fitur yang sesuai dengan training
        |
        v
Buat DMatrix (format XGBoost)
        |
        v
model.predict() -> probabilitas harga NAIK (0-1)
        |
        v
Tentukan signal:
  prob_up > 0.65   -> BUY
  prob_down > 0.65 -> SELL
  lainnya          -> HOLD
        |
        v
Output: PredictionResult
  - signal: "BUY" / "SELL" / "HOLD"
  - probability: 0-1 (prob naik)
  - confidence: max(prob_up, prob_down)
  - feature_importance: {fitur: skor}
```

### Proses Training

```
1. Ambil 10,000 bar M15 XAUUSD
2. Feature engineering (40+ kolom)
3. SMC analysis (swing, FVG, OB, BOS, CHoCH)
4. Buat target: 1 jika close[t+1] > close[t], else 0
5. Split: 70% train, 30% test
6. Train XGBoost 50 round + early stopping (patience=5)
7. Evaluasi: Train AUC vs Test AUC
8. Simpan model + feature names ke .pkl
```

---

## Output & Dampak ke Trading

### 1. Validasi Signal SMC

```
SMC bilang BUY + XGBoost setuju (>55%)    -> TRADE
SMC bilang BUY + XGBoost netral (<55%)    -> SKIP
SMC bilang BUY + XGBoost bilang SELL >75% -> TOLAK (veto)
```

### 2. Confidence Gate

```
ML confidence < 55%  -> Tidak boleh entry (terlalu tidak yakin)
ML confidence 55-65% -> Entry dengan lot kecil
ML confidence > 65%  -> Entry dengan lot penuh
```

### 3. Exit Signal (Penutupan Posisi)

```
Posisi BUY terbuka
XGBoost prediksi SELL dengan confidence > 75%
-> TUTUP posisi (ML reversal exit)
```

### 4. Feature Importance

```python
# Contoh output (top 5)
{
    "market_structure": 0.85,   # Fitur paling penting
    "rsi": 0.68,
    "atr_percent": 0.65,
    "macd_histogram": 0.52,
    "bos": 0.48,
}
```

Menunjukkan fitur mana yang paling berpengaruh dalam keputusan model.

---

## Metrik Evaluasi

```python
{
    "train_auc": 0.6234,      # Performa di data training
    "test_auc": 0.5932,       # Performa di data testing
    "train_samples": 7000,
    "test_samples": 3000,
    "num_features": 24,
}
```

| AUC | Interpretasi |
|-----|-------------|
| 0.50 | Sama dengan tebak acak |
| 0.55 | Sedikit lebih baik dari acak |
| 0.60 | Cukup baik untuk trading |
| 0.70+ | Sangat baik |

**Rollback threshold:** Jika test AUC < 0.52, model otomatis rollback ke versi sebelumnya.

---

## Auto-Retraining

- **Jadwal:** Harian pukul 05:00 WIB
- **Data:** 5,000 bar terakhir
- **Proses:** Backup lama -> retrain -> validasi AUC -> simpan/rollback
- **Minimum interval:** 20 jam antar retrain (cegah overfitting)

---

## Contoh Skenario

**Skenario 1: Signal kuat**
```
RSI=35 (oversold), MACD rising, BOS bullish, market_structure=1
-> XGBoost: prob_up=0.78 -> BUY (confidence 78%)
-> Lot penuh, entry dieksekusi
```

**Skenario 2: Konflik dengan SMC**
```
SMC signal: BUY
XGBoost: prob_down=0.82 -> SELL (confidence 82%)
-> Signal DITOLAK (ML strongly disagrees >75%)
-> Tidak ada trade
```

**Skenario 3: Tidak yakin**
```
RSI=50, MACD flat, regime=1
-> XGBoost: prob_up=0.53 -> HOLD (confidence 53% < 55%)
-> Tidak ada trade — tunggu signal lebih jelas
```
