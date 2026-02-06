# Feature Engineering

> **File:** `src/feature_eng.py`
> **Class:** `FeatureEngineer`
> **Framework:** Pure Polars (vectorized, tanpa loop, tanpa TA-Lib)

---

## Apa Itu Feature Engineering?

Feature Engineering adalah proses **mengubah data harga mentah (OHLCV) menjadi 40+ fitur numerik** yang bisa dibaca oleh model machine learning. Ini adalah "mata" dari AI — tanpa fitur yang baik, model tidak bisa belajar apapun.

**Analogi:** Feature Engineering adalah **alat ukur** — thermometer, barometer, kompas — yang mengubah data mentah menjadi informasi bermakna.

---

## Flow Utama: `calculate_all()`

```
Input: DataFrame OHLCV (open, high, low, close, volume)
    |
    |-- calculate_rsi()               -> rsi
    |-- calculate_atr()               -> atr, atr_percent
    |-- calculate_macd()              -> macd, macd_signal, macd_histogram
    |-- calculate_bollinger_bands()   -> bb_upper, bb_lower, bb_width, bb_percent_b
    |-- calculate_ema_crossover()     -> ema_9, ema_21, ema_cross_bull/bear
    |-- calculate_volume_features()   -> volume_ratio, high_volume
    |
    |-- [jika include_ml_features=True]
    |   calculate_ml_features()       -> returns, volatility, lags, trends, time
    |
    v
Output: DataFrame dengan 40+ kolom fitur
```

**Data minimum:** 26 bar (kebutuhan MACD slow EMA) agar semua fitur stabil.

---

## Kategori 1: Indikator Teknikal

### RSI (Relative Strength Index) — Period 14

```
Formula: RSI = 100 - (100 / (1 + RS))
         RS  = Average Gain / Average Loss
Smoothing: Wilder's EMA (alpha = 1/14)
```

| Nilai | Interpretasi |
|-------|-------------|
| RSI > 70 | Overbought (potensi turun) |
| RSI < 30 | Oversold (potensi naik) |
| RSI ~ 50 | Netral |

**Output:** `rsi`

---

### ATR (Average True Range) — Period 14

```
True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR = Wilder's EMA dari True Range
ATR% = (ATR / Close) * 100
```

| Kondisi | Interpretasi |
|---------|-------------|
| ATR tinggi | Pasar volatile (pergerakan besar) |
| ATR rendah | Pasar tenang (pergerakan kecil) |

**Output:** `atr`, `atr_percent`

---

### MACD (Moving Average Convergence Divergence) — 12/26/9

```
MACD Line  = EMA(12) - EMA(26)
Signal     = EMA(MACD Line, 9)
Histogram  = MACD Line - Signal
```

| Kondisi | Interpretasi |
|---------|-------------|
| Histogram > 0 & naik | Bullish momentum menguat |
| Histogram < 0 & turun | Bearish momentum menguat |
| MACD cross Signal ke atas | Potensi reversal naik |
| MACD cross Signal ke bawah | Potensi reversal turun |

**Output:** `macd`, `macd_signal`, `macd_histogram`

---

### Bollinger Bands — Period 20, StdDev 2.0

```
Middle = SMA(20)
Upper  = Middle + 2 * StdDev
Lower  = Middle - 2 * StdDev
Width  = (Upper - Lower) / Middle
%B     = (Close - Lower) / (Upper - Lower)
```

| Kondisi | Interpretasi |
|---------|-------------|
| %B > 1 | Harga di atas upper band (extreme bullish) |
| %B < 0 | Harga di bawah lower band (extreme bearish) |
| %B ~ 0.5 | Harga di tengah |
| Width melebar | Volatilitas meningkat |
| Width menyempit | Volatilitas menurun (squeeze) |

**Output:** `bb_middle`, `bb_upper`, `bb_lower`, `bb_width`, `bb_percent_b`

---

### EMA Crossover — 9/21

```
EMA9  = Exponential Moving Average (cepat)
EMA21 = Exponential Moving Average (lambat)
```

| Kondisi | Interpretasi |
|---------|-------------|
| EMA9 > EMA21 | Tren naik |
| EMA9 < EMA21 | Tren turun |
| EMA9 cross atas EMA21 | Sinyal beli |
| EMA9 cross bawah EMA21 | Sinyal jual |

**Output:** `ema_9`, `ema_21`, `ema_cross_bull`, `ema_cross_bear`

---

## Kategori 2: Volume Features — Period 20

```
volume_sma        = Rolling Mean(volume, 20)
volume_ratio      = volume / volume_sma
volume_increasing = 1 jika volume > volume sebelumnya
high_volume       = 1 jika volume_ratio > 1.5
```

**Fungsi:** Konfirmasi breakout — pergerakan besar harus didukung volume tinggi.

**Catatan:** Jika kolom volume tidak ada di data, fitur ini di-skip (graceful degradation).

---

## Kategori 3: ML-Specific Features

### Returns & Momentum

```
returns_1   = (Close[t] / Close[t-1]) - 1     # Return 1 bar
returns_5   = (Close[t] / Close[t-5]) - 1     # Return 5 bar
returns_20  = (Close[t] / Close[t-20]) - 1    # Return 20 bar
log_returns = ln(Close[t] / Close[t-1])        # Log return
```

**Fungsi:** Mengukur kecepatan dan arah pergerakan harga dalam berbagai timeframe.

---

### Price Position

```
price_position   = (Close - Low) / (High - Low)   # Posisi 0-1 dalam range candle
dist_from_sma_20 = (Close / SMA20) - 1            # Jarak (%) dari rata-rata
```

**Fungsi:** Mengukur dimana harga relatif terhadap range dan rata-rata.

---

### Volatility

```
volatility_20       = StdDev(log_returns, 20)       # Realized volatility
normalized_range    = (High - Low) / Close           # Range sebagai % harga
avg_normalized_range = SMA(normalized_range, 14)     # Rata-rata range 14 bar
```

**Fungsi:** Input penting untuk HMM regime detection dan risk sizing.

---

### Lag Features

```
close_lag_1 = Close[t-1]
close_lag_2 = Close[t-2]
close_lag_3 = Close[t-3]
close_lag_5 = Close[t-5]
```

**Fungsi:** Auto-regressive features — menangkap pola harga berulang.

---

### Trend Features

```
higher_high = 1 jika High[t] > High[t-1], else 0
lower_low   = 1 jika Low[t] < Low[t-1], else 0
hh_count_5  = Sum(higher_high, 5 bar)   # Berapa kali HH dalam 5 bar
ll_count_5  = Sum(lower_low, 5 bar)     # Berapa kali LL dalam 5 bar
```

**Fungsi:** Mengukur konsistensi tren — banyak HH = strong uptrend.

---

### Time Features

```
hour           = Jam (0-23)
weekday        = Hari (0=Senin, 6=Minggu)
london_session = 1 jika jam 08:00-16:00 UTC
ny_session     = 1 jika jam 13:00-21:00 UTC
```

**Fungsi:** Pasar berperilaku berbeda tiap sesi — London volatile, Asian tenang.

**Catatan:** Hanya dihitung jika kolom `time` bertipe Datetime.

---

## Kategori 4: SMC sebagai Fitur Numerik

Dari SMC Analyzer, dikonversi jadi angka untuk XGBoost:

```
swing_high       = 1 / 0
swing_low        = -1 / 0
fvg_signal       = 1 (bull) / -1 (bear) / 0
ob               = 1 (bull) / -1 (bear) / 0
bos              = 1 (bull) / -1 (bear) / 0
choch            = 1 (bull) / -1 (bear) / 0
market_structure = 1 (bull) / -1 (bear) / 0
regime           = 0 / 1 / 2 (dari HMM)
```

---

## Target Variable (Label Training)

```python
create_target(df, lookahead=1, threshold=0.0):
    target = 1 jika close[t+1] > close[t]   # Harga naik
    target = 0 jika close[t+1] <= close[t]   # Harga turun/tetap
    target_return = close[t+1] / close[t] - 1  # Return kontinu
```

**Catatan:** Target dibuat saat training saja, tidak saat live trading.

---

## Preprocessing untuk ML

### Penanganan Null
```python
# Bar awal memiliki NaN karena lookback period
# Saat training: baris dengan NaN di-drop
df_clean = df.select(features + [target]).drop_nulls()
```

### Penanganan Infinity
```python
# Saat prediksi: NaN & infinity diganti 0
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

### Normalisasi
**Tidak dilakukan** — XGBoost berbasis tree, scale-invariant (tidak perlu scaling).

### Cleanup Kolom Temporary
Setiap method membersihkan kolom sementara yang diawali `_` (misal `_delta`, `_avg_gain`, dll).

---

## Fitur yang Digunakan vs Tidak

### Digunakan oleh XGBoost (24+ fitur)
Semua indikator teknikal, returns, volatility, trend, time, SMC numerik, regime.

### Tidak Digunakan (Excluded)
- Kolom OHLCV asli: `time`, `open`, `high`, `low`, `close`, `volume`
- Kolom meta: `spread`, `real_volume`, `target`, `target_return`
- Kolom SMC level: `fvg_top`, `fvg_bottom`, `ob_top`, `ob_bottom`, dll
- Kolom temporary: apapun yang diawali `_`

---

## Parameter Konfigurasi

| Indikator | Parameter | Default | Configurable |
|-----------|-----------|---------|-------------|
| RSI | period | 14 | Ya |
| ATR | period | 14 | Ya |
| MACD | fast/slow/signal | 12/26/9 | Ya |
| Bollinger Bands | period, std_dev | 20, 2.0 | Ya |
| EMA Crossover | fast/slow | 9/21 | Ya |
| Volume | period | 20 | Ya |
| Returns | lookback | [1, 5, 20] | Hardcoded |
| Volatility | window | 20 | Hardcoded |
| Session | London hours | 08-16 UTC | Hardcoded |
| Session | NY hours | 13-21 UTC | Hardcoded |

---

## Performa

- **5000 bar features:** < 100ms (sangat cepat)
- **Framework:** Polars vectorized (10-100x lebih cepat dari Pandas loop)
- **Memory:** ~1.6MB untuk 40 fitur x 5000 bar
