# HMM (Hidden Markov Model) — Regime Detector

> **File:** `src/regime_detector.py`
> **Model:** `models/hmm_regime.pkl`
> **Library:** `hmmlearn.GaussianHMM`

---

## Apa Itu HMM?

Hidden Markov Model adalah model statistik yang mendeteksi **"hidden state" (kondisi tersembunyi)** dari data yang terlihat. Dalam konteks trading, HMM membaca pola volatilitas dan return harga untuk mengklasifikasikan **kondisi pasar saat ini**.

**Analogi:** HMM adalah **radar cuaca** untuk pasar — menentukan apakah pasar sedang cerah, mendung, atau badai.

---

## Fungsi Utama

HMM bertugas **mengklasifikasikan kondisi pasar** ke dalam 3 regime:

| Regime | Nama | Aksi Trading | Lot Multiplier |
|--------|------|-------------|----------------|
| 0 | `LOW_VOLATILITY` | Trade normal | 1.0x |
| 1 | `MEDIUM_VOLATILITY` | Trade normal | 1.0x |
| 2 | `HIGH_VOLATILITY` | Kurangi lot | 0.5x |
| - | `CRISIS` | Stop trading | 0.0x |

---

## Arsitektur Model

```python
GaussianHMM(
    n_components=3,          # 3 regime (low/medium/high volatility)
    covariance_type="diag",  # Diagonal covariance (stabil)
    n_iter=200,              # Iterasi training
    random_state=42,
)
```

**Konfigurasi** (`config.py`):
```
n_regimes        = 3     # Jumlah regime
lookback_periods = 500   # Bar untuk training
retrain_frequency = 20   # Retrain setiap 20 bar
```

---

## Input (Fitur)

HMM hanya menggunakan **2 fitur sederhana**:

| Fitur | Formula | Fungsi |
|-------|---------|--------|
| **Log Returns** | `ln(close[t] / close[t-1])` | Momentum & arah harga |
| **Rolling Volatility** | `StdDev(log_returns, 20)` | Gejolak pasar 20 bar |

**Kenapa hanya 2?** HMM bekerja optimal dengan fitur sedikit tapi representatif. Dua fitur ini sudah cukup menangkap pola volatilitas pasar.

---

## Cara Kerja

### Proses Prediksi (Setiap Loop)

```
200 bar M15 terakhir dari MT5
        |
        v
prepare_features()
  - Hitung log_returns = ln(close[t] / close[t-1])
  - Hitung rolling volatility = StdDev(20 bar)
        |
        v
model.predict(features)
  - Output: regime per bar (0, 1, atau 2)
        |
        v
model.predict_proba(features)
  - Output: probabilitas tiap regime (0-1)
        |
        v
Mapping ke nama regime:
  - Sort berdasarkan volatilitas
  - Volatilitas terendah = LOW_VOLATILITY
  - Volatilitas tertinggi = HIGH_VOLATILITY
        |
        v
Output per bar:
  - regime: 0/1/2
  - regime_name: "low_volatility" / "medium_volatility" / "high_volatility"
  - regime_confidence: 0.0 - 1.0
```

### Proses Training

```
1. Ambil 10,000 bar M15 XAUUSD dari MT5
2. Hitung fitur: log_returns + volatility
3. Fit GaussianHMM dengan 3 komponen
   -> Model belajar transition probability antar regime
   -> Model belajar emission probability (pola tiap state)
4. Map state ke nama regime berdasarkan sorting volatilitas
5. Simpan ke models/hmm_regime.pkl
```

---

## Output & Dampak ke Trading

### 1. Position Size Multiplier

```python
get_position_multiplier(regime):
    LOW_VOLATILITY      -> 1.0x (lot penuh)
    MEDIUM_VOLATILITY   -> 1.0x (lot penuh)
    HIGH_VOLATILITY     -> 0.5x (lot setengah)
    CRISIS              -> 0.0x (tidak trading)

# Contoh:
base_lot = 0.02
actual_lot = base_lot * multiplier
# HIGH_VOL: 0.02 * 0.5 = 0.01
```

### 2. Trading Gate

```
if regime == CRISIS:
    return None  # STOP — tidak boleh trading sama sekali
```

### 3. Fitur Input untuk XGBoost

Kolom `regime` (0/1/2) juga dikirim sebagai salah satu dari 24 fitur XGBoost, sehingga model ML tahu kondisi pasar saat membuat prediksi.

---

## Transition Matrix

HMM menghasilkan **matriks transisi** yang menunjukkan probabilitas perpindahan antar regime:

```
              Ke:
Dari:     LOW    MED    HIGH
LOW     [ 0.85   0.12   0.03 ]   <- 85% tetap low
MED     [ 0.10   0.78   0.12 ]   <- 78% tetap medium
HIGH    [ 0.05   0.15   0.80 ]   <- 80% tetap high
```

**Kegunaan:** Memprediksi seberapa lama regime saat ini akan bertahan.

---

## Auto-Retraining

- **Jadwal:** Harian pukul 05:00 WIB (saat pasar tutup)
- **Data:** 5,000 bar terakhir
- **Validasi:** Jika log-likelihood terlalu rendah, rollback ke model lama
- **Backup:** Model lama disimpan di `models/backups/[timestamp]/`

---

## Metrik Evaluasi

```python
{
    "samples": 10000,           # Bar yang digunakan
    "n_regimes": 3,             # Jumlah state
    "log_likelihood": -1234.5,  # Kualitas fit (makin tinggi makin baik)
}
```

---

## Contoh Skenario

**Skenario 1: Pasar tenang**
```
Input:  Volatilitas rendah, return stabil
Output: regime=0 (LOW_VOLATILITY), confidence=0.92
Aksi:   Trading normal, lot penuh (1.0x)
```

**Skenario 2: Volatilitas melonjak (berita NFP)**
```
Input:  Volatilitas tinggi, return besar
Output: regime=2 (HIGH_VOLATILITY), confidence=0.88
Aksi:   Lot dikurangi 50% (0.5x), melindungi modal
```

**Skenario 3: Flash crash**
```
Input:  Volatilitas ekstrem, return sangat besar
Output: regime=CRISIS, confidence=0.95
Aksi:   STOP trading — 0% lot, lindungi akun
```
