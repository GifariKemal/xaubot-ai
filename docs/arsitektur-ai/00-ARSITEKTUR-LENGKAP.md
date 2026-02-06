# Arsitektur Lengkap — Smart AI Trading Bot

> **Dokumen:** Arsitektur keseluruhan sistem dalam 1 file
> **Instrumen:** XAUUSD (Gold) M15
> **Platform:** MetaTrader 5
> **Bahasa:** Python 3.11+ (async, Polars, XGBoost, HMM)
> **Database:** PostgreSQL + CSV fallback
> **Notifikasi:** Telegram Bot API

---

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Diagram Arsitektur](#2-diagram-arsitektur)
3. [23 Komponen](#3-23-komponen)
4. [Pipeline Data: Dari OHLCV ke Keputusan Trading](#4-pipeline-data)
5. [Alur Entry: 11 Filter](#5-alur-entry-11-filter)
6. [Alur Exit: 10 Kondisi](#6-alur-exit-10-kondisi)
7. [Sistem Proteksi Risiko 4 Lapis](#7-sistem-proteksi-risiko-4-lapis)
8. [AI/ML Engine](#8-aiml-engine)
9. [Smart Money Concepts (SMC)](#9-smart-money-concepts)
10. [Position Lifecycle](#10-position-lifecycle)
11. [Auto-Retraining & Model Management](#11-auto-retraining)
12. [Infrastruktur & Database](#12-infrastruktur--database)
13. [Konfigurasi & Parameter Kritis](#13-konfigurasi--parameter-kritis)
14. [Performa & Timing](#14-performa--timing)
15. [Error Handling & Fault Tolerance](#15-error-handling--fault-tolerance)
16. [Daftar File Source Code](#16-daftar-file-source-code)

---

## 1. Gambaran Umum

### Apa Ini?

Bot trading otomatis yang menggabungkan **3 otak kecerdasan buatan** untuk trading XAUUSD (Emas) di MetaTrader 5:

```
OTAK 1: Smart Money Concepts (SMC)
        → Membaca pola institusi besar (bank, hedge fund)
        → Menentukan DIMANA entry, SL, dan TP

OTAK 2: XGBoost Machine Learning
        → Memprediksi ARAH harga (naik/turun)
        → Memberikan tingkat keyakinan (confidence)

OTAK 3: Hidden Markov Model (HMM)
        → Membaca KONDISI pasar (tenang/volatile/krisis)
        → Menyesuaikan ukuran posisi dan agresivitas
```

### Filosofi Desain

```
1. KESELAMATAN MODAL NOMOR 1
   → 4 lapis proteksi stop loss
   → Lot ultra-kecil (0.01-0.02)
   → Circuit breaker otomatis

2. TIDAK PERNAH CRASH
   → Setiap error di-catch, bot terus jalan
   → Database gagal? CSV fallback
   → MT5 putus? Auto-reconnect

3. SELF-IMPROVING
   → Model AI dilatih ulang otomatis setiap hari
   → Rollback otomatis jika model baru lebih buruk
   → Threshold confidence menyesuaikan kondisi pasar

4. TRANSPARAN
   → Setiap keputusan dicatat ke database
   → Notifikasi Telegram real-time
   → Laporan harian, jam-an, dan per-trade
```

### Angka-Angka Kunci

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| Modal target | $5,000 | Small account mode |
| Risiko per trade | 1% ($50) | Maksimum kerugian per posisi |
| Lot size | 0.01 - 0.02 | Ultra-konservatif |
| Max daily loss | 3% ($150) | Circuit breaker harian |
| Max total loss | 10% ($500) | Stop total trading |
| Max posisi bersamaan | 2-3 | Menghindari overexposure |
| Cooldown antar trade | 5 menit | Mencegah overtrading |
| Loop speed | ~50ms | Cepat tapi efisien |
| Timeframe | M15 | 15 menit per candle |

---

## 2. Diagram Arsitektur

### Diagram Keseluruhan Sistem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MAIN LIVE (Orchestrator)                       │
│                          main_live.py — TradingBot                      │
│                          Candle-based (M15) + position check ~10 detik  │
│                                                                         │
│  ┌─────────────── PHASE 1: DATA ──────────────────────────────────┐    │
│  │                                                                 │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │    │
│  │  │   MT5    │    │ Feature  │    │   SMC    │    │   HMM    │ │    │
│  │  │Connector │ ──→│  Engine  │ ──→│ Analyzer │    │ Regime   │ │    │
│  │  │(broker)  │    │(40+ fitur│    │(institusi│    │(3 state) │ │    │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │    │
│  │       │               │               │               │       │    │
│  │       │               │               ▼               ▼       │    │
│  │       │               │         ┌──────────┐    ┌──────────┐ │    │
│  │       │               └────────→│ XGBoost  │    │ Dynamic  │ │    │
│  │       │                         │ ML Model │    │Confidence│ │    │
│  │       │                         │(prediksi)│    │(threshold│ │    │
│  │       │                         └──────────┘    └──────────┘ │    │
│  └───────┼─────────────────────────────┼───────────────┼────────┘    │
│          │                             │               │             │
│  ┌───────┼──── PHASE 2: MONITORING ────┼───────────────┼────────┐    │
│  │       │                             │               │         │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │    │
│  │  │ Position │    │  Smart   │    │  Risk    │              │    │
│  │  │ Manager  │    │  Risk    │    │  Engine  │              │    │
│  │  │(trailing)│    │ Manager  │    │ (Kelly)  │              │    │
│  │  └──────────┘    └──────────┘    └──────────┘              │    │
│  └────────────────────────────────────────────────────────────┘    │
│          │                             │               │             │
│  ┌───────┼──── PHASE 3: ENTRY ─────────┼───────────────┼────────┐    │
│  │       │                             │               │         │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │    │
│  │  │ Session  │    │  News    │    │ 11-Gate  │              │    │
│  │  │ Filter   │    │  Agent   │    │  Entry   │              │    │
│  │  │(waktu)   │    │(berita)  │    │  Filter  │──→ EXECUTE   │    │
│  │  └──────────┘    └──────────┘    └──────────┘              │    │
│  └────────────────────────────────────────────────────────────┘    │
│          │                                                          │
│  ┌───────┼──── PHASE 4: PERIODIK ──────────────────────────────┐    │
│  │       │                                                      │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │    │
│  │  │  Auto    │    │ Telegram │    │  Trade   │              │    │
│  │  │ Trainer  │    │ Notifier │    │  Logger  │              │    │
│  │  │(retrain) │    │(laporan) │    │(DB+CSV)  │              │    │
│  │  └──────────┘    └──────────┘    └──────────┘              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             │                                        │
│                    ┌────────┴────────┐                               │
│                    │   PostgreSQL    │                               │
│                    │   + CSV Backup  │                               │
│                    └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Alur Data (Data Flow)

```
MT5 Broker (XAUUSD M15)
       │
       │  200 bar OHLCV
       ▼
┌─────────────────┐
│ MT5 Connector   │  numpy → Polars (tanpa Pandas)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engineer │  OHLCV → 40+ fitur teknikal
│                  │  RSI, ATR, MACD, BB, EMA, Volume,
│                  │  Returns, Volatility, Lags, Trend
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│  SMC  │ │  HMM  │
│Analyzer│ │Regime │
│       │ │Detect │
└───┬───┘ └───┬───┘
    │         │
    │    ┌────┘
    │    │
    ▼    ▼
┌─────────────────┐
│    XGBoost      │  24 fitur → Prediksi BUY/SELL/HOLD
│  ML Predictor   │  + Confidence 0-100%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dynamic         │  Sesuaikan threshold berdasarkan
│ Confidence      │  sesi, regime, volatilitas, trend
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal Combiner │  SMC + ML harus setuju
│ (11 Filter)     │  + Session + Risk + Cooldown
└────────┬────────┘
         │
    ┌────┴────┐
    │  PASS?  │
    │         │
   YES       NO → tunggu loop berikutnya
    │
    ▼
┌─────────────────┐
│ Risk Engine     │  Kelly Criterion → lot size
│ + Risk Manager  │  Validasi order → approve/reject
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute Order   │  Kirim ke MT5 dengan SL & TP
│ via MT5         │  Register ke Position Manager
└────────┬────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
┌───────────┐    ┌────────────────┐
│ Telegram  │    │ Trade Logger   │
│ Notifier  │    │ (DB + CSV)     │
└───────────┘    └────────────────┘
```

---

## 3. 23 Komponen

### Tabel Komponen Lengkap

| # | Komponen | File | Kategori | Fungsi Utama |
|---|----------|------|----------|-------------|
| 1 | HMM Regime Detector | `src/regime_detector.py` | AI/ML | Deteksi kondisi pasar (3 regime) |
| 2 | XGBoost Predictor | `src/ml_model.py` | AI/ML | Prediksi arah harga + confidence |
| 3 | SMC Analyzer | `src/smc_polars.py` | Analisis | Pola institusi: FVG, OB, BOS, CHoCH |
| 4 | Feature Engineering | `src/feature_eng.py` | Data | OHLCV → 40+ fitur numerik |
| 5 | Smart Risk Manager | `src/smart_risk_manager.py` | Risiko | 4 mode trading, 10 kondisi exit |
| 6 | Session Filter | `src/session_filter.py` | Filter | Waktu trading optimal (WIB) |
| 7 | Stop Loss (4 Lapis) | Multi-file | Proteksi | SMC → Software → Emergency → Circuit |
| 8 | Take Profit (6 Layer) | Multi-file | Proteksi | Hard → Momentum → Peak → Probability → Early → Broker |
| 9 | Entry Trade | `main_live.py` | Eksekusi | 11 filter berurutan |
| 10 | Exit Trade | `main_live.py` | Eksekusi | 10 kondisi exit real-time |
| 11 | News Agent | `src/news_agent.py` | Monitor | Monitoring berita (TIDAK memblokir) |
| 12 | Telegram Notifier | `src/telegram_notifier.py` | Notifikasi | 11 tipe notifikasi real-time |
| 13 | Auto Trainer | `src/auto_trainer.py` | ML Ops | Retraining harian otomatis |
| 14 | Backtest | `backtests/backtest_live_sync.py` | Validasi | Simulasi 100% sync dengan live |
| 15 | Dynamic Confidence | `src/dynamic_confidence.py` | Adaptif | Threshold ML adaptif (60-85%) |
| 16 | MT5 Connector | `src/mt5_connector.py` | Koneksi | Bridge ke broker, auto-reconnect |
| 17 | Configuration | `src/config.py` | Config | 6 sub-config, auto-adjust modal |
| 18 | Trade Logger | `src/trade_logger.py` | Logging | Dual storage DB + CSV |
| 19 | Position Manager | `src/position_manager.py` | Manajemen | Trailing SL, breakeven, market close |
| 20 | Risk Engine | `src/risk_engine.py` | Risiko | Kelly Criterion, circuit breaker |
| 21 | Database | `src/db/` | Storage | PostgreSQL, 6 repository |
| 22 | Train Models | `train_models.py` | Training | Script training awal |
| 23 | Main Live | `main_live.py` | Orchestrator | Koordinasi semua komponen |

### Hubungan Antar Komponen

```
                    ┌─────────────────────────┐
                    │     CONFIGURATION (17)   │
                    │   Sumber parameter semua │
                    └────────────┬─────────────┘
                                 │ dikonsumsi oleh semua
                                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ MT5 (16) │───→│FeatEng(4)│───→│  SMC (3) │
│  Broker  │    │ 40+ fitur│    │ Institusi│
└──────────┘    └────┬─────┘    └────┬─────┘
                     │               │
                     ▼               ▼
               ┌──────────┐    ┌──────────┐
               │ HMM  (1) │    │XGBoost(2)│
               │ Regime   │    │ Prediksi │
               └────┬─────┘    └────┬─────┘
                    │               │
                    ▼               ▼
               ┌──────────────────────────┐
               │  Dynamic Confidence (15) │
               │  Threshold adaptif       │
               └────────────┬─────────────┘
                            │
                            ▼
┌──────────┐    ┌──────────────────────────┐    ┌──────────┐
│Session(6)│───→│    ENTRY TRADE (9)       │←───│ News(11) │
│  Waktu   │    │    11 Filter Gate        │    │ Berita   │
└──────────┘    └────────────┬─────────────┘    └──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              ┌──────────┐    ┌──────────────┐
              │RiskEng(20│    │SmartRisk (5) │
              │Kelly Lot │    │ 4 Mode       │
              └────┬─────┘    └──────┬───────┘
                   │                 │
                   ▼                 ▼
              ┌──────────────────────────┐
              │     EXECUTE ORDER        │
              │     via MT5 (16)         │
              └────────────┬─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │PosMgr(19)│ │Logger(18)│ │Telegram  │
        │Trailing  │ │ DB+CSV   │ │  (12)    │
        └────┬─────┘ └────┬─────┘ └──────────┘
             │            │
             ▼            ▼
        ┌──────────┐ ┌──────────┐
        │EXIT (10) │ │  DB (21) │
        │10 Kondisi│ │PostgreSQL│
        └──────────┘ └──────────┘

Periodik:
┌──────────┐    ┌──────────┐    ┌──────────┐
│AutoTrain │    │ Backtest │    │  Train   │
│  (13)    │    │   (14)   │    │Models(22)│
│Harian    │    │Validasi  │    │Setup awal│
└──────────┘    └──────────┘    └──────────┘
```

---

## 4. Pipeline Data

### Dari OHLCV Mentah ke Keputusan Trading

#### Tahap 1: Data Fetching (MT5 Connector)

```
MT5 Broker
    │
    │  mt5.copy_rates_from_pos("XAUUSD", MT5_TIMEFRAME_M15, 0, 200)
    │
    ▼
NumPy Structured Array
    │
    │  Konversi langsung ke Polars (TANPA Pandas)
    │
    ▼
Polars DataFrame:
┌──────────────────────────────────────────────────────┐
│ time │ open │ high │ low │ close │ tick_volume │ spread │
│ i64  │ f64  │ f64  │ f64 │ f64   │ f64         │ f64    │
├──────┼──────┼──────┼─────┼───────┼─────────────┼────────┤
│ ...  │ 2645 │ 2648 │ 2643│ 2647  │ 5234        │ 25     │
└──────────────────────────────────────────────────────┘
```

**Kenapa Polars, bukan Pandas?**
- 3-5x lebih cepat untuk operasi vectorized
- Memory-efficient (zero-copy)
- Native lazy evaluation
- Konsisten di seluruh codebase (tidak ada konversi bolak-balik)

#### Tahap 2: Feature Engineering (40+ Fitur)

```
Input: Polars DataFrame (200 bar OHLCV)
    │
    ├── Momentum Indicators
    │   ├── RSI(14)          → 0-100, overbought/oversold
    │   ├── MACD(12,26,9)    → trend strength & direction
    │   └── MACD Histogram   → momentum acceleration
    │
    ├── Volatility Indicators
    │   ├── ATR(14)          → average true range (pips)
    │   ├── Bollinger Bands(20,2.0) → upper, lower, width
    │   └── Volatility(20)   → rolling std of returns
    │
    ├── Trend Indicators
    │   ├── EMA(9) / EMA(21) → fast/slow crossover
    │   ├── EMA Cross Signal → 1 (bullish) / -1 (bearish)
    │   └── SMA(20)          → simple moving average
    │
    ├── Price Action
    │   ├── Returns(1,5,20)  → % perubahan harga
    │   ├── Log Returns       → untuk distribusi normal
    │   ├── Price Position    → posisi dalam range BB
    │   └── Higher High/Lower Low count → trend structure
    │
    ├── Volume Features
    │   ├── Volume SMA(20)   → rata-rata volume
    │   └── Volume Ratio      → current / average
    │
    ├── Lag Features
    │   ├── close_lag_1..5   → harga sebelumnya
    │   └── returns_lag_1..3 → return sebelumnya
    │
    └── Time Features
        ├── Hour, Weekday    → waktu candle
        └── Session flags    → london, ny, overlap
    │
    ▼
Output: DataFrame + 40 kolom baru (semua numerik, siap ML)
```

**Minimum data:** 26 bar untuk semua indikator stabil

#### Tahap 3: SMC Analysis (Pola Institusi)

```
Input: DataFrame dengan OHLCV
    │
    ├── Swing Points (Fractal)
    │   Window: 11 bar (swing_length=5, ±5 dari tengah)
    │   Output: swing_high (1/0), swing_low (-1/0), level harga
    │
    ├── Fair Value Gaps (FVG)
    │   Bullish: bar[i-2].high < bar[i].low (gap up)
    │   Bearish: bar[i-2].low > bar[i].high (gap down)
    │   Output: fvg_bull, fvg_bear, fvg_top, fvg_bottom, fvg_mid
    │
    ├── Order Blocks (OB)
    │   Lookback: 10 bar
    │   Bullish: candle bearish terakhir sebelum move up besar
    │   Bearish: candle bullish terakhir sebelum move down besar
    │   Output: ob (1/-1), ob_top, ob_bottom, ob_mitigated
    │
    ├── Break of Structure (BOS)
    │   BOS: harga break swing high/low → trend continuation
    │   Output: bos (1/-1), level yang di-break
    │
    ├── Change of Character (CHoCH)
    │   CHoCH: harga break berlawanan arah trend → reversal signal
    │   Output: choch (1/-1), level yang di-break
    │
    └── Liquidity Zones
        BSL: Buy Side Liquidity (above swing highs)
        SSL: Sell Side Liquidity (below swing lows)
        Output: bsl_level, ssl_level
    │
    ▼
Signal Generation:
    Syarat: Structure break + (FVG ATAU Order Block)
    │
    ├── Entry: harga saat ini
    ├── SL: ATR-based, minimum 1.5 × ATR dari entry
    ├── TP: 2:1 Risk-Reward minimum, cap 4 × ATR
    ├── Confidence: 55-85% (berdasarkan confluence)
    └── Reason: "BOS + Bullish FVG at 2645.50"
```

#### Tahap 4: Regime Detection (HMM)

```
Input: log_returns + normalized_range (volatilitas)
    │
    │  GaussianHMM(n_components=3, lookback=500)
    │
    ▼
3 Regime:
┌──────────────────────────────────────────────────┐
│ REGIME 0: Low Volatility                         │
│   → Pasar tenang, range kecil                    │
│   → Lot multiplier: 1.0x (normal)               │
│   → Rekomendasi: TRADE                           │
│                                                   │
│ REGIME 1: Medium Volatility                      │
│   → Pasar aktif, trend jelas                     │
│   → Lot multiplier: 1.0x (normal)               │
│   → Rekomendasi: TRADE                           │
│                                                   │
│ REGIME 2: High Volatility                        │
│   → Pasar sangat volatile, berbahaya             │
│   → Lot multiplier: 0.5x (setengah)             │
│   → Rekomendasi: REDUCE                          │
│                                                   │
│ CRISIS (detected by FlashCrashDetector):         │
│   → Move > 2.5% dalam 1 menit                   │
│   → Lot multiplier: 0.0x (STOP)                 │
│   → Rekomendasi: EMERGENCY CLOSE ALL             │
└──────────────────────────────────────────────────┘
```

#### Tahap 5: ML Prediction (XGBoost)

```
Input: 24 fitur terpilih dari Feature Engineering + SMC + Regime
    │
    │  XGBoost Binary Classifier
    │  Anti-overfitting config:
    │    max_depth=3, learning_rate=0.05
    │    min_child_weight=10, subsample=0.7
    │    colsample_bytree=0.6
    │    reg_alpha=1.0 (L1), reg_lambda=5.0 (L2)
    │
    ▼
Output:
┌──────────────────────────────────────────┐
│ prob_up: 0.72  (probabilitas naik)       │
│ prob_down: 0.28 (probabilitas turun)     │
│                                           │
│ → Signal: BUY (prob_up > 0.50)           │
│ → Confidence: 72%                        │
│                                           │
│ Threshold Keputusan:                      │
│   prob > 0.50 → ada sinyal (minimum)     │
│   prob > 0.65 → sinyal kuat              │
│   prob > 0.75 → sinyal sangat kuat       │
│   prob > 0.80 → lot bisa naik ke 0.02    │
└──────────────────────────────────────────┘
```

#### Tahap 6: Dynamic Confidence (Threshold Adaptif)

```
Scoring (0-100 poin):

Base score: 50
    │
    ├── Session Modifier:
    │   Golden Time (20:00-23:59 WIB) → +20
    │   London (15:00-23:59)          → +15
    │   New York (20:00-05:00)        → +10
    │   Tokyo/Sydney                  → +0
    │   Market Closed                 → -30
    │
    ├── Regime Modifier:
    │   Medium Volatility             → +15
    │   Low Volatility                → +5
    │   High Volatility               → -5
    │   Crisis                        → -25
    │
    ├── Volatility Modifier:
    │   Medium (ideal)                → +10
    │   Low                           → +0
    │   High                          → -5
    │   Extreme                       → -10
    │
    ├── Trend Modifier:
    │   Trending (jelas)              → +10
    │   Ranging (sideways)            → -5
    │
    ├── SMC Confluence:
    │   Ada konfluensi                → +10
    │   Tidak ada                     → +0
    │
    └── ML Confidence:
        ≥ 70%                         → +5
        ≥ 60%                         → +2
        < 60%                         → +0
    │
    ▼
Quality Level → Threshold:
┌───────────────────────────────────────────────────┐
│ EXCELLENT (≥80 poin) → Threshold: 60% (longgar)   │
│ GOOD      (65-79)    → Threshold: 65%             │
│ MODERATE  (50-64)    → Threshold: 70%             │
│ POOR      (35-49)    → Threshold: 80% (ketat)    │
│ AVOID     (<35)      → Threshold: 85% (SKIP)     │
└───────────────────────────────────────────────────┘

Contoh: Golden Time + Medium Vol + Trending + SMC + ML 72%
        = 50 + 20 + 15 + 10 + 10 + 10 + 5 = 120 → cap 100
        = EXCELLENT → Threshold 60% → ML 72% PASS ✓
```

---

## 5. Alur Entry: 11 Filter

Setiap sinyal harus melewati **11 gerbang berurutan**. Satu saja gagal = TIDAK trading.

```
SINYAL SMC + ML MASUK
        │
        ▼
┌─ FILTER 1: Session Filter ──────────────────────────────┐
│  Apakah sekarang jam trading yang dibolehkan?            │
│  ✗ 00:00-06:00 WIB (dead zone) → BLOCK                  │
│  ✗ Jumat ≥ 23:00 WIB (weekend risk) → BLOCK             │
│  ✓ London/NY/Golden Time → PASS                          │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 2: Risk Mode ───────────────────────────────────┐
│  Apakah mode trading bukan STOPPED?                      │
│  ✗ STOPPED (daily/total limit hit) → BLOCK               │
│  ✓ NORMAL / RECOVERY / PROTECTED → PASS                  │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 3: SMC Signal ──────────────────────────────────┐
│  Apakah ada setup SMC yang valid?                        │
│  ✗ Tidak ada FVG/OB + BOS/CHoCH → BLOCK                 │
│  ✓ Ada sinyal BUY/SELL dengan SL & TP → PASS            │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 4: ML Confidence ───────────────────────────────┐
│  Apakah ML confidence ≥ dynamic threshold?               │
│  ✗ ML confidence < threshold → BLOCK                     │
│  ✓ ML confidence ≥ threshold → PASS                      │
│  (threshold 60-85% tergantung market quality)            │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 5: ML Agreement ────────────────────────────────┐
│  Apakah ML TIDAK strongly disagree dengan SMC?           │
│  ✗ ML > 65% berlawanan arah SMC → BLOCK (conflict)      │
│  ✓ ML setuju atau netral → PASS                          │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 6: Market Quality ──────────────────────────────┐
│  Apakah Dynamic Confidence bukan AVOID?                  │
│  ✗ Quality == AVOID (score < 35) → BLOCK                 │
│  ✓ EXCELLENT/GOOD/MODERATE/POOR → PASS                   │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 7: Signal Confirmation ─────────────────────────┐
│  Apakah sinyal konsisten 2x berturut-turut?              │
│  ✗ Sinyal baru muncul 1x → BLOCK (tunggu konfirmasi)    │
│  ✓ Sinyal sudah 2x berturut → PASS                      │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 8: Pullback Filter ─────────────────────────────┐
│  Apakah momentum selaras dengan arah sinyal?             │
│  ✗ BUY tapi RSI > 80 (overbought) → BLOCK               │
│  ✗ SELL tapi RSI < 20 (oversold) → BLOCK                 │
│  ✗ MACD histogram berlawanan → BLOCK                     │
│  ✗ Harga bounce > $2 dalam 3 candle → BLOCK             │
│  ✓ Momentum selaras → PASS                               │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 9: Trade Cooldown ──────────────────────────────┐
│  Apakah sudah ≥ 5 menit sejak trade terakhir?            │
│  ✗ < 300 detik sejak trade terakhir → BLOCK              │
│  ✓ ≥ 300 detik → PASS                                    │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 10: Position Limit ─────────────────────────────┐
│  Apakah jumlah posisi terbuka < limit?                   │
│  ✗ Sudah 2+ posisi terbuka → BLOCK                       │
│  ✓ < 2 posisi → PASS                                     │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
┌─ FILTER 11: Lot Size ───────────────────────────────────┐
│  Apakah lot size yang dihitung > 0?                      │
│  ✗ Lot = 0 (regime crisis / risk terlalu tinggi) → BLOCK│
│  ✓ Lot ≥ 0.01 → PASS                                    │
└──────────────────────────────────────────────────────────┘
        │ PASS
        ▼
   ╔═══════════════════════╗
   ║   EXECUTE TRADE       ║
   ║   BUY atau SELL       ║
   ║   via MT5 Connector   ║
   ╚═══════════════════════╝
        │
        ├── Register ke SmartRiskManager
        ├── Log ke TradeLogger (DB + CSV)
        └── Kirim notifikasi Telegram
```

---

## 6. Alur Exit: 10 Kondisi

Setiap posisi terbuka dievaluasi **setiap ~10 detik** (di antara candle) atau **setiap candle baru** (full analysis) terhadap 10 kondisi exit:

```
POSISI TERBUKA (dicek setiap ~10 detik)
        │
        │  Update: profit, momentum, peak, durasi
        │
        ▼
┌─ KONDISI 1: Smart Take Profit ──────────────────────────┐
│  (a) Profit ≥ $40 → TUTUP (hard TP)                     │
│  (b) Profit ≥ $25 + momentum < -30 → TUTUP              │
│  (c) Peak > $30, sekarang < 60% peak → TUTUP            │
│  (d) Profit ≥ $20 + TP probability < 25% → TUTUP        │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 2: Early Exit (Profit Kecil) ──────────────────┐
│  Profit $5-$15 + ML reversal ≥ 65% + momentum < -50     │
│  → TUTUP (amankan profit kecil sebelum hilang)           │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 3: Early Cut (v4 — Smart Hold DIHAPUS) ────────┐
│  Loss >= 30% max ($15) DAN momentum < -30?               │
│  → TUTUP CEPAT (early cut, jangan tunggu recovery)       │
│                                                           │
│  v4: "Smart Hold" dihapus — tidak ada lagi hold losers    │
│  menunggu golden time atau sesi London.                   │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 4: ML Trend Reversal ──────────────────────────┐
│  ML confidence ≥ 65% BERLAWANAN ARAH posisi              │
│  + 3x warning berturut-turut                             │
│  → TUTUP (AI mendeteksi pembalikan trend)                │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 5: Maximum Loss ───────────────────────────────┐
│  Loss ≥ 50% dari max_loss_per_trade ($25 dari $50)      │
│  → TUTUP (kerugian terlalu besar)                        │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 6: Stall Detection ────────────────────────────┐
│  Posisi sudah 10+ bar tanpa profit signifikan            │
│  → TUTUP (pasar tidak bergerak sesuai ekspektasi)        │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 7: Daily Limit ────────────────────────────────┐
│  Total daily loss mendekati limit                        │
│  → TUTUP semua posisi (proteksi sisa modal hari ini)     │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 8: Weekend Close ──────────────────────────────┐
│  Mendekati market close weekend + ada posisi profit      │
│  → TUTUP (hindari gap risk Senin)                        │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 9: Time-Based ─────────────────────────────────┐
│  (a) > 4 jam + profit < $5 → TUTUP (terlalu lama)       │
│  (b) > 6 jam → FORCE TUTUP (apapun kondisi)             │
└──────────────────────────────────────────────────────────┘
        │ tidak trigger
        ▼
┌─ KONDISI 10: Default HOLD ──────────────────────────────┐
│  Tidak ada kondisi terpenuhi                             │
│  → HOLD (biarkan posisi berjalan)                        │
└──────────────────────────────────────────────────────────┘
```

### Position Manager (Tambahan per Posisi)

Selain 10 kondisi di atas, Position Manager juga menjalankan:

```
┌─ Market Close Handler (Prioritas Tertinggi) ────────────┐
│  Dekat close harian/weekend?                             │
│  ├── Profit ≥ $10 + dekat close → TUTUP (amankan)       │
│  ├── Loss + weekend + SL > 50% → TUTUP (gap risk)       │
│  └── Loss kecil + weekend → HOLD (bisa recovery)        │
└──────────────────────────────────────────────────────────┘

┌─ Breakeven Protection ──────────────────────────────────┐
│  Profit ≥ 15 pips → Pindah SL ke entry + 2 buffer       │
│  (posisi tidak bisa rugi lagi)                           │
└──────────────────────────────────────────────────────────┘

┌─ Trailing Stop ─────────────────────────────────────────┐
│  Profit ≥ 25 pips → SL mengikuti harga, jarak 10 pips   │
│  (kunci profit sambil biarkan berjalan)                  │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Sistem Proteksi Risiko 4 Lapis

```
╔══════════════════════════════════════════════════════════════════╗
║                    LAPIS 1: BROKER STOP LOSS                     ║
║                    (Otomatis oleh MT5)                            ║
║                                                                   ║
║  SL = Entry ± (1.5 × ATR)    minimum 10 pips                    ║
║  Dikirim bersama order ke broker                                 ║
║  Aktif 24/7, bahkan jika bot mati                                ║
║  Max loss: ~$50-80 per trade                                     ║
╠══════════════════════════════════════════════════════════════════╣
║                    LAPIS 2: SOFTWARE SMART EXIT                  ║
║                    (Bot mengevaluasi setiap ~10 detik)           ║
║                                                                   ║
║  10 kondisi exit (lihat bagian 6)                                ║
║  Biasanya menutup SEBELUM broker SL kena                        ║
║  Target close: loss ≤ $25 (lebih ketat dari broker)              ║
║  Termasuk: momentum, ML reversal, stall, time limit             ║
╠══════════════════════════════════════════════════════════════════╣
║                    LAPIS 3: EMERGENCY STOP LOSS                  ║
║                    (Backup jika software gagal)                  ║
║                                                                   ║
║  Max loss per trade: 2% modal ($100 untuk $5K)                   ║
║  Diset sebagai broker SL terpisah                                ║
║  Aktif jika software error/hang                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                    LAPIS 4: CIRCUIT BREAKER                      ║
║                    (Hentikan semua trading)                       ║
║                                                                   ║
║  Trigger 1: Daily loss ≥ 3% ($150) → Stop hari ini              ║
║  Trigger 2: Total loss ≥ 10% ($500) → Stop total                ║
║  Trigger 3: Flash crash > 2.5% / 1 menit → CLOSE ALL            ║
║  Reset: Otomatis di hari baru (daily), manual (total)            ║
╚══════════════════════════════════════════════════════════════════╝
```

### 4 Mode Trading (Smart Risk Manager)

```
┌──────────────────────────────────────────────────────────┐
│ MODE: NORMAL                                             │
│ Kondisi: Semua aman, tidak ada masalah                   │
│ Lot: 0.01 - 0.02 (berdasarkan confidence)                │
│ Max posisi: 2-3                                          │
│                                                           │
│            │ 3x loss berturut-turut                       │
│            ▼                                              │
│ MODE: RECOVERY                                           │
│ Kondisi: Setelah kerugian beruntun                        │
│ Lot: 0.01 (minimum saja)                                 │
│ Max posisi: 1                                            │
│                                                           │
│            │ mendekati 80% daily limit                    │
│            ▼                                              │
│ MODE: PROTECTED                                          │
│ Kondisi: Hampir kena daily limit                         │
│ Lot: 0.01 (minimum saja)                                 │
│ Max posisi: 1                                            │
│                                                           │
│            │ daily/total limit tercapai                   │
│            ▼                                              │
│ MODE: STOPPED                                            │
│ Kondisi: Batas kerugian tercapai                          │
│ Lot: 0 (TIDAK BOLEH trading)                             │
│ Max posisi: 0 (tutup semua)                              │
│ Reset: Otomatis hari baru                                │
└──────────────────────────────────────────────────────────┘
```

### Lot Sizing: Risk-Constrained Half-Kelly

```
Langkah 1: Hitung Kelly Fraction
    f* = (win_rate × avg_rr - (1 - win_rate)) / avg_rr

    Contoh: win_rate=55%, avg_rr=2.0
    f* = (0.55 × 2.0 - 0.45) / 2.0 = 0.325 (32.5%)

Langkah 2: Cap Kelly (max 25%)
    f* = min(0.325, 0.25) = 0.25

Langkah 3: Half-Kelly (safety)
    f* = 0.25 × 0.5 = 0.125 (12.5%)

Langkah 4: Apply regime multiplier
    High volatility: × 0.5 = 0.0625
    Normal: × 1.0 = 0.125

Langkah 5: Cap di config limit
    config risk_per_trade = 1%
    actual_risk = min(0.125, 0.01) = 0.01 (1%)

Langkah 6: Hitung lot
    risk_amount = $5000 × 1% = $50
    SL distance = 50 pips → pip_value ~$1/pip/0.01lot
    lot = $50 / (50 × $1) = 0.01 lot

Langkah 7: ML Confidence boost
    ML ≥ 80% → lot × 2 = 0.02 lot (maximum)
    ML < 65% → lot = 0.01 (minimum)

Langkah 8: Session multiplier
    Golden Time: × 1.2
    Sydney: × 0.5

    Final lot: 0.01 - 0.02 (ultra-konservatif)
```

---

## 8. AI/ML Engine

### Hidden Markov Model (HMM) — Otak Regime

```
┌──────────────────────────────────────────────────────────┐
│ HIDDEN MARKOV MODEL                                      │
│                                                           │
│ Library: hmmlearn.GaussianHMM                            │
│ Input: log_returns + rolling_volatility (2 fitur)        │
│ States: 3 (Low, Medium, High Volatility)                 │
│ Lookback: 500 bar untuk training                         │
│ Retrain: setiap 20 bar (auto-update)                     │
│                                                           │
│ Transition Matrix (contoh):                              │
│          To Low   To Med   To High                       │
│ Fr Low  [ 0.85    0.12     0.03  ]                       │
│ Fr Med  [ 0.10    0.80     0.10  ]                       │
│ Fr High [ 0.05    0.15     0.80  ]                       │
│                                                           │
│ Distribusi Emisi (per state):                            │
│ Low:  μ_return ≈ 0, σ_return = kecil                    │
│ Med:  μ_return ≈ 0, σ_return = sedang                   │
│ High: μ_return ≈ 0, σ_return = besar                    │
│                                                           │
│ Output:                                                   │
│ ├── regime: 0/1/2 (low/medium/high)                     │
│ ├── confidence: 0.0 - 1.0                               │
│ ├── lot_multiplier: 1.0 / 0.5 / 0.0                    │
│ └── recommendation: TRADE / REDUCE / SLEEP              │
└──────────────────────────────────────────────────────────┘
```

### XGBoost — Otak Prediksi

```
┌──────────────────────────────────────────────────────────┐
│ XGBOOST BINARY CLASSIFIER                               │
│                                                           │
│ Library: xgboost                                         │
│ Objective: binary:logistic                               │
│ Target: UP (1) / DOWN (0) pada bar berikutnya            │
│                                                           │
│ Anti-Overfitting Config:                                 │
│ ├── max_depth: 3       (shallow trees)                   │
│ ├── learning_rate: 0.05 (slow learning)                  │
│ ├── min_child_weight: 10 (min samples per leaf)          │
│ ├── subsample: 0.7     (70% data per tree)               │
│ ├── colsample_bytree: 0.6 (60% features per tree)       │
│ ├── reg_alpha: 1.0     (L1 regularization)               │
│ ├── reg_lambda: 5.0    (L2 regularization)               │
│ ├── gamma: 1.0         (min loss reduction)              │
│ └── num_boost_round: 50 (few rounds)                     │
│                                                           │
│ 24 Fitur Input (Top 10):                                 │
│ 1. RSI(14)              6. price_position                │
│ 2. MACD_histogram       7. volatility_20                 │
│ 3. ATR(14)              8. returns_5                     │
│ 4. bb_width             9. ema_cross                     │
│ 5. returns_1           10. regime                        │
│                                                           │
│ Output:                                                   │
│ ├── signal: BUY / SELL / HOLD                            │
│ ├── probability: 0.0 - 1.0 (prob of UP)                 │
│ └── confidence: 0.0 - 1.0 (prob of winning side)        │
│                                                           │
│ Validation:                                               │
│ ├── Train/Test: 70% / 30% (50-bar gap, anti leakage)     │
│ ├── Walk-forward: 500 train / 50 test / 50 step         │
│ ├── Target AUC: > 0.65                                   │
│ ├── Rollback AUC: < 0.60 (v4: dinaikkan dari 0.52)       │
│ └── Overfitting ratio: train_AUC/test_AUC < 1.15        │
└──────────────────────────────────────────────────────────┘
```

### Kombinasi Sinyal (SMC + ML)

```
SMC Signal: "BUY at 2645, SL 2635, TP 2665, conf 75%"
ML Signal:  "BUY, confidence 72%"
                    │
                    ▼
┌─ KOMBINASI ──────────────────────────────────────────────┐
│                                                           │
│ CASE 1: SMC BUY + ML BUY (≥50%)                         │
│   → Combined confidence = (75% + 72%) / 2 = 73.5%       │
│   → ENTRY (jika pass 11 filter lainnya)                  │
│                                                           │
│ CASE 2: SMC BUY + ML SELL (≥65%)                         │
│   → ML strongly disagrees → BLOCK (filter #5)            │
│   → TIDAK entry                                          │
│                                                           │
│ CASE 3: SMC BUY + ML uncertain (<50%)                    │
│   → ML tidak yakin → BLOCK (filter #4)                   │
│   → TIDAK entry                                          │
│                                                           │
│ CASE 4: Tidak ada SMC signal                             │
│   → Tidak ada entry point → SKIP                         │
│   → SMC adalah sinyal PRIMER (wajib ada)                 │
│                                                           │
│ PRINSIP:                                                  │
│   SMC = sinyal UTAMA (menentukan entry/SL/TP)            │
│   ML  = KONFIRMASI (bisa memblokir, tidak bisa inisiasi)│
│   HMM = PENYESUAI (mengatur agresivitas)                 │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Smart Money Concepts (SMC)

### 6 Konsep yang Dianalisis

```
┌──────────────────────────────────────────────────────────┐
│ 1. SWING POINTS (Fractal High/Low)                       │
│                                                           │
│    Swing High: titik tertinggi dalam window 11 bar       │
│        ↑                                                  │
│   ____/\____        ← 5 bar kiri lebih rendah            │
│              \____  ← 5 bar kanan lebih rendah           │
│                                                           │
│    Swing Low: titik terendah dalam window 11 bar         │
│   ____      ____    ← 5 bar kiri lebih tinggi            │
│       \____/        ← 5 bar kanan lebih tinggi           │
│        ↑                                                  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 2. FAIR VALUE GAP (FVG) — Ketidakseimbangan Harga       │
│                                                           │
│    Bullish FVG (gap up):                                 │
│    Bar[i-2].high < Bar[i].low  (ada gap)                 │
│                                                           │
│    │   │                                                  │
│    │   │ ← gap (FVG zone)                                │
│    │   ├────┐                                            │
│    ├───┘    │                                            │
│    │        │                                            │
│                                                           │
│    Harga cenderung kembali mengisi FVG → entry zone      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 3. ORDER BLOCK (OB) — Zona Institusi                    │
│                                                           │
│    Bullish OB: candle bearish terakhir sebelum rally     │
│    (zona dimana institusi menempatkan buy order besar)    │
│                                                           │
│              /───\                                        │
│             /     \                                       │
│    ────\   /                                             │
│     OB  \_/ ← entry zone                                │
│                                                           │
│    Lookback: 10 bar untuk deteksi                        │
│    Mitigated: true jika harga sudah revisit              │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 4. BREAK OF STRUCTURE (BOS) — Kelanjutan Trend          │
│                                                           │
│    Uptrend BOS:                                          │
│    Harga break di ATAS swing high sebelumnya             │
│    → Trend bullish berlanjut                             │
│                                                           │
│         /\    /\                                         │
│        /  \  /  \  /\ ← BOS (break above prev high)     │
│       /    \/    \/                                       │
│      /                                                    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 5. CHANGE OF CHARACTER (CHoCH) — Perubahan Trend        │
│                                                           │
│    Uptrend → Downtrend:                                  │
│    Harga break di BAWAH swing low terakhir               │
│    → Trend berubah dari bullish ke bearish               │
│                                                           │
│    /\    /\                                              │
│   /  \  /  \                                             │
│  /    \/    \                                            │
│              \____  ← CHoCH (break below prev low)       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 6. LIQUIDITY ZONES — Target Likuiditas                  │
│                                                           │
│    BSL (Buy Side Liquidity): di atas swing highs         │
│    SSL (Sell Side Liquidity): di bawah swing lows        │
│                                                           │
│    ---- BSL ---- (stop loss para seller berkumpul)       │
│    /\    /\                                              │
│   /  \  /  \                                             │
│  /    \/    \                                            │
│  ---- SSL ---- (stop loss para buyer berkumpul)          │
│                                                           │
│    Institusi sering "hunt" liquidity zone ini            │
└──────────────────────────────────────────────────────────┘
```

### Signal Generation

```
SYARAT SINYAL SMC:

    Structure Break (BOS atau CHoCH)
              +
    Zone (FVG atau Order Block)
              =
    VALID SIGNAL

BUY Signal:
    ├── BOS bullish ATAU CHoCH bearish→bullish
    ├── + Bullish FVG ATAU Bullish OB di bawah harga
    ├── Entry: harga saat ini
    ├── SL: di bawah zone, minimum 1.5 × ATR
    ├── TP: 2:1 R:R, maximum 4 × ATR
    └── Confidence: 55-85% (lebih banyak confluence = lebih tinggi)

SELL Signal:
    ├── BOS bearish ATAU CHoCH bullish→bearish
    ├── + Bearish FVG ATAU Bearish OB di atas harga
    ├── Entry: harga saat ini
    ├── SL: di atas zone, minimum 1.5 × ATR
    ├── TP: 2:1 R:R, maximum 4 × ATR
    └── Confidence: 55-85%
```

---

## 10. Position Lifecycle

### Dari Lahir Sampai Mati (Siklus Hidup Posisi)

```
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 1: SINYAL TERDETEKSI                                    ║
║                                                                ║
║ SMC menemukan setup + ML konfirmasi + 11 filter PASS          ║
║ → Keputusan: BUKA POSISI                                      ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 2: LOT SIZE CALCULATION                                 ║
║                                                                ║
║ Risk Engine (Kelly):                                          ║
║   Balance $5000 × 1% risk = $50 max loss                     ║
║   SL distance 50 pips → lot = 0.01                            ║
║                                                                ║
║ ML Confidence boost:                                          ║
║   ML ≥ 80% → 0.02 lot (double)                               ║
║   ML < 65% → 0.01 lot (minimum)                               ║
║                                                                ║
║ Session multiplier:                                           ║
║   Golden: × 1.2, Sydney: × 0.5                               ║
║                                                                ║
║ Regime multiplier:                                            ║
║   Normal: × 1.0, High Vol: × 0.5, Crisis: × 0.0             ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 3: ORDER VALIDATION                                     ║
║                                                                ║
║ Risk Engine memvalidasi:                                      ║
║   ✓ SL di sisi yang benar (BUY: SL < entry)                  ║
║   ✓ TP di sisi yang benar (BUY: TP > entry)                  ║
║   ✓ Lot dalam range (0.01 - 0.05)                            ║
║   ✓ Entry dekat harga saat ini (< 0.1%)                      ║
║   ✓ Risk% ≤ 1.5× config limit                                ║
║   ✓ Circuit breaker TIDAK aktif                               ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 4: ORDER EXECUTION                                      ║
║                                                                ║
║ MT5 Connector mengirim order:                                 ║
║   → Symbol: XAUUSD                                           ║
║   → Type: BUY/SELL                                            ║
║   → Lot: 0.01-0.02                                           ║
║   → SL: ATR-based (broker level)                              ║
║   → TP: 2:1 R:R (broker level)                               ║
║   → Deviation: 20 points (slippage tolerance)                 ║
║   → Retry: max 3 attempts jika gagal                          ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 5: POSITION REGISTERED                                  ║
║                                                                ║
║ Smart Risk Manager:                                           ║
║   → Catat entry price, direction, lot, timestamp              ║
║   → Inisialisasi peak_profit = 0                              ║
║   → Mulai tracking momentum                                  ║
║                                                                ║
║ Trade Logger:                                                 ║
║   → Insert ke PostgreSQL (30+ field)                          ║
║   → Backup ke CSV                                             ║
║                                                                ║
║ Telegram:                                                     ║
║   → Kirim notifikasi trade open                               ║
║   → Detail: entry, SL, TP, R:R, confidence, regime            ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 6: ACTIVE MONITORING (setiap ~10 detik + candle baru)   ║
║                                                                ║
║ ┌── Update profit/loss real-time                              ║
║ ├── Update peak profit (tertinggi yang pernah dicapai)        ║
║ ├── Hitung momentum (kecepatan perubahan profit)              ║
║ ├── Hitung TP probability                                     ║
║ ├── Cek 10 kondisi exit (lihat bagian 6)                      ║
║ ├── Cek Position Manager (trailing, breakeven)                ║
║ └── Cek Market Close Handler (dekat close?)                   ║
║                                                                ║
║ Setiap ~10 detik (atau candle baru), posisi dievaluasi:       ║
║   → HOLD (lanjut)                                             ║
║   → CLOSE (tutup dengan alasan spesifik)                      ║
╚═══════════════════════════════════════════════════════════════╝
        │
        │  trigger close
        ▼
╔═══════════════════════════════════════════════════════════════╗
║ TAHAP 7: POSITION CLOSED                                      ║
║                                                                ║
║ MT5 Connector:                                                ║
║   → Close position via market order                           ║
║                                                                ║
║ Smart Risk Manager:                                           ║
║   → Record profit/loss                                        ║
║   → Update daily/total loss counters                          ║
║   → Update win/loss streak                                    ║
║   → Check mode transition (NORMAL→RECOVERY→PROTECTED→STOPPED)║
║                                                                ║
║ Trade Logger:                                                 ║
║   → Update trade record: exit price, profit, duration, reason ║
║   → Update PostgreSQL + CSV                                   ║
║                                                                ║
║ Telegram:                                                     ║
║   → Kirim notifikasi trade close                              ║
║   → Detail: profit, duration, exit reason, balance            ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 11. Auto-Retraining & Model Management

### Lifecycle Model AI

```
┌──────────────────────────────────────────────────────────────┐
│              INITIAL TRAINING (train_models.py)               │
│              Dijalankan 1x saat setup                        │
│                                                               │
│  1. Fetch 10,000 bar M15 dari MT5 (~104 hari)                │
│  2. Feature Engineering → 40+ fitur                          │
│  3. SMC Analysis → struktur pasar                            │
│  4. Create target → UP/DOWN (lookahead=1)                    │
│  4b. Split 70/30 dengan 50-bar gap (anti temporal leakage)   │
│  5. Train HMM (3 regime, lookback=500)                       │
│  6. Train XGBoost (50 rounds, early_stop=5)                  │
│  7. Walk-forward validation (500 train/50 test/50 step)      │
│  8. Save → models/hmm_regime.pkl + xgboost_model.pkl         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              DAILY AUTO-RETRAINING (Auto Trainer)             │
│              Otomatis setiap hari 05:00 WIB                  │
│                                                               │
│  Schedule:                                                    │
│  ├── Harian (05:00 WIB): 8,000 bar, 50 rounds               │
│  ├── Weekend (05:00 Sabtu): 15,000 bar, 80 rounds (deep)    │
│  └── Emergency: jika AUC < 0.65 (kualitas turun)            │
│                                                               │
│  Proses:                                                      │
│  1. Backup model saat ini → models/backup/                   │
│  2. Fetch data baru dari MT5                                 │
│  3. Feature Engineering + SMC                                │
│  4. Train HMM baru + XGBoost baru                           │
│  5. Validasi: test AUC ≥ 0.60? (v4: dinaikkan dari 0.52)    │
│     ├── Ya → Save model baru, reload di memory              │
│     └── Tidak → ROLLBACK ke model sebelumnya                │
│  6. Log hasil ke PostgreSQL                                  │
│  7. Kirim laporan via Telegram                               │
│                                                               │
│  Safety:                                                      │
│  ├── Max 5 backup disimpan (rotasi)                          │
│  ├── Min 20 jam antar retrain (cooldown)                     │
│  ├── Auto-rollback jika AUC < 0.60 (v4 threshold)           │
│  └── Model lama selalu tersedia untuk rollback               │
└──────────────────────────────────────────────────────────────┘
```

### Perbandingan Initial vs Auto Training

| Aspek | train_models.py | Auto Trainer |
|-------|-----------------|-------------|
| Kapan | Manual, 1x setup | Otomatis, harian |
| Data | 10,000 bar | 8K (harian) / 15K (weekend) |
| Boost rounds | 50 | 50 (harian) / 80 (weekend) |
| Walk-forward | Ya | Tidak |
| Backup | Tidak | Ya (5 terakhir) |
| Rollback | Tidak | Ya (AUC < 0.60) |
| Database | Tidak | Ya (PostgreSQL) |
| Tujuan | Setup awal | Maintenance rutin |

---

## 12. Infrastruktur & Database

### PostgreSQL Schema

```
trading_db
├── trades                 (Semua trade: open, close, profit, SMC, ML, features)
├── training_runs          (Log setiap training: AUC, akurasi, durasi, rollback)
├── signals                (Setiap sinyal yang dihasilkan: executed atau tidak)
├── market_snapshots       (Snapshot periodik: harga, regime, volatilitas)
├── bot_status             (Status bot: uptime, loop count, balance, risk mode)
└── daily_summaries        (Ringkasan harian: win rate, profit factor, per sesi)
```

### Tabel `trades` (Detail)

```sql
-- Identifikasi
ticket, symbol, direction (BUY/SELL)

-- Harga
entry_price, exit_price, stop_loss, take_profit

-- Hasil
lot_size, profit_usd, profit_pips
opened_at, closed_at, duration_seconds

-- Konteks Entry
entry_regime, entry_volatility, entry_session
smc_signal, smc_confidence, smc_reason
smc_fvg_detected, smc_ob_detected, smc_bos_detected, smc_choch_detected
ml_signal, ml_confidence
market_quality, market_score, dynamic_threshold

-- Konteks Exit
exit_reason, exit_regime, exit_ml_signal

-- Keuangan
balance_before, balance_after, equity_at_entry

-- Data Lengkap
features_entry (JSON), features_exit (JSON)
bot_version, trade_mode
```

### Connection Architecture

```
Bot Components
├── TradeLogger    → TradeRepository, SignalRepository, MarketSnapshotRepository
├── AutoTrainer    → TrainingRepository
├── main_live.py   → BotStatusRepository, DailySummaryRepository
└── Dashboard      → Semua repository (READ)
         │
         ▼
    DatabaseConnection (Singleton)
         │
         ▼
    ThreadedConnectionPool (1-10 koneksi)
         │
         ▼
    PostgreSQL Server
```

### Graceful Degradation

```
PostgreSQL tersedia?
├── Ya → Gunakan DB + CSV backup (dual write)
└── Tidak → CSV saja (bot tetap berjalan 100%)

Bot TIDAK PERNAH crash karena database.
Semua operasi DB dibungkus try-except.
```

---

## 13. Konfigurasi & Parameter Kritis

### Configuration System

```
.env file
    │
    ▼
TradingConfig.from_env()
    │
    ├── RiskConfig
    │   ├── risk_per_trade: 1.0% (SMALL) / 0.5% (MEDIUM)
    │   ├── max_daily_loss: 3.0% (SMALL) / 2.0% (MEDIUM)
    │   ├── max_total_loss: 10.0%
    │   ├── max_positions: 3 (SMALL) / 5 (MEDIUM)
    │   ├── min_lot: 0.01
    │   ├── max_lot: 0.05 (SMALL) / 2.0 (MEDIUM)
    │   └── max_leverage: 1:100 (SMALL) / 1:30 (MEDIUM)
    │
    ├── SMCConfig
    │   ├── swing_length: 5
    │   ├── fvg_min_gap_pips: 2.0
    │   ├── ob_lookback: 10
    │   └── bos_close_break: true
    │
    ├── MLConfig
    │   ├── confidence_threshold: 0.65
    │   ├── entry_confidence: 0.70
    │   ├── high_confidence: 0.75
    │   ├── very_high_confidence: 0.80
    │   └── retrain_frequency_days: 7
    │
    ├── ThresholdsConfig
    │   ├── ml_min_confidence: 0.65
    │   ├── ml_high_confidence: 0.75
    │   ├── trade_cooldown_seconds: 300
    │   ├── min_profit_to_secure: $15
    │   ├── good_profit: $25
    │   ├── great_profit: $40
    │   ├── flash_crash_threshold: 2.5%
    │   └── sydney_lot_multiplier: 0.5
    │
    └── RegimeConfig
        ├── n_regimes: 3
        ├── lookback: 500
        └── retrain_frequency: 20
```

### Capital Mode Auto-Detection

```
Balance ≤ $10,000 → SMALL MODE
    ├── Risk: 1% per trade ($50 pada $5K)
    ├── Daily limit: 3% ($150)
    ├── Lot: 0.01-0.05
    ├── Leverage: 1:100
    ├── Timeframe: M15
    └── Max posisi: 3

Balance > $10,000 → MEDIUM MODE
    ├── Risk: 0.5% per trade
    ├── Daily limit: 2%
    ├── Lot: 0.01-2.0
    ├── Leverage: 1:30
    ├── Timeframe: H1
    └── Max posisi: 5
```

### Session Schedule (WIB = GMT+7)

```
┌────────────────────────────────────────────────────────────┐
│ WAKTU (WIB)  │ SESI          │ LOT MULT │ KETERANGAN      │
├──────────────┼───────────────┼──────────┼─────────────────┤
│ 00:00-04:00  │ DEAD ZONE     │ BLOCKED  │ Likuiditas rendah│
│ 04:00-06:00  │ ROLLOVER      │ BLOCKED  │ Spread melebar  │
│ 06:00-07:00  │ Sydney        │ 0.5x     │ Pasar baru buka │
│ 07:00-13:00  │ Tokyo+Sydney  │ 0.7x     │ Asia aktif      │
│ 13:00-15:00  │ Tokyo akhir   │ 0.7x     │ Transisi        │
│ 15:00-20:00  │ London        │ 1.0x     │ Volatilitas naik │
│ 20:00-23:59  │ ★ GOLDEN TIME │ 1.2x     │ London+NY overlap│
│ Jumat ≥23:00 │ WEEKEND RISK  │ BLOCKED  │ Gap risk        │
└────────────────────────────────────────────────────────────┘

★ Golden Time (20:00-23:59 WIB) = waktu paling optimal
  → Spread ketat, likuiditas maksimal, volatilitas ideal
  → Lot multiplier 1.2x (bonus)
  → v4: Smart Hold dihapus — tidak ada lagi hold losers menunggu sesi tertentu
```

---

## 14. Performa & Timing

### Main Loop Breakdown

```
Target: < 50ms per iterasi analisis

FULL ANALYSIS (saat candle baru M15):
┌────────────────────────────────────────────────────────────┐
│ Komponen               │ Waktu    │ Keterangan             │
├────────────────────────┼──────────┼────────────────────────┤
│ MT5 data fetch         │  ~10ms   │ 200 bar M15 via API    │
│ Feature engineering    │   ~5ms   │ 40+ fitur, Polars      │
│ SMC analysis           │   ~5ms   │ 6 konsep, Polars native│
│ HMM predict            │   ~2ms   │ 2 fitur → 1 regime    │
│ XGBoost predict        │   ~3ms   │ 24 fitur → 1 signal   │
│ Position monitoring    │   ~5ms   │ Per posisi terbuka     │
│ Entry logic            │   ~5ms   │ 11 filter check        │
│ Overhead               │  ~15ms   │ Logging, state update  │
├────────────────────────┼──────────┼────────────────────────┤
│ TOTAL                  │  ~50ms   │                        │
└────────────────────────────────────────────────────────────┘

POSITION CHECK ONLY (di antara candle, setiap ~10 detik):
┌────────────────────────────────────────────────────────────┐
│ Komponen               │ Waktu    │ Keterangan             │
├────────────────────────┼──────────┼────────────────────────┤
│ MT5 data fetch         │   ~5ms   │ 50 bar saja            │
│ Feature engineering    │   ~3ms   │ Minimal fitur          │
│ ML prediction          │   ~3ms   │ Untuk exit evaluation  │
│ Position evaluation    │   ~5ms   │ 10 kondisi exit        │
│ Overhead               │   ~5ms   │ Logging                │
├────────────────────────┼──────────┼────────────────────────┤
│ TOTAL                  │  ~21ms   │                        │
└────────────────────────────────────────────────────────────┘
```

### Timer Periodik

```
┌──────────────────────────────────────────────────────────┐
│ Event                  │ Interval       │ Cara Trigger    │
├────────────────────────┼────────────────┼─────────────────┤
│ Full analysis + entry  │ Candle baru M15│ Deteksi candle  │
│ Position monitoring    │ ~10 detik      │ Di antara candle│
│ Performance logging    │ 4 candle (~1j) │ candle_count % 4│
│ Auto-retrain check     │ 20 candle (~5j)│ candle_count %20│
│ Market update Telegram │ 30 menit       │ Timer           │
│ Hourly analysis        │ 1 jam          │ Timer           │
│ Daily summary + reset  │ Ganti hari     │ Date check      │
└──────────────────────────────────────────────────────────┘
```

---

## 15. Error Handling & Fault Tolerance

### Prinsip: Bot TIDAK PERNAH Crash

```
┌──────────────────────────────────────────────────────────┐
│ LEVEL 1: Per-Loop Error Handling                         │
│                                                           │
│ try:                                                     │
│     # Fetch data, analyze, trade                         │
│ except ConnectionError:                                  │
│     # MT5 disconnected → reconnect()                     │
│ except Exception as e:                                   │
│     # Log error → lanjut loop berikutnya                 │
│     # Bot TIDAK crash dari error tunggal                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ LEVEL 2: MT5 Auto-Reconnect                             │
│                                                           │
│ MT5 putus?                                               │
│ ├── Attempt 1: reconnect (tunggu 2 detik)                │
│ ├── Attempt 2: reconnect (tunggu 4 detik)                │
│ ├── Attempt 3: reconnect (tunggu 8 detik)                │
│ ├── Cooldown 60 detik                                    │
│ └── Retry cycle (max 5 per cooldown)                     │
│                                                           │
│ Selama disconnected:                                     │
│   → Position monitoring PAUSE                            │
│   → Entry baru DITUNDA                                   │
│   → Posisi terbuka dilindungi broker SL (lapis 1)        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ LEVEL 3: Database Graceful Degradation                   │
│                                                           │
│ PostgreSQL down?                                         │
│ ├── Switch ke CSV-only mode                              │
│ ├── Semua data tetap dicatat                             │
│ ├── Trading tetap berjalan normal                        │
│ └── Retry DB connection periodik                         │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ LEVEL 4: Telegram Failure                                │
│                                                           │
│ Telegram API error?                                      │
│ ├── Log error secara silent                              │
│ ├── Trading tetap jalan 100%                             │
│ └── Retry di notifikasi berikutnya                       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ LEVEL 5: Model File Missing                              │
│                                                           │
│ .pkl file tidak ditemukan?                                │
│ ├── Log warning                                          │
│ ├── Skip prediksi (ML/HMM)                               │
│ ├── Trading bisa jalan tanpa ML (SMC only)               │
│ └── Trigger: jalankan train_models.py                    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ LEVEL 6: Flash Crash Protection                          │
│                                                           │
│ Harga bergerak > 2.5% dalam 1 menit?                    │
│ ├── EMERGENCY: Close ALL positions                       │
│ ├── Circuit breaker AKTIF                                │
│ ├── Kirim alert KRITIS via Telegram                      │
│ └── Bot masuk mode STOPPED                               │
└──────────────────────────────────────────────────────────┘
```

### Startup & Shutdown

```
STARTUP SEQUENCE:
  1. Load konfigurasi dari .env
  2. Connect ke MT5 (max 3 retry)
  3. Load model HMM dari models/hmm_regime.pkl
  4. Load model XGBoost dari models/xgboost_model.pkl
  5. Initialize SmartRiskManager (set balance, limits)
  6. Initialize SessionFilter (WIB timezone)
  7. Initialize TelegramNotifier
  8. Initialize TradeLogger (connect DB)
  9. Initialize AutoTrainer
  10. Send Telegram: "BOT STARTED" (config, balance, risk settings)
  11. Mulai main loop

SHUTDOWN SEQUENCE (SIGINT/SIGTERM):
  1. Signal diterima
  2. Hentikan loop utama
  3. Kirim Telegram: "BOT STOPPED" (balance, trades, uptime)
  4. Disconnect MT5
  5. Close database connections
  6. Exit
```

---

## 16. Daftar File Source Code

```
Smart Automatic Trading BOT + AI/
│
├── main_live.py                    # Orchestrator utama (TradingBot)
├── train_models.py                 # Script training awal
├── .env                            # Environment variables (credentials)
│
├── src/
│   ├── config.py                   # Konfigurasi terpusat (6 sub-config)
│   ├── mt5_connector.py            # Bridge ke MetaTrader 5
│   ├── feature_eng.py              # Feature Engineering (40+ fitur)
│   ├── regime_detector.py          # HMM Regime Detection (3 state)
│   ├── ml_model.py                 # XGBoost Signal Predictor
│   ├── smc_polars.py               # Smart Money Concepts (6 konsep)
│   ├── smart_risk_manager.py       # 4-Mode Risk Manager
│   ├── risk_engine.py              # Kelly Criterion + Circuit Breaker
│   ├── session_filter.py           # Session Time Filter (WIB)
│   ├── dynamic_confidence.py       # Dynamic Threshold Manager
│   ├── news_agent.py               # News Event Monitor
│   ├── telegram_notifier.py        # Telegram Push Notifications
│   ├── auto_trainer.py             # Daily Auto-Retraining
│   ├── trade_logger.py             # Dual Storage Logger (DB+CSV)
│   ├── position_manager.py         # Position Manager + Market Close
│   │
│   └── db/
│       ├── __init__.py             # DB exports
│       ├── connection.py           # PostgreSQL Singleton + Pool
│       └── repository.py           # 6 Repository classes
│
├── models/
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── hmm_regime.pkl              # Trained HMM model
│   └── backup/                     # Auto-backup (5 terakhir)
│
├── data/
│   ├── training_data.parquet       # Data training terakhir
│   └── trade_logs/                 # CSV backup (per bulan)
│       ├── trades_2025_01.csv
│       ├── trades_2025_02.csv
│       └── ...
│
├── backtests/
│   └── backtest_live_sync.py       # Backtest 100% sync live
│
├── logs/
│   └── training_YYYY-MM-DD.log     # Log training detail
│
└── docs/
    └── arsitektur-ai/
        ├── 00-ARSITEKTUR-LENGKAP.md  # Dokumen ini
        ├── README.md                  # Index komponen
        └── 01-23 (per komponen)       # Detail per modul
```

---

## Ringkasan Eksekutif

**Smart AI Trading Bot** adalah sistem trading otomatis yang menggabungkan:

1. **Smart Money Concepts (SMC)** sebagai sinyal UTAMA — mendeteksi zona institusi (FVG, Order Block, BOS, CHoCH) untuk menentukan entry, SL, dan TP yang presisi.

2. **XGBoost Machine Learning** sebagai KONFIRMASI — memprediksi arah harga dengan 24 fitur teknikal, memblokir trade jika tidak setuju dengan SMC.

3. **Hidden Markov Model (HMM)** sebagai PENYESUAI — mendeteksi kondisi pasar (tenang/volatile/krisis) untuk menyesuaikan agresivitas.

4. **4-Lapis Proteksi Risiko** — dari broker SL, software smart exit, emergency stop, hingga circuit breaker. Lot ultra-kecil (0.01-0.02) memastikan kerugian per trade maximum $50 (1%).

5. **Self-Improving** — model AI dilatih ulang otomatis setiap hari dengan auto-rollback jika model baru lebih buruk.

6. **Fault-Tolerant** — bot tidak pernah crash. MT5 putus? Auto-reconnect. Database mati? CSV fallback. Error? Log dan lanjut.

Semua ini dikoordinasikan oleh **Main Live Orchestrator** yang menjalankan loop **candle-based** — analisis penuh hanya saat candle M15 baru terbentuk (~50ms per iterasi), dengan pengecekan posisi setiap ~10 detik di antara candle (~21ms). Mengevaluasi 11 filter entry dan 10 kondisi exit secara real-time, dengan notifikasi Telegram untuk setiap kejadian penting.

```
TARGET: Trading XAUUSD M15 yang KONSISTEN dan AMAN
        dengan kerugian terkontrol dan profit teroptimasi.
```
