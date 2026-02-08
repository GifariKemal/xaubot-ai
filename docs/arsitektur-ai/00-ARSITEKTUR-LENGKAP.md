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
5. [Alur Entry: 14 Filter](#5-alur-entry-14-filter)
6. [Alur Exit: 12 Kondisi](#6-alur-exit-12-kondisi)
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

```mermaid
graph TD
    subgraph MAIN["MAIN LIVE — Orchestrator<br/>main_live.py — TradingBot<br/>Candle-based M15 + position check ~10 detik"]
        direction TB

        subgraph P1["PHASE 1: DATA"]
            MT5C["MT5 Connector<br/>(broker)"] --> FEng["Feature Engine<br/>(40+ fitur)"]
            FEng --> SMCA["SMC Analyzer<br/>(institusi)"]
            MT5C ~~~ HMMD["HMM Regime<br/>(3 state)"]
            FEng --> XGB["XGBoost ML Model<br/>(prediksi)"]
            SMCA --> XGB
            HMMD --> DC["Dynamic Confidence<br/>(threshold)"]
        end

        subgraph P2["PHASE 2: MONITORING"]
            PM["Position Manager<br/>(trailing)"]
            SRM["Smart Risk Manager"]
            RiskE["Risk Engine<br/>(Kelly)"]
        end

        subgraph P3["PHASE 3: ENTRY"]
            SF["Session Filter<br/>(waktu)"]
            NA["News Agent<br/>(berita)"]
            EF["14-Gate Entry Filter"] --> EXEC["EXECUTE"]
        end

        subgraph P4["PHASE 4: PERIODIK"]
            AT["Auto Trainer<br/>(retrain)"]
            TN["Telegram Notifier<br/>(laporan)"]
            TL["Trade Logger<br/>(DB+CSV)"]
        end

        P1 --> P2
        P2 --> P3
        P3 --> P4
        P4 --> DB["PostgreSQL + CSV Backup"]
    end
```

### Alur Data (Data Flow)

```mermaid
flowchart TD
    MT5B["MT5 Broker<br/>(XAUUSD M15)"] -->|"200 bar OHLCV"| MT5C["MT5 Connector<br/>numpy → Polars (tanpa Pandas)"]
    MT5C --> FE["Feature Engineer<br/>OHLCV → 40+ fitur teknikal<br/>RSI, ATR, MACD, BB, EMA, Volume"]
    FE --> SMC["SMC Analyzer"]
    FE --> HMM["HMM Regime Detect"]
    SMC --> XGB["XGBoost ML Predictor<br/>24 fitur → BUY/SELL/HOLD<br/>+ Confidence 0-100%"]
    HMM --> XGB
    XGB --> DCM["Dynamic Confidence<br/>Threshold berdasarkan sesi,<br/>regime, volatilitas, trend"]
    DCM --> SC["Signal Combiner (14 Filter)<br/>SMC + ML harus setuju<br/>+ Session + Risk + Cooldown"]
    SC --> PASS{"PASS?"}
    PASS -->|"NO"| WAIT["Tunggu loop berikutnya"]
    PASS -->|"YES"| RE["Risk Engine + Risk Manager<br/>Kelly Criterion → lot size<br/>Validasi order → approve/reject"]
    RE --> EXEC["Execute Order via MT5<br/>Kirim ke MT5 dengan SL dan TP<br/>Register ke Position Manager"]
    EXEC --> TN["Telegram Notifier"]
    EXEC --> TL["Trade Logger<br/>(DB + CSV)"]
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
| 5 | Smart Risk Manager | `src/smart_risk_manager.py` | Risiko | 4 mode *trading*, 12 kondisi *exit* |
| 6 | Session Filter | `src/session_filter.py` | Filter | Waktu trading optimal (WIB) |
| 7 | Stop Loss (4 Lapis) | Multi-file | Proteksi | SMC → Software → Emergency → Circuit |
| 8 | Take Profit (6 Layer) | Multi-file | Proteksi | Hard → Momentum → Peak → Probability → Early → Broker |
| 9 | Entry Trade | `main_live.py` | Eksekusi | 14 *filter* berurutan |
| 10 | Exit Trade | `main_live.py` | Eksekusi | 12 kondisi *exit real-time* |
| 11 | News Agent | `src/news_agent.py` | Monitor | **NONAKTIF** — dikomentari di kode |
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

```mermaid
flowchart TD
    CONFIG["CONFIGURATION (17)<br/>Sumber parameter semua"] -->|"dikonsumsi oleh semua"| MT5_16
    CONFIG --> FE4
    CONFIG --> SMC3

    MT5_16["MT5 (16)<br/>Broker"] --> FE4["FeatEng (4)<br/>40+ fitur"]
    FE4 --> SMC3["SMC (3)<br/>Institusi"]
    FE4 --> HMM1["HMM (1)<br/>Regime"]
    FE4 --> XGB2["XGBoost (2)<br/>Prediksi"]
    SMC3 --> XGB2

    HMM1 --> DC15["Dynamic Confidence (15)<br/>Threshold adaptif"]
    XGB2 --> DC15

    SESSION6["Session (6)<br/>Waktu"] --> ENTRY9["ENTRY TRADE (9)<br/>14 Filter Gate"]
    DC15 --> ENTRY9
    NEWS11["News (11)<br/>Berita"] --> ENTRY9

    ENTRY9 --> RISK20["RiskEng (20)<br/>Kelly Lot"]
    ENTRY9 --> SRISK5["SmartRisk (5)<br/>4 Mode"]

    RISK20 --> EXEC["EXECUTE ORDER<br/>via MT5 (16)"]
    SRISK5 --> EXEC

    EXEC --> PM19["PosMgr (19)<br/>Trailing"]
    EXEC --> LOG18["Logger (18)<br/>DB+CSV"]
    EXEC --> TG12["Telegram (12)"]

    PM19 --> EXIT10["EXIT (10)<br/>12 Kondisi"]
    LOG18 --> DB21["DB (21)<br/>PostgreSQL"]

    subgraph PERIODIK["Periodik"]
        AT13["AutoTrain (13)<br/>Harian"]
        BT14["Backtest (14)<br/>Validasi"]
        TM22["Train Models (22)<br/>Setup awal"]
    end
```

---

## 4. Pipeline Data

### Dari OHLCV Mentah ke Keputusan Trading

#### Tahap 1: Data Fetching (MT5 Connector)

```mermaid
flowchart TD
    MT5B["MT5 Broker"] -->|"mt5.copy_rates_from_pos<br/>(XAUUSD, M15, 0, 200)"| NPA["NumPy Structured Array"]
    NPA -->|"Konversi langsung ke Polars<br/>(TANPA Pandas)"| PDF["Polars DataFrame:<br/>time (i64), open (f64), high (f64),<br/>low (f64), close (f64),<br/>tick_volume (f64), spread (f64)"]
```

**Kenapa Polars, bukan Pandas?**
- 3-5x lebih cepat untuk operasi vectorized
- Memory-efficient (zero-copy)
- Native lazy evaluation
- Konsisten di seluruh codebase (tidak ada konversi bolak-balik)

#### Tahap 2: Feature Engineering (40+ Fitur)

```mermaid
flowchart TD
    INPUT["Input: Polars DataFrame<br/>(200 bar OHLCV)"] --> MOM["Momentum Indicators"]
    INPUT --> VOL["Volatility Indicators"]
    INPUT --> TREND["Trend Indicators"]
    INPUT --> PA["Price Action"]
    INPUT --> VOLF["Volume Features"]
    INPUT --> LAG["Lag Features"]
    INPUT --> TIME["Time Features"]

    MOM --> M1["RSI(14) - 0-100, overbought/oversold"]
    MOM --> M2["MACD(12,26,9) - trend strength and direction"]
    MOM --> M3["MACD Histogram - momentum acceleration"]

    VOL --> V1["ATR(14) - average true range (pips)"]
    VOL --> V2["Bollinger Bands(20,2.0) - upper, lower, width"]
    VOL --> V3["Volatility(20) - rolling std of returns"]

    TREND --> T1["EMA(9) / EMA(21) - fast/slow crossover"]
    TREND --> T2["EMA Cross Signal - 1 bullish / -1 bearish"]
    TREND --> T3["SMA(20) - simple moving average"]

    PA --> P1["Returns(1,5,20) - % perubahan harga"]
    PA --> P2["Log Returns - untuk distribusi normal"]
    PA --> P3["Price Position - posisi dalam range BB"]
    PA --> P4["Higher High/Lower Low count - trend structure"]

    VOLF --> VF1["Volume SMA(20) - rata-rata volume"]
    VOLF --> VF2["Volume Ratio - current / average"]

    LAG --> L1["close_lag_1..5 - harga sebelumnya"]
    LAG --> L2["returns_lag_1..3 - return sebelumnya"]

    TIME --> TI1["Hour, Weekday - waktu candle"]
    TIME --> TI2["Session flags - london, ny, overlap"]

    M1 & M2 & M3 & V1 & V2 & V3 & T1 & T2 & T3 & P1 & P2 & P3 & P4 & VF1 & VF2 & L1 & L2 & TI1 & TI2 --> OUTPUT["Output: DataFrame + 40 kolom baru<br/>(semua numerik, siap ML)"]
```

**Minimum data:** 26 bar untuk semua indikator stabil

#### Tahap 3: SMC Analysis (Pola Institusi)

```mermaid
flowchart TD
    INPUT["Input: DataFrame dengan OHLCV"] --> SP["Swing Points (Fractal)<br/>Window: 11 bar (swing_length=5, +/-5 dari tengah)<br/>Output: swing_high (1/0), swing_low (-1/0), level harga"]
    INPUT --> FVG["Fair Value Gaps (FVG)<br/>Bullish: bar i-2 high < bar i low (gap up)<br/>Bearish: bar i-2 low > bar i high (gap down)<br/>Output: fvg_bull, fvg_bear, fvg_top, fvg_bottom, fvg_mid"]
    INPUT --> OB["Order Blocks (OB)<br/>Lookback: 10 bar<br/>Bullish: candle bearish terakhir sebelum move up besar<br/>Bearish: candle bullish terakhir sebelum move down besar<br/>Output: ob (1/-1), ob_top, ob_bottom, ob_mitigated"]
    INPUT --> BOS["Break of Structure (BOS)<br/>Harga break swing high/low = trend continuation<br/>Output: bos (1/-1), level yang di-break"]
    INPUT --> CHOCH["Change of Character (CHoCH)<br/>Harga break berlawanan arah trend = reversal signal<br/>Output: choch (1/-1), level yang di-break"]
    INPUT --> LIQ["Liquidity Zones<br/>BSL: Buy Side Liquidity (above swing highs)<br/>SSL: Sell Side Liquidity (below swing lows)<br/>Output: bsl_level, ssl_level"]

    SP & FVG & OB & BOS & CHOCH & LIQ --> SIG["Signal Generation<br/>Syarat: Structure break + (FVG ATAU Order Block)"]

    SIG --> ENTRY["Entry: harga saat ini"]
    SIG --> SL["SL: ATR-based, minimum 1.5 x ATR dari entry"]
    SIG --> TP["TP: 2:1 Risk-Reward minimum, cap 4 x ATR"]
    SIG --> CONF["Confidence: 40-85%<br/>(v5: calibrated weighted scoring)"]
    SIG --> REASON["Reason: BOS + Bullish FVG at 2645.50"]
```

#### Tahap 4: Regime Detection (HMM)

```mermaid
flowchart TD
    INPUT["Input: log_returns + normalized_range (volatilitas)"] -->|"GaussianHMM(n_components=3, lookback=500)"| REGIME["3 Regime Output"]

    REGIME --> R0["REGIME 0: Low Volatility<br/>Pasar tenang, range kecil<br/>Lot multiplier: 1.0x (normal)<br/>Rekomendasi: TRADE"]
    REGIME --> R1["REGIME 1: Medium Volatility<br/>Pasar aktif, trend jelas<br/>Lot multiplier: 1.0x (normal)<br/>Rekomendasi: TRADE"]
    REGIME --> R2["REGIME 2: High Volatility<br/>Pasar sangat volatile, berbahaya<br/>Lot multiplier: 0.5x (setengah)<br/>Rekomendasi: REDUCE"]
    REGIME --> RC["CRISIS (FlashCrashDetector)<br/>Move > 2.5% dalam 1 menit<br/>Lot multiplier: 0.0x (STOP)<br/>Rekomendasi: EMERGENCY CLOSE ALL"]

    style R0 fill:#4CAF50,color:#fff
    style R1 fill:#2196F3,color:#fff
    style R2 fill:#FF9800,color:#fff
    style RC fill:#F44336,color:#fff
```

#### Tahap 5: ML Prediction (XGBoost)

```mermaid
flowchart TD
    INPUT["Input: 24 fitur terpilih dari<br/>Feature Engineering + SMC + Regime"] -->|"XGBoost Binary Classifier<br/>max_depth=3, learning_rate=0.05<br/>min_child_weight=10, subsample=0.7<br/>colsample_bytree=0.6<br/>reg_alpha=1.0 (L1), reg_lambda=5.0 (L2)"| OUTPUT["Output:<br/>prob_up: 0.72 (probabilitas naik)<br/>prob_down: 0.28 (probabilitas turun)"]

    OUTPUT --> SIG["Signal: BUY (prob_up > 0.50)<br/>Confidence: 72%"]
    SIG --> TH1["prob > 0.50 - ada sinyal (minimum)"]
    SIG --> TH2["prob > 0.65 - sinyal kuat"]
    SIG --> TH3["prob > 0.75 - sinyal sangat kuat"]
    SIG --> TH4["prob > 0.80 - lot bisa naik ke 0.02"]
```

#### Tahap 6: Dynamic Confidence (Threshold Adaptif)

```mermaid
flowchart TD
    BASE["Base Score: 50"] --> SESS["Session Modifier<br/>Golden Time (20:00-23:59 WIB): +20<br/>London (15:00-23:59): +15<br/>New York (20:00-05:00): +10<br/>Tokyo/Sydney: +0<br/>Market Closed: -30"]
    BASE --> REG["Regime Modifier<br/>Medium Volatility: +15<br/>Low Volatility: +5<br/>High Volatility: -5<br/>Crisis: -25"]
    BASE --> VOLM["Volatility Modifier<br/>Medium (ideal): +10<br/>Low: +0<br/>High: -5<br/>Extreme: -10"]
    BASE --> TREN["Trend Modifier<br/>Trending (jelas): +10<br/>Ranging (sideways): -5"]
    BASE --> SMCC["SMC Confluence<br/>Ada konfluensi: +10<br/>Tidak ada: +0"]
    BASE --> MLC["ML Confidence<br/>ge 70%: +5<br/>ge 60%: +2<br/>lt 60%: +0"]

    SESS & REG & VOLM & TREN & SMCC & MLC --> TOTAL["Total Score (0-100)"]

    TOTAL --> EXC["EXCELLENT (ge 80 poin)<br/>Threshold: 60% (longgar)"]
    TOTAL --> GOOD["GOOD (65-79)<br/>Threshold: 65%"]
    TOTAL --> MOD["MODERATE (50-64)<br/>Threshold: 70%"]
    TOTAL --> POOR["POOR (35-49)<br/>Threshold: 80% (ketat)"]
    TOTAL --> AVOID["AVOID (lt 35)<br/>Threshold: 85% (SKIP)"]

    style EXC fill:#4CAF50,color:#fff
    style GOOD fill:#8BC34A,color:#fff
    style MOD fill:#FF9800,color:#fff
    style POOR fill:#FF5722,color:#fff
    style AVOID fill:#F44336,color:#fff
```

Contoh: Golden Time + Medium Vol + Trending + SMC + ML 72%
= 50 + 20 + 15 + 10 + 10 + 10 + 5 = 120 (cap 100)
= EXCELLENT -> Threshold 60% -> ML 72% PASS

---

## 5. Alur Entry: 14 Filter

Setiap sinyal harus melewati **14 gerbang berurutan**. Satu saja gagal = TIDAK trading.

```mermaid
flowchart TD
    S["SINYAL SMC + ML MASUK"] --> F1{"Filter 1: Session<br/>Jam trading dibolehkan?<br/>BLOCK: 00:00-06:00 WIB (dead zone)<br/>BLOCK: Jumat ge 23:00 WIB"}
    F1 -->|"BLOCK"| SKIP1["SKIP"]
    F1 -->|"PASS: London/NY/Golden Time"| F2{"Filter 2: Risk Mode<br/>Bukan STOPPED?<br/>BLOCK: daily/total limit hit"}
    F2 -->|"BLOCK"| SKIP2["SKIP"]
    F2 -->|"PASS: NORMAL/RECOVERY/PROTECTED"| F3{"Filter 3: SMC Signal<br/>Ada setup SMC valid?<br/>BLOCK: Tidak ada FVG/OB + BOS/CHoCH"}
    F3 -->|"BLOCK"| SKIP3["SKIP"]
    F3 -->|"PASS: Ada sinyal BUY/SELL + SL + TP"| F4{"Filter 4: ML Confidence<br/>ML confidence ge dynamic threshold?<br/>Threshold 60-85% tergantung quality"}
    F4 -->|"BLOCK"| SKIP4["SKIP"]
    F4 -->|"PASS: confidence ge threshold"| F5{"Filter 5: ML Agreement<br/>ML TIDAK strongly disagree?<br/>BLOCK: ML > 65% berlawanan arah SMC"}
    F5 -->|"BLOCK"| SKIP5["SKIP"]
    F5 -->|"PASS: ML setuju atau netral"| F6{"Filter 6: Market Quality<br/>Dynamic Confidence bukan AVOID?<br/>BLOCK: Quality == AVOID (score lt 35)"}
    F6 -->|"BLOCK"| SKIP6["SKIP"]
    F6 -->|"PASS: EXCELLENT/GOOD/MODERATE/POOR"| F7{"Filter 7: Signal Confirmation<br/>Sinyal konsisten 2x berturut?<br/>BLOCK: Baru muncul 1x"}
    F7 -->|"BLOCK"| SKIP7["SKIP"]
    F7 -->|"PASS: 2x berturut"| F8{"Filter 8: Pullback (v5: ATR-based)<br/>Momentum selaras?<br/>BLOCK: RSI overbought/oversold<br/>BLOCK: MACD berlawanan<br/>BLOCK: Bounce > 15% ATR (3 candle)"}
    F8 -->|"BLOCK"| SKIP8["SKIP"]
    F8 -->|"PASS: Momentum selaras"| F9{"Filter 9: Trade Cooldown<br/>Sudah ge 5 menit sejak trade terakhir?<br/>BLOCK: lt 300 detik"}
    F9 -->|"BLOCK"| SKIP9["SKIP"]
    F9 -->|"PASS: ge 300 detik"| F10{"Filter 10: Position Limit<br/>Posisi terbuka lt limit?<br/>BLOCK: Sudah 2+ posisi terbuka"}
    F10 -->|"BLOCK"| SKIP10["SKIP"]
    F10 -->|"PASS: lt 2 posisi"| F11{"Filter 11: Lot Size<br/>Lot size > 0?<br/>BLOCK: Lot = 0 (crisis/risk tinggi)"}
    F11 -->|"BLOCK"| SKIP11["SKIP"]
    F11 -->|"PASS: Lot ge 0.01"| F12{"Filter 12: Flash Crash Guard<br/>TIDAK ada flash crash?<br/>BLOCK: Move > 2.5% dalam 1 menit"}
    F12 -->|"BLOCK + CLOSE ALL"| SKIP12["SKIP"]
    F12 -->|"PASS: Normal"| F13{"Filter 13: H1 Bias (#31B)<br/>EMA20 H1 selaras arah sinyal?<br/>BLOCK: BUY tapi harga lt EMA20 H1<br/>BLOCK: SELL tapi harga > EMA20 H1"}
    F13 -->|"BLOCK"| SKIP13["SKIP"]
    F13 -->|"PASS: Selaras"| F14{"Filter 14: Time Filter (#34A)<br/>Bukan jam transisi?<br/>BLOCK: Jam 9 atau 21 WIB"}
    F14 -->|"BLOCK"| SKIP14["SKIP"]
    F14 -->|"PASS"| EXEC["EXECUTE TRADE<br/>BUY atau SELL<br/>via MT5 Connector"]

    EXEC --> REG["Register ke SmartRiskManager"]
    EXEC --> LOG["Log ke TradeLogger (DB + CSV)"]
    EXEC --> TG["Kirim notifikasi Telegram"]

    style EXEC fill:#4CAF50,color:#fff,stroke:#388E3C,stroke-width:3px
```

---

## 6. Alur Exit: 12 Kondisi

Setiap posisi terbuka dievaluasi **setiap ~10 detik** (di antara candle) atau **setiap candle baru** (full analysis) terhadap 12 kondisi exit:

```mermaid
flowchart TD
    POS["POSISI TERBUKA<br/>(dicek setiap ~10 detik)<br/>Update: profit, momentum, peak, durasi"] --> K1{"Kondisi 1: Smart Take Profit<br/>(a) Profit ge $40 = hard TP<br/>(b) Profit ge $25 + momentum lt -30<br/>(c) Peak > $30, sekarang lt 60% peak<br/>(d) Profit ge $20 + TP prob lt 25%"}
    K1 -->|"TRIGGER"| CLOSE1["TUTUP: Smart TP"]
    K1 -->|"tidak trigger"| K2{"Kondisi 2: Early Exit (Profit Kecil)<br/>Profit $5-$15 + ML reversal ge 65%<br/>+ momentum lt -50"}
    K2 -->|"TRIGGER"| CLOSE2["TUTUP: Amankan profit kecil"]
    K2 -->|"tidak trigger"| K3{"Kondisi 3: Early Cut<br/>(v4: Smart Hold DIHAPUS)<br/>Loss ge 30% max ($15)<br/>DAN momentum lt -30"}
    K3 -->|"TRIGGER"| CLOSE3["TUTUP CEPAT: Early cut"]
    K3 -->|"tidak trigger"| K4{"Kondisi 4: ML Trend Reversal<br/>ML confidence ge 65%<br/>BERLAWANAN ARAH posisi<br/>+ 3x warning berturut-turut"}
    K4 -->|"TRIGGER"| CLOSE4["TUTUP: AI deteksi pembalikan"]
    K4 -->|"tidak trigger"| K5{"Kondisi 5: Maximum Loss<br/>Loss ge 50% dari max_loss_per_trade<br/>($25 dari $50)"}
    K5 -->|"TRIGGER"| CLOSE5["TUTUP: Kerugian terlalu besar"]
    K5 -->|"tidak trigger"| K6{"Kondisi 6: Stall Detection<br/>Posisi 10+ bar<br/>tanpa profit signifikan"}
    K6 -->|"TRIGGER"| CLOSE6["TUTUP: Pasar tidak bergerak"]
    K6 -->|"tidak trigger"| K7{"Kondisi 7: Daily Limit<br/>Total daily loss<br/>mendekati limit"}
    K7 -->|"TRIGGER"| CLOSE7["TUTUP SEMUA: Proteksi modal"]
    K7 -->|"tidak trigger"| K8{"Kondisi 8: Weekend Close<br/>Dekat market close weekend<br/>+ ada posisi profit"}
    K8 -->|"TRIGGER"| CLOSE8["TUTUP: Hindari gap risk"]
    K8 -->|"tidak trigger"| K9{"Kondisi 9: Smart Time-Based<br/>(v5: Dont Cut Winners)<br/>(a) > 4 jam + no growth = TUTUP<br/>(b) > 4 jam + growing + ML = HOLD<br/>(c) > 6 jam + profit lt $10 = TUTUP<br/>(d) > 6 jam + profit > $10 = extend 8j<br/>(e) > 8 jam = TUTUP final"}
    K9 -->|"TRIGGER"| CLOSE9["TUTUP: Time-based"]
    K9 -->|"tidak trigger"| K10{"Kondisi 10: Trailing Stop Loss<br/>Profit ge 25 pips?<br/>SL ikuti harga jarak 10 pips"}
    K10 -->|"TRIGGER"| CLOSE10["TRAILING: Kunci profit"]
    K10 -->|"tidak trigger"| K11{"Kondisi 11: Breakeven Protection<br/>Profit ge 15 pips?<br/>SL ke entry + 2 poin buffer"}
    K11 -->|"TRIGGER"| CLOSE11["BREAKEVEN: Posisi aman"]
    K11 -->|"tidak trigger"| K12["Kondisi 12: Default HOLD<br/>Tidak ada kondisi terpenuhi<br/>Biarkan posisi berjalan"]

    style CLOSE1 fill:#F44336,color:#fff
    style CLOSE2 fill:#F44336,color:#fff
    style CLOSE3 fill:#F44336,color:#fff
    style CLOSE4 fill:#F44336,color:#fff
    style CLOSE5 fill:#F44336,color:#fff
    style CLOSE6 fill:#F44336,color:#fff
    style CLOSE7 fill:#F44336,color:#fff
    style CLOSE8 fill:#F44336,color:#fff
    style CLOSE9 fill:#F44336,color:#fff
    style CLOSE10 fill:#FF9800,color:#fff
    style CLOSE11 fill:#FF9800,color:#fff
    style K12 fill:#4CAF50,color:#fff
```

### *Position Manager* (Tambahan per Posisi)

Selain 12 kondisi di atas, *Position Manager* juga menjalankan:

```mermaid
flowchart TD
    PM["Position Manager (per posisi)"] --> MCH{"Market Close Handler<br/>(Prioritas Tertinggi)<br/>Dekat close harian/weekend?"}
    MCH -->|"Profit ge $10 + dekat close"| CLOSE_MC["TUTUP (amankan)"]
    MCH -->|"Loss + weekend + SL > 50%"| CLOSE_GAP["TUTUP (gap risk)"]
    MCH -->|"Loss kecil + weekend"| HOLD_MC["HOLD (bisa recovery)"]
    MCH -->|"Tidak dekat close"| BE{"Breakeven Protection<br/>Profit ge 15 pips?"}
    BE -->|"Ya"| BE_ACT["Pindah SL ke entry + 2 buffer<br/>(posisi tidak bisa rugi lagi)"]
    BE -->|"Tidak"| TS{"Trailing Stop<br/>Profit ge 25 pips?"}
    TS -->|"Ya"| TS_ACT["SL mengikuti harga, jarak 10 pips<br/>(kunci profit sambil biarkan berjalan)"]
    TS -->|"Tidak"| CONT["Lanjut monitoring"]
```

---

## 7. Sistem Proteksi Risiko 4 Lapis

```mermaid
flowchart TD
    subgraph L1["LAPIS 1: BROKER STOP LOSS (Otomatis oleh MT5)"]
        L1D["SL = Entry +/- (1.5 x ATR), minimum 10 pips<br/>Dikirim bersama order ke broker<br/>Aktif 24/7, bahkan jika bot mati<br/>Max loss: ~$50-80 per trade"]
    end
    subgraph L2["LAPIS 2: SOFTWARE SMART EXIT (Bot evaluasi setiap ~10 detik)"]
        L2D["12 kondisi exit (lihat bagian 6)<br/>Biasanya menutup SEBELUM broker SL kena<br/>Target close: loss le $25 (lebih ketat dari broker)<br/>Termasuk: momentum, ML reversal, stall, time limit"]
    end
    subgraph L3["LAPIS 3: EMERGENCY STOP LOSS (Backup jika software gagal)"]
        L3D["Max loss per trade: 2% modal ($100 untuk $5K)<br/>Diset sebagai broker SL terpisah<br/>Aktif jika software error/hang"]
    end
    subgraph L4["LAPIS 4: CIRCUIT BREAKER (Hentikan semua trading)"]
        L4D["Trigger 1: Daily loss ge 3% ($150) = Stop hari ini<br/>Trigger 2: Total loss ge 10% ($500) = Stop total<br/>Trigger 3: Flash crash > 2.5% / 1 menit = CLOSE ALL<br/>Reset: Otomatis hari baru (daily), manual (total)"]
    end

    L1 --> L2 --> L3 --> L4

    style L1 fill:#4CAF50,color:#fff
    style L2 fill:#2196F3,color:#fff
    style L3 fill:#FF9800,color:#fff
    style L4 fill:#F44336,color:#fff
```

### 4 Mode Trading (Smart Risk Manager)

```mermaid
flowchart TD
    NORMAL["MODE: NORMAL<br/>Kondisi: Semua aman, tidak ada masalah<br/>Lot: 0.01 - 0.02 (berdasarkan confidence)<br/>Max posisi: 2-3"]
    NORMAL -->|"3x loss berturut-turut"| RECOVERY["MODE: RECOVERY<br/>Kondisi: Setelah kerugian beruntun<br/>Lot: 0.01 (minimum saja)<br/>Max posisi: 1"]
    RECOVERY -->|"mendekati 80% daily limit"| PROTECTED["MODE: PROTECTED<br/>Kondisi: Hampir kena daily limit<br/>Lot: 0.01 (minimum saja)<br/>Max posisi: 1"]
    PROTECTED -->|"daily/total limit tercapai"| STOPPED["MODE: STOPPED<br/>Kondisi: Batas kerugian tercapai<br/>Lot: 0 (TIDAK BOLEH trading)<br/>Max posisi: 0 (tutup semua)<br/>Reset: Otomatis hari baru"]

    style NORMAL fill:#4CAF50,color:#fff
    style RECOVERY fill:#FF9800,color:#fff
    style PROTECTED fill:#FF5722,color:#fff
    style STOPPED fill:#F44336,color:#fff
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

```mermaid
flowchart TD
    subgraph HMM["HIDDEN MARKOV MODEL"]
        direction TB
        INFO["Library: hmmlearn.GaussianHMM<br/>Input: log_returns + rolling_volatility (2 fitur)<br/>States: 3 (Low, Medium, High Volatility)<br/>Lookback: 500 bar untuk training<br/>Retrain: setiap 20 bar (auto-update)"]
        TRANS["Transition Matrix (contoh):<br/>Fr Low: To Low 0.85, To Med 0.12, To High 0.03<br/>Fr Med: To Low 0.10, To Med 0.80, To High 0.10<br/>Fr High: To Low 0.05, To Med 0.15, To High 0.80"]
        EMISI["Distribusi Emisi (per state):<br/>Low: mu_return ~ 0, sigma = kecil<br/>Med: mu_return ~ 0, sigma = sedang<br/>High: mu_return ~ 0, sigma = besar"]
        OUTPUT["Output:<br/>regime: 0/1/2 (low/medium/high)<br/>confidence: 0.0 - 1.0<br/>lot_multiplier: 1.0 / 0.5 / 0.0<br/>recommendation: TRADE / REDUCE / SLEEP"]
        INFO --> TRANS --> EMISI --> OUTPUT
    end
```

### XGBoost — Otak Prediksi

```mermaid
flowchart TD
    subgraph XGB["XGBOOST BINARY CLASSIFIER"]
        direction TB
        INFO2["Library: xgboost<br/>Objective: binary:logistic<br/>Target: UP (1) / DOWN (0) pada bar berikutnya"]
        CONFIG2["Anti-Overfitting Config:<br/>max_depth: 3 (shallow trees)<br/>learning_rate: 0.05 (slow learning)<br/>min_child_weight: 10 (min samples/leaf)<br/>subsample: 0.7 (70% data/tree)<br/>colsample_bytree: 0.6 (60% features/tree)<br/>reg_alpha: 1.0 (L1), reg_lambda: 5.0 (L2)<br/>gamma: 1.0 (min loss reduction)<br/>num_boost_round: 50 (few rounds)"]
        FITUR2["24 Fitur Input (Top 10):<br/>1. RSI(14), 2. MACD_histogram<br/>3. ATR(14), 4. bb_width<br/>5. returns_1, 6. price_position<br/>7. volatility_20, 8. returns_5<br/>9. ema_cross, 10. regime"]
        OUT2["Output:<br/>signal: BUY / SELL / HOLD<br/>probability: 0.0-1.0 (prob of UP)<br/>confidence: 0.0-1.0 (prob of winning side)"]
        VAL2["Validation:<br/>Train/Test: 70%/30% (50-bar gap, anti leakage)<br/>Walk-forward: 500 train / 50 test / 50 step<br/>Target AUC: > 0.65<br/>Rollback AUC: lt 0.60 (v4: dinaikkan dari 0.52)<br/>Overfitting ratio: train_AUC/test_AUC lt 1.15"]
        INFO2 --> CONFIG2 --> FITUR2 --> OUT2 --> VAL2
    end
```

### Kombinasi Sinyal (SMC + ML)

```mermaid
flowchart TD
    SMC_SIG["SMC Signal: BUY at 2645, SL 2635, TP 2665, conf 75%"]
    ML_SIG["ML Signal: BUY, confidence 72%"]

    SMC_SIG & ML_SIG --> COMBINE{"KOMBINASI SMC + ML"}

    COMBINE --> C1["CASE 1: SMC BUY + ML BUY (ge 50%)<br/>Combined confidence = (75%+72%)/2 = 73.5%<br/>= ENTRY (jika pass 14 filter)"]
    COMBINE --> C2["CASE 2: SMC BUY + ML SELL (ge 65%)<br/>ML strongly disagrees = BLOCK (filter #5)<br/>= TIDAK entry"]
    COMBINE --> C3["CASE 3: SMC BUY + ML uncertain (lt 50%)<br/>ML tidak yakin = BLOCK (filter #4)<br/>= TIDAK entry"]
    COMBINE --> C4["CASE 4: Tidak ada SMC signal<br/>Tidak ada entry point = SKIP<br/>SMC adalah sinyal PRIMER (wajib ada)"]

    C1 & C2 & C3 & C4 --> PRINSIP["PRINSIP:<br/>SMC = sinyal UTAMA (entry/SL/TP)<br/>ML = KONFIRMASI (bisa blokir, tidak bisa inisiasi)<br/>HMM = PENYESUAI (mengatur agresivitas)"]

    style C1 fill:#4CAF50,color:#fff
    style C2 fill:#F44336,color:#fff
    style C3 fill:#F44336,color:#fff
    style C4 fill:#9E9E9E,color:#fff
```

---

## 9. Smart Money Concepts (SMC)

### 6 Konsep yang Dianalisis

```mermaid
flowchart TD
    subgraph SP["1. SWING POINTS (Fractal High/Low)"]
        SP_D["Swing High: titik tertinggi dalam window 11 bar<br/>5 bar kiri lebih rendah, 5 bar kanan lebih rendah<br/><br/>Swing Low: titik terendah dalam window 11 bar<br/>5 bar kiri lebih tinggi, 5 bar kanan lebih tinggi"]
    end
    subgraph FVG2["2. FAIR VALUE GAP (FVG) - Ketidakseimbangan Harga"]
        FVG_D["Bullish FVG (gap up): Bar i-2 high lt Bar i low (ada gap)<br/>Bearish FVG (gap down): Bar i-2 low > Bar i high<br/>Harga cenderung kembali mengisi FVG = entry zone"]
    end
    subgraph OB2["3. ORDER BLOCK (OB) - Zona Institusi"]
        OB_D["Bullish OB: candle bearish terakhir sebelum rally<br/>(zona dimana institusi menempatkan buy order besar)<br/>Lookback: 10 bar untuk deteksi<br/>Mitigated: true jika harga sudah revisit"]
    end
    subgraph BOS2["4. BREAK OF STRUCTURE (BOS) - Kelanjutan Trend"]
        BOS_D["Uptrend BOS: harga break di ATAS swing high sebelumnya<br/>= Trend bullish berlanjut<br/>Downtrend BOS: harga break di BAWAH swing low<br/>= Trend bearish berlanjut"]
    end
    subgraph CHOCH2["5. CHANGE OF CHARACTER (CHoCH) - Perubahan Trend"]
        CHOCH_D["Uptrend ke Downtrend:<br/>Harga break di BAWAH swing low terakhir<br/>= Trend berubah dari bullish ke bearish<br/>(sinyal reversal)"]
    end
    subgraph LIQ2["6. LIQUIDITY ZONES - Target Likuiditas"]
        LIQ_D["BSL (Buy Side Liquidity): di atas swing highs<br/>(stop loss para seller berkumpul)<br/>SSL (Sell Side Liquidity): di bawah swing lows<br/>(stop loss para buyer berkumpul)<br/>Institusi sering hunt liquidity zone ini"]
    end
```

### Signal Generation

```mermaid
flowchart TD
    SB["Structure Break<br/>(BOS atau CHoCH)"] --> VALID["VALID SIGNAL"]
    ZN["Zone<br/>(FVG atau Order Block)"] --> VALID

    VALID --> BUY_SIG["BUY Signal:<br/>BOS bullish ATAU CHoCH bearish-to-bullish<br/>+ Bullish FVG ATAU Bullish OB di bawah harga<br/>Entry: harga saat ini<br/>SL: di bawah zone, minimum 1.5 x ATR<br/>TP: 2:1 R:R, maximum 4 x ATR<br/>Confidence: 40-85% (v5: calibrated weighted scoring)"]
    VALID --> SELL_SIG["SELL Signal:<br/>BOS bearish ATAU CHoCH bullish-to-bearish<br/>+ Bearish FVG ATAU Bearish OB di atas harga<br/>Entry: harga saat ini<br/>SL: di atas zone, minimum 1.5 x ATR<br/>TP: 2:1 R:R, maximum 4 x ATR<br/>Confidence: 40-85% (v5: calibrated weighted scoring)"]

    style BUY_SIG fill:#4CAF50,color:#fff
    style SELL_SIG fill:#F44336,color:#fff
```

---

## 10. Position Lifecycle

### Dari Lahir Sampai Mati (Siklus Hidup Posisi)

```mermaid
flowchart TD
    T1["TAHAP 1: SINYAL TERDETEKSI<br/>SMC menemukan setup + ML konfirmasi + 14 filter PASS<br/>Keputusan: BUKA POSISI"]
    T1 --> T2["TAHAP 2: LOT SIZE CALCULATION<br/>Risk Engine (Kelly): Balance $5000 x 1% = $50 max loss<br/>SL distance 50 pips = lot 0.01<br/>ML ge 80% = 0.02 lot, ML lt 65% = 0.01 lot<br/>Session: Golden x1.2, Sydney x0.5<br/>Regime: Normal x1.0, High Vol x0.5, Crisis x0.0"]
    T2 --> T3["TAHAP 3: ORDER VALIDATION<br/>SL di sisi benar (BUY: SL lt entry)<br/>TP di sisi benar (BUY: TP > entry)<br/>Lot dalam range (0.01 - 0.05)<br/>Entry dekat harga saat ini (lt 0.1%)<br/>Risk% le 1.5x config limit<br/>Circuit breaker TIDAK aktif"]
    T3 --> T4["TAHAP 4: ORDER EXECUTION<br/>MT5 Connector mengirim order:<br/>Symbol: XAUUSD, Type: BUY/SELL<br/>Lot: 0.01-0.02, SL: ATR-based<br/>TP: 2:1 R:R, Deviation: 20 points<br/>Retry: max 3 attempts"]
    T4 --> T4B["TAHAP 4b: POST-EXECUTION VALIDATION (v5)<br/>Slippage Validation: max 0.15% (~$4 XAUUSD)<br/>Log WARNING jika melebihi batas<br/>Gunakan harga AKTUAL untuk tracking<br/>Partial Fill: cek volume, update lot aktual"]
    T4B --> T5["TAHAP 5: POSITION REGISTERED<br/>Smart Risk Manager: catat entry AKTUAL, peak=0<br/>Trade Logger: insert PostgreSQL (30+ field) + CSV<br/>Telegram: notifikasi trade open<br/>(entry, SL, TP, R:R, confidence, regime)"]
    T5 --> T6["TAHAP 6: ACTIVE MONITORING<br/>(setiap ~10 detik + candle baru)<br/>Update profit/loss, peak profit, momentum<br/>Hitung TP probability<br/>Cek 12 kondisi exit<br/>Cek Position Manager (trailing, breakeven)<br/>Cek Market Close Handler"]
    T6 -->|"trigger close"| T7["TAHAP 7: POSITION CLOSED<br/>MT5: Close via market order<br/>Risk Manager: record P/L, update counters,<br/>check mode transition<br/>Logger: update exit price, profit, duration, reason<br/>Telegram: notifikasi trade close"]
    T6 -->|"HOLD"| T6

    style T1 fill:#1565C0,color:#fff
    style T7 fill:#C62828,color:#fff
```

---

## 11. Auto-Retraining & Model Management

### Lifecycle Model AI

```mermaid
flowchart TD
    INIT["INITIAL TRAINING (train_models.py)<br/>Dijalankan 1x saat setup"]
    INIT --> I1["1. Fetch 10,000 bar M15 dari MT5 (~104 hari)"]
    I1 --> I2["2. Feature Engineering = 40+ fitur"]
    I2 --> I3["3. SMC Analysis = struktur pasar"]
    I3 --> I4["4. Create target UP/DOWN (lookahead=1)<br/>4b. Split 70/30, 50-bar gap (anti leakage)"]
    I4 --> I5["5. Train HMM (3 regime, lookback=500)"]
    I5 --> I6["6. Train XGBoost (50 rounds, early_stop=5)"]
    I6 --> I7["7. Walk-forward validation (500/50/50)"]
    I7 --> I8["8. Save: hmm_regime.pkl + xgboost_model.pkl"]

    I8 --> DAILY["DAILY AUTO-RETRAINING (Auto Trainer)<br/>Otomatis setiap hari 05:00 WIB"]

    DAILY --> SCHED["Schedule:<br/>Harian (05:00 WIB): 8,000 bar, 50 rounds<br/>Weekend (Sabtu): 15,000 bar, 80 rounds (deep)<br/>Emergency: jika AUC lt 0.65"]

    SCHED --> D1["1. Backup model saat ini"]
    D1 --> D2["2. Fetch data baru dari MT5"]
    D2 --> D3["3. Feature Engineering + SMC"]
    D3 --> D4["4. Train HMM baru + XGBoost baru"]
    D4 --> D5{"5. Validasi: test AUC ge 0.60?"}
    D5 -->|"Ya"| SAVE["Save model baru, reload di memory"]
    D5 -->|"Tidak"| ROLLBACK["ROLLBACK ke model sebelumnya"]
    SAVE --> D6["6. Log hasil ke PostgreSQL"]
    ROLLBACK --> D6
    D6 --> D7["7. Kirim laporan via Telegram"]

    D7 --> SAFETY["Safety:<br/>Max 5 backup (rotasi)<br/>Min 20 jam antar retrain (cooldown)<br/>Auto-rollback jika AUC lt 0.60<br/>Model lama selalu tersedia"]
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

```mermaid
flowchart TD
    DB["trading_db"] --> T["trades<br/>Semua trade: open, close, profit, SMC, ML, features"]
    DB --> TR["training_runs<br/>Log setiap training: AUC, akurasi, durasi, rollback"]
    DB --> SG["signals<br/>Setiap sinyal yang dihasilkan: executed atau tidak"]
    DB --> MS["market_snapshots<br/>Snapshot periodik: harga, regime, volatilitas"]
    DB --> BS["bot_status<br/>Status bot: uptime, loop count, balance, risk mode"]
    DB --> DS["daily_summaries<br/>Ringkasan harian: win rate, profit factor, per sesi"]
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

```mermaid
flowchart TD
    TL["TradeLogger"] -->|"TradeRepository, SignalRepository,<br/>MarketSnapshotRepository"| DBC["DatabaseConnection (Singleton)"]
    AT2["AutoTrainer"] -->|"TrainingRepository"| DBC
    ML["main_live.py"] -->|"BotStatusRepository,<br/>DailySummaryRepository"| DBC
    DASH["Dashboard"] -->|"Semua repository (READ)"| DBC

    DBC --> POOL["ThreadedConnectionPool (1-10 koneksi)"]
    POOL --> PG["PostgreSQL Server"]
```

### Graceful Degradation

```mermaid
flowchart TD
    CHECK{"PostgreSQL tersedia?"}
    CHECK -->|"Ya"| DUAL["Gunakan DB + CSV backup (dual write)"]
    CHECK -->|"Tidak"| CSV["CSV saja (bot tetap berjalan 100%)"]
    DUAL --> NOTE["Bot TIDAK PERNAH crash karena database.<br/>Semua operasi DB dibungkus try-except."]
    CSV --> NOTE
```

---

## 13. Konfigurasi & Parameter Kritis

### Configuration System

```mermaid
flowchart TD
    ENV[".env file"] --> TC["TradingConfig.from_env()"]

    TC --> RC["RiskConfig<br/>risk_per_trade: 1.0% (SMALL) / 0.5% (MEDIUM)<br/>max_daily_loss: 3.0% (SMALL) / 2.0% (MEDIUM)<br/>max_total_loss: 10.0%<br/>max_positions: 3 (SMALL) / 5 (MEDIUM)<br/>min_lot: 0.01<br/>max_lot: 0.05 (SMALL) / 2.0 (MEDIUM)<br/>max_leverage: 1:100 (SMALL) / 1:30 (MEDIUM)"]

    TC --> SC["SMCConfig<br/>swing_length: 5<br/>fvg_min_gap_pips: 2.0<br/>ob_lookback: 10<br/>bos_close_break: true"]

    TC --> MLC2["MLConfig<br/>confidence_threshold: 0.65<br/>entry_confidence: 0.70<br/>high_confidence: 0.75<br/>very_high_confidence: 0.80<br/>retrain_frequency_days: 7"]

    TC --> THC["ThresholdsConfig<br/>ml_min_confidence: 0.65<br/>ml_high_confidence: 0.75<br/>trade_cooldown_seconds: 300<br/>min_profit_to_secure: $15<br/>good_profit: $25, great_profit: $40<br/>flash_crash_threshold: 2.5%<br/>sydney_lot_multiplier: 0.5"]

    TC --> RGC["RegimeConfig<br/>n_regimes: 3<br/>lookback: 500<br/>retrain_frequency: 20"]
```

### Capital Mode Auto-Detection

```mermaid
flowchart TD
    BAL{"Balance?"}
    BAL -->|"le $10,000"| SMALL["SMALL MODE<br/>Risk: 1% per trade ($50 pada $5K)<br/>Daily limit: 3% ($150)<br/>Lot: 0.01-0.05<br/>Leverage: 1:100<br/>Timeframe: M15<br/>Max posisi: 3"]
    BAL -->|"> $10,000"| MEDIUM["MEDIUM MODE<br/>Risk: 0.5% per trade<br/>Daily limit: 2%<br/>Lot: 0.01-2.0<br/>Leverage: 1:30<br/>Timeframe: H1<br/>Max posisi: 5"]
```

### Session Schedule (WIB = GMT+7)

```mermaid
flowchart LR
    subgraph SESSION["Session Schedule (WIB = GMT+7)"]
        direction TB
        DZ["00:00-04:00 DEAD ZONE<br/>BLOCKED - Likuiditas rendah"]
        RO["04:00-06:00 ROLLOVER<br/>BLOCKED - Spread melebar"]
        SY["06:00-07:00 Sydney<br/>0.5x - Pasar baru buka"]
        TK["07:00-13:00 Tokyo+Sydney<br/>0.7x - Asia aktif"]
        TA["13:00-15:00 Tokyo akhir<br/>0.7x - Transisi"]
        LD["15:00-20:00 London<br/>1.0x - Volatilitas naik"]
        GT["20:00-23:59 GOLDEN TIME<br/>1.2x - London+NY overlap"]
        WR["Jumat ge 23:00 WEEKEND RISK<br/>BLOCKED - Gap risk"]
    end

    style DZ fill:#F44336,color:#fff
    style RO fill:#F44336,color:#fff
    style SY fill:#FF9800,color:#fff
    style TK fill:#FFC107,color:#000
    style TA fill:#FFC107,color:#000
    style LD fill:#2196F3,color:#fff
    style GT fill:#4CAF50,color:#fff
    style WR fill:#F44336,color:#fff
```

Golden Time (20:00-23:59 WIB) = waktu paling optimal
- Spread ketat, likuiditas maksimal, volatilitas ideal
- Lot multiplier 1.2x (bonus)
- v4: Smart Hold dihapus -- tidak ada lagi hold losers menunggu sesi tertentu

---

## 14. Performa & Timing

### Main Loop Breakdown

Target: < 50ms per iterasi analisis

**FULL ANALYSIS (saat candle baru M15):**

| Komponen | Waktu | Keterangan |
|----------|-------|------------|
| MT5 data fetch | ~10ms | 200 bar M15 via API |
| Feature engineering | ~5ms | 40+ fitur, Polars |
| SMC analysis | ~5ms | 6 konsep, Polars native |
| HMM predict | ~2ms | 2 fitur -> 1 regime |
| XGBoost predict | ~3ms | 24 fitur -> 1 signal |
| Position monitoring | ~5ms | Per posisi terbuka |
| Entry logic | ~5ms | 14 filter check |
| Overhead | ~15ms | Logging, state update |
| **TOTAL** | **~50ms** | |

**POSITION CHECK ONLY (di antara candle, setiap ~10 detik):**

| Komponen | Waktu | Keterangan |
|----------|-------|------------|
| MT5 data fetch | ~5ms | 50 bar saja |
| Feature engineering | ~3ms | Minimal fitur |
| ML prediction | ~3ms | Untuk exit evaluation |
| Position evaluation | ~5ms | 12 kondisi exit |
| Overhead | ~5ms | Logging |
| **TOTAL** | **~21ms** | |

### Timer Periodik

| Event | Interval | Cara Trigger |
|-------|----------|--------------|
| Full analysis + entry | Candle baru M15 | Deteksi candle |
| Position monitoring | ~10 detik | Di antara candle |
| Performance logging | 4 candle (~1j) | candle_count % 4 |
| Auto-retrain check | 20 candle (~5j) | candle_count % 20 |
| Market update Telegram | 30 menit | Timer |
| Hourly analysis | 1 jam | Timer |
| Daily summary + reset | Ganti hari | Date check |

---

## 15. Error Handling & Fault Tolerance

### Prinsip: Bot TIDAK PERNAH Crash

```mermaid
flowchart TD
    subgraph LV1["LEVEL 1: Per-Loop Error Handling"]
        LV1D["try: Fetch data, analyze, trade<br/>except ConnectionError: reconnect()<br/>except Exception: Log error, lanjut loop berikutnya<br/>Bot TIDAK crash dari error tunggal"]
    end
    subgraph LV2["LEVEL 2: MT5 Auto-Reconnect"]
        LV2D["MT5 putus?<br/>Attempt 1: reconnect (tunggu 2 detik)<br/>Attempt 2: reconnect (tunggu 4 detik)<br/>Attempt 3: reconnect (tunggu 8 detik)<br/>Cooldown 60 detik, Retry cycle (max 5/cooldown)<br/>Selama disconnected: monitoring PAUSE,<br/>entry DITUNDA, posisi dilindungi broker SL"]
    end
    subgraph LV3["LEVEL 3: Database Graceful Degradation"]
        LV3D["PostgreSQL down?<br/>Switch ke CSV-only mode<br/>Semua data tetap dicatat<br/>Trading tetap berjalan normal<br/>Retry DB connection periodik"]
    end
    subgraph LV4["LEVEL 4: Telegram Failure"]
        LV4D["Telegram API error?<br/>Log error secara silent<br/>Trading tetap jalan 100%<br/>Retry di notifikasi berikutnya"]
    end
    subgraph LV5["LEVEL 5: Model File Missing"]
        LV5D[".pkl file tidak ditemukan?<br/>Log warning<br/>Skip prediksi (ML/HMM)<br/>Trading bisa jalan tanpa ML (SMC only)<br/>Trigger: jalankan train_models.py"]
    end
    subgraph LV6["LEVEL 6: Flash Crash Protection"]
        LV6D["Harga bergerak > 2.5% dalam 1 menit?<br/>EMERGENCY: Close ALL positions<br/>Circuit breaker AKTIF<br/>Kirim alert KRITIS via Telegram<br/>Bot masuk mode STOPPED"]
    end

    LV1 --> LV2 --> LV3 --> LV4 --> LV5 --> LV6
```

### Startup & Shutdown

```mermaid
flowchart TD
    subgraph STARTUP["STARTUP SEQUENCE"]
        direction TB
        S1["1. Load konfigurasi dari .env"] --> S2["2. Connect ke MT5 (max 3 retry)"]
        S2 --> S3["3. Load model HMM dari models/hmm_regime.pkl"]
        S3 --> S4["4. Load model XGBoost dari models/xgboost_model.pkl"]
        S4 --> S5["5. Initialize SmartRiskManager (set balance, limits)"]
        S5 --> S6["6. Initialize SessionFilter (WIB timezone)"]
        S6 --> S7["7. Initialize TelegramNotifier"]
        S7 --> S8["8. Initialize TradeLogger (connect DB)"]
        S8 --> S9["9. Initialize AutoTrainer"]
        S9 --> S10["10. Send Telegram: BOT STARTED"]
        S10 --> S11["11. Mulai main loop"]
    end

    subgraph SHUTDOWN["SHUTDOWN SEQUENCE (SIGINT/SIGTERM)"]
        direction TB
        D1["1. Signal diterima"] --> D2["2. Hentikan loop utama"]
        D2 --> D3["3. Kirim Telegram: BOT STOPPED"]
        D3 --> D4["4. Disconnect MT5"]
        D4 --> D5["5. Close database connections"]
        D5 --> D6["6. Exit"]
    end
```

---

## 16. Daftar File Source Code

```mermaid
flowchart TD
    ROOT["Smart Automatic Trading BOT + AI/"]
    ROOT --> MAIN["main_live.py - Orchestrator utama"]
    ROOT --> TRAIN["train_models.py - Script training awal"]
    ROOT --> ENVF[".env - Environment variables"]

    ROOT --> SRC["src/"]
    SRC --> CFG["config.py - Konfigurasi terpusat (6 sub-config)"]
    SRC --> MT5F["mt5_connector.py - Bridge ke MetaTrader 5"]
    SRC --> FEF["feature_eng.py - Feature Engineering (40+ fitur)"]
    SRC --> RDF["regime_detector.py - HMM Regime Detection"]
    SRC --> MLF["ml_model.py - XGBoost Signal Predictor"]
    SRC --> SMCF["smc_polars.py - Smart Money Concepts (6 konsep)"]
    SRC --> SRMF["smart_risk_manager.py - 4-Mode Risk Manager"]
    SRC --> REF["risk_engine.py - Kelly Criterion + Circuit Breaker"]
    SRC --> SFF["session_filter.py - Session Time Filter (WIB)"]
    SRC --> DCF["dynamic_confidence.py - Dynamic Threshold Manager"]
    SRC --> NAF["news_agent.py - News Event Monitor"]
    SRC --> TNF["telegram_notifier.py - Telegram Push Notifications"]
    SRC --> ATF["auto_trainer.py - Daily Auto-Retraining"]
    SRC --> TLF["trade_logger.py - Dual Storage Logger (DB+CSV)"]
    SRC --> PMF["position_manager.py - Position Manager + Market Close"]
    SRC --> DBD["db/"]
    DBD --> DBINIT["__init__.py - DB exports"]
    DBD --> DBCONN["connection.py - PostgreSQL Singleton + Pool"]
    DBD --> DBREPO["repository.py - 6 Repository classes"]

    ROOT --> MODELS["models/"]
    MODELS --> XGBM["xgboost_model.pkl - Trained XGBoost model"]
    MODELS --> HMMM["hmm_regime.pkl - Trained HMM model"]
    MODELS --> BKUP["backup/ - Auto-backup (5 terakhir)"]

    ROOT --> DATA["data/"]
    DATA --> TDATA["training_data.parquet"]
    DATA --> TLOGS["trade_logs/ - CSV backup (per bulan)"]

    ROOT --> BTESTS["backtests/"]
    BTESTS --> BTSYNC["backtest_live_sync.py - 100% sync live"]

    ROOT --> LOGS["logs/"]
    LOGS --> LOGF["training_YYYY-MM-DD.log"]

    ROOT --> DOCS["docs/arsitektur-ai/"]
    DOCS --> DOC0["00-ARSITEKTUR-LENGKAP.md - Dokumen ini"]
    DOCS --> DOCR["README.md - Index komponen"]
    DOCS --> DOC1["01-23 (per komponen) - Detail per modul"]
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

Semua ini dikoordinasikan oleh **Main Live Orchestrator** yang menjalankan loop **candle-based** — analisis penuh hanya saat candle M15 baru terbentuk (~50ms per iterasi), dengan pengecekan posisi setiap ~10 detik di antara candle (~21ms). Mengevaluasi 14 *filter entry* dan 12 kondisi *exit* secara *real-time*, dengan notifikasi Telegram untuk setiap kejadian penting.

```
TARGET: Trading XAUUSD M15 yang KONSISTEN dan AMAN
        dengan kerugian terkontrol dan profit teroptimasi.
```
