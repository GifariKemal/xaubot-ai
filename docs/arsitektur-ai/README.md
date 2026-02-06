# Arsitektur AI — Smart Trading Bot

> Dokumentasi lengkap semua komponen AI dan sistem pendukung.

---

## Daftar Komponen

### Inti AI & Analisis

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 1 | [HMM Regime Detector](01-HMM-Regime-Detector.md) | `src/regime_detector.py` | Deteksi kondisi pasar (radar cuaca) |
| 2 | [XGBoost Signal Predictor](02-XGBoost-Signal-Predictor.md) | `src/ml_model.py` | Prediksi arah harga (navigator AI) |
| 3 | [SMC Analyzer](03-SMC-Analyzer.md) | `src/smc_polars.py` | Analisis struktur pasar institusi (peta jalan) |
| 4 | [Feature Engineering](04-Feature-Engineering.md) | `src/feature_eng.py` | Pengolahan data mentah ke fitur ML (alat ukur) |

### Proteksi & Manajemen Risiko

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 5 | [Risk Management](05-Risk-Management.md) | `src/smart_risk_manager.py` | Perlindungan modal (sabuk pengaman) |
| 6 | [Session Filter](06-Session-Filter.md) | `src/session_filter.py` | Pengaturan waktu trading (jadwal kerja) |
| 7 | [Stop Loss (S/L)](07-Stop-Loss.md) | Multi-file | Proteksi 4 lapis dari kerugian |
| 8 | [Take Profit (T/P)](08-Take-Profit.md) | Multi-file | Pengambilan profit cerdas 6 layer |

### Proses Trading

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 9 | [Entry Trade](09-Entry-Trade.md) | `main_live.py` | Proses masuk posisi (11 filter) |
| 10 | [Exit Trade](10-Exit-Trade.md) | `main_live.py` | Proses keluar posisi (10 kondisi) |

### Koneksi & Konfigurasi

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 16 | [MT5 Connector](16-MT5-Connector.md) | `src/mt5_connector.py` | Jembatan ke broker MT5 (auto-reconnect) |
| 17 | [Configuration](17-Configuration.md) | `src/config.py` | Konfigurasi terpusat (6 sub-config) |

### Pendukung

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 11 | [News Agent](11-News-Agent.md) | `src/news_agent.py` | Monitoring berita ekonomi |
| 12 | [Telegram Notifications](12-Telegram-Notifications.md) | `src/telegram_notifier.py` | Notifikasi real-time ke Telegram |
| 18 | [Trade Logger](18-Trade-Logger.md) | `src/trade_logger.py` | Pencatatan trade dual-storage (DB + CSV) |
| 19 | [Position Manager](19-Position-Manager.md) | `src/position_manager.py` | Manajemen posisi aktif (trailing, breakeven) |
| 20 | [Risk Engine](20-Risk-Engine.md) | `src/risk_engine.py` | Mesin risiko & circuit breaker (Kelly Criterion) |
| 21 | [Database](21-Database.md) | `src/db/` | PostgreSQL integration (6 repository) |

### Training & Validasi

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 13 | [Auto Trainer](13-Auto-Trainer.md) | `src/auto_trainer.py` | Retraining model otomatis (pelatih malam) |
| 14 | [Backtest](14-Backtest.md) | `backtests/backtest_live_sync.py` | Simulasi trading 100% sync dengan live |
| 15 | [Dynamic Confidence](15-Dynamic-Confidence.md) | `src/dynamic_confidence.py` | Penyesuaian threshold otomatis (termometer) |
| 22 | [Train Models](22-Train-Models.md) | `train_models.py` | Script training awal (HMM + XGBoost) |

### Orchestrator

| # | Komponen | File Source | Fungsi |
|---|----------|------------|--------|
| 23 | [Main Live Orchestrator](23-Main-Live-Orchestrator.md) | `main_live.py` | Otak pusat bot, koordinasi semua komponen |

---

## Pipeline Lengkap

```
Raw OHLCV dari MT5
       |
       v
[Feature Engineering]  ->  40+ fitur numerik (RSI, ATR, MACD, BB, EMA, ...)
       |
       v
[SMC Analyzer]         ->  Swing, FVG, OB, BOS, CHoCH, Liquidity
       |                   + Signal (entry, SL ATR-based, TP ATR-capped)
       |
   +---+---+
   |       |
   v       v
 [HMM]  [XGBoost]
Regime   Signal
   |       |
   +---+---+
       |
       v
[Signal Combination]   ->  SMC + ML harus setuju
       |
       v
[News Agent]           ->  Monitor berita (tidak blocking)
       |
       v
[Session Filter]       ->  Cek waktu boleh trading?
       |
       v
[ENTRY TRADE]          ->  11 filter harus PASS:
  | Session, Risk Mode, SMC Signal, ML Confirm,
  | ML Agree, Quality, Confirmation 2x, Pullback,
  | Cooldown, Position Limit, Lot Size
       |
       v
[Risk Management]      ->  Hitung lot aman, apply multiplier
       |
       v
[Execute Order]        ->  Kirim ke MT5 dengan broker SL & TP
       |
       v
[Telegram]             ->  Notifikasi trade open
       |
       v
[EXIT MONITORING]      ->  Setiap 1 detik, 10 kondisi exit:
  | Smart TP, Early Exit, Golden Hold, ML Reversal,
  | Max Loss, Stall, Daily Limit, Weekend, Time-based, Hold
       |
       v
[Close Position]       ->  Record result, update risk, notify
```

---

## Ringkasan Peran Setiap Komponen

| Komponen | Pertanyaan yang Dijawab |
|----------|------------------------|
| Feature Engineering | "Data mentah ini berarti apa?" |
| SMC Analyzer | "Dimana institusi besar trading? Entry/SL/TP dimana?" |
| HMM | "Kondisi pasar bagaimana sekarang?" |
| XGBoost | "Harga akan naik atau turun?" |
| Session Filter | "Sekarang waktu yang tepat untuk trading?" |
| News Agent | "Ada berita high-impact yang perlu diperhatikan?" |
| Risk Management | "Berapa besar boleh trading? Sudah aman?" |
| Stop Loss | "Bagaimana melindungi dari kerugian?" |
| Take Profit | "Kapan mengambil profit?" |
| Entry Trade | "Apakah semua syarat terpenuhi untuk masuk?" |
| Exit Trade | "Apakah sudah waktunya keluar?" |
| Telegram | "Apa yang sedang terjadi?" |
| Auto Trainer | "Apakah model AI masih akurat? Perlu dilatih ulang?" |
| Backtest | "Apakah strategi ini profitable di data historis?" |
| Dynamic Confidence | "Seberapa selektif bot harus trading saat ini?" |
| MT5 Connector | "Bagaimana bot terhubung ke broker dan mengirim order?" |
| Configuration | "Bagaimana semua parameter dikonfigurasi?" |
| Trade Logger | "Dimana semua data trade disimpan?" |
| Position Manager | "Bagaimana posisi terbuka dikelola secara aktif?" |
| Risk Engine | "Berapa ukuran lot yang aman? Sudah lewat batas harian?" |
| Database | "Bagaimana data persisten disimpan dan di-query?" |
| Train Models | "Bagaimana model AI dilatih pertama kali?" |
| Main Live | "Siapa yang mengorkestrasi semua komponen?" |
