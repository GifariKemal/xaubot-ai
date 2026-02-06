# Main Live — Orchestrator Utama

> **File:** `main_live.py`
> **Class:** `TradingBot`
> **Runtime:** Async event loop (asyncio)
> **Mode:** Candle-based (analisis penuh hanya saat candle baru M15)
> **Target:** < 0.05 detik per iterasi analisis

---

## Apa Itu Main Live?

Main Live adalah **otak pusat** yang mengorkestrasi semua komponen bot. Menjalankan loop **candle-based** — analisis penuh hanya dijalankan saat candle M15 baru terbentuk, dengan pengecekan posisi setiap 10 detik di antara candle. Mengkoordinasikan 15+ komponen dari data fetching hingga order execution.

**Analogi:** Main Live seperti **konduktor orkestra** — tidak memainkan alat musik sendiri, tapi mengarahkan semua pemain (komponen) agar bermain harmonis pada waktu yang tepat.

---

## Komponen yang Dimuat

```python
class TradingBot:
    def __init__(self):
        # Koneksi
        self.mt5 = MT5Connector(...)              # Jembatan ke broker
        self.telegram = TelegramNotifier(...)       # Notifikasi

        # AI Models
        self.ml_model = TradingModel(...)           # XGBoost predictor
        self.regime_detector = MarketRegimeDetector(...) # HMM regime
        self.smc = SMCAnalyzer(...)                 # Smart Money Concepts

        # Analisis
        self.features = FeatureEngineer()           # 40+ fitur
        self.dynamic_confidence = DynamicConfidenceManager(...) # Threshold
        self.session_filter = SessionFilter(...)     # Waktu trading
        self.news_agent = NewsAgent(...)            # Monitor berita

        # Risiko
        self.smart_risk = SmartRiskManager(...)     # Risk management
        self.risk_engine = RiskEngine(...)          # Kelly criterion
        self.position_manager = SmartPositionManager(...) # Position mgmt

        # Logging & Training
        self.trade_logger = TradeLogger(...)        # Pencatat trade
        self.auto_trainer = AutoTrainer(...)        # Retraining otomatis

        # State
        self.flash_crash_detector = FlashCrashDetector(...) # Proteksi
```

---

## Main Loop (Candle-Based)

```
STARTUP:
  Load models → Connect MT5 → Send Telegram startup
    |
    v
LOOP UTAMA (cek setiap ~5 detik, analisis pada candle baru):
    |
    ├── Fetch 2 bar terakhir → cek apakah candle baru terbentuk
    |
    |===[CANDLE BARU? YA → FULL ANALYSIS]=================
    |
    |  |===[PHASE 1: DATA]================================
    |  |
    |  ├── Fetch 200 bar M15 XAUUSD dari MT5
    |  ├── Feature Engineering (40+ fitur)
    |  ├── SMC Analysis (Swing, FVG, OB, BOS, CHoCH)
    |  ├── HMM Regime Detection
    |  └── XGBoost Prediction
    |  |
    |  |===[PHASE 2: MONITORING]===========================
    |  |
    |  ├── Cek posisi terbuka
    |  │   └── Untuk setiap posisi:
    |  │       ├── Update profit & momentum
    |  │       ├── 10 kondisi exit (smart_risk.evaluate_position)
    |  │       └── Jika should_close → tutup → log → Telegram
    |  |
    |  ├── Position Manager (trailing SL, breakeven)
    |  │   └── Smart Market Close Handler
    |  |
    |  |===[PHASE 3: ENTRY]================================
    |  |
    |  ├── [1] Session Filter → boleh trading?
    |  ├── [2] Risk Mode → bukan STOPPED?
    |  ├── [3] SMC Signal → ada setup?
    |  ├── [4] ML Confidence → >= threshold?
    |  ├── [5] ML Agreement → tidak strongly disagree?
    |  ├── [6] Dynamic Quality → bukan AVOID?
    |  ├── [7] Confirmation → 2x candle berturut?
    |  ├── [8] Pullback Filter → momentum selaras? (v5: ATR-based)
    |  ├── [9] Cooldown → 5 menit sejak trade terakhir?
    |  ├── [10] Position Limit → < 2 posisi?
    |  ├── [11] Lot Size → > 0?
    |  ├── SEMUA PASS → Execute trade
    |  ├── Slippage validation (v5: cek harga aktual vs expected)
    |  ├── Partial fill check (v5: cek volume aktual vs requested)
    |  └── Register position (harga & volume AKTUAL) → Log → Telegram
    |  |
    |  |===[PHASE 4: PERIODIK]=============================
    |  |
    |  ├── Setiap 20 candle (~5 jam M15): Cek auto-retrain
    |  ├── Setiap 30 menit: Market update (Telegram)
    |  ├── Setiap 1 jam: Hourly analysis (Telegram)
    |  ├── Pergantian hari: Daily summary + reset
    |  └── News Agent: Monitor (non-blocking)
    |
    |===[CANDLE BARU? TIDAK → POSITION CHECK ONLY]========
    |
    ├── Setiap 10 detik: cek posisi terbuka saja
    │   ├── Fetch 50 bar (minimal data)
    │   ├── Hitung fitur untuk ML check
    │   └── Evaluasi exit conditions per posisi
    |
    v
    Tunggu ~5 detik → Loop lagi
```

---

## Startup Sequence

```
1. Load konfigurasi dari .env
2. Connect ke MT5 (max 3 retry)
3. Load model HMM dari models/hmm_regime.pkl
4. Load model XGBoost dari models/xgboost_model.pkl
5. Initialize SmartRiskManager (set balance, limits)
6. Initialize SessionFilter (WIB timezone)
7. Initialize TelegramNotifier
8. Initialize TradeLogger
9. Initialize AutoTrainer
10. Send Telegram: "BOT STARTED" (config, balance, risk settings)
11. Mulai main loop
```

---

## Shutdown Sequence

```
1. Signal SIGINT/SIGTERM diterima
2. Hentikan loop utama
3. Kirim Telegram: "BOT STOPPED" (balance, trades, uptime)
4. Disconnect MT5
5. Close database connections
6. Exit
```

---

## Error Handling

```
Setiap iterasi loop dibungkus try-except:

try:
    # Fetch data, analyze, trade
except ConnectionError:
    # MT5 disconnected → reconnect()
except Exception as e:
    # Log error → lanjut loop berikutnya
    # Bot TIDAK crash dari error tunggal

Prinsip: NEVER STOP TRADING karena error non-kritis
```

---

## Timer Periodik

| Event | Interval | Aksi |
|-------|----------|------|
| Full analysis + entry | Setiap candle baru M15 | Saat candle terbentuk |
| Position monitoring | ~10 detik | Di antara candle |
| Performance logging | 4 candle (~1 jam) | `loop_count % 4` |
| Auto-retrain check | 20 candle (~5 jam) | `loop_count % 20` |
| Market update Telegram | 30 menit | Timer |
| Hourly analysis Telegram | 1 jam | Timer |
| Daily summary | Pergantian hari | Date check |

---

## Performa Target

```
Target: < 0.05 detik per iterasi analisis (50ms)

Full Analysis (saat candle baru):
├── MT5 data fetch:      ~10ms  (200 bar)
├── Feature engineering:  ~5ms  (Polars, vectorized)
├── SMC analysis:         ~5ms  (Polars native)
├── HMM predict:          ~2ms
├── XGBoost predict:      ~3ms
├── Position monitoring:  ~5ms
├── Entry logic:          ~5ms
└── Overhead:             ~15ms
                          ------
                          ~50ms total

Position Check Only (di antara candle):
├── MT5 data fetch:      ~5ms   (50 bar saja)
├── Feature engineering:  ~3ms
├── ML prediction:        ~3ms
├── Position evaluation:  ~5ms
└── Overhead:             ~5ms
                          ------
                          ~21ms total
```

---

## Hubungan Semua Komponen

```
┌─────────────────────────────────────────────────────────┐
│                     main_live.py                         │
│                    (TradingBot)                           │
│                                                          │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
│  │   MT5   │  │ Feature │  │   SMC    │  │   HMM    │  │
│  │Connector│→ │  Eng    │→ │ Analyzer │→ │ Detector │  │
│  └─────────┘  └─────────┘  └──────────┘  └──────────┘  │
│       ↑                          ↓              ↓        │
│       │              ┌──────────────────────┐           │
│       │              │  Dynamic Confidence  │           │
│       │              └──────────────────────┘           │
│       │                        ↓                        │
│       │              ┌──────────────────┐               │
│       │              │ XGBoost Model    │               │
│       │              └──────────────────┘               │
│       │                        ↓                        │
│       │    ┌─────────────────────────────────┐          │
│       │    │  Entry Logic (11 Filters)       │          │
│       │    │  Session, Risk, SMC, ML, ...    │          │
│       │    └─────────────────────────────────┘          │
│       │                        ↓                        │
│       │    ┌────────────┐  ┌───────────────┐            │
│       ├────│ Risk Engine│  │Smart Risk Mgr │            │
│       │    └────────────┘  └───────────────┘            │
│       │                        ↓                        │
│       │←── Execute Order (BUY/SELL)                     │
│       │                        ↓                        │
│       │    ┌────────────┐  ┌───────────────┐            │
│       │    │  Position  │  │ Trade Logger  │            │
│       │    │  Manager   │  │ (DB + CSV)    │            │
│       │    └────────────┘  └───────────────┘            │
│       │                        ↓                        │
│       │    ┌────────────┐  ┌───────────────┐            │
│       │    │  Telegram  │  │ Auto Trainer  │            │
│       │    │ Notifier   │  │ (retraining)  │            │
│       │    └────────────┘  └───────────────┘            │
│       │                                                  │
│       │    ┌────────────┐  ┌───────────────┐            │
│       │    │News Agent  │  │Session Filter │            │
│       │    │(monitor)   │  │(waktu trading)│            │
│       │    └────────────┘  └───────────────┘            │
└─────────────────────────────────────────────────────────┘
```
