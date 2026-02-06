# Configuration — Pusat Pengaturan Bot

> **File:** `src/config.py`
> **Class:** `TradingConfig`, `RiskConfig`, `SMCConfig`, `MLConfig`, `ThresholdsConfig`, `RegimeConfig`
> **Sumber:** Environment variables (`.env`)

---

## Apa Itu Configuration?

Configuration adalah **pusat pengaturan** seluruh parameter bot — dari kredensial MT5 hingga threshold AI. Semua pengaturan otomatis menyesuaikan berdasarkan ukuran modal (small/medium).

**Analogi:** Configuration seperti **kokpit pesawat** — semua tombol dan dial pengaturan ada di satu tempat, dan bisa diubah sebelum "terbang" (trading).

---

## Capital Mode (Otomatis)

| Mode | Modal | Risk/Trade | Max Daily Loss | Leverage | Max Lot | Max Posisi | Timeframe |
|------|-------|-----------|----------------|----------|---------|-----------|-----------|
| **SMALL** | ≤ $10K | 1% | 3% | 1:100 | 0.05 | 3 | M15 |
| **MEDIUM** | > $10K | 0.5% | 2% | 1:30 | 2.0 | 5 | H1 |

```python
# Otomatis berdasarkan capital
if capital <= 10000:
    mode = SMALL    # Growth mode
else:
    mode = MEDIUM   # Preservation mode
```

---

## 6 Sub-Konfigurasi

### 1. RiskConfig

```python
RiskConfig(
    risk_per_trade=1.0,     # 1% per trade ($50 dari $5K)
    max_daily_loss=3.0,     # 3% max daily loss ($150)
    max_leverage=100,       # 1:100
    max_positions=3,        # Max 3 posisi bersamaan
    max_lot_size=0.05,      # Max 0.05 lot
    min_lot_size=0.01,      # Min 0.01 lot
    lot_step=0.01,          # Increment 0.01
)
```

### 2. SMCConfig

```python
SMCConfig(
    swing_length=5,         # 5 bar untuk swing detection
    fvg_min_gap_pips=2.0,   # Min gap FVG: 2 pips
    ob_lookback=10,         # Order block lookback: 10 bar
    bos_close_break=True,   # Butuh close break untuk BOS
)
```

### 3. MLConfig

```python
MLConfig(
    model_path="models/xgboost_model.json",
    confidence_threshold=0.65,   # Min confidence untuk entry
    retrain_frequency_days=7,    # Retrain setiap 7 hari
    lookback_periods=1000,       # Data lookback
)
```

### 4. ThresholdsConfig

```python
ThresholdsConfig(
    # ML Confidence
    ml_min_confidence=0.65,          # Minimum confidence
    ml_entry_confidence=0.70,        # Default entry
    ml_high_confidence=0.75,         # High confidence
    ml_very_high_confidence=0.80,    # Lot multiplier trigger

    # Risk
    trend_reversal_confidence=0.75,  # Trigger reversal close
    protected_mode_threshold=0.80,   # Enter protected mode

    # Profit/Loss (USD)
    min_profit_to_secure=15.0,       # Min profit to consider secure
    good_profit_level=25.0,          # Good profit
    great_profit_level=40.0,         # Take it!

    # Timing
    trade_cooldown_seconds=300,      # 5 menit antar trade
    loop_interval_seconds=30.0,      # Main loop interval

    # Session
    sydney_lot_multiplier=0.5,       # Sydney lot reduction
)
```

### 5. RegimeConfig

```python
RegimeConfig(
    n_regimes=3,            # 3 HMM states
    lookback_periods=500,   # HMM training lookback
    retrain_frequency=20,   # Retrain setiap 20 bar
)
```

---

## Environment Variables (.env)

| Variable | Contoh | Wajib | Keterangan |
|----------|--------|-------|------------|
| `MT5_LOGIN` | `12345678` | Ya | Akun MT5 |
| `MT5_PASSWORD` | `p@ssw0rd` | Ya | Password MT5 |
| `MT5_SERVER` | `BrokerName-Live` | Ya | Server broker |
| `MT5_PATH` | `C:\...\terminal64.exe` | Tidak | Path MT5 |
| `CAPITAL` | `5000` | Tidak | Modal ($5000 default) |
| `SYMBOL` | `XAUUSD` | Tidak | Simbol trading |
| `RISK_PER_TRADE` | `1.0` | Tidak | Override risk % |
| `MAX_DAILY_LOSS_PERCENT` | `3.0` | Tidak | Override daily loss |
| `AI_CONFIDENCE_THRESHOLD` | `0.65` | Tidak | Override ML threshold |
| `TELEGRAM_BOT_TOKEN` | `123:ABC...` | Tidak | Token Telegram |
| `TELEGRAM_CHAT_ID` | `-1001234...` | Tidak | Chat ID Telegram |
| `DB_HOST` | `localhost` | Tidak | PostgreSQL host |
| `DB_NAME` | `trading_db` | Tidak | Database name |

---

## Position Sizing (Kelly Criterion)

```python
def calculate_position_size(entry_price, stop_loss_price, balance):
    """
    Risk-Constrained Kelly Criterion:

    risk_amount = balance × risk% ($5000 × 1% = $50)
    sl_pips = |entry - SL| / 0.1
    lot = risk_amount / (sl_pips × pip_value)
    lot × 0.5 (Half-Kelly untuk safety)

    Clamp: min_lot ≤ lot ≤ max_lot
    """
```

---

## Validasi Otomatis

```
Saat TradingConfig dibuat:
    |
    v
_validate_required_settings():
├── MT5_LOGIN != 0?
├── MT5_PASSWORD tidak kosong?
├── MT5_SERVER tidak kosong?
└── Capital > 0?
    |
    ├── Ada yang gagal → ValueError
    └── Semua OK → _configure_by_capital()
```
