# Risk Engine — Mesin Risiko & Circuit Breaker

> **File:** `src/risk_engine.py`
> **Class:** `RiskEngine`
> **Digunakan oleh:** `main_live.py`

---

## Apa Itu Risk Engine?

Risk Engine adalah **lapisan proteksi fundamental** yang menghitung ukuran posisi, memvalidasi order, dan mengaktifkan circuit breaker saat batas risiko terlampaui.

**Analogi:** Risk Engine seperti **sistem rem ABS di mobil** — menghitung kecepatan aman, memvalidasi manuver, dan menghentikan paksa jika ada bahaya.

---

## 4 Fungsi Utama

### 1. Position Sizing (Kelly Criterion)

```
Risk-Constrained Half-Kelly:

1. Hitung Kelly fraction:
   f* = (p × b - q) / b
   dimana:
     p = win rate (misal 0.55)
     q = 1 - p (0.45)
     b = avg win/loss ratio (misal 2.0)

2. Cap Kelly: max 25%

3. Half-Kelly: f* × 0.5 (safety)

4. Apply regime multiplier (0.5x - 1.0x)

5. Cap di config limit: max risk_per_trade%

6. Hitung lot:
   risk_amount = balance × actual_risk%
   lot = risk_amount / (SL_pips × pip_value)

7. Round ke lot_step, clamp ke min/max
```

**Contoh:**

```
Balance: $5,000
Win rate: 55%
Win/Loss ratio: 2.0
Kelly: (0.55 × 2.0 - 0.45) / 2.0 = 0.325 (32.5%)
Half-Kelly: 16.25%
Cap: min(16.25%, 1.0%) = 1.0%
Risk amount: $50
SL distance: 50 pips ($5 per pip per 0.01 lot)
Lot: $50 / (50 × $1) = 0.01 lot (menambahkan regime multiplier)
```

### 2. Risk Check (Real-time)

```python
check_risk(balance, equity, open_positions, current_price)
    |
    v
Hitung daily P/L: equity - starting_balance
    |
    v
Cek circuit breaker aktif? → can_trade = False
    |
    v
Daily loss >= max_daily_loss%? → CIRCUIT BREAKER
    |
    v
Posisi >= max_positions? → can_trade = False
    |
    v
Return RiskMetrics(daily_pnl, drawdown, can_trade, reason)
```

### 3. Order Validation

```python
validate_order(type, entry, sl, tp, lot, price, balance)
    |
    ├── Circuit breaker aktif? → REJECT
    ├── BUY: SL >= entry? → REJECT ("SL harus di bawah entry")
    ├── BUY: TP <= entry? → REJECT ("TP harus di atas entry")
    ├── Lot < minimum? → REJECT
    ├── Lot > maximum? → REJECT
    ├── Entry terlalu jauh dari current price (>0.1%)? → REJECT
    ├── Risk% > 1.5× config limit? → REJECT
    └── Semua OK → APPROVED
```

### 4. Circuit Breaker

```
TRIGGER:
  Daily loss >= max_daily_loss% (3% untuk $5K account)

EFEK:
  → can_trade = False
  → Semua entry baru DITOLAK
  → TIDAK menutup posisi yang ada

RESET:
  → Otomatis pada hari baru
  → Manual via reset_circuit_breaker()
```

---

## Daily Stats Tracking

```python
# Auto-initialize setiap hari baru
_daily_stats[today] = {
    "starting_balance": equity,   # Basis untuk % hitung
    "trades": 0,                  # Total trade hari ini
    "wins": 0,                    # Trade profit
    "losses": 0,                  # Trade loss
}
```

---

## Return Types

### RiskMetrics

```python
@dataclass
class RiskMetrics:
    daily_pnl: float           # P/L hari ini ($)
    daily_pnl_percent: float   # P/L hari ini (%)
    open_exposure: float       # Total exposure ($)
    max_drawdown: float        # Drawdown dari peak (%)
    position_count: int        # Jumlah posisi terbuka
    can_trade: bool            # Boleh buka posisi baru?
    reason: str                # Alasan
```

### PositionSizeResult

```python
@dataclass
class PositionSizeResult:
    lot_size: float            # Ukuran lot yang dihitung
    risk_amount: float         # Risk dalam USD
    risk_percent: float        # Risk dalam %
    stop_distance: float       # Jarak SL (harga)
    take_profit_distance: float # Jarak TP (harga)
    approved: bool             # Disetujui?
    rejection_reason: str      # Alasan penolakan
```

---

## Hubungan dengan Smart Risk Manager

```
RiskEngine (modul ini)
├── Kelly Criterion position sizing
├── Circuit breaker (daily loss limit)
├── Order validation
└── Foundational risk checks

SmartRiskManager (05-Risk-Management.md)
├── 4 trading modes (NORMAL/RECOVERY/PROTECTED/STOPPED)
├── Smart exit logic (10 kondisi)
├── Position monitoring per-detik
└── Higher-level risk decisions
```

**RiskEngine** adalah mesin kalkulasi dasar, **SmartRiskManager** adalah manajer tingkat tinggi yang menggunakannya.
