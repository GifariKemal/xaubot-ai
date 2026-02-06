# Risk Management

> **File utama:** `src/smart_risk_manager.py`
> **File pendukung:** `src/risk_engine.py`, `src/position_manager.py`
> **Konfigurasi:** `src/config.py`

---

## Apa Itu Risk Management?

Risk Management adalah sistem **pelindung modal** yang menentukan **seberapa besar** boleh trading, **kapan harus berhenti**, dan **bagaimana mengelola posisi terbuka**. Ini adalah komponen paling kritis — tanpa risk management yang baik, bahkan strategi terbaik pun bisa bangkrut.

**Analogi:** Risk Management adalah **sabuk pengaman + airbag + rem ABS** — melindungi dari kerugian fatal.

---

## 3 Modul Risk Management

| Modul | File | Fungsi |
|-------|------|--------|
| **SmartRiskManager** | `smart_risk_manager.py` | Ultra-safe position sizing & daily limits |
| **RiskEngine** | `risk_engine.py` | Kelly Criterion & circuit breaker |
| **SmartPositionManager** | `position_manager.py` | Trailing stop & profit protection |

---

## Trading Mode (4 State)

Bot beroperasi dalam salah satu dari 4 mode:

```
NORMAL -> RECOVERY -> PROTECTED -> STOPPED
  |         |            |            |
  |    3 loss berturut   |    80% limit tercapai
  |                      |            |
  |                      |    100% limit -> STOP total
  v                      v
Trading penuh      Lot minimum saja
```

| Mode | Bisa Trade? | Lot Size | Kondisi |
|------|------------|----------|---------|
| **NORMAL** | Ya | 0.01 - 0.02 | Operasi standar |
| **RECOVERY** | Ya | 0.01 saja | Setelah 3 loss berturut |
| **PROTECTED** | Ya | 0.01 saja | Daily loss 80% dari limit |
| **STOPPED** | Tidak | 0.00 | Daily/total limit tercapai |

### Transisi Mode (Prioritas tinggi ke rendah)

```
1. Cek total_loss >= $500 (10%)      -> STOPPED
2. Cek daily_loss >= $250 (5%)       -> STOPPED
3. Cek total_loss >= $400 (80%)      -> PROTECTED
4. Cek daily_loss >= $200 (80%)      -> PROTECTED
5. Cek consecutive_losses >= 3       -> RECOVERY
6. Sisanya                           -> NORMAL
```

---

## Kalkulasi Lot Size

### Formula

```
calculate_lot_size(entry_price, confidence, regime, ml_confidence):

1. Base lot = 0.01

2. Cek trading mode:
   NORMAL    -> lot 0.01 - 0.02
   RECOVERY  -> lot 0.01 (fixed)
   PROTECTED -> lot 0.01 (fixed)
   STOPPED   -> lot 0.00 (tidak trade)

3. Cek ML confidence:
   effective = min(confidence, ml_confidence)

   >= 0.65 -> lot 0.02 (HIGH)
   >= 0.55 -> lot 0.01 (MEDIUM)
   <  0.55 -> lot 0.01 (LOW)

4. Cek regime:
   high_volatility / crisis -> paksa lot 0.01

5. Apply session multiplier:
   Sydney session -> lot * 0.5
   London-NY overlap -> lot * 1.2

6. Cap ke max_allowed_lot berdasarkan state
7. Round ke increment 0.01
```

### Contoh Perhitungan

```
Input:
  confidence = 0.78 (SMC)
  ml_confidence = 0.72 (XGBoost)
  regime = "medium_volatility"
  session = "London"

Langkah:
  1. Mode = NORMAL
  2. effective = min(0.78, 0.72) = 0.72 >= 0.65 -> lot = 0.02
  3. Regime = medium -> tidak override
  4. Session = London (1.0x) -> lot tetap 0.02
  5. Final lot = 0.02
```

```
Input:
  confidence = 0.65
  ml_confidence = 0.60
  regime = "high_volatility"
  session = "Sydney"

Langkah:
  1. Mode = NORMAL
  2. effective = min(0.65, 0.60) = 0.60 >= 0.55 -> lot = 0.01
  3. Regime = high_volatility -> paksa lot 0.01
  4. Session = Sydney (0.5x) -> lot = max(0.01, 0.01*0.5) = 0.01
  5. Final lot = 0.01
```

---

## Limit Proteksi (untuk modal $5,000)

### Per Trade
| Proteksi | Persentase | Nilai | Mekanisme |
|----------|-----------|-------|-----------|
| **Software S/L** | 1.0% | $50 | Bot tutup posisi otomatis |
| **Emergency Broker S/L** | 2.0% | $100 | SL broker sebagai safety net |

### Per Hari
| Proteksi | Persentase | Nilai | Aksi |
|----------|-----------|-------|------|
| **Warning** | 4.0% (80%) | $200 | Mode -> PROTECTED (lot minimum) |
| **Daily Loss Limit** | 5.0% | $250 | Mode -> STOPPED (berhenti total) |

### Total (Kumulatif)
| Proteksi | Persentase | Nilai | Aksi |
|----------|-----------|-------|------|
| **Warning** | 8.0% (80%) | $400 | Mode -> PROTECTED |
| **Total Loss Limit** | 10.0% | $500 | Mode -> STOPPED permanen |

---

## Position Limit

```
Max concurrent positions: 2

Cek sebelum buka posisi baru:
  can_open_position():
    jika active_positions >= 2:
      return False, "Max positions reached (2/2)"
    else:
      return True, "OK"
```

---

## Manajemen Posisi Terbuka

### Evaluasi Posisi (`evaluate_position()`)

Setiap posisi terbuka dievaluasi setiap loop:

```
1. TAKE PROFIT CHECK
   Jika profit >= $40:
     -> TUTUP (exit_reason: TAKE_PROFIT)

2. ML REVERSAL CHECK (v3: threshold diturunkan)
   Jika ML confidence > 65% berlawanan arah:       <- sebelumnya 70%
     DAN loss >= 40% dari max ($20):
       -> TUTUP (exit_reason: TREND_REVERSAL)

3. MAX LOSS CHECK (50% threshold)
   Jika loss >= $25 (50% dari $50 max):
     Kecuali golden time DAN momentum > -40:
       -> TUTUP (exit_reason: POSITION_LIMIT)

4. STALL DETECTION
   Jika harga stall 10+ candle DAN loss >= $15:
     stall_count++
     Jika stall_count >= 5:
       -> TUTUP (exit_reason: STALL)

5. PROFIT PROTECTION (Peak Tracking)
   Jika peak_profit > $30 DAN current < 60% dari peak:
     -> TUTUP (lindungi profit)

6. TIME-BASED EXIT (v3: BARU)
   Jika posisi terbuka >= 4 jam DAN profit < $5:
     - Jika profit >= $0 -> TUTUP (breakeven/profit kecil)
     - Jika profit > -$15 -> TUTUP (loss kecil, daripada makin besar)
   Jika posisi terbuka >= 6 jam:
     -> FORCE EXIT (tutup apapun kondisinya)
```

### Time-Based Exit Detail (v3 Update)

```
Jam 0        Jam 4              Jam 6
|------------|------------------|-----> waktu
             |                  |
             | profit < $5?     | FORCE EXIT
             | Ya -> evaluasi:  | (apapun kondisinya)
             |   profit >= 0    |
             |     -> tutup OK  |
             |   loss > -$15    |
             |     -> tutup     |
             |   loss <= -$15   |
             |     -> tahan     |

Kenapa perlu time-based exit?
  - Trade yang stuck tanpa progress = buang waktu & margin
  - Lebih baik exit kecil daripada menunggu loss besar
  - Mencegah posisi "zombie" yang tidak kemana-mana
```

### Broker Stop Loss (v3: ATR-Based Protection)

**Perubahan utama v3:** Bot sekarang mengirim **SL ke broker** (bukan SL=0 seperti sebelumnya).

```python
# v2 (lama): Tidak ada proteksi broker
result = mt5.send_order(sl=0, ...)  # Bergantung 100% pada software

# v3 (baru): ATR-based broker protection
broker_sl = signal.stop_loss  # SL dari SMC (ATR-based, min 1.5 ATR)

# Validasi jarak minimum (10 pips untuk XAUUSD)
min_sl_distance = 1.0  # $1 = 10 pips
if direction == "BUY" and current_price - broker_sl < min_sl_distance:
    broker_sl = current_price - (min_sl_distance * 2)  # Paksa lebih lebar
if direction == "SELL" and broker_sl - current_price < min_sl_distance:
    broker_sl = current_price + (min_sl_distance * 2)  # Paksa lebih lebar

result = mt5.send_order(sl=broker_sl, ...)  # SL AKTIF di broker
```

**Fallback jika broker reject SL:**
```python
# Error code 10016 = SL/TP rejected
if not result.success and result.retcode == 10016:
    # Fallback ke software SL (tanpa broker protection)
    result = mt5.send_order(sl=0, ...)  # Software tetap mengelola
```

### Emergency Stop Loss (Safety Net Terakhir)

```python
calculate_emergency_sl(entry_price, lot_size, direction):
    pip_value = lot_size * 10  # XAUUSD
    emergency_pips = emergency_sl_usd / pip_value  # $100 / pip_value
    price_distance = emergency_pips * 0.01

    if direction == "BUY":
        sl = entry_price - price_distance
    else:
        sl = entry_price + price_distance
```

### Perbandingan Proteksi Lama vs Baru

```
┌─────────────────┬───────────────────────┬──────────────────────────┐
│ Skenario        │ Sebelum (v2)          │ Sesudah (v3)             │
├─────────────────┼───────────────────────┼──────────────────────────┤
│ Weekend Gap     │ Loss unlimited        │ Broker SL aktif          │
├─────────────────┼───────────────────────┼──────────────────────────┤
│ Flash Crash     │ Bergantung software   │ Broker SL aktif          │
├─────────────────┼───────────────────────┼──────────────────────────┤
│ Connection Lost │ Loss unlimited        │ Broker SL aktif          │
├─────────────────┼───────────────────────┼──────────────────────────┤
│ Trade Stuck     │ Ditahan selamanya     │ Exit max 6 jam           │
├─────────────────┼───────────────────────┼──────────────────────────┤
│ Reversal Lambat │ Tunggu 70% confidence │ Exit di 65% (lebih cepat)│
└─────────────────┴───────────────────────┴──────────────────────────┘
```

---

## Circuit Breaker (RiskEngine)

```python
# Automatic halt jika kondisi darurat
if daily_pnl_percent <= -max_daily_loss:
    activate_circuit_breaker("Daily loss limit breached")
    can_trade = False

# Flash crash protection
if price_move > flash_crash_threshold (2.5%):
    activate_circuit_breaker("Flash crash detected")
    can_trade = False
```

---

## Drawdown Tracking

### Daily Drawdown
```python
# Saat loss:
daily_loss += abs(profit)
total_loss += abs(profit)
consecutive_losses += 1

# Saat profit:
total_loss = max(0, total_loss - profit)  # Recovery
consecutive_losses = 0  # Reset
```

### Peak Equity Drawdown
```python
# Track peak equity
if equity > peak_equity:
    peak_equity = equity

# Hitung drawdown
drawdown = ((peak_equity - equity) / peak_equity) * 100
```

### Per-Position Peak Tracking
```python
# Track peak profit per posisi
peak_profits[ticket] = max(peak_profits[ticket], current_profit)

# Profit protection: tutup jika profit turun 40% dari peak
if current_profit < peak_profit * 0.6:
    close_position()  # Lindungi profit
```

---

## Daily Reset

```python
check_new_day():
    if date.today() != current_date:
        # Reset semua counter harian
        daily_loss = 0
        daily_trades = 0
        consecutive_losses = 0
        mode = NORMAL (jika total_loss OK)
        current_date = today
```

---

## Integrasi dalam Main Loop

```
Main Trading Loop (setiap 1 detik)
    |
    v
1. check_new_day()         <- Reset harian
    |
    v
2. get_trading_recommendation()
    |-- can_trade? -> Jika False, skip
    |-- mode? -> Tentukan lot limit
    |
    v
3. calculate_lot_size()    <- Hitung lot aman
    |-- Input: confidence, regime, ml_confidence
    |-- Output: lot 0.01-0.02
    |
    v
4. Apply session_multiplier <- Sydney 0.5x, Golden 1.2x
    |
    v
5. can_open_position()     <- Cek limit posisi (max 2)
    |
    v
6. execute_trade()         <- Kirim order ke MT5 (v3: DENGAN broker SL)
    |-- broker_sl = signal.stop_loss (ATR-based)
    |-- Fallback sl=0 jika broker reject
    |-- register_position() <- Track posisi baru + entry_time
    |
    v
7. evaluate_position()     <- Monitor posisi terbuka
    |-- Cek TP, ML reversal (65%), max loss, stall
    |-- Cek time-based exit (4 jam / 6 jam)   <- v3 BARU
    |
    v
8. record_trade_result()   <- Catat profit/loss
    |-- Update daily_loss, total_loss
    |-- Cek apakah limit tercapai
```

---

## Semua Parameter Konfigurasi

| Parameter | Nilai | Fungsi |
|-----------|-------|--------|
| `capital` | $5,000 | Modal awal |
| `max_daily_loss_percent` | 5.0% | Limit harian ($250) |
| `max_total_loss_percent` | 10.0% | Limit kumulatif ($500) |
| `max_loss_per_trade_percent` | 1.0% | Software SL ($50) |
| `emergency_sl_percent` | 2.0% | Broker SL ($100) |
| `base_lot_size` | 0.01 | Lot minimum |
| `max_lot_size` | 0.02 | Lot maximum |
| `recovery_lot_size` | 0.01 | Lot saat recovery |
| `trend_reversal_threshold` | **0.65** | ML confidence untuk tutup (v3: diturunkan dari 0.70) |
| `max_concurrent_positions` | 2 | Posisi terbuka max |
| `flash_crash_threshold` | 2.5% | Deteksi crash |
| `breakeven_pips` | 15.0 | Pindah SL ke breakeven |
| `trail_start_pips` | 25.0 | Mulai trailing stop |
| `trail_step_pips` | 10.0 | Jarak trailing |

---

## Sinkronisasi Backtest (backtest_live_sync.py)

Backtest menggunakan **logika exit yang identik** dengan live trading:

```
Exit reversal:   0.65 (65% ML confidence)    <- synced dengan live
Time-based exit:
  16 bars (4 jam M15) + profit < $5 -> exit
  24 bars (6 jam M15)              -> force exit

Perhitungan bar:
  bars_since_entry = current_bar_index - entry_bar_index
  16 bars * 15 menit = 4 jam
  24 bars * 15 menit = 6 jam
```

**Kenapa penting disinkronkan?** Agar hasil backtest akurat mewakili performa live trading.

---

## Filosofi Kunci

1. **Dual-Layer SL** — ATR-based broker SL + software-managed exit (v3 update)
2. **Ultra-Conservative** — Lot 0.01-0.02 saja, tidak pernah agresif
3. **Multi-Layer Protection** — Per-trade, per-day, total limit, circuit breaker
4. **Recovery First** — Setelah loss, otomatis masuk mode defensif
5. **Profit Protection** — Jika profit sudah besar, lindungi dari drawback
6. **Time-Bounded** — Tidak ada posisi "zombie", max 6 jam (v3 update)
7. **Faster Reversal** — Exit lebih cepat di 65% ML confidence (v3 update)
