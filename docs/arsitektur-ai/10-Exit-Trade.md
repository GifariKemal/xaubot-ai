# Exit Trade — Proses Keluar Posisi

> **File utama:** `main_live.py`, `src/smart_risk_manager.py`
> **File pendukung:** `src/position_manager.py`

---

## Apa Itu Exit Trade?

Exit Trade adalah keseluruhan proses **monitoring posisi terbuka** dan **memutuskan kapan menutup**. Bot memeriksa setiap posisi terbuka **setiap ~10 detik** (di antara candle) atau **setiap candle baru** (full analysis) dengan 10 kondisi exit berbeda.

**Analogi:** Exit Trade seperti **pilot otomatis di pesawat** — terus monitor ketinggian (profit), cuaca (momentum), bahan bakar (waktu), dan bisa landing darurat kapan saja.

---

## 2 Jalur Exit

```
Jalur 1: BROKER EXIT (otomatis, independen)
  -> Harga hit TP level -> tutup otomatis
  -> Harga hit SL level -> tutup otomatis
  -> Tidak perlu bot online

Jalur 2: SOFTWARE EXIT (cerdas, kontekstual)
  -> Bot evaluasi setiap 1 detik
  -> Mempertimbangkan momentum, ML, waktu, dll
  -> 10 kondisi exit berbeda
```

---

## Monitoring Loop

```python
# main_live.py Lines 1117-1215
# Setiap 1 detik, untuk SETIAP posisi terbuka:

for position in open_positions:
    # Update data posisi
    current_price = mt5.get_tick(symbol)
    current_profit = position.profit

    # Update history untuk analisis momentum
    guard.update_history(current_price, current_profit, ml_confidence)

    # Evaluasi: haruskah ditutup?
    should_close, reason, message = smart_risk.evaluate_position(
        ticket=ticket,
        current_price=current_price,
        current_profit=profit,
        ml_signal=ml_prediction.signal,
        ml_confidence=ml_prediction.confidence,
        regime=regime_state,
    )

    if should_close:
        # Tutup posisi
        close_position(ticket, reason)
```

---

## 10 Kondisi Exit (Urutan Pengecekan)

### CHECK 1: Smart Take Profit (profit >= $15)

```
Ketika profit sudah cukup besar, evaluasi apakah harus diamankan:

a) Hard TP: profit >= $40
   -> TUTUP langsung, target tercapai

b) Momentum TP: profit >= $25 DAN momentum < -30
   -> TUTUP, profit sedang turun cepat

c) Peak Protection: peak > $30 DAN current < 60% peak
   -> TUTUP, lindungi dari drawback lebih dalam

d) Probability TP: TP_prob < 25% DAN profit >= $20
   -> TUTUP, kemungkinan capai TP sudah rendah

e) Strong Momentum: momentum >= 0
   -> HOLD, biarkan profit berjalan (let it run)
```

---

### CHECK 2: Early Exit Small Profit ($5-$15)

```
Profit masih kecil tapi ada tanda bahaya:

IF profit $5-$15
AND momentum < -50 (turun sangat cepat)
AND ML confidence >= 65% berlawanan arah:
  -> TUTUP, ambil profit kecil sebelum hilang
```

---

### CHECK 3: Early Cut (v4 — Smart Hold DIHAPUS)

```
Posisi sedang rugi — potong cepat jika sinyal buruk:

IF profit < 0:
  Loss >= 30% max ($15 dari $50) DAN momentum < -30
     -> TUTUP CEPAT (early cut, jangan tunggu recovery)

v4 PERUBAHAN: "Smart Hold" DIHAPUS
  - SEBELUMNYA: Hold posisi rugi menunggu golden time (20:00-23:59)
  - SEBELUMNYA: Hold posisi rugi kecil di sesi London (15:00-19:00)
  - SEKARANG: Tidak ada lagi "hold losers hoping for recovery"
  - ALASAN: Menahan posisi rugi menunggu sesi tertentu = perilaku
    berbahaya (martingale mentality). Proper risk management:
    ikuti aturan SL, jangan berharap recovery.
```

---

### CHECK 4: Trend Reversal Detection

```
ML mendeteksi perubahan tren:

IF ML confidence >= 65% berlawanan dengan posisi:
  a) Loss > 40% max DAN profit < -$8
     -> TUTUP (reversal + loss signifikan)

  b) Akumulasi 3x reversal warning DAN loss < -$10
     -> TUTUP (multiple warnings = konfirmasi reversal)

  c) Belum memenuhi threshold
     -> reversal_warnings += 1 (catat warning)
```

---

### CHECK 5: Maximum Loss Per Trade

```
Loss mencapai batas toleransi:

IF loss >= 50% dari max_loss ($25 dari $50):
  Exception: golden_time <= 1 jam DAN momentum > -40
    -> HOLD (kesempatan terakhir recovery)

  Selain itu:
    -> TUTUP [S/L] Position loss limit
```

---

### CHECK 6: Stall Detection

```
Harga tidak bergerak kemana-mana:

IF 10 candle terakhir range profit < $3
AND current_profit < -$15:
  stall_count += 1

  IF stall_count >= 5:
    -> TUTUP [STALL] Harga stuck, buang waktu & margin
```

---

### CHECK 7: Daily Loss Limit

```
Mencegah daily loss limit terlampaui:

potential_daily_loss = daily_loss + abs(min(0, current_profit))

IF potential_daily_loss >= max_daily_loss ($250):
  -> TUTUP [LIMIT] Akan melampaui batas harian
```

---

### CHECK 8: Weekend Close

```
Proteksi dari gap weekend:

IF hari Jumat setelah 04:00 WIB:
  a) profit > 0
     -> TUTUP [WEEKEND] Amankan profit

  b) profit > -$10
     -> TUTUP [WEEKEND] Loss kecil, hindari gap

  c) profit <= -$10
     -> HOLD (loss terlalu besar untuk cut, evaluasi manual)
```

---

### CHECK 9: Smart Time-Based Exit (v5: Smarter — Don't Cut Winners)

```
Mencegah posisi "zombie" TAPI jangan potong posisi profit yang masih tumbuh.

trade_duration = (sekarang - entry_time) dalam jam
profit_growing = momentum > 0
ml_agrees      = ML signal searah dengan posisi

IF 4+ jam:
  a) profit < $5 DAN NOT profit_growing:
     - profit >= $0 -> TUTUP [TIMEOUT] Breakeven + no growth
     - profit > -$15 -> TUTUP [TIMEOUT] Small loss + no growth
  b) profit >= $5 DAN profit_growing DAN ml_agrees:
     -> HOLD (extend time — profit masih tumbuh!)
     -> Log: "extending time"

IF 6+ jam:
  a) profit < $10 ATAU NOT profit_growing:
     -> TUTUP [MAX TIME] (sebelumnya: force close apapun kondisi)
  b) profit >= $10 DAN profit_growing:
     -> EXTEND ke max 8 jam (v5 BARU — biarkan profit berjalan)
     -> Di jam 8+ -> TUTUP [MAX TIME] Take profit

v5 PERUBAHAN:
  - 4h: Sekarang cek profit growth, bukan hanya profit < $5
  - 6h: BUKAN force close lagi — extend ke 8h jika profitable + growing
  - ML agreement dipertimbangkan sebelum timeout
  - Prinsip: jangan potong pemenang yang masih berjalan
```

**Visualisasi:**

```
Jam:  0     1     2     3     4     5     6     7     8
      |-----|-----|-----|-----|-----|-----|-----|-----|
      entry                   |           |           |
                              |           |           |
                        4h check:    6h check:   8h FINAL EXIT
                        stuck?       profitable?
                        no growth?   growing?
                        -> exit      -> extend!
                                     not growing?
                                     -> exit
```

---

### CHECK 10: Default — HOLD

```
Tidak ada kondisi exit terpenuhi:

-> HOLD posisi
-> Log status: momentum, TP probability, ML signal
-> Evaluasi ulang di cek berikutnya (~10 detik atau candle baru)
```

---

## Exit Reason Enum

| Reason | Kode | Deskripsi |
|--------|------|-----------|
| `TAKE_PROFIT` | take_profit | Target profit tercapai |
| `TREND_REVERSAL` | trend_reversal | ML deteksi reversal |
| `DAILY_LIMIT` | daily_limit | Batas harian tercapai |
| `POSITION_LIMIT` | position_limit | Max loss per trade |
| `TOTAL_LIMIT` | total_limit | Batas total tercapai |
| `WEEKEND_CLOSE` | weekend_close | Penutupan Jumat |
| `TIMEOUT` | timeout | Time-based exit (4h/6h) |
| `STALL` | stall | Harga stuck |
| `MANUAL` | manual | Penutupan manual |

---

## Post-Exit Flow

```python
# Setelah posisi ditutup:

# 1. Record hasil trade
risk_result = smart_risk.record_trade_result(profit)
# Update: daily_loss, total_loss, consecutive_losses, mode

# 2. Unregister dari monitoring
smart_risk.unregister_position(ticket)

# 3. Log trade
trade_logger.log_trade_close(
    ticket, entry_price, exit_price, profit, pips,
    duration, exit_reason, ml_signal, regime, ...
)

# 4. Kirim notifikasi Telegram
await telegram.send_trade_close(trade_info)
# Format: WIN/LOSS/BE, P/L, pips, duration, balance

# 5. Cek limit violations
if risk_result["daily_limit_hit"]:
    await send_critical_alert("DAILY LOSS LIMIT")
    # Mode -> STOPPED, tidak ada trade lagi hari ini

if risk_result["total_limit_hit"]:
    await send_critical_alert("TOTAL LOSS LIMIT")
    # Mode -> STOPPED permanen
```

---

## Diagram Exit Flow

```
Setiap ~10 detik (atau candle baru), per posisi terbuka:
    |
    v
Update profit & momentum
    |
    v
[1] Profit >= $15? ----YES---> Smart TP evaluation
    |NO                         (hard/$40, momentum, peak, prob)
    v
[2] Profit $5-$15? ----YES---> Reversal + momentum drop?
    |NO                         -> Early exit
    v
[3] Profit < 0? -------YES---> Loss>30% + momentum<-30?
    |NO                         -> EARLY CUT
    v
[4] ML Reversal 65%+? -YES---> Loss > 40%? -> TUTUP
    |NO                         Else warning++
    v
[5] Loss >= 50% max? --YES---> TUTUP (kecuali golden time)
    |NO
    v
[6] Stall 10+ candle? -YES---> stall++ -> 5x? TUTUP
    |NO
    v
[7] Daily limit? ------YES---> TUTUP
    |NO
    v
[8] Friday close? -----YES---> TUTUP (profit>0 atau loss>-$10)
    |NO
    v
[9] Time >= 4h? -------YES---> profit<$5? TUTUP
    |   Time >= 6h? ---YES---> FORCE EXIT
    |NO
    v
[10] HOLD -> evaluasi ulang 1 detik kemudian
```
