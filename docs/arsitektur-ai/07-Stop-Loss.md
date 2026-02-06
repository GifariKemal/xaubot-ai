# Stop Loss (S/L) — Sistem Proteksi Berlapis

> **File terkait:** `src/smc_polars.py`, `main_live.py`, `src/smart_risk_manager.py`

---

## Apa Itu Stop Loss di Bot Ini?

Stop Loss bukan hanya satu angka — ini adalah **sistem proteksi 4 lapis** yang bekerja bersamaan. Jika satu layer gagal, layer berikutnya siap melindungi.

**Analogi:** SL di bot ini seperti sistem keamanan gedung — ada CCTV (software monitoring), security (broker SL), alarm kebakaran (emergency SL), dan sprinkler otomatis (circuit breaker).

---

## 4 Layer Stop Loss

```
Layer 1: SMC ATR-Based SL      <- Dikirim ke broker sebagai SL aktif
Layer 2: Software Smart Exit    <- Bot monitor & tutup posisi secara cerdas
Layer 3: Emergency Broker SL    <- Safety net 2% jika software gagal
Layer 4: Circuit Breaker        <- Halt total jika flash crash / daily limit
```

```
Harga masuk (Entry)
    |
    |-- Layer 1: SMC SL (1.5 ATR)         contoh: -$15 ~ -$30
    |
    |-- Layer 2: Software SL ($25-$50)     50% dari max loss
    |
    |-- Layer 3: Emergency SL ($100)       2% modal (jaring terakhir)
    |
    |-- Layer 4: Circuit Breaker           Flash crash 2.5% -> HALT
    |
    v
Semakin jauh = semakin jarang tercapai (backup)
```

---

## Layer 1: SMC ATR-Based Stop Loss

**Sumber:** `smc_polars.py` (Lines 631-652, 694-702)
**Dikirim ke:** Broker MT5 sebagai SL order aktif

### Perhitungan

```python
# Ambil ATR dari Feature Engineering
atr = latest["atr"]                    # Contoh: ATR = $8.50
min_sl_distance = 1.5 * atr            # 1.5 * 8.50 = $12.75

# Untuk BUY:
swing_sl = last_swing_low              # Contoh: $4935.00
atr_sl   = entry - min_sl_distance     # $4950 - $12.75 = $4937.25
SL       = MIN(swing_sl, atr_sl)       # $4935.00 (pilih yang LEBIH JAUH)

# Untuk SELL:
swing_sl = last_swing_high             # Contoh: $4965.00
atr_sl   = entry + min_sl_distance     # $4950 + $12.75 = $4962.75
SL       = MAX(swing_sl, atr_sl)       # $4965.00 (pilih yang LEBIH JAUH)
```

### Kenapa MIN/MAX (Pilih yang Lebih Jauh)?

```
Sebelum (v2): SL = swing_low ATAU entry * 0.995
  -> Bisa sangat dekat, gampang kena whipsaw

Sesudah (v3): SL = MIN(swing_low, entry - 1.5*ATR)
  -> Selalu minimal 1.5 ATR dari entry
  -> Lebih protektif terhadap noise pasar
```

---

## Layer 2: Software Smart Exit

**Sumber:** `smart_risk_manager.py` (Lines 559-724)
**Mekanisme:** Bot monitor posisi setiap detik dan tutup otomatis

### Kondisi Software SL

```
Max loss per trade: $50 (1% dari modal $5,000)

Trigger exit jika:
  1. Loss >= $25 (50% dari max)
     Kecuali golden time DAN momentum > -40 -> hold

  2. Loss >= $20 (40%) + ML reversal 65%+ berlawanan
     -> Tutup karena trend reversal

  3. Loss >= $15 + harga stall 10+ candle
     stall_count >= 5 -> tutup

  4. 4+ jam terbuka + profit < $5
     -> Time-based exit

  5. 6+ jam terbuka
     -> Force exit (apapun kondisinya)
```

### Kelebihan Software SL vs Hard SL

```
Hard SL (broker):
  - Kaku, tidak bisa diubah
  - Bisa kena whipsaw lalu harga balik
  - Tidak bisa mempertimbangkan konteks

Software SL (bot):
  - Dinamis, mempertimbangkan momentum
  - Bisa hold jika golden time & momentum positif
  - Bisa exit lebih cepat jika ML deteksi reversal
  - Mempertimbangkan durasi posisi
```

---

## Layer 3: Emergency Broker Stop Loss

**Sumber:** `smart_risk_manager.py` (Lines 305-346)
**Fungsi:** Jaring pengaman TERAKHIR jika software gagal (disconnect, crash, dll)

### Perhitungan

```python
# Konfigurasi: 2% dari modal
emergency_sl_percent = 2.0
emergency_sl_usd = 5000 * 0.02 = $100    # Max loss jika software gagal

# Hitung jarak SL
pip_value = lot_size * 10                 # 0.01 lot -> $0.10/pip
emergency_pips = $100 / $0.10 = 1000 pips
price_distance = 1000 * 0.01 = $10.00

# SL price
BUY:  SL = entry - $10.00 = $4940.00
SELL: SL = entry + $10.00 = $4960.00
```

### Kapan Emergency SL Tercapai?

Seharusnya **tidak pernah** — software SL ($50) akan menutup jauh sebelum emergency SL ($100). Emergency SL hanya tercapai jika:
- Bot crash / disconnect
- Server bermasalah
- Internet putus
- Harga gap melewati semua level

---

## Layer 4: Circuit Breaker

**Sumber:** `risk_engine.py` (Lines 143-151)
**Fungsi:** Halt trading total saat kondisi darurat

```python
# Flash crash: Pergerakan > 2.5% dalam waktu singkat
if price_move > flash_crash_threshold:
    activate_circuit_breaker("Flash crash detected")
    # Tutup SEMUA posisi
    # Block semua trade baru
    # Kirim alert Telegram

# Daily loss limit
if daily_pnl_percent <= -5.0%:
    activate_circuit_breaker("Daily loss limit breached")
```

---

## Pengiriman SL ke Broker (main_live.py)

### Flow Pengiriman

```python
# Step 1: Ambil SL dari SMC signal (ATR-based)
broker_sl = signal.stop_loss

# Step 2: Validasi jarak minimum (10 pips untuk XAUUSD)
min_sl_distance = 1.0  # $1 = 10 pips

if direction == "BUY":
    if current_price - broker_sl < 1.0:
        broker_sl = current_price - 2.0    # Paksa lebih lebar

if direction == "SELL":
    if broker_sl - current_price < 1.0:
        broker_sl = current_price + 2.0    # Paksa lebih lebar

# Step 3: Kirim order DENGAN SL
result = mt5.send_order(
    sl=broker_sl,       # SL AKTIF di broker
    tp=signal.take_profit,
    ...
)

# Step 4: Fallback jika broker reject (error 10016)
if not result.success and retcode == 10016:
    # SL terlalu dekat / tidak valid
    result = mt5.send_order(
        sl=0,            # Tanpa broker SL
        comment="AI Safe v3 NoSL"
    )
    # Software SL tetap aktif sebagai proteksi
```

---

## Tabel Ringkasan Layer SL

| Layer | Sumber | Jarak dari Entry | Max Loss | Kondisi Trigger |
|-------|--------|-----------------|----------|-----------------|
| **1. SMC ATR** | Broker SL aktif | 1.5 ATR (~$12-15) | ~$15-30 | Harga hit SL level |
| **2. Software** | Bot monitoring | Dinamis | $25-50 | Loss threshold + konteks |
| **3. Emergency** | Broker safety net | 2% modal ($10) | $100 | Software gagal |
| **4. Circuit** | Halt total | Semua posisi | Unlimited cap | Flash crash / daily limit |

---

## Skenario Proteksi

### Skenario 1: Trading Normal

```
Entry BUY @ $4950, SL broker @ $4937 (1.5 ATR)
  -> Harga turun ke $4938 -> Masih aman
  -> Harga turun ke $4936 -> BROKER SL HIT -> Tutup otomatis
  -> Loss: ~$14 (0.01 lot)
```

### Skenario 2: Connection Lost

```
Entry BUY @ $4950, SL broker @ $4937
  -> Bot disconnect
  -> Harga turun drastis ke $4920
  -> BROKER SL sudah aktif di $4937 -> Tutup otomatis
  -> Loss: ~$13 (bukan unlimited!)
```

### Skenario 3: Weekend Gap

```
Jumat: Entry BUY @ $4950, SL broker @ $4937
  -> Senin buka gap di $4910 (melewati SL)
  -> Broker eksekusi SL di harga terbaik ~$4910
  -> Loss: ~$40 (lebih dari SL tapi terproteksi)
```

### Skenario 4: Flash Crash (Tanpa Broker SL Fallback)

```
Entry BUY @ $4950, sl=0 (broker reject)
  -> Harga jatuh cepat ke $4925
  -> Software: loss = $25 >= 50% max -> TUTUP
  -> Loss: ~$25 (software protect)

  -> Jika software juga gagal:
     Emergency SL @ $4940 -> TUTUP
     -> Loss: ~$100 max
```
