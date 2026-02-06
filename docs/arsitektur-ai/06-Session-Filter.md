# Session Filter

> **File:** `src/session_filter.py`
> **Class:** `SessionFilter`
> **Timezone:** WIB (Waktu Indonesia Barat / GMT+7)

---

## Apa Itu Session Filter?

Session Filter menentukan **kapan bot boleh trading** berdasarkan sesi pasar global. Setiap sesi memiliki karakteristik berbeda — volatilitas, likuiditas, dan spread. Bot menyesuaikan perilaku berdasarkan sesi yang sedang aktif.

**Analogi:** Session Filter adalah **jadwal kerja** — bot tahu kapan harus bekerja keras, kapan santai, dan kapan istirahat.

---

## 7 Sesi yang Didefinisikan

| Sesi | Enum | Waktu (WIB) | Volatilitas | Multiplier |
|------|------|-------------|-------------|------------|
| **Sydney** | `SYDNEY` | 06:00 - 13:00 | Low | 0.5x |
| **Tokyo** | `TOKYO` | 07:00 - 16:00 | Medium | 0.7x |
| **London** | `LONDON` | 15:00 - 23:59 | High | 1.0x |
| **New York** | `NEW_YORK` | 20:00 - 23:59 | Extreme | 1.0x |
| **Tokyo-London Overlap** | `OVERLAP_TOKYO_LONDON` | 15:00 - 16:00 | High | 1.0x |
| **London-NY Overlap** | `OVERLAP_LONDON_NY` | 20:00 - 23:59 | Extreme | **1.2x** |
| **Off Hours** | `OFF_HOURS` | Diluar sesi | - | 0.0x |

---

## Visualisasi Timeline (WIB)

```
JAM WIB:  00  02  04  06  08  10  12  14  16  18  20  22  24
          |---|---|---|---|---|---|---|---|---|---|---|---|---|
DANGER:   [=========]   <- Dead Zone (00-04)
DANGER:       [===]     <- Rollover (04-06)
SYDNEY:           [===========]                     0.5x
TOKYO:                [=============]               0.7x
OVERLAP T-L:                      [=]               1.0x
LONDON:                       [===================] 1.0x
NEW YORK:                                 [=======] 1.0x
GOLDEN:                                   [=======] 1.2x ★
          |---|---|---|---|---|---|---|---|---|---|---|---|---|
          00  02  04  06  08  10  12  14  16  18  20  22  24
```

**★ GOLDEN TIME (20:00-23:59 WIB):** Waktu terbaik — likuiditas tertinggi, London & NY overlap.

---

## Zona Bahaya (Danger Zones)

| Zona | Waktu (WIB) | Alasan | Aksi |
|------|-------------|--------|------|
| **Dead Zone** | 00:00 - 04:00 | Likuiditas rendah, spread tinggi | Block trading |
| **Rollover** | 04:00 - 06:00 | Spread melebar saat rollover broker | Block trading |

---

## Logika `can_trade()` — Keputusan Utama

```
can_trade() -> (bool, str, float)
               bisa?  alasan  multiplier

Langkah pengecekan (urut prioritas):

1. Weekend?
   |-- Sabtu / Minggu -> (False, "Market tutup", 0.0)

2. Jumat >= 23:00?
   |-- Ya -> (False, "Hindari gap weekend", 0.0)

3. Danger Zone?
   |-- 00:00-04:00 -> (False, "Likuiditas rendah", 0.0)
   |-- 04:00-06:00 -> (False, "Spread melebar", 0.0)

4. Sesi saat ini?
   |-- Cek overlap dulu (prioritas tertinggi)
   |-- Lalu cek sesi utama

5. allow_trading flag?
   |-- False -> (False, "Tidak diizinkan", 0.0)

6. Aggressive Mode?
   |-- Sydney -> (True, "SAFE MODE 0.5x", 0.5)
   |-- Low volatility -> (False, "Tunggu sesi volatile", mult)
   |-- High/Extreme -> (True, "Trading OK", mult)

7. Default
   |-- (True, "Trading OK - {sesi}", multiplier)
```

---

## Prioritas Deteksi Sesi

```python
# Overlap dicek PERTAMA (prioritas tertinggi)
1. London-NY Overlap (20:00-23:59)  -> GOLDEN TIME 1.2x
2. Tokyo-London Overlap (15:00-16:00)

# Lalu sesi utama
3. London (15:00-23:59)
4. New York (20:00-23:59)
5. Tokyo (07:00-16:00)
6. Sydney (06:00-13:00)

# Terakhir
7. Off Hours (default)
```

---

## Dampak ke Position Sizing

Session multiplier diterapkan **setelah** kalkulasi lot dari SmartRiskManager:

```
Lot dasar dari Risk Manager: 0.02
    |
    v
Session multiplier:
  Sydney (0.5x):     0.02 * 0.5 = 0.01
  Tokyo (0.7x):      0.02 * 0.7 = 0.014 -> 0.01 (rounded)
  London (1.0x):     0.02 * 1.0 = 0.02
  Golden (1.2x):     0.02 * 1.2 = 0.024 -> 0.02 (capped)
    |
    v
Final lot (min 0.01, max 0.02)
```

---

## Weekend & Friday Handling

### Weekend
```
Sabtu (weekday=5): Market tutup -> tidak trading
Minggu (weekday=6): Market tutup -> tidak trading
```

### Friday Close
```
Jumat >= 23:00 WIB:
  -> Block semua trade baru
  -> Alasan: Hindari gap weekend (harga bisa gap besar saat buka Senin)
```

---

## News Event Monitoring

### Event yang Dipantau

| Event | Waktu (WIB) | Buffer Sebelum | Buffer Sesudah |
|-------|-------------|---------------|----------------|
| **NFP** (Non-Farm Payroll) | 19:30 | 15 menit | 30 menit |
| **FOMC** (Fed Decision) | 01:00 | 15 menit | 45 menit |
| **CPI** (Inflation) | 19:30 | 15 menit | 30 menit |

### Kebijakan News: MONITORING ONLY (Tidak Blocking)

```
Backtest menunjukkan:
  - Win rate saat news: 62.1%
  - Win rate normal: 64.9%
  - Selisih kecil, tapi BLOCKING news KEHILANGAN $178 profit

Keputusan: ML model sudah cukup menangani volatilitas news.
News hanya di-LOG, TIDAK memblokir trading.
```

---

## Aggressive Mode

Bot default menggunakan `aggressive_mode=True`:

```python
create_wib_session_filter(aggressive=True)
```

### Efek Aggressive Mode

| Sesi | Tanpa Aggressive | Dengan Aggressive |
|------|-----------------|-------------------|
| Sydney | Block | **Allow** (0.5x, proven profitable) |
| Tokyo | Allow | Block (volatilitas kurang) |
| London | Allow | Allow |
| New York | Allow | Allow |
| Golden | Allow | Allow (boost 1.2x) |

**Alasan Sydney diizinkan:** Backtest menunjukkan win rate 62% dan profit $5,934 di sesi Sydney.

---

## Golden Time (London-NY Overlap)

```
Waktu: 20:00 - 23:59 WIB
Multiplier: 1.2x (BOOSTED)
Volatilitas: Extreme

Kenapa spesial?
  - London dan New York sama-sama aktif
  - Likuiditas TERTINGGI sepanjang hari
  - Pergerakan harga paling signifikan
  - Volume trading terbesar

Aturan tambahan di main_live.py:
  - Require ML + SMC alignment (keduanya harus setuju)
  - Lot boleh lebih besar (1.2x multiplier)
```

---

## Integrasi dalam Main Loop

```python
# 1. Inisialisasi
self.session_filter = create_wib_session_filter(aggressive=True)

# 2. Cek setiap loop
session_ok, session_reason, session_multiplier = self.session_filter.can_trade()

if not session_ok:
    # Log setiap 5 menit
    logger.info(f"Session: {session_reason}")
    next = self.session_filter.get_next_trading_window()
    logger.info(f"Next: {next['session']} in {next['hours_until']} hours")
    return  # Skip, tidak trading

# 3. Simpan multiplier untuk lot sizing
self._current_session_multiplier = session_multiplier

# 4. Apply ke lot size (setelah risk calculation)
safe_lot = max(0.01, safe_lot * session_multiplier)
```

---

## Status Report

```python
get_status_report() -> {
    "current_time_wib": "2026-02-06 20:15:00",
    "current_session": "London-NY Overlap",
    "volatility": "extreme",
    "can_trade": True,
    "reason": "Trading OK - GOLDEN TIME (1.2x)",
    "position_multiplier": 1.2,
    "is_weekend": False,
    "is_friday_close": False,
    "is_danger_zone": False,
}
```

---

## Contoh Skenario

**Skenario 1: Golden Time**
```
Waktu: 21:30 WIB (Rabu)
Sesi: London-NY Overlap
-> can_trade = True
-> multiplier = 1.2x
-> Lot 0.02 * 1.2 = 0.024 -> cap 0.02
-> Trading optimal!
```

**Skenario 2: Sydney pagi**
```
Waktu: 08:00 WIB (Selasa)
Sesi: Sydney
-> can_trade = True (aggressive mode)
-> multiplier = 0.5x
-> Lot 0.02 * 0.5 = 0.01
-> SAFE MODE: lot minimum
```

**Skenario 3: Dead zone**
```
Waktu: 02:30 WIB (Kamis)
Sesi: Off Hours (Danger Zone)
-> can_trade = False
-> Alasan: "Likuiditas rendah, spread tinggi"
-> Bot istirahat, tunggu sesi berikutnya
```

**Skenario 4: Jumat malam**
```
Waktu: 23:15 WIB (Jumat)
-> can_trade = False
-> Alasan: "Hindari gap weekend"
-> Tidak buka posisi baru
```
