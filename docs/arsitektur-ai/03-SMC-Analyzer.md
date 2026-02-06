# SMC Analyzer (Smart Money Concepts)

> **File:** `src/smc_polars.py`
> **Framework:** Pure Polars (vectorized, tanpa loop)

---

## Apa Itu SMC?

Smart Money Concepts adalah metode analisis berdasarkan **cara institusi besar (bank, hedge fund) trading**. SMC membaca **struktur pasar** dan **jejak uang besar** untuk menemukan zona entry yang presisi.

**Analogi:** SMC adalah **peta jalan** — menunjukkan zona penting, rambu lalu lintas, dan rute terbaik.

---

## 6 Konsep yang Diimplementasikan

| # | Konsep | Fungsi | Lines |
|---|--------|--------|-------|
| 1 | Swing Points | Puncak & lembah penting | 185-261 |
| 2 | Fair Value Gap (FVG) | Imbalance/gap harga | 84-183 |
| 3 | Order Block (OB) | Zona order institusi | 263-368 |
| 4 | Break of Structure (BOS) | Kelanjutan tren | 370-457 |
| 5 | Change of Character (CHoCH) | Pembalikan tren | 370-457 |
| 6 | Liquidity Zones | Kumpulan stop loss | 459-551 |

---

## 1. Swing Points (Fractal High/Low)

**Fungsi:** Mendeteksi puncak dan lembah penting di chart.

### Algoritma

```
Window: 2 x swing_length + 1 = 11 candle (default swing_length=5)

Swing High: High saat ini = Maximum dalam 11 candle
Swing Low:  Low saat ini = Minimum dalam 11 candle
```

### Visualisasi

```
         /\ <- Swing High (high = max 11 candle)
        /  \
       /    \
      /      \
     /        \/ <- Swing Low (low = min 11 candle)
    /
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `swing_high` | 1 / 0 | 1 jika swing high |
| `swing_low` | -1 / 0 | -1 jika swing low |
| `swing_high_level` | float | Harga di swing high |
| `swing_low_level` | float | Harga di swing low |
| `last_swing_high` | float | Swing high terakhir (forward fill) |
| `last_swing_low` | float | Swing low terakhir (forward fill) |

---

## 2. Fair Value Gap (FVG)

**Fungsi:** Mendeteksi **imbalance/gap** di harga — zona yang belum "diisi" oleh pasar.

### Algoritma

```
Bullish FVG:                    Bearish FVG:
Candle T-2: ████ high           Candle T-2: ████ low
                  |                           |
                  | GAP (celah)               | GAP (celah)
                  |                           |
Candle T+1: ████ low            Candle T+1: ████ high

Syarat Bullish: high[T-2] < low[T+1]
Syarat Bearish: low[T-2]  > high[T+1]
```

### Zona FVG

```
Bullish FVG Zone:
  Top    = low[T+1]    (batas atas gap)
  Bottom = high[T-2]   (batas bawah gap)
  Mid    = (top + bottom) / 2  (50% retracement)
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `fvg_signal` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| `fvg_top` | float | Batas atas gap |
| `fvg_bottom` | float | Batas bawah gap |
| `fvg_mid` | float | Titik tengah (target retracement) |

**Peran:** Zona entry ideal — harga cenderung **kembali mengisi gap** sebelum melanjutkan.

---

## 3. Order Block (OB)

**Fungsi:** Mendeteksi candle terakhir sebelum pergerakan besar — zona dimana institusi menaruh order.

### Algoritma

```
Bullish OB:
  1. Temukan swing low
  2. Lihat 10 candle ke belakang
  3. Cari candle bearish terakhir (close < open)
  4. Jika candle berikutnya close di atas high candle tersebut:
     -> Candle itu = Bullish Order Block

Bearish OB:
  1. Temukan swing high
  2. Lihat 10 candle ke belakang
  3. Cari candle bullish terakhir (close > open)
  4. Jika candle berikutnya close di bawah low candle tersebut:
     -> Candle itu = Bearish Order Block
```

### Visualisasi

```
Bullish OB:                     Bearish OB:
                                ████ <- Bullish candle terakhir
████ <- Bearish candle terakhir        sebelum jatuh
        sebelum naik           ═══════════════
═══════════════                     ||| turun
     ||| naik
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `ob` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| `ob_top` | float | Batas atas zona OB |
| `ob_bottom` | float | Batas bawah zona OB |
| `ob_mitigated` | bool | True jika OB sudah dikunjungi ulang |

**Peran:** Zona support/resistance berdasarkan aksi institusi besar.

---

## 4. Break of Structure (BOS)

**Fungsi:** Mendeteksi **kelanjutan tren** — harga menembus swing point searah tren.

### Algoritma

```python
# Tren sudah BULLISH, lalu:
if close > last_swing_high:
    bos = 1  # Bullish BOS — tren naik BERLANJUT

# Tren sudah BEARISH, lalu:
if close < last_swing_low:
    bos = -1  # Bearish BOS — tren turun BERLANJUT
```

### Visualisasi

```
Bullish BOS:
     SH1        SH2 (baru ditembus!)
    /    \      / close >>>
   /      \    /
  /        SL1           -> BOS! Tren naik lanjut

Bearish BOS:
  \        SH1
   \      /    \
    \    /      \ close <<<
     SL1        SL2 (baru ditembus!)  -> BOS! Tren turun lanjut
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `bos` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |

**Peran:** Konfirmasi bahwa **tren masih kuat** dan lanjut.

---

## 5. Change of Character (CHoCH)

**Fungsi:** Mendeteksi **pembalikan tren** — harga menembus swing point berlawanan tren.

### Algoritma

```python
# Tren sedang BEARISH, lalu:
if close > last_swing_high:
    choch = 1  # Bullish CHoCH — REVERSAL naik!

# Tren sedang BULLISH, lalu:
if close < last_swing_low:
    choch = -1  # Bearish CHoCH — REVERSAL turun!
```

### Visualisasi

```
Bearish CHoCH (tren naik -> balik turun):
     SH <- gagal naik
    /  \
   /    \
  /      close menembus SL >>> CHoCH! Reversal turun!
 SL

Bullish CHoCH (tren turun -> balik naik):
 SH
  \      close menembus SH >>> CHoCH! Reversal naik!
   \    /
    \  /
     SL <- gagal turun
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `choch` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| `market_structure` | 1 / -1 / 0 | Bullish / Bearish / Netral |

**Peran:** **Early warning** perubahan arah tren.

---

## 6. Liquidity Zones

**Fungsi:** Mendeteksi kumpulan stop loss (equal highs/lows) yang bisa "disapu" oleh institusi.

### Algoritma

```
1. Hitung rolling std & mean dari highs dan lows (window=20)
2. Coefficient of Variation = std / mean
3. Jika CV < 0.001 (0.1%):
   -> Harga sangat mirip = cluster likuiditas
   -> BSL (Buy Side Liquidity) = level high
   -> SSL (Sell Side Liquidity) = level low
4. Deteksi sweep:
   -> BSL sweep: High > BSL lalu close < BSL
   -> SSL sweep: Low < SSL lalu close > SSL
```

### Visualisasi

```
Buy Side Liquidity (BSL):        Sell Side Liquidity (SSL):
═══════ equal highs ═══════
████ ████ ████ ████              ████ ████ ████ ████
                                 ═══════ equal lows ═══════
^ Stop loss short sellers        ^ Stop loss long traders
^ Institusi sweep ke atas        ^ Institusi sweep ke bawah
```

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| `bsl_level` | float | Level buy side liquidity |
| `ssl_level` | float | Level sell side liquidity |
| `liquidity_sweep` | "BSL" / "SSL" / None | Sweep terdeteksi |

---

## Signal Generation

### ATR-Based Dynamic SL/TP (v3 Update)

Sebelum menghitung SL dan TP, sistem mengambil nilai ATR untuk kalkulasi dinamis:

```python
# Line 631-634
atr = latest["atr"]              # Dari Feature Engineering
min_sl_distance = 1.5 * atr      # Minimum jarak SL = 1.5 ATR
max_tp_distance = 4.0 * atr      # Maximum jarak TP = 4.0 ATR

# Fallback jika ATR tidak tersedia:
atr = current_close * 0.01       # 1% dari harga
```

### Kondisi Bullish Signal

```
IF (market_structure == BULLISH ATAU ada BOS/CHoCH bullish)
AND (ada FVG bullish ATAU Order Block bullish):

  Entry  = FVG bottom atau OB bottom

  SL (v3 - ATR-based, lebih protektif):
    swing_sl  = last_swing_low (jika ada & di bawah entry)
    atr_sl    = entry - 1.5 * ATR
    SL        = MIN(swing_sl, atr_sl)  <- pilih yang LEBIH JAUH

  TP (v3 - dibatasi realistis):
    risk = entry - SL
    tp   = entry + (risk * 2)          <- minimum 2:1 RR
    IF tp > entry + 4*ATR:
       tp = entry + 4*ATR              <- cap TP agar realistis
```

### Kondisi Bearish Signal

```
IF (market_structure == BEARISH ATAU ada BOS/CHoCH bearish)
AND (ada FVG bearish ATAU Order Block bearish):

  Entry  = FVG top atau OB top

  SL (v3 - ATR-based, lebih protektif):
    swing_sl  = last_swing_high (jika ada & di atas entry)
    atr_sl    = entry + 1.5 * ATR
    SL        = MAX(swing_sl, atr_sl)  <- pilih yang LEBIH JAUH

  TP (v3 - dibatasi realistis):
    risk = SL - entry
    tp   = entry - (risk * 2)          <- minimum 2:1 RR
    IF tp < entry - 4*ATR:
       tp = entry - 4*ATR              <- cap TP agar realistis
```

### Perbandingan SL/TP Lama vs Baru

```
┌────────────┬───────────────────────────┬──────────────────────────────┐
│ Komponen   │ Sebelum (v2)              │ Sesudah (v3)                 │
├────────────┼───────────────────────────┼──────────────────────────────┤
│ SL (BUY)   │ swing_low atau            │ MIN(swing_low, entry-1.5ATR) │
│            │ entry * 0.995 (bisa dekat)│ <- selalu cukup jauh         │
├────────────┼───────────────────────────┼──────────────────────────────┤
│ SL (SELL)  │ swing_high atau           │ MAX(swing_high, entry+1.5ATR)│
│            │ entry * 1.005 (bisa dekat)│ <- selalu cukup jauh         │
├────────────┼───────────────────────────┼──────────────────────────────┤
│ TP         │ risk * 2                  │ MIN(risk*2, 4*ATR)           │
│            │ (bisa sangat jauh)        │ <- dibatasi realistis        │
└────────────┴───────────────────────────┴──────────────────────────────┘
```

### Sistem Confidence

```
Base confidence:  55%
+ BOS/CHoCH:     +10%
+ FVG:           +10%
+ Order Block:   +10%
Maximum:          85%
```

### Output Signal

```python
SMCSignal:
  signal_type: "BUY" / "SELL"
  entry_price: float
  stop_loss: float        # ATR-based (min 1.5 ATR dari entry)
  take_profit: float      # 2:1 RR, capped di 4 ATR
  confidence: 0.55 - 0.85
  reason: "Bullish BOS + FVG + OB"
  risk_reward: float      # Minimum 2.0
```

---

## Konfigurasi

```python
SMCConfig:
  swing_length: 5          # Window untuk deteksi swing (11 bar total)
  fvg_min_gap_pips: 2.0    # Minimum ukuran FVG
  ob_lookback: 10          # Berapa jauh cari OB ke belakang
  bos_close_break: True    # Harus close (bukan wick) yang break
```

---

## Integrasi dalam Pipeline

```
Data OHLCV
    |
    v
smc.calculate_all(df)
    |--- calculate_fair_value_gaps()
    |--- calculate_swing_points()
    |--- calculate_order_blocks()     <- butuh swing points
    |--- calculate_structure_breaks() <- butuh swing points
    |--- calculate_liquidity_zones()
    |
    v
smc.generate_signal(df)
    |
    v
SMCSignal (entry, SL, TP, confidence)
    |
    v
Dikombinasikan dengan XGBoost + HMM
```
