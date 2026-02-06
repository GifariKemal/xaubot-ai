# News Agent — Monitoring Berita Ekonomi

> **File:** `src/news_agent.py`
> **Class:** `NewsAgent`
> **Status:** Aktif tapi **TIDAK MEMBLOKIR** trading (monitoring only)

---

## Apa Itu News Agent?

News Agent memonitor **berita ekonomi high-impact** (NFP, FOMC, CPI) yang bisa menyebabkan volatilitas ekstrem di pasar gold. Awalnya dirancang untuk memblokir trading saat news, tapi setelah backtest menunjukkan bahwa blocking justru **kehilangan $178 profit**, sekarang hanya berfungsi sebagai **monitor dan logger**.

**Analogi:** News Agent seperti **stasiun cuaca** — melaporkan badai yang datang, tapi pilot (bot) tetap terbang karena pesawat (ML model) sudah cukup tangguh menangani turbulensi.

---

## Kenapa Tidak Blocking?

```
Hasil Backtest (29 trades):
  - Win rate tanpa filter: 64.9%
  - Win rate saat news:    62.1%  (selisih hanya 2.8%)
  - Profit yang hilang jika filter aktif: $178.15

Kesimpulan:
  -> ML model sudah cukup menangani volatilitas news
  -> Blocking justru kehilangan peluang profit
  -> Monitoring cukup, tidak perlu blocking
```

---

## Event yang Dipantau

### 3 Event High-Impact

| Event | Waktu (WIB) | Hari | Dampak ke Gold |
|-------|-------------|------|---------------|
| **NFP** (Non-Farm Payroll) | 20:30 | Jumat pertama bulan | Sangat tinggi |
| **FOMC** (Fed Decision) | 02:00 | ~8x per tahun | Sangat tinggi |
| **CPI** (Inflation) | 20:30 | Tgl 10-15 (Sel/Rab/Kam) | Tinggi |

### Deteksi Event

```python
# NFP: Jumat pertama bulan
if weekday == 4 and day <= 7:          # Friday, day 1-7
    if 19 <= hour <= 21:               # 19:00-21:00 WIB
        return "NFP (Non-Farm Payroll) - HIGH IMPACT"

# FOMC: Tanggal spesifik (hardcoded schedule)
fomc_dates = [
    (1,29), (3,19), (5,7), (6,18), (7,30),  # 2025
    (9,17), (11,5), (12,17),
    (1,29), (3,18), (5,6), (6,17), (7,29),  # 2026
]
if (month, day) in fomc_dates:
    if 1 <= hour <= 3:                 # 01:00-03:00 WIB
        return "FOMC Decision - HIGH IMPACT"

# CPI: Sekitar tanggal 10-15, hari kerja
if 10 <= day <= 15 and 19 <= hour <= 21:
    if weekday in [1, 2, 3]:           # Selasa-Kamis
        return "CPI (Inflation) - HIGH IMPACT"
```

---

## Buffer Times

| Parameter | Default | Aktif di Production |
|-----------|---------|-------------------|
| `news_buffer_minutes` | 30 menit | **0** (disabled) |
| `high_impact_buffer_minutes` | 60 menit | **0** (disabled) |

```python
# Inisialisasi di main_live.py
self.news_agent = create_news_agent(
    news_buffer_minutes=0,          # No blocking
    high_impact_buffer_minutes=0,   # No blocking
)
```

---

## Market Condition States

| Kondisi | Bisa Trade? | Lot Multiplier | Trigger |
|---------|------------|---------------|---------|
| `SAFE` | Ya | 1.0x | Tidak ada news |
| `CAUTION` | Ya | 0.5x | News medium-impact |
| `DANGER_NEWS` | Tidak* | 0.0x | High-impact news |
| `DANGER_SENTIMENT` | Tidak* | 0.5x | Sentimen sangat bearish |

*\*Di production, DANGER tetap diizinkan trading (monitoring only)*

---

## Analisis Sentimen

News Agent juga bisa menganalisis headline berita berdasarkan keyword:

### Keyword Bullish (untuk Gold)

```
Geopolitical: war, conflict, invasion, crisis, escalation
Economic:     rate cut, dovish, easing, recession, stimulus
Market:       safe haven, gold surge, gold rally, buy gold
```

### Keyword Bearish (untuk Gold)

```
Geopolitical: peace deal, ceasefire, de-escalation
Economic:     rate hike, hawkish, tightening, strong dollar
Market:       risk on, stocks rally, sell gold, gold crash
```

### Keyword Volatile

```
breaking, urgent, flash, sudden, unexpected, shock, crash, spike
```

### Scoring

```
Setiap keyword match:
  Bullish:  +0.3
  Bearish:  -0.3
  Volatile: -0.1 (penalty)

Score range: -1.0 (sangat bearish) sampai +1.0 (sangat bullish)
Confidence berdasarkan jumlah keyword yang match
```

---

## Method `should_trade()`

```python
def should_trade(headlines=None) -> (bool, str, float):
    """
    Returns:
      can_trade: bool      <- Apakah aman trading
      reason: str          <- Alasan
      lot_multiplier: float <- Pengali lot (0.0-1.0)
    """
    # 1. Cek economic calendar (MT5 + hardcoded events)
    # 2. Analisis sentimen (jika ada headlines)
    # 3. Tentukan kondisi pasar
    # 4. Return rekomendasi
```

---

## Integrasi di Main Loop

```python
# main_live.py Lines 485-496
# NEWS AGENT MONITORING (NO BLOCKING)

can_trade_news, news_reason, news_lot_mult = self.news_agent.should_trade()

# Hanya LOG, TIDAK block
if not can_trade_news and loop_count % 300 == 0:  # Setiap 5 menit
    logger.info(f"News Agent: HIGH IMPACT NEWS - {news_reason} (trading allowed)")

# Catatan:
# - news_lot_mult dihitung tapi TIDAK diterapkan
# - Trading tetap berjalan normal
# - Informasi digunakan untuk logging dan analisis
```

---

## Sumber Data

| Sumber | Status | Keterangan |
|--------|--------|-----------|
| **MT5 Calendar** | Aktif | Cek economic calendar dari terminal |
| **Hardcoded Events** | Aktif (Fallback) | NFP, FOMC, CPI schedule |
| **NewsAPI** | Tersedia, tidak digunakan | External API (butuh API key) |
| **ForexFactory** | Tersedia, tidak diimplementasi | Placeholder untuk scraping |

---

## Konfigurasi

```python
NewsAgent(
    news_buffer_minutes=30,           # Buffer news biasa (disabled: 0)
    high_impact_buffer_minutes=60,    # Buffer high-impact (disabled: 0)
    enable_mt5_calendar=True,         # Cek MT5 calendar
    enable_sentiment=True,            # Analisis sentimen
)

# Cache
_cache_duration = 15 menit            # Cache hasil calendar check
```

---

## Contoh Output Log

```
[14:30] News Agent: HIGH IMPACT NEWS - NFP (Non-Farm Payroll) (trading allowed)
[14:35] News Agent: Market condition SAFE - no upcoming events
[20:25] News Agent: HIGH IMPACT NEWS - CPI (Inflation) (trading allowed)
```

**Catatan:** Meskipun terdeteksi "HIGH IMPACT NEWS", bot tetap trading. Log ini berguna untuk analisis post-trade — apakah trade yang terjadi saat news perform baik atau buruk.
