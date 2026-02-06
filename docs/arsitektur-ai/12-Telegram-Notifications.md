# Telegram Notifications — Sistem Notifikasi Real-Time

> **File:** `src/telegram_notifier.py`
> **Class:** `TelegramNotifier`
> **API:** Telegram Bot API (async via aiohttp)

---

## Apa Itu Telegram Notifications?

Telegram Notifications mengirimkan **laporan real-time** ke grup Telegram setiap kali terjadi event penting — trade dibuka/ditutup, laporan harian, alert darurat, dan status sistem.

**Analogi:** Telegram Notifications seperti **dashboard pilot di cockpit** — menampilkan semua informasi penting secara real-time tanpa harus melihat layar trading.

---

## Konfigurasi

```
Bot Token:  Dari environment variable TELEGRAM_BOT_TOKEN
Chat ID:    Dari environment variable TELEGRAM_CHAT_ID
Format:     HTML (parse_mode)
Transport:  Async HTTP POST via aiohttp
Timezone:   WIB (Asia/Jakarta)
```

```python
# Inisialisasi
from dotenv import load_dotenv
load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
enabled = bool(bot_token and chat_id)  # Auto-disable jika tidak dikonfigurasi
```

---

## 11 Tipe Notifikasi

| # | Tipe | Trigger | Frekuensi |
|---|------|---------|-----------|
| 1 | Trade Open | Order berhasil dieksekusi | Per trade |
| 2 | Trade Close | Posisi ditutup | Per trade |
| 3 | Market Update | Timer 30 menit | Setiap 30 menit |
| 4 | Hourly Analysis | Timer 1 jam | Setiap 1 jam |
| 5 | Daily Summary | Pergantian hari | 1x per hari |
| 6 | Startup | Bot dinyalakan | 1x per sesi |
| 7 | Shutdown | Bot dimatikan | 1x per sesi |
| 8 | News Alert | Event ekonomi terdeteksi | Per event |
| 9 | Critical Limit | Daily/total loss limit | Per event |
| 10 | Emergency Close | Flash crash / darurat | Per event |
| 11 | System Status | Status berkala | Per request |

---

## Format Pesan

### 1. Trade Open

```
🟢 LONG #123456
├ XAUUSD
├ Entry: 4950.00
├ Lot: 0.02
├ SL: 4937.00 (-$13)
├ TP: 4976.00 (+$26)
├ R:R: 1:2.0
├ AI: 75% | medium_volatility
└ SMC Bullish BOS + FVG
⏰ 14:35 WIB
```

| Elemen | Arti |
|--------|------|
| 🟢/🔴 | BUY (hijau) / SELL (merah) |
| LONG/SHORT | Arah posisi |
| #123456 | Ticket ID dari broker |
| R:R | Risk to Reward ratio |
| AI: 75% | ML confidence |
| medium_volatility | HMM regime |

---

### 2. Trade Close

```
✅ WIN #123456
├ XAUUSD BUY
├ Entry: 4950.00
├ Exit: 4965.00
├ Lot: 0.02
├ P/L: +$30.00 (+0.49%)
├ Pips: +150.0
├ Duration: 2m
├ Bal Before: $6130.00
└ Bal After: $6160.00
⏰ 14:40 WIB
```

| Emoji | Arti |
|-------|------|
| ✅ | WIN (profit) |
| ❌ | LOSS (rugi) |
| ➖ | BREAKEVEN (impas) |

---

### 3. Market Update (Setiap 30 Menit)

```
📊 XAUUSD $4965.00
├ 🟢 BUY 75%
├ UPTREND
├ medium_volatility
├ London-NY Overlap
└ ✅
⏰ 14:45
```

---

### 4. Hourly Analysis (Setiap 1 Jam)

```
📊 HOURLY 14:00 WIB

Account
├ Bal: $5,094.68
├ Eq: $5,120.50
├ Float: +$25.82
└ Day: +$150.00 (12 trades)

Positions (2)
├ #123456 BUY: +$30.00 M:+45
└ #123457 SELL: -$15.00 M:-20

Market
├ XAUUSD $4,965.00
├ London-NY Overlap
└ medium_volatility | high

AI Signal
├ BUY 75% / thresh 70%
└ Quality: EXCELLENT (score:85) → READY

Risk NORMAL
└ Daily Loss: $0.00 / $148.34

✅ News: SAFE
```

---

### 5. Daily Summary

```
🎉 DAILY REPORT 2025-02-06

Result
├ P/L: +$150.00 (+3.03%)
├ Gross Win: +$500.00
├ Gross Loss: -$350.00
├ Bal Start: $4,944.68
└ Bal End: $5,094.68

Stats
├ Total: 12 trades
├ Wins: 8 | Losses: 4
├ Win Rate: 66.7%
├ Profit Factor: 1.43
└ Avg/Trade: $12.50

Recent Trades
├ ✅ BUY: +$30.00
├ ❌ SELL: -$25.00
├ ✅ BUY: +$45.00
├ ➖ SELL: $0.00
└ ✅ BUY: +$100.00
```

| Emoji Hari | Arti |
|-----------|------|
| 🎉 | Hari profit |
| 📉 | Hari loss |
| ➖ | Hari breakeven |

---

### 6. Startup

```
🚀 BOT STARTED

Config
├ Symbol: XAUUSD
├ Mode: small
├ Capital: $5,000.00
├ Balance: $4,944.68
└ ML: Loaded (37 features)

Risk Settings
├ Risk/Trade: 1%
├ Max Daily Loss: 5%
├ Max Total Loss: 10%
└ SL: Smart (ATR-based)

✅ News: SAFE
⏰ 2025-02-06 08:15 WIB
```

---

### 7. Shutdown

```
🔴 BOT STOPPED

Session Summary
├ Balance: $5,094.68
├ Total Trades: 12
├ ✅ P/L: +$150.00
└ Uptime: 8.5h

⏰ 2025-02-06 16:45 WIB
```

---

### 8. News Alert

```
🚨 NEWS DANGER_NEWS
├ NFP (Non-Farm Payroll) - HIGH IMPACT
├ High volatility expected during release
└ Buffer: 60m
⏰ 20:25
```

| Emoji | Kondisi |
|-------|---------|
| 🚨 | DANGER_NEWS |
| ⚠️ | CAUTION / DANGER_SENTIMENT |
| ✅ | SAFE |

---

### 9. Critical Limit Alert

```
🚨 DAILY LOSS LIMIT REACHED 🚨

Daily Loss: $250.00
Limit: $250.00 (5%)

⛔ TRADING STOPPED FOR TODAY
Will resume tomorrow automatically.
```

---

### 10. Emergency Close

```
🚨 EMERGENCY CLOSE COMPLETE

Closed 3 positions due to flash crash detection
Total P/L: -$45.00
```

---

### 11. Alert (Berbagai Tipe)

| Alert Type | Emoji | Contoh |
|-----------|-------|--------|
| flash_crash | 🚨 | "Flash crash detected on XAUUSD" |
| high_volatility | ⚡ | "Volatility spike detected" |
| connection_error | 📡 | "MT5 connection lost" |
| model_retrain | 🔄 | "ML model retrained successfully" |
| market_close | 🔔 | "Market closing in 30 minutes" |
| low_balance | 💰 | "Account balance below threshold" |

---

## 3 Metode Pengiriman

| Metode | Endpoint | Kegunaan |
|--------|----------|---------|
| `send_message()` | `/sendMessage` | Teks biasa (semua notifikasi) |
| `send_photo()` | `/sendPhoto` | Chart/grafik (daily report) |
| `send_document()` | `/sendDocument` | File PDF (laporan detail) |

---

## Error Handling

```
Strategi: GRACEFUL DEGRADATION

1. Try-Except di setiap send method
   -> Gagal kirim? Log warning, lanjut trading

2. HTTP status check
   -> Status != 200? Log error, return False

3. Emergency close
   -> Telegram gagal? TETAP tutup posisi
   -> Trading > notifikasi dalam prioritas

4. Disabled mode
   -> Token/ChatID kosong? Auto-disable, return True
   -> Bot tetap berjalan tanpa notifikasi
```

```python
# Contoh: Emergency close TIDAK boleh gagal karena Telegram
try:
    await telegram.send_message("Emergency close...")
except:
    pass  # Jangan biarkan Telegram failure menghentikan close
```

---

## Rate Limiting

| Notifikasi | Interval |
|-----------|----------|
| Trade Open/Close | Langsung (per event) |
| Market Update | 30 menit |
| Hourly Analysis | 1 jam |
| Daily Summary | 1x per hari |
| Startup/Shutdown | 1x per sesi |
| Min message interval | 1 detik (variable) |

---

## Kapan Notifikasi Dikirim di Main Loop

```
Main Loop (setiap 1 detik)
    |
    |-- Cek new day? ------> Daily Summary + Reset
    |
    |-- Cek hourly timer? -> Hourly Analysis (setiap 1 jam)
    |
    |-- Cek 30min timer? --> Market Update (setiap 30 menit)
    |
    |-- Trade executed? ---> Trade Open notification
    |
    |-- Position closed? --> Trade Close notification
    |
    |-- Limit hit? --------> Critical Limit Alert
    |
    |-- Flash crash? ------> Emergency Close Alert
    |
    |-- (startup) ---------> Startup message
    |
    |-- (shutdown) --------> Shutdown message
```

---

## Formatting HTML

Semua pesan menggunakan HTML parse mode:

```html
<b>Bold</b>           -> Label penting
<code>Monospace</code> -> Angka, harga, nilai
<i>Italic</i>          -> Info tambahan, alasan signal
```

Tree structure menggunakan box-drawing characters:

```
├  -> Item tengah
└  -> Item terakhir
```
