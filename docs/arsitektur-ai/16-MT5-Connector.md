# MT5 Connector — Jembatan ke MetaTrader 5

> **File:** `src/mt5_connector.py`
> **Class:** `MT5Connector`, `MT5SimulationConnector`
> **Library:** MetaTrader5 (Python API)

---

## Apa Itu MT5 Connector?

MT5 Connector adalah **jembatan komunikasi** antara bot AI dan terminal MetaTrader 5. Semua interaksi dengan broker — ambil data harga, kirim order, cek posisi — dilakukan melalui modul ini.

**Analogi:** MT5 Connector seperti **penerjemah di bandara** — menerjemahkan perintah bot (Python) ke bahasa yang dipahami broker (MT5 API), dan sebaliknya.

---

## Fungsi Utama

| Method | Fungsi | Return |
|--------|--------|--------|
| `connect()` | Koneksi ke MT5 terminal | `bool` |
| `disconnect()` | Putus koneksi | - |
| `reconnect()` | Reconnect otomatis | `bool` |
| `ensure_connected()` | Cek & auto-reconnect | `bool` |
| `get_market_data()` | Ambil data OHLCV | `pl.DataFrame` |
| `get_tick()` | Ambil harga real-time | `TickData` |
| `send_order()` | Kirim order BUY/SELL | `OrderResult` |
| `close_position()` | Tutup posisi | `OrderResult` |
| `get_open_positions()` | Cek posisi terbuka | `pl.DataFrame` |
| `get_symbol_info()` | Info simbol (spread, dll) | `Dict` |

---

## Koneksi & Auto-Reconnect

```
connect(max_retries=3)
    |
    v
Shutdown koneksi lama (jika ada)
    |
    v
mt5.initialize(login, password, server)
    |
    v
Tunggu 2 detik (stabilisasi terminal)
    |
    v
Verifikasi: terminal_info() != None?
    |
    ├── Ya → Cek terminal.connected?
    │         ├── Ya → ✅ Connected!
    │         └── Tidak → Tunggu 3 detik → Retry
    │
    └── Tidak → Exponential backoff (2s, 4s, 8s) → Retry
```

### Auto-Reconnect

```python
ensure_connected():
    """
    Dipanggil sebelum setiap operasi penting.

    1. Cek flag _connected
    2. Coba mt5.account_info()
    3. Gagal? → reconnect()
    4. Max 5 attempts, lalu cooldown 60 detik
    """
```

---

## Data Fetching (Polars Native)

```python
get_market_data(symbol="XAUUSD", timeframe="M15", count=200)
```

**Proses:**

```
MT5 Terminal
    |
    v
mt5.copy_rates_from_pos() → numpy structured array
    |
    v
LANGSUNG ke Polars DataFrame (TANPA Pandas)
    |
    v
Cast types:
├── time: Unix timestamp → Datetime
├── open/high/low/close: Float64
├── tick_volume → volume (Int64)
└── spread, real_volume: Int64
    |
    v
Return pl.DataFrame
```

**Kolom output:**

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `time` | Datetime | Waktu candle |
| `open` | Float64 | Harga buka |
| `high` | Float64 | Harga tertinggi |
| `low` | Float64 | Harga terendah |
| `close` | Float64 | Harga tutup |
| `volume` | Int64 | Tick volume |
| `spread` | Int64 | Spread |
| `real_volume` | Int64 | Real volume |

---

## Order Execution

```python
send_order(
    symbol="XAUUSD",
    order_type="BUY",     # atau "SELL"
    volume=0.01,          # Lot size
    sl=4937.00,           # Stop Loss
    tp=4976.00,           # Take Profit
    deviation=20,         # Max slippage (points)
    magic=123456,         # Bot ID
    comment="AI Bot",
    max_retries=3,
)
```

**Retry Logic:**

```
Kirim order
    |
    ├── RETCODE 10009 (DONE) → ✅ Success
    |
    ├── RETCODE 10013-10016 (INVALID) → ❌ Non-retryable
    |
    ├── RETCODE 10027 (TRADE DISABLED) → ❌ Raise error
    |
    └── RETCODE lain (requote/reject) → 🔄 Retry (max 3x)
```

---

## Timeframe Mapping

| String | MT5 Constant | Penggunaan |
|--------|-------------|------------|
| `M1` | TIMEFRAME_M1 | 1 menit |
| `M5` | TIMEFRAME_M5 | 5 menit |
| `M15` | TIMEFRAME_M15 | **Utama** (execution) |
| `M30` | TIMEFRAME_M30 | 30 menit |
| `H1` | TIMEFRAME_H1 | 1 jam |
| `H4` | TIMEFRAME_H4 | Trend analysis |
| `D1` | TIMEFRAME_D1 | 1 hari |

---

## Error Codes

| Code | Nama | Aksi |
|------|------|------|
| 10009 | DONE | Order berhasil |
| 10004 | REQUOTE | Retry |
| 10006 | REJECT | Retry |
| 10013 | INVALID | Stop, order salah |
| 10014 | INVALID_VOLUME | Stop, lot salah |
| 10015 | INVALID_PRICE | Stop, harga salah |
| 10016 | INVALID_STOPS | Stop, SL/TP salah |
| 10027 | TRADE_DISABLED | AutoTrading off |
| -10003 | NO_CONNECTION | Reconnect |
| -10004 | NO_IPC | Reconnect |

---

## Simulation Mode

```python
class MT5SimulationConnector(MT5Connector):
    """
    Untuk testing tanpa MT5 terminal.

    - connect() selalu berhasil
    - get_market_data() generate data sintetis (random walk)
    - Base price XAUUSD: $2000
    """
```

---

## Konfigurasi Koneksi

```python
MT5Connector(
    login=12345678,                    # Dari .env MT5_LOGIN
    password="password123",            # Dari .env MT5_PASSWORD
    server="BrokerServer-Live",        # Dari .env MT5_SERVER
    path="C:/Program Files/MT5/...",   # Dari .env MT5_PATH (opsional)
    timeout=60000,                     # 60 detik timeout
)
```
