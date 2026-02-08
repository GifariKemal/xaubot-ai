# Trade Logger — Pencatat Trade Otomatis

> **File:** `src/trade_logger.py`
> **Class:** `TradeLogger`
> **Storage:** PostgreSQL (primary) + CSV (*fallback*)

---

## Apa Itu *Trade Logger*?

*Trade Logger* mencatat **setiap trade, sinyal, dan kondisi pasar** secara otomatis ke database dan file CSV. Data ini digunakan untuk analisis performa, retraining ML model, dan debugging.

**Analogi:** *Trade Logger* seperti ***black box* di pesawat** — merekam semua yang terjadi untuk analisis setelah penerbangan (trading).

---

## Alur *Dual Storage*

```mermaid
flowchart TD
    A["Event Terjadi\n(trade / signal / snapshot)"] --> B[TradeLogger]
    B --> C{DB tersedia?}
    C -- Ya --> D["PostgreSQL\n(Primary)"]
    C -- Ya --> E["CSV\n(Backup)"]
    C -- Tidak --> E
    D --> F["trades table\nsignals table\nmarket_snapshots\nbot_status"]
    E --> G["data/trade_logs/\ntrades/ | signals/ | snapshots/\n(file bulanan YYYY_MM.csv)"]
    style A fill:#2d333b,stroke:#adbac7,color:#adbac7
    style B fill:#1f6feb,stroke:#58a6ff,color:#fff
    style C fill:#3d444d,stroke:#adbac7,color:#adbac7
    style D fill:#238636,stroke:#3fb950,color:#fff
    style E fill:#9e6a03,stroke:#d29922,color:#fff
    style F fill:#238636,stroke:#3fb950,color:#fff
    style G fill:#9e6a03,stroke:#d29922,color:#fff
```

**Prinsip *dual storage*:**

- **DB tersedia?** — Tulis ke PostgreSQL **DAN** CSV (double safety)
- **DB tidak tersedia?** — CSV saja (*graceful degradation*)
- CSV **selalu** ditulis sebagai *fallback*, tidak peduli status DB

---

## 3 Tipe Data yang Dicatat

### 1. Trade Record (Per Trade)

Setiap trade dibuka/ditutup dicatat lengkap:

| Kategori | Field |
|----------|-------|
| **Identitas** | ticket, symbol |
| **Trade** | direction, lot_size, entry_price, exit_price, SL, TP |
| **Hasil** | profit_usd, profit_pips, duration_seconds |
| **Waktu** | open_time, close_time |
| **Market** | regime, volatility, session, spread, ATR |
| **SMC** | *signal*, confidence, reason, FVG/OB/BOS/CHoCH flags |
| **ML** | *signal*, confidence |
| **Dynamic** | market_quality, market_score, threshold |
| **Exit** | exit_reason, exit_regime, exit_ml_signal |
| **Balance** | balance_before, balance_after, equity_at_entry |
| **Features** | JSON *snapshot* fitur saat entry & exit |

### 2. *Signal* Record (Per Sinyal)

Setiap sinyal yang dihasilkan (termasuk yang **tidak** dieksekusi):

```
timestamp, symbol, price
signal_type, signal_source, confidence
smc_*, ml_*
regime, session, volatility, market_score
trade_executed (bool)
execution_reason ("executed" / "below_threshold" / "max_positions" / ...)
```

### 3. Market *Snapshot* (Periodik)

*Snapshot* kondisi pasar secara berkala:

```
timestamp, symbol, price, OHLC
regime, volatility, session, ATR, spread
ml_signal, ml_confidence
smc_signal, smc_confidence
open_positions, floating_pnl
features (JSON)
```

---

## *Dual Storage*

```mermaid
flowchart TD
    EV["Event Terjadi<br/>(trade / signal / snapshot)"] --> PG["PostgreSQL (Primary)<br/>trades, signals,<br/>market_snapshots, bot_status<br/>Cepat, queryable, thread-safe pooling"]
    EV --> CSV["CSV (Fallback)<br/>data/trade_logs/<br/>trades/, signals/, snapshots/<br/>Selalu ditulis (backup)"]
```

- **DB tidak tersedia?** → CSV saja (*graceful degradation*)
- **DB tersedia?** → Tulis ke DB **DAN** CSV (double safety)

---

## Proses Log Trade

```mermaid
flowchart TD
    OPEN["Trade Dibuka"] --> LOG_OPEN["log_trade_open()<br/>ticket, entry_price, regime, smc, ml"]
    LOG_OPEN --> MEM["Simpan ke _pending_trades di memory"]
    LOG_OPEN --> DB_INS["INSERT ke database (trades table)"]
    MEM --> WAIT["... trading berjalan ..."]
    DB_INS --> WAIT
    WAIT --> CLOSE["Trade Ditutup"]
    CLOSE --> LOG_CLOSE["log_trade_close()<br/>ticket, exit_price, profit, exit_reason"]
    LOG_CLOSE --> FETCH["Ambil data pending dari memory"]
    LOG_CLOSE --> DUR["Hitung durasi: close - open"]
    FETCH --> UPD["UPDATE database<br/>(exit_price, profit, duration)"]
    DUR --> UPD
    UPD --> CSV["APPEND ke CSV<br/>(trades_YYYY_MM.csv)"]
```

Data *pending* disimpan dalam dictionary `_pending_trades[ticket]` selama trade masih terbuka. Ketika trade ditutup, data entry digabung dengan data exit menjadi satu `TradeRecord` lengkap sebelum ditulis ke CSV.

---

## Analisis Helper

| Method | Fungsi |
|--------|--------|
| `get_recent_trades(10)` | 10 trade terakhir |
| `get_win_rate(30)` | Win rate 30 hari |
| `get_smc_performance(30)` | Performa per pattern SMC |
| `get_trades_for_training(30)` | Data untuk ML retraining |
| `get_stats()` | Statistik logger |

Setiap helper method mencoba query dari PostgreSQL terlebih dahulu. Jika DB tidak tersedia, otomatis *fallback* ke pembacaan file CSV — konsisten dengan prinsip *graceful degradation*.

---

## *Thread Safety*

```python
self._lock = threading.Lock()

# Setiap operasi CSV dilindungi lock
with self._lock:
    # Write to CSV
```

Semua operasi tulis ke file CSV dilindungi oleh `threading.Lock()` untuk menjamin *thread safety*. Ini mencegah korupsi data ketika multiple thread mencoba menulis ke file yang sama secara bersamaan (misalnya log trade close dan log *signal* terjadi hampir bersamaan).

---

## File CSV (Terorganisir per Bulan)

```
data/trade_logs/
├── trades/
│   ├── trades_2025_01.csv
│   └── trades_2025_02.csv
├── signals/
│   ├── signals_2025_01.csv
│   └── signals_2025_02.csv
└── snapshots/
    ├── snapshots_2025_01.csv
    └── snapshots_2025_02.csv
```
