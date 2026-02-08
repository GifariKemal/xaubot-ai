# *Database Module* — *PostgreSQL* Integration

> **File:** `src/db/connection.py`, `src/db/repository.py`
> **Database:** *PostgreSQL*
> **Library:** psycopg2 (*connection pooling*)

---

## Apa Itu *Database Module*?

*Database Module* menyediakan **penyimpanan persisten** untuk semua data trading — trade history, training log, sinyal, snapshot pasar, dan status bot. Menggunakan *PostgreSQL* dengan *connection pooling* untuk performa tinggi.

**Analogi:** *Database Module* seperti **arsip perpustakaan** — menyimpan semua catatan trading secara terorganisir, bisa dicari kapan saja, dan tidak hilang meski bot di-restart.

---

## Arsitektur

```mermaid
graph TD
    TL[TradeLogger] -->|write| TR[TradeRepository]
    TL -->|write| SigR[SignalRepository]
    TL -->|write| MSR[MarketSnapshotRepository]
    AT[AutoTrainer] -->|write| TrR[TrainingRepository]
    ML[main_live.py] -->|write| BSR[BotStatusRepository]
    ML -->|write| DSR[DailySummaryRepository]
    DASH[Dashboard] -.->|read| TR
    DASH -.->|read| SigR
    DASH -.->|read| MSR
    DASH -.->|read| TrR
    DASH -.->|read| BSR
    DASH -.->|read| DSR

    TR --> DC[DatabaseConnection<br/><i>Singleton</i>]
    SigR --> DC
    MSR --> DC
    TrR --> DC
    BSR --> DC
    DSR --> DC

    DC --> POOL[ThreadedConnectionPool<br/>1 – 10 koneksi]
    POOL --> PG[(PostgreSQL Server)]

    style DC fill:#2d6a4f,stroke:#1b4332,color:#fff
    style PG fill:#1b4332,stroke:#081c15,color:#fff
    style POOL fill:#40916c,stroke:#2d6a4f,color:#fff
```

---

## Connection (*Singleton* + Pooling)

```python
class DatabaseConnection:
    """
    Thread-safe singleton dengan connection pooling.

    - Hanya 1 instance (singleton pattern)
    - Pool: 1-10 koneksi (ThreadedConnectionPool)
    - Auto-reconnect jika putus
    - Context manager support
    """
```

`DatabaseConnection` menerapkan pola *singleton* yang *thread-safe* — hanya satu instance yang pernah dibuat selama proses berjalan. Akses ke database dilakukan melalui *context manager* (`with db.get_cursor() as cur`) sehingga koneksi selalu dikembalikan ke pool setelah selesai.

### Konfigurasi

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=trading_bot
DB_PASSWORD=trading_bot_2026
```

### Penggunaan

```python
from src.db import get_db, init_db

# Initialize
if init_db():
    db = get_db()

    # Query dengan context manager
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM trades WHERE profit_usd > 0")
        rows = cur.fetchall()

    # Simple execute
    result = db.execute("SELECT count(*) FROM trades", fetch=True)
```

---

## 6 *Repository*

Setiap *repository* bertanggung jawab atas satu tabel dan menyediakan method khusus untuk operasi CRUD.

### 1. TradeRepository

| Method | Fungsi |
|--------|--------|
| `insert_trade()` | Insert trade baru (saat open) |
| `update_trade_close()` | Update exit data (saat close) |
| `get_trade_by_ticket()` | Cari trade per ticket |
| `get_open_trades()` | Trade yang belum ditutup |
| `get_recent_trades(100)` | 100 trade terakhir |
| `get_trades_for_training(30)` | Trade 30 hari untuk ML |
| `get_daily_stats(date)` | Statistik per hari |
| `get_session_stats("London", 30)` | Statistik per sesi |
| `get_smc_pattern_stats(30)` | Performa per pola SMC |

### 2. TrainingRepository

| Method | Fungsi |
|--------|--------|
| `insert_training_run()` | Catat mulai training |
| `update_training_complete()` | Update hasil training |
| `mark_rollback()` | Tandai model di-rollback |
| `get_latest_successful()` | Training sukses terakhir |
| `get_training_history(20)` | 20 training terakhir |

### 3. SignalRepository

| Method | Fungsi |
|--------|--------|
| `insert_signal()` | Catat sinyal yang dihasilkan |
| `mark_executed()` | Tandai sinyal yang dieksekusi |
| `get_recent_signals(100)` | 100 sinyal terakhir |
| `get_signal_stats(24)` | Statistik 24 jam |

### 4. MarketSnapshotRepository

| Method | Fungsi |
|--------|--------|
| `insert_snapshot()` | Simpan snapshot pasar |
| `get_recent_snapshots(60)` | Snapshot 60 menit terakhir |

### 5. BotStatusRepository

| Method | Fungsi |
|--------|--------|
| `insert_status()` | Catat status bot |
| `get_latest_status()` | Status terbaru |

### 6. DailySummaryRepository

| Method | Fungsi |
|--------|--------|
| `upsert_summary()` | Insert/update ringkasan harian |
| `get_summary(date)` | Ringkasan per tanggal |
| `get_recent_summaries(30)` | 30 hari terakhir |

---

## Tabel Database

### Entity-Relationship Diagram

```mermaid
erDiagram
    trades {
        bigint ticket PK
        varchar symbol
        varchar direction
        float entry_price
        float exit_price
        float stop_loss
        float take_profit
        float lot_size
        float profit_usd
        float profit_pips
        timestamp opened_at
        timestamp closed_at
        int duration_seconds
        varchar entry_regime
        float entry_volatility
        varchar entry_session
        varchar smc_signal
        float smc_confidence
        text smc_reason
        bool smc_fvg_detected
        bool smc_ob_detected
        bool smc_bos_detected
        bool smc_choch_detected
        varchar ml_signal
        float ml_confidence
        varchar market_quality
        float market_score
        float dynamic_threshold
        varchar exit_reason
        varchar exit_regime
        varchar exit_ml_signal
        float balance_before
        float balance_after
        float equity_at_entry
        json features_entry
        json features_exit
        varchar bot_version
        varchar trade_mode
    }

    training_runs {
        serial id PK
        varchar training_type
        int bars_used
        int num_boost_rounds
        bool hmm_trained
        int hmm_n_regimes
        bool xgb_trained
        float train_auc
        float test_auc
        float train_accuracy
        float test_accuracy
        varchar model_path
        varchar backup_path
        bool success
        text error_message
        timestamp started_at
        timestamp completed_at
        int duration_seconds
        bool rolled_back
        text rollback_reason
        timestamp rollback_at
    }

    signals {
        serial id PK
        timestamp signal_time
        varchar symbol
        float price
        varchar signal_type
        varchar signal_source
        float combined_confidence
        varchar regime
        varchar session
        float volatility
        float market_score
        bool executed
        text execution_reason
        bigint trade_ticket FK
    }

    market_snapshots {
        serial id PK
        timestamp snapshot_time
        varchar symbol
        float price
        float open
        float high
        float low
        float close
        varchar regime
        float volatility
        varchar session
        float atr
        float spread
        varchar ml_signal
        float ml_confidence
        varchar smc_signal
        float smc_confidence
        int open_positions
        float floating_pnl
        json features
    }

    bot_status {
        serial id PK
        timestamp status_time
        bool is_running
        varchar status
        int loop_count
        float avg_execution_ms
        int uptime_seconds
        float balance
        float equity
        float margin_used
        int open_positions
        float floating_pnl
        float daily_pnl
        varchar risk_mode
        varchar current_session
        bool is_golden_time
    }

    daily_summaries {
        date summary_date PK
        int total_trades
        int winning_trades
        int losing_trades
        int breakeven_trades
        float gross_profit
        float gross_loss
        float net_profit
        float start_balance
        float end_balance
        float win_rate
        float profit_factor
        float avg_win
        float avg_loss
        int sydney_trades
        int tokyo_trades
        int london_trades
        int ny_trades
        int golden_trades
        int fvg_trades
        int fvg_wins
        int ob_trades
        int ob_wins
    }

    trades ||--o{ signals : "trade_ticket"
    daily_summaries ||--o{ trades : "summary_date covers opened_at"
```

### trades

```sql
├── ticket, symbol, direction
├── entry_price, exit_price, stop_loss, take_profit
├── lot_size, profit_usd, profit_pips
├── opened_at, closed_at, duration_seconds
├── entry_regime, entry_volatility, entry_session
├── smc_signal, smc_confidence, smc_reason
├── smc_fvg_detected, smc_ob_detected, smc_bos_detected, smc_choch_detected
├── ml_signal, ml_confidence
├── market_quality, market_score, dynamic_threshold
├── exit_reason, exit_regime, exit_ml_signal
├── balance_before, balance_after, equity_at_entry
├── features_entry (JSON), features_exit (JSON)
└── bot_version, trade_mode
```

### training_runs

```sql
├── training_type, bars_used, num_boost_rounds
├── hmm_trained, hmm_n_regimes
├── xgb_trained, train_auc, test_auc
├── train_accuracy, test_accuracy
├── model_path, backup_path
├── success, error_message
├── started_at, completed_at, duration_seconds
└── rolled_back, rollback_reason, rollback_at
```

### signals

```sql
├── signal_time, symbol, price
├── signal_type, signal_source, combined_confidence
├── smc_*, ml_*
├── regime, session, volatility, market_score
└── executed, execution_reason, trade_ticket
```

### market_snapshots

```sql
├── snapshot_time, symbol, price, OHLC
├── regime, volatility, session, ATR, spread
├── ml_signal, ml_confidence, smc_signal, smc_confidence
└── open_positions, floating_pnl, features (JSON)
```

### bot_status

```sql
├── status_time, is_running, status
├── loop_count, avg_execution_ms, uptime_seconds
├── balance, equity, margin_used
├── open_positions, floating_pnl, daily_pnl
└── risk_mode, current_session, is_golden_time
```

### daily_summaries

```sql
├── summary_date
├── total/winning/losing/breakeven_trades
├── gross_profit, gross_loss, net_profit
├── start_balance, end_balance
├── win_rate, profit_factor, avg win/loss
├── trades per session (sydney/tokyo/london/ny/golden)
└── SMC pattern stats (fvg/ob trades & wins)
```

---

## *Graceful Degradation*

```
PostgreSQL tersedia?
├── Ya → Gunakan DB + CSV backup
└── Tidak → CSV saja (semua tetap berjalan)

Bot TIDAK pernah crash karena database.
```

*Graceful degradation* memastikan bot tetap beroperasi penuh meskipun *PostgreSQL* tidak tersedia. Semua operasi database dibungkus dengan `try/except` — jika koneksi gagal, data ditulis ke CSV sebagai fallback. Saat database kembali online, bot otomatis menggunakan koneksi pool kembali tanpa restart.
