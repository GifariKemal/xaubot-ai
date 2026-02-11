# CLAUDE.md — XAUBot AI

## Project Overview

XAUBot AI is an automated XAUUSD (Gold) trading bot that combines Machine Learning (XGBoost), Smart Money Concepts (SMC), and Hidden Markov Model (HMM) regime detection. It runs on MetaTrader 5 via an async Python loop, executing trades on M15 candles.

## Directory Structure

```
.
├── main_live.py              # Main async trading orchestrator
├── train_models.py           # Model training script
├── Dockerfile                # Docker image (must be at root)
├── docker-compose.yml        # Docker orchestration (must be at root)
├── .dockerignore             # Docker build exclusions (must be at root)
├── src/                      # Core modules
│   ├── config.py             # Trading configuration & capital modes
│   ├── mt5_connector.py      # MetaTrader 5 connection layer
│   ├── smc_polars.py         # Smart Money Concepts (Polars-based)
│   ├── ml_model.py           # XGBoost trading model
│   ├── feature_eng.py        # Feature engineering (37 features)
│   ├── regime_detector.py    # HMM market regime detection
│   ├── risk_engine.py        # Risk calculations & validation
│   ├── smart_risk_manager.py # Dynamic risk management
│   ├── session_filter.py     # Trading session filter (Sydney/London/NY)
│   ├── position_manager.py   # Open position management
│   ├── dynamic_confidence.py # Adaptive confidence thresholds
│   ├── auto_trainer.py       # Auto-retraining pipeline
│   ├── news_agent.py         # Economic news filtering
│   ├── telegram_notifier.py  # Telegram alerts
│   ├── trade_logger.py       # Trade logging to DB
│   ├── utils.py              # Utility functions
│   └── db/                   # Database schemas
├── backtests/                # Backtesting scripts
│   ├── backtest_live_sync.py # Main backtest (synced with live logic)
│   └── archive/              # Old backtest versions
├── scripts/                  # Utility scripts
│   ├── check_market.py       # Quick SMC market analysis
│   ├── check_positions.py    # View open positions
│   ├── check_status.py       # Account status check
│   ├── close_positions.py    # Close all positions
│   ├── modify_tp.py          # Modify take-profit levels
│   └── get_trade_history.py  # Pull trade history from MT5
├── tests/                    # Test scripts
│   ├── test_modules.py       # Module integration tests
│   ├── test_mt5_connection.py# MT5 connection test
│   └── test_risk_settings.py # Risk settings test
├── models/                   # Trained models (.pkl)
├── data/                     # Market data & trade logs
├── docs/                     # Documentation
│   ├── arsitektur-ai/        # Architecture docs (23 components)
│   └── research/             # Research & analysis files
├── web-dashboard/            # Next.js monitoring dashboard
├── docker/                   # Docker configuration
│   ├── .env.docker.example   # Docker environment template
│   ├── requirements-docker.txt # Docker-specific Python deps
│   ├── init-db/              # Database init scripts
│   ├── scripts/              # Docker helper scripts (.bat/.sh)
│   └── docs/                 # Docker documentation
├── archive/                  # Deprecated files (gitignored)
└── logs/                     # Runtime logs
```

## Key Commands

```bash
# Run the live trading bot
python main_live.py

# Train/retrain ML models
python train_models.py

# Run backtest with threshold tuning
python backtests/backtest_live_sync.py --tune

# Run backtest with specific threshold
python backtests/backtest_live_sync.py --threshold 0.50 --save

# Run module tests
python tests/test_modules.py

# Check market status
python scripts/check_market.py
```

## Architecture

The bot runs an **async candle-based loop** on M15 timeframe:

1. **Data Fetch** — Pull OHLCV from MT5, convert to Polars DataFrame
2. **Feature Engineering** — Calculate 37 technical features (RSI, ATR, MACD, Bollinger, etc.)
3. **SMC Analysis** — Detect Order Blocks, Fair Value Gaps, BOS, CHoCH
4. **Regime Detection** — HMM classifies market as trending/ranging/volatile
5. **ML Prediction** — XGBoost outputs BUY/SELL/HOLD with confidence score
6. **Entry Filtering** — 11 entry filters must pass (session, regime, spread, cooldown, etc.)
7. **Risk Sizing** — ATR-based SL, dynamic position sizing, Kelly criterion
8. **Trade Execution** — Send order to MT5 with broker-level SL/TP
9. **Position Management** — 10 exit conditions (trailing SL, time exit, regime change, etc.)
10. **Logging** — Trade logged to PostgreSQL + Telegram notification

## Tech Stack

- **Python 3.11+** — Main runtime
- **Polars** — Data engine (not Pandas)
- **XGBoost** — ML model for signal prediction
- **hmmlearn** — Hidden Markov Model for regime detection
- **MetaTrader5** — Broker connection
- **asyncio + aiohttp** — Async execution & HTTP
- **loguru** — Structured logging
- **PostgreSQL** — Trade database
- **Next.js** — Web dashboard (optional)

## Configuration

All secrets in `.env`:
- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH` — Broker credentials
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — Notifications
- `CAPITAL` — Trading capital
- `SYMBOL` — Default: XAUUSD

Capital modes auto-configure risk parameters:
- **MICRO** (<$500): 2% risk/trade
- **SMALL** ($500-$10k): 1.5% risk/trade
- **MEDIUM** ($10k-$100k): 0.5% risk/trade
- **LARGE** (>$100k): 0.25% risk/trade

## Important Notes

- All data processing uses **Polars**, not Pandas
- The bot targets **< 50ms per loop** iteration
- Models are stored as `.pkl` files in `models/`
- Backtest logic is **synced with live** (`backtest_live_sync.py` mirrors `main_live.py`)
- Scripts in `scripts/` and `tests/` include `sys.path` fix so they work from any directory

---

## Versioning System

### **Semantic Versioning (SemVer)**

XAUBot AI uses **Semantic Versioning 2.0.0**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes, breaking changes (e.g., 1.0.0 → 2.0.0)
- **MINOR**: New features, backward compatible (e.g., 0.1.0 → 0.2.0)
- **PATCH**: Bug fixes, backward compatible (e.g., 0.1.0 → 0.1.1)

### **Version Files**

1. **`VERSION`** - Single source of truth (base version)
2. **`CHANGELOG.md`** - Detailed change history (Keep a Changelog format)
3. **`src/version.py`** - Centralized version manager

### **Auto-Versioning**

Version is **automatically calculated** based on enabled features:

```python
# Base version from VERSION file
Base: 0.0.0

# Feature increments (cumulative):
+ Kalman Filter        → +0.1.0 = 0.1.0
+ Fuzzy Logic         → +0.1.0 = 0.2.0
+ Kelly Criterion     → +0.1.0 = 0.3.0
+ Trajectory Predictor → +0.1.0 = 0.4.0
+ Momentum Persistence → +0.1.0 = 0.5.0
+ Recovery Detector    → +0.1.0 = 0.6.0

# Effective version: v0.6.0 (Kalman + Fuzzy + Kelly + Predictive)
```

### **Feature Detection**

Features auto-detected from:
- **Environment variables**: `KALMAN_ENABLED`, `ADVANCED_EXITS_ENABLED`, `PREDICTIVE_ENABLED`
- **Import availability**: Modules in `src/` directory
- **Runtime checks**: Component initialization

### **Version Display**

```python
from src.version import get_version, get_detailed_version

print(get_version())           # "0.6.0"
print(get_detailed_version())  # "v0.6.0 (Kalman + Fuzzy + Kelly + Predictive)"
```

### **Changelog Management**

All changes documented in `CHANGELOG.md`:

```markdown
## [0.6.0] - 2026-02-11
### Added
- Trajectory Predictor for profit forecasting
- Momentum Persistence Detector
- Recovery Strength Analyzer

### Changed
- Exit strategy version: v6.2 → v6.3
- Fuzzy threshold now dynamic (85-98%)

### Fixed
- ExitReason.STOP_LOSS → POSITION_LIMIT
```

### **Version Update Workflow**

1. **Add new feature** → Automatically increments MINOR version
2. **Fix bug** → Manually increment PATCH in `VERSION` file
3. **Breaking change** → Manually increment MAJOR in `VERSION` file
4. **Update CHANGELOG.md** → Document all changes
5. **Commit** → Version updates committed with changes

### **When to Update VERSION File**

**Auto-incremented** (no manual change needed):
- Adding new predictive modules
- Enabling/disabling feature flags
- Adding new exit strategies

**Manual increment required**:
- Bug fixes → Increment PATCH (0.6.0 → 0.6.1)
- Breaking changes → Increment MAJOR (0.6.0 → 1.0.0)
- Resetting versions → Edit `VERSION` file directly

### **Example Version History**

```
v0.0.0 - Initial release (baseline)
v0.1.0 - Added Kalman Filter
v0.2.0 - Added Fuzzy Logic Controller
v0.3.0 - Added Kelly Criterion
v0.4.0 - Added Trajectory Predictor
v0.5.0 - Added Momentum Persistence
v0.6.0 - Added Recovery Detector (v6.3 Predictive Intelligence complete)
v0.6.1 - Fixed variable scope bug (PATCH)
v0.7.0 - Added new session filter (MINOR)
v1.0.0 - Complete rewrite with new ML architecture (MAJOR)
```

### **Version in Logs**

```
============================================================
XAUBOT AI v0.6.0 (Kalman + Fuzzy + Kelly + Predictive)
Strategy: Exit v6.3 Predictive Intelligence
============================================================
SMART RISK MANAGER v0.6.0 (Exit v6.3 Predictive Intelligence) INITIALIZED
  [OK] Fuzzy Exit Controller initialized
  [OK] Kelly Position Scaler initialized
  [OK] Trajectory Predictor initialized
  [OK] Momentum Persistence initialized
  [OK] Recovery Detector initialized
  Advanced Exits: ENABLED (Kalman + Fuzzy + Kelly + Predictive)
============================================================
```

### **Best Practices**

1. **Always update CHANGELOG.md** when making changes
2. **Use semantic commit messages**: `feat:`, `fix:`, `docs:`, `refactor:`
3. **Version tags in git**: `git tag v0.6.0` after stable release
4. **Document breaking changes** clearly in CHANGELOG
5. **Test version detection**: `python src/version.py` to verify

### **Quick Reference**

| Action | Version Impact | Example |
|--------|---------------|---------|
| Add feature | +0.1.0 (MINOR) | Predictive Intelligence |
| Fix bug | +0.0.1 (PATCH) | Variable scope fix |
| Breaking change | +1.0.0 (MAJOR) | API redesign |
| Enable feature flag | Auto-detected | `PREDICTIVE_ENABLED=1` |
| Disable feature | Auto-detected | `KALMAN_ENABLED=0` |

---
