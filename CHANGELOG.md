# Changelog

All notable changes to XAUBot AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional versioning system with semantic versioning (MAJOR.MINOR.PATCH)
- Automated version detection based on enabled features
- Centralized version management via `src/version.py`
- Comprehensive changelog following Keep a Changelog format

---

## [0.0.0] - 2026-02-11

### Initial Release
Starting point for versioned releases. All previous development consolidated into v0.0.0 baseline.

#### Core Features
- **MT5 Integration**: Real-time connection to MetaTrader 5
- **Smart Money Concepts (SMC)**: Order Blocks, Fair Value Gaps, BOS/CHoCH detection
- **Machine Learning**: XGBoost model for trade signal prediction (37 features)
- **HMM Regime Detection**: Market classification (trending/ranging/volatile)
- **Risk Management**: Multi-tier capital modes (MICRO/SMALL/MEDIUM/LARGE)
- **Session Filtering**: Sydney/London/NY session optimization
- **Telegram Notifications**: Real-time trade alerts and commands

#### Advanced Exit Systems
- **v6.0 Kalman Intelligence**: Kalman filter for velocity smoothing
- **v6.1 Profit-Tier Strategy**: Dynamic exit thresholds based on profit magnitude
- **v6.2 Bug Fixes**: ExitReason.STOP_LOSS → POSITION_LIMIT correction
- **v6.3 Predictive Intelligence**:
  - Trajectory Predictor (profit forecasting 1-5min ahead)
  - Momentum Persistence Detector (continuation probability)
  - Recovery Strength Analyzer (loss recovery optimization)

#### Technical Infrastructure
- **Framework**: Python 3.11+, Polars (not Pandas), asyncio
- **Models**: XGBoost (binary classification), HMM (regime detection)
- **Database**: PostgreSQL for trade logging
- **Dashboard**: Next.js web monitoring interface
- **Deployment**: Docker support with multi-environment configs

### Performance Metrics (Baseline)
- Win Rate: 56-58%
- Average Win: $2.78 (v6.2) → Target $6-8 (v6.3)
- Peak Capture: 71% → Target 85%+
- Daily Loss Limit: 5% of capital
- Risk per Trade: 0.5-2% (capital-mode dependent)

---

## Version History Format

### [MAJOR.MINOR.PATCH] - YYYY-MM-DD

#### Added
- New features that are backward compatible

#### Changed
- Changes in existing functionality

#### Deprecated
- Features that will be removed in future versions

#### Removed
- Features that have been removed

#### Fixed
- Bug fixes

#### Security
- Security vulnerability fixes

---

## Semantic Versioning Guidelines

### MAJOR version (x.0.0)
Increment when making incompatible API changes:
- Breaking changes to core trading logic
- Removal of major features
- Database schema changes requiring migration
- Configuration format changes

Examples:
- Switching from Pandas to Polars
- Changing ML model architecture completely
- Removing hard stop-loss system

### MINOR version (0.x.0)
Increment when adding functionality in a backward-compatible manner:
- New exit strategies (e.g., v6.3 Predictive Intelligence)
- New indicators or features
- New filters or risk management modes
- Enhanced logging or monitoring

Examples:
- Adding Trajectory Predictor
- Adding new session filter
- Implementing Kelly Criterion

### PATCH version (0.0.x)
Increment when making backward-compatible bug fixes:
- Bug fixes that don't change behavior
- Performance optimizations
- Documentation updates
- Code refactoring (no logic changes)

Examples:
- Fixing ExitReason.STOP_LOSS typo
- Fixing variable scope errors
- Correcting log messages

---

## Feature Tracking

Current feature set determines version automatically:

| Feature | Version Component | Impact |
|---------|------------------|--------|
| Basic Trading (SMC + ML + MT5) | 0.x.x | Core |
| Exit v6.0 (Kalman) | 0.1.x | MINOR |
| Exit v6.1 (Profit-Tier) | 0.2.x | MINOR |
| Exit v6.2 (Bug Fixes) | 0.2.1 | PATCH |
| Exit v6.3 (Predictive) | 0.3.x | MINOR |
| Fuzzy Logic Controller | +0.1 | MINOR |
| Kelly Criterion | +0.1 | MINOR |
| Recovery Detector | +0.1 | MINOR |

---

## Links
- [Repository](https://github.com/GifariKemal/xaubot-ai)
- [Documentation](./docs/)
- [Issues](https://github.com/GifariKemal/xaubot-ai/issues)
