# Smart Automatic Trading BOT + AI

An intelligent automated trading system for XAUUSD (Gold) using Machine Learning and Smart Money Concepts (SMC).

## Features

- **ML-Powered Predictions**: XGBoost model with 37 features for market direction prediction
- **Smart Money Concepts (SMC)**: Order Blocks, Fair Value Gaps, Break of Structure, Change of Character
- **HMM Regime Detection**: Hidden Markov Model for market regime classification
- **Dynamic Risk Management**: ATR-based stop loss, position sizing, and smart exits
- **Session-Aware Trading**: Optimized for different market sessions (Sydney, London, NY)
- **Auto-Retraining**: Models automatically retrain based on market conditions
- **Telegram Notifications**: Real-time trade alerts and market updates
- **Web Dashboard**: Real-time monitoring interface

## Performance (Backtest Jan 2025 - Feb 2026)

| Metric | Value |
|--------|-------|
| Total Trades | 654 |
| Win Rate | 63.9% |
| Net P/L | $4,189.52 |
| Profit Factor | 2.64 |
| Max Drawdown | 2.2% |
| Sharpe Ratio | 4.83 |

## Architecture

```
├── main_live.py           # Main trading orchestrator
├── src/
│   ├── ml_model.py        # XGBoost ML model
│   ├── smc_polars.py      # Smart Money Concepts analyzer
│   ├── regime_detector.py # HMM market regime detection
│   ├── smart_risk_manager.py # Risk management system
│   ├── feature_eng.py     # Feature engineering
│   ├── mt5_connector.py   # MetaTrader 5 connection
│   ├── session_filter.py  # Trading session management
│   └── ...
├── backtests/
│   ├── backtest_live_sync.py # Main backtest (synced with live)
│   └── archive/           # Historical backtest scripts
├── models/                # Trained ML models (.pkl)
├── data/                  # Market data and trade logs
├── docs/                  # Documentation
└── web-dashboard/         # Next.js monitoring dashboard
```

## Risk Management

- **ATR-Based Stop Loss**: Minimum 1.5 ATR distance
- **Broker-Level Protection**: Emergency SL at broker level
- **Time-Based Exit**: Max 6 hours per trade
- **Daily Loss Limit**: 5% of capital
- **Position Limit**: Max 2 concurrent positions

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure:
   - MT5 credentials
   - Telegram bot token
   - Database connection
4. Train models:
   ```bash
   python train_models.py
   ```
5. Run the bot:
   ```bash
   python main_live.py
   ```

## Configuration

Key settings in `.env`:
- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` - MetaTrader 5 credentials
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` - Telegram notifications
- `CAPITAL` - Trading capital amount
- `SYMBOL` - Trading symbol (default: XAUUSD)

## Backtest

Run backtest with threshold tuning:
```bash
python backtests/backtest_live_sync.py --tune
```

Run backtest with specific threshold:
```bash
python backtests/backtest_live_sync.py --threshold 0.50 --save
```

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk.

## License

MIT License
