# XAUBot AI — Feature Reference

## Overview

XAUBot AI is an automated XAUUSD (Gold) trading bot that combines **XGBoost Machine Learning**, **Smart Money Concepts (SMC)**, and **Hidden Markov Model (HMM)** regime detection. It operates on MetaTrader 5 via an asynchronous Python loop, executing trades on the M15 (15-minute) timeframe.

The bot follows a strict pipeline: data is fetched, features are engineered, market structure is analyzed, regime is classified, ML predictions are generated, and a series of 14 sequential filters determine whether a trade is executed. Once in a position, 12 exit conditions are monitored every 5-10 seconds.

---

## Entry Filter Pipeline

There are **14 filters** that run in order during `_trading_iteration()`. A signal must pass **ALL** of them to execute a trade.

### 1. Data Fetch
- Pulls **200 M15 bars** from MetaTrader 5.
- Data is converted to a **Polars DataFrame** (not Pandas).

### 2. Feature Engineering
- Calculates **37 technical features** from the OHLCV data.
- Includes: RSI, ATR, MACD, Bollinger Bands, EMA (multiple periods), Stochastic, volume-based indicators, and more.
- All computations use Polars for performance.

### 3. SMC Analysis
- Detects institutional **Smart Money Concepts** structures:
  - **Order Blocks (OB)** — supply/demand zones from institutional activity.
  - **Fair Value Gaps (FVG)** — imbalances in price action.
  - **Break of Structure (BOS)** — continuation signals.
  - **Change of Character (CHoCH)** — reversal signals.

### 4. Regime Detection
- **HMM (Hidden Markov Model)** classifies the current market state:
  - `TRENDING` — directional movement, favorable for entries.
  - `RANGING` — sideways consolidation, reduced sizing.
  - `HIGH_VOLATILITY` — erratic movement, caution required.
  - `CRISIS` — extreme conditions, trading blocked.

### 5. Flash Crash Guard
- Emergency protection: if price move exceeds a threshold percentage, **all positions are immediately closed**.
- Prevents catastrophic loss during sudden market dislocations.

### 6. Regime Filter
- Blocks trading entirely if the regime recommendation is `SLEEP`.
- Prevents entries during unfavorable market conditions identified by the HMM.

### 7. Risk Check
- Blocks trading if:
  - **Daily loss limit** has been reached (5% of capital).
  - **Equity** is too low relative to required margin.
  - **Total loss limit** has been breached (10% of capital).

### 8. Session Filter
- Filters based on **WIB (Western Indonesian Time)** trading sessions.
- Each session applies a **lot size multiplier** to control exposure:
  - **Sydney** (06:00-13:00 WIB) — 0.5x multiplier (low volatility).
  - **Tokyo** (07:00-16:00 WIB) — 0.7x multiplier (medium volatility).
  - **London** (15:00-24:00 WIB) — 1.0x multiplier (high volatility).
  - **New York** (20:00-24:00 WIB) — 1.0x multiplier (extreme volatility).
  - **Off-Hours** (00:00-06:00 WIB) — **blocked entirely**.

### 9. H1 Bias Filter (#31B)
- Multi-timeframe confirmation using **EMA20 on the H1 chart**.
- Price position relative to H1 EMA20 determines directional bias:
  - **BULLISH** (price above EMA20) — only BUY signals allowed.
  - **BEARISH** (price below EMA20) — only SELL signals allowed.
  - **NEUTRAL** (price near EMA20) — **all signals blocked**.
- Backtest result: **+$343 improvement, 81.8% win rate, Sharpe 3.97**.

### 10. SMC Signal Generation
- Generates a **BUY or SELL signal** based on SMC structure analysis.
- Each signal includes a **confidence score** derived from the quality of the detected structures (OB proximity, FVG alignment, BOS/CHoCH context).

### 11. Signal Combination
- Combines **SMC signal + ML (XGBoost) prediction**.
- Applies a **dynamic confidence threshold** that adapts based on:
  - Current trading session.
  - Market regime.
  - Recent volatility.
- Both signals must agree on direction; combined confidence must exceed the threshold.

### 12. Time Filter (#34A)
- Skips specific WIB hours known for poor conditions:
  - **Hour 9 WIB** — end of New York session, low liquidity.
  - **Hour 21 WIB** — London-New York transition, prone to whipsaw.
- Backtest result: **+$356 improvement**.

### 13. Trade Cooldown
- Enforces a minimum **150 seconds (2.5 minutes)** between consecutive trades.
- Prevents overtrading and rapid-fire entries from noisy signals.

### 14. Smart Risk Gate
- Final gate before execution. Checks:
  - **Trading mode**: `NORMAL`, `RECOVERY`, `PROTECTED`, or `STOPPED`.
  - **Lot size calculation**: Based on ATR, capital mode, and session multiplier.
  - **Position limit**: Maximum **2 concurrent positions** allowed.
- If mode is `STOPPED`, no trade is executed regardless of signal quality.

---

## Exit Conditions

**12 exit conditions** are checked every **5-10 seconds** while a position is open.

### 1. Take Profit (Broker-Level TP)
- TP is set at the broker level at entry time.
- Calculated using ATR-based risk-reward ratios.

### 2. Trailing Stop (#24B)
- **ATR-adaptive trailing stop**:
  - Activation distance: **ATR x 4.0**.
  - Step size: **ATR x 3.0**.
- Locks in profits as price moves favorably.

### 3. Breakeven Move (#24B)
- Moves stop loss to **entry price** (breakeven) when unrealized profit exceeds **ATR x 2.0**.
- Eliminates risk on the trade after a favorable move.

### 4. ML Reversal Exit
- Closes the position if the ML model's confidence **flips direction** with confidence exceeding **75%**.
- Responds to changing market conditions detected by XGBoost.

### 5. Max Loss Per Trade
- **Software-level stop loss** at **1% of capital**.
- Acts as a safety net in addition to broker SL.

### 6. Daily Loss Limit
- If cumulative daily loss reaches **5% of capital**, **all positions are closed** and trading halts for the day.

### 7. Total Loss Limit
- If cumulative total loss reaches **10% of capital**, **trading is stopped entirely** until manual intervention.

### 8. Market Close Handler
- Before daily close or weekend close:
  - Takes profit on positions with unrealized profit **> $5**.
  - Prevents gap risk from overnight/weekend holds.

### 9. Flash Crash Emergency
- Triggered by sudden extreme price movement.
- **Immediately closes all open positions** without delay.

### 10. Drawdown Protection
- Monitors drawdown from equity peak.
- Closes all positions if drawdown exceeds **50%** from the peak.

### 11. Impulse Trail (#33B)
- Enhanced trailing stop using **impulse candle detection**.
- Identifies strong momentum candles and trails the stop behind them.
- More responsive than standard ATR trailing in trending conditions.

### 12. Smart Breakeven (#28B)
- Enhanced breakeven logic with **ATR multiplier triggers**:
  - Trigger: profit exceeds **ATR x 2.0**.
  - Moves SL to entry + small buffer.
- More adaptive than fixed-pip breakeven.

---

## Backtest Optimization History

Summary of key optimizations applied to the live bot, tested and validated through backtesting.

| # | Name | Key Change | Result |
|---|------|------------|--------|
| #24B | ATR-Adaptive Exit | ATR-based trailing (4.0x) and breakeven (2.0x) multipliers | Base optimization for exit logic |
| #28B | Smart Breakeven | Enhanced breakeven with ATR x 2.0 trigger | Improved exit timing on winning trades |
| #31B | H1 EMA20 Filter | H1 price vs EMA20 multi-timeframe filter | +$343, WR 81.8%, Sharpe 3.97 |
| #33B | Impulse Trail | Trail using impulse candle detection | Better trailing in trending markets |
| #34A | Skip Hours | Skip WIB hours 9 and 21 | +$356, reduced whipsaw losses |

---

## Risk Management

### Capital Modes

Capital modes are auto-configured based on account balance. Each mode sets risk parameters appropriate for the account size.

| Mode | Capital Range | Risk/Trade | Max Lot |
|------|--------------|------------|---------|
| MICRO | < $500 | 2% | 0.02 |
| SMALL | $500 - $10,000 | 1.5% | 0.05 |
| MEDIUM | $10,000 - $100,000 | 0.5% | 0.10 |
| LARGE | > $100,000 | 0.25% | 0.50 |

### Trading Modes

The Smart Risk Manager dynamically adjusts the trading mode based on recent performance.

| Mode | Trigger | Lot Adjustment |
|------|---------|---------------|
| NORMAL | Default state | Base lot (0.01-0.03) |
| RECOVERY | After a losing trade | Recovery lot (0.01) |
| PROTECTED | Approaching daily loss limit | Minimum lot (0.01) |
| STOPPED | Daily or total loss limit hit | No trading allowed |

### Risk Limits

| Limit | Value | Action |
|-------|-------|--------|
| Max daily loss | 5% of capital | Close all positions, halt trading for the day |
| Max total loss | 10% of capital | Stop all trading until manual reset |
| Max loss per trade | 1% of capital | Software stop loss |
| Emergency broker SL | 2% of capital | Broker-level hard stop |
| Max concurrent positions | 2 | Reject new entries if at limit |

---

## Session Filter (WIB)

All session times are in **WIB (Western Indonesian Time, UTC+7)**.

| Session | Hours (WIB) | Volatility | Lot Multiplier |
|---------|-------------|------------|----------------|
| Sydney | 06:00 - 13:00 | Low | 0.5x |
| Tokyo | 07:00 - 16:00 | Medium | 0.7x |
| London | 15:00 - 24:00 | High | 1.0x |
| New York | 20:00 - 24:00 | Extreme | 1.0x |
| Off-Hours | 00:00 - 06:00 | N/A | **Blocked** |

### Golden Hour
- **19:00 - 23:00 WIB** (London-New York Overlap).
- Highest liquidity and volatility period for XAUUSD.
- Best trading conditions; full lot multiplier applied.

### Skip Hours (#34A)
- **Hour 9 WIB** — End of New York session; low liquidity leads to erratic fills.
- **Hour 21 WIB** — London-New York transition; prone to whipsaw and false breakouts.

---

## Auto-Trainer

The bot includes an automatic model retraining pipeline to keep the ML model current with market conditions.

| Parameter | Value |
|-----------|-------|
| Check interval | Every 20 candles (~5 hours on M15) |
| Daily retrain | 05:00 WIB (during market close) |
| Weekend training | Deep training with expanded data window |
| Min AUC threshold | 0.65 |
| Rollback policy | If new model performs worse, revert to backup |

### Retraining Flow
1. Every 20 candles, the auto-trainer checks model performance metrics.
2. If AUC drops below **0.65**, a retrain is triggered.
3. At **05:00 WIB daily** (market close), a scheduled retrain runs.
4. On **weekends**, deep training uses a larger historical dataset.
5. After training, the new model is validated against the previous one.
6. If the new model underperforms, the system **rolls back** to the backup model.

---

## ML Model

### Algorithm
- **XGBoost** gradient-boosted decision trees.

### Features
- **37 technical indicators** computed by `src/feature_eng.py`:
  - Trend: EMA (multiple periods), MACD, ADX.
  - Momentum: RSI, Stochastic K/D.
  - Volatility: ATR, Bollinger Bands (width, %B).
  - Volume: Volume-weighted indicators.
  - Custom: SMC-derived features, regime features.

### Output
- **Signal**: BUY, SELL, or HOLD.
- **Confidence score**: 0.0 to 1.0, used in combination with SMC confidence.

### Dynamic Threshold
- The confidence threshold for trade execution is not fixed.
- It adjusts based on:
  - **Session**: Higher threshold during low-volatility sessions.
  - **Regime**: Higher threshold during ranging/volatile regimes.
  - **Recent performance**: Tightens after losses, relaxes after wins.

---

## Active Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| SMC Analyzer | `src/smc_polars.py` | Active | Order Block, FVG, BOS, CHoCH detection |
| XGBoost ML | `src/ml_model.py` | Active | Signal prediction with confidence |
| HMM Regime | `src/regime_detector.py` | Active | Market regime classification |
| Feature Engine | `src/feature_eng.py` | Active | 37 technical feature computation |
| Risk Engine | `src/risk_engine.py` | Active | ATR-based SL/TP, position sizing |
| Smart Risk Manager | `src/smart_risk_manager.py` | Active | Dynamic mode management |
| Position Manager | `src/position_manager.py` | Active | Exit condition monitoring |
| Session Filter | `src/session_filter.py` | Active | WIB session-based filtering |
| Dynamic Confidence | `src/dynamic_confidence.py` | Active | Adaptive threshold adjustment |
| Auto Trainer | `src/auto_trainer.py` | Active | Scheduled model retraining |
| Telegram Notifier | `src/telegram_notifier.py` | Active | Trade alerts via Telegram |
| Trade Logger | `src/trade_logger.py` | Active | PostgreSQL trade logging |
| News Agent | `src/news_agent.py` | **DISABLED** | Economic news filter (costs $178 profit in backtest) |
| Flash Crash Detector | `src/regime_detector.py` | Active | Emergency position closure |

---

## Architecture Diagram

```
MT5 Broker
    |
    v
[Data Fetch] --> [Feature Eng (37)] --> [SMC Analysis] --> [Regime Detection (HMM)]
                                                                    |
                                                                    v
                                                          [Flash Crash Guard]
                                                                    |
                                                                    v
                                                           [Regime Filter]
                                                                    |
                                                                    v
                                                            [Risk Check]
                                                                    |
                                                                    v
                                                          [Session Filter]
                                                                    |
                                                                    v
                                                        [H1 Bias Filter (#31B)]
                                                                    |
                                                                    v
                                                          [SMC Signal Gen]
                                                                    |
                                                                    v
                                                       [Signal Combination (ML+SMC)]
                                                                    |
                                                                    v
                                                        [Time Filter (#34A)]
                                                                    |
                                                                    v
                                                         [Trade Cooldown]
                                                                    |
                                                                    v
                                                        [Smart Risk Gate]
                                                                    |
                                                                    v
                                                        [TRADE EXECUTION]
                                                                    |
                                                                    v
                                                   [Position Manager (12 exits)]
                                                                    |
                                                                    v
                                                  [Telegram + PostgreSQL Logging]
```
