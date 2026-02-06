# Backtest Tuning Report
**Date**: February 6, 2026
**Period**: January 2, 2025 - February 4, 2026
**Data**: 25,807 bars (M15 timeframe)

## Threshold Tuning Results

| ML Threshold | Total Trades | Win Rate | Net P/L | Profit Factor |
|--------------|--------------|----------|---------|---------------|
| **50%** | **485** | **61.6%** | **$3,120.55** | **2.02** |
| 52% | 463 | 59.0% | $1,868.08 | 1.55 |
| 55% | 306 | 59.5% | $1,443.56 | 1.74 |

## Optimal Configuration

```python
ML_THRESHOLD = 0.50        # Optimal from tuning
SIGNAL_CONFIRMATION = 2    # Consecutive signals
PULLBACK_FILTER = True     # Enabled
TRADE_COOLDOWN = 300s      # 5 minutes
```

## Performance Metrics (50% Threshold)

### Overall
- **Total Trades**: 485
- **Wins**: 299 (61.6%)
- **Losses**: 186 (38.4%)
- **Net P/L**: $3,120.55
- **Profit Factor**: 2.02

### Risk Metrics
- **Max Drawdown**: 2.4% ($163.74)
- **Avg Win**: $20.69
- **Avg Loss**: $16.48
- **Expectancy**: $6.43 per trade
- **Sharpe Ratio**: 3.69 (Excellent)

### Exit Reasons
| Reason | Count | Percentage |
|--------|-------|------------|
| Take Profit | 271 | 55.9% |
| Trend Reversal | 181 | 37.3% |
| Timeout | 31 | 6.4% |
| Max Loss | 2 | 0.4% |

### Session Performance
| Session | Trades | Win Rate | Net P/L |
|---------|--------|----------|---------|
| **Golden Time (London-NY)** | 103 | **68.0%** | $1,012.72 |
| Tokyo-London Overlap | 25 | **72.0%** | $248.24 |
| Sydney-Tokyo | 215 | 61.4% | $1,233.13 |
| NY Session | 73 | 53.4% | $398.98 |
| London Early | 69 | 58.0% | $227.47 |

## Key Findings

1. **Lower threshold = Better performance**: 50% threshold outperforms 55% significantly
   - 58% more trades (485 vs 306)
   - 2.1% higher win rate (61.6% vs 59.5%)
   - 116% more profit ($3,120 vs $1,443)

2. **Golden Time is still best**: 68% WR with significant profits

3. **Smart exit is effective**:
   - 55.9% take profit (good!)
   - Only 0.4% max loss exits (risk well managed)

4. **Excellent risk-adjusted returns**:
   - Sharpe Ratio 3.69 (>2 is excellent)
   - Max drawdown only 2.4%

## Recommendation

Update main_live.py with:
- ML Threshold: 50% (changed from 55%)
- Keep other filters (pullback, confirmation, session)

**Expected monthly profit**: ~$240 (based on 13-month backtest)
