"""
Deep Analysis: News Filter Impact on Trading Performance
=========================================================
Analisis mendalam apakah news filter tepat diterapkan.

Metodologi:
1. Gunakan model ML ASLI (XGBoost) untuk prediksi
2. Simulasikan trading logic seperti di main_live.py
3. Bandingkan beberapa skenario news filter
4. Analisis trades saat news vs non-news
5. Hitung opportunity cost dari news filter
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>", level="INFO")

# ============================================================
# HISTORICAL NEWS CALENDAR 2025-2026
# ============================================================

HISTORICAL_NEWS = [
    # Format: (date, hour_wib, event_name, impact)
    # May 2025
    (date(2025, 5, 2), 19, "NFP", "HIGH"),
    (date(2025, 5, 7), 1, "FOMC", "HIGH"),
    (date(2025, 5, 13), 19, "CPI", "HIGH"),
    (date(2025, 5, 14), 19, "PPI", "MEDIUM"),
    (date(2025, 5, 29), 19, "GDP", "MEDIUM"),

    # June 2025
    (date(2025, 6, 6), 19, "NFP", "HIGH"),
    (date(2025, 6, 11), 19, "CPI", "HIGH"),
    (date(2025, 6, 12), 19, "PPI", "MEDIUM"),
    (date(2025, 6, 18), 1, "FOMC", "HIGH"),
    (date(2025, 6, 26), 19, "GDP", "MEDIUM"),

    # July 2025
    (date(2025, 7, 3), 19, "NFP", "HIGH"),
    (date(2025, 7, 11), 19, "CPI", "HIGH"),
    (date(2025, 7, 15), 19, "PPI", "MEDIUM"),
    (date(2025, 7, 30), 1, "FOMC", "HIGH"),
    (date(2025, 7, 31), 19, "GDP", "HIGH"),

    # August 2025
    (date(2025, 8, 1), 19, "NFP", "HIGH"),
    (date(2025, 8, 13), 19, "CPI", "HIGH"),
    (date(2025, 8, 14), 19, "PPI", "MEDIUM"),
    (date(2025, 8, 28), 19, "GDP", "MEDIUM"),

    # September 2025
    (date(2025, 9, 5), 19, "NFP", "HIGH"),
    (date(2025, 9, 10), 19, "CPI", "HIGH"),
    (date(2025, 9, 11), 19, "PPI", "MEDIUM"),
    (date(2025, 9, 17), 1, "FOMC", "HIGH"),
    (date(2025, 9, 25), 19, "GDP", "MEDIUM"),

    # October 2025
    (date(2025, 10, 3), 19, "NFP", "HIGH"),
    (date(2025, 10, 10), 19, "CPI", "HIGH"),
    (date(2025, 10, 14), 19, "PPI", "MEDIUM"),
    (date(2025, 10, 30), 19, "GDP", "HIGH"),

    # November 2025
    (date(2025, 11, 7), 19, "NFP", "HIGH"),
    (date(2025, 11, 5), 1, "FOMC", "HIGH"),
    (date(2025, 11, 13), 19, "CPI", "HIGH"),
    (date(2025, 11, 14), 19, "PPI", "MEDIUM"),
    (date(2025, 11, 26), 19, "GDP", "MEDIUM"),

    # December 2025
    (date(2025, 12, 5), 19, "NFP", "HIGH"),
    (date(2025, 12, 10), 19, "CPI", "HIGH"),
    (date(2025, 12, 11), 19, "PPI", "MEDIUM"),
    (date(2025, 12, 17), 1, "FOMC", "HIGH"),

    # January 2026
    (date(2026, 1, 10), 20, "NFP", "HIGH"),
    (date(2026, 1, 15), 20, "CPI", "HIGH"),
    (date(2026, 1, 29), 2, "FOMC", "HIGH"),

    # February 2026
    (date(2026, 2, 5), 20, "NFP", "HIGH"),
]


class NewsFilterMode:
    """Different news filter configurations."""

    @staticmethod
    def no_filter(dt: datetime, news_list: list) -> Tuple[bool, str]:
        """No filtering - always allow trading."""
        return False, "No filter"

    @staticmethod
    def conservative(dt: datetime, news_list: list) -> Tuple[bool, str]:
        """Block entire day for HIGH impact news."""
        current_date = dt.date()
        for news_date, hour, name, impact in news_list:
            if news_date == current_date and impact == "HIGH":
                return True, f"{name} day"
        return False, "Clear"

    @staticmethod
    def moderate(dt: datetime, news_list: list) -> Tuple[bool, str]:
        """Block 2 hours before and after HIGH impact news."""
        current_date = dt.date()
        current_hour = dt.hour

        for news_date, news_hour, name, impact in news_list:
            if news_date == current_date:
                if impact == "HIGH":
                    # 2 hours before and after
                    if abs(current_hour - news_hour) <= 2:
                        return True, f"{name} (+/-2h)"
                elif impact == "MEDIUM":
                    # 1 hour before and after for medium
                    if abs(current_hour - news_hour) <= 1:
                        return True, f"{name} (+/-1h)"
        return False, "Clear"

    @staticmethod
    def aggressive(dt: datetime, news_list: list) -> Tuple[bool, str]:
        """Block only 1 hour around HIGH impact news."""
        current_date = dt.date()
        current_hour = dt.hour

        for news_date, news_hour, name, impact in news_list:
            if news_date == current_date and impact == "HIGH":
                if abs(current_hour - news_hour) <= 1:
                    return True, f"{name} (+/-1h)"
        return False, "Clear"


@dataclass
class Trade:
    """Trade record with news context."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    ml_confidence: float
    during_news: bool = False
    news_event: str = ""


@dataclass
class AnalysisResult:
    """Comprehensive analysis result."""
    filter_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

    # News-specific
    trades_blocked: int
    trades_during_news: int
    pnl_during_news: float
    pnl_outside_news: float

    trades: List[Trade] = field(default_factory=list)


def load_data_and_model():
    """Load market data and ML model."""
    try:
        import MetaTrader5 as mt5
        from src.config import get_config
        from src.ml_model import TradingModel
        from src.feature_eng import FeatureEngineer
        from src.smc_polars import SMCAnalyzer
        from src.regime_detector import MarketRegimeDetector
        import time

        config = get_config()

        # Initialize MT5
        if not mt5.initialize(
            path=config.mt5_path,
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
        ):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return None, None, None

        logger.info(f"MT5 connected: {mt5.account_info().server}")

        # Enable symbol
        symbol = "XAUUSD"
        mt5.symbol_select(symbol, True)
        time.sleep(0.5)

        # Get data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60000)
        mt5.shutdown()

        if rates is None:
            logger.error("No data received")
            return None, None, None

        # Convert to DataFrame
        df = pl.DataFrame({
            "time": [datetime.fromtimestamp(r[0]) for r in rates],
            "open": [r[1] for r in rates],
            "high": [r[2] for r in rates],
            "low": [r[3] for r in rates],
            "close": [r[4] for r in rates],
            "volume": [float(r[5]) for r in rates],
        })

        logger.info(f"Loaded {len(df)} bars: {df['time'].min()} to {df['time'].max()}")

        # Calculate technical features
        fe = FeatureEngineer()
        df = fe.calculate_all(df, include_ml_features=True)

        # Calculate SMC features
        smc = SMCAnalyzer()
        df = smc.calculate_all(df)

        # Calculate HMM Regime
        logger.info("Calculating HMM regime...")
        regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
        regime_detector.load()
        if regime_detector.fitted:
            df = regime_detector.predict(df)
            logger.info("HMM regime calculated")
        else:
            # Add default regime if model not loaded
            logger.warning("HMM model not fitted, using default regime")
            df = df.with_columns(pl.lit(0).alias("regime"))

        logger.info(f"Features calculated: {len(df.columns)} columns")

        # Load ML model
        ml_model = TradingModel(model_path="models/xgboost_model.pkl")
        ml_model.load()

        if not ml_model.fitted:
            logger.error("ML model not loaded")
            return df, None, None

        logger.info(f"ML model loaded: {len(ml_model.feature_names)} features")

        return df, ml_model, ml_model.feature_names

    except Exception as e:
        logger.error(f"Error loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def is_during_news_window(dt: datetime, window_hours: int = 2) -> Tuple[bool, str]:
    """Check if datetime is within news window."""
    current_date = dt.date()
    current_hour = dt.hour

    for news_date, news_hour, name, impact in HISTORICAL_NEWS:
        if news_date == current_date:
            if abs(current_hour - news_hour) <= window_hours:
                return True, name
    return False, ""


def run_backtest(
    df: pl.DataFrame,
    ml_model,
    feature_names: List[str],
    filter_func,
    filter_name: str,
) -> AnalysisResult:
    """Run backtest with specific news filter."""

    logger.info(f"Running backtest: {filter_name}")

    trades: List[Trade] = []
    trades_blocked = 0

    position = None
    capital = 5000.0
    lot_size = 0.02

    # Get available features
    available_features = [f for f in feature_names if f in df.columns]

    for idx in range(200, len(df) - 1):
        row = df.row(idx, named=True)
        current_time = row["time"]

        # Filter by date range
        if current_time.date() < date(2025, 5, 22):
            continue
        if current_time.date() > date(2026, 2, 5):
            break

        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row.get("atr_14", close * 0.003)
        if atr is None or atr == 0:
            atr = close * 0.003

        # Manage position
        if position is not None:
            if position["direction"] == "BUY":
                if low <= position["sl"]:
                    pnl = (position["sl"] - position["entry_price"]) * lot_size * 100
                    during_news, news_name = is_during_news_window(position["entry_time"])
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="BUY",
                        entry_price=position["entry_price"],
                        exit_price=position["sl"],
                        lot_size=lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                        during_news=during_news,
                        news_event=news_name,
                    ))
                    capital += pnl
                    position = None
                elif high >= position["tp"]:
                    pnl = (position["tp"] - position["entry_price"]) * lot_size * 100
                    during_news, news_name = is_during_news_window(position["entry_time"])
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="BUY",
                        entry_price=position["entry_price"],
                        exit_price=position["tp"],
                        lot_size=lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                        during_news=during_news,
                        news_event=news_name,
                    ))
                    capital += pnl
                    position = None
            else:  # SELL
                if high >= position["sl"]:
                    pnl = (position["entry_price"] - position["sl"]) * lot_size * 100
                    during_news, news_name = is_during_news_window(position["entry_time"])
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="SELL",
                        entry_price=position["entry_price"],
                        exit_price=position["sl"],
                        lot_size=lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                        during_news=during_news,
                        news_event=news_name,
                    ))
                    capital += pnl
                    position = None
                elif low <= position["tp"]:
                    pnl = (position["entry_price"] - position["tp"]) * lot_size * 100
                    during_news, news_name = is_during_news_window(position["entry_time"])
                    trades.append(Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        direction="SELL",
                        entry_price=position["entry_price"],
                        exit_price=position["tp"],
                        lot_size=lot_size,
                        pnl=pnl,
                        ml_confidence=position["confidence"],
                        during_news=during_news,
                        news_event=news_name,
                    ))
                    capital += pnl
                    position = None

        if position is not None:
            continue

        # Session filter (London/NY only: 14:00-23:00 WIB)
        hour = current_time.hour
        if hour < 14 or hour > 23:
            continue

        # NEWS FILTER CHECK
        is_blocked, block_reason = filter_func(current_time, HISTORICAL_NEWS)
        if is_blocked:
            trades_blocked += 1
            continue

        # ML Prediction using actual model
        try:
            # Get slice for prediction
            df_slice = df.slice(max(0, idx - 100), 101)
            prediction = ml_model.predict(df_slice, available_features)

            signal = prediction.signal
            confidence = prediction.confidence
        except Exception as e:
            continue

        # Check threshold (ML-Only = 70%)
        if confidence < 0.70:
            continue

        # Entry
        if signal == "BUY":
            sl = close - (atr * 1.5)
            tp = close + (atr * 3.0)
            position = {
                "direction": "BUY",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }
        elif signal == "SELL":
            sl = close + (atr * 1.5)
            tp = close - (atr * 3.0)
            position = {
                "direction": "SELL",
                "entry_price": close,
                "entry_time": current_time,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
            }

    # Calculate metrics
    total_trades = len(trades)
    if total_trades == 0:
        return AnalysisResult(
            filter_name=filter_name,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, avg_win=0, avg_loss=0,
            profit_factor=0, max_drawdown=0, sharpe_ratio=0,
            trades_blocked=trades_blocked, trades_during_news=0,
            pnl_during_news=0, pnl_outside_news=0,
        )

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    win_rate = len(winning) / total_trades * 100
    total_pnl = sum(t.pnl for t in trades)

    avg_win = np.mean([t.pnl for t in winning]) if winning else 0
    avg_loss = np.mean([abs(t.pnl) for t in losing]) if losing else 0

    total_wins = sum(t.pnl for t in winning) if winning else 0
    total_losses = sum(abs(t.pnl) for t in losing) if losing else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Max drawdown
    equity = [5000.0]
    for t in trades:
        equity.append(equity[-1] + t.pnl)

    peak = equity[0]
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe ratio (simplified)
    returns = [t.pnl for t in trades]
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0

    # News-specific analysis
    news_trades = [t for t in trades if t.during_news]
    non_news_trades = [t for t in trades if not t.during_news]

    pnl_during_news = sum(t.pnl for t in news_trades)
    pnl_outside_news = sum(t.pnl for t in non_news_trades)

    return AnalysisResult(
        filter_name=filter_name,
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        trades_blocked=trades_blocked,
        trades_during_news=len(news_trades),
        pnl_during_news=pnl_during_news,
        pnl_outside_news=pnl_outside_news,
        trades=trades,
    )


def analyze_news_impact(trades: List[Trade]) -> Dict:
    """Analyze impact of news on trades."""
    news_trades = [t for t in trades if t.during_news]
    non_news_trades = [t for t in trades if not t.during_news]

    if not news_trades:
        return {
            "news_trades": 0,
            "news_win_rate": 0,
            "news_avg_pnl": 0,
            "non_news_trades": len(non_news_trades),
            "non_news_win_rate": sum(1 for t in non_news_trades if t.pnl > 0) / len(non_news_trades) * 100 if non_news_trades else 0,
            "non_news_avg_pnl": np.mean([t.pnl for t in non_news_trades]) if non_news_trades else 0,
        }

    news_wins = sum(1 for t in news_trades if t.pnl > 0)
    non_news_wins = sum(1 for t in non_news_trades if t.pnl > 0)

    return {
        "news_trades": len(news_trades),
        "news_win_rate": news_wins / len(news_trades) * 100,
        "news_avg_pnl": np.mean([t.pnl for t in news_trades]),
        "news_total_pnl": sum(t.pnl for t in news_trades),
        "non_news_trades": len(non_news_trades),
        "non_news_win_rate": non_news_wins / len(non_news_trades) * 100 if non_news_trades else 0,
        "non_news_avg_pnl": np.mean([t.pnl for t in non_news_trades]) if non_news_trades else 0,
        "non_news_total_pnl": sum(t.pnl for t in non_news_trades),
    }


def main():
    """Run comprehensive analysis."""
    print("=" * 70)
    print("DEEP ANALYSIS: NEWS FILTER IMPACT")
    print("=" * 70)
    print()

    # Load data and model
    logger.info("Loading data and ML model...")
    df, ml_model, feature_names = load_data_and_model()

    if df is None or ml_model is None:
        logger.error("Failed to load data or model")
        return

    print()
    print("=" * 70)
    print("RUNNING BACKTESTS WITH DIFFERENT NEWS FILTERS")
    print("=" * 70)
    print()

    # Define filter scenarios
    filters = [
        (NewsFilterMode.no_filter, "NO FILTER"),
        (NewsFilterMode.aggressive, "AGGRESSIVE (+/-1h HIGH only)"),
        (NewsFilterMode.moderate, "MODERATE (+/-2h HIGH, +/-1h MED)"),
        (NewsFilterMode.conservative, "CONSERVATIVE (Block entire day)"),
    ]

    results = []
    for filter_func, filter_name in filters:
        result = run_backtest(df, ml_model, feature_names, filter_func, filter_name)
        results.append(result)
        print(f"\n{filter_name}:")
        print(f"  Trades: {result.total_trades} | WR: {result.win_rate:.1f}% | P/L: ${result.total_pnl:.2f}")
        print(f"  PF: {result.profit_factor:.2f} | MaxDD: {result.max_drawdown:.1f}% | Blocked: {result.trades_blocked}")

    print()
    print("=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Filter':<35} {'Trades':>8} {'WinRate':>8} {'P/L':>12} {'PF':>6} {'MaxDD':>8} {'Sharpe':>8}")
    print("-" * 85)

    for r in results:
        print(f"{r.filter_name:<35} {r.total_trades:>8} {r.win_rate:>7.1f}% ${r.total_pnl:>10.2f} {r.profit_factor:>6.2f} {r.max_drawdown:>7.1f}% {r.sharpe_ratio:>8.2f}")

    print()
    print("=" * 70)
    print("NEWS IMPACT ANALYSIS (from NO FILTER scenario)")
    print("=" * 70)

    # Analyze trades from no-filter scenario
    no_filter_result = results[0]
    impact = analyze_news_impact(no_filter_result.trades)

    print(f"""
Trades DURING News Window (+/-2h):
  Total Trades : {impact['news_trades']}
  Win Rate     : {impact['news_win_rate']:.1f}%
  Avg P/L      : ${impact['news_avg_pnl']:.2f}
  Total P/L    : ${impact.get('news_total_pnl', 0):.2f}

Trades OUTSIDE News Window:
  Total Trades : {impact['non_news_trades']}
  Win Rate     : {impact['non_news_win_rate']:.1f}%
  Avg P/L      : ${impact['non_news_avg_pnl']:.2f}
  Total P/L    : ${impact.get('non_news_total_pnl', 0):.2f}
""")

    # Calculate opportunity cost
    print("=" * 70)
    print("OPPORTUNITY COST ANALYSIS")
    print("=" * 70)

    baseline = results[0]  # No filter
    for r in results[1:]:
        trades_lost = baseline.total_trades - r.total_trades
        pnl_diff = r.total_pnl - baseline.total_pnl
        wr_diff = r.win_rate - baseline.win_rate
        dd_diff = baseline.max_drawdown - r.max_drawdown

        print(f"\n{r.filter_name}:")
        pct_lost = (trades_lost/baseline.total_trades*100) if baseline.total_trades > 0 else 0
        print(f"  Trades Lost     : {trades_lost} ({pct_lost:.1f}%)")
        print(f"  P/L Difference  : ${pnl_diff:+.2f}")
        print(f"  WinRate Change  : {wr_diff:+.1f}%")
        print(f"  MaxDD Reduction : {dd_diff:+.1f}%")

        # Score calculation
        # Positive if: better P/L, better WR, lower DD
        score = 0
        if pnl_diff > 0:
            score += 2
        if wr_diff > 0:
            score += 1
        if dd_diff > 0:
            score += 1
        print(f"  Score           : {score}/4")

    print()
    print("=" * 70)
    print("VERDICT & RECOMMENDATION")
    print("=" * 70)

    # Find best filter based on criteria
    best_pnl = max(results, key=lambda x: x.total_pnl)
    best_wr = max(results, key=lambda x: x.win_rate)
    best_dd = min(results, key=lambda x: x.max_drawdown)
    best_pf = max(results, key=lambda x: x.profit_factor)

    print(f"""
Best Total P/L      : {best_pnl.filter_name} (${best_pnl.total_pnl:.2f})
Best Win Rate       : {best_wr.filter_name} ({best_wr.win_rate:.1f}%)
Best Max Drawdown   : {best_dd.filter_name} ({best_dd.max_drawdown:.1f}%)
Best Profit Factor  : {best_pf.filter_name} ({best_pf.profit_factor:.2f})
""")

    # Final recommendation
    print("-" * 70)

    # Compare no filter vs moderate (our current implementation)
    no_filter = results[0]
    moderate = results[2]

    if moderate.total_pnl > no_filter.total_pnl:
        verdict = "RECOMMENDED"
        reason = "Meningkatkan profit"
    elif moderate.max_drawdown < no_filter.max_drawdown and moderate.win_rate >= no_filter.win_rate - 2:
        verdict = "RECOMMENDED"
        reason = "Mengurangi risk (drawdown) dengan trade quality tetap"
    elif moderate.win_rate > no_filter.win_rate:
        verdict = "RECOMMENDED"
        reason = "Meningkatkan win rate"
    elif no_filter.total_pnl > moderate.total_pnl and (no_filter.total_pnl - moderate.total_pnl) > 50:
        verdict = "NOT RECOMMENDED"
        reason = f"Kehilangan profit ${no_filter.total_pnl - moderate.total_pnl:.2f} tidak worth it"
    else:
        verdict = "OPTIONAL"
        reason = "Impact minimal, gunakan sesuai preferensi risk"

    print(f"""
FINAL VERDICT: {verdict}

Alasan: {reason}

Perbandingan NO FILTER vs MODERATE:
  P/L        : ${no_filter.total_pnl:.2f} vs ${moderate.total_pnl:.2f} ({moderate.total_pnl - no_filter.total_pnl:+.2f})
  Win Rate   : {no_filter.win_rate:.1f}% vs {moderate.win_rate:.1f}% ({moderate.win_rate - no_filter.win_rate:+.1f}%)
  Max DD     : {no_filter.max_drawdown:.1f}% vs {moderate.max_drawdown:.1f}% ({no_filter.max_drawdown - moderate.max_drawdown:+.1f}% reduction)
  PF         : {no_filter.profit_factor:.2f} vs {moderate.profit_factor:.2f}
""")

    # News trade analysis verdict
    if impact['news_trades'] > 0:
        if impact['news_avg_pnl'] < impact['non_news_avg_pnl']:
            print(f"""
ANALISIS TRADING SAAT NEWS:
  - Avg P/L saat news: ${impact['news_avg_pnl']:.2f}
  - Avg P/L diluar news: ${impact['non_news_avg_pnl']:.2f}

  Trades saat news cenderung LEBIH BURUK.
  News filter membantu menghindari trades dengan expected value lebih rendah.
""")
        else:
            print(f"""
ANALISIS TRADING SAAT NEWS:
  - Avg P/L saat news: ${impact['news_avg_pnl']:.2f}
  - Avg P/L diluar news: ${impact['non_news_avg_pnl']:.2f}

  Trades saat news TIDAK lebih buruk dari biasa.
  News filter mungkin tidak diperlukan untuk profitability,
  tapi tetap berguna untuk menghindari volatilitas ekstrem.
""")

    print("=" * 70)
    print("Analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
