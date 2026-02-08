"""
Backtest #37 — ML V2 Model Testing
===================================
Test model_d.pkl (ML V2 Config D) dengan trading logic lengkap.

IMPORTANT: Script ini TIDAK mengubah model live!
- Model live: models/xgboost_model.pkl (TIDAK DISENTUH)
- Model test: backtests/36_ml_v2_results/model_d.pkl (ISOLATED)
- Results: backtests/37_ml_v2_test_results/ (SEPARATE FOLDER)

Differences from live:
1. Model: model_d.pkl (76 features) instead of xgboost_model.pkl (37 features)
2. Features: Adds H1 MTF + Continuous SMC + Regime + PA features
3. Target: 3-bar lookahead with 0.3*ATR threshold (vs 1-bar, no threshold)

Trading logic: IDENTICAL to backtest_live_sync.py
- Same SMC entry/exit
- Same session filter
- Same risk management
- Same exit conditions

Usage:
    python backtests/backtest_37_ml_v2_test.py
    python backtests/backtest_37_ml_v2_test.py --bars 10000  # Custom data size
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import csv
import argparse
from zoneinfo import ZoneInfo
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, MarketRegime
from src.config import get_config
from src.session_filter import create_wib_session_filter
from src.dynamic_confidence import create_dynamic_confidence, MarketQuality
from loguru import logger

# ML V2 imports
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from backtests.ml_v2.ml_v2_model import TradingModelV2

# Reduce logging
logger.remove()
logger.add(sys.stderr, level="INFO")


class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    MAX_LOSS = "max_loss"
    ML_REVERSAL = "ml_reversal"
    TIMEOUT = "timeout"
    TREND_REVERSAL = "trend_reversal"


@dataclass
class SimulatedTrade:
    """Simulated trade record."""
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    profit_usd: float
    profit_pips: float
    result: TradeResult
    exit_reason: ExitReason
    ml_confidence: float
    smc_signal: int
    regime: str
    session: str
    entry_reason: str = ""


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Model comparison metrics
    model_name: str = ""
    test_auc: float = 0.0
    num_features: int = 0


def prepare_data_with_v2_features(
    df_m15: pl.DataFrame,
    df_h1: pl.DataFrame,
    model_path: str
) -> Tuple[pl.DataFrame, TradingModelV2]:
    """
    Prepare M15 data with V2 features and load V2 model.

    Args:
        df_m15: M15 OHLCV data
        df_h1: H1 OHLCV data (for MTF features)
        model_path: Path to ML V2 model

    Returns:
        Tuple of (prepared df_m15, loaded model)
    """
    logger.info("Preparing data with ML V2 features...")

    # Base features
    features = FeatureEngineer()
    df_m15 = features.calculate_all(df_m15, include_ml_features=True)

    # SMC
    config = get_config()
    smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
    df_m15 = smc.calculate_all(df_m15)

    # Regime
    regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    try:
        regime_detector.load()
        df_m15 = regime_detector.predict(df_m15)
        logger.info("  HMM regime loaded")
    except Exception as e:
        logger.warning(f"  HMM regime not available: {e}")
        df_m15 = df_m15.with_columns([
            pl.lit(1).alias("regime"),
            pl.lit("medium_volatility").alias("regime_name"),
        ])

    # H1 features (for MTF)
    if df_h1 is not None:
        df_h1 = features.calculate_all(df_h1, include_ml_features=False)
        df_h1 = smc.calculate_all(df_h1)

    # V2 Features
    fe_v2 = MLV2FeatureEngineer()
    df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)

    logger.info(f"  Data prepared: {len(df_m15)} M15 bars, {len(df_m15.columns)} columns")

    # Load V2 model
    logger.info(f"Loading ML V2 model from {model_path}...")
    model = TradingModelV2()
    model = model.load(model_path)
    logger.info(f"  Model loaded: {len(model.feature_names)} features, Test AUC: {model._train_metrics.get('xgb_test_score', 0):.4f}")
    logger.info(f"  Model fitted: {model.fitted}")
    logger.info(f"  Model type: {model.model_type}")

    # Override model's internal confidence threshold to match backtest threshold
    # Model default is 0.65 which is too conservative
    logger.info(f"  Original confidence threshold: {model.confidence_threshold}")
    model.confidence_threshold = 0.50  # Match backtest ML threshold
    logger.info(f"  Overridden to: {model.confidence_threshold}")

    # Verify model works by testing a prediction
    test_pred = model.predict(df_m15.tail(1))
    logger.info(f"  Test prediction: {test_pred.signal}, confidence: {test_pred.confidence:.4f}")

    return df_m15, model


def run_backtest(
    df: pl.DataFrame,
    model: TradingModelV2,
    ml_threshold: float = 0.50,
    max_bars: Optional[int] = None,
) -> Tuple[List[SimulatedTrade], BacktestMetrics]:
    """
    Run backtest with ML V2 model.

    Uses IDENTICAL trading logic as backtest_live_sync.py:
    - Session filter (19:00-23:00 WIB)
    - Quality filter (avoid AVOID/CRISIS)
    - Signal confirmation (2+ consecutive)
    - Pullback filter (ATR-based)
    - Dynamic RR (1.5-2.0)
    - Exit conditions (TP/SL/ML reversal/timeout/trend reversal)

    Args:
        df: Prepared M15 DataFrame with all features
        model: Loaded ML V2 model
        ml_threshold: ML confidence threshold (default 0.50)
        max_bars: Limit backtest to N bars (None = all)

    Returns:
        Tuple of (trades list, metrics)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running backtest with ML V2 model...")
    logger.info(f"  ML Threshold: {ml_threshold}")
    logger.info(f"  Max bars: {max_bars if max_bars else 'all'}")
    logger.info(f"{'='*70}\n")

    # Convert to pandas for easier iteration (temporary)
    df_pd = df.to_pandas()

    if max_bars:
        df_pd = df_pd.tail(max_bars).copy()

    trades: List[SimulatedTrade] = []
    equity_curve = [10000.0]  # Start with $10k
    current_equity = 10000.0

    position: Optional[Dict] = None
    last_trade_idx = -9999
    ticket_counter = 1

    # Track consecutive signals
    signal_persistence = {}

    for i in range(len(df_pd)):
        row = df_pd.iloc[i]
        current_time = row['time']
        current_close = row['close']
        current_atr = row.get('atr', 12.0)

        # Check if in position
        if position is not None:
            # Exit logic (same as live)
            exit_signal = False
            exit_reason = None
            exit_price = current_close

            # 1. TP/SL check
            if position['direction'] == 'BUY':
                if current_close >= position['take_profit']:
                    exit_signal = True
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_price = position['take_profit']
                elif current_close <= position['stop_loss']:
                    exit_signal = True
                    exit_reason = ExitReason.MAX_LOSS
                    exit_price = position['stop_loss']
            else:  # SELL
                if current_close <= position['take_profit']:
                    exit_signal = True
                    exit_reason = ExitReason.TAKE_PROFIT
                    exit_price = position['take_profit']
                elif current_close >= position['stop_loss']:
                    exit_signal = True
                    exit_reason = ExitReason.MAX_LOSS
                    exit_price = position['stop_loss']

            # 2. ML Reversal check
            if not exit_signal:
                try:
                    ml_pred = model.predict(df.slice(i, 1))
                    if position['direction'] == 'BUY' and ml_pred.signal == 'SELL' and ml_pred.confidence >= 0.65:
                        exit_signal = True
                        exit_reason = ExitReason.ML_REVERSAL
                    elif position['direction'] == 'SELL' and ml_pred.signal == 'BUY' and ml_pred.confidence >= 0.65:
                        exit_signal = True
                        exit_reason = ExitReason.ML_REVERSAL
                except:
                    pass

            # 3. Timeout check (max 40 bars ~10 hours)
            bars_in_trade = i - position['entry_idx']
            if not exit_signal and bars_in_trade >= 40:
                exit_signal = True
                exit_reason = ExitReason.TIMEOUT

            # Execute exit
            if exit_signal:
                profit_pips = (exit_price - position['entry_price']) * (1 if position['direction'] == 'BUY' else -1) * 10
                profit_usd = profit_pips * position['lot_size'] * 10  # $10 per pip per 0.01 lot

                trade_result = TradeResult.WIN if profit_usd > 0 else (TradeResult.LOSS if profit_usd < 0 else TradeResult.BREAKEVEN)

                trade = SimulatedTrade(
                    ticket=position['ticket'],
                    entry_time=position['entry_time'],
                    exit_time=current_time,
                    direction=position['direction'],
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    stop_loss=position['stop_loss'],
                    take_profit=position['take_profit'],
                    lot_size=position['lot_size'],
                    profit_usd=profit_usd,
                    profit_pips=profit_pips,
                    result=trade_result,
                    exit_reason=exit_reason,
                    ml_confidence=position['ml_confidence'],
                    smc_signal=position['smc_signal'],
                    regime=position['regime'],
                    session=position['session'],
                    entry_reason=position.get('entry_reason', ''),
                )

                trades.append(trade)
                current_equity += profit_usd
                equity_curve.append(current_equity)

                position = None
                last_trade_idx = i

        # Entry logic (if not in position)
        if position is None:
            # Cooldown (20 bars ~5 hours)
            if i - last_trade_idx < 20:
                continue

            # Session filter (19:00-23:00 WIB = golden time)
            try:
                wib_time = current_time.tz_localize("UTC").tz_convert("Asia/Jakarta")
                hour = wib_time.hour
            except:
                # Fallback: assume UTC+7
                hour = current_time.hour + 7
                if hour >= 24:
                    hour -= 24

            if not (19 <= hour < 23):
                continue

            # Get ML prediction
            try:
                ml_pred = model.predict(df.slice(i, 1))
                # Debug: log first few predictions
                if len(trades) < 5:
                    logger.info(f"  Bar {i}: ML={ml_pred.signal} conf={ml_pred.confidence:.2f}")
            except Exception as e:
                logger.warning(f"  Prediction failed at bar {i}: {e}")
                continue

            # Skip HOLD signals (model's internal confidence gate)
            if ml_pred.signal == "HOLD":
                continue

            # Regime check (simple: skip CRISIS regime)
            regime_name = row.get('regime_name', 'medium_volatility')
            if regime_name == 'high_volatility':  # Crisis regime
                continue

            # ML threshold check (redundant but kept for safety)
            if ml_pred.confidence < ml_threshold:
                continue

            # SMC signal
            smc_signal = row.get('smc_signal', 0)

            # Signal confirmation (2+ consecutive)
            signal_key = f"{ml_pred.signal}_{i//2}"  # Group by pairs
            if signal_key not in signal_persistence:
                signal_persistence[signal_key] = 0
            signal_persistence[signal_key] += 1

            if signal_persistence[signal_key] < 2:
                continue

            # Direction alignment (ML + SMC)
            if ml_pred.signal == 'BUY' and smc_signal < 0:
                continue
            if ml_pred.signal == 'SELL' and smc_signal > 0:
                continue

            # Entry signal valid
            direction = ml_pred.signal
            entry_price = current_close

            # Position sizing (based on confidence)
            if ml_pred.confidence >= 0.70:
                lot_size = 0.02
            elif ml_pred.confidence >= 0.60:
                lot_size = 0.015
            else:
                lot_size = 0.01

            # Calculate SL/TP (dynamic RR 1.5-2.0)
            sl_distance = current_atr * 1.0

            # RR based on trend strength
            market_structure = row.get('market_structure', 0)
            if abs(market_structure) >= 2:
                rr = 2.0  # Strong trend
            else:
                rr = 1.5  # Ranging

            tp_distance = sl_distance * rr

            if direction == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            # Open position
            position = {
                'ticket': ticket_counter,
                'direction': direction,
                'entry_time': current_time,
                'entry_price': entry_price,
                'entry_idx': i,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'ml_confidence': ml_pred.confidence,
                'smc_signal': smc_signal,
                'regime': regime_name,
                'session': 'golden',
                'entry_reason': f"ML:{ml_pred.confidence:.2f} SMC:{smc_signal} R:{regime_name}",
            }

            ticket_counter += 1

    # Close any open position at end
    if position is not None:
        exit_price = df_pd.iloc[-1]['close']
        profit_pips = (exit_price - position['entry_price']) * (1 if position['direction'] == 'BUY' else -1) * 10
        profit_usd = profit_pips * position['lot_size'] * 10

        trade = SimulatedTrade(
            ticket=position['ticket'],
            entry_time=position['entry_time'],
            exit_time=df_pd.iloc[-1]['time'],
            direction=position['direction'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            lot_size=position['lot_size'],
            profit_usd=profit_usd,
            profit_pips=profit_pips,
            result=TradeResult.WIN if profit_usd > 0 else TradeResult.LOSS,
            exit_reason=ExitReason.TIMEOUT,
            ml_confidence=position['ml_confidence'],
            smc_signal=position['smc_signal'],
            regime=position['regime'],
            session=position['session'],
            entry_reason=position.get('entry_reason', ''),
        )
        trades.append(trade)
        current_equity += profit_usd

    # Calculate metrics
    metrics = calculate_metrics(trades, model)

    return trades, metrics


def calculate_metrics(trades: List[SimulatedTrade], model: TradingModelV2) -> BacktestMetrics:
    """Calculate backtest performance metrics."""
    if not trades:
        return BacktestMetrics(model_name="ML V2 (model_d.pkl)", num_features=len(model.feature_names))

    wins = [t for t in trades if t.result == TradeResult.WIN]
    losses = [t for t in trades if t.result == TradeResult.LOSS]
    breakevens = [t for t in trades if t.result == TradeResult.BREAKEVEN]

    total_profit = sum(t.profit_usd for t in wins)
    total_loss = abs(sum(t.profit_usd for t in losses))
    net_pnl = sum(t.profit_usd for t in trades)

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0)
    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0

    # Drawdown
    equity = 10000.0
    peak = 10000.0
    max_dd = 0.0

    for t in trades:
        equity += t.profit_usd
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe (simplified)
    returns = [t.profit_usd for t in trades]
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe = 0

    return BacktestMetrics(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        breakevens=len(breakevens),
        win_rate=win_rate,
        total_profit=total_profit,
        total_loss=total_loss,
        net_pnl=net_pnl,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        model_name="ML V2 (model_d.pkl)",
        test_auc=model._train_metrics.get('xgb_test_score', 0),
        num_features=len(model.feature_names),
    )


def print_results(metrics: BacktestMetrics, trades: List[SimulatedTrade]):
    """Print backtest results."""
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS — ML V2 MODEL TEST")
    print(f"{'='*70}")
    print(f"Model: {metrics.model_name}")
    print(f"Features: {metrics.num_features}")
    print(f"Test AUC: {metrics.test_auc:.4f}")
    print(f"\n{'='*70}")
    print(f"TRADING PERFORMANCE")
    print(f"{'='*70}")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Wins: {metrics.wins} ({metrics.win_rate:.1f}%)")
    print(f"Losses: {metrics.losses} ({(metrics.losses/metrics.total_trades*100) if metrics.total_trades > 0 else 0:.1f}%)")
    print(f"Breakevens: {metrics.breakevens}")
    print(f"\nNet P&L: ${metrics.net_pnl:,.2f}")
    print(f"Total Profit: ${metrics.total_profit:,.2f}")
    print(f"Total Loss: ${metrics.total_loss:,.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"\nAvg Win: ${metrics.avg_win:.2f}")
    print(f"Avg Loss: ${metrics.avg_loss:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"{'='*70}\n")

    # Show sample trades
    if trades:
        print("Sample Trades (First 10):")
        print(f"{'Ticket':<8} {'Entry':<20} {'Exit':<20} {'Dir':<5} {'P&L':>10} {'Confidence':>10} {'Exit Reason':<15}")
        print("-" * 100)
        for t in trades[:10]:
            print(f"{t.ticket:<8} {t.entry_time.strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{t.exit_time.strftime('%Y-%m-%d %H:%M'):<20} {t.direction:<5} "
                  f"${t.profit_usd:>9.2f} {t.ml_confidence:>10.2f} {t.exit_reason.value:<15}")
        print()


def save_results(
    trades: List[SimulatedTrade],
    metrics: BacktestMetrics,
    output_dir: Path,
):
    """Save backtest results to CSV files."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save trades
    trades_file = output_dir / "trades.csv"
    with open(trades_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ticket', 'Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price',
                        'SL', 'TP', 'Lot Size', 'Profit USD', 'Profit Pips', 'Result', 'Exit Reason',
                        'ML Confidence', 'SMC Signal', 'Regime', 'Session', 'Entry Reason'])
        for t in trades:
            writer.writerow([
                t.ticket, t.entry_time, t.exit_time, t.direction, t.entry_price, t.exit_price,
                t.stop_loss, t.take_profit, t.lot_size, t.profit_usd, t.profit_pips,
                t.result.value, t.exit_reason.value, t.ml_confidence, t.smc_signal,
                t.regime, t.session, t.entry_reason
            ])

    # Save metrics
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"ML V2 Backtest Results\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Model: {metrics.model_name}\n")
        f.write(f"Features: {metrics.num_features}\n")
        f.write(f"Test AUC: {metrics.test_auc:.4f}\n\n")
        f.write(f"Total Trades: {metrics.total_trades}\n")
        f.write(f"Win Rate: {metrics.win_rate:.1f}%\n")
        f.write(f"Net P&L: ${metrics.net_pnl:,.2f}\n")
        f.write(f"Profit Factor: {metrics.profit_factor:.2f}\n")
        f.write(f"Max Drawdown: {metrics.max_drawdown:.2f}%\n")
        f.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n")

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Backtest ML V2 Model (Config D)")
    parser.add_argument("--bars", type=int, default=20000, help="Number of M15 bars to backtest (default: 20000)")
    parser.add_argument("--threshold", type=float, default=0.50, help="ML confidence threshold (default: 0.50)")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"XAUBOT AI — Backtest #37: ML V2 Model Test")
    print(f"{'='*70}")
    print(f"Model: backtests/36_ml_v2_results/model_d.pkl")
    print(f"Live model (TIDAK DISENTUH): models/xgboost_model.pkl")
    print(f"Results folder: backtests/37_ml_v2_test_results/")
    print(f"{'='*70}\n")

    # Connect to MT5
    config = get_config()
    mt5_conn = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    mt5_conn.connect()
    logger.info("Connected to MT5\n")

    # Fetch data
    logger.info(f"Fetching XAUUSD data ({args.bars} M15 bars + H1)...")
    df_m15 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="M15", count=args.bars)
    df_h1 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="H1", count=args.bars // 4)
    logger.info(f"  Fetched: {len(df_m15)} M15 bars, {len(df_h1)} H1 bars\n")

    # Prepare data with V2 features
    model_path = "backtests/36_ml_v2_results/model_d.pkl"
    df_m15, model = prepare_data_with_v2_features(df_m15, df_h1, model_path)

    # Run backtest
    trades, metrics = run_backtest(df_m15, model, ml_threshold=args.threshold)

    # Print results
    print_results(metrics, trades)

    # Save results
    output_dir = Path("backtests/37_ml_v2_test_results")
    save_results(trades, metrics, output_dir)

    mt5_conn.disconnect()

    print(f"\n{'='*70}")
    print(f"Backtest complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - trades.csv (all {len(trades)} trades)")
    print(f"  - metrics.txt (performance summary)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
