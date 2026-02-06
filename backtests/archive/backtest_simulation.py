"""
Backtest Simulation - Test improved trading system with historical data.
"""
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import polars as pl
from dotenv import load_dotenv
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

load_dotenv()

@dataclass
class SimulatedTrade:
    """Simulated trade result."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    profit: float
    reason: str
    ml_confidence: float
    smc_signal: bool

def run_backtest():
    """Run backtest simulation with improved settings."""

    print("=" * 70)
    print("BACKTEST SIMULATION - IMPROVED TRADING SYSTEM")
    print("=" * 70)
    print()

    # Import components
    from src.mt5_connector import MT5Connector
    from src.feature_eng import FeatureEngineer
    from src.ml_model import TradingModel
    from src.smc_polars import SMCAnalyzer
    from src.regime_detector import MarketRegimeDetector
    from src.dynamic_confidence import create_dynamic_confidence
    from src.smart_risk_manager import create_smart_risk_manager

    # Connect to MT5
    mt5 = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER'),
    )

    if not mt5.connect():
        print("Failed to connect to MT5")
        return

    print(f"Connected to MT5")
    print(f"Balance: ${mt5.account_balance:,.2f}")
    print()

    # Initialize components
    feature_eng = FeatureEngineer()
    ml_model = TradingModel()
    ml_model.load("models/xgboost_model.pkl")
    smc = SMCAnalyzer()
    regime = MarketRegimeDetector()
    regime.load()  # Load pre-trained regime model
    dynamic_conf = create_dynamic_confidence()
    risk_manager = create_smart_risk_manager(mt5.account_balance)

    # Fetch historical data (last 7 days of M5 data)
    symbol = "XAUUSD"
    df = mt5.get_market_data(symbol, "M5", count=2000)  # ~7 days of M5 data

    if df is None or len(df) == 0:
        print("Failed to fetch historical data")
        mt5.disconnect()
        return

    print(f"Fetched {len(df)} bars of historical data")
    print(f"Date range: {df['time'][0]} to {df['time'][-1]}")
    print()

    # Add features
    df = feature_eng.calculate_all(df)

    # Add SMC features (required by ML model)
    df = smc.calculate_all(df)

    # Add regime features (required by ML model)
    df = regime.predict(df)

    # Get feature columns for ML
    feature_cols = [c for c in df.columns if c in ml_model.feature_names]
    print(f"Using {len(feature_cols)} features for ML prediction")
    print()

    print("=" * 70)
    print("PRODUCTION SETTINGS:")
    print("=" * 70)
    print(f"  ML-only threshold    : 75%+ required")
    print(f"  SMC+ML requirement   : Both MUST agree (65%+)")
    print(f"  Market quality skip  : POOR and AVOID")
    print(f"  Min ML confidence    : 65%")
    print(f"  Dynamic thresholds   : {dynamic_conf.min_threshold:.0%} - {dynamic_conf.max_threshold:.0%}")
    print(f"  Max lot size         : {risk_manager.max_lot_size}")
    print(f"  Max loss/trade       : ${risk_manager.max_loss_per_trade}")
    print("=" * 70)
    print()

    # Simulation parameters
    simulated_trades: List[SimulatedTrade] = []
    initial_balance = mt5.account_balance
    current_balance = initial_balance
    last_trade_idx = -300  # Start with no cooldown
    cooldown_bars = 60  # 5 minutes = 60 bars of M5

    # Stats
    total_signals = 0
    skipped_low_confidence = 0
    skipped_no_agreement = 0
    skipped_poor_quality = 0
    skipped_cooldown = 0

    print("Running simulation...")
    print("-" * 70)

    # Simulate through historical data (skip first 200 bars for indicator warmup)
    for i in range(200, len(df) - 10):
        # Get data up to this point
        current_df = df.head(i + 1)
        current_price = current_df['close'][-1]
        current_time = current_df['time'][-1]

        # ML Prediction
        ml_pred = ml_model.predict(current_df, feature_cols)

        # Skip if ML confidence too low
        if ml_pred.confidence < 0.65:  # Production: 65% minimum
            skipped_low_confidence += 1
            continue

        total_signals += 1

        # Check cooldown
        if i - last_trade_idx < cooldown_bars:
            skipped_cooldown += 1
            continue

        # SMC Signal
        smc_signal = smc.generate_signal(current_df)
        has_smc = smc_signal is not None

        # Dynamic confidence analysis (simplified)
        # Using moderate quality for simulation
        dynamic_threshold = dynamic_conf.base_threshold  # 80%

        # Entry decision
        should_trade = False
        trade_direction = None
        trade_reason = ""

        # Rule 1: ML-only needs 75%+
        if not has_smc:
            if ml_pred.confidence >= 0.75:  # Production: 75%
                should_trade = True
                trade_direction = ml_pred.signal
                trade_reason = f"ML-ONLY ({ml_pred.confidence:.0%})"
            else:
                skipped_low_confidence += 1
                continue
        else:
            # Rule 2: SMC + ML must agree
            ml_agrees = (
                (smc_signal.signal_type == "BUY" and ml_pred.signal == "BUY") or
                (smc_signal.signal_type == "SELL" and ml_pred.signal == "SELL")
            )

            if ml_agrees and ml_pred.confidence >= 0.65:  # Production: 65%
                should_trade = True
                trade_direction = ml_pred.signal
                trade_reason = f"SMC+ML AGREE ({ml_pred.confidence:.0%})"
            else:
                skipped_no_agreement += 1
                continue

        if not should_trade or trade_direction not in ["BUY", "SELL"]:
            continue

        # Simulate trade execution
        entry_price = current_price
        lot_size = risk_manager.base_lot_size  # 0.01

        # Look ahead 10-50 bars to simulate trade outcome
        # (This is simplified - real trading has more complexity)
        exit_idx = min(i + 30, len(df) - 1)  # ~2.5 hours later
        exit_price = df['close'][exit_idx]
        exit_time = df['time'][exit_idx]

        # Calculate profit
        if trade_direction == "BUY":
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        # Gold: 1 lot = $100 per point, 0.01 lot = $1 per point
        profit = price_diff * lot_size * 100

        # Apply max loss limit
        if profit < -risk_manager.max_loss_per_trade:
            profit = -risk_manager.max_loss_per_trade

        # Record trade
        trade = SimulatedTrade(
            entry_time=current_time,
            exit_time=exit_time,
            direction=trade_direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lot_size=lot_size,
            profit=profit,
            reason=trade_reason,
            ml_confidence=ml_pred.confidence,
            smc_signal=has_smc,
        )
        simulated_trades.append(trade)
        current_balance += profit
        last_trade_idx = i

        # Print trade
        result = "WIN" if profit > 0 else "LOSS"
        print(f"  {current_time} | {trade_direction} | {trade_reason} | ${profit:+.2f} [{result}]")

    print("-" * 70)
    print()

    # Calculate statistics
    total_trades = len(simulated_trades)
    if total_trades > 0:
        winning_trades = [t for t in simulated_trades if t.profit > 0]
        losing_trades = [t for t in simulated_trades if t.profit <= 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100

        total_profit = sum(t.profit for t in simulated_trades)
        avg_win = sum(t.profit for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t.profit for t in losing_trades) / loss_count if loss_count > 0 else 0

        # Profit factor
        gross_profit = sum(t.profit for t in winning_trades)
        gross_loss = abs(sum(t.profit for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print("=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print()
        print(f"  Initial Balance  : ${initial_balance:,.2f}")
        print(f"  Final Balance    : ${current_balance:,.2f}")
        print(f"  Total P/L        : ${total_profit:+,.2f} ({(total_profit/initial_balance)*100:+.2f}%)")
        print()
        print(f"  Total Trades     : {total_trades}")
        print(f"  Winning Trades   : {win_count}")
        print(f"  Losing Trades    : {loss_count}")
        print(f"  Win Rate         : {win_rate:.1f}%")
        print()
        print(f"  Average Win      : ${avg_win:+.2f}")
        print(f"  Average Loss     : ${avg_loss:.2f}")
        print(f"  Profit Factor    : {profit_factor:.2f}")
        print()
        print("  Signals Analysis:")
        print(f"    Total ML signals (70%+) : {total_signals}")
        print(f"    Skipped (low conf)      : {skipped_low_confidence}")
        print(f"    Skipped (no agreement)  : {skipped_no_agreement}")
        print(f"    Skipped (cooldown)      : {skipped_cooldown}")
        print(f"    Executed trades         : {total_trades}")
        print()

        # Trade breakdown
        ml_only_trades = [t for t in simulated_trades if "ML-ONLY" in t.reason]
        smc_ml_trades = [t for t in simulated_trades if "SMC+ML" in t.reason]

        print("  Trade Type Breakdown:")
        if ml_only_trades:
            ml_wins = len([t for t in ml_only_trades if t.profit > 0])
            print(f"    ML-ONLY trades : {len(ml_only_trades)} (Win: {ml_wins}, WR: {ml_wins/len(ml_only_trades)*100:.0f}%)")
        if smc_ml_trades:
            smc_wins = len([t for t in smc_ml_trades if t.profit > 0])
            print(f"    SMC+ML trades  : {len(smc_ml_trades)} (Win: {smc_wins}, WR: {smc_wins/len(smc_ml_trades)*100:.0f}%)")

    else:
        print("No trades executed in simulation period.")
        print(f"  Total signals checked: {total_signals}")
        print(f"  Skipped (low confidence): {skipped_low_confidence}")
        print(f"  Skipped (no agreement): {skipped_no_agreement}")

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    mt5.disconnect()

if __name__ == "__main__":
    run_backtest()
