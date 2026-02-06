"""
Backtest Simulation - 1 Month Historical Data
=============================================
Simulasi sistem trading dengan data market real 1 bulan kebelakang.
"""
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    market_quality: str

def run_backtest_1month():
    """Run 1 month backtest simulation."""

    print("=" * 70)
    print("BACKTEST SIMULATION - 1 MONTH HISTORICAL DATA")
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
    from src.session_filter import SessionFilter

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
    regime.load()
    dynamic_conf = create_dynamic_confidence()
    risk_manager = create_smart_risk_manager(mt5.account_balance)
    session_filter = SessionFilter()

    # Fetch 1 month of M5 data (~8640 bars)
    # M5 = 5 minutes, 1 month = 30 days * 24 hours * 12 bars/hour = 8640
    symbol = "XAUUSD"
    print("Fetching 1 month of historical data...")
    df = mt5.get_market_data(symbol, "M5", count=9000)  # ~1 month of M5 data

    if df is None or len(df) == 0:
        print("Failed to fetch historical data")
        mt5.disconnect()
        return

    print(f"Fetched {len(df)} bars of historical data")
    print(f"Date range: {df['time'][0]} to {df['time'][-1]}")

    # Calculate date range
    start_date = df['time'][0]
    end_date = df['time'][-1]
    days_covered = (end_date - start_date).days
    print(f"Period covered: {days_covered} days")
    print()

    # Add all features
    print("Calculating features...")
    df = feature_eng.calculate_all(df)
    df = smc.calculate_all(df)
    df = regime.predict(df)

    # Get feature columns for ML
    feature_cols = [c for c in df.columns if c in ml_model.feature_names]
    print(f"Using {len(feature_cols)} features for ML prediction")
    print()

    print("=" * 70)
    print("IMPROVED SYSTEM SETTINGS:")
    print("=" * 70)
    print(f"  Min ML confidence    : 65%")
    print(f"  ML-only threshold    : 75%+")
    print(f"  SMC+ML requirement   : Both MUST agree (65%+)")
    print(f"  Session filter       : Only London, NY, Overlap")
    print(f"  Trade cooldown       : 60 bars (5 hours)")
    print(f"  Max lot size         : {risk_manager.max_lot_size}")
    print(f"  Max loss/trade       : ${risk_manager.max_loss_per_trade}")
    print("=" * 70)
    print()

    # Simulation parameters
    simulated_trades: List[SimulatedTrade] = []
    initial_balance = mt5.account_balance
    current_balance = initial_balance
    last_trade_idx = -100  # Start with no cooldown
    cooldown_bars = 60  # 5 hours cooldown (60 * 5min = 300min = 5h)

    # Stats
    total_signals = 0
    skipped_low_confidence = 0
    skipped_no_agreement = 0
    skipped_poor_quality = 0
    skipped_cooldown = 0
    skipped_session = 0
    skipped_wrong_direction = 0

    # Daily tracking
    daily_pnl = {}

    print("Running simulation...")
    print("-" * 70)

    # Simulate through historical data (skip first 300 bars for indicator warmup)
    for i in range(300, len(df) - 60):
        # Get data up to this point
        current_df = df.head(i + 1)
        current_price = current_df['close'][-1]
        current_time = current_df['time'][-1]
        current_date = current_time.date()

        # Initialize daily PnL tracking
        if current_date not in daily_pnl:
            daily_pnl[current_date] = 0

        # Check session (simplified - check hour)
        hour = current_time.hour
        # London: 14:00-22:00 WIB, NY: 19:00-04:00 WIB, Overlap: 19:00-22:00 WIB
        # In UTC: London 07:00-15:00, NY 12:00-21:00, Overlap 12:00-15:00
        is_good_session = (7 <= hour <= 21)  # Simplified: 07:00-21:00 UTC

        if not is_good_session:
            continue

        # ML Prediction
        ml_pred = ml_model.predict(current_df, feature_cols)

        # Skip if ML confidence too low (min 65%)
        if ml_pred.confidence < 0.65:
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

        # Get market quality (simplified)
        market_quality = "good"

        # Entry decision
        should_trade = False
        trade_direction = None
        trade_reason = ""

        # Rule 1: ML-only needs 75%+
        if not has_smc:
            if ml_pred.confidence >= 0.75:
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

            if ml_agrees and ml_pred.confidence >= 0.65:
                should_trade = True
                trade_direction = ml_pred.signal
                trade_reason = f"SMC+ML ({ml_pred.confidence:.0%})"
            else:
                skipped_no_agreement += 1
                continue

        if not should_trade or trade_direction not in ["BUY", "SELL"]:
            continue

        # Simulate trade execution
        entry_price = current_price
        lot_size = risk_manager.base_lot_size  # 0.01

        # Look ahead to find exit (simplified: 12-60 bars, ~1-5 hours)
        # Use ATR-based TP/SL
        atr = current_df['atr'][-1] if 'atr' in current_df.columns else current_price * 0.003

        tp_distance = atr * 2.0  # 2 ATR for TP
        sl_distance = atr * 1.5  # 1.5 ATR for SL

        if trade_direction == "BUY":
            tp_price = entry_price + tp_distance
            sl_price = entry_price - sl_distance
        else:
            tp_price = entry_price - tp_distance
            sl_price = entry_price + sl_distance

        # Simulate price movement over next 60 bars
        exit_price = entry_price
        exit_time = current_time
        exit_reason = "TIMEOUT"

        for j in range(1, min(61, len(df) - i)):
            future_high = df['high'][i + j]
            future_low = df['low'][i + j]
            future_time = df['time'][i + j]

            if trade_direction == "BUY":
                # Check SL first
                if future_low <= sl_price:
                    exit_price = sl_price
                    exit_time = future_time
                    exit_reason = "SL"
                    break
                # Check TP
                if future_high >= tp_price:
                    exit_price = tp_price
                    exit_time = future_time
                    exit_reason = "TP"
                    break
            else:  # SELL
                # Check SL first
                if future_high >= sl_price:
                    exit_price = sl_price
                    exit_time = future_time
                    exit_reason = "SL"
                    break
                # Check TP
                if future_low <= tp_price:
                    exit_price = tp_price
                    exit_time = future_time
                    exit_reason = "TP"
                    break

            exit_price = df['close'][i + j]
            exit_time = future_time

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
            market_quality=market_quality,
        )
        simulated_trades.append(trade)
        current_balance += profit
        last_trade_idx = i

        # Track daily PnL
        daily_pnl[current_date] = daily_pnl.get(current_date, 0) + profit

        # Print trade (limit output)
        if len(simulated_trades) <= 30 or len(simulated_trades) % 10 == 0:
            result = "WIN" if profit > 0 else "LOSS"
            print(f"  {current_time.strftime('%Y-%m-%d %H:%M')} | {trade_direction} | {trade_reason} | ${profit:+.2f} [{result}] ({exit_reason})")

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

        # Max drawdown
        running_balance = initial_balance
        peak_balance = initial_balance
        max_drawdown = 0
        max_drawdown_pct = 0

        for trade in simulated_trades:
            running_balance += trade.profit
            if running_balance > peak_balance:
                peak_balance = running_balance
            drawdown = peak_balance - running_balance
            drawdown_pct = (drawdown / peak_balance) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in simulated_trades:
            if trade.profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        print("=" * 70)
        print("BACKTEST RESULTS - 1 MONTH")
        print("=" * 70)
        print()
        print(f"  Period           : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days_covered} days)")
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
        print(f"  Max Drawdown     : ${max_drawdown:,.2f} ({max_drawdown_pct:.1f}%)")
        print(f"  Max Consec. Wins : {max_consecutive_wins}")
        print(f"  Max Consec. Loss : {max_consecutive_losses}")
        print()

        # Signals Analysis
        print("  Signals Analysis:")
        print(f"    Total ML signals (65%+) : {total_signals}")
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
            ml_profit = sum(t.profit for t in ml_only_trades)
            print(f"    ML-ONLY trades : {len(ml_only_trades)} (Win: {ml_wins}, WR: {ml_wins/len(ml_only_trades)*100:.0f}%, P/L: ${ml_profit:+.2f})")
        if smc_ml_trades:
            smc_wins = len([t for t in smc_ml_trades if t.profit > 0])
            smc_profit = sum(t.profit for t in smc_ml_trades)
            print(f"    SMC+ML trades  : {len(smc_ml_trades)} (Win: {smc_wins}, WR: {smc_wins/len(smc_ml_trades)*100:.0f}%, P/L: ${smc_profit:+.2f})")
        print()

        # Daily breakdown
        print("  Daily Performance (last 10 days with trades):")
        sorted_days = sorted(daily_pnl.items(), key=lambda x: x[0], reverse=True)
        days_with_trades = [(d, p) for d, p in sorted_days if p != 0][:10]
        for date, pnl in days_with_trades:
            result = "[+]" if pnl > 0 else "[-]"
            print(f"    {date} : ${pnl:+.2f} {result}")
        print()

        # Monthly projection
        trades_per_day = total_trades / days_covered if days_covered > 0 else 0
        profit_per_day = total_profit / days_covered if days_covered > 0 else 0
        monthly_projection = profit_per_day * 30

        print("  Projections:")
        print(f"    Avg trades/day    : {trades_per_day:.1f}")
        print(f"    Avg profit/day    : ${profit_per_day:+.2f}")
        print(f"    Monthly projection: ${monthly_projection:+.2f}")

    else:
        print("No trades executed in simulation period.")
        print(f"  Total signals checked: {total_signals}")
        print(f"  Skipped (low confidence): {skipped_low_confidence}")
        print(f"  Skipped (no agreement): {skipped_no_agreement}")
        print(f"  Skipped (cooldown): {skipped_cooldown}")

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    mt5.disconnect()

if __name__ == "__main__":
    run_backtest_1month()
