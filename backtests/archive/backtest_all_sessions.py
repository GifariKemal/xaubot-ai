"""
Backtest All Sessions - Test trading outside golden time
=========================================================
Menguji apakah sistem bisa profit di semua session dengan threshold lebih rendah.

Test scenarios:
1. Current settings (conservative)
2. Lower ML threshold (55% instead of 65%)
3. SMC-only mode (ignore ML threshold when SMC has signal)
"""

import os
import sys
sys.path.insert(0, 'src')

import polars as pl
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import our modules
from mt5_connector import MT5Connector
from feature_eng import FeatureEngineer
from smc_polars import SMCAnalyzer
from ml_model import TradingModel
from regime_detector import MarketRegimeDetector

@dataclass
class BacktestTrade:
    entry_time: datetime
    entry_price: float
    direction: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: str = ""
    session: str = ""
    ml_confidence: float = 0.0
    smc_signal: str = ""

@dataclass
class BacktestResult:
    scenario: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_pips: float
    profit_factor: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    trades: List[BacktestTrade]

def get_session_name(hour: int) -> str:
    """Get session name based on WIB hour."""
    if 4 <= hour < 6:
        return "Rollover (AVOID)"
    elif 6 <= hour < 15:
        return "Sydney-Tokyo"
    elif 15 <= hour < 16:
        return "Tokyo-London Overlap"
    elif 16 <= hour < 20:
        return "London"
    elif 20 <= hour < 24:
        return "London-NY Overlap (GOLDEN)"
    else:
        return "Off-Hours"

def run_backtest_scenario(
    df: pl.DataFrame,
    scenario_name: str,
    ml_threshold: float = 0.65,
    require_smc: bool = True,
    smc_only_mode: bool = False,  # Trade on SMC signal even if ML below threshold
    allowed_sessions: List[str] = None,  # None = all sessions
    lot_size: float = 0.01,
    take_profit_pips: float = 150,  # $15 for 0.01 lot
    stop_loss_pips: float = 100,    # $10 for 0.01 lot
) -> BacktestResult:
    """Run backtest with specific parameters."""

    trades: List[BacktestTrade] = []
    position = None
    equity_curve = [10000.0]  # Start with $10k
    max_equity = 10000.0
    max_drawdown = 0.0

    # Convert to list for iteration
    rows = df.to_dicts()

    for i, row in enumerate(rows):
        if i < 50:  # Skip initial rows for indicator warmup
            continue

        current_time = row.get('time', datetime.now())
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)

        hour = current_time.hour
        session = get_session_name(hour)

        # Skip if session not allowed
        if allowed_sessions and session not in allowed_sessions:
            continue

        # Skip dangerous sessions
        if "AVOID" in session or "Off-Hours" in session:
            continue

        price = row.get('close', 0)
        ml_conf = row.get('ml_confidence', row.get('pred_prob_up', 0.5))
        if ml_conf is None:
            ml_conf = 0.5
        ml_signal = row.get('ml_signal', 'HOLD')

        # Determine SMC signal from components
        market_structure = row.get('market_structure', 0)
        bos = row.get('bos', 0)
        choch = row.get('choch', 0)
        fvg_bull = row.get('is_fvg_bull', False)
        fvg_bear = row.get('is_fvg_bear', False)
        ob = row.get('ob', 0)

        # Generate SMC signal
        smc_signal = "NONE"
        if market_structure == 1 and (bos == 1 or choch == 1) and fvg_bull:
            smc_signal = "BUY"
        elif market_structure == -1 and (bos == -1 or choch == -1) and fvg_bear:
            smc_signal = "SELL"

        # Determine ML direction from confidence
        if ml_conf > 0.5:
            ml_direction = "BUY"
            ml_conf_adj = ml_conf
        else:
            ml_direction = "SELL"
            ml_conf_adj = 1 - ml_conf

        # Check for exit if in position
        if position:
            pnl_pips = 0
            if position.direction == "BUY":
                pnl_pips = (price - position.entry_price) * 10  # XAUUSD: $1 = 10 pips
            else:
                pnl_pips = (position.entry_price - price) * 10

            # Check exit conditions
            exit_reason = None
            if pnl_pips >= take_profit_pips:
                exit_reason = "Take Profit"
            elif pnl_pips <= -stop_loss_pips:
                exit_reason = "Stop Loss"
            elif i >= len(rows) - 1:
                exit_reason = "End of Data"
            # Exit on reversal signal
            elif smc_signal != "NONE" and smc_signal != position.direction:
                exit_reason = f"Reversal ({smc_signal})"

            if exit_reason:
                pnl_usd = pnl_pips * lot_size  # $1 per pip for 0.01 lot
                position.exit_time = current_time
                position.exit_price = price
                position.pnl = pnl_usd
                position.pnl_pips = pnl_pips
                position.exit_reason = exit_reason
                trades.append(position)

                equity_curve.append(equity_curve[-1] + pnl_usd)
                max_equity = max(max_equity, equity_curve[-1])
                drawdown = (max_equity - equity_curve[-1]) / max_equity * 100
                max_drawdown = max(max_drawdown, drawdown)

                position = None
                continue

        # Check for entry if no position
        if not position:
            should_enter = False
            direction = None

            if smc_only_mode:
                # SMC-only: Enter when SMC has signal, ML just confirms direction
                if smc_signal in ["BUY", "SELL"]:
                    should_enter = True
                    direction = smc_signal
            else:
                # Normal mode: Need both SMC and ML agreement
                if require_smc:
                    if smc_signal in ["BUY", "SELL"] and ml_conf_adj >= ml_threshold:
                        if smc_signal == ml_direction:
                            should_enter = True
                            direction = smc_signal
                else:
                    # ML-only mode
                    if ml_conf_adj >= ml_threshold:
                        should_enter = True
                        direction = ml_direction

            if should_enter and direction:
                position = BacktestTrade(
                    entry_time=current_time,
                    entry_price=price,
                    direction=direction,
                    session=session,
                    ml_confidence=ml_conf_adj,
                    smc_signal=smc_signal,
                )

    # Calculate results
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_wins = sum(t.pnl for t in wins)
    total_losses = abs(sum(t.pnl for t in losses))

    return BacktestResult(
        scenario=scenario_name,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades) * 100 if trades else 0,
        total_pnl=sum(t.pnl for t in trades),
        total_pips=sum(t.pnl_pips for t in trades),
        profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
        max_drawdown=max_drawdown,
        avg_win=total_wins / len(wins) if wins else 0,
        avg_loss=total_losses / len(losses) if losses else 0,
        trades=trades,
    )

def main():
    print("=" * 70)
    print("BACKTEST ALL SESSIONS - Testing Non-Golden Time Trading")
    print("=" * 70)

    # Connect to MT5
    mt5 = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER'),
    )
    if not mt5.connect():
        print("Failed to connect to MT5")
        return

    print(f"\nConnected to MT5")
    print(f"Balance: ${mt5.account_balance:,.2f}")

    # Get historical data (2 weeks for more data)
    print("\nFetching historical data (14 days M15)...")
    df = mt5.get_market_data("XAUUSD", "M15", count=14 * 24 * 4)  # 14 days

    if df is None or len(df) == 0:
        print("Failed to get historical data")
        return

    print(f"Got {len(df)} candles")

    # Add features
    print("\nCalculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_all(df)

    # Add SMC signals
    print("Calculating SMC signals...")
    smc = SMCAnalyzer()
    df = smc.calculate_all(df)

    # Add ML predictions
    print("Loading ML model and predicting...")
    try:
        ml = TradingModel()
        ml.load("models/xgboost_model.pkl")
        df = ml.predict_batch(df)

        # Create ml_confidence column
        df = df.with_columns([
            pl.when(pl.col("pred_prob_up") > 0.5)
            .then(pl.col("pred_prob_up"))
            .otherwise(1 - pl.col("pred_prob_up"))
            .alias("ml_confidence")
        ])
    except Exception as e:
        print(f"ML model error: {e}")
        # Create dummy predictions
        df = df.with_columns([
            pl.lit(0.5).alias("pred_prob_up"),
            pl.lit(0.5).alias("ml_confidence"),
        ])

    print(f"\nData ready: {len(df)} rows")

    # Define test scenarios
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST SCENARIOS")
    print("=" * 70)

    scenarios = [
        # Scenario 1: Current conservative settings
        {
            "name": "1. Conservative (Current)",
            "ml_threshold": 0.65,
            "require_smc": True,
            "smc_only_mode": False,
            "allowed_sessions": None,  # All sessions
        },
        # Scenario 2: Lower threshold
        {
            "name": "2. Lower Threshold (55%)",
            "ml_threshold": 0.55,
            "require_smc": True,
            "smc_only_mode": False,
            "allowed_sessions": None,
        },
        # Scenario 3: SMC-only mode
        {
            "name": "3. SMC-Only (Ignore ML)",
            "ml_threshold": 0.50,
            "require_smc": True,
            "smc_only_mode": True,
            "allowed_sessions": None,
        },
        # Scenario 4: Golden time only
        {
            "name": "4. Golden Time Only",
            "ml_threshold": 0.60,
            "require_smc": True,
            "smc_only_mode": False,
            "allowed_sessions": ["London-NY Overlap (GOLDEN)"],
        },
        # Scenario 5: London + Golden
        {
            "name": "5. London + Golden",
            "ml_threshold": 0.60,
            "require_smc": True,
            "smc_only_mode": False,
            "allowed_sessions": ["London", "London-NY Overlap (GOLDEN)"],
        },
        # Scenario 6: All sessions with SMC-only
        {
            "name": "6. All Sessions SMC-Only",
            "ml_threshold": 0.50,
            "require_smc": True,
            "smc_only_mode": True,
            "allowed_sessions": ["Sydney-Tokyo", "Tokyo-London Overlap", "London", "London-NY Overlap (GOLDEN)"],
        },
        # Scenario 7: Very aggressive (50% threshold)
        {
            "name": "7. Aggressive (50% threshold)",
            "ml_threshold": 0.50,
            "require_smc": True,
            "smc_only_mode": False,
            "allowed_sessions": None,
        },
    ]

    results = []

    for scenario in scenarios:
        print(f"\nRunning: {scenario['name']}...")
        result = run_backtest_scenario(
            df=df,
            scenario_name=scenario["name"],
            ml_threshold=scenario["ml_threshold"],
            require_smc=scenario["require_smc"],
            smc_only_mode=scenario["smc_only_mode"],
            allowed_sessions=scenario["allowed_sessions"],
        )
        results.append(result)

        # Print quick summary
        print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%, PnL: ${result.total_pnl:.2f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Scenario':<35} {'Trades':>7} {'WinRate':>8} {'PnL':>10} {'PF':>6} {'MaxDD':>7}")
    print("-" * 70)

    for r in results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "INF"
        print(f"{r.scenario:<35} {r.total_trades:>7} {r.win_rate:>7.1f}% ${r.total_pnl:>8.2f} {pf_str:>6} {r.max_drawdown:>6.1f}%")

    print("-" * 70)

    # Find best scenario
    valid_results = [r for r in results if r.total_trades >= 5]
    if valid_results:
        best_pnl = max(valid_results, key=lambda x: x.total_pnl)
        best_wr = max(valid_results, key=lambda x: x.win_rate)

        print(f"\nBEST BY PnL: {best_pnl.scenario}")
        print(f"  ${best_pnl.total_pnl:.2f} profit, {best_pnl.win_rate:.1f}% win rate")

        print(f"\nBEST BY WIN RATE: {best_wr.scenario}")
        print(f"  {best_wr.win_rate:.1f}% win rate, ${best_wr.total_pnl:.2f} profit")

    # Detailed analysis of best scenario
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if valid_results:
        # Find balanced best (high PnL + reasonable win rate)
        scored = [(r, r.total_pnl * (r.win_rate / 100)) for r in valid_results if r.win_rate >= 40]
        if scored:
            best = max(scored, key=lambda x: x[1])[0]
            print(f"\nRECOMMENDED SCENARIO: {best.scenario}")
            print(f"  - Trades: {best.total_trades}")
            print(f"  - Win Rate: {best.win_rate:.1f}%")
            print(f"  - Total PnL: ${best.total_pnl:.2f}")
            print(f"  - Profit Factor: {best.profit_factor:.2f}")
            print(f"  - Max Drawdown: {best.max_drawdown:.1f}%")

            # Session breakdown
            print(f"\n  Session Breakdown:")
            session_stats = {}
            for t in best.trades:
                if t.session not in session_stats:
                    session_stats[t.session] = {"trades": 0, "wins": 0, "pnl": 0}
                session_stats[t.session]["trades"] += 1
                session_stats[t.session]["wins"] += 1 if t.pnl > 0 else 0
                session_stats[t.session]["pnl"] += t.pnl

            for session, stats in sorted(session_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                print(f"    {session}: {stats['trades']} trades, {wr:.0f}% WR, ${stats['pnl']:.2f}")

    print("\n" + "=" * 70)
    mt5.disconnect()

if __name__ == "__main__":
    main()
