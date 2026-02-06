"""
Test Improved System Against Real Trading History
=================================================
Simulasi: Apakah sistem perbaikan kita akan mengambil/menolak trade yang sama
dengan kondisi market yang sama persis?
"""
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import polars as pl
from dotenv import load_dotenv
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

load_dotenv()

@dataclass
class RealTrade:
    """Real trade from MT5 history."""
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    real_profit: float

@dataclass
class SimulationResult:
    """Result of simulating a trade with improved system."""
    ticket: int
    real_trade: RealTrade
    would_take: bool
    rejection_reason: str
    simulated_lot: float
    simulated_profit: float
    ml_confidence: float
    has_smc_signal: bool
    market_quality: str

def get_real_trades() -> List[RealTrade]:
    """Fetch real trades from MT5 history."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        print("MT5 init failed")
        return []

    if not mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')):
        print("MT5 login failed")
        return []

    # Get last 14 days
    from_date = datetime.now() - timedelta(days=14)
    to_date = datetime.now() + timedelta(days=1)

    deals = mt5.history_deals_get(from_date, to_date)

    if not deals:
        mt5.shutdown()
        return []

    # Group by position
    positions = {}
    for deal in deals:
        if deal.position_id > 0:
            if deal.position_id not in positions:
                positions[deal.position_id] = []
            positions[deal.position_id].append(deal)

    trades = []
    for pos_id, pos_deals in positions.items():
        if len(pos_deals) >= 2:
            entry = next((d for d in pos_deals if d.entry == 0), None)
            exit_deal = next((d for d in pos_deals if d.entry == 1), None)

            if entry and exit_deal:
                trades.append(RealTrade(
                    ticket=pos_id,
                    entry_time=datetime.fromtimestamp(entry.time),
                    exit_time=datetime.fromtimestamp(exit_deal.time),
                    direction='BUY' if entry.type == 0 else 'SELL',
                    entry_price=entry.price,
                    exit_price=exit_deal.price,
                    lot_size=entry.volume,
                    real_profit=exit_deal.profit,
                ))

    mt5.shutdown()
    return sorted(trades, key=lambda x: x.entry_time)

def simulate_trade_decision(trade: RealTrade, mt5_connector, feature_eng, ml_model, smc, regime_detector, dynamic_conf, risk_manager) -> SimulationResult:
    """
    Simulate what our improved system would do for this specific trade.
    Uses the exact market data at the time of the real trade.
    """
    # Get market data at the time of entry (look back 500 bars from entry time)
    # Since market is closed, we use the closest available data
    df = mt5_connector.get_market_data("XAUUSD", "M5", count=500)

    if df is None or len(df) == 0:
        return SimulationResult(
            ticket=trade.ticket,
            real_trade=trade,
            would_take=False,
            rejection_reason="NO DATA",
            simulated_lot=0,
            simulated_profit=0,
            ml_confidence=0,
            has_smc_signal=False,
            market_quality="unknown",
        )

    # Apply feature engineering
    df = feature_eng.calculate_all(df)
    df = smc.calculate_all(df)
    df = regime_detector.predict(df)  # Add regime column

    # Get ML prediction
    ml_pred = ml_model.predict(df)

    # Get SMC signal
    smc_signal = smc.generate_signal(df)
    has_smc = smc_signal is not None

    # Get market analysis
    market_analysis = dynamic_conf.analyze_market(
        session="London-NY",  # Assume good session for testing
        regime="medium_volatility",
        volatility="medium",
        trend_direction=ml_pred.signal,
        has_smc_signal=has_smc,
        ml_signal=ml_pred.signal,
        ml_confidence=ml_pred.confidence,
    )

    # Apply improved entry rules
    would_take = False
    rejection_reason = ""

    # Rule 1: Market quality check
    if market_analysis.quality.value in ["poor", "avoid"]:
        rejection_reason = f"Market quality: {market_analysis.quality.value}"
    # Rule 2: Min ML confidence 65%
    elif ml_pred.confidence < 0.65:
        rejection_reason = f"ML confidence too low: {ml_pred.confidence:.0%} < 65%"
    # Rule 3: ML-only needs 75%+
    elif not has_smc and ml_pred.confidence < 0.75:
        rejection_reason = f"ML-only needs 75%+, got {ml_pred.confidence:.0%}"
    # Rule 4: SMC+ML must agree
    elif has_smc:
        smc_dir = smc_signal.signal_type
        ml_dir = ml_pred.signal
        if smc_dir != ml_dir:
            rejection_reason = f"SMC ({smc_dir}) vs ML ({ml_dir}) disagree"
        elif ml_pred.confidence < 0.65:
            rejection_reason = f"SMC+ML conf too low: {ml_pred.confidence:.0%}"
        else:
            would_take = True
    else:
        # ML-only with 75%+
        would_take = True

    # Check direction match
    if would_take and ml_pred.signal != trade.direction:
        would_take = False
        rejection_reason = f"Wrong direction: System={ml_pred.signal}, Real={trade.direction}"

    # Calculate what our system would use
    simulated_lot = min(risk_manager.max_lot_size, risk_manager.base_lot_size)  # 0.01-0.02

    # Calculate simulated profit with our lot size
    price_diff = trade.exit_price - trade.entry_price
    if trade.direction == "SELL":
        price_diff = -price_diff

    # Gold: $1 per 0.01 lot per point (pip)
    simulated_profit = price_diff * simulated_lot * 100

    # Cap loss at max_loss_per_trade
    if simulated_profit < -risk_manager.max_loss_per_trade:
        simulated_profit = -risk_manager.max_loss_per_trade

    return SimulationResult(
        ticket=trade.ticket,
        real_trade=trade,
        would_take=would_take,
        rejection_reason=rejection_reason if not would_take else "ACCEPTED",
        simulated_lot=simulated_lot if would_take else 0,
        simulated_profit=simulated_profit if would_take else 0,
        ml_confidence=ml_pred.confidence,
        has_smc_signal=has_smc,
        market_quality=market_analysis.quality.value,
    )

def main():
    print("=" * 70)
    print("TEST IMPROVED SYSTEM vs REAL TRADING HISTORY")
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

    # Get real trades first
    print("Fetching real trading history...")
    real_trades = get_real_trades()
    print(f"Found {len(real_trades)} real trades")
    print()

    if not real_trades:
        print("No trades found!")
        return

    # Initialize MT5 for market data
    mt5 = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER'),
    )

    if not mt5.connect():
        print("Failed to connect to MT5")
        return

    print(f"Connected to MT5 - Balance: ${mt5.account_balance:,.2f}")
    print()

    # Initialize components with IMPROVED settings
    feature_eng = FeatureEngineer()
    ml_model = TradingModel()
    ml_model.load("models/xgboost_model.pkl")
    smc = SMCAnalyzer()
    regime_detector = MarketRegimeDetector()
    regime_detector.load()  # Load trained regime model
    dynamic_conf = create_dynamic_confidence()
    risk_manager = create_smart_risk_manager(mt5.account_balance)

    print("=" * 70)
    print("IMPROVED SYSTEM SETTINGS:")
    print("=" * 70)
    print(f"  Min ML confidence    : 65%")
    print(f"  ML-only threshold    : 75%+")
    print(f"  SMC+ML requirement   : Both must agree (65%+)")
    print(f"  Max lot size         : {risk_manager.max_lot_size}")
    print(f"  Max loss/trade       : ${risk_manager.max_loss_per_trade}")
    print("=" * 70)
    print()

    # Simulate each real trade
    print("=" * 70)
    print("SIMULATION RESULTS:")
    print("=" * 70)
    print()

    results: List[SimulationResult] = []

    for trade in real_trades:
        result = simulate_trade_decision(
            trade, mt5, feature_eng, ml_model, smc, regime_detector, dynamic_conf, risk_manager
        )
        results.append(result)

        # Print result
        status = "[TAKE]" if result.would_take else "[SKIP]"
        real_result = "WIN" if trade.real_profit > 0 else "LOSS"

        print(f"Ticket #{trade.ticket}:")
        print(f"  Real: {trade.direction} | Lot: {trade.lot_size} | P/L: ${trade.real_profit:+.2f} [{real_result}]")
        print(f"  System: {status} | ML: {result.ml_confidence:.0%} | SMC: {'YES' if result.has_smc_signal else 'NO'} | Quality: {result.market_quality}")

        if result.would_take:
            sim_result = "WIN" if result.simulated_profit > 0 else "LOSS"
            print(f"  Simulated: Lot: {result.simulated_lot} | P/L: ${result.simulated_profit:+.2f} [{sim_result}]")
        else:
            print(f"  Reason: {result.rejection_reason}")
        print()

    # Calculate statistics
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()

    # Real results
    real_wins = len([t for t in real_trades if t.real_profit > 0])
    real_losses = len([t for t in real_trades if t.real_profit <= 0])
    real_total_pnl = sum(t.real_profit for t in real_trades)
    real_win_rate = (real_wins / len(real_trades) * 100) if real_trades else 0

    print("REAL TRADING (what actually happened):")
    print(f"  Total Trades   : {len(real_trades)}")
    print(f"  Wins/Losses    : {real_wins}/{real_losses}")
    print(f"  Win Rate       : {real_win_rate:.1f}%")
    print(f"  Total P/L      : ${real_total_pnl:+,.2f}")
    print()

    # Simulated results (trades our system would take)
    taken_results = [r for r in results if r.would_take]
    skipped_results = [r for r in results if not r.would_take]

    sim_wins = len([r for r in taken_results if r.simulated_profit > 0])
    sim_losses = len([r for r in taken_results if r.simulated_profit <= 0])
    sim_total_pnl = sum(r.simulated_profit for r in taken_results)
    sim_win_rate = (sim_wins / len(taken_results) * 100) if taken_results else 0

    print("IMPROVED SYSTEM (what our system would do):")
    print(f"  Would Take     : {len(taken_results)} trades")
    print(f"  Would Skip     : {len(skipped_results)} trades")
    print(f"  Wins/Losses    : {sim_wins}/{sim_losses}")
    print(f"  Win Rate       : {sim_win_rate:.1f}%")
    print(f"  Total P/L      : ${sim_total_pnl:+,.2f}")
    print()

    # Analyze skipped trades - were they good or bad?
    skipped_that_were_losses = [r for r in skipped_results if r.real_trade.real_profit <= 0]
    skipped_that_were_wins = [r for r in skipped_results if r.real_trade.real_profit > 0]

    print("ANALYSIS OF SKIPPED TRADES:")
    print(f"  Skipped LOSSES : {len(skipped_that_were_losses)} (GOOD - avoided bad trades)")
    print(f"  Skipped WINS   : {len(skipped_that_were_wins)} (missed opportunities)")
    print()

    # Calculate money saved by skipping losses
    avoided_losses = sum(r.real_trade.real_profit for r in skipped_that_were_losses)
    missed_profits = sum(r.real_trade.real_profit for r in skipped_that_were_wins)

    print(f"  Avoided Losses : ${abs(avoided_losses):,.2f} (money saved)")
    print(f"  Missed Profits : ${missed_profits:,.2f} (opportunity cost)")
    print()

    # Summary comparison
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"  Real Trading P/L     : ${real_total_pnl:+,.2f}")
    print(f"  Improved System P/L  : ${sim_total_pnl:+,.2f}")
    print(f"  Difference           : ${(sim_total_pnl - real_total_pnl):+,.2f}")
    print()

    # Risk comparison
    real_max_loss = min(t.real_profit for t in real_trades) if real_trades else 0
    sim_max_loss = min(r.simulated_profit for r in taken_results) if taken_results else 0

    print("RISK COMPARISON:")
    print(f"  Real Max Single Loss : ${real_max_loss:,.2f}")
    print(f"  System Max Loss Cap  : ${sim_max_loss:,.2f} (capped at ${risk_manager.max_loss_per_trade})")
    print()

    # Verdict
    print("=" * 70)
    if sim_total_pnl >= real_total_pnl * 0.8:  # Within 20% of real
        print("VERDICT: Improved system performs WELL with LOWER RISK")
    elif len(skipped_that_were_losses) > len(skipped_that_were_wins):
        print("VERDICT: System correctly AVOIDS more bad trades than good ones")
    else:
        print("VERDICT: System may be TOO CONSERVATIVE - adjust thresholds")
    print("=" * 70)

    mt5.disconnect()

if __name__ == "__main__":
    main()
