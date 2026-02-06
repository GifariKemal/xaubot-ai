"""
Simulation Test - Test the improved trading system without real trades.
Uses real market data but only simulates decisions.
"""
import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

load_dotenv()

async def run_simulation():
    """Run simulation test with improved settings."""

    print("=" * 60)
    print("SIMULATION TEST - IMPROVED TRADING SYSTEM")
    print("=" * 60)
    print()

    # Import components
    from src.mt5_connector import MT5Connector
    from src.feature_eng import FeatureEngineer
    from src.ml_model import TradingModel
    from src.smc_polars import SMCAnalyzer, SMCSignal
    from src.regime_detector import MarketRegimeDetector
    from src.session_filter import SessionFilter
    from src.dynamic_confidence import create_dynamic_confidence
    from src.smart_risk_manager import create_smart_risk_manager

    # Initialize
    import os
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
    print(f"Equity: ${mt5.account_equity:,.2f}")
    print()

    # Components
    feature_eng = FeatureEngineer()
    ml_model = TradingModel()
    ml_model.load("models/xgboost_model.pkl")
    smc = SMCAnalyzer()
    regime = MarketRegimeDetector()
    regime.load()
    session_filter = SessionFilter()
    dynamic_conf = create_dynamic_confidence()
    risk_manager = create_smart_risk_manager(mt5.account_balance)

    print("=" * 60)
    print("IMPROVED SETTINGS:")
    print("=" * 60)
    print(f"  ML-only threshold: 85%+ required")
    print(f"  SMC+ML: Both MUST agree")
    print(f"  Market quality: Skip POOR and AVOID")
    print(f"  Min ML confidence: 70%")
    print(f"  Trade cooldown: 5 minutes")
    print(f"  Max lot: 0.02")
    print(f"  Max loss/trade: $30")
    print(f"  Max daily loss: 2%")
    print("=" * 60)
    print()

    # Fetch data
    symbol = "XAUUSD"
    df = mt5.get_market_data(symbol, "M5", count=500)
    if df is None or len(df) == 0:
        print("Failed to fetch data (market might be closed)")
        print("Using last available data...")
        df = mt5.get_market_data(symbol, "M5", count=500)
        if df is None or len(df) == 0:
            print("Still no data - market is closed")
            mt5.disconnect()
            return

    print(f"Fetched {len(df)} bars of {symbol} M5 data")
    print(f"Latest price: ${df['close'][-1]:,.2f}")
    print()

    # Feature engineering
    df = feature_eng.calculate_all(df)

    # Add SMC features (required by ML model)
    df = smc.calculate_all(df)

    # Regime detection
    df = regime.predict(df)  # Adds regime columns to df
    regime_state = regime.get_current_state(df)  # Get regime state object
    print(f"Current Regime: {regime_state.regime.value if regime_state else 'N/A'}")
    print(f"Recommendation: {regime_state.recommendation if regime_state else 'N/A'}")
    print()

    # Session check
    can_trade, reason, _ = session_filter.can_trade()
    session_info = session_filter.get_status_report()
    print(f"Session: {session_info.get('current_session', 'Unknown')}")
    print(f"Can Trade: {can_trade} - {reason}")
    print()

    # ML Prediction
    feature_cols = [c for c in df.columns if c in ml_model.feature_names]
    ml_pred = ml_model.predict(df, feature_cols)
    print(f"ML Prediction: {ml_pred.signal} ({ml_pred.confidence:.0%})")
    print()

    # SMC Signal
    smc_signal = smc.generate_signal(df)
    if smc_signal:
        print(f"SMC Signal: {smc_signal.signal_type} ({smc_signal.confidence:.0%})")
        print(f"  Entry: {smc_signal.entry_price:.2f}")
        print(f"  SL: {smc_signal.stop_loss:.2f}")
        print(f"  TP: {smc_signal.take_profit:.2f}")
    else:
        print("SMC Signal: NONE")
    print()

    # Dynamic Confidence Analysis
    market_analysis = dynamic_conf.analyze_market(
        session=session_info.get('current_session', 'Unknown'),
        regime=regime_state.regime.value,
        volatility=session_info.get('volatility', 'medium'),
        trend_direction=regime_state.regime.value,
        has_smc_signal=(smc_signal is not None),
        ml_signal=ml_pred.signal,
        ml_confidence=ml_pred.confidence,
    )

    print("=" * 60)
    print("MARKET ANALYSIS:")
    print("=" * 60)
    print(f"  Quality: {market_analysis.quality.value.upper()}")
    print(f"  Score: {market_analysis.score}")
    print(f"  Threshold: {market_analysis.confidence_threshold:.0%}")
    print()
    for reason in market_analysis.reasons:
        print(f"  {reason}")
    print()

    # Entry Decision
    print("=" * 60)
    print("ENTRY DECISION (SIMULATION):")
    print("=" * 60)

    # Check conditions
    should_trade = False
    trade_reason = ""

    # 1. Market quality check
    if market_analysis.quality.value in ["poor", "avoid"]:
        trade_reason = f"SKIP: Market quality {market_analysis.quality.value}"
    # 2. ML confidence check
    elif ml_pred.confidence < 0.70:
        trade_reason = f"SKIP: ML confidence {ml_pred.confidence:.0%} < 70%"
    # 3. ML-only (no SMC)
    elif smc_signal is None:
        if ml_pred.confidence >= 0.85:
            should_trade = True
            trade_reason = f"TRADE (ML-ONLY): {ml_pred.signal} at {ml_pred.confidence:.0%}"
        else:
            trade_reason = f"SKIP: ML-only needs 85%+, got {ml_pred.confidence:.0%}"
    # 4. SMC + ML combination
    else:
        ml_agrees = (
            (smc_signal.signal_type == "BUY" and ml_pred.signal == "BUY") or
            (smc_signal.signal_type == "SELL" and ml_pred.signal == "SELL")
        )
        if ml_agrees:
            should_trade = True
            trade_reason = f"TRADE (SMC+ML): {smc_signal.signal_type} - Both agree!"
        else:
            trade_reason = f"SKIP: SMC={smc_signal.signal_type} vs ML={ml_pred.signal} - Disagree"

    print(f"  {trade_reason}")
    print()

    if should_trade:
        # Calculate lot size
        lot = risk_manager.calculate_lot_size(
            entry_price=df['close'][-1],
            confidence=ml_pred.confidence,
            regime=regime_state.regime.value,
        )
        print(f"  Simulated Trade:")
        print(f"    Direction: {ml_pred.signal}")
        print(f"    Lot Size: {lot}")
        print(f"    Entry: ${df['close'][-1]:,.2f}")
    else:
        print(f"  No trade - waiting for better conditions")

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    mt5.disconnect()

if __name__ == "__main__":
    asyncio.run(run_simulation())
