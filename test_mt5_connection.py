"""
Test MT5 Connection
===================
Quick test to verify MT5 connection and data retrieval.
"""

import os
import sys
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# Load environment
from dotenv import load_dotenv
load_dotenv()

def test_connection():
    """Test MT5 connection."""
    logger.info("=" * 60)
    logger.info("MT5 CONNECTION TEST")
    logger.info("=" * 60)
    
    # Check environment variables
    login = os.getenv("MT5_LOGIN")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")
    
    logger.info(f"Login: {login}")
    logger.info(f"Server: {server}")
    logger.info(f"Path: {path}")
    
    if not all([login, server]):
        logger.error("Missing MT5 credentials in .env file")
        return False
    
    # Try to import MetaTrader5
    try:
        import MetaTrader5 as mt5
        logger.info(f"MetaTrader5 version: {mt5.__version__}")
    except ImportError:
        logger.error("MetaTrader5 not installed!")
        logger.info("Install with: pip install MetaTrader5")
        return False
    
    # Try to connect
    logger.info("Attempting to connect...")
    
    init_kwargs = {
        "login": int(login),
        "password": os.getenv("MT5_PASSWORD"),
        "server": server,
    }
    
    if path and os.path.exists(path):
        init_kwargs["path"] = path
    
    if mt5.initialize(**init_kwargs):
        logger.info("CONNECTION SUCCESSFUL!")
        
        # Get account info
        account = mt5.account_info()
        if account:
            logger.info(f"Account: {account.login}")
            logger.info(f"Server: {account.server}")
            logger.info(f"Balance: ${account.balance:,.2f}")
            logger.info(f"Equity: ${account.equity:,.2f}")
            logger.info(f"Leverage: 1:{account.leverage}")
            logger.info(f"Currency: {account.currency}")
        
        # Get symbol info
        symbol = os.getenv("SYMBOL", "XAUUSD")
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            logger.info(f"\nSymbol: {symbol}")
            logger.info(f"  Digits: {symbol_info.digits}")
            logger.info(f"  Spread: {symbol_info.spread}")
            logger.info(f"  Min lot: {symbol_info.volume_min}")
            logger.info(f"  Max lot: {symbol_info.volume_max}")
        else:
            logger.warning(f"Symbol {symbol} not found")
        
        # Get tick data
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            logger.info(f"  Bid: {tick.bid}")
            logger.info(f"  Ask: {tick.ask}")
        
        # Get some bars
        logger.info("\nFetching M15 data...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 10)
        if rates is not None:
            logger.info(f"  Received {len(rates)} bars")
            logger.info(f"  Latest close: {rates[-1]['close']}")
        else:
            error = mt5.last_error()
            logger.error(f"  Failed to get data: {error}")
        
        mt5.shutdown()
        logger.info("\nMT5 connection test: PASSED")
        return True
    
    else:
        error = mt5.last_error()
        logger.error(f"CONNECTION FAILED: {error}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Is MT5 terminal running?")
        logger.info("  2. Is AutoTrading enabled? (Algo Trading button)")
        logger.info("  3. Check login/password/server")
        logger.info("  4. Check terminal path")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
