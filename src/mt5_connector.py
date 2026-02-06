"""
MetaTrader 5 Connector Module
=============================
Handles all communication with MT5 terminal.

CRITICAL: Data is converted to Polars DataFrame immediately after fetching.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    logger.warning("MetaTrader5 not installed. Running in simulation mode.")
    mt5 = None


@dataclass
class TickData:
    """Real-time tick data structure."""
    time: datetime
    bid: float
    ask: float
    last: float
    volume: float
    spread: float


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[int] = None
    retcode: Optional[int] = None
    comment: str = ""
    price: float = 0.0
    volume: float = 0.0


class MT5Connector:
    """
    MetaTrader 5 connection handler with Polars integration.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Direct conversion to Polars DataFrame
    - Order execution with retry logic
    - Real-time tick streaming
    """
    
    # MT5 Timeframe mapping
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
        "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
        "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
        "M30": mt5.TIMEFRAME_M30 if mt5 else 30,
        "H1": mt5.TIMEFRAME_H1 if mt5 else 16385,
        "H4": mt5.TIMEFRAME_H4 if mt5 else 16388,
        "D1": mt5.TIMEFRAME_D1 if mt5 else 16408,
        "W1": mt5.TIMEFRAME_W1 if mt5 else 32769,
    }
    
    # Trade return codes
    RETCODE_DONE = 10009
    RETCODE_REQUOTE = 10004
    RETCODE_REJECT = 10006
    RETCODE_INVALID = 10013
    RETCODE_INVALID_VOLUME = 10014
    RETCODE_INVALID_PRICE = 10015
    RETCODE_INVALID_STOPS = 10016
    RETCODE_TRADE_DISABLED = 10027

    # Connection error codes (for auto-reconnect detection)
    ERR_NO_IPC_CONNECTION = -10004    # No IPC connection to terminal
    ERR_NO_CONNECTION = -10003        # No connection to trade server
    ERR_TERMINAL_CALL_FAILED = -1     # Terminal call failed
    ERR_COMMON_ERROR = -10001         # Common error
    ERR_INVALID_PARAMS = -10002       # Invalid parameters

    # List of connection errors that trigger reconnect
    CONNECTION_ERRORS = [-10004, -10003, -1, -10001, -10002]
    
    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: Optional[str] = None,
        timeout: int = 60000,
    ):
        """
        Initialize MT5 connector.
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: Broker server name
            path: Path to MT5 terminal (optional)
            timeout: Connection timeout in ms
        """
        self.login = login
        self._password = password  # Prefixed with _ to indicate private
        self.server = server
        self.path = path
        self.timeout = timeout
        self._connected = False
        self._account_info: Optional[Dict] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._last_reconnect_time: Optional[datetime] = None
        
    def connect(self, max_retries: int = 3) -> bool:
        """
        Connect to MT5 terminal with retry logic.

        IMPROVED: Properly handles existing connections and ensures clean startup.

        Args:
            max_retries: Maximum connection attempts

        Returns:
            True if connected successfully
        """
        if mt5 is None:
            logger.warning("MT5 not available - simulation mode")
            return False

        for attempt in range(max_retries):
            try:
                # IMPORTANT: Shutdown any existing connection first
                try:
                    mt5.shutdown()
                    time.sleep(0.5)  # Brief pause after shutdown
                except Exception:
                    pass

                # Build initialization kwargs
                kwargs = {
                    "login": self.login,
                    "password": self._password,
                    "server": self.server,
                    "timeout": self.timeout,
                }
                if self.path:
                    kwargs["path"] = self.path

                if mt5.initialize(**kwargs):
                    # Wait for terminal to fully initialize
                    time.sleep(2)  # Increased delay for stability

                    # Verify connection by getting terminal info
                    terminal_info = mt5.terminal_info()
                    if terminal_info is None:
                        logger.warning("Terminal info not available, retrying...")
                        mt5.shutdown()
                        time.sleep(2)
                        continue

                    # Check if terminal is connected to trade server
                    if not terminal_info.connected:
                        logger.warning("Terminal not connected to trade server, waiting...")
                        time.sleep(3)
                        terminal_info = mt5.terminal_info()
                        if not terminal_info or not terminal_info.connected:
                            logger.warning("Still not connected, retrying...")
                            mt5.shutdown()
                            continue

                    self._connected = True
                    self._account_info = self._get_account_info()
                    logger.info(f"Connected to MT5: {self.server} (Account: {self.login})")

                    # Pre-select common symbols to ensure they're ready
                    mt5.symbol_select("XAUUSD", True)
                    time.sleep(0.5)

                    return True

                error = mt5.last_error()
                logger.warning(f"Connection attempt {attempt + 1} failed: {error}")

            except Exception as e:
                logger.error(f"Connection error: {e}")

            # Exponential backoff with longer delays
            wait_time = 2 ** (attempt + 1)
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

        raise ConnectionError(f"Failed to connect to MT5 after {max_retries} attempts")
    
    def disconnect(self):
        """Safely disconnect from MT5."""
        if self._connected and mt5:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5.

        IMPROVED: More robust reconnection with proper cleanup.

        Returns:
            True if reconnection successful
        """
        logger.warning("Attempting to reconnect to MT5...")

        # Full shutdown and cleanup
        if mt5:
            try:
                mt5.shutdown()
            except:
                pass

        self._connected = False
        self._reconnect_attempts += 1

        # Wait longer before reconnecting to let MT5 stabilize
        wait_time = min(5, 2 + self._reconnect_attempts)
        logger.info(f"Waiting {wait_time}s before reconnect (attempt {self._reconnect_attempts})...")
        time.sleep(wait_time)

        try:
            success = self.connect(max_retries=3)
            if success:
                self._reconnect_attempts = 0  # Reset on success
            return success
        except ConnectionError as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def ensure_connected(self) -> bool:
        """
        Ensure MT5 is connected, auto-reconnect if needed.

        Returns:
            True if connected (or reconnected successfully)
        """
        if not mt5:
            return False

        # If explicitly marked as disconnected, need to reconnect first
        if not self._connected:
            logger.debug("Connection flag is False, attempting reconnect...")
            return self.reconnect()

        # Check if actually connected by trying to get account info
        try:
            info = mt5.account_info()
            if info is not None:
                self._reconnect_attempts = 0  # Reset on success
                return True
        except:
            pass

        # Connection lost - attempt reconnect
        self._connected = False
        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._max_reconnect_attempts:
            # Cooldown period - wait 60 seconds before trying again
            if self._last_reconnect_time:
                elapsed = (datetime.now() - self._last_reconnect_time).total_seconds()
                if elapsed < 60:
                    return False
            self._reconnect_attempts = 0  # Reset after cooldown

        logger.warning(f"MT5 connection lost. Reconnect attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
        self._last_reconnect_time = datetime.now()

        if self.reconnect():
            logger.info("MT5 reconnected successfully!")
            self._reconnect_attempts = 0
            return True

        return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        if not mt5:
            return {}
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }
    
    @property
    def account_balance(self) -> float:
        """Get current account balance."""
        if mt5:
            info = mt5.account_info()
            return info.balance if info else 0.0
        return 0.0
    
    @property
    def account_equity(self) -> float:
        """Get current account equity."""
        if mt5:
            info = mt5.account_info()
            return info.equity if info else 0.0
        return 0.0
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str = "M15",
        count: int = 1000,
        max_retries: int = 3,
    ) -> pl.DataFrame:
        """
        Fetch market data and convert to Polars DataFrame.

        CRITICAL: This is the main data fetching function.
        Data is converted to Polars immediately - NO PANDAS.

        IMPROVED: Better retry logic and error handling.

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1)
            count: Number of bars to fetch
            max_retries: Maximum fetch attempts before giving up

        Returns:
            Polars DataFrame with columns:
            ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        """
        if not mt5:
            logger.warning("MT5 not available, returning empty DataFrame")
            return self._create_empty_dataframe()

        # Convert timeframe string to MT5 constant
        tf = self.TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        rates = None

        for attempt in range(max_retries):
            # Auto-reconnect if disconnected
            if not self.ensure_connected():
                logger.warning(f"Not connected to MT5, attempt {attempt + 1}/{max_retries}")
                time.sleep(2)
                continue

            # Ensure symbol is selected in Market Watch
            select_result = mt5.symbol_select(symbol, True)
            if not select_result:
                error = mt5.last_error()
                logger.warning(f"Failed to select symbol {symbol}, attempt {attempt + 1}: {error}")

                # Check if symbol exists at all
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol} not found in MT5 - check if symbol name is correct")

                # Symbol select failure often indicates connection issue - force reconnect
                self._connected = False
                if attempt >= 1:  # After 2 failed attempts, do full reconnect
                    logger.info("Symbol select failing repeatedly, forcing full reconnect...")
                    self.reconnect()
                else:
                    time.sleep(1)
                continue

            # Wait for symbol data to be ready
            time.sleep(0.2)

            # Fetch rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

            if rates is not None and len(rates) > 0:
                # Success!
                if attempt > 0:
                    logger.info(f"Data fetched successfully on attempt {attempt + 1}")
                break

            # Failed to get data
            error = mt5.last_error()
            error_code = error[0] if error else 0
            logger.warning(f"Failed to get market data (attempt {attempt + 1}/{max_retries}): {error}")

            # Check if error is connection-related
            if error_code in self.CONNECTION_ERRORS:
                self._connected = False
                # Force full reconnect
                logger.info("Connection error detected, forcing reconnect...")
                self.reconnect()
            else:
                # Non-connection error, wait and retry
                time.sleep(1)

        # Final check
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get market data for {symbol} after {max_retries} attempts")
            return self._create_empty_dataframe()
        
        # CRITICAL: Convert numpy structured array directly to Polars
        # This is the key optimization - no Pandas intermediate
        df = pl.DataFrame({
            "time": rates["time"],
            "open": rates["open"],
            "high": rates["high"],
            "low": rates["low"],
            "close": rates["close"],
            "tick_volume": rates["tick_volume"],
            "spread": rates["spread"],
            "real_volume": rates["real_volume"],
        })
        
        # Cast columns to correct types
        df = df.with_columns([
            # Convert Unix timestamp to datetime
            pl.from_epoch(pl.col("time"), time_unit="s").alias("time"),
            # Ensure price columns are Float64
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            # Rename tick_volume to volume for convenience
            pl.col("tick_volume").cast(pl.Int64).alias("volume"),
        ]).drop("tick_volume")
        
        logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df
    
    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str],
        count: int = 1000,
    ) -> Dict[str, pl.DataFrame]:
        """
        Fetch data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframe strings
            count: Number of bars per timeframe
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            data[tf] = self.get_market_data(symbol, tf, count)
        return data
    
    def get_tick(self, symbol: str) -> Optional[TickData]:
        """
        Get current tick data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TickData object or None
        """
        if not mt5:
            return None
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return TickData(
            time=datetime.fromtimestamp(tick.time),
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            spread=(tick.ask - tick.bid),
        )
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information."""
        if not mt5:
            return None
            
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            "name": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_tick_size": info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "spread": info.spread,
            "trade_mode": info.trade_mode,
        }
    
    def send_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        magic: int = 123456,
        comment: str = "AI Bot",
        max_retries: int = 3,
    ) -> OrderResult:
        """
        Send market order with retry logic.
        
        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Lot size
            price: Price (None for market order)
            sl: Stop loss price
            tp: Take profit price
            deviation: Maximum price deviation in points
            magic: Magic number for identification
            comment: Order comment
            max_retries: Maximum retry attempts
            
        Returns:
            OrderResult with execution details
        """
        if not mt5:
            return OrderResult(success=False, comment="MT5 not available")
        
        # Get current prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, comment="Failed to get tick data")
        
        # Determine order type and price
        if order_type.upper() == "BUY":
            mt5_type = mt5.ORDER_TYPE_BUY
            order_price = price or tick.ask
        else:
            mt5_type = mt5.ORDER_TYPE_SELL
            order_price = price or tick.bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_type,
            "price": float(order_price),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)
        
        # Execute with retry
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed (None): {error}")
                continue
            
            if result.retcode == self.RETCODE_DONE:
                logger.info(f"Order executed: {order_type} {volume} {symbol} @ {result.price}")
                return OrderResult(
                    success=True,
                    order_id=result.order,
                    retcode=result.retcode,
                    comment=result.comment,
                    price=result.price,
                    volume=result.volume,
                )
            
            # Non-retryable errors
            if result.retcode in [
                self.RETCODE_INVALID,
                self.RETCODE_INVALID_VOLUME,
                self.RETCODE_INVALID_PRICE,
                self.RETCODE_INVALID_STOPS,
            ]:
                return OrderResult(
                    success=False,
                    retcode=result.retcode,
                    comment=result.comment,
                )
            
            # AutoTrading disabled
            if result.retcode == self.RETCODE_TRADE_DISABLED:
                raise RuntimeError("AutoTrading is disabled in MT5 terminal")
            
            # Retryable errors (requote, reject)
            logger.warning(f"Order attempt {attempt + 1} failed: {result.retcode} - {result.comment}")
            time.sleep(0.5)
        
        return OrderResult(
            success=False,
            retcode=result.retcode if result else None,
            comment=result.comment if result else "Max retries exceeded",
        )
    
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        deviation: int = 20,
        magic: int = 123456,
    ) -> OrderResult:
        """
        Close an open position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None for full close)
            deviation: Maximum price deviation
            magic: Magic number
            
        Returns:
            OrderResult with execution details
        """
        if not mt5:
            return OrderResult(success=False, comment="MT5 not available")
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(success=False, comment="Position not found")
        
        position = position[0]
        symbol = position.symbol
        pos_volume = volume or position.volume
        
        # Determine close direction
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos_volume),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": "AI Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == self.RETCODE_DONE:
            logger.info(f"Position {ticket} closed")
            return OrderResult(
                success=True,
                order_id=result.order,
                retcode=result.retcode,
                comment=result.comment,
                price=result.price,
                volume=result.volume,
            )
        
        return OrderResult(
            success=False,
            retcode=result.retcode if result else None,
            comment=result.comment if result else "Close failed",
        )
    
    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get open positions as Polars DataFrame.
        
        Args:
            symbol: Filter by symbol (optional)
            magic: Filter by magic number (optional)
            
        Returns:
            DataFrame with position details
        """
        if not mt5:
            return pl.DataFrame()
        
        # Get positions
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return pl.DataFrame({
                "ticket": [],
                "symbol": [],
                "type": [],
                "volume": [],
                "price_open": [],
                "sl": [],
                "tp": [],
                "profit": [],
                "magic": [],
            })
        
        # Convert to Polars
        data = {
            "ticket": [p.ticket for p in positions],
            "symbol": [p.symbol for p in positions],
            "type": ["BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL" for p in positions],
            "volume": [p.volume for p in positions],
            "price_open": [p.price_open for p in positions],
            "sl": [p.sl for p in positions],
            "tp": [p.tp for p in positions],
            "profit": [p.profit for p in positions],
            "magic": [p.magic for p in positions],
        }
        
        df = pl.DataFrame(data)
        
        # Filter by magic if provided
        if magic is not None:
            df = df.filter(pl.col("magic") == magic)
        
        return df
    
    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create empty DataFrame with correct schema."""
        return pl.DataFrame({
            "time": pl.Series([], dtype=pl.Datetime("us")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Int64),
            "spread": pl.Series([], dtype=pl.Int64),
            "real_volume": pl.Series([], dtype=pl.Int64),
        })


# Simulation connector for testing without MT5
class MT5SimulationConnector(MT5Connector):
    """Simulated MT5 connector for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(login=0, password="", server="Simulation")
        self._connected = True
    
    def connect(self, max_retries: int = 3) -> bool:
        self._connected = True
        logger.info("Simulation mode - connected")
        return True
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str = "M15",
        count: int = 1000,
    ) -> pl.DataFrame:
        """Generate simulated market data."""
        import numpy as np
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        
        # Base price for XAUUSD
        base_price = 2000.0
        
        # Generate random walk prices
        returns = np.random.randn(count) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from prices
        opens = prices
        closes = prices * (1 + np.random.randn(count) * 0.0005)
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(count)) * 0.0003)
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(count)) * 0.0003)
        volumes = np.random.randint(100, 10000, count)
        
        # Generate timestamps
        end_time = datetime.now()
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        minutes = tf_minutes.get(timeframe, 15)
        times = [
            end_time - pd.Timedelta(minutes=minutes * (count - i - 1))
            for i in range(count)
        ]
        
        return pl.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "spread": np.full(count, 2),
            "real_volume": volumes,
        })


# Import pandas only for simulation timestamp generation
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == "__main__":
    # Test with simulation
    connector = MT5SimulationConnector()
    connector.connect()
    
    df = connector.get_market_data("XAUUSD", "M15", 100)
    print(df.head(10))
    print(f"\nSchema: {df.schema}")
    print(f"Shape: {df.shape}")
