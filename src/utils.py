"""
Utility Functions
=================
Helper functions for the trading system.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


def validate_ohlcv_data(df: pl.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV DataFrame structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Required columns
    required = ["time", "open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    if issues:
        return False, issues
    
    # Check data types
    if df["time"].dtype not in [pl.Datetime, pl.Date]:
        issues.append(f"'time' should be datetime, got {df['time'].dtype}")
    
    for col in ["open", "high", "low", "close"]:
        if df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            issues.append(f"'{col}' should be numeric, got {df[col].dtype}")
    
    # Check for null values
    null_counts = df.select([
        pl.col(c).is_null().sum().alias(c) for c in required
    ]).row(0)
    
    for i, col in enumerate(required):
        if null_counts[i] > 0:
            issues.append(f"Column '{col}' has {null_counts[i]} null values")
    
    # Check OHLC relationship
    invalid_candles = df.filter(
        (pl.col("high") < pl.col("low")) |
        (pl.col("high") < pl.col("open")) |
        (pl.col("high") < pl.col("close")) |
        (pl.col("low") > pl.col("open")) |
        (pl.col("low") > pl.col("close"))
    )
    
    if len(invalid_candles) > 0:
        issues.append(f"Found {len(invalid_candles)} invalid OHLC relationships")
    
    # Check time ordering
    if df["time"].is_sorted():
        pass  # OK
    else:
        issues.append("Time column is not sorted")
    
    return len(issues) == 0, issues


def resample_ohlcv(
    df: pl.DataFrame,
    target_timeframe: str,
) -> pl.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    
    Args:
        df: Source DataFrame with OHLCV data
        target_timeframe: Target timeframe ("5m", "15m", "1h", "4h", "1d")
        
    Returns:
        Resampled DataFrame
    """
    # Map timeframe strings to durations
    tf_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
    }
    
    every = tf_map.get(target_timeframe, target_timeframe)
    
    df = df.sort("time")
    
    resampled = df.group_by_dynamic("time", every=every).agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum() if "volume" in df.columns else pl.lit(0).alias("volume"),
    ])
    
    return resampled


def calculate_pip_value(
    symbol: str,
    lot_size: float,
    account_currency: str = "USD",
) -> float:
    """
    Calculate pip value for a symbol.
    
    Args:
        symbol: Trading symbol
        lot_size: Lot size
        account_currency: Account currency
        
    Returns:
        Pip value in account currency
    """
    # Standard forex pairs (per standard lot)
    pip_values = {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "AUDUSD": 10.0,
        "NZDUSD": 10.0,
        "USDJPY": 9.1,  # Approximate, varies
        "USDCHF": 10.0,
        "USDCAD": 7.5,  # Approximate
        "XAUUSD": 1.0,  # Per 0.1 move
        "XAGUSD": 0.5,  # Per 0.01 move
    }
    
    base_pip = pip_values.get(symbol, 10.0)
    return base_pip * lot_size


def calculate_trade_statistics(trades: List[Dict]) -> Dict:
    """
    Calculate trading statistics from trade history.
    
    Args:
        trades: List of trade dictionaries with 'pnl', 'is_win' keys
        
    Returns:
        Dictionary of statistics
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_win": 0,
            "max_loss": 0,
            "total_pnl": 0,
            "sharpe_ratio": 0,
        }
    
    wins = [t for t in trades if t.get("is_win", False)]
    losses = [t for t in trades if not t.get("is_win", True)]
    
    total_trades = len(trades)
    win_count = len(wins)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    win_pnls = [t.get("pnl", 0) for t in wins]
    loss_pnls = [abs(t.get("pnl", 0)) for t in losses]
    all_pnls = [t.get("pnl", 0) for t in trades]
    
    total_wins = sum(win_pnls)
    total_losses = sum(loss_pnls)
    
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    
    max_win = max(win_pnls) if win_pnls else 0
    max_loss = max(loss_pnls) if loss_pnls else 0
    
    total_pnl = sum(all_pnls)
    
    # Sharpe ratio (simplified)
    if len(all_pnls) > 1:
        returns = np.array(all_pnls)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
    }


def format_price(price: float, digits: int = 5) -> str:
    """Format price with correct decimal places."""
    return f"{price:.{digits}f}"


def format_lot(lot: float) -> str:
    """Format lot size."""
    return f"{lot:.2f}"


def format_percentage(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.2f}%"


def format_currency(value: float, currency: str = "USD") -> str:
    """Format as currency."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{value:,.2f}"


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation", log: bool = True):
        self.name = name
        self.log = log
        self.elapsed = 0.0
    
    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self._start
        if self.log:
            logger.debug(f"{self.name}: {self.elapsed*1000:.2f}ms")
        return False


def create_synthetic_data(
    n_bars: int = 1000,
    base_price: float = 2000.0,
    volatility: float = 0.002,
    seed: Optional[int] = 42,
) -> pl.DataFrame:
    """
    Create synthetic OHLCV data for testing.
    
    Args:
        n_bars: Number of bars to generate
        base_price: Starting price
        volatility: Daily volatility
        seed: Random seed
        
    Returns:
        Polars DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random walk prices
    returns = np.random.randn(n_bars) * volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    opens = prices
    closes = prices * (1 + np.random.randn(n_bars) * volatility * 0.5)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * volatility * 0.3)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * volatility * 0.3)
    volumes = np.random.randint(1000, 10000, n_bars)
    
    # Generate timestamps
    end_time = datetime.now()
    times = [end_time - timedelta(minutes=15 * (n_bars - i - 1)) for i in range(n_bars)]
    
    return pl.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


if __name__ == "__main__":
    # Test utilities
    print("=== Utility Tests ===\n")
    
    # Test synthetic data
    df = create_synthetic_data(100)
    print(f"Synthetic data shape: {df.shape}")
    
    # Validate
    is_valid, issues = validate_ohlcv_data(df)
    print(f"Data valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Test resampling
    df_resampled = resample_ohlcv(df, "1h")
    print(f"Resampled shape: {df_resampled.shape}")
    
    # Test statistics
    trades = [
        {"pnl": 100, "is_win": True},
        {"pnl": -50, "is_win": False},
        {"pnl": 75, "is_win": True},
        {"pnl": -30, "is_win": False},
        {"pnl": 120, "is_win": True},
    ]
    stats = calculate_trade_statistics(trades)
    print(f"\nTrade Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test formatting
    print(f"\nFormatting:")
    print(f"  Price: {format_price(2000.12345)}")
    print(f"  Lot: {format_lot(0.05)}")
    print(f"  Percentage: {format_percentage(0.55)}")
    print(f"  Currency: {format_currency(1234.56)}")
    
    # Test timer
    with PerformanceTimer("Test operation"):
        import time
        time.sleep(0.1)
