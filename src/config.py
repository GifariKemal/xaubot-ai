"""
Configuration Module for Smart Trading Bot
==========================================
Defines trading parameters based on capital size.

Capital Modes:
- Small ($5,000): Risk 1.5%, Leverage 1:100 (Growth mode)
- Medium ($50,000): Risk 0.5%, Leverage 1:30 (Preservation mode)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class CapitalMode(Enum):
    """Trading mode based on capital size."""
    SMALL = "small"      # $5,000 - Aggressive growth
    MEDIUM = "medium"    # $50,000 - Capital preservation


@dataclass
class RiskConfig:
    """Risk management configuration."""
    risk_per_trade: float          # Percentage of capital to risk per trade
    max_daily_loss: float          # Maximum daily loss percentage
    max_leverage: int              # Maximum leverage ratio
    max_positions: int             # Maximum concurrent positions
    max_lot_size: float            # Maximum lot size per trade
    min_lot_size: float            # Minimum lot size
    lot_step: float                # Lot size increment
    

@dataclass
class SMCConfig:
    """Smart Money Concepts configuration."""
    swing_length: int = 5          # Bars for swing detection
    fvg_min_gap_pips: float = 2.0  # Minimum FVG gap in pips
    ob_lookback: int = 10          # Order block lookback period
    bos_close_break: bool = True   # Require close above/below for BOS


@dataclass
class MLConfig:
    """Machine Learning model configuration."""
    model_path: str = "models/xgboost_model.json"
    confidence_threshold: float = 0.65  # Minimum AI confidence for entry
    retrain_frequency_days: int = 7     # Retrain model every N days
    lookback_periods: int = 1000        # Training data lookback


@dataclass
class ThresholdsConfig:
    """
    Centralized trading thresholds configuration.
    All hard-coded thresholds should be defined here.
    """
    # ML Confidence Thresholds
    ml_min_confidence: float = 0.65        # Minimum confidence to consider signal
    ml_entry_confidence: float = 0.70      # Default confidence for entry
    ml_high_confidence: float = 0.75       # High confidence threshold
    ml_very_high_confidence: float = 0.80  # Very high confidence (lot multiplier)

    # Dynamic Threshold Adjustments
    dynamic_threshold_aggressive: float = 0.65   # Aggressive mode threshold
    dynamic_threshold_moderate: float = 0.70     # Moderate mode threshold
    dynamic_threshold_conservative: float = 0.75 # Conservative mode threshold

    # Risk Management Thresholds
    trend_reversal_confidence: float = 0.75  # ML confidence to trigger reversal close
    protected_mode_threshold: float = 0.80   # % of daily limit to enter protected mode

    # Profit/Loss Thresholds (USD)
    min_profit_to_secure: float = 15.0       # Minimum profit before considering secure
    good_profit_level: float = 25.0          # Good profit level
    great_profit_level: float = 40.0         # Great profit - take it

    # Trade Timing
    trade_cooldown_seconds: int = 300        # Minimum seconds between trades
    loop_interval_seconds: float = 30.0      # Main loop interval

    # Session Multipliers
    sydney_lot_multiplier: float = 0.5       # Sydney session lot reduction
    

@dataclass
class RegimeConfig:
    """Market regime detection configuration."""
    n_regimes: int = 3             # Number of HMM states
    lookback_periods: int = 500    # HMM training lookback
    retrain_frequency: int = 20    # Retrain every N bars


@dataclass
class TradingConfig:
    """
    Main trading configuration.
    Automatically adjusts parameters based on capital mode.
    """
    # MT5 Connection
    mt5_login: int = field(default_factory=lambda: int(os.getenv("MT5_LOGIN", "0")))
    mt5_password: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: os.getenv("MT5_SERVER", ""))
    mt5_path: Optional[str] = field(default_factory=lambda: os.getenv("MT5_PATH"))
    
    # Trading Symbol
    symbol: str = "XAUUSD"
    
    # Timeframes
    execution_timeframe: str = "M15"    # Entry timeframe
    trend_timeframe: str = "H4"         # Trend analysis timeframe
    
    # Capital
    capital: float = 5000.0
    capital_mode: CapitalMode = CapitalMode.SMALL
    
    # Sub-configurations
    risk: RiskConfig = field(default_factory=lambda: RiskConfig(
        risk_per_trade=1.5,
        max_daily_loss=3.0,
        max_leverage=100,
        max_positions=3,
        max_lot_size=0.5,
        min_lot_size=0.01,
        lot_step=0.01,
    ))
    smc: SMCConfig = field(default_factory=SMCConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)

    # Execution
    slippage_points: int = 20          # Maximum slippage in points
    magic_number: int = 123456         # Order identification
    
    # Circuit Breaker
    flash_crash_threshold: float = 2.5  # 2.5% move in 1 minute triggers halt (Gold-friendly)
    
    def __post_init__(self):
        """Adjust configuration based on capital mode and validate required settings."""
        self._validate_required_settings()
        self._configure_by_capital()

    def _validate_required_settings(self):
        """
        Validate that required environment variables are set.
        Raises ValueError if critical settings are missing.
        """
        missing = []

        # MT5 credentials validation
        if self.mt5_login == 0:
            missing.append("MT5_LOGIN")
        if not self.mt5_password:
            missing.append("MT5_PASSWORD")
        if not self.mt5_server:
            missing.append("MT5_SERVER")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please set them in your .env file."
            )

        # Capital validation
        if self.capital <= 0:
            raise ValueError(f"Invalid capital: {self.capital}. Must be positive.")
    
    def _configure_by_capital(self):
        """Set parameters based on capital size."""
        if self.capital <= 10000:
            self.capital_mode = CapitalMode.SMALL
            self._configure_small_account()
        else:
            self.capital_mode = CapitalMode.MEDIUM
            self._configure_medium_account()
    
    def _configure_small_account(self):
        """
        Small Account Configuration ($5,000)
        Strategy: SMART SAFE - Mental health FIRST!

        Features:
        - Risk 1% per trade
        - Multiple positions allowed (max 3) based on market
        - Smart management (no hard SL)
        """
        self.risk = RiskConfig(
            risk_per_trade=1.0,        # 1% = $50 risk per trade
            max_daily_loss=3.0,        # 3% daily loss limit
            max_leverage=100,          # 1:100 leverage
            max_positions=3,           # Max 3 positions (based on market)
            max_lot_size=0.05,         # Max 0.05 lot
            min_lot_size=0.01,         # Min 0.01 lot
            lot_step=0.01,
        )
        # Focus on single high-liquidity pair
        self.execution_timeframe = "M15"  # Scalping/day trading
        
    def _configure_medium_account(self):
        """
        Medium Account Configuration ($50,000)
        Strategy: Conservative, capital preservation
        """
        self.risk = RiskConfig(
            risk_per_trade=0.5,        # 0.5% = $250 risk per trade
            max_daily_loss=2.0,        # 2% daily loss limit
            max_leverage=30,           # 1:30 leverage (safer)
            max_positions=5,           # More diversification
            max_lot_size=2.0,          # Max 2 lots
            min_lot_size=0.01,
            lot_step=0.01,
        )
        # Swing trading approach
        self.execution_timeframe = "H1"   # Longer timeframe
        self.trend_timeframe = "H4"
    
    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Create configuration from environment variables."""
        capital = float(os.getenv("CAPITAL", "5000"))
        symbol = os.getenv("SYMBOL", "XAUUSD")
        
        config = cls(
            capital=capital,
            symbol=symbol,
        )
        
        # Override from env if provided
        if os.getenv("RISK_PER_TRADE"):
            config.risk.risk_per_trade = float(os.getenv("RISK_PER_TRADE"))

        if os.getenv("MAX_DAILY_LOSS_PERCENT"):
            config.risk.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_PERCENT"))

        if os.getenv("MAX_POSITION_SIZE"):
            config.risk.max_lot_size = float(os.getenv("MAX_POSITION_SIZE"))

        if os.getenv("MIN_LOT_SIZE"):
            config.risk.min_lot_size = float(os.getenv("MIN_LOT_SIZE"))

        if os.getenv("AI_CONFIDENCE_THRESHOLD"):
            config.ml.confidence_threshold = float(os.getenv("AI_CONFIDENCE_THRESHOLD"))

        if os.getenv("FLASH_CRASH_THRESHOLD"):
            config.flash_crash_threshold = float(os.getenv("FLASH_CRASH_THRESHOLD"))

        return config
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_balance: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on Risk-Constrained Kelly Criterion.
        
        Formula: Lot Size = (Account Balance * Risk%) / (SL Distance in pips * Pip Value)
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance (uses capital if None)
        
        Returns:
            Calculated lot size (rounded to lot_step)
        """
        balance = account_balance or self.capital
        risk_amount = balance * (self.risk.risk_per_trade / 100)
        
        # Calculate SL distance in price
        sl_distance = abs(entry_price - stop_loss_price)
        
        if sl_distance == 0:
            return self.risk.min_lot_size
        
        # For XAUUSD: 1 pip = 0.1, pip value per lot ~$1
        # Simplified calculation - adjust pip_value based on symbol
        if "XAU" in self.symbol:
            pip_value_per_lot = 1.0  # $1 per 0.1 move per lot
            sl_pips = sl_distance / 0.1
        else:
            pip_value_per_lot = 10.0  # Standard forex
            sl_pips = sl_distance / 0.0001
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Apply Half-Kelly for safety (reduces volatility)
        lot_size *= 0.5
        
        # Round to lot step
        lot_size = round(lot_size / self.risk.lot_step) * self.risk.lot_step
        
        # Apply limits
        lot_size = max(self.risk.min_lot_size, min(lot_size, self.risk.max_lot_size))
        
        return lot_size
    
    def __repr__(self) -> str:
        return (
            f"TradingConfig(\n"
            f"  symbol={self.symbol},\n"
            f"  capital=${self.capital:,.2f},\n"
            f"  mode={self.capital_mode.value},\n"
            f"  risk_per_trade={self.risk.risk_per_trade}%,\n"
            f"  max_leverage=1:{self.risk.max_leverage},\n"
            f"  execution_tf={self.execution_timeframe},\n"
            f"  trend_tf={self.trend_timeframe}\n"
            f")"
        )


# Global configuration instance
def get_config() -> TradingConfig:
    """Get the global trading configuration."""
    return TradingConfig.from_env()


if __name__ == "__main__":
    # Test configuration
    print("=== Small Account ($5,000) ===")
    config_small = TradingConfig(capital=5000)
    print(config_small)
    print(f"Position size for 50 pip SL: {config_small.calculate_position_size(2000, 1995)} lots")
    
    print("\n=== Medium Account ($50,000) ===")
    config_medium = TradingConfig(capital=50000)
    print(config_medium)
    print(f"Position size for 50 pip SL: {config_medium.calculate_position_size(2000, 1995)} lots")
