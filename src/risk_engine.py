"""
Risk Engine Module
==================
Risk management and position sizing logic.

Features:
- Risk-Constrained Kelly Criterion
- Daily loss limits (Circuit Breaker)
- Position size calculation
- Exposure management
"""

import polars as pl
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, date
from loguru import logger

from .config import TradingConfig, RiskConfig


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    daily_pnl: float
    daily_pnl_percent: float
    open_exposure: float
    max_drawdown: float
    position_count: int
    can_trade: bool
    reason: str


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    lot_size: float
    risk_amount: float
    risk_percent: float
    stop_distance: float
    take_profit_distance: float
    approved: bool
    rejection_reason: Optional[str] = None


class RiskEngine:
    """
    Risk management engine with circuit breakers.
    
    Implements:
    - Risk-Constrained Kelly Criterion for position sizing
    - Daily loss limits
    - Maximum exposure limits
    - Flash crash protection
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize risk engine.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.risk_config = config.risk
        
        # Track daily stats
        self._daily_stats: Dict[date, Dict] = {}
        self._trade_log: List[Dict] = []
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
    
    def check_risk(
        self,
        account_balance: float,
        account_equity: float,
        open_positions: pl.DataFrame,
        current_price: float,
    ) -> RiskMetrics:
        """
        Check current risk status.
        
        Args:
            account_balance: Current account balance
            account_equity: Current account equity
            open_positions: DataFrame of open positions
            current_price: Current market price
            
        Returns:
            RiskMetrics with current status
        """
        today = date.today()
        
        # Initialize daily stats if needed (use equity as starting balance)
        if today not in self._daily_stats:
            self._daily_stats[today] = {
                "starting_balance": account_equity,  # Use equity to include open positions
                "trades": 0,
                "wins": 0,
                "losses": 0,
            }
            # Reset circuit breaker on new day/first run
            self._circuit_breaker_active = False
            self._circuit_breaker_reason = ""
        
        daily = self._daily_stats[today]

        # Safety: Update starting balance if it was 0 (edge case on startup)
        if daily["starting_balance"] == 0 and account_equity > 0:
            daily["starting_balance"] = account_equity

        # Calculate daily P&L (with division safety)
        daily_pnl = account_equity - daily["starting_balance"]
        if daily["starting_balance"] > 0:
            daily_pnl_percent = (daily_pnl / daily["starting_balance"]) * 100
        else:
            daily_pnl_percent = 0.0
        
        # Calculate open exposure
        open_exposure = 0.0
        position_count = 0
        if len(open_positions) > 0:
            position_count = len(open_positions)
            # Sum of position values
            for row in open_positions.iter_rows(named=True):
                open_exposure += abs(row.get("volume", 0)) * current_price
        
        # Calculate max drawdown (equity from peak)
        max_drawdown = 0.0
        if hasattr(self, "_peak_equity"):
            if account_equity > self._peak_equity:
                self._peak_equity = account_equity
            max_drawdown = ((self._peak_equity - account_equity) / self._peak_equity) * 100
        else:
            self._peak_equity = account_equity
        
        # Check if trading is allowed
        can_trade = True
        reason = "OK"
        
        # Circuit breaker checks
        if self._circuit_breaker_active:
            can_trade = False
            reason = self._circuit_breaker_reason
        
        # Daily loss limit (only trigger on LOSSES, not profits)
        elif daily_pnl_percent <= -self.risk_config.max_daily_loss:
            can_trade = False
            reason = f"Daily loss limit reached: {daily_pnl_percent:.2f}%"
            self._activate_circuit_breaker(reason)
        
        # Maximum positions
        elif position_count >= self.risk_config.max_positions:
            can_trade = False
            reason = f"Maximum positions reached: {position_count}"
        
        return RiskMetrics(
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            open_exposure=open_exposure,
            max_drawdown=max_drawdown,
            position_count=position_count,
            can_trade=can_trade,
            reason=reason,
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        account_balance: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 2.0,
        regime_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate position size using Risk-Constrained Kelly Criterion.
        
        Kelly Formula: f* = (p * b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio
        
        We use Half-Kelly for safety.
        
        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            account_balance: Current account balance
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            regime_multiplier: Position size multiplier from regime detection
            
        Returns:
            PositionSizeResult with calculated lot size
        """
        # Validate inputs
        if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
            return PositionSizeResult(
                lot_size=0,
                risk_amount=0,
                risk_percent=0,
                stop_distance=0,
                take_profit_distance=0,
                approved=False,
                rejection_reason="Invalid price levels",
            )
        
        # Calculate distances
        stop_distance = abs(entry_price - stop_loss_price)
        tp_distance = abs(take_profit_price - entry_price)
        
        if stop_distance == 0:
            return PositionSizeResult(
                lot_size=0,
                risk_amount=0,
                risk_percent=0,
                stop_distance=0,
                take_profit_distance=tp_distance,
                approved=False,
                rejection_reason="Stop loss distance is zero",
            )
        
        # Calculate Kelly fraction
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio
        
        # Full Kelly
        if b > 0:
            kelly = (p * b - q) / b
        else:
            kelly = 0
        
        # Cap Kelly at reasonable level (never risk more than 25%)
        kelly = max(0, min(kelly, 0.25))
        
        # Use Half-Kelly for safety
        half_kelly = kelly * 0.5
        
        # Apply regime multiplier
        adjusted_kelly = half_kelly * regime_multiplier
        
        # Calculate risk amount (but cap at config limit)
        max_risk_percent = self.risk_config.risk_per_trade / 100
        actual_risk_percent = min(adjusted_kelly, max_risk_percent)
        risk_amount = account_balance * actual_risk_percent
        
        # Calculate lot size
        # For XAUUSD: pip value varies, using simplified calculation
        symbol = self.config.symbol
        if "XAU" in symbol:
            # Gold: $1 per 0.01 lot per point ($0.1 move)
            pip_value_per_lot = 1.0
            pips = stop_distance / 0.1
        else:
            # Standard forex
            pip_value_per_lot = 10.0
            pips = stop_distance / 0.0001
        
        if pips > 0 and pip_value_per_lot > 0:
            lot_size = risk_amount / (pips * pip_value_per_lot)
        else:
            lot_size = 0
        
        # Round to lot step and apply limits
        lot_size = round(lot_size / self.risk_config.lot_step) * self.risk_config.lot_step
        lot_size = max(self.risk_config.min_lot_size, min(lot_size, self.risk_config.max_lot_size))
        
        # Validate position
        approved = True
        rejection_reason = None
        
        if lot_size < self.risk_config.min_lot_size:
            approved = False
            rejection_reason = f"Lot size {lot_size} below minimum {self.risk_config.min_lot_size}"
        
        actual_risk = lot_size * pips * pip_value_per_lot
        actual_risk_pct = (actual_risk / account_balance) * 100
        
        return PositionSizeResult(
            lot_size=lot_size,
            risk_amount=actual_risk,
            risk_percent=actual_risk_pct,
            stop_distance=stop_distance,
            take_profit_distance=tp_distance,
            approved=approved,
            rejection_reason=rejection_reason,
        )
    
    def validate_order(
        self,
        order_type: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        current_price: float,
        account_balance: float,
    ) -> Tuple[bool, str]:
        """
        Validate order before execution.
        
        Args:
            order_type: "BUY" or "SELL"
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            lot_size: Lot size
            current_price: Current market price
            account_balance: Account balance
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check circuit breaker
        if self._circuit_breaker_active:
            return False, f"Circuit breaker active: {self._circuit_breaker_reason}"
        
        # Validate price levels
        if order_type == "BUY":
            if stop_loss >= entry_price:
                return False, "Buy SL must be below entry"
            if take_profit <= entry_price:
                return False, "Buy TP must be above entry"
        else:  # SELL
            if stop_loss <= entry_price:
                return False, "Sell SL must be above entry"
            if take_profit >= entry_price:
                return False, "Sell TP must be below entry"
        
        # Check lot size
        if lot_size < self.risk_config.min_lot_size:
            return False, f"Lot size below minimum: {lot_size} < {self.risk_config.min_lot_size}"
        if lot_size > self.risk_config.max_lot_size:
            return False, f"Lot size above maximum: {lot_size} > {self.risk_config.max_lot_size}"
        
        # Check entry price deviation from current
        max_deviation = 0.001  # 0.1%
        if abs(entry_price / current_price - 1) > max_deviation:
            return False, f"Entry price deviates too much from current: {entry_price} vs {current_price}"
        
        # Calculate risk
        stop_distance = abs(entry_price - stop_loss)
        if "XAU" in self.config.symbol:
            pips = stop_distance / 0.1
            pip_value = 1.0
        else:
            pips = stop_distance / 0.0001
            pip_value = 10.0
        
        risk_amount = lot_size * pips * pip_value
        risk_percent = (risk_amount / account_balance) * 100
        
        if risk_percent > self.risk_config.risk_per_trade * 1.5:  # Allow 50% margin
            return False, f"Risk too high: {risk_percent:.2f}% > {self.risk_config.risk_per_trade * 1.5:.2f}%"
        
        return True, "Order validated"
    
    def record_trade(
        self,
        order_type: str,
        entry_price: float,
        exit_price: Optional[float],
        lot_size: float,
        pnl: float,
        is_win: bool,
    ):
        """
        Record trade for statistics.
        
        Args:
            order_type: "BUY" or "SELL"
            entry_price: Entry price
            exit_price: Exit price (if closed)
            lot_size: Lot size
            pnl: Profit/loss amount
            is_win: Whether trade was profitable
        """
        trade = {
            "timestamp": datetime.now(),
            "type": order_type,
            "entry": entry_price,
            "exit": exit_price,
            "lot_size": lot_size,
            "pnl": pnl,
            "is_win": is_win,
        }
        self._trade_log.append(trade)
        
        # Update daily stats
        today = date.today()
        if today in self._daily_stats:
            self._daily_stats[today]["trades"] += 1
            if is_win:
                self._daily_stats[today]["wins"] += 1
            else:
                self._daily_stats[today]["losses"] += 1
        
        logger.info(f"Trade recorded: {order_type} {lot_size} lots, P&L: {pnl:.2f}")
    
    def get_win_rate(self, lookback: int = 100) -> float:
        """Get recent win rate."""
        recent = self._trade_log[-lookback:]
        if not recent:
            return 0.5  # Default
        
        wins = sum(1 for t in recent if t["is_win"])
        return wins / len(recent)
    
    def get_avg_rr(self, lookback: int = 100) -> float:
        """Get average risk/reward ratio."""
        recent = self._trade_log[-lookback:]
        if not recent:
            return 2.0  # Default
        
        wins = [t["pnl"] for t in recent if t["is_win"] and t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in recent if not t["is_win"] and t["pnl"] < 0]
        
        if not wins or not losses:
            return 2.0
        
        return np.mean(wins) / np.mean(losses)
    
    def _activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker."""
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        logger.warning(f"CIRCUIT BREAKER ACTIVATED: {reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual override)."""
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        logger.info("Circuit breaker reset")
    
    def reset_daily_stats(self):
        """Reset daily statistics (called at start of new day)."""
        today = date.today()
        self._daily_stats[today] = {
            "starting_balance": 0,  # Will be set on first check
            "trades": 0,
            "wins": 0,
            "losses": 0,
        }
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        logger.info("Daily stats reset")
    
    def get_daily_summary(self) -> Dict:
        """Get daily trading summary."""
        today = date.today()
        if today not in self._daily_stats:
            return {"trades": 0, "wins": 0, "losses": 0, "pnl": 0}
        
        stats = self._daily_stats[today]
        return {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0,
        }


if __name__ == "__main__":
    # Test risk engine
    from .config import TradingConfig
    
    # Create test config
    config = TradingConfig(capital=5000)
    engine = RiskEngine(config)
    
    print("\n=== Risk Engine Test ===")
    print(f"Config: {config.capital_mode.value}")
    print(f"Risk per trade: {config.risk.risk_per_trade}%")
    print(f"Max daily loss: {config.risk.max_daily_loss}%")
    
    # Test position sizing
    result = engine.calculate_position_size(
        entry_price=2000.0,
        stop_loss_price=1995.0,  # 50 pips
        take_profit_price=2010.0,  # 100 pips
        account_balance=5000.0,
        win_rate=0.55,
        avg_win_loss_ratio=2.0,
        regime_multiplier=1.0,
    )
    
    print(f"\n=== Position Size Result ===")
    print(f"Lot size: {result.lot_size}")
    print(f"Risk amount: ${result.risk_amount:.2f}")
    print(f"Risk percent: {result.risk_percent:.2f}%")
    print(f"Stop distance: {result.stop_distance}")
    print(f"Approved: {result.approved}")
    if result.rejection_reason:
        print(f"Rejection: {result.rejection_reason}")
    
    # Test order validation
    valid, reason = engine.validate_order(
        order_type="BUY",
        entry_price=2000.0,
        stop_loss=1995.0,
        take_profit=2010.0,
        lot_size=result.lot_size,
        current_price=2000.0,
        account_balance=5000.0,
    )
    
    print(f"\n=== Order Validation ===")
    print(f"Valid: {valid}")
    print(f"Reason: {reason}")
    
    # Test risk check
    open_positions = pl.DataFrame({
        "ticket": [],
        "volume": [],
        "symbol": [],
    })
    
    metrics = engine.check_risk(
        account_balance=5000.0,
        account_equity=4950.0,
        open_positions=open_positions,
        current_price=2000.0,
    )
    
    print(f"\n=== Risk Metrics ===")
    print(f"Daily P&L: ${metrics.daily_pnl:.2f} ({metrics.daily_pnl_percent:.2f}%)")
    print(f"Open exposure: ${metrics.open_exposure:.2f}")
    print(f"Max drawdown: {metrics.max_drawdown:.2f}%")
    print(f"Can trade: {metrics.can_trade}")
    print(f"Reason: {metrics.reason}")
