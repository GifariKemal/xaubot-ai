"""
Kelly Criterion for Dynamic Position Scaling
=============================================
Optimal position sizing based on win probability and payoff ratio.

Kelly Formula:
  f* = (p × b - q) / b
  where:
    p = win probability
    q = loss probability (1 - p)
    b = win/loss ratio (avg_win / avg_loss)

Application:
- Partial exits when exit_confidence is medium (0.50-0.70)
- Scale position down if Kelly fraction suggests reducing exposure
- Full exit if Kelly fraction < 0.3

Integration with Fuzzy Logic:
- High exit_confidence (>0.75) → adjust win probability down → Kelly suggests reduce
- Low exit_confidence (<0.50) → maintain position → Kelly suggests hold

Author: AI Assistant (Phase 6 - Advanced Exit Strategies)
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


class KellyPositionScaler:
    """
    Kelly criterion calculator for position scaling.

    Dynamically adjusts position size based on:
    - Exit confidence (from fuzzy logic)
    - Trade statistics (win rate, avg win/loss)
    """

    def __init__(
        self,
        base_win_rate: float = 0.55,
        avg_win: float = 8.0,
        avg_loss: float = 4.0,
        kelly_fraction: float = 0.5,
    ):
        """
        Initialize Kelly scaler.

        Args:
            base_win_rate: Historical win rate (0-1)
            avg_win: Average winning trade ($)
            avg_loss: Average losing trade ($)
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly for safety)
        """
        self.base_win_rate = base_win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction

        # Running statistics (updated from trade history)
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.sum_wins = 0.0
        self.sum_losses = 0.0

    def calculate_optimal_fraction(
        self,
        exit_confidence: float,
        current_profit: float,
        target_profit: float,
    ) -> float:
        """
        Calculate optimal position fraction to hold.

        Args:
            exit_confidence: Fuzzy exit confidence (0-1)
            current_profit: Current profit ($)
            target_profit: Target TP ($)

        Returns:
            Fraction of position to hold (0-1)
            1.0 = hold 100%
            0.5 = close 50%
            0.0 = close 100%
        """
        # Adjust win probability based on exit confidence
        # High exit_confidence = lower win probability for continuing
        p_continue_win = self.base_win_rate * (1 - exit_confidence * 0.7)

        # Win/loss ratio
        if self.avg_loss > 0:
            b = self.avg_win / self.avg_loss
        else:
            b = 2.0  # Default

        # Kelly formula
        q = 1 - p_continue_win
        kelly_optimal = (p_continue_win * b - q) / b

        # Apply fractional Kelly for safety
        kelly_optimal *= self.kelly_fraction

        # Clamp to [0, 1]
        kelly_optimal = np.clip(kelly_optimal, 0, 1)

        return kelly_optimal

    def get_exit_action(
        self,
        exit_confidence: float,
        current_profit: float,
        target_profit: float,
    ) -> Tuple[bool, float, str]:
        """
        Get exit action based on Kelly criterion.

        Args:
            exit_confidence: Fuzzy exit confidence (0-1)
            current_profit: Current profit ($)
            target_profit: Target TP ($)

        Returns:
            (should_exit, close_fraction, reason)
            should_exit: True if any exit recommended
            close_fraction: 0-1 (0=hold, 1=full exit)
            reason: Exit reason string
        """
        kelly_hold = self.calculate_optimal_fraction(
            exit_confidence, current_profit, target_profit
        )

        # Full exit: Kelly suggests 0% hold
        if kelly_hold < 0.25:
            return True, 1.0, f"Kelly full exit: hold={kelly_hold:.2f}"

        # Partial exit: Kelly suggests 25-70% hold
        elif kelly_hold < 0.70:
            close_fraction = 1 - kelly_hold
            return True, close_fraction, f"Kelly partial: close {close_fraction:.0%} (hold={kelly_hold:.2f})"

        # Hold: Kelly suggests 70%+ hold
        else:
            return False, 0.0, f"Kelly hold: {kelly_hold:.2%}"

    def update_statistics(self, profit: float):
        """
        Update running statistics from completed trade.

        Args:
            profit: Trade profit/loss ($)
        """
        self.total_trades += 1

        if profit > 0:
            self.total_wins += 1
            self.sum_wins += profit
        else:
            self.total_losses += 1
            self.sum_losses += abs(profit)

        # Recalculate base parameters
        if self.total_trades > 0:
            self.base_win_rate = self.total_wins / self.total_trades

        if self.total_wins > 0:
            self.avg_win = self.sum_wins / self.total_wins

        if self.total_losses > 0:
            self.avg_loss = self.sum_losses / self.total_losses

    def get_statistics(self) -> dict:
        """Get current statistics."""
        win_loss_ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 0

        return {
            "total_trades": self.total_trades,
            "win_rate": self.base_win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "kelly_fraction": self.kelly_fraction,
        }

    def set_parameters(
        self,
        base_win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ):
        """Update parameters manually."""
        if base_win_rate is not None:
            self.base_win_rate = base_win_rate
        if avg_win is not None:
            self.avg_win = avg_win
        if avg_loss is not None:
            self.avg_loss = avg_loss
        if kelly_fraction is not None:
            self.kelly_fraction = kelly_fraction
