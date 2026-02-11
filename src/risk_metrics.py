"""
Risk Analytics Module
=====================
Professional-grade risk metrics for XAUBot AI.

Implements:
- Value at Risk (VaR) at 95% and 99% confidence
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk-adjusted returns)
- Calmar Ratio (return / max drawdown)
- Maximum Drawdown analysis
- Win/Loss statistics
- Risk-Reward ratios

Author: AI Assistant (Phase 8 - FinceptTerminal Enhancement)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger


class RiskAnalytics:
    """
    Comprehensive risk analytics for trading performance.

    Calculates professional metrics used by hedge funds and institutional traders.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize risk analytics.

        Args:
            risk_free_rate: Annual risk-free rate (default 4% = US Treasury)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """
        Calculate returns from equity curve.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Array of percentage returns
        """
        if len(equity_curve) < 2:
            return np.array([])

        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        return returns

    def value_at_risk(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR estimates the maximum expected loss over a time period
        at a given confidence level.

        Args:
            returns: Array of returns
            confidence: Confidence level (0.95 = 95%, 0.99 = 99%)

        Returns:
            VaR value (negative = loss)
        """
        if len(returns) == 0:
            return 0.0

        # Sort returns and find percentile
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))

        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[0]
        return var

    def conditional_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that VaR has been exceeded.
        More conservative than VaR.

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            CVaR value (negative = loss)
        """
        if len(returns) == 0:
            return 0.0

        var = self.value_at_risk(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var

    def sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted returns).

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Higher is better. >1.0 is good, >2.0 is excellent.

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year (252 for daily)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        # Annualized mean return
        mean_return = np.mean(returns) * periods_per_year

        # Annualized volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)

        if volatility == 0:
            return 0.0

        sharpe = (mean_return - self.risk_free_rate) / volatility
        return sharpe

    def sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted returns).

        Like Sharpe but only penalizes downside volatility.
        Better for strategies with asymmetric returns.

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Annualized mean return
        mean_return = np.mean(returns) * periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No losses = infinite Sortino

        downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year)

        if downside_deviation == 0:
            return 0.0

        sortino = (mean_return - self.risk_free_rate) / downside_deviation
        return sortino

    def calmar_ratio(
        self,
        returns: np.ndarray,
        max_drawdown: float,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).

        Calmar = Annualized Return / Max Drawdown

        Higher is better. >2.0 is good.

        Args:
            returns: Array of returns
            max_drawdown: Maximum drawdown (positive value)
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0

        annualized_return = np.mean(returns) * periods_per_year
        calmar = annualized_return / abs(max_drawdown)
        return calmar

    def maximum_drawdown(self, equity_curve: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: List of equity values

        Returns:
            (max_drawdown_pct, peak_idx, trough_idx)
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0

        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        max_dd = drawdown.min()
        trough_idx = drawdown.argmin()
        peak_idx = running_max[:trough_idx + 1].argmax() if trough_idx > 0 else 0

        return abs(max_dd), peak_idx, trough_idx

    def win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of winning trades).

        Args:
            returns: Array of returns

        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0

        wins = (returns > 0).sum()
        total = len(returns)
        return wins / total

    def profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Array of returns

        Returns:
            Profit factor (>1.0 = profitable)
        """
        if len(returns) == 0:
            return 0.0

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_win_loss_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate average win / average loss ratio.

        Args:
            returns: Array of returns

        Returns:
            Win/loss ratio
        """
        if len(returns) == 0:
            return 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return float('inf')

        return avg_win / avg_loss

    def get_comprehensive_report(
        self,
        equity_curve: List[float],
        trade_returns: Optional[List[float]] = None,
        periods_per_year: int = 252
    ) -> Dict:
        """
        Generate comprehensive risk report.

        Args:
            equity_curve: List of equity values over time
            trade_returns: Optional list of individual trade returns
            periods_per_year: Trading periods per year

        Returns:
            Dictionary with all risk metrics
        """
        if len(equity_curve) < 2:
            return {
                "error": "Insufficient data",
                "data_points": len(equity_curve)
            }

        # Calculate returns from equity curve
        returns = self.calculate_returns(equity_curve)

        # Use trade returns if provided, otherwise use equity returns
        if trade_returns and len(trade_returns) > 0:
            trade_ret = np.array(trade_returns)
        else:
            trade_ret = returns

        # Maximum drawdown
        max_dd, peak_idx, trough_idx = self.maximum_drawdown(equity_curve)

        # Risk metrics
        var_95 = self.value_at_risk(returns, 0.95)
        var_99 = self.value_at_risk(returns, 0.99)
        cvar_95 = self.conditional_var(returns, 0.95)

        sharpe = self.sharpe_ratio(returns, periods_per_year)
        sortino = self.sortino_ratio(returns, periods_per_year)
        calmar = self.calmar_ratio(returns, max_dd, periods_per_year)

        # Win/loss statistics
        win_rate = self.win_rate(trade_ret)
        profit_fac = self.profit_factor(trade_ret)
        win_loss_ratio = self.average_win_loss_ratio(trade_ret)

        # Return statistics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annualized_return = (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1

        return {
            # Return Metrics
            "total_return": total_return,
            "annualized_return": annualized_return,
            "avg_return": np.mean(returns),

            # Risk Metrics
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,

            # Value at Risk
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,

            # Drawdown
            "max_drawdown": max_dd,
            "max_dd_peak_idx": peak_idx,
            "max_dd_trough_idx": trough_idx,

            # Win/Loss Stats
            "win_rate": win_rate,
            "profit_factor": profit_fac,
            "win_loss_ratio": win_loss_ratio,

            # Volatility
            "volatility": np.std(returns),
            "annualized_volatility": np.std(returns) * np.sqrt(periods_per_year),

            # Data
            "total_trades": len(trade_ret),
            "data_points": len(equity_curve),
        }

    def format_report(self, report: Dict) -> str:
        """
        Format risk report as human-readable string.

        Args:
            report: Report from get_comprehensive_report()

        Returns:
            Formatted string
        """
        if "error" in report:
            return f"⚠️ {report['error']}"

        # Sharpe rating
        sharpe = report["sharpe_ratio"]
        if sharpe < 0:
            sharpe_rating = "❌ Negative"
        elif sharpe < 1.0:
            sharpe_rating = "⚠️ Poor"
        elif sharpe < 2.0:
            sharpe_rating = "✅ Good"
        else:
            sharpe_rating = "🎯 Excellent"

        # Win rate rating
        win_rate = report["win_rate"]
        if win_rate < 0.45:
            wr_rating = "❌ Low"
        elif win_rate < 0.55:
            wr_rating = "⚠️ Average"
        else:
            wr_rating = "✅ High"

        report_text = f"""
📊 RISK ANALYTICS REPORT
{'=' * 50}

📈 RETURN METRICS
  Total Return: {report['total_return']:.2%}
  Annualized: {report['annualized_return']:.2%}
  Avg Daily: {report['avg_return']:.3%}

⚖️ RISK-ADJUSTED RETURNS
  Sharpe Ratio: {sharpe:.2f} {sharpe_rating}
  Sortino Ratio: {report['sortino_ratio']:.2f}
  Calmar Ratio: {report['calmar_ratio']:.2f}

⚠️ VALUE AT RISK
  VaR 95%: {report['var_95']:.2%} (worst 5% day)
  VaR 99%: {report['var_99']:.2%} (worst 1% day)
  CVaR 95%: {report['cvar_95']:.2%} (expected shortfall)

📉 DRAWDOWN ANALYSIS
  Max Drawdown: {report['max_drawdown']:.2%}
  Peak -> Trough: {report['max_dd_peak_idx']} -> {report['max_dd_trough_idx']}

🎯 WIN/LOSS STATISTICS
  Win Rate: {win_rate:.1%} {wr_rating}
  Profit Factor: {report['profit_factor']:.2f}
  Avg Win/Loss: {report['win_loss_ratio']:.2f}x

📊 VOLATILITY
  Daily Vol: {report['volatility']:.2%}
  Annual Vol: {report['annualized_volatility']:.2%}

📈 PERFORMANCE SUMMARY
  Total Trades: {report['total_trades']}
  Data Points: {report['data_points']}

{'=' * 50}
"""
        return report_text


# Convenience functions for quick calculations

def quick_sharpe(returns: List[float], risk_free_rate: float = 0.04) -> float:
    """Quick Sharpe ratio calculation."""
    analytics = RiskAnalytics(risk_free_rate)
    return analytics.sharpe_ratio(np.array(returns))


def quick_var(returns: List[float], confidence: float = 0.95) -> float:
    """Quick VaR calculation."""
    analytics = RiskAnalytics()
    return analytics.value_at_risk(np.array(returns), confidence)


def quick_max_drawdown(equity_curve: List[float]) -> float:
    """Quick max drawdown calculation."""
    analytics = RiskAnalytics()
    max_dd, _, _ = analytics.maximum_drawdown(equity_curve)
    return max_dd


if __name__ == "__main__":
    # Example usage
    import random

    # Simulate equity curve
    equity = [5000]
    for _ in range(100):
        change = random.gauss(0.001, 0.02)  # 0.1% avg return, 2% volatility
        equity.append(equity[-1] * (1 + change))

    # Calculate risk metrics
    analytics = RiskAnalytics()
    report = analytics.get_comprehensive_report(equity)

    print(analytics.format_report(report))
