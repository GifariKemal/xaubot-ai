"""
Risk Analytics Report Generator
================================
Analyzes trade history and generates professional risk metrics.

Usage:
    python scripts/generate_risk_report.py
    python scripts/generate_risk_report.py --days 30
    python scripts/generate_risk_report.py --output report.txt

Author: AI Assistant (Phase 8 - FinceptTerminal Enhancement)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
from datetime import datetime, timedelta
from loguru import logger

from src.risk_metrics import RiskAnalytics
from src.mt5_connector import MT5Connector
from src.config import TradingConfig


async def fetch_trade_history(days: int = 30):
    """
    Fetch trade history from MT5.

    Args:
        days: Number of days to look back

    Returns:
        List of trades with profit/loss
    """
    config = TradingConfig()
    mt5 = MT5Connector(config)

    if not mt5.connect():
        logger.error("Failed to connect to MT5")
        return None

    try:
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Fetch history
        deals = mt5.get_deals_history(from_date, to_date)

        if not deals:
            logger.warning(f"No trade history found in last {days} days")
            return None

        # Build equity curve and trade returns
        equity_curve = [config.capital]  # Starting capital
        trade_returns = []

        for deal in deals:
            profit = deal.profit
            trade_returns.append(profit)
            equity_curve.append(equity_curve[-1] + profit)

        logger.info(f"Fetched {len(trade_returns)} trades from last {days} days")
        return equity_curve, trade_returns

    finally:
        mt5.disconnect()


def generate_risk_report(equity_curve, trade_returns, output_file=None):
    """
    Generate comprehensive risk analytics report.

    Args:
        equity_curve: List of equity values over time
        trade_returns: List of individual trade P&L
        output_file: Optional file to save report
    """
    if not equity_curve or len(equity_curve) < 2:
        logger.error("Insufficient data for risk analysis")
        return

    # Initialize analytics
    analytics = RiskAnalytics(risk_free_rate=0.04)  # 4% US Treasury

    # Calculate comprehensive metrics
    report = analytics.get_comprehensive_report(
        equity_curve=equity_curve,
        trade_returns=trade_returns,
        periods_per_year=252  # Trading days
    )

    if "error" in report:
        logger.error(f"Risk calculation error: {report['error']}")
        return

    # Format report
    report_text = analytics.format_report(report)

    # Additional context
    initial_capital = equity_curve[0]
    final_capital = equity_curve[-1]
    net_profit = final_capital - initial_capital

    header = f"""
{'=' * 50}
XAUBOT AI - RISK ANALYTICS REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: Last {len(trade_returns)} trades
Initial Capital: ${initial_capital:,.2f}
Final Capital: ${final_capital:,.2f}
Net P&L: ${net_profit:,.2f} ({(net_profit/initial_capital)*100:.2f}%)
{'=' * 50}
"""

    full_report = header + report_text

    # Print to console
    print(full_report)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        logger.info(f"Report saved to {output_file}")

    # Interpretation guide
    interpretation = """
📖 INTERPRETATION GUIDE
{'=' * 50}

Sharpe Ratio:
  < 0    : Strategy is losing vs risk-free rate
  0-1    : Poor risk-adjusted returns
  1-2    : Good risk-adjusted returns
  > 2    : Excellent risk-adjusted returns

Win Rate:
  < 45%  : Low (need high win/loss ratio)
  45-55% : Average
  > 55%  : High

Profit Factor:
  < 1.0  : Losing strategy
  1.0-1.5: Break-even to marginal
  1.5-2.0: Good
  > 2.0  : Excellent

Max Drawdown:
  < 10%  : Very safe
  10-20% : Acceptable
  20-30% : High risk
  > 30%  : Dangerous

Sortino Ratio:
  Like Sharpe but only penalizes downside volatility.
  Higher is better. > 2.0 is excellent.

Calmar Ratio:
  Return / Max Drawdown
  > 2.0 is good, > 3.0 is excellent

VaR (Value at Risk):
  95% VaR = Worst expected loss 5% of the time
  99% VaR = Worst expected loss 1% of the time
  CVaR = Average loss when VaR is exceeded

{'=' * 50}
"""

    print(interpretation)

    return report


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate risk analytics report for XAUBot AI"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to file (optional)"
    )

    args = parser.parse_args()

    logger.info(f"Fetching trade history for last {args.days} days...")

    # Fetch data
    result = await fetch_trade_history(args.days)

    if result is None:
        logger.error("Failed to fetch trade history")
        return

    equity_curve, trade_returns = result

    # Generate report
    logger.info("Generating risk analytics report...")
    generate_risk_report(equity_curve, trade_returns, args.output)


if __name__ == "__main__":
    asyncio.run(main())
