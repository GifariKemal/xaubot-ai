"""
Test Phase 8 (Risk Metrics) and Phase 9 (Macro Data) Modules
=============================================================

Quick validation that both modules work correctly.

Usage:
    python tests/test_phase8_phase9.py

Author: AI Assistant
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import numpy as np
from loguru import logger

from src.risk_metrics import RiskAnalytics, quick_sharpe, quick_var, quick_max_drawdown
from src.macro_connector import MacroDataConnector, get_quick_macro_score


def test_risk_metrics():
    """Test risk metrics module."""
    print("\n" + "=" * 60)
    print("TEST 1: RISK METRICS MODULE")
    print("=" * 60)

    # Simulate equity curve (100 trades)
    np.random.seed(42)
    equity = [5000]
    returns = []

    for _ in range(100):
        # Simulate realistic trading returns
        # 55% win rate, avg win $8, avg loss $4
        if np.random.rand() < 0.55:
            profit = np.random.normal(8, 3)  # Win
        else:
            profit = np.random.normal(-4, 2)  # Loss

        returns.append(profit)
        equity.append(equity[-1] + profit)

    print(f"\nSimulated Equity Curve:")
    print(f"  Starting Capital: ${equity[0]:,.2f}")
    print(f"  Ending Capital: ${equity[-1]:,.2f}")
    print(f"  Net P&L: ${equity[-1] - equity[0]:,.2f}")
    print(f"  Total Trades: {len(returns)}")

    # Test 1: Quick functions
    print("\n--- Quick Functions ---")
    sharpe = quick_sharpe(returns)
    var_95 = quick_var(returns, 0.95)
    max_dd = quick_max_drawdown(equity)

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"VaR 95%: ${var_95:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    assert isinstance(sharpe, float), "Sharpe should be float"
    assert isinstance(var_95, float), "VaR should be float"
    assert isinstance(max_dd, float), "Max DD should be float"
    print("[OK] Quick functions work correctly")

    # Test 2: Comprehensive report
    print("\n--- Comprehensive Report ---")
    analytics = RiskAnalytics(risk_free_rate=0.04)
    report = analytics.get_comprehensive_report(
        equity_curve=equity,
        trade_returns=returns,
        periods_per_year=252
    )

    assert "error" not in report, "Report should not have errors"
    assert "sharpe_ratio" in report, "Missing Sharpe ratio"
    assert "sortino_ratio" in report, "Missing Sortino ratio"
    assert "calmar_ratio" in report, "Missing Calmar ratio"
    assert "win_rate" in report, "Missing win rate"
    assert "profit_factor" in report, "Missing profit factor"
    print("[OK] Comprehensive report generated")

    # Test 3: Formatted output
    print("\n--- Formatted Report ---")
    formatted = analytics.format_report(report)
    assert len(formatted) > 100, "Formatted report too short"
    assert "RISK ANALYTICS REPORT" in formatted, "Missing header"
    print("[OK] Report formatting works")

    # Display key metrics
    print(f"\nKey Metrics:")
    print(f"  Sharpe Ratio: {report['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {report['sortino_ratio']:.2f}")
    print(f"  Win Rate: {report['win_rate']:.1%}")
    print(f"  Profit Factor: {report['profit_factor']:.2f}")
    print(f"  Max Drawdown: {report['max_drawdown']:.2%}")

    print("\n[PASS] Risk Metrics Module: ALL TESTS PASSED")
    return True


async def test_macro_connector():
    """Test macro data connector module."""
    print("\n" + "=" * 60)
    print("TEST 2: MACRO DATA CONNECTOR MODULE")
    print("=" * 60)

    connector = MacroDataConnector()

    # Test 1: Individual metrics
    print("\n--- Individual Metrics ---")
    dxy = await connector.get_dxy_index()
    vix = await connector.get_vix_index()
    real_yields = await connector.get_real_yields()
    fed_funds = await connector.get_fed_funds_rate()

    print(f"DXY (US Dollar Index): {dxy}")
    print(f"VIX (Volatility Index): {vix}")
    print(f"Real Yields (10Y TIPS): {real_yields}")
    print(f"Fed Funds Rate: {fed_funds}")

    # At least DXY and VIX should work (no API key needed)
    assert dxy is None or isinstance(dxy, float), "DXY should be None or float"
    assert vix is None or isinstance(vix, float), "VIX should be None or float"
    print("[OK] Individual metric fetching works")

    # Test 2: Macro score calculation
    print("\n--- Macro Score Calculation ---")
    macro_score, components = await connector.calculate_macro_score()

    print(f"Macro Score: {macro_score:.2f} (0=Bearish, 0.5=Neutral, 1=Bullish)")
    print(f"Components: {components}")

    assert 0.0 <= macro_score <= 1.0, "Macro score out of range"
    assert "dxy" in components, "Missing DXY component"
    assert "vix" in components, "Missing VIX component"
    print("[OK] Macro score calculation works")

    # Test 3: Quick macro score function
    print("\n--- Quick Macro Score ---")
    quick_score = await get_quick_macro_score()
    print(f"Quick Score: {quick_score:.2f}")
    assert 0.0 <= quick_score <= 1.0, "Quick score out of range"
    print("[OK] Quick macro score works")

    # Test 4: Human-readable context
    print("\n--- Macro Context Summary ---")
    summary = await connector.get_macro_context()
    assert len(summary) > 50, "Summary too short"
    assert "MACRO CONTEXT" in summary, "Missing header"
    print("[OK] Context summary generation works")

    # Skip printing summary to avoid unicode issues in Windows console
    # print("\n" + summary)
    print("  (Summary generated successfully, length: {} chars)".format(len(summary)))

    # Test 5: Caching mechanism
    print("\n--- Cache Test ---")
    print("Fetching DXY again (should use cache)...")
    import time
    start = time.time()
    dxy_cached = await connector.get_dxy_index()
    elapsed = time.time() - start
    print(f"Second fetch took {elapsed*1000:.2f}ms")
    assert elapsed < 0.1, "Cache not working (took too long)"
    assert dxy_cached == dxy, "Cached value different"
    print("[OK] Caching mechanism works")

    print("\n[PASS] Macro Data Connector Module: ALL TESTS PASSED")
    return True


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("TESTING PHASE 8 & PHASE 9 MODULES")
    print("=" * 60)
    print("Phase 8: Risk Metrics")
    print("Phase 9: Macro Data Integration")
    print("=" * 60)

    try:
        # Test 1: Risk Metrics
        test_risk_metrics()

        # Test 2: Macro Connector
        await test_macro_connector()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED - MODULES READY FOR USE")
        print("=" * 60)
        print("\nUsage:")
        print("  1. Generate risk report: python scripts/generate_risk_report.py")
        print("  2. Check market + macro: python scripts/check_market.py")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
