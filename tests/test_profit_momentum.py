"""
Test Profit Momentum Tracker
=============================
Demo dan test untuk profit momentum tracking system.
"""

import sys
import os
import time
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profit_momentum_tracker import ProfitMomentumTracker
from loguru import logger


def simulate_profit_pattern_1():
    """
    Simulate Pattern 1: Steady Growth then Reversal
    - Profit grows steadily
    - Peaks at $50
    - Then reverses slowly

    Expected: Should exit around $45-$47 (90-94% of peak)
    """
    logger.info("=" * 60)
    logger.info("PATTERN 1: Steady Growth → Reversal")
    logger.info("=" * 60)

    tracker = ProfitMomentumTracker(
        enable_logging=True,
        min_profit_for_momentum_exit=5.0,
        grace_period_seconds=3.0,
    )

    ticket = 123456
    price = 2650.0

    # Phase 1: Steady growth (0-10s)
    logger.info("\n📈 Phase 1: Steady Growth (0-10s)")
    for i in range(20):  # 10 seconds at 500ms interval
        profit = i * 2.5  # Linear growth to $50
        price += 0.5

        tracker.update(ticket, profit, price)
        time.sleep(0.5)

        if i % 4 == 0:  # Log every 2 seconds
            metrics = tracker.calculate_metrics(ticket)
            if metrics:
                logger.info(
                    f"  t={i*0.5:.1f}s | Profit: ${profit:.2f} | "
                    f"Velocity: {metrics.velocity:.2f} $/s | "
                    f"Momentum: {metrics.momentum_direction}"
                )

    # Phase 2: Peak stagnation (10-13s)
    logger.info("\n⏸️  Phase 2: Peak Stagnation (10-13s)")
    for i in range(6):  # 3 seconds
        profit = 50.0 + random.uniform(-0.5, 0.5)  # Stagnant around $50
        price += random.uniform(-0.1, 0.1)

        tracker.update(ticket, profit, price)
        should_exit, reason = tracker.should_exit(ticket, profit)

        if should_exit:
            logger.warning(f"🚨 EXIT SIGNAL: {reason}")
            break

        time.sleep(0.5)

    # Phase 3: Slow reversal (13-20s)
    logger.info("\n📉 Phase 3: Slow Reversal (13-20s)")
    for i in range(14):  # 7 seconds
        profit = 50.0 - (i * 1.5)  # Decline from $50
        price -= 0.3

        tracker.update(ticket, profit, price)
        should_exit, reason = tracker.should_exit(ticket, profit)

        metrics = tracker.calculate_metrics(ticket)
        if metrics and i % 2 == 0:
            logger.info(
                f"  t={13+i*0.5:.1f}s | Profit: ${profit:.2f} | "
                f"Velocity: {metrics.velocity:.2f} $/s | "
                f"Peak Drawdown: {metrics.drawdown_from_peak:.1f}%"
            )

        if should_exit:
            logger.warning(f"🚨 EXIT SIGNAL at ${profit:.2f}: {reason}")
            summary = tracker.get_position_summary(ticket)
            logger.success(
                f"✅ Exit Summary: Peak ${summary['peak_profit']:.2f} → "
                f"Exit ${profit:.2f} ({summary['drawdown_pct']:.1f}% from peak)"
            )
            break

        time.sleep(0.5)


def simulate_profit_pattern_2():
    """
    Simulate Pattern 2: Quick Spike then Sharp Reversal
    - Profit spikes quickly to $40
    - Reverses sharply

    Expected: Should exit quickly on velocity reversal
    """
    logger.info("\n" + "=" * 60)
    logger.info("PATTERN 2: Quick Spike → Sharp Reversal")
    logger.info("=" * 60)

    tracker = ProfitMomentumTracker(
        enable_logging=True,
        velocity_reversal_threshold=-1.0,  # More sensitive
        min_profit_for_momentum_exit=5.0,
    )

    ticket = 234567
    price = 2650.0

    # Phase 1: Quick spike (0-4s)
    logger.info("\n🚀 Phase 1: Quick Spike (0-4s)")
    for i in range(8):  # 4 seconds
        profit = i * 5.0  # Fast growth to $40
        price += 1.0

        tracker.update(ticket, profit, price)
        time.sleep(0.5)

        metrics = tracker.calculate_metrics(ticket)
        if metrics and i % 2 == 0:
            logger.info(
                f"  t={i*0.5:.1f}s | Profit: ${profit:.2f} | "
                f"Velocity: {metrics.velocity:.2f} $/s"
            )

    # Phase 2: Sharp reversal (4-8s)
    logger.info("\n💥 Phase 2: Sharp Reversal (4-8s)")
    for i in range(8):  # 4 seconds
        profit = 40.0 - (i * 4.0)  # Fast decline
        price -= 0.8

        tracker.update(ticket, profit, price)
        should_exit, reason = tracker.should_exit(ticket, profit)

        metrics = tracker.calculate_metrics(ticket)
        if metrics:
            logger.info(
                f"  t={4+i*0.5:.1f}s | Profit: ${profit:.2f} | "
                f"Velocity: {metrics.velocity:.2f} $/s | "
                f"Accel: {metrics.acceleration:.2f} $/s²"
            )

        if should_exit:
            logger.warning(f"🚨 EXIT SIGNAL at ${profit:.2f}: {reason}")
            summary = tracker.get_position_summary(ticket)
            logger.success(
                f"✅ Exit Summary: Peak ${summary['peak_profit']:.2f} → "
                f"Exit ${profit:.2f}"
            )
            break

        time.sleep(0.5)


def simulate_profit_pattern_3():
    """
    Simulate Pattern 3: Healthy Trend (No Exit)
    - Profit grows steadily
    - Small pullbacks but momentum stays positive

    Expected: Should NOT exit (healthy momentum)
    """
    logger.info("\n" + "=" * 60)
    logger.info("PATTERN 3: Healthy Trend (No Exit Expected)")
    logger.info("=" * 60)

    tracker = ProfitMomentumTracker(
        enable_logging=True,
        peak_drawdown_threshold=50.0,  # Allow larger drawdown
    )

    ticket = 345678
    price = 2650.0

    # Simulate 15 seconds of healthy growth with small pullbacks
    logger.info("\n📊 Simulating healthy trend with pullbacks...")
    for i in range(30):  # 15 seconds
        # Add some volatility but overall uptrend
        base_profit = i * 1.5
        noise = random.uniform(-2.0, 3.0)  # Slight upward bias
        profit = base_profit + noise

        price += random.uniform(-0.2, 0.5)

        tracker.update(ticket, profit, price)
        should_exit, reason = tracker.should_exit(ticket, profit)

        if i % 4 == 0:  # Log every 2 seconds
            metrics = tracker.calculate_metrics(ticket)
            if metrics:
                logger.info(
                    f"  t={i*0.5:.1f}s | Profit: ${profit:.2f} | "
                    f"Peak: ${metrics.peak_profit:.2f} | "
                    f"Velocity: {metrics.velocity:.2f} $/s | "
                    f"Status: {metrics.momentum_direction}"
                )

        if should_exit:
            logger.warning(f"⚠️  Unexpected exit: {reason}")
            break

        time.sleep(0.5)

    if not should_exit:
        logger.success("✅ No exit triggered - Healthy trend maintained!")
        summary = tracker.get_position_summary(ticket)
        if summary:
            logger.info(
                f"Final Stats: Peak ${summary['peak_profit']:.2f}, "
                f"Current ${summary['current_profit']:.2f}, "
                f"Velocity {summary['velocity']:.2f} $/s"
            )


def main():
    """Run all simulation patterns."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{message}</level>",
        level="INFO",
    )

    logger.info("🧪 Profit Momentum Tracker - Simulation Tests")
    logger.info("=" * 60)

    try:
        # Run pattern simulations
        simulate_profit_pattern_1()
        time.sleep(2)

        simulate_profit_pattern_2()
        time.sleep(2)

        simulate_profit_pattern_3()

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Simulation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.success("🎉 All simulations completed!")


if __name__ == "__main__":
    main()
