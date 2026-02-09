"""
Profit Momentum Tracker - Code Snippets for Integration
========================================================

Copy-paste these snippets into main_live.py for integration.
"""

# ============================================================
# SNIPPET 1: Import Statement (add to top of main_live.py)
# ============================================================
from src.profit_momentum_tracker import ProfitMomentumTracker


# ============================================================
# SNIPPET 2: Initialize Tracker (add to TradingBot.__init__)
# ============================================================
# Initialize Profit Momentum Tracker (NEW)
self.momentum_tracker = ProfitMomentumTracker(
    # Velocity thresholds
    velocity_reversal_threshold=-0.5,  # Exit if velocity < -0.5 $/s
    deceleration_threshold=-1.0,       # Exit if accel < -1.0 $/s²
    stagnation_threshold=0.1,          # Velocity < 0.1 $/s = stagnant
    stagnation_count_max=8,            # Exit after 8 stagnant samples (4s)

    # Drawdown protection
    peak_drawdown_threshold=40.0,      # Exit if 40% drawdown from peak
    min_peak_to_protect=10.0,          # Only protect peaks > $10

    # Anti-early-exit protection
    min_profit_for_momentum_exit=5.0,  # Don't exit on momentum if profit < $5
    grace_period_seconds=10.0,         # Minimum 10s in profit before momentum exit
    min_samples_required=6,            # Minimum 6 samples (3s) before analyzing

    # Logging
    enable_logging=True,
)

# Pass tracker to Position Manager
self.position_manager = SmartPositionManager(
    breakeven_pips=30.0,
    trail_start_pips=50.0,
    trail_step_pips=30.0,
    atr_be_mult=2.0,
    atr_trail_start_mult=4.0,
    atr_trail_step_mult=3.0,
    min_profit_to_protect=5.0,
    max_drawdown_from_peak=50.0,
    enable_market_close_handler=True,
    min_profit_before_close=10.0,
    max_loss_to_hold=100.0,
    momentum_tracker=self.momentum_tracker,  # NEW: Pass tracker
    enable_momentum_exit=True,               # NEW: Enable momentum exits
)

# Initialize momentum log throttle
self._last_momentum_log = {}


# ============================================================
# SNIPPET 3: Monitoring Method (add to TradingBot class)
# ============================================================
async def _monitor_positions_momentum(self):
    """
    Monitor open positions momentum every 500ms.
    Updates profit tracker for real-time momentum analysis.
    """
    logger.info("🎯 Profit momentum monitoring started (500ms interval)")

    while self.running:
        try:
            # Get open positions
            positions_df = self.mt5.get_positions()

            if len(positions_df) > 0:
                # Update momentum tracker for each position
                for row in positions_df.iter_rows(named=True):
                    ticket = row["ticket"]
                    profit = row.get("profit", 0.0)
                    current_price = row.get("price_current", 0.0)

                    # Update tracker
                    self.momentum_tracker.update(ticket, profit, current_price)

                    # Log metrics every 2 seconds per ticket
                    if self._should_log_momentum(ticket):
                        summary = self.momentum_tracker.get_position_summary(ticket)
                        if summary:
                            logger.debug(
                                f"#{ticket} | "
                                f"Profit: ${summary['current_profit']:.2f} | "
                                f"Peak: ${summary['peak_profit']:.2f} | "
                                f"Vel: {summary['velocity']:.2f} $/s | "
                                f"Momentum: {summary['momentum']} | "
                                f"Drawdown: {summary['drawdown_pct']:.1f}%"
                            )

            # Wait 500ms before next update
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Momentum monitoring error: {e}")
            await asyncio.sleep(0.5)


def _should_log_momentum(self, ticket: int) -> bool:
    """
    Throttle momentum logging to every 2 seconds per ticket.

    Args:
        ticket: MT5 ticket number

    Returns:
        bool: True if should log now
    """
    now = time.time()
    last_log = self._last_momentum_log.get(ticket, 0)

    if now - last_log >= 2.0:  # Log every 2 seconds
        self._last_momentum_log[ticket] = now
        return True
    return False


# ============================================================
# SNIPPET 4: Start Monitoring Task (modify run() method)
# ============================================================
async def run(self):
    """Main trading loop with momentum monitoring."""
    self.running = True

    logger.info("🚀 Starting trading bot...")
    logger.info(f"Capital Mode: {self.config.capital_mode.value}")
    logger.info(f"Risk per Trade: {self.risk_engine.risk_percent}%")
    logger.info(f"Symbol: {self.config.symbol}")

    # Start background tasks
    tasks = [
        asyncio.create_task(self._trading_loop(), name="trading_loop"),
        asyncio.create_task(self._monitor_positions_momentum(), name="momentum_monitor"),  # NEW
    ]

    try:
        # Wait for all tasks
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        logger.warning("⚠️  Shutdown signal received")
        self.running = False

    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
        self.running = False

    finally:
        # Cleanup
        logger.info("🛑 Shutting down...")
        for task in tasks:
            if not task.done():
                task.cancel()

        # Disconnect MT5
        if not self.simulation:
            self.mt5.disconnect()

        logger.success("✅ Shutdown complete")


# ============================================================
# SNIPPET 5: Optional - Enhanced Position Summary Logging
# ============================================================
def log_position_summary_with_momentum(self):
    """
    Log detailed position summary including momentum metrics.
    Call this periodically in trading loop.
    """
    positions_df = self.mt5.get_positions()

    if len(positions_df) > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPEN POSITIONS: {len(positions_df)}")
        logger.info(f"{'='*60}")

        for row in positions_df.iter_rows(named=True):
            ticket = row["ticket"]
            pos_type = row.get("type", "UNKNOWN")
            profit = row.get("profit", 0.0)
            volume = row.get("volume", 0.0)

            # Get momentum summary
            momentum_summary = self.momentum_tracker.get_position_summary(ticket)

            if momentum_summary:
                logger.info(
                    f"  #{ticket} | {pos_type} {volume:.2f} lot | "
                    f"Profit: ${profit:.2f} | "
                    f"Peak: ${momentum_summary['peak_profit']:.2f} | "
                    f"Velocity: {momentum_summary['velocity']:.2f} $/s | "
                    f"Momentum: {momentum_summary['momentum']} | "
                    f"Samples: {momentum_summary['samples']} | "
                    f"Time in Profit: {momentum_summary['time_in_profit']:.1f}s"
                )
            else:
                logger.info(
                    f"  #{ticket} | {pos_type} {volume:.2f} lot | "
                    f"Profit: ${profit:.2f} (no momentum data yet)"
                )

        logger.info(f"{'='*60}\n")


# ============================================================
# EXAMPLE USAGE IN MAIN
# ============================================================
if __name__ == "__main__":
    # Create bot instance
    bot = TradingBot(simulation=False)

    # Run with asyncio
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
