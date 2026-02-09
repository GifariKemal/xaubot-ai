"""
Telegram Notification Helpers
=============================
Extracts all notification logic from main_live.py into a single module.

This module handles:
  - Building context dicts from bot state
  - Sending trade open/close notifications
  - Sending market updates, hourly reports
  - Sending critical alerts, emergency notifications
  - Startup & shutdown notifications

Integration:
    from src.telegram_notifications import TelegramNotifications
    self.notifications = TelegramNotifications(bot)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import datetime
from loguru import logger


class TelegramNotifications:
    """
    High-level notification helper that reads bot state and sends
    formatted Telegram messages via bot.telegram (TelegramNotifier).
    """

    def __init__(self, bot):
        """
        Args:
            bot: TradingBot instance (has .telegram, .mt5, .smart_risk, etc.)
        """
        self.bot = bot

    # ------------------------------------------------------------------
    # Startup notification
    # ------------------------------------------------------------------
    async def send_startup(self):
        """Send bot startup notification with full context."""
        bot = self.bot
        balance = bot.mt5.account_balance or bot.config.capital
        session_status = bot.session_filter.get_status_report()
        risk_state = bot.smart_risk.get_state()
        risk_rec = bot.smart_risk.get_trading_recommendation()
        ml_status = (
            f"Loaded ({len(bot.ml_model.feature_names)} features)"
            if bot.ml_model.fitted
            else "Not loaded"
        )

        ctx = {
            "risk_per_trade": bot.config.risk.risk_per_trade,
            "max_daily_loss": bot.config.risk.max_daily_loss,
            "max_total_loss": bot.smart_risk.max_total_loss_percent,
            "max_lot": bot.smart_risk.max_lot_size,
            "max_positions": bot.smart_risk.max_concurrent_positions,
            "cooldown_seconds": bot._trade_cooldown_seconds,
            "daily_loss": risk_state.daily_loss,
            "total_loss": bot.smart_risk._total_loss,
            "consecutive_losses": risk_state.consecutive_losses,
            "risk_mode": risk_rec.get("mode", "normal"),
            "session": session_status.get("current_session", "Unknown"),
            "can_trade": session_status.get("can_trade", False),
            "volatility": session_status.get("volatility", "unknown"),
        }

        await bot.telegram.send_startup_message(
            symbol=bot.config.symbol,
            capital=bot.config.capital,
            balance=balance,
            mode=bot.config.capital_mode.value,
            ml_model_status=ml_status,
            news_status="DISABLED",
            context=ctx,
        )

    # ------------------------------------------------------------------
    # Shutdown notification
    # ------------------------------------------------------------------
    async def send_shutdown(self):
        """Send bot shutdown notification with session summary."""
        bot = self.bot
        try:
            balance = bot.mt5.account_balance or bot.config.capital
            uptime_hours = (datetime.now() - bot._start_time).total_seconds() / 3600
            risk_state = bot.smart_risk.get_state()

            ctx = {
                "risk_mode": bot.smart_risk.get_trading_recommendation().get("mode", "normal"),
                "daily_loss": risk_state.daily_loss,
                "daily_profit": risk_state.daily_profit,
                "total_loss": bot.smart_risk._total_loss,
                "consecutive_losses": risk_state.consecutive_losses,
                "session": bot.session_filter.get_status_report().get("current_session", "Unknown"),
            }

            await bot.telegram.send_shutdown_message(
                balance=balance,
                total_trades=bot._total_session_trades,
                total_profit=bot._total_session_profit,
                uptime_hours=uptime_hours,
                context=ctx,
            )
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")

    # ------------------------------------------------------------------
    # Trade close — smart position manager
    # ------------------------------------------------------------------
    async def notify_trade_close_smart(
        self,
        ticket: int,
        profit: float,
        current_price: float,
        reason: str,
    ):
        """Send notification for smart close (from SmartRiskManager)."""
        bot = self.bot
        try:
            trade_info = bot._open_trade_info.pop(ticket, {})

            balance_before = trade_info.get("balance_before", 0)
            balance_after = bot.mt5.account_balance or 0
            entry_price = trade_info.get("entry_price", current_price)
            duration = int(
                (datetime.now() - trade_info.get("open_time", datetime.now())).total_seconds()
            )

            # Track stats
            bot._total_session_profit += profit
            bot._total_session_trades += 1
            if profit > 0:
                bot._total_session_wins += 1

            ctx = self._build_close_context(reason)

            await bot.telegram.notify_trade_close(
                ticket=ticket,
                symbol=bot.config.symbol,
                order_type=trade_info.get("direction", "BUY"),
                lot_size=trade_info.get("lot_size", 0.01),
                entry_price=entry_price,
                close_price=current_price,
                profit=profit,
                profit_pips=(current_price - entry_price) / 0.1,
                balance_before=balance_before,
                balance_after=balance_after,
                duration_seconds=duration,
                ml_confidence=trade_info.get("ml_confidence", 0),
                regime=trade_info.get("regime", "unknown"),
                volatility=trade_info.get("volatility", "unknown"),
                context=ctx,
            )
        except Exception as e:
            logger.warning(f"Failed to send close notification: {e}")

    # ------------------------------------------------------------------
    # Trade close — position manager action
    # ------------------------------------------------------------------
    async def notify_trade_close_action(self, action, current_price: float):
        """Send notification for close via PositionManager action."""
        bot = self.bot
        try:
            ticket = action.ticket
            trade_info = bot._open_trade_info.pop(ticket, {})
            entry_price = trade_info.get("entry_price", current_price)
            open_time = trade_info.get("open_time", datetime.now())
            balance_before = trade_info.get("balance_before", bot._daily_start_balance)
            ml_confidence = trade_info.get("ml_confidence", 0)
            regime = trade_info.get("regime", "unknown")
            volatility = trade_info.get("volatility", "unknown")

            balance_after = bot.mt5.account_balance or bot.config.capital

            profit = action.profit if hasattr(action, "profit") else 0
            if profit == 0:
                profit = balance_after - balance_before

            duration_seconds = int((datetime.now() - open_time).total_seconds())

            price_diff = current_price - entry_price
            profit_pips = price_diff / 0.1 if "XAU" in bot.config.symbol else price_diff / 0.0001

            # Track session stats
            bot._total_session_profit += profit
            bot._total_session_trades += 1
            if profit > 0:
                bot._total_session_wins += 1

            exit_reason = action.reason if hasattr(action, "reason") else "position_manager"
            ctx = self._build_close_context(exit_reason)

            await bot.telegram.notify_trade_close(
                ticket=ticket,
                symbol=bot.config.symbol,
                order_type=trade_info.get("direction", "BUY"),
                lot_size=trade_info.get("lot_size", 0.01),
                entry_price=entry_price,
                close_price=current_price,
                profit=profit,
                profit_pips=profit_pips,
                balance_before=balance_before,
                balance_after=balance_after,
                duration_seconds=duration_seconds,
                ml_confidence=ml_confidence,
                regime=regime,
                volatility=volatility,
                context=ctx,
            )
        except Exception as e:
            logger.warning(f"Failed to send trade close notification: {e}")

    # ------------------------------------------------------------------
    # Trade open notification
    # ------------------------------------------------------------------
    async def notify_trade_open(
        self,
        result,
        signal,
        position,
        regime: str,
        volatility: str,
        session_status: dict,
        *,
        safe_mode: bool = False,
        smc_fvg: bool = False,
        smc_ob: bool = False,
        smc_bos: bool = False,
        smc_choch: bool = False,
        dynamic_threshold=None,
        market_quality=None,
        market_score=None,
    ):
        """Send trade open notification."""
        bot = self.bot

        # Store trade info for close notification (only if not already stored)
        # Safe mode pre-stores with actual fill price/lot, so don't overwrite
        if result.order_id not in bot._open_trade_info:
            bot._open_trade_info[result.order_id] = {
                "entry_price": signal.entry_price,
                "open_time": datetime.now(),
                "balance_before": bot.mt5.account_balance,
                "ml_confidence": signal.confidence,
                "regime": regime,
                "volatility": volatility,
                "direction": signal.signal_type,
                "lot_size": position.lot_size,
            }

        risk_state = bot.smart_risk.get_state()
        risk_rec = bot.smart_risk.get_trading_recommendation()

        ctx = {
            "dynamic_threshold": (
                float(dynamic_threshold)
                if dynamic_threshold is not None
                else getattr(bot, "_last_dynamic_threshold", bot.config.ml.confidence_threshold)
            ),
            "market_quality": (
                str(market_quality)
                if market_quality is not None
                else getattr(bot, "_last_market_quality", "unknown")
            ),
            "market_score": (
                int(market_score)
                if market_score is not None
                else getattr(bot, "_last_market_score", 0)
            ),
            "smc_signal": getattr(bot, "_last_raw_smc_signal", ""),
            "smc_confidence": getattr(bot, "_last_raw_smc_confidence", 0),
            "smc_fvg": smc_fvg or getattr(signal, "fvg_detected", False),
            "smc_ob": smc_ob or getattr(signal, "ob_detected", False),
            "smc_bos": smc_bos or getattr(signal, "bos_detected", False),
            "smc_choch": smc_choch or getattr(signal, "choch_detected", False),
            "session": session_status.get("current_session", "Unknown"),
            "h1_bias": getattr(bot, "_h1_bias_cache", "NEUTRAL"),
            "risk_mode": risk_rec.get("mode", "normal"),
            "daily_loss": risk_state.daily_loss,
            "consecutive_losses": risk_state.consecutive_losses,
            "entry_filters": getattr(bot, "_last_filter_results", []),
        }

        reason = f"SAFE MODE: {signal.reason}" if safe_mode else signal.reason
        sl = 0 if safe_mode else signal.stop_loss

        try:
            await bot.telegram.notify_trade_open(
                ticket=result.order_id,
                symbol=bot.config.symbol,
                order_type=signal.signal_type,
                lot_size=position.lot_size,
                entry_price=signal.entry_price,
                stop_loss=sl,
                take_profit=signal.take_profit,
                ml_confidence=signal.confidence,
                signal_reason=reason,
                regime=regime,
                volatility=volatility,
                context=ctx,
            )
        except Exception as e:
            logger.warning(f"Failed to send trade open notification: {e}")

    # ------------------------------------------------------------------
    # Critical limit alert
    # ------------------------------------------------------------------
    async def send_critical_limit_alert(
        self,
        limit_type: str,
        current_loss: float,
        max_loss: float,
        max_percent: float,
    ):
        """Send critical alert when loss limits are reached."""
        logger.critical("=" * 60)
        logger.critical(f"CRITICAL: {limit_type} REACHED!")
        logger.critical(f"Loss: ${current_loss:.2f} / ${max_loss:.2f} ({max_percent}%)")
        logger.critical("TRADING HAS BEEN STOPPED!")
        logger.critical("=" * 60)

        try:
            if limit_type == "TOTAL LOSS LIMIT":
                message = (
                    f"🚨🚨 CRITICAL: TOTAL LOSS LIMIT REACHED 🚨🚨\n\n"
                    f"Total Loss: ${current_loss:.2f}\n"
                    f"Limit: ${max_loss:.2f} ({max_percent}%)\n\n"
                    f"⛔ TRADING STOPPED PERMANENTLY\n"
                    f"Manual reset required to resume trading.\n\n"
                    f"Please review your trading strategy."
                )
            else:
                message = (
                    f"🚨 DAILY LOSS LIMIT REACHED 🚨\n\n"
                    f"Daily Loss: ${current_loss:.2f}\n"
                    f"Limit: ${max_loss:.2f} ({max_percent}%)\n\n"
                    f"⛔ TRADING STOPPED FOR TODAY\n"
                    f"Will resume tomorrow automatically."
                )

            await self.bot.telegram.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")

    # ------------------------------------------------------------------
    # Emergency close notification
    # ------------------------------------------------------------------
    async def send_emergency_close_result(
        self,
        closed_count: int,
        failed_tickets: list,
    ):
        """Send notification after emergency close attempt."""
        try:
            if failed_tickets:
                await self.bot.telegram.send_message(
                    f"🚨 EMERGENCY CLOSE FAILED!\n\n"
                    f"Failed tickets: {failed_tickets}\n"
                    f"Please close manually!"
                )
            else:
                await self.bot.telegram.send_message(
                    f"🚨 EMERGENCY CLOSE COMPLETE\n\n"
                    f"Closed {closed_count} positions due to flash crash detection"
                )
        except Exception:
            pass  # Don't let telegram failure stop us

    async def send_flash_crash_critical(self, move_pct: float, error):
        """Send critical alert when flash crash emergency close fails."""
        try:
            await self.bot.telegram.send_message(
                f"🚨🚨 CRITICAL ERROR 🚨🚨\n\n"
                f"Flash crash detected but emergency close FAILED!\n"
                f"Error: {error}\n\n"
                f"MANUAL INTERVENTION REQUIRED!"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Market update (on-demand via command, not auto-sent)
    # ------------------------------------------------------------------
    async def send_market_update(self, df, regime_state, ml_prediction):
        """Send market update to Telegram."""
        bot = self.bot
        try:
            now = datetime.now()

            if bot._last_market_update_time:
                time_since = (now - bot._last_market_update_time).total_seconds()
                if time_since < 1800:
                    return

            session_status = bot.session_filter.get_status_report()

            atr = df["atr"].tail(1).item() if "atr" in df.columns else 0
            tick = bot.mt5.get_tick(bot.config.symbol)
            spread = (tick.ask - tick.bid) if tick else 0

            if "ema_9" in df.columns and "ema_21" in df.columns:
                ema_9 = df["ema_9"].tail(1).item()
                ema_21 = df["ema_21"].tail(1).item()
                trend_direction = "UPTREND" if ema_9 > ema_21 else "DOWNTREND"
            else:
                trend_direction = "NEUTRAL"

            ctx = {
                "h1_bias": getattr(bot, "_h1_bias_cache", "NEUTRAL"),
                "dynamic_threshold": getattr(bot, "_last_dynamic_threshold", 0.55),
                "market_quality": getattr(bot, "_last_market_quality", "unknown"),
                "market_score": getattr(bot, "_last_market_score", 0),
                "smc_signal": getattr(bot, "_last_raw_smc_signal", ""),
                "smc_confidence": getattr(bot, "_last_raw_smc_confidence", 0),
                "consecutive_losses": bot.smart_risk.get_state().consecutive_losses,
                "risk_mode": bot.smart_risk.get_trading_recommendation().get("mode", "normal"),
                "daily_loss": bot.smart_risk.get_state().daily_loss,
                "session_trades": bot._total_session_trades,
                "session_profit": bot._total_session_profit,
            }

            await bot.telegram.notify_market_update(
                symbol=bot.config.symbol,
                price=df["close"].tail(1).item(),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "unknown"),
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                trend_direction=trend_direction,
                session=session_status.get("current_session", "Unknown"),
                can_trade=session_status.get("can_trade", True),
                atr=atr,
                spread=spread,
                context=ctx,
            )

            bot._last_market_update_time = now
            logger.info("Telegram: Market update sent")

        except Exception as e:
            logger.warning(f"Failed to send market update: {e}")

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------
    async def send_daily_summary(self):
        """Send daily trading summary to Telegram."""
        bot = self.bot
        try:
            balance = bot.mt5.account_balance or bot.config.capital
            await bot.telegram.send_daily_summary(
                start_balance=bot._daily_start_balance,
                end_balance=balance,
            )
            logger.info("Telegram: Daily summary sent")
        except Exception as e:
            logger.warning(f"Failed to send daily summary: {e}")

    # ------------------------------------------------------------------
    # Hourly analysis report
    # ------------------------------------------------------------------
    async def send_hourly_analysis_if_due(
        self,
        df,
        regime_state,
        ml_prediction,
        open_positions,
        current_price: float,
    ):
        """Send comprehensive hourly analysis report. Interval: 1 hour."""
        bot = self.bot
        now = datetime.now()

        if bot._last_hourly_report_time:
            time_since = (now - bot._last_hourly_report_time).total_seconds()
            if time_since < 3600:
                return

        try:
            balance = bot.mt5.account_balance or bot.config.capital
            equity = bot.mt5.account_equity or bot.config.capital
            floating_pnl = equity - balance

            # Position details with Smart Risk data
            position_details = []
            for row in open_positions.iter_rows(named=True):
                ticket = row["ticket"]
                profit = row.get("profit", 0)
                position_type = row.get("type", 0)
                direction = "BUY" if position_type == 0 else "SELL"

                guard = bot.smart_risk._position_guards.get(ticket)
                momentum = guard.momentum_score if guard else 0
                tp_prob = guard.get_tp_probability() if guard else 50

                position_details.append({
                    "ticket": ticket,
                    "direction": direction,
                    "profit": profit,
                    "momentum": momentum,
                    "tp_probability": tp_prob,
                })

            session_status = bot.session_filter.get_status_report()

            market_analysis = bot.dynamic_confidence.analyze_market(
                session=session_status.get("current_session", "Unknown"),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "medium"),
                trend_direction=regime_state.regime.value if regime_state else "neutral",
                has_smc_signal=False,
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
            )

            risk_rec = bot.smart_risk.get_trading_recommendation()

            avg_exec = (
                (sum(bot._execution_times) / len(bot._execution_times) * 1000)
                if bot._execution_times
                else 0
            )
            uptime = (now - bot._start_time).total_seconds() / 3600

            # Get ATR and spread
            atr = 0
            spread = 0
            try:
                df_latest = bot.mt5.get_market_data(
                    symbol=bot.config.symbol,
                    timeframe=bot.config.execution_timeframe,
                    count=50,
                )
                if len(df_latest) > 0 and "atr" in df_latest.columns:
                    atr = df_latest["atr"].tail(1).item()
                tick = bot.mt5.get_tick(bot.config.symbol)
                spread = (tick.ask - tick.bid) if tick else 0
            except Exception:
                pass

            ctx = {
                "h1_bias": getattr(bot, "_h1_bias_cache", "NEUTRAL"),
                "smc_signal": getattr(bot, "_last_raw_smc_signal", ""),
                "smc_confidence": getattr(bot, "_last_raw_smc_confidence", 0),
                "atr": atr,
                "spread": spread,
                "total_loss": bot.smart_risk._total_loss,
                "consecutive_losses": bot.smart_risk.get_state().consecutive_losses,
                "entry_filters": getattr(bot, "_last_filter_results", []),
            }

            await bot.telegram.send_hourly_analysis(
                balance=balance,
                equity=equity,
                floating_pnl=floating_pnl,
                open_positions=len(open_positions),
                position_details=position_details,
                symbol=bot.config.symbol,
                current_price=current_price,
                session=session_status.get("current_session", "Unknown"),
                regime=regime_state.regime.value if regime_state else "unknown",
                volatility=session_status.get("volatility", "unknown"),
                ml_signal=ml_prediction.signal,
                ml_confidence=ml_prediction.confidence,
                dynamic_threshold=market_analysis.confidence_threshold,
                market_quality=market_analysis.quality.value,
                market_score=market_analysis.score,
                daily_pnl=bot._total_session_profit,
                daily_trades=bot._total_session_trades,
                risk_mode=risk_rec.get("mode", "normal"),
                max_daily_loss=bot.smart_risk.max_daily_loss_usd,
                uptime_hours=uptime,
                total_loops=bot._loop_count,
                avg_execution_ms=avg_exec,
                news_status="DISABLED",
                news_reason="News agent disabled",
                context=ctx,
            )

            bot._last_hourly_report_time = now
            logger.info("Telegram: Hourly analysis report sent")

        except Exception as e:
            logger.warning(f"Failed to send hourly analysis: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_close_context(self, exit_reason: str) -> dict:
        """Build context dict for trade close notifications."""
        bot = self.bot
        risk_state = bot.smart_risk.get_state()
        win_rate = (
            (bot._total_session_wins / bot._total_session_trades * 100)
            if bot._total_session_trades > 0
            else 0
        )
        return {
            "exit_reason": exit_reason,
            "risk_mode": bot.smart_risk.get_trading_recommendation().get("mode", "normal"),
            "daily_loss": risk_state.daily_loss,
            "daily_profit": risk_state.daily_profit,
            "consecutive_losses": risk_state.consecutive_losses,
            "total_loss": bot.smart_risk._total_loss,
            "session_trades": bot._total_session_trades,
            "session_wins": bot._total_session_wins,
            "session_profit": bot._total_session_profit,
            "win_rate": win_rate,
            "session": bot.session_filter.get_status_report().get("current_session", "Unknown"),
        }
