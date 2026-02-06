"""
Telegram Notifier Module
========================
Smart Telegram integration for AI Trading Bot.

Features:
- Trade notifications with detailed P/L
- Market condition updates (educational)
- ML prediction insights
- Volatility alerts
- Daily summary with charts
- Interactive commands
- PDF report generation
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo
import io

from loguru import logger

# Timezone
WIB = ZoneInfo("Asia/Jakarta")


class NotificationType(Enum):
    """Types of Telegram notifications."""
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    MARKET_UPDATE = "market_update"
    DAILY_SUMMARY = "daily_summary"
    ALERT = "alert"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class TradeInfo:
    """Trade information for notifications."""
    ticket: int
    symbol: str
    order_type: str  # BUY or SELL
    lot_size: float
    entry_price: float
    close_price: Optional[float] = None
    stop_loss: float = 0
    take_profit: float = 0
    profit: float = 0
    profit_pips: float = 0
    balance_before: float = 0
    balance_after: float = 0
    duration_seconds: int = 0
    ml_confidence: float = 0
    signal_reason: str = ""
    regime: str = ""
    volatility: str = ""


@dataclass
class MarketCondition:
    """Market condition information."""
    symbol: str
    price: float
    regime: str
    volatility: str
    ml_signal: str
    ml_confidence: float
    trend_direction: str
    session: str
    can_trade: bool
    atr: float = 0
    spread: float = 0


class TelegramNotifier:
    """
    Smart Telegram notification system for trading bot.

    Sends formatted messages with trade info, market conditions,
    and educational content.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._session = None

        # Track daily stats
        self._daily_trades: List[TradeInfo] = []
        self._daily_start_balance: float = 0
        self._last_daily_report: Optional[datetime] = None

        # Rate limiting
        self._last_message_time: Optional[datetime] = None
        self._min_message_interval = 1  # seconds

        # API URL
        self._api_url = f"https://api.telegram.org/bot{bot_token}"

        # Chart storage
        self._charts_dir = Path("data/charts")
        self._charts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Telegram notifier initialized (enabled={enabled})")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """Send a text message to Telegram."""
        if not self.enabled:
            return True

        try:
            session = await self._get_session()

            url = f"{self._api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }

            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Telegram send failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    async def send_photo(
        self,
        photo_path: str,
        caption: str = "",
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a photo to Telegram."""
        if not self.enabled:
            return True

        try:
            session = await self._get_session()

            url = f"{self._api_url}/sendPhoto"

            import aiohttp
            data = aiohttp.FormData()
            data.add_field("chat_id", self.chat_id)
            data.add_field("caption", caption)
            data.add_field("parse_mode", parse_mode)

            with open(photo_path, "rb") as f:
                data.add_field("photo", f, filename="chart.png")

                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Telegram photo send failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
            return False

    async def send_document(
        self,
        doc_path: str,
        caption: str = "",
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a document (PDF) to Telegram."""
        if not self.enabled:
            return True

        try:
            session = await self._get_session()

            url = f"{self._api_url}/sendDocument"

            import aiohttp
            data = aiohttp.FormData()
            data.add_field("chat_id", self.chat_id)
            data.add_field("caption", caption)
            data.add_field("parse_mode", parse_mode)

            with open(doc_path, "rb") as f:
                filename = Path(doc_path).name
                data.add_field("document", f, filename=filename)

                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Telegram doc send failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"Telegram doc error: {e}")
            return False

    # ========== FORMATTED MESSAGES ==========

    def _format_trade_open(self, trade: TradeInfo) -> str:
        """Format trade open notification - Compact Mobile Style."""
        emoji = "🟢" if trade.order_type == "BUY" else "🔴"
        direction = "LONG" if trade.order_type == "BUY" else "SHORT"

        # Calculate risk/reward
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        tp_distance = abs(trade.take_profit - trade.entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # SL display
        sl_display = f"{trade.stop_loss:.2f}" if trade.stop_loss > 0 else "Smart"

        # Calculate potential profit/loss
        potential_loss = abs(trade.entry_price - trade.stop_loss) * trade.lot_size * 100 if trade.stop_loss > 0 else 0
        potential_profit = abs(trade.take_profit - trade.entry_price) * trade.lot_size * 100

        msg = f"""{emoji} <b>{direction}</b> #{trade.ticket}
├ <b>{trade.symbol}</b>
├ Entry: <code>{trade.entry_price:.2f}</code>
├ Lot: <code>{trade.lot_size}</code>
├ SL: <code>{sl_display}</code> (-${potential_loss:.0f})
├ TP: <code>{trade.take_profit:.2f}</code> (+${potential_profit:.0f})
├ R:R: <code>1:{rr_ratio:.1f}</code>
├ AI: <code>{trade.ml_confidence:.0%}</code> | {trade.regime}
└ <i>{trade.signal_reason[:50]}</i>
⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg

    def _format_trade_close(self, trade: TradeInfo) -> str:
        """Format trade close notification - Compact Mobile Style."""
        # Determine profit/loss styling
        if trade.profit > 0:
            emoji = "✅"
            profit_str = f"+${trade.profit:.2f}"
        elif trade.profit < 0:
            emoji = "❌"
            profit_str = f"-${abs(trade.profit):.2f}"
        else:
            emoji = "➖"
            profit_str = "$0"

        # Calculate percentage change
        pct_change = (trade.profit / trade.balance_before * 100) if trade.balance_before > 0 else 0
        pct_str = f"+{pct_change:.2f}%" if pct_change >= 0 else f"{pct_change:.2f}%"

        # Duration formatting
        duration_mins = trade.duration_seconds // 60
        duration_str = f"{duration_mins}m" if duration_mins > 0 else f"{trade.duration_seconds}s"

        # Result label
        if trade.profit > 0:
            result = "WIN"
        elif trade.profit < 0:
            result = "LOSS"
        else:
            result = "BE"

        msg = f"""{emoji} <b>{result}</b> #{trade.ticket}
├ <b>{trade.symbol}</b> {trade.order_type}
├ Entry: <code>{trade.entry_price:.2f}</code>
├ Exit: <code>{trade.close_price:.2f}</code>
├ Lot: <code>{trade.lot_size}</code>
├ <b>P/L: {profit_str}</b> ({pct_str})
├ Pips: <code>{trade.profit_pips:+.1f}</code>
├ Duration: <code>{duration_str}</code>
├ Bal Before: <code>${trade.balance_before:,.2f}</code>
└ Bal After: <code>${trade.balance_after:,.2f}</code>
⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg

    def _format_market_update(self, condition: MarketCondition) -> str:
        """Format market condition update - Compact Mobile Style."""
        # Signal emoji
        if condition.ml_signal == "BUY":
            signal_emoji = "🟢"
        elif condition.ml_signal == "SELL":
            signal_emoji = "🔴"
        else:
            signal_emoji = "⚪"

        status = "✅" if condition.can_trade else "⛔"

        msg = f"""📊 <b>{condition.symbol}</b> ${condition.price:.2f}
├ {signal_emoji} {condition.ml_signal} {condition.ml_confidence:.0%}
├ {condition.trend_direction}
├ {condition.regime}
├ {condition.session}
└ {status}
⏰ {datetime.now(WIB).strftime('%H:%M')}"""
        return msg

    def _format_daily_summary(
        self,
        trades: List[TradeInfo],
        start_balance: float,
        end_balance: float,
        market_condition: Optional[MarketCondition] = None,
    ) -> str:
        """Format daily trading summary - Mobile Responsive with Code."""
        # Calculate stats
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit > 0)
        losing_trades = sum(1 for t in trades if t.profit < 0)

        total_profit = sum(t.profit for t in trades)
        gross_profit = sum(t.profit for t in trades if t.profit > 0)
        gross_loss = sum(abs(t.profit) for t in trades if t.profit < 0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        pf_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"

        # Average trade
        avg_profit = (total_profit / total_trades) if total_trades > 0 else 0

        # Day result
        day_pct = ((end_balance - start_balance) / start_balance * 100) if start_balance > 0 else 0

        if total_profit > 0:
            day_emoji = "🎉"
            day_result = "PROFIT"
        elif total_profit < 0:
            day_emoji = "📉"
            day_result = "LOSS"
        else:
            day_emoji = "➖"
            day_result = "BE"

        profit_str = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        pct_str = f"+{day_pct:.2f}%" if day_pct >= 0 else f"{day_pct:.2f}%"

        # Build trade history (last 5)
        trade_lines = []
        for i, t in enumerate(trades[-5:]):
            sign = "+" if t.profit >= 0 else "-"
            amt = abs(t.profit)
            result_emoji = "✅" if t.profit > 0 else "❌" if t.profit < 0 else "➖"
            prefix = "└" if i == len(trades[-5:]) - 1 else "├"
            trade_lines.append(f"{prefix} {result_emoji} {t.order_type}: {sign}${amt:.2f}")
        trade_str = "\n".join(trade_lines) if trade_lines else "└ No trades"

        msg = f"""{day_emoji} <b>DAILY REPORT</b> {datetime.now(WIB).strftime('%Y-%m-%d')}

<b>Result</b>
├ P/L: <b>{profit_str}</b> ({pct_str})
├ Gross Win: <code>+${gross_profit:.2f}</code>
├ Gross Loss: <code>-${gross_loss:.2f}</code>
├ Bal Start: <code>${start_balance:,.2f}</code>
└ Bal End: <code>${end_balance:,.2f}</code>

<b>Stats</b>
├ Total: <code>{total_trades}</code> trades
├ Wins: <code>{winning_trades}</code> | Losses: <code>{losing_trades}</code>
├ Win Rate: <code>{win_rate:.1f}%</code>
├ Profit Factor: <code>{pf_str}</code>
└ Avg/Trade: <code>${avg_profit:.2f}</code>

<b>Recent Trades</b>
{trade_str}"""
        return msg

    def _format_alert(self, alert_type: str, message: str) -> str:
        """Format alert message - Compact Mobile Style."""
        alert_emojis = {
            "flash_crash": "🚨",
            "high_volatility": "⚡",
            "connection_error": "📡",
            "model_retrain": "🔄",
            "market_close": "🔔",
            "low_balance": "💰",
        }
        emoji = alert_emojis.get(alert_type, "⚠️")
        title = alert_type.upper().replace('_', ' ')

        msg = f"""{emoji} <b>{title}</b>
└ {message}
⏰ {datetime.now(WIB).strftime('%H:%M')}"""
        return msg

    def _format_system_status(
        self,
        balance: float,
        equity: float,
        open_positions: int,
        session: str,
        ml_status: str,
        uptime_hours: float,
    ) -> str:
        """Format system status message - Compact Mobile Style."""
        msg = f"""🤖 <b>STATUS</b> 🟢
├ Bal: ${balance:,.0f}
├ Eq: ${equity:,.0f}
├ Pos: {open_positions}
├ {session}
├ ML: {ml_status}
└ Up: {uptime_hours:.1f}h
⏰ {datetime.now(WIB).strftime('%H:%M')}"""
        return msg

    # ========== HIGH-LEVEL NOTIFICATION METHODS ==========

    async def notify_trade_open(
        self,
        ticket: int,
        symbol: str,
        order_type: str,
        lot_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        ml_confidence: float,
        signal_reason: str,
        regime: str,
        volatility: str,
    ):
        """Send trade open notification."""
        trade = TradeInfo(
            ticket=ticket,
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ml_confidence=ml_confidence,
            signal_reason=signal_reason,
            regime=regime,
            volatility=volatility,
        )

        msg = self._format_trade_open(trade)
        await self.send_message(msg)
        logger.info(f"Telegram: Trade open notification sent for #{ticket}")

    async def notify_trade_close(
        self,
        ticket: int,
        symbol: str,
        order_type: str,
        lot_size: float,
        entry_price: float,
        close_price: float,
        profit: float,
        profit_pips: float,
        balance_before: float,
        balance_after: float,
        duration_seconds: int,
        ml_confidence: float = 0,
        regime: str = "",
        volatility: str = "",
    ):
        """Send trade close notification with detailed P/L."""
        trade = TradeInfo(
            ticket=ticket,
            symbol=symbol,
            order_type=order_type,
            lot_size=lot_size,
            entry_price=entry_price,
            close_price=close_price,
            profit=profit,
            profit_pips=profit_pips,
            balance_before=balance_before,
            balance_after=balance_after,
            duration_seconds=duration_seconds,
            ml_confidence=ml_confidence,
            regime=regime,
            volatility=volatility,
        )

        # Track for daily summary
        self._daily_trades.append(trade)

        msg = self._format_trade_close(trade)
        await self.send_message(msg)
        logger.info(f"Telegram: Trade close notification sent for #{ticket}")

    async def notify_market_update(
        self,
        symbol: str,
        price: float,
        regime: str,
        volatility: str,
        ml_signal: str,
        ml_confidence: float,
        trend_direction: str,
        session: str,
        can_trade: bool,
        atr: float = 0,
        spread: float = 0,
    ):
        """Send market condition update."""
        condition = MarketCondition(
            symbol=symbol,
            price=price,
            regime=regime,
            volatility=volatility,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            trend_direction=trend_direction,
            session=session,
            can_trade=can_trade,
            atr=atr,
            spread=spread,
        )

        msg = self._format_market_update(condition)
        await self.send_message(msg, disable_notification=True)
        logger.info("Telegram: Market update sent")

    async def notify_alert(self, alert_type: str, message: str):
        """Send alert notification."""
        msg = self._format_alert(alert_type, message)
        await self.send_message(msg)
        logger.info(f"Telegram: Alert sent - {alert_type}")

    async def notify_system_status(
        self,
        balance: float,
        equity: float,
        open_positions: int,
        session: str,
        ml_status: str,
        uptime_hours: float,
    ):
        """Send system status update."""
        msg = self._format_system_status(
            balance, equity, open_positions,
            session, ml_status, uptime_hours
        )
        await self.send_message(msg, disable_notification=True)
        logger.info("Telegram: System status sent")

    async def send_daily_summary(
        self,
        start_balance: float,
        end_balance: float,
        market_condition: Optional[MarketCondition] = None,
    ):
        """Send daily trading summary."""
        msg = self._format_daily_summary(
            self._daily_trades,
            start_balance,
            end_balance,
            market_condition,
        )
        await self.send_message(msg)

        # Generate and send chart if possible
        chart_path = await self._generate_daily_chart(
            self._daily_trades,
            start_balance,
            end_balance,
        )
        if chart_path:
            await self.send_photo(
                chart_path,
                caption=f"📊 Daily Performance Chart - {datetime.now(WIB).strftime('%Y-%m-%d')}"
            )

        # Reset daily tracking
        self._daily_trades = []
        self._last_daily_report = datetime.now(WIB)

        logger.info("Telegram: Daily summary sent")

    async def _generate_daily_chart(
        self,
        trades: List[TradeInfo],
        start_balance: float,
        end_balance: float,
    ) -> Optional[str]:
        """Generate daily performance chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if not trades:
                return None

            # Create figure with dark theme (shadcn-inspired)
            plt.style.use('dark_background')
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.patch.set_facecolor('#0a0a0a')

            # Color palette (shadcn-inspired)
            colors = {
                'profit': '#22c55e',  # Green
                'loss': '#ef4444',    # Red
                'neutral': '#64748b', # Slate
                'primary': '#3b82f6', # Blue
                'bg': '#0a0a0a',
                'card': '#1c1c1c',
                'text': '#fafafa',
            }

            # 1. Equity Curve
            ax1 = axes[0, 0]
            ax1.set_facecolor(colors['card'])

            balance_curve = [start_balance]
            for t in trades:
                balance_curve.append(balance_curve[-1] + t.profit)

            x = range(len(balance_curve))
            ax1.fill_between(x, balance_curve, alpha=0.3, color=colors['primary'])
            ax1.plot(x, balance_curve, color=colors['primary'], linewidth=2)
            ax1.set_title('Equity Curve', color=colors['text'], fontsize=12, fontweight='bold')
            ax1.set_xlabel('Trade #', color=colors['text'])
            ax1.set_ylabel('Balance ($)', color=colors['text'])
            ax1.tick_params(colors=colors['text'])
            ax1.grid(True, alpha=0.2)

            # 2. P/L per Trade
            ax2 = axes[0, 1]
            ax2.set_facecolor(colors['card'])

            profits = [t.profit for t in trades]
            bar_colors = [colors['profit'] if p > 0 else colors['loss'] for p in profits]
            ax2.bar(range(len(profits)), profits, color=bar_colors, alpha=0.8)
            ax2.axhline(y=0, color=colors['neutral'], linestyle='-', linewidth=1)
            ax2.set_title('P/L per Trade', color=colors['text'], fontsize=12, fontweight='bold')
            ax2.set_xlabel('Trade #', color=colors['text'])
            ax2.set_ylabel('Profit ($)', color=colors['text'])
            ax2.tick_params(colors=colors['text'])
            ax2.grid(True, alpha=0.2)

            # 3. Win/Loss Pie Chart
            ax3 = axes[1, 0]
            ax3.set_facecolor(colors['card'])

            wins = sum(1 for t in trades if t.profit > 0)
            losses = sum(1 for t in trades if t.profit < 0)
            be = sum(1 for t in trades if t.profit == 0)

            sizes = [wins, losses, be] if be > 0 else [wins, losses]
            pie_colors = [colors['profit'], colors['loss'], colors['neutral']][:len(sizes)]
            labels = ['Wins', 'Losses', 'BE'][:len(sizes)]

            if sum(sizes) > 0:
                wedges, texts, autotexts = ax3.pie(
                    sizes, labels=labels, autopct='%1.1f%%',
                    colors=pie_colors, startangle=90
                )
                for text in texts:
                    text.set_color(colors['text'])
                for autotext in autotexts:
                    autotext.set_color(colors['text'])
            ax3.set_title('Win Rate', color=colors['text'], fontsize=12, fontweight='bold')

            # 4. Summary Stats Box
            ax4 = axes[1, 1]
            ax4.set_facecolor(colors['card'])
            ax4.axis('off')

            total_profit = sum(t.profit for t in trades)
            win_rate = (wins / len(trades) * 100) if trades else 0
            avg_profit = total_profit / len(trades) if trades else 0

            stats_text = f"""
Daily Summary
─────────────────
Total Trades:  {len(trades)}
Win Rate:      {win_rate:.1f}%
Net P/L:       ${total_profit:+,.2f}
Avg Trade:     ${avg_profit:+,.2f}

Start Balance: ${start_balance:,.2f}
End Balance:   ${end_balance:,.2f}
Day Change:    {((end_balance-start_balance)/start_balance*100):+.2f}%
"""
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top',
                    fontfamily='monospace', color=colors['text'])
            ax4.set_title('Statistics', color=colors['text'], fontsize=12, fontweight='bold')

            plt.tight_layout()

            # Save chart
            chart_path = self._charts_dir / f"daily_{datetime.now(WIB).strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, facecolor=colors['bg'], edgecolor='none')
            plt.close()

            return str(chart_path)

        except ImportError:
            logger.warning("matplotlib not available for chart generation")
            return None
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

    def set_daily_start_balance(self, balance: float):
        """Set the starting balance for daily tracking."""
        self._daily_start_balance = balance
        self._daily_trades = []

    async def send_startup_message(
        self,
        symbol: str,
        capital: float,
        balance: float,
        mode: str,
        ml_model_status: str,
        news_status: str = "SAFE",
    ):
        """Send bot startup notification - Compact Mobile Style."""
        news_emoji = "✅" if news_status == "SAFE" else "⚠️"
        msg = f"""🚀 <b>BOT STARTED</b>

<b>Config</b>
├ Symbol: <code>{symbol}</code>
├ Mode: <code>{mode}</code>
├ Capital: <code>${capital:,.2f}</code>
├ Balance: <code>${balance:,.2f}</code>
└ ML: <code>{ml_model_status}</code>

<b>Risk Settings</b>
├ Risk/Trade: <code>1%</code>
├ Max Daily Loss: <code>5%</code>
├ Max Total Loss: <code>10%</code>
└ SL: <code>Smart (No Hard)</code>

{news_emoji} News: {news_status}
⏰ {datetime.now(WIB).strftime('%Y-%m-%d %H:%M')} WIB"""
        await self.send_message(msg)
        logger.info("Telegram: Startup message sent")

    async def send_news_alert(
        self,
        event_name: str,
        condition: str,
        reason: str,
        buffer_minutes: int = 60,
    ):
        """Send news alert when high-impact news blocks trading."""
        emoji_map = {
            "DANGER_NEWS": "🚨",
            "DANGER_SENTIMENT": "⚠️",
            "CAUTION": "⚡",
            "SAFE": "✅",
        }
        emoji = emoji_map.get(condition, "📰")

        msg = f"""{emoji} <b>NEWS</b> {condition}
├ {event_name[:30]}
├ {reason[:35]}
└ Buffer: {buffer_minutes}m
⏰ {datetime.now(WIB).strftime('%H:%M')}"""

        await self.send_message(msg)
        logger.info(f"Telegram: News alert sent - {event_name}")

    async def send_hourly_analysis(
        self,
        # Account info
        balance: float,
        equity: float,
        floating_pnl: float,
        # Position info
        open_positions: int,
        position_details: list,  # List of dicts with ticket, direction, profit, momentum, tp_prob
        # Market info
        symbol: str,
        current_price: float,
        session: str,
        regime: str,
        volatility: str,
        # ML/AI info
        ml_signal: str,
        ml_confidence: float,
        dynamic_threshold: float,
        market_quality: str,
        market_score: int,
        # Risk info
        daily_pnl: float,
        daily_trades: int,
        risk_mode: str,
        max_daily_loss: float,
        # Bot info
        uptime_hours: float,
        total_loops: int,
        avg_execution_ms: float,
        # News info (optional)
        news_status: str = "SAFE",
        news_reason: str = "No high-impact news",
    ):
        """
        Send comprehensive hourly analysis report.
        Interval: Every 1 hour
        """
        now = datetime.now(WIB)

        # Floating P/L emoji
        float_emoji = "+" if floating_pnl >= 0 else ""
        daily_emoji = "+" if daily_pnl >= 0 else ""

        # Risk mode indicator
        risk_indicators = {
            "normal": "NORMAL",
            "recovery": "RECOVERY",
            "protected": "PROTECTED",
            "stopped": "STOPPED",
        }
        risk_display = risk_indicators.get(risk_mode.lower(), risk_mode.upper())

        # Market quality indicator
        quality_indicators = {
            "excellent": "EXCELLENT",
            "good": "GOOD",
            "moderate": "MODERATE",
            "poor": "POOR",
            "avoid": "AVOID",
        }
        quality_display = quality_indicators.get(market_quality.lower(), market_quality.upper())

        # Build position details string
        pos_lines = []
        for pos in position_details[:5]:  # Max 5 positions
            ticket = pos.get("ticket", 0)
            direction = pos.get("direction", "?")
            profit = pos.get("profit", 0)
            momentum = pos.get("momentum", 0)
            tp_prob = pos.get("tp_probability", 50)

            profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
            mom_str = f"+{momentum:.0f}" if momentum >= 0 else f"{momentum:.0f}"

            pos_lines.append(f"  #{ticket} {direction}: {profit_str} | M:{mom_str} | TP:{tp_prob:.0f}%")

        positions_str = "\n".join(pos_lines) if pos_lines else "  No open positions"

        # ML signal strength
        if ml_confidence >= 0.75:
            signal_strength = "STRONG"
        elif ml_confidence >= 0.65:
            signal_strength = "MODERATE"
        else:
            signal_strength = "WEAK"

        # Can trade indicator
        can_trade = ml_confidence >= dynamic_threshold and market_quality.lower() != "avoid"
        trade_status = "READY" if can_trade else "WAIT"

        # Build position list with details
        pos_lines = []
        for i, pos in enumerate(position_details[:5]):  # Max 5
            t = pos.get("ticket", 0)
            d = pos.get("direction", "?")
            p = pos.get("profit", 0)
            m = pos.get("momentum", 0)
            tp_prob = pos.get("tp_probability", 50)
            ps = f"+${p:.2f}" if p >= 0 else f"-${abs(p):.2f}"
            prefix = "└" if i == len(position_details[:5]) - 1 else "├"
            pos_lines.append(f"{prefix} #{t} {d}: <b>{ps}</b> M:{m:+.0f}")
        pos_str = "\n".join(pos_lines) if pos_lines else "└ No positions"

        msg = f"""📊 <b>HOURLY</b> {now.strftime('%H:%M')} WIB

<b>Account</b>
├ Bal: <code>${balance:,.2f}</code>
├ Eq: <code>${equity:,.2f}</code>
├ Float: <b>{float_emoji}${floating_pnl:.2f}</b>
└ Day: <b>{daily_emoji}${daily_pnl:.2f}</b> ({daily_trades} trades)

<b>Positions ({open_positions})</b>
{pos_str}

<b>Market</b>
├ {symbol} <code>${current_price:,.2f}</code>
├ {session}
└ {regime} | {volatility}

<b>AI Signal</b>
├ {ml_signal} <code>{ml_confidence:.0%}</code> / thresh <code>{dynamic_threshold:.0%}</code>
└ Quality: {quality_display} (score:{market_score}) → {trade_status}

<b>Risk</b> {risk_display}
└ Daily Loss: <code>${abs(min(0, daily_pnl)):.2f}</code> / <code>${max_daily_loss:.2f}</code>

{"✅" if news_status == "SAFE" else "⚠️"} News: {news_status}"""

        await self.send_message(msg, disable_notification=True)
        logger.info("Telegram: Hourly analysis report sent")

    async def send_shutdown_message(
        self,
        balance: float,
        total_trades: int,
        total_profit: float,
        uptime_hours: float,
    ):
        """Send bot shutdown notification - Compact Mobile Style."""
        profit_str = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        emoji = "✅" if total_profit >= 0 else "❌"

        msg = f"""🔴 <b>BOT STOPPED</b>

<b>Session Summary</b>
├ Balance: <code>${balance:,.2f}</code>
├ Total Trades: <code>{total_trades}</code>
├ {emoji} P/L: <b>{profit_str}</b>
└ Uptime: <code>{uptime_hours:.1f}h</code>

⏰ {datetime.now(WIB).strftime('%Y-%m-%d %H:%M')} WIB"""
        await self.send_message(msg)
        logger.info("Telegram: Shutdown message sent")


def create_telegram_notifier() -> TelegramNotifier:
    """Create Telegram notifier from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = bool(bot_token and chat_id)

    if not enabled:
        logger.warning("Telegram notifier disabled - missing BOT_TOKEN or CHAT_ID")

    return TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=enabled,
    )


if __name__ == "__main__":
    # Test telegram notifier
    import asyncio

    async def test():
        notifier = create_telegram_notifier()

        # Test startup message
        await notifier.send_startup_message(
            symbol="XAUUSD",
            capital=5000,
            balance=6160,
            mode="small",
            ml_model_status="Loaded (37 features)",
        )

        # Test trade close notification
        await notifier.notify_trade_close(
            ticket=12345678,
            symbol="XAUUSD",
            order_type="BUY",
            lot_size=0.2,
            entry_price=4950.00,
            close_price=4965.00,
            profit=30.00,
            profit_pips=150,
            balance_before=6130.00,
            balance_after=6160.00,
            duration_seconds=125,
            ml_confidence=0.71,
            regime="medium_volatility",
            volatility="high",
        )

        # Test market update
        await notifier.notify_market_update(
            symbol="XAUUSD",
            price=4965.00,
            regime="medium_volatility",
            volatility="high",
            ml_signal="BUY",
            ml_confidence=0.71,
            trend_direction="UPTREND",
            session="London-NY Overlap",
            can_trade=True,
            atr=15.5,
            spread=2.1,
        )

        await notifier.close()

    asyncio.run(test())
