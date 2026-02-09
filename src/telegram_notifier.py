"""
Telegram Notifier Module
========================
Smart Telegram integration for AI Trading Bot.

Features:
- Trade notifications with ALL features as text array
- Market condition updates with full context
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

        # Command polling
        self._last_update_id: int = 0
        self._command_handlers: Dict[str, Any] = {}

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

    # ========== COMMAND SYSTEM ==========

    def register_command(self, command: str, handler):
        """Register a command handler. Handler is an async callable returning str."""
        self._command_handlers[command.lstrip("/")] = handler

    async def poll_commands(self) -> int:
        """
        Poll Telegram for new commands and dispatch handlers.
        Returns number of commands processed.
        """
        if not self.enabled:
            return 0

        try:
            session = await self._get_session()
            url = f"{self._api_url}/getUpdates"
            params = {"offset": self._last_update_id + 1, "timeout": 0, "limit": 10}

            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json()

            if not data.get("ok") or not data.get("result"):
                return 0

            processed = 0
            for update in data["result"]:
                self._last_update_id = update["update_id"]

                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # Only respond to our chat
                if chat_id != self.chat_id:
                    continue

                if not text.startswith("/"):
                    continue

                # Parse command (e.g., "/status" or "/status@botname")
                cmd = text.split()[0].split("@")[0].lstrip("/").lower()

                if cmd in self._command_handlers:
                    try:
                        response = await self._command_handlers[cmd]()
                        if response:
                            await self.send_message(response)
                        processed += 1
                    except Exception as e:
                        logger.warning(f"Command /{cmd} error: {e}")
                        await self.send_message(f"⚠️ Error: <code>{e}</code>")
                elif cmd == "help":
                    await self._send_help()
                    processed += 1
                else:
                    await self.send_message(f"❓ Unknown: <code>/{cmd}</code>\nKetik /help untuk daftar command.")
                    processed += 1

            return processed

        except asyncio.TimeoutError:
            return 0
        except Exception as e:
            logger.debug(f"Command poll error: {e}")
            return 0

    async def _send_help(self):
        """Send help message with all available commands."""
        cmd_list = sorted(self._command_handlers.keys())
        help_items = []
        for cmd in cmd_list:
            doc = getattr(self._command_handlers[cmd], "_cmd_desc", "")
            help_items.append(f"/{cmd} — {doc}" if doc else f"/{cmd}")

        msg = f"""📋 <b>COMMANDS</b>

{self._build_section("Available", help_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        await self.send_message(msg.strip())

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

    # ========== HELPER: Build text array ==========

    @staticmethod
    def _build_section(title: str, items: List[str]) -> str:
        """Build a section with tree-style connectors."""
        if not items:
            return ""
        lines = [f"<b>{title}</b>"]
        for i, item in enumerate(items):
            prefix = "└" if i == len(items) - 1 else "├"
            lines.append(f"{prefix} {item}")
        return "\n".join(lines)

    # ========== FORMATTED MESSAGES ==========

    def _format_trade_open(self, trade: TradeInfo, ctx: dict) -> str:
        """Format trade open notification with ALL features as text array."""
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

        # === Section 1: Trade ===
        trade_items = [
            f"<b>{trade.symbol}</b>",
            f"Entry: <code>{trade.entry_price:.2f}</code>",
            f"Lot: <code>{trade.lot_size}</code>",
            f"SL: <code>{sl_display}</code> (-${potential_loss:.0f})",
            f"TP: <code>{trade.take_profit:.2f}</code> (+${potential_profit:.0f})",
            f"R:R: <code>1:{rr_ratio:.1f}</code>",
        ]

        # === Section 2: AI / ML ===
        ml_conf = trade.ml_confidence
        threshold = ctx.get("dynamic_threshold", 0.5)
        quality = ctx.get("market_quality", "unknown")
        score = ctx.get("market_score", 0)
        ai_items = [
            f"ML: <code>{ml_conf:.0%}</code> / thresh <code>{threshold:.0%}</code>",
            f"Quality: <code>{quality.upper()}</code> (score:{score})",
        ]

        # === Section 3: SMC ===
        smc_signal = ctx.get("smc_signal", "")
        smc_conf = ctx.get("smc_confidence", 0)
        smc_fvg = ctx.get("smc_fvg", False)
        smc_ob = ctx.get("smc_ob", False)
        smc_bos = ctx.get("smc_bos", False)
        smc_choch = ctx.get("smc_choch", False)

        patterns = []
        if smc_fvg: patterns.append("FVG")
        if smc_ob: patterns.append("OB")
        if smc_bos: patterns.append("BOS")
        if smc_choch: patterns.append("CHoCH")

        smc_items = [
            f"Signal: <code>{smc_signal or 'NONE'}</code> ({smc_conf:.0%})",
            f"Patterns: <code>{', '.join(patterns) if patterns else 'None'}</code>",
        ]

        # === Section 4: Market ===
        session = ctx.get("session", "Unknown")
        h1_bias = ctx.get("h1_bias", "NEUTRAL")
        regime = trade.regime or "unknown"
        vol = trade.volatility or "unknown"
        market_items = [
            f"Session: <code>{session}</code>",
            f"Regime: <code>{regime}</code> | Vol: <code>{vol}</code>",
            f"H1 Bias: <code>{h1_bias}</code>",
        ]

        # === Section 5: Risk ===
        risk_mode = ctx.get("risk_mode", "normal")
        daily_loss = ctx.get("daily_loss", 0)
        consec = ctx.get("consecutive_losses", 0)
        risk_items = [
            f"Mode: <code>{risk_mode.upper()}</code>",
            f"Daily Loss: <code>${daily_loss:.2f}</code> | Streak: <code>{consec}L</code>",
        ]

        # === Section 6: Entry Filters ===
        filters = ctx.get("entry_filters", [])
        filter_items = []
        for f in filters:
            passed = f.get("passed", True)
            name = f.get("name", "")
            detail = f.get("detail", "")
            icon = "✅" if passed else "❌"
            filter_items.append(f"{icon} {name}: {detail}")

        # === Build message ===
        sections = [
            self._build_section("Trade", trade_items),
            self._build_section("AI Signal", ai_items),
            self._build_section("SMC", smc_items),
            self._build_section("Market", market_items),
            self._build_section("Risk", risk_items),
        ]
        if filter_items:
            sections.append(self._build_section("Entry Filters", filter_items))

        body = "\n\n".join(s for s in sections if s)

        msg = f"""{emoji} <b>{direction}</b> #{trade.ticket}

{body}

<i>{trade.signal_reason[:80]}</i>
⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

    def _format_trade_close(self, trade: TradeInfo, ctx: dict) -> str:
        """Format trade close notification with ALL status as text array."""
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

        # Balance change
        bal_change = trade.balance_after - trade.balance_before
        bal_change_str = f"+${bal_change:.2f}" if bal_change >= 0 else f"-${abs(bal_change):.2f}"

        # Win rate
        win_rate = ctx.get("win_rate", 0)
        session_trades = ctx.get("session_trades", 0)
        session_wins = ctx.get("session_wins", 0)

        # === Section 1: Trade Result ===
        trade_items = [
            f"<b>{trade.symbol}</b> {trade.order_type}",
            f"Entry: <code>{trade.entry_price:.2f}</code> → Exit: <code>{trade.close_price:.2f}</code>",
            f"Lot: <code>{trade.lot_size}</code> | Pips: <code>{trade.profit_pips:+.1f}</code>",
            f"<b>P/L: {profit_str}</b> ({pct_str})",
            f"Duration: <code>{duration_str}</code>",
        ]

        # === Section 2: Exit ===
        exit_reason = ctx.get("exit_reason", "unknown")
        exit_items = [
            f"Reason: <code>{exit_reason}</code>",
            f"Regime: <code>{trade.regime or 'unknown'}</code> | Vol: <code>{trade.volatility or 'unknown'}</code>",
            f"Session: <code>{ctx.get('session', 'Unknown')}</code>",
        ]

        # === Section 3: Balance ===
        balance_items = [
            f"Before: <code>${trade.balance_before:,.2f}</code>",
            f"After: <code>${trade.balance_after:,.2f}</code> (<b>{bal_change_str}</b>)",
        ]

        # === Section 4: Session Stats ===
        session_profit = ctx.get("session_profit", 0)
        session_pnl_str = f"+${session_profit:.2f}" if session_profit >= 0 else f"-${abs(session_profit):.2f}"
        consec = ctx.get("consecutive_losses", 0)

        stats_items = [
            f"Trades: <code>{session_wins}W</code> / <code>{session_trades}T</code>",
            f"Win Rate: <code>{win_rate:.1f}%</code>",
            f"Session P/L: <b>{session_pnl_str}</b>",
            f"Streak: <code>{consec}L</code> | Mode: <code>{ctx.get('risk_mode', 'normal').upper()}</code>",
        ]

        # === Build message ===
        msg = f"""{emoji} <b>{result}</b> #{trade.ticket}

{self._build_section("Trade", trade_items)}

{self._build_section("Exit", exit_items)}

{self._build_section("Balance", balance_items)}

{self._build_section("Session Stats", stats_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

    def _format_market_update(self, condition: MarketCondition, ctx: dict) -> str:
        """Format market condition update with full context as text array."""
        # Signal emoji
        if condition.ml_signal == "BUY":
            signal_emoji = "🟢"
        elif condition.ml_signal == "SELL":
            signal_emoji = "🔴"
        else:
            signal_emoji = "⚪"

        status = "✅ READY" if condition.can_trade else "⛔ WAIT"

        # Extra context
        h1_bias = ctx.get("h1_bias", "NEUTRAL")
        threshold = ctx.get("dynamic_threshold", 0.5)
        quality = ctx.get("market_quality", "unknown")
        score = ctx.get("market_score", 0)
        smc_signal = ctx.get("smc_signal", "")
        smc_conf = ctx.get("smc_confidence", 0)

        # === Section 1: Price ===
        price_items = [
            f"<b>{condition.symbol}</b> <code>${condition.price:.2f}</code>",
            f"ATR: <code>{condition.atr:.2f}</code> | Spread: <code>{condition.spread:.1f}</code>",
        ]

        # === Section 2: AI Signal ===
        signal_items = [
            f"{signal_emoji} ML: <code>{condition.ml_signal}</code> {condition.ml_confidence:.0%} / thresh {threshold:.0%}",
            f"SMC: <code>{smc_signal or 'NONE'}</code> ({smc_conf:.0%})",
            f"Quality: <code>{quality.upper()}</code> (score:{score})",
            f"Trend: <code>{condition.trend_direction}</code> | H1: <code>{h1_bias}</code>",
        ]

        # === Section 3: Market ===
        market_items = [
            f"Regime: <code>{condition.regime}</code> | Vol: <code>{condition.volatility}</code>",
            f"Session: <code>{condition.session}</code>",
            f"Status: {status}",
        ]

        # === Section 4: Risk ===
        risk_mode = ctx.get("risk_mode", "normal")
        daily_loss = ctx.get("daily_loss", 0)
        consec = ctx.get("consecutive_losses", 0)
        session_trades = ctx.get("session_trades", 0)
        session_profit = ctx.get("session_profit", 0)
        sp_str = f"+${session_profit:.2f}" if session_profit >= 0 else f"-${abs(session_profit):.2f}"

        risk_items = [
            f"Mode: <code>{risk_mode.upper()}</code> | Streak: <code>{consec}L</code>",
            f"Daily Loss: <code>${daily_loss:.2f}</code>",
            f"Session: <code>{session_trades}</code> trades, <b>{sp_str}</b>",
        ]

        msg = f"""📊 <b>MARKET UPDATE</b>

{self._build_section("Price", price_items)}

{self._build_section("AI Signal", signal_items)}

{self._build_section("Market", market_items)}

{self._build_section("Risk", risk_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

    def _format_daily_summary(
        self,
        trades: List[TradeInfo],
        start_balance: float,
        end_balance: float,
        market_condition: Optional[MarketCondition] = None,
    ) -> str:
        """Format daily trading summary with ALL stats as text array."""
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
        elif total_profit < 0:
            day_emoji = "📉"
        else:
            day_emoji = "➖"

        profit_str = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        pct_str = f"+{day_pct:.2f}%" if day_pct >= 0 else f"{day_pct:.2f}%"

        # === Section 1: Result ===
        result_items = [
            f"<b>P/L: {profit_str}</b> ({pct_str})",
            f"Gross Win: <code>+${gross_profit:.2f}</code>",
            f"Gross Loss: <code>-${gross_loss:.2f}</code>",
            f"Bal Start: <code>${start_balance:,.2f}</code>",
            f"Bal End: <code>${end_balance:,.2f}</code>",
        ]

        # === Section 2: Stats ===
        stats_items = [
            f"Total: <code>{total_trades}</code> trades",
            f"Wins: <code>{winning_trades}</code> | Losses: <code>{losing_trades}</code>",
            f"Win Rate: <code>{win_rate:.1f}%</code>",
            f"Profit Factor: <code>{pf_str}</code>",
            f"Avg/Trade: <code>${avg_profit:.2f}</code>",
        ]

        # === Section 3: Recent Trades ===
        recent = trades[-5:]
        trade_items = []
        for t in recent:
            sign = "+" if t.profit >= 0 else "-"
            amt = abs(t.profit)
            result_emoji = "✅" if t.profit > 0 else "❌" if t.profit < 0 else "➖"
            trade_items.append(f"{result_emoji} {t.order_type}: {sign}${amt:.2f}")
        if not trade_items:
            trade_items = ["No trades"]

        msg = f"""{day_emoji} <b>DAILY REPORT</b> {datetime.now(WIB).strftime('%Y-%m-%d')}

{self._build_section("Result", result_items)}

{self._build_section("Stats", stats_items)}

{self._build_section("Recent Trades", trade_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

    def _format_alert(self, alert_type: str, message: str) -> str:
        """Format alert message as text array."""
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

        alert_items = [message]

        msg = f"""{emoji} <b>{title}</b>

{self._build_section("Detail", alert_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

    def _format_system_status(
        self,
        balance: float,
        equity: float,
        open_positions: int,
        session: str,
        ml_status: str,
        uptime_hours: float,
    ) -> str:
        """Format system status message as text array."""
        status_items = [
            f"Bal: <code>${balance:,.0f}</code>",
            f"Eq: <code>${equity:,.0f}</code>",
            f"Pos: <code>{open_positions}</code>",
            f"Session: <code>{session}</code>",
            f"ML: <code>{ml_status}</code>",
            f"Uptime: <code>{uptime_hours:.1f}h</code>",
        ]

        msg = f"""🤖 <b>STATUS</b> 🟢

{self._build_section("System", status_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""
        return msg.strip()

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
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send trade open notification with ALL features."""
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

        msg = self._format_trade_open(trade, context or {})
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
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send trade close notification with ALL status."""
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

        msg = self._format_trade_close(trade, context or {})
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
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send market condition update with full context."""
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

        msg = self._format_market_update(condition, context or {})
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
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send bot startup notification with ALL features as text array."""
        ctx = context or {}

        config_items = [
            f"Symbol: <code>{symbol}</code>",
            f"Mode: <code>{mode}</code>",
            f"Capital: <code>${capital:,.2f}</code>",
            f"Balance: <code>${balance:,.2f}</code>",
            f"ML: <code>{ml_model_status}</code>",
        ]

        risk_items = [
            f"Risk/Trade: <code>{ctx.get('risk_per_trade', 1)}%</code>",
            f"Max Daily Loss: <code>{ctx.get('max_daily_loss', 5)}%</code>",
            f"Max Total Loss: <code>{ctx.get('max_total_loss', 10)}%</code>",
            f"SL: <code>Smart (ATR-based + Broker safety net)</code>",
            f"Max Lot: <code>{ctx.get('max_lot', 0.02)}</code>",
            f"Max Positions: <code>{ctx.get('max_positions', 2)}</code>",
            f"Cooldown: <code>{ctx.get('cooldown_seconds', 150)}s</code>",
        ]

        # Risk state (loaded from file)
        daily_loss = ctx.get("daily_loss", 0)
        total_loss = ctx.get("total_loss", 0)
        consec = ctx.get("consecutive_losses", 0)
        risk_mode = ctx.get("risk_mode", "normal")

        state_items = [
            f"Mode: <code>{risk_mode.upper()}</code>",
            f"Daily Loss: <code>${daily_loss:.2f}</code>",
            f"Total Loss: <code>${total_loss:.2f}</code>",
            f"Streak: <code>{consec}L</code>",
        ]

        session = ctx.get("session", "Unknown")
        can_trade = ctx.get("can_trade", False)
        vol = ctx.get("volatility", "unknown")
        session_icon = "✅" if can_trade else "⛔"

        session_items = [
            f"{session_icon} {session}",
            f"Volatility: <code>{vol}</code>",
        ]

        news_emoji = "✅" if news_status == "SAFE" else "⚠️"

        msg = f"""🚀 <b>BOT STARTED</b>

{self._build_section("Config", config_items)}

{self._build_section("Risk Settings", risk_items)}

{self._build_section("Risk State", state_items)}

{self._build_section("Session", session_items)}

{news_emoji} News: <code>{news_status}</code>
⏰ {datetime.now(WIB).strftime('%Y-%m-%d %H:%M')} WIB"""
        await self.send_message(msg.strip())
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

        news_items = [
            f"Event: <code>{event_name[:40]}</code>",
            f"Reason: <code>{reason[:50]}</code>",
            f"Buffer: <code>{buffer_minutes}m</code>",
        ]

        msg = f"""{emoji} <b>NEWS</b> {condition}

{self._build_section("Detail", news_items)}

⏰ {datetime.now(WIB).strftime('%H:%M')} WIB"""

        await self.send_message(msg.strip())
        logger.info(f"Telegram: News alert sent - {event_name}")

    async def send_hourly_analysis(
        self,
        # Account info
        balance: float,
        equity: float,
        floating_pnl: float,
        # Position info
        open_positions: int,
        position_details: list,
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
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send comprehensive hourly analysis report with ALL features."""
        now = datetime.now(WIB)
        ctx = context or {}

        # Floating P/L emoji
        float_prefix = "+" if floating_pnl >= 0 else ""
        daily_prefix = "+" if daily_pnl >= 0 else ""

        # Risk mode indicator
        risk_display = risk_mode.upper()

        # Market quality indicator
        quality_display = market_quality.upper()

        # Can trade indicator
        can_trade = ml_confidence >= dynamic_threshold and market_quality.lower() != "avoid"
        trade_status = "READY" if can_trade else "WAIT"

        # === Section 1: Account ===
        account_items = [
            f"Bal: <code>${balance:,.2f}</code>",
            f"Eq: <code>${equity:,.2f}</code>",
            f"Float: <b>{float_prefix}${floating_pnl:.2f}</b>",
            f"Day: <b>{daily_prefix}${daily_pnl:.2f}</b> ({daily_trades} trades)",
        ]

        # === Section 2: Positions ===
        pos_items = []
        for pos in position_details[:5]:
            t = pos.get("ticket", 0)
            d = pos.get("direction", "?")
            p = pos.get("profit", 0)
            m = pos.get("momentum", 0)
            tp_prob = pos.get("tp_probability", 50)
            ps = f"+${p:.2f}" if p >= 0 else f"-${abs(p):.2f}"
            pos_items.append(f"#{t} {d}: <b>{ps}</b> M:{m:+.0f} TP:{tp_prob:.0f}%")
        if not pos_items:
            pos_items = ["No positions"]

        # === Section 3: Market ===
        h1_bias = ctx.get("h1_bias", "NEUTRAL")
        atr = ctx.get("atr", 0)
        spread = ctx.get("spread", 0)
        market_items = [
            f"{symbol} <code>${current_price:,.2f}</code>",
            f"ATR: <code>{atr:.2f}</code> | Spread: <code>{spread:.1f}</code>",
            f"Session: <code>{session}</code>",
            f"Regime: <code>{regime}</code> | Vol: <code>{volatility}</code>",
            f"H1 Bias: <code>{h1_bias}</code>",
        ]

        # === Section 4: AI Signal ===
        smc_signal = ctx.get("smc_signal", "")
        smc_conf = ctx.get("smc_confidence", 0)
        ai_items = [
            f"ML: <code>{ml_signal}</code> {ml_confidence:.0%} / thresh {dynamic_threshold:.0%}",
            f"SMC: <code>{smc_signal or 'NONE'}</code> ({smc_conf:.0%})",
            f"Quality: <code>{quality_display}</code> (score:{market_score}) → {trade_status}",
        ]

        # === Section 5: Risk ===
        consec = ctx.get("consecutive_losses", 0)
        total_loss = ctx.get("total_loss", 0)
        risk_items = [
            f"Mode: <code>{risk_display}</code>",
            f"Daily Loss: <code>${abs(min(0, daily_pnl)):.2f}</code> / <code>${max_daily_loss:.2f}</code>",
            f"Total Loss: <code>${total_loss:.2f}</code> | Streak: <code>{consec}L</code>",
        ]

        # === Section 6: Entry Filters ===
        filters = ctx.get("entry_filters", [])
        filter_items = []
        for f in filters:
            passed = f.get("passed", True)
            name = f.get("name", "")
            detail = f.get("detail", "")
            icon = "✅" if passed else "❌"
            filter_items.append(f"{icon} {name}: {detail}")

        # === Section 7: Bot ===
        bot_items = [
            f"Uptime: <code>{uptime_hours:.1f}h</code> | Loops: <code>{total_loops}</code>",
            f"Avg Exec: <code>{avg_execution_ms:.0f}ms</code>",
        ]

        news_emoji = "✅" if news_status == "SAFE" else "⚠️"

        # Build sections list (skip empty)
        sections = [
            self._build_section("Account", account_items),
            self._build_section(f"Positions ({open_positions})", pos_items),
            self._build_section("Market", market_items),
            self._build_section("AI Signal", ai_items),
            self._build_section("Risk", risk_items),
        ]
        if filter_items:
            sections.append(self._build_section("Entry Filters", filter_items))
        sections.append(self._build_section("Bot", bot_items))

        body = "\n\n".join(s for s in sections if s)

        msg = f"""📊 <b>HOURLY</b> {now.strftime('%H:%M')} WIB

{body}

{news_emoji} News: <code>{news_status}</code>
⏰ {now.strftime('%Y-%m-%d %H:%M')} WIB"""

        await self.send_message(msg.strip(), disable_notification=True)
        logger.info("Telegram: Hourly analysis report sent")

    async def send_shutdown_message(
        self,
        balance: float,
        total_trades: int,
        total_profit: float,
        uptime_hours: float,
        # ALL extra context as dict
        context: dict = None,
    ):
        """Send bot shutdown notification with ALL status."""
        ctx = context or {}
        profit_str = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        emoji = "✅" if total_profit >= 0 else "❌"

        session_items = [
            f"Balance: <code>${balance:,.2f}</code>",
            f"Total Trades: <code>{total_trades}</code>",
            f"{emoji} P/L: <b>{profit_str}</b>",
            f"Uptime: <code>{uptime_hours:.1f}h</code>",
        ]

        risk_mode = ctx.get("risk_mode", "normal")
        daily_loss = ctx.get("daily_loss", 0)
        daily_profit = ctx.get("daily_profit", 0)
        total_loss = ctx.get("total_loss", 0)
        consec = ctx.get("consecutive_losses", 0)
        session = ctx.get("session", "Unknown")
        risk_items = [
            f"Mode: <code>{risk_mode.upper()}</code> | Streak: <code>{consec}L</code>",
            f"Daily: <code>+${daily_profit:.2f}</code> / <code>-${daily_loss:.2f}</code>",
            f"Total Loss: <code>${total_loss:.2f}</code>",
            f"Session: <code>{session}</code>",
        ]

        msg = f"""🔴 <b>BOT STOPPED</b>

{self._build_section("Session Summary", session_items)}

{self._build_section("Risk State", risk_items)}

⏰ {datetime.now(WIB).strftime('%Y-%m-%d %H:%M')} WIB"""
        await self.send_message(msg.strip())
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
            ml_model_status="Loaded (76 features)",
            context={
                "risk_per_trade": 1,
                "max_daily_loss": 5,
                "max_total_loss": 10,
                "max_lot": 0.02,
                "max_positions": 2,
                "cooldown_seconds": 150,
                "daily_loss": 0,
                "total_loss": 0,
                "consecutive_losses": 0,
                "risk_mode": "normal",
                "session": "London-NY Overlap",
                "can_trade": True,
                "volatility": "medium",
            },
        )

        # Test trade open
        await notifier.notify_trade_open(
            ticket=12345678,
            symbol="XAUUSD",
            order_type="BUY",
            lot_size=0.02,
            entry_price=2850.00,
            stop_loss=2840.00,
            take_profit=2870.00,
            ml_confidence=0.71,
            signal_reason="Bullish BOS + FVG confirmed by ML",
            regime="medium_volatility",
            volatility="medium",
            context={
                "dynamic_threshold": 0.55,
                "market_quality": "good",
                "market_score": 72,
                "smc_signal": "BUY",
                "smc_confidence": 0.75,
                "smc_fvg": True,
                "smc_ob": True,
                "smc_bos": True,
                "smc_choch": False,
                "session": "London-NY Overlap",
                "h1_bias": "BULLISH",
                "risk_mode": "normal",
                "daily_loss": 0,
                "consecutive_losses": 0,
                "entry_filters": [
                    {"name": "Flash Crash", "passed": True, "detail": "OK"},
                    {"name": "Regime Filter", "passed": True, "detail": "medium_volatility"},
                    {"name": "Risk Check", "passed": True, "detail": "OK"},
                    {"name": "Session Filter", "passed": True, "detail": "London-NY Overlap"},
                    {"name": "ML Confidence", "passed": True, "detail": "71% >= 55%"},
                    {"name": "Cooldown", "passed": True, "detail": "OK"},
                ],
            },
        )

        # Test trade close
        await notifier.notify_trade_close(
            ticket=12345678,
            symbol="XAUUSD",
            order_type="BUY",
            lot_size=0.02,
            entry_price=2850.00,
            close_price=2865.00,
            profit=30.00,
            profit_pips=150,
            balance_before=6130.00,
            balance_after=6160.00,
            duration_seconds=2700,
            ml_confidence=0.71,
            regime="medium_volatility",
            volatility="medium",
            context={
                "exit_reason": "take_profit",
                "risk_mode": "normal",
                "daily_loss": 0,
                "daily_profit": 30.00,
                "consecutive_losses": 0,
                "total_loss": 0,
                "session_trades": 3,
                "session_wins": 2,
                "session_profit": 30.00,
                "win_rate": 66.7,
                "session": "London-NY Overlap",
            },
        )

        # Test market update
        await notifier.notify_market_update(
            symbol="XAUUSD",
            price=2855.50,
            regime="medium_volatility",
            volatility="medium",
            ml_signal="BUY",
            ml_confidence=0.68,
            trend_direction="UPTREND",
            session="London-NY Overlap",
            can_trade=True,
            atr=12.5,
            spread=3.2,
            context={
                "h1_bias": "BULLISH",
                "dynamic_threshold": 0.55,
                "market_quality": "good",
                "market_score": 72,
                "smc_signal": "BUY",
                "smc_confidence": 0.75,
                "risk_mode": "normal",
                "daily_loss": 0,
                "consecutive_losses": 0,
                "session_trades": 1,
                "session_profit": 30.00,
            },
        )

        # Test shutdown
        await notifier.send_shutdown_message(
            balance=6160.00,
            total_trades=3,
            total_profit=45.00,
            uptime_hours=8.5,
            context={
                "risk_mode": "normal",
                "daily_loss": 0,
                "daily_profit": 45.00,
                "total_loss": 0,
                "consecutive_losses": 0,
                "session": "NY Close",
            },
        )

        await notifier.close()

    asyncio.run(test())
