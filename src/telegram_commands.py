"""
Telegram Command Handlers
=========================
Handles all Telegram bot commands separately from main_live.py.

Commands:
  /status     — Bot status & account overview
  /market     — Current market analysis & signals
  /risk       — Risk management state & settings
  /positions  — Open positions detail
  /pos        — Alias for /positions
  /daily      — Daily trading summary
  /filters    — Entry filter status
  /help       — List all available commands

Integration:
    from src.telegram_commands import register_commands
    register_commands(bot)  # bot = TradingBot instance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date
from zoneinfo import ZoneInfo
from loguru import logger

WIB = ZoneInfo("Asia/Jakarta")


def _fmt_usd(value: float) -> str:
    """Format USD value with sign."""
    if value >= 0:
        return f"+${value:.2f}"
    return f"-${abs(value):.2f}"


def _timestamp() -> str:
    """Return formatted WIB timestamp."""
    return datetime.now(WIB).strftime('%H:%M')


def register_commands(bot):
    """
    Register all Telegram commands on the bot instance.

    Args:
        bot: TradingBot instance (has .telegram, .mt5, .smart_risk, etc.)
    """
    tg = bot.telegram
    build = tg._build_section

    # ------------------------------------------------------------------
    # /status — Bot status & account overview
    # ------------------------------------------------------------------
    async def cmd_status():
        balance = bot.mt5.account_balance or 0
        equity = bot.mt5.account_equity or 0
        floating = equity - balance
        session_status = bot.session_filter.get_status_report()
        risk_rec = bot.smart_risk.get_trading_recommendation()
        risk_state = bot.smart_risk.get_state()
        uptime = (datetime.now() - bot._start_time).total_seconds() / 3600
        avg_ms = (sum(bot._execution_times) / len(bot._execution_times) * 1000) if bot._execution_times else 0
        wr = (bot._total_session_wins / bot._total_session_trades * 100) if bot._total_session_trades > 0 else 0
        can_icon = "✅" if session_status.get("can_trade", False) else "⛔"

        items_acct = [
            f"Bal: <code>${balance:,.2f}</code>",
            f"Eq: <code>${equity:,.2f}</code>",
            f"Float: <b>{_fmt_usd(floating)}</b>",
        ]
        items_session = [
            f"Trades: <code>{bot._total_session_trades}</code> ({bot._total_session_wins}W) | WR: <code>{wr:.1f}%</code>",
            f"P/L: <b>{_fmt_usd(bot._total_session_profit)}</b>",
        ]
        items_risk = [
            f"Mode: <code>{risk_rec.get('mode', 'normal').upper()}</code>",
            f"Daily Loss: <code>${risk_state.daily_loss:.2f}</code> | Streak: <code>{risk_state.consecutive_losses}L</code>",
            f"Total Loss: <code>${bot.smart_risk._total_loss:.2f}</code>",
        ]
        items_bot = [
            f"{can_icon} {session_status.get('current_session', 'Unknown')} | Vol: <code>{session_status.get('volatility', '?')}</code>",
            f"Uptime: <code>{uptime:.1f}h</code> | Loops: <code>{bot._loop_count}</code> | Exec: <code>{avg_ms:.0f}ms</code>",
        ]

        return f"""🤖 <b>STATUS</b>

{build("Account", items_acct)}

{build("Session", items_session)}

{build("Risk", items_risk)}

{build("Bot", items_bot)}

⏰ {_timestamp()} WIB""".strip()

    cmd_status._cmd_desc = "Bot status & account"

    # ------------------------------------------------------------------
    # /market — Current market analysis
    # ------------------------------------------------------------------
    async def cmd_market():
        tick = bot.mt5.get_tick(bot.config.symbol)
        price = tick.bid if tick else 0
        spread = (tick.ask - tick.bid) if tick else 0

        session_status = bot.session_filter.get_status_report()
        h1_bias = getattr(bot, "_h1_bias_cache", "NEUTRAL")
        regime = getattr(bot, "_last_regime", None)
        regime_str = regime.value if regime else "unknown"
        ml_signal = getattr(bot, "_last_ml_signal", "HOLD")
        ml_conf = getattr(bot, "_last_ml_confidence", 0)
        smc_signal = getattr(bot, "_last_raw_smc_signal", "")
        smc_conf = getattr(bot, "_last_raw_smc_confidence", 0)
        threshold = getattr(bot, "_last_dynamic_threshold", 0.55)
        quality = getattr(bot, "_last_market_quality", "unknown")
        score = getattr(bot, "_last_market_score", 0)

        try:
            df = bot.mt5.get_market_data(bot.config.symbol, bot.config.execution_timeframe, 50)
            atr = df["atr"].tail(1).item() if "atr" in df.columns else 0
        except Exception:
            atr = 0

        sig_emoji = {"BUY": "🟢", "SELL": "🔴"}.get(ml_signal, "⚪")
        can_icon = "✅ READY" if session_status.get("can_trade", False) else "⛔ WAIT"

        items_price = [
            f"<b>{bot.config.symbol}</b> <code>${price:.2f}</code>",
            f"ATR: <code>{atr:.2f}</code> | Spread: <code>{spread:.1f}</code>",
        ]
        items_signal = [
            f"{sig_emoji} ML: <code>{ml_signal}</code> {ml_conf:.0%} / thresh {threshold:.0%}",
            f"SMC: <code>{smc_signal or 'NONE'}</code> ({smc_conf:.0%})",
            f"Quality: <code>{quality.upper()}</code> (score:{score})",
            f"H1 Bias: <code>{h1_bias}</code>",
        ]
        items_market = [
            f"Regime: <code>{regime_str}</code> | Vol: <code>{session_status.get('volatility', '?')}</code>",
            f"Session: <code>{session_status.get('current_session', 'Unknown')}</code>",
            f"Status: {can_icon}",
        ]

        return f"""📊 <b>MARKET</b>

{build("Price", items_price)}

{build("AI Signal", items_signal)}

{build("Market", items_market)}

⏰ {_timestamp()} WIB""".strip()

    cmd_market._cmd_desc = "Market analysis & signals"

    # ------------------------------------------------------------------
    # /risk — Risk management state
    # ------------------------------------------------------------------
    async def cmd_risk():
        risk_state = bot.smart_risk.get_state()
        risk_rec = bot.smart_risk.get_trading_recommendation()
        balance = bot.mt5.account_balance or bot.config.capital
        max_daily_usd = bot.smart_risk.max_daily_loss_usd
        max_total_usd = balance * bot.smart_risk.max_total_loss_percent / 100

        items_settings = [
            f"Risk/Trade: <code>{bot.config.risk.risk_per_trade}%</code>",
            f"Max Daily Loss: <code>{bot.config.risk.max_daily_loss}%</code> (${max_daily_usd:.2f})",
            f"Max Total Loss: <code>{bot.smart_risk.max_total_loss_percent}%</code> (${max_total_usd:.2f})",
            f"Max Lot: <code>{bot.smart_risk.max_lot_size}</code>",
            f"Max Positions: <code>{bot.smart_risk.max_concurrent_positions}</code>",
        ]
        items_state = [
            f"Mode: <code>{risk_rec.get('mode', 'normal').upper()}</code>",
            f"Daily Loss: <code>${risk_state.daily_loss:.2f}</code> / <code>${max_daily_usd:.2f}</code>",
            f"Daily Profit: <code>${risk_state.daily_profit:.2f}</code>",
            f"Total Loss: <code>${bot.smart_risk._total_loss:.2f}</code>",
            f"Streak: <code>{risk_state.consecutive_losses}L</code>",
        ]
        items_rec = [
            f"Lot: <code>{risk_rec.get('recommended_lot', 0)}</code>",
            f"Reason: <code>{risk_rec.get('reason', '')[:60]}</code>",
        ]

        return f"""🛡 <b>RISK</b>

{build("Settings", items_settings)}

{build("Current State", items_state)}

{build("Recommendation", items_rec)}

⏰ {_timestamp()} WIB""".strip()

    cmd_risk._cmd_desc = "Risk management state"

    # ------------------------------------------------------------------
    # /positions — Open positions detail
    # ------------------------------------------------------------------
    async def cmd_positions():
        positions = bot.mt5.get_open_positions(
            symbol=bot.config.symbol,
            magic=bot.config.magic_number,
        )
        if positions is None or len(positions) == 0:
            return f"📭 <b>POSITIONS</b>\n\n└ No open positions\n\n⏰ {_timestamp()} WIB"

        pos_items = []
        total_profit = 0
        for row in positions.iter_rows(named=True):
            ticket = row.get("ticket", 0)
            direction = "BUY" if row.get("type", 0) == 0 else "SELL"
            profit = row.get("profit", 0)
            total_profit += profit
            open_price = row.get("price_open", 0)
            current = row.get("price_current", 0)
            sl = row.get("sl", 0)
            tp = row.get("tp", 0)
            lot = row.get("volume", 0)

            guard = bot.smart_risk._position_guards.get(ticket)
            momentum = guard.momentum_score if guard else 0

            pos_items.append(f"#{ticket} {direction} <code>{lot}</code>")
            pos_items.append(f"  Open: <code>{open_price:.2f}</code> -> Now: <code>{current:.2f}</code>")
            pos_items.append(f"  SL: <code>{sl:.2f}</code> | TP: <code>{tp:.2f}</code>")
            pos_items.append(f"  P/L: <b>{_fmt_usd(profit)}</b> | M: <code>{momentum:+.0f}</code>")

        summary = [f"Total: <code>{len(positions)}</code> positions, <b>{_fmt_usd(total_profit)}</b>"]

        return f"""📈 <b>POSITIONS</b>

{build("Open", pos_items)}

{build("Summary", summary)}

⏰ {_timestamp()} WIB""".strip()

    cmd_positions._cmd_desc = "Open positions detail"

    # ------------------------------------------------------------------
    # /daily — Daily trading summary
    # ------------------------------------------------------------------
    async def cmd_daily():
        balance = bot.mt5.account_balance or bot.config.capital
        trades = bot._total_session_trades
        wins = bot._total_session_wins
        losses = trades - wins
        wr = (wins / trades * 100) if trades > 0 else 0
        day_change = ((balance - bot._daily_start_balance) / bot._daily_start_balance * 100) if bot._daily_start_balance > 0 else 0
        day_str = f"+{day_change:.2f}%" if day_change >= 0 else f"{day_change:.2f}%"

        items_balance = [
            f"Start: <code>${bot._daily_start_balance:,.2f}</code>",
            f"Now: <code>${balance:,.2f}</code> (<b>{day_str}</b>)",
            f"P/L: <b>{_fmt_usd(bot._total_session_profit)}</b>",
        ]
        items_stats = [
            f"Trades: <code>{trades}</code>",
            f"Wins: <code>{wins}</code> | Losses: <code>{losses}</code>",
            f"Win Rate: <code>{wr:.1f}%</code>",
        ]

        return f"""📋 <b>DAILY</b> {date.today().strftime('%Y-%m-%d')}

{build("Balance", items_balance)}

{build("Stats", items_stats)}

⏰ {_timestamp()} WIB""".strip()

    cmd_daily._cmd_desc = "Daily trading summary"

    # ------------------------------------------------------------------
    # /filters — Entry filter status
    # ------------------------------------------------------------------
    async def cmd_filters():
        filters = getattr(bot, "_last_filter_results", [])
        if not filters:
            return f"🔍 <b>FILTERS</b>\n\n└ No filter data yet (wait for next candle)\n\n⏰ {_timestamp()} WIB"

        filter_items = []
        for f in filters:
            icon = "✅" if f.get("passed", True) else "❌"
            filter_items.append(f"{icon} {f.get('name', '')}: <code>{f.get('detail', '')}</code>")

        passed = sum(1 for f in filters if f.get("passed", True))
        total = len(filters)

        return f"""🔍 <b>FILTERS</b> ({passed}/{total} passed)

{build("Entry Filters", filter_items)}

⏰ {_timestamp()} WIB""".strip()

    cmd_filters._cmd_desc = "Entry filter status"

    # ------------------------------------------------------------------
    # Register all commands
    # ------------------------------------------------------------------
    tg.register_command("status", cmd_status)
    tg.register_command("s", cmd_status)          # alias
    tg.register_command("market", cmd_market)
    tg.register_command("m", cmd_market)           # alias
    tg.register_command("risk", cmd_risk)
    tg.register_command("positions", cmd_positions)
    tg.register_command("pos", cmd_positions)      # alias
    tg.register_command("p", cmd_positions)        # alias
    tg.register_command("daily", cmd_daily)
    tg.register_command("d", cmd_daily)            # alias
    tg.register_command("filters", cmd_filters)
    tg.register_command("f", cmd_filters)          # alias

    logger.info("Telegram commands registered: /status /market /risk /positions /daily /filters /help")
