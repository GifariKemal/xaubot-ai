"""
Smart Position Manager
======================
Intelligent position management with:
- Trailing Stop Loss
- Profit Protection
- Market-based Exit Signals
- Dynamic SL/TP Adjustment
- Smart Market Close Handler (NEW)
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Timezone constants
WIB = ZoneInfo("Asia/Jakarta")  # GMT+7
EST = ZoneInfo("America/New_York")  # Market timezone


@dataclass
class PositionAction:
    """Action to take on a position."""
    ticket: int
    action: str  # "HOLD", "CLOSE", "TRAIL_SL", "TAKE_PARTIAL"
    reason: str
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    close_percent: float = 100.0  # For partial close


@dataclass
class MarketCloseAnalysis:
    """Analysis result for market close decision."""
    near_close: bool
    near_weekend: bool
    hours_to_close: float
    recommendation: str  # "CLOSE_PROFIT", "HOLD_LOSS", "CUT_LOSS_WEEKEND", "NORMAL"
    reason: str


class SmartMarketCloseHandler:
    """
    Intelligent market close handler.

    Logic:
    1. Profit + Near Close -> Close to secure profit (jangan sampai hilang TP)
    2. Loss + Still in range -> Hold, wait for volatility on reopen
    3. Loss + Weekend approaching -> Consider cut loss (gap risk)

    Market Hours (XAUUSD):
    - Sunday 5pm EST - Friday 5pm EST (24/5)
    - Daily close around 5pm EST = 05:00 WIB (next day)
    - Weekend gap risk on Monday open
    """

    def __init__(
        self,
        daily_close_hour_wib: int = 5,      # 05:00 WIB = 5pm EST (previous day)
        hours_before_close: float = 2.0,     # Consider "near close" within 2 hours
        weekend_close_hour_wib: int = 5,     # Friday 5pm EST = Saturday 05:00 WIB
        min_profit_to_take: float = 10.0,    # Minimum profit $ to take before close
        max_loss_to_hold: float = 100.0,     # Max loss $ to hold over close
        weekend_loss_cut_percent: float = 50.0,  # Cut loss if > 50% of SL hit before weekend
    ):
        self.daily_close_hour_wib = daily_close_hour_wib
        self.hours_before_close = hours_before_close
        self.weekend_close_hour_wib = weekend_close_hour_wib
        self.min_profit_to_take = min_profit_to_take
        self.max_loss_to_hold = max_loss_to_hold
        self.weekend_loss_cut_percent = weekend_loss_cut_percent

    def analyze(self, profit: float, sl_distance_percent: float = 0.0) -> MarketCloseAnalysis:
        """
        Analyze position status relative to market close.

        Args:
            profit: Current position profit/loss in $
            sl_distance_percent: How much of SL has been hit (0-100%)

        Returns:
            MarketCloseAnalysis with recommendation
        """
        now_wib = datetime.now(WIB)

        # Check if near daily close (05:00 WIB)
        hours_to_daily_close = self._hours_until_time(now_wib, self.daily_close_hour_wib)
        near_daily_close = hours_to_daily_close <= self.hours_before_close

        # Check if near weekend (Friday -> Saturday 05:00 WIB)
        near_weekend, hours_to_weekend = self._check_weekend_proximity(now_wib)

        # Determine hours to relevant close
        if near_weekend:
            hours_to_close = hours_to_weekend
            near_close = True
        else:
            hours_to_close = hours_to_daily_close
            near_close = near_daily_close

        # Make recommendation
        recommendation, reason = self._make_recommendation(
            profit=profit,
            near_close=near_close,
            near_weekend=near_weekend,
            hours_to_close=hours_to_close,
            sl_distance_percent=sl_distance_percent,
        )

        return MarketCloseAnalysis(
            near_close=near_close,
            near_weekend=near_weekend,
            hours_to_close=hours_to_close,
            recommendation=recommendation,
            reason=reason,
        )

    def _hours_until_time(self, now: datetime, target_hour: int) -> float:
        """Calculate hours until target hour today or tomorrow."""
        target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

        if now >= target:
            # Target already passed today, calculate for tomorrow
            target = target + timedelta(days=1)

        delta = target - now
        return delta.total_seconds() / 3600

    def _check_weekend_proximity(self, now: datetime) -> Tuple[bool, float]:
        """
        Check if we're approaching weekend close.

        Weekend close = Saturday 05:00 WIB (Friday 5pm EST)

        Returns:
            (near_weekend, hours_to_weekend_close)
        """
        weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

        # Calculate hours until Saturday 05:00 WIB
        if weekday == 5:  # Saturday
            # Already weekend
            return False, 0
        elif weekday == 6:  # Sunday
            # Market opening soon, not approaching close
            return False, 0
        else:
            # Monday-Friday
            days_until_saturday = (5 - weekday) % 7
            if days_until_saturday == 0:
                days_until_saturday = 7  # Should not happen, but safety

            target = now.replace(hour=self.weekend_close_hour_wib, minute=0, second=0, microsecond=0)
            target = target + timedelta(days=days_until_saturday)

            delta = target - now
            hours_to_weekend = delta.total_seconds() / 3600

            # Consider "near weekend" if within 30 min of close (Saturday ~04:30 WIB)
            # Market closes Saturday 05:00 WIB — Friday night trading is OK
            near_weekend = hours_to_weekend <= 0.5 and weekday == 4  # Friday only

            return near_weekend, hours_to_weekend

    def _make_recommendation(
        self,
        profit: float,
        near_close: bool,
        near_weekend: bool,
        hours_to_close: float,
        sl_distance_percent: float,
    ) -> Tuple[str, str]:
        """
        Make smart recommendation based on conditions.

        Returns:
            (recommendation, reason)
        """
        # Case 1: In profit and near close -> TAKE PROFIT
        if profit >= self.min_profit_to_take and near_close:
            urgency = "WEEKEND" if near_weekend else "daily"
            return (
                "CLOSE_PROFIT",
                f"Take profit ${profit:.2f} before {urgency} close ({hours_to_close:.1f}h remaining)"
            )

        # Case 2: In loss, near weekend, and significant SL hit -> CUT LOSS
        if profit < 0 and near_weekend:
            if sl_distance_percent >= self.weekend_loss_cut_percent:
                return (
                    "CUT_LOSS_WEEKEND",
                    f"Cut loss ${profit:.2f} before weekend (SL {sl_distance_percent:.0f}% hit, gap risk)"
                )
            elif abs(profit) > self.max_loss_to_hold:
                return (
                    "CUT_LOSS_WEEKEND",
                    f"Cut large loss ${profit:.2f} before weekend (gap risk)"
                )
            else:
                return (
                    "HOLD_LOSS",
                    f"Hold small loss ${profit:.2f} over weekend (may recover on Monday volatility)"
                )

        # Case 3: In loss, near daily close but not weekend -> HOLD
        if profit < 0 and near_close and not near_weekend:
            if abs(profit) <= self.max_loss_to_hold:
                return (
                    "HOLD_LOSS",
                    f"Hold loss ${profit:.2f} over daily close (may recover tomorrow)"
                )
            else:
                return (
                    "CUT_LOSS_WEEKEND",  # Reuse for large daily loss
                    f"Consider cutting large loss ${profit:.2f} before close"
                )

        # Case 4: Small profit near close -> Consider taking
        if profit > 0 and profit < self.min_profit_to_take and near_close:
            if hours_to_close < 0.5:  # Very close to close (30 min)
                return (
                    "CLOSE_PROFIT",
                    f"Take small profit ${profit:.2f} (only {hours_to_close*60:.0f}min to close)"
                )

        # Default: Normal operation
        return ("NORMAL", "No market close action needed")

    def get_market_status(self) -> Dict:
        """Get current market status for logging."""
        now_wib = datetime.now(WIB)
        weekday = now_wib.weekday()
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        near_weekend, hours_to_weekend = self._check_weekend_proximity(now_wib)
        hours_to_daily = self._hours_until_time(now_wib, self.daily_close_hour_wib)

        return {
            "time_wib": now_wib.strftime("%H:%M:%S"),
            "day": weekday_names[weekday],
            "hours_to_daily_close": hours_to_daily,
            "hours_to_weekend_close": hours_to_weekend if weekday < 5 else 0,
            "near_weekend": near_weekend,
            "market_open": weekday < 5 or (weekday == 6 and now_wib.hour >= 22),  # Sunday 10pm WIB
        }


class SmartPositionManager:
    """
    Smart position manager with profit protection.

    Features:
    - Trailing stop loss (lock in profits)
    - Breakeven protection
    - Market condition-based exits
    - Momentum reversal detection
    - Regime-based position adjustment
    - Smart Market Close Handler (take profit before close, hold loss if recoverable)
    """

    def __init__(
        self,
        breakeven_pips: float = 15.0,      # Fallback if ATR unavailable
        trail_start_pips: float = 25.0,    # Fallback if ATR unavailable
        trail_step_pips: float = 10.0,     # Fallback if ATR unavailable
        min_profit_to_protect: float = 50.0,  # Minimum $ profit to protect
        max_drawdown_from_peak: float = 30.0,  # Max % drawdown from peak profit
        # ATR-adaptive exit multipliers (#24B: backtest +$373)
        atr_be_mult: float = 2.0,          # Breakeven = ATR * 2.0
        atr_trail_start_mult: float = 4.0, # Trail start = ATR * 4.0
        atr_trail_step_mult: float = 3.0,  # Trail step = ATR * 3.0
        # Market Close Handler settings
        enable_market_close_handler: bool = True,
        min_profit_before_close: float = 10.0,  # Take profit if >= $10 near close
        max_loss_to_hold: float = 100.0,   # Hold loss up to $100 over close
    ):
        self.breakeven_pips = breakeven_pips
        self.trail_start_pips = trail_start_pips
        self.trail_step_pips = trail_step_pips
        self.atr_be_mult = atr_be_mult
        self.atr_trail_start_mult = atr_trail_start_mult
        self.atr_trail_step_mult = atr_trail_step_mult
        self.min_profit_to_protect = min_profit_to_protect
        self.max_drawdown_from_peak = max_drawdown_from_peak

        # Initialize market close handler
        self.enable_market_close_handler = enable_market_close_handler
        self.market_close_handler = SmartMarketCloseHandler(
            min_profit_to_take=min_profit_before_close,
            max_loss_to_hold=max_loss_to_hold,
        )

        # Track peak profit per position
        self._peak_profits: Dict[int, float] = {}
        self._entry_times: Dict[int, datetime] = {}

    def analyze_positions(
        self,
        positions: pl.DataFrame,
        df_market: pl.DataFrame,
        regime_state,
        ml_prediction,
        current_price: float,
    ) -> List[PositionAction]:
        """
        Analyze all positions and decide actions.

        Args:
            positions: DataFrame of open positions
            df_market: Market data DataFrame with indicators
            regime_state: Current market regime
            ml_prediction: Current ML prediction
            current_price: Current market price

        Returns:
            List of PositionAction for each position
        """
        actions = []

        if len(positions) == 0:
            return actions

        # Get market analysis
        market_analysis = self._analyze_market(df_market, regime_state, ml_prediction)

        # Get current ATR for adaptive exit levels (#24B)
        current_atr = None
        if "atr" in df_market.columns:
            atr_val = df_market["atr"].tail(1).item()
            if atr_val is not None and atr_val > 0:
                current_atr = atr_val

        # #33B: Get last candle range for impulse detection
        last_candle_range = None
        if len(df_market) >= 1:
            last_row = df_market.tail(1)
            last_high = last_row["high"].item()
            last_low = last_row["low"].item()
            if last_high is not None and last_low is not None:
                last_candle_range = last_high - last_low

        for row in positions.iter_rows(named=True):
            action = self._analyze_single_position(
                row, market_analysis, current_price, current_atr, last_candle_range
            )
            if action:
                actions.append(action)

        return actions

    def _analyze_market(
        self,
        df: pl.DataFrame,
        regime_state,
        ml_prediction,
    ) -> Dict:
        """Analyze current market conditions."""
        analysis = {
            "trend": "NEUTRAL",
            "momentum": "NEUTRAL",
            "regime": "medium_volatility",
            "ml_signal": "HOLD",
            "ml_confidence": 0.5,
            "should_exit_longs": False,
            "should_exit_shorts": False,
            "urgency": 0,  # 0-10 scale
        }

        if len(df) < 20:
            return analysis

        # Get recent data
        close = df["close"].tail(20).to_numpy()

        # Trend analysis (simple MA comparison)
        ma_fast = np.mean(close[-5:])
        ma_slow = np.mean(close[-20:])

        if ma_fast > ma_slow * 1.001:
            analysis["trend"] = "BULLISH"
        elif ma_fast < ma_slow * 0.999:
            analysis["trend"] = "BEARISH"

        # Momentum analysis (rate of change)
        roc = (close[-1] / close[-5] - 1) * 100
        if roc > 0.3:
            analysis["momentum"] = "BULLISH"
        elif roc < -0.3:
            analysis["momentum"] = "BEARISH"

        # Regime
        if regime_state:
            analysis["regime"] = regime_state.regime.value

            # High volatility = be careful
            if regime_state.regime.value in ["high_volatility", "crisis"]:
                analysis["urgency"] += 3

        # ML signal
        if ml_prediction:
            analysis["ml_signal"] = ml_prediction.signal
            analysis["ml_confidence"] = ml_prediction.confidence

            # Strong opposite signal = consider exit
            if ml_prediction.confidence > 0.75:
                if ml_prediction.signal == "SELL":
                    analysis["should_exit_longs"] = True
                    analysis["urgency"] += 2
                elif ml_prediction.signal == "BUY":
                    analysis["should_exit_shorts"] = True
                    analysis["urgency"] += 2

        # RSI analysis (if available)
        if "rsi" in df.columns:
            rsi = df["rsi"].tail(1).item()
            if rsi and rsi > 75:
                analysis["should_exit_longs"] = True
                analysis["urgency"] += 2
            elif rsi and rsi < 25:
                analysis["should_exit_shorts"] = True
                analysis["urgency"] += 2

        # Trend reversal detection
        if analysis["trend"] == "BEARISH" and analysis["momentum"] == "BEARISH":
            analysis["should_exit_longs"] = True
            analysis["urgency"] += 3
        elif analysis["trend"] == "BULLISH" and analysis["momentum"] == "BULLISH":
            analysis["should_exit_shorts"] = True
            analysis["urgency"] += 3

        return analysis

    def _analyze_single_position(
        self,
        pos: Dict,
        market: Dict,
        current_price: float,
        current_atr: float = None,
        last_candle_range: float = None,
    ) -> Optional[PositionAction]:
        """Analyze a single position and decide action."""
        ticket = pos["ticket"]
        pos_type = pos.get("type", 0)  # Can be int (0=BUY, 1=SELL) or str ("BUY"/"SELL")
        entry_price = pos["price_open"]
        current_sl = pos.get("sl", 0)
        current_tp = pos.get("tp", 0)
        profit = pos.get("profit", 0)
        volume = pos.get("volume", 0.01)

        # Handle both int (MT5 raw) and string (from DataFrame) type formats
        is_buy = pos_type in [0, "BUY", mt5.POSITION_TYPE_BUY if mt5 else 0]

        # Calculate pip profit
        if is_buy:
            pip_profit = (current_price - entry_price) / 0.1  # Gold pips
        else:
            pip_profit = (entry_price - current_price) / 0.1

        # Track peak profit
        if ticket not in self._peak_profits:
            self._peak_profits[ticket] = profit
        else:
            self._peak_profits[ticket] = max(self._peak_profits[ticket], profit)

        peak_profit = self._peak_profits[ticket]

        # === CLOSE CONDITIONS ===

        # 0. SMART MARKET CLOSE HANDLER - Priority check before other conditions
        if self.enable_market_close_handler:
            # Calculate SL distance percent (how much of SL has been hit)
            sl_distance_percent = 0.0
            if current_sl > 0 and entry_price > 0:
                max_loss_distance = abs(entry_price - current_sl)
                if max_loss_distance > 0:
                    current_loss_distance = abs(current_price - entry_price) if profit < 0 else 0
                    sl_distance_percent = (current_loss_distance / max_loss_distance) * 100

            close_analysis = self.market_close_handler.analyze(
                profit=profit,
                sl_distance_percent=sl_distance_percent,
            )

            if close_analysis.recommendation == "CLOSE_PROFIT":
                # Take profit before market close - jangan sampai hilang TP!
                return PositionAction(
                    ticket=ticket,
                    action="CLOSE",
                    reason=f"Market Close: {close_analysis.reason}",
                )
            elif close_analysis.recommendation == "CUT_LOSS_WEEKEND":
                # Cut loss before weekend to avoid gap risk
                return PositionAction(
                    ticket=ticket,
                    action="CLOSE",
                    reason=f"Weekend Risk: {close_analysis.reason}",
                )
            elif close_analysis.recommendation == "HOLD_LOSS":
                # Hold loss - might recover on reopen with volatility
                # Log but don't close, let other conditions potentially trigger
                logger.debug(f"Market Close Hold: {close_analysis.reason}")
                # Continue to check other conditions, but this gives context

        # 1. Regime change to dangerous
        if market["regime"] in ["crisis", "high_volatility"] and profit > self.min_profit_to_protect:
            return PositionAction(
                ticket=ticket,
                action="CLOSE",
                reason=f"Regime danger ({market['regime']}) - Securing ${profit:.2f} profit",
            )

        # 2. Strong opposite signal — only exit at substantial profit (v5)
        # min_profit_to_protect / 2 was too low ($4), now requires 75% of threshold
        signal_exit_threshold = self.min_profit_to_protect * 0.75
        if is_buy and market["should_exit_longs"] and profit > signal_exit_threshold:
            return PositionAction(
                ticket=ticket,
                action="CLOSE",
                reason=f"Bearish signal detected - Securing ${profit:.2f} profit (threshold ${signal_exit_threshold:.0f})",
            )
        elif not is_buy and market["should_exit_shorts"] and profit > signal_exit_threshold:
            return PositionAction(
                ticket=ticket,
                action="CLOSE",
                reason=f"Bullish signal detected - Securing ${profit:.2f} profit (threshold ${signal_exit_threshold:.0f})",
            )

        # 3. Drawdown from peak profit
        if peak_profit > self.min_profit_to_protect:
            drawdown_pct = ((peak_profit - profit) / peak_profit) * 100 if peak_profit > 0 else 0
            if drawdown_pct > self.max_drawdown_from_peak:
                return PositionAction(
                    ticket=ticket,
                    action="CLOSE",
                    reason=f"Profit protection: {drawdown_pct:.0f}% drawdown from peak ${peak_profit:.2f}",
                )

        # 4. High urgency — only exit at substantial profit (v4: raised from $0)
        if market["urgency"] >= 8 and profit > self.min_profit_to_protect:
            return PositionAction(
                ticket=ticket,
                action="CLOSE",
                reason=f"High urgency exit (score: {market['urgency']}) - Securing ${profit:.2f}",
            )

        # === TRAILING STOP CONDITIONS (ATR-adaptive #24B) ===

        # Compute adaptive levels from ATR (fall back to fixed pips if ATR unavailable)
        if current_atr is not None and current_atr > 0:
            # ATR is in price terms; convert to pips (1 pip = 0.1 for gold)
            be_pips = current_atr * self.atr_be_mult / 0.1
            trail_start = current_atr * self.atr_trail_start_mult / 0.1
            trail_step = current_atr * self.atr_trail_step_mult / 0.1
        else:
            be_pips = self.breakeven_pips
            trail_start = self.trail_start_pips
            trail_step = self.trail_step_pips

        # 5. Breakeven protection (#28B: smart BE locks profit at 0.5*ATR instead of fixed $2)
        if pip_profit >= be_pips and current_sl != 0:
            be_lock_distance = current_atr * 0.5 if (current_atr is not None and current_atr > 0) else 2.0
            breakeven_sl = entry_price + (1 if is_buy else -1) * be_lock_distance

            if is_buy and current_sl < breakeven_sl:
                return PositionAction(
                    ticket=ticket,
                    action="TRAIL_SL",
                    reason=f"Moving SL to breakeven ({pip_profit:.1f}/{be_pips:.0f} pips)",
                    new_sl=breakeven_sl,
                )
            elif not is_buy and current_sl > breakeven_sl:
                return PositionAction(
                    ticket=ticket,
                    action="TRAIL_SL",
                    reason=f"Moving SL to breakeven ({pip_profit:.1f}/{be_pips:.0f} pips)",
                    new_sl=breakeven_sl,
                )

        # 6. Trailing stop (after trail_start pips)
        # #33B: Impulse detection — tighten trail when candle range > 1.5x ATR
        if pip_profit >= trail_start:
            is_impulse = False
            if last_candle_range is not None and current_atr is not None and current_atr > 0:
                if last_candle_range > current_atr * 1.5:
                    is_impulse = True

            active_trail_step = trail_step
            if is_impulse:
                active_trail_step = (current_atr * 1.5) / 0.1  # 1.5x ATR in pips
            trail_distance = active_trail_step * 0.1  # Convert to price

            if is_buy:
                new_trail_sl = current_price - trail_distance
                if current_sl < new_trail_sl:
                    return PositionAction(
                        ticket=ticket,
                        action="TRAIL_SL",
                        reason=f"Trailing SL ({pip_profit:.1f}/{trail_start:.0f} pips)",
                        new_sl=new_trail_sl,
                    )
            else:
                new_trail_sl = current_price + trail_distance
                if current_sl > new_trail_sl or current_sl == 0:
                    return PositionAction(
                        ticket=ticket,
                        action="TRAIL_SL",
                        reason=f"Trailing SL ({pip_profit:.1f}/{trail_start:.0f} pips)",
                        new_sl=new_trail_sl,
                    )

        # 7. Default: HOLD
        return PositionAction(
            ticket=ticket,
            action="HOLD",
            reason=f"Holding position ({pip_profit:.1f} pips, ${profit:.2f})",
        )

    def execute_actions(self, actions: List[PositionAction]) -> List[Dict]:
        """Execute position actions via MT5."""
        results = []

        if mt5 is None:
            logger.error("MT5 not available")
            return results

        for action in actions:
            result = {"ticket": action.ticket, "action": action.action, "success": False}

            if action.action == "HOLD":
                result["success"] = True
                result["message"] = action.reason

            elif action.action == "CLOSE":
                close_result = self._close_position(action.ticket)
                result["success"] = close_result["success"]
                result["message"] = close_result.get("message", action.reason)
                if close_result["success"]:
                    logger.info(f"CLOSED #{action.ticket}: {action.reason}")
                    # Clean up tracking
                    self._peak_profits.pop(action.ticket, None)

            elif action.action == "TRAIL_SL":
                trail_result = self._modify_sl(action.ticket, action.new_sl)
                result["success"] = trail_result["success"]
                result["message"] = trail_result.get("message", action.reason)
                if trail_result["success"]:
                    logger.info(f"TRAILED SL #{action.ticket} to {action.new_sl:.2f}: {action.reason}")

            results.append(result)

        return results

    def _close_position(self, ticket: int) -> Dict:
        """Close a position by ticket."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "message": "Position not found"}

        pos = position[0]
        symbol = pos.symbol
        volume = pos.volume
        pos_type = pos.type

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "message": "Cannot get tick"}

        close_price = tick.bid if pos_type == 0 else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Smart exit",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "message": f"Closed at {close_price:.2f}"}
        else:
            return {"success": False, "message": f"Failed: {result.comment} ({result.retcode})"}

    def _modify_sl(self, ticket: int, new_sl: float) -> Dict:
        """Modify stop loss of a position."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "message": "Position not found"}

        pos = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,  # Keep existing TP
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "message": f"SL modified to {new_sl:.2f}"}
        else:
            return {"success": False, "message": f"Failed: {result.comment} ({result.retcode})"}

    def get_position_summary(self, positions: pl.DataFrame) -> Dict:
        """Get summary of all positions."""
        if len(positions) == 0:
            return {"count": 0, "total_profit": 0, "avg_profit": 0}

        total_profit = 0
        for row in positions.iter_rows(named=True):
            total_profit += row.get("profit", 0)

        return {
            "count": len(positions),
            "total_profit": total_profit,
            "avg_profit": total_profit / len(positions),
            "peak_profits": dict(self._peak_profits),
        }
