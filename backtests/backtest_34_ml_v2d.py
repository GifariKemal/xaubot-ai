"""
Backtest #34 ML-V2D -- Time Filter + ML V2 Model D (76 features)
=================================================================
Clone of backtest_34_time_filter.py but using model_d.pkl from
backtests/36_ml_v2_results/ instead of the V1 xgboost_model.pkl.

Model D: 76 features (53 base + 8 H1 + 7 continuous SMC + 4 regime + 4 PA)
Test AUC: 0.7339 (+5.5% vs live model)

Usage:
    python backtests/backtest_34_ml_v2d.py
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys
import os
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, MarketRegime
from src.config import get_config
from src.dynamic_confidence import DynamicConfidenceManager, create_dynamic_confidence, MarketQuality
from backtests.ml_v2.ml_v2_model import TradingModelV2
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

WIB = ZoneInfo("Asia/Jakarta")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Path to model_d.pkl
MODEL_D_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "36_ml_v2_results", "model_d.pkl"
)


# --- Enums & Dataclasses ---

class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"

class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    SMART_TP = "smart_tp"
    PEAK_PROTECT = "peak_protect"
    EARLY_EXIT = "early_exit"
    EARLY_CUT = "early_cut"
    MAX_LOSS = "max_loss"
    STALL = "stall"
    TREND_REVERSAL = "trend_reversal"
    TIMEOUT = "timeout"
    WEEKEND_CLOSE = "weekend_close"
    TRAILING_SL = "trailing_sl"
    BREAKEVEN_EXIT = "breakeven_exit"
    DAILY_LIMIT = "daily_limit"
    REGIME_DANGER = "regime_danger"
    MARKET_SIGNAL = "market_signal"

class TradingMode(Enum):
    NORMAL = "normal"
    RECOVERY = "recovery"
    PROTECTED = "protected"
    STOPPED = "stopped"

@dataclass
class SimulatedTrade:
    ticket: int
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    profit_usd: float
    profit_pips: float
    result: TradeResult
    exit_reason: ExitReason
    smc_confidence: float
    regime: str
    session: str
    signal_reason: str
    has_bos: bool = False
    has_choch: bool = False
    has_fvg: bool = False
    has_ob: bool = False
    atr_at_entry: float = 0.0
    rr_ratio: float = 0.0
    trading_mode: str = "normal"
    h1_trend: str = "NEUTRAL"
    wib_hour: int = 0
    weekday: int = 0

@dataclass
class BacktestStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_usd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    avoided_signals: int = 0
    daily_limit_stops: int = 0
    recovery_mode_trades: int = 0
    session_blocked: int = 0
    h1_filtered: int = 0
    time_filtered: int = 0


# --- Time Filter Backtest with ML V2 Model D ---

class TimeFilterBacktestV2D:
    """#31B base + time-of-hour/day-of-week filtering + ML V2 Model D."""

    def __init__(
        self,
        capital: float = 5000.0,
        max_daily_loss_percent: float = 5.0,
        max_loss_per_trade_percent: float = 1.0,
        base_lot_size: float = 0.01,
        max_lot_size: float = 0.02,
        recovery_lot_size: float = 0.01,
        max_concurrent_positions: int = 2,
        min_profit_to_protect: float = 5.0,
        max_drawdown_from_peak: float = 50.0,
        trade_cooldown_bars: int = 10,
        # #24B base
        skip_tokyo_london: bool = True,
        early_cut_momentum: float = -50.0,
        early_cut_loss_pct: float = 30.0,
        be_mult: float = 2.0,
        trail_start_mult: float = 4.0,
        trail_step_mult: float = 3.0,
        # #28B: Smart breakeven
        be_profit_lock_atr_mult: float = 0.5,
        # === #34 TIME FILTER PARAMS ===
        skip_wib_hours: Set[int] = None,      # Set of WIB hours to skip
        skip_weekdays: Set[int] = None,        # Set of weekdays to skip (0=Mon, 4=Fri)
        trend_reversal_mult: float = 0.6,
    ):
        self.capital = capital
        self.max_daily_loss_usd = capital * (max_daily_loss_percent / 100)
        self.max_loss_per_trade = capital * (max_loss_per_trade_percent / 100)
        self.base_lot_size = base_lot_size
        self.max_lot_size = max_lot_size
        self.recovery_lot_size = recovery_lot_size
        self.max_concurrent_positions = max_concurrent_positions
        self.min_profit_to_protect = min_profit_to_protect
        self.max_drawdown_from_peak = max_drawdown_from_peak
        self.trade_cooldown_bars = trade_cooldown_bars
        self.trend_reversal_mult = trend_reversal_mult

        self.skip_tokyo_london = skip_tokyo_london
        self.early_cut_momentum = early_cut_momentum
        self.early_cut_loss_pct = early_cut_loss_pct
        self.be_mult = be_mult
        self.trail_start_mult = trail_start_mult
        self.trail_step_mult = trail_step_mult
        self.be_profit_lock_atr_mult = be_profit_lock_atr_mult

        # #34 params
        self.skip_wib_hours = skip_wib_hours or set()
        self.skip_weekdays = skip_weekdays or set()

        config = get_config()
        self.smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
        self.features = FeatureEngineer()
        self.dynamic_confidence = create_dynamic_confidence()

        # === ML V2 Model D (instead of V1) ===
        self.ml_model = TradingModelV2(model_path=MODEL_D_PATH)
        try:
            self.ml_model.load()
            print(f"  ML V2 Model D loaded: {len(self.ml_model.feature_names)} features")
        except Exception as e:
            print(f"  [WARN] ML V2 Model D load failed: {e}")

        self.regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
        try:
            self.regime_detector.load()
        except Exception:
            pass

        self._ticket_counter = 2340000

    def _get_session_from_time(self, dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        wib_time = dt.astimezone(WIB)
        hour = wib_time.hour
        if 6 <= hour < 15:
            return "Sydney-Tokyo", True, 0.5
        elif 15 <= hour < 16:
            if self.skip_tokyo_london:
                return "Tokyo-London Overlap", False, 0.0
            return "Tokyo-London Overlap", True, 0.75
        elif 16 <= hour < 19:
            return "London Early", True, 0.8
        elif 19 <= hour < 24:
            return "London-NY Overlap (Golden)", True, 1.0
        elif 0 <= hour < 4:
            return "NY Session", True, 0.9
        else:
            return "Off Hours", False, 0.0

    def _get_wib_hour(self, dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(WIB).hour

    def _get_wib_weekday(self, dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(WIB).weekday()

    def _hours_to_golden(self, dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        wib = dt.astimezone(WIB)
        if 19 <= wib.hour < 24:
            return 0
        target = wib.replace(hour=19, minute=0, second=0, microsecond=0)
        if wib.hour >= 19:
            target += timedelta(days=1)
        return max(0, (target - wib).total_seconds() / 3600)

    def _is_near_weekend_close(self, dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        wib = dt.astimezone(WIB)
        return wib.weekday() == 5 and wib.hour >= 4 and wib.minute >= 30

    def _calculate_lot_size(self, confidence, regime, trading_mode, session_mult):
        if trading_mode == TradingMode.STOPPED:
            return 0
        lot = self.base_lot_size
        if trading_mode in (TradingMode.RECOVERY, TradingMode.PROTECTED):
            lot = self.recovery_lot_size
        else:
            if confidence >= 0.65:
                lot = self.max_lot_size
            elif confidence >= 0.55:
                lot = self.base_lot_size
            else:
                lot = self.recovery_lot_size
        if regime.lower() in ["high_volatility", "crisis"]:
            lot = self.recovery_lot_size
        lot = max(0.01, lot * session_mult)
        return round(lot, 2)

    def _calc_ema(self, data, period):
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        for val in data[period:]:
            ema = (val - ema) * multiplier + ema
        return ema

    def _get_h1_trend(self, df_h1_slice):
        if df_h1_slice is None or len(df_h1_slice) < 20:
            return "NEUTRAL"
        closes = df_h1_slice["close"].to_list()
        ema20 = self._calc_ema(closes, 20)
        current_price = closes[-1]
        if current_price > ema20 * 1.001:
            return "BULLISH"
        elif current_price < ema20 * 0.999:
            return "BEARISH"
        return "NEUTRAL"

    def _simulate_trade_exit(
        self, df, entry_idx, direction, entry_price, take_profit, stop_loss,
        lot_size, daily_loss_so_far, feature_cols, max_bars=100,
    ):
        pip_value = 10
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        closes = df["close"].to_list()
        times = df["time"].to_list()

        atr = 12.0
        if "atr" in df.columns:
            atr_list = df["atr"].to_list()
            if entry_idx < len(atr_list) and atr_list[entry_idx] is not None:
                atr = atr_list[entry_idx]

        adaptive_breakeven_pips = atr * self.be_mult
        adaptive_trail_start_pips = atr * self.trail_start_mult
        adaptive_trail_step_pips = atr * self.trail_step_mult
        reversal_momentum_threshold = atr * self.trend_reversal_mult
        min_loss_for_reversal_exit = atr * 0.8

        if self.be_profit_lock_atr_mult > 0:
            be_lock_distance = atr * self.be_profit_lock_atr_mult
        else:
            be_lock_distance = 2.0

        profit_history = []
        peak_profit = 0.0
        stall_count = 0
        reversal_warnings = 0
        current_sl = stop_loss
        breakeven_moved = False

        if direction == "BUY":
            target_tp_profit = (take_profit - entry_price) / 0.1 * pip_value * lot_size
        else:
            target_tp_profit = (entry_price - take_profit) / 0.1 * pip_value * lot_size

        cached_ml_signal = ""
        cached_ml_confidence = 0.5

        for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            high = highs[i]
            low = lows[i]
            close = closes[i]
            current_time = times[i]

            if direction == "BUY":
                current_pips = (close - entry_price) / 0.1
                pip_profit_from_entry = current_pips
            else:
                current_pips = (entry_price - close) / 0.1
                pip_profit_from_entry = current_pips
            current_profit = current_pips * pip_value * lot_size

            profit_history.append(current_profit)
            if current_profit > peak_profit:
                peak_profit = current_profit

            bars_since_entry = i - entry_idx

            if bars_since_entry % 4 == 0 and self.ml_model.fitted:
                try:
                    df_slice = df.head(i + 1)
                    ml_pred = self.ml_model.predict(df_slice, feature_cols)
                    cached_ml_signal = ml_pred.signal
                    cached_ml_confidence = ml_pred.confidence
                except Exception:
                    pass

            momentum = 0.0
            if len(profit_history) >= 3:
                recent = profit_history[-5:] if len(profit_history) >= 5 else profit_history
                profit_change = recent[-1] - recent[0]
                momentum = max(-100, min(100, (profit_change / 10) * 50))
            profit_growing = momentum > 0

            # A.0 TP hit
            if direction == "BUY" and high >= take_profit:
                pips = (take_profit - entry_price) / 0.1
                return pips * pip_value * lot_size, pips, ExitReason.TAKE_PROFIT, i, take_profit
            elif direction == "SELL" and low <= take_profit:
                pips = (entry_price - take_profit) / 0.1
                return pips * pip_value * lot_size, pips, ExitReason.TAKE_PROFIT, i, take_profit

            # A.0b Trailing SL hit
            if breakeven_moved and current_sl > 0:
                if direction == "BUY" and low <= current_sl:
                    pips = (current_sl - entry_price) / 0.1
                    reason = ExitReason.TRAILING_SL if pip_profit_from_entry >= adaptive_trail_start_pips else ExitReason.BREAKEVEN_EXIT
                    return pips * pip_value * lot_size, pips, reason, i, current_sl
                elif direction == "SELL" and high >= current_sl:
                    pips = (entry_price - current_sl) / 0.1
                    reason = ExitReason.TRAILING_SL if pip_profit_from_entry >= adaptive_trail_start_pips else ExitReason.BREAKEVEN_EXIT
                    return pips * pip_value * lot_size, pips, reason, i, current_sl

            # A.1 Breakeven (#28B: Smart)
            if pip_profit_from_entry >= adaptive_breakeven_pips and not breakeven_moved:
                if direction == "BUY":
                    current_sl = entry_price + be_lock_distance
                else:
                    current_sl = entry_price - be_lock_distance
                breakeven_moved = True

            # A.2 Trailing SL
            if pip_profit_from_entry >= adaptive_trail_start_pips:
                trail_distance = adaptive_trail_step_pips * 0.1
                if direction == "BUY":
                    new_trail_sl = close - trail_distance
                    if new_trail_sl > current_sl:
                        current_sl = new_trail_sl
                else:
                    new_trail_sl = close + trail_distance
                    if current_sl == 0 or new_trail_sl < current_sl:
                        current_sl = new_trail_sl

            # A.3 Peak protect
            if peak_profit > self.min_profit_to_protect:
                drawdown_pct = ((peak_profit - current_profit) / peak_profit) * 100 if peak_profit > 0 else 0
                if drawdown_pct > self.max_drawdown_from_peak:
                    return current_profit, current_pips, ExitReason.PEAK_PROTECT, i, close

            # A.4 Market analysis
            if bars_since_entry % 5 == 0 and bars_since_entry >= 5 and i >= 20:
                ma_fast = np.mean(closes[i-4:i+1])
                ma_slow = np.mean(closes[i-19:i+1])
                trend = "BULLISH" if ma_fast > ma_slow * 1.001 else ("BEARISH" if ma_fast < ma_slow * 0.999 else "NEUTRAL")
                roc = (closes[i] / closes[max(0,i-4)] - 1) * 100
                mom_dir = "BULLISH" if roc > 0.3 else ("BEARISH" if roc < -0.3 else "NEUTRAL")

                rsi_val = None
                if "rsi" in df.columns:
                    rsi_list = df["rsi"].to_list()
                    if i < len(rsi_list):
                        rsi_val = rsi_list[i]

                urgency = 0
                should_exit = False
                if cached_ml_confidence > 0.75:
                    if (direction == "BUY" and cached_ml_signal == "SELL") or (direction == "SELL" and cached_ml_signal == "BUY"):
                        should_exit = True; urgency += 2
                if rsi_val:
                    if (rsi_val > 75 and direction == "BUY") or (rsi_val < 25 and direction == "SELL"):
                        should_exit = True; urgency += 2
                if (direction == "BUY" and trend == "BEARISH" and mom_dir == "BEARISH") or \
                   (direction == "SELL" and trend == "BULLISH" and mom_dir == "BULLISH"):
                    should_exit = True; urgency += 3

                if should_exit and current_profit > self.min_profit_to_protect / 2:
                    return current_profit, current_pips, ExitReason.MARKET_SIGNAL, i, close
                if urgency >= 7 and current_profit > 0:
                    return current_profit, current_pips, ExitReason.MARKET_SIGNAL, i, close

            # A.5 Weekend close
            if self._is_near_weekend_close(current_time):
                if current_profit > 0 or current_profit > -10:
                    return current_profit, current_pips, ExitReason.WEEKEND_CLOSE, i, close

            # B.1 Smart TP
            if current_profit >= 15:
                if current_profit >= 40:
                    return current_profit, current_pips, ExitReason.SMART_TP, i, close
                if current_profit >= 25 and momentum < -30:
                    return current_profit, current_pips, ExitReason.SMART_TP, i, close
                if peak_profit > 30 and current_profit < peak_profit * 0.6:
                    return current_profit, current_pips, ExitReason.PEAK_PROTECT, i, close
                if current_profit >= 20:
                    progress = (current_profit / target_tp_profit) * 100 if target_tp_profit > 0 else 0
                    progress_score = min(40, max(0, progress * 0.4))
                    momentum_score = ((momentum + 100) / 200) * 30
                    time_penalty = min(10, bars_since_entry / 4 * 2)
                    tp_probability = progress_score + momentum_score + 10 - time_penalty
                    if tp_probability < 25:
                        return current_profit, current_pips, ExitReason.SMART_TP, i, close

            # B.2 Smart Early Exit
            if 5 <= current_profit < 15:
                if momentum < -50 and cached_ml_confidence >= 0.65:
                    is_reversal = (direction == "BUY" and cached_ml_signal == "SELL") or (direction == "SELL" and cached_ml_signal == "BUY")
                    if is_reversal:
                        return current_profit, current_pips, ExitReason.EARLY_EXIT, i, close

            # B.3 Early cut
            if current_profit < 0:
                loss_percent_of_max = abs(current_profit) / self.max_loss_per_trade * 100
                if momentum < self.early_cut_momentum and loss_percent_of_max >= self.early_cut_loss_pct:
                    return current_profit, current_pips, ExitReason.EARLY_CUT, i, close

            # B.4 Trend Reversal
            is_ml_reversal = False
            if (direction == "BUY" and cached_ml_signal == "SELL" and cached_ml_confidence >= 0.75) or \
               (direction == "SELL" and cached_ml_signal == "BUY" and cached_ml_confidence >= 0.75):
                is_ml_reversal = True
                reversal_warnings += 1
            loss_moderate = abs(current_profit) > (self.max_loss_per_trade * 0.4)
            if is_ml_reversal and current_profit < -8 and loss_moderate:
                return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close
            if reversal_warnings >= 3 and current_profit < -10:
                return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close

            # B.5 Max loss
            if current_profit <= -(self.max_loss_per_trade * 0.50):
                htg = self._hours_to_golden(current_time)
                if htg <= 1 and htg > 0 and momentum > -40:
                    pass
                else:
                    return current_profit, current_pips, ExitReason.MAX_LOSS, i, close

            # B.6 Stall
            if len(profit_history) >= 10:
                recent_range = max(profit_history[-10:]) - min(profit_history[-10:])
                if recent_range < 3 and current_profit < -15:
                    stall_count += 1
                    if stall_count >= 5:
                        return current_profit, current_pips, ExitReason.STALL, i, close

            # B.7 Daily loss limit
            potential_daily_loss = daily_loss_so_far + abs(min(0, current_profit))
            if potential_daily_loss >= self.max_daily_loss_usd:
                return current_profit, current_pips, ExitReason.DAILY_LIMIT, i, close

            # C) Time-based
            if bars_since_entry >= 16 and current_profit < 5 and not profit_growing:
                if current_profit >= 0 or current_profit > -15:
                    return current_profit, current_pips, ExitReason.TIMEOUT, i, close
            if bars_since_entry >= 24 and (current_profit < 10 or not profit_growing):
                return current_profit, current_pips, ExitReason.TIMEOUT, i, close
            if bars_since_entry >= 32:
                return current_profit, current_pips, ExitReason.TIMEOUT, i, close

            # C.2 ATR trend reversal
            if bars_since_entry > 10:
                recent_closes = closes[i-5:i+1]
                mom = recent_closes[-1] - recent_closes[0]
                if (direction == "BUY" and mom < -reversal_momentum_threshold) or \
                   (direction == "SELL" and mom > reversal_momentum_threshold):
                    if current_profit < -min_loss_for_reversal_exit:
                        return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close

        final_idx = min(entry_idx + max_bars - 1, len(df) - 1)
        final_price = closes[final_idx]
        pips = ((final_price - entry_price) if direction == "BUY" else (entry_price - final_price)) / 0.1
        return pips * pip_value * lot_size, pips, ExitReason.TIMEOUT, final_idx, final_price

    def run(self, df_m15, df_h1, start_date=None, end_date=None, initial_capital=5000.0):
        stats = BacktestStats()
        capital = initial_capital
        peak_capital = initial_capital
        stats.equity_curve.append(capital)

        daily_loss = 0.0
        daily_profit = 0.0
        daily_trades = 0
        consecutive_losses = 0
        trading_mode = TradingMode.NORMAL
        current_date = None

        feature_cols = []
        if self.ml_model.fitted and self.ml_model.feature_names:
            feature_cols = [f for f in self.ml_model.feature_names if f in df_m15.columns]
            missing = [f for f in self.ml_model.feature_names if f not in df_m15.columns]
            if missing:
                print(f"  [WARN] Missing {len(missing)} features: {missing[:5]}...")

        times_m15 = df_m15["time"].to_list()
        times_h1 = df_h1["time"].to_list() if df_h1 is not None else []

        start_idx = next((i for i, t in enumerate(times_m15) if t >= start_date), 100) if start_date else 100
        end_idx = next((i for i, t in enumerate(times_m15) if t > end_date), len(df_m15) - 100) if end_date else len(df_m15) - 100

        last_trade_idx = -self.trade_cooldown_bars * 2

        skip_hours_str = ",".join(str(h) for h in sorted(self.skip_wib_hours)) if self.skip_wib_hours else "none"
        skip_days_str = ",".join(DAY_NAMES[d] for d in sorted(self.skip_weekdays)) if self.skip_weekdays else "none"
        print(f"  #34 ML-V2D skip hours(WIB): [{skip_hours_str}], skip days: [{skip_days_str}]")
        print(f"  ML features available: {len(feature_cols)}/{len(self.ml_model.feature_names) if self.ml_model.fitted else 0}")
        print(f"  Date range: {times_m15[start_idx]} to {times_m15[end_idx - 1]}")
        print(f"  Total bars: {end_idx - start_idx}")

        for i in range(start_idx, end_idx):
            if i - last_trade_idx < self.trade_cooldown_bars:
                continue

            current_time = times_m15[i]
            trade_date = current_time.date() if hasattr(current_time, 'date') else current_time
            if current_date is None or trade_date != current_date:
                daily_loss = 0.0
                daily_profit = 0.0
                daily_trades = 0
                current_date = trade_date
                if consecutive_losses < 2:
                    trading_mode = TradingMode.NORMAL

            if trading_mode == TradingMode.STOPPED:
                continue

            session_name, can_trade, lot_mult = self._get_session_from_time(current_time)
            if not can_trade:
                if session_name == "Tokyo-London Overlap":
                    stats.session_blocked += 1
                continue

            if hasattr(current_time, 'weekday') and current_time.weekday() >= 5:
                continue

            # #34: Time-of-hour filter
            wib_hour = self._get_wib_hour(current_time)
            if wib_hour in self.skip_wib_hours:
                stats.time_filtered += 1
                continue

            # #34: Day-of-week filter
            wib_weekday = self._get_wib_weekday(current_time)
            if wib_weekday in self.skip_weekdays:
                stats.time_filtered += 1
                continue

            df_slice = df_m15.head(i + 1)

            regime = "normal"
            try:
                if self.regime_detector.fitted:
                    regime_state = self.regime_detector.get_current_state(df_slice)
                    if regime_state:
                        regime = regime_state.regime.value
                        if regime_state.regime == MarketRegime.CRISIS:
                            continue
                        if regime_state.recommendation == "SLEEP":
                            continue
            except Exception:
                pass

            try:
                ml_signal = ""
                ml_confidence = 0.5
                if self.ml_model.fitted and feature_cols:
                    ml_pred = self.ml_model.predict(df_slice, feature_cols)
                    ml_signal = ml_pred.signal
                    ml_confidence = ml_pred.confidence

                market_analysis = self.dynamic_confidence.analyze_market(
                    session=session_name, regime=regime, volatility="medium",
                    trend_direction=regime, has_smc_signal=True,
                    ml_signal=ml_signal, ml_confidence=ml_confidence,
                )
                if market_analysis.quality == MarketQuality.AVOID:
                    stats.avoided_signals += 1
                    continue
            except Exception:
                pass

            try:
                smc_signal = self.smc.generate_signal(df_slice)
            except Exception:
                continue

            if smc_signal is None:
                continue

            # #31B: H1 Price vs EMA20 filter
            h1_trend = "NEUTRAL"
            if df_h1 is not None and len(times_h1) > 0:
                h1_idx = 0
                for j, t in enumerate(times_h1):
                    if t <= current_time:
                        h1_idx = j
                    else:
                        break
                if h1_idx > 20:
                    df_h1_slice = df_h1.head(h1_idx + 1)
                    h1_trend = self._get_h1_trend(df_h1_slice)

            if smc_signal.signal_type == "BUY" and h1_trend != "BULLISH":
                stats.h1_filtered += 1
                continue
            if smc_signal.signal_type == "SELL" and h1_trend != "BEARISH":
                stats.h1_filtered += 1
                continue

            recent_df = df_slice.tail(10)
            recent_bos = recent_df["bos"].to_list() if "bos" in df_slice.columns else []
            recent_choch = recent_df["choch"].to_list() if "choch" in df_slice.columns else []
            recent_fvg_bull = recent_df["is_fvg_bull"].to_list() if "is_fvg_bull" in df_slice.columns else []
            recent_fvg_bear = recent_df["is_fvg_bear"].to_list() if "is_fvg_bear" in df_slice.columns else []
            recent_obs = recent_df["ob"].to_list() if "ob" in df_slice.columns else []

            has_bos = 1 in recent_bos or -1 in recent_bos
            has_choch = 1 in recent_choch or -1 in recent_choch
            has_fvg = any(recent_fvg_bull) or any(recent_fvg_bear)
            has_ob = 1 in recent_obs or -1 in recent_obs

            atr_at_entry = 12.0
            if "atr" in df_slice.columns:
                atr_val = df_slice.tail(1)["atr"].item()
                if atr_val is not None and atr_val > 0:
                    atr_at_entry = atr_val

            confidence = smc_signal.confidence
            ml_agrees = (smc_signal.signal_type == "BUY" and ml_signal == "BUY") or \
                        (smc_signal.signal_type == "SELL" and ml_signal == "SELL")
            if ml_agrees:
                confidence = (smc_signal.confidence + ml_confidence) / 2
            if regime == "high_volatility":
                confidence *= 0.9

            lot_size = self._calculate_lot_size(confidence, regime, trading_mode, lot_mult)
            if lot_size <= 0:
                continue

            if trading_mode == TradingMode.RECOVERY:
                stats.recovery_mode_trades += 1

            entry_price = smc_signal.entry_price
            take_profit_price = smc_signal.take_profit
            stop_loss_price = smc_signal.stop_loss
            risk = abs(entry_price - stop_loss_price)
            rr = abs(take_profit_price - entry_price) / risk if risk > 0 else 0

            profit, pips, exit_reason, exit_idx, exit_price = self._simulate_trade_exit(
                df=df_m15, entry_idx=i, direction=smc_signal.signal_type,
                entry_price=entry_price, take_profit=take_profit_price,
                stop_loss=stop_loss_price, lot_size=lot_size,
                daily_loss_so_far=daily_loss, feature_cols=feature_cols,
            )

            self._ticket_counter += 1
            result = TradeResult.WIN if profit > 0 else (TradeResult.LOSS if profit < 0 else TradeResult.BREAKEVEN)

            trade = SimulatedTrade(
                ticket=self._ticket_counter,
                entry_time=current_time,
                exit_time=times_m15[exit_idx] if exit_idx < len(times_m15) else times_m15[-1],
                direction=smc_signal.signal_type,
                entry_price=entry_price, exit_price=exit_price,
                stop_loss=stop_loss_price, take_profit=take_profit_price,
                lot_size=lot_size, profit_usd=profit, profit_pips=pips,
                result=result, exit_reason=exit_reason,
                smc_confidence=confidence, regime=regime,
                session=session_name, signal_reason=smc_signal.reason,
                has_bos=has_bos, has_choch=has_choch,
                has_fvg=has_fvg, has_ob=has_ob,
                atr_at_entry=atr_at_entry, rr_ratio=rr,
                trading_mode=trading_mode.value,
                h1_trend=h1_trend,
                wib_hour=wib_hour,
                weekday=wib_weekday,
            )
            stats.trades.append(trade)
            stats.total_trades += 1
            daily_trades += 1
            capital += profit

            if profit > 0:
                stats.wins += 1
                stats.total_profit += profit
                daily_profit += profit
                consecutive_losses = 0
                if trading_mode == TradingMode.RECOVERY:
                    trading_mode = TradingMode.NORMAL
            else:
                stats.losses += 1
                stats.total_loss += abs(profit)
                daily_loss += abs(profit)
                consecutive_losses += 1

            if daily_loss >= self.max_daily_loss_usd:
                trading_mode = TradingMode.STOPPED
                stats.daily_limit_stops += 1
            elif consecutive_losses >= 3 or daily_loss >= self.max_daily_loss_usd * 0.6:
                trading_mode = TradingMode.PROTECTED
            elif consecutive_losses >= 2:
                trading_mode = TradingMode.RECOVERY

            if capital > peak_capital:
                peak_capital = capital
            drawdown_pct = (peak_capital - capital) / peak_capital * 100
            drawdown_usd = peak_capital - capital
            if drawdown_pct > stats.max_drawdown:
                stats.max_drawdown = drawdown_pct
                stats.max_drawdown_usd = drawdown_usd

            stats.equity_curve.append(capital)
            last_trade_idx = exit_idx

            if stats.total_trades % 100 == 0:
                print(f"  {stats.total_trades} trades processed...")

        if stats.total_trades > 0:
            stats.win_rate = stats.wins / stats.total_trades * 100
            stats.avg_win = stats.total_profit / stats.wins if stats.wins > 0 else 0
            stats.avg_loss = stats.total_loss / stats.losses if stats.losses > 0 else 0
            stats.avg_trade = (stats.total_profit - stats.total_loss) / stats.total_trades
            stats.profit_factor = stats.total_profit / stats.total_loss if stats.total_loss > 0 else float("inf")
            win_prob = stats.wins / stats.total_trades
            loss_prob = stats.losses / stats.total_trades
            stats.expectancy = (win_prob * stats.avg_win) - (loss_prob * stats.avg_loss)
            returns = [t.profit_usd for t in stats.trades]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                stats.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        return stats


def analyze_hourly_daily(stats):
    """Analyze trade performance by WIB hour and weekday."""
    hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    day_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

    for t in stats.trades:
        h = t.wib_hour
        hour_stats[h]["trades"] += 1
        hour_stats[h]["pnl"] += t.profit_usd
        if t.result == TradeResult.WIN:
            hour_stats[h]["wins"] += 1

        d = t.weekday
        day_stats[d]["trades"] += 1
        day_stats[d]["pnl"] += t.profit_usd
        if t.result == TradeResult.WIN:
            day_stats[d]["wins"] += 1

    return hour_stats, day_stats


# --- Main ---

def main():
    print("=" * 70)
    print("XAUBOT AI -- #34 ML-V2D: Time Filter + ML V2 Model D (76 features)")
    print("Base: #34 Time Filter | ML: model_d.pkl (Test AUC 0.7339)")
    print("=" * 70)

    config = get_config()
    mt5_conn = MT5Connector(
        login=config.mt5_login, password=config.mt5_password,
        server=config.mt5_server, path=config.mt5_path,
    )
    mt5_conn.connect()
    print(f"\nConnected to MT5")

    print("Fetching XAUUSD M15 historical data...")
    df_m15 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="M15", count=50000)
    print(f"  M15: {len(df_m15)} bars")

    print("Fetching XAUUSD H1 historical data...")
    df_h1 = mt5_conn.get_market_data(symbol="XAUUSD", timeframe="H1", count=15000)
    print(f"  H1: {len(df_h1)} bars")

    times = df_m15["time"].to_list()
    print(f"  M15 range: {times[0]} to {times[-1]}")

    end_date = datetime.now()
    start_date = datetime(2025, 8, 1)
    data_start = times[0]
    if hasattr(data_start, 'replace') and data_start.tzinfo:
        start_date = start_date.replace(tzinfo=data_start.tzinfo)
        end_date = end_date.replace(tzinfo=data_start.tzinfo)
    if data_start > start_date:
        start_date = data_start + timedelta(days=5)

    print(f"\n  Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # === Calculate base indicators ===
    print("\nCalculating M15 indicators...")
    features = FeatureEngineer()
    smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
    df_m15 = features.calculate_all(df_m15, include_ml_features=True)
    df_m15 = smc.calculate_all(df_m15)

    regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
    try:
        regime_detector.load()
        df_m15 = regime_detector.predict(df_m15)
        print("  HMM regime loaded")
    except Exception:
        print("  [WARN] HMM not available")
        df_m15 = df_m15.with_columns([
            pl.lit(1).alias("regime"),
            pl.lit("medium_volatility").alias("regime_name"),
        ])

    print("Calculating H1 indicators...")
    df_h1 = features.calculate_all(df_h1, include_ml_features=False)
    # H1 also needs SMC for V2 H1 features (ob_top, fvg_top, bos, etc.)
    df_h1 = smc.calculate_all(df_h1)
    print("  H1 base + SMC indicators calculated")

    # === Add V2 features (23 new features for model_d) ===
    print("\nAdding ML V2 features (23 new features)...")
    fe_v2 = MLV2FeatureEngineer()
    df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)
    v2_cols = fe_v2.get_v2_feature_columns()
    available_v2 = [c for c in v2_cols if c in df_m15.columns]
    print(f"  V2 features available: {len(available_v2)}/{len(v2_cols)}")
    print(f"  Total M15 columns: {len(df_m15.columns)}")

    baseline_34_pnl = 2806.56  # #31B baseline for comparison

    # ===============================================================
    # PHASE 1: Run baseline to analyze per-hour and per-day performance
    # ===============================================================
    print(f"\n{'=' * 60}")
    print("  PHASE 1: Baseline analysis (no time filter, ML V2 Model D)")
    bt_base = TimeFilterBacktestV2D()
    stats_base = bt_base.run(df_m15=df_m15, df_h1=df_h1, start_date=start_date, end_date=end_date)
    net_base = stats_base.total_profit - stats_base.total_loss

    hour_stats, day_stats = analyze_hourly_daily(stats_base)

    print(f"\n  Baseline (V2D): {stats_base.total_trades} trades, {stats_base.win_rate:.1f}% WR, ${net_base:,.2f}")

    # Print hourly analysis
    print(f"\n  === HOURLY ANALYSIS (WIB) ===")
    print(f"  {'Hour':>4} {'Trades':>7} {'Wins':>5} {'WR':>7} {'PnL':>10} {'Avg':>8}")
    print(f"  {'-' * 45}")
    hour_ranking = []
    for h in sorted(hour_stats.keys()):
        s = hour_stats[h]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
        marker = " <-- WORST" if s["trades"] >= 5 and (wr < 75 or s["pnl"] < 0) else ""
        print(f"  {h:>4} {s['trades']:>7} {s['wins']:>5} {wr:>6.1f}% ${s['pnl']:>9,.2f} ${avg:>7,.2f}{marker}")
        if s["trades"] >= 5:
            hour_ranking.append((h, wr, s["pnl"], s["trades"]))

    # Sort by PnL (worst first)
    hour_ranking.sort(key=lambda x: x[2])
    worst_2_hours = set(h[0] for h in hour_ranking[:2])
    worst_3_hours = set(h[0] for h in hour_ranking[:3])

    print(f"\n  Worst 2 hours (by PnL): {sorted(worst_2_hours)}")
    print(f"  Worst 3 hours (by PnL): {sorted(worst_3_hours)}")

    # Print daily analysis
    print(f"\n  === DAY-OF-WEEK ANALYSIS ===")
    print(f"  {'Day':>4} {'Trades':>7} {'Wins':>5} {'WR':>7} {'PnL':>10} {'Avg':>8}")
    print(f"  {'-' * 45}")
    day_ranking = []
    for d in sorted(day_stats.keys()):
        s = day_stats[d]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
        marker = " <-- WORST" if s["trades"] >= 10 and (wr < 78 or s["pnl"] < 0) else ""
        print(f"  {DAY_NAMES[d]:>4} {s['trades']:>7} {s['wins']:>5} {wr:>6.1f}% ${s['pnl']:>9,.2f} ${avg:>7,.2f}{marker}")
        if s["trades"] >= 10:
            day_ranking.append((d, wr, s["pnl"], s["trades"]))

    day_ranking.sort(key=lambda x: x[2])
    worst_day = {day_ranking[0][0]} if day_ranking else set()

    print(f"\n  Worst day: {[DAY_NAMES[d] for d in sorted(worst_day)]}")

    # ===============================================================
    # PHASE 2: Run filtered configs based on Phase 1 analysis
    # ===============================================================
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Testing filtered configurations (ML V2 Model D)")

    configs = [
        ("A: Skip worst 2 hours", {
            "skip_wib_hours": worst_2_hours,
        }),
        ("B: Skip worst 3 hours", {
            "skip_wib_hours": worst_3_hours,
        }),
        ("C: Skip worst day", {
            "skip_weekdays": worst_day,
        }),
        ("D: Worst 2h + worst day", {
            "skip_wib_hours": worst_2_hours,
            "skip_weekdays": worst_day,
        }),
        ("E: Worst 3h + worst day", {
            "skip_wib_hours": worst_3_hours,
            "skip_weekdays": worst_day,
        }),
    ]

    all_results = []

    for cfg_name, cfg_params in configs:
        print(f"\n{'=' * 60}")
        print(f"  Config: {cfg_name}")

        bt = TimeFilterBacktestV2D(**cfg_params)
        stats = bt.run(df_m15=df_m15, df_h1=df_h1, start_date=start_date, end_date=end_date, initial_capital=5000.0)
        net_pnl = stats.total_profit - stats.total_loss
        diff = net_pnl - baseline_34_pnl

        buy_trades = [t for t in stats.trades if t.direction == "BUY"]
        sell_trades = [t for t in stats.trades if t.direction == "SELL"]
        buy_wins = sum(1 for t in buy_trades if t.result == TradeResult.WIN)
        sell_wins = sum(1 for t in sell_trades if t.result == TradeResult.WIN)
        buy_wr = buy_wins / len(buy_trades) * 100 if buy_trades else 0
        sell_wr = sell_wins / len(sell_trades) * 100 if sell_trades else 0
        buy_pnl = sum(t.profit_usd for t in buy_trades)
        sell_pnl = sum(t.profit_usd for t in sell_trades)

        print(f"\n  [{cfg_name}] Results:")
        print(f"    Trades: {stats.total_trades} | WR: {stats.win_rate:.1f}%")
        print(f"    Net PnL: ${net_pnl:,.2f} | PF: {stats.profit_factor:.2f}")
        print(f"    Max DD: {stats.max_drawdown:.1f}% | Sharpe: {stats.sharpe_ratio:.2f}")
        print(f"    BUY: {len(buy_trades)}, {buy_wr:.1f}% WR, ${buy_pnl:,.2f}")
        print(f"    SELL: {len(sell_trades)}, {sell_wr:.1f}% WR, ${sell_pnl:,.2f}")
        print(f"    Time-filtered: {stats.time_filtered} signals blocked")
        print(f"    vs #31B: ${diff:+,.2f}")

        all_results.append((cfg_name, stats, net_pnl, diff))

    # === FINAL SUMMARY ===
    print(f"\n{'=' * 70}")
    print("#34 ML-V2D: TIME FILTER + ML V2 MODEL D -- ALL CONFIGURATIONS")
    print("=" * 70)

    print(f"\n  {'Config':<25} {'Trades':>6} {'WR':>6} {'Net PnL':>10} {'DD':>6} {'Sharpe':>7} {'PF':>5} {'Blocked':>8} {'vs #31B':>10}")
    print(f"  {'-' * 90}")
    print(f"  {'#31B (V1 model)':<25} {'625':>6} {'81.8%':>6} {'$2,807':>10} {'2.5%':>6} {'3.97':>7} {'2.19':>5} {'--':>8} {'--':>10}")
    print(f"  {'V2D Baseline (no filt)':<25} {stats_base.total_trades:>6} {stats_base.win_rate:>5.1f}% ${net_base:>9,.2f} {stats_base.max_drawdown:>5.1f}% {stats_base.sharpe_ratio:>7.2f} {stats_base.profit_factor:>5.2f} {'--':>8} ${net_base - baseline_34_pnl:>+9,.2f}")
    for cfg_name, stats, net_pnl, diff in all_results:
        blocked = stats.time_filtered
        print(f"  {cfg_name:<25} {stats.total_trades:>6} {stats.win_rate:>5.1f}% ${net_pnl:>9,.2f} {stats.max_drawdown:>5.1f}% {stats.sharpe_ratio:>7.2f} {stats.profit_factor:>5.2f} {blocked:>8} ${diff:>+9,.2f}")

    best_pnl = -999999
    best_name = ""
    best_stats = None
    for entry in all_results:
        if entry[2] > best_pnl:
            best_pnl = entry[2]
            best_name = entry[0]
            best_stats = entry[1]

    # Also compare baseline (no filter)
    if net_base > best_pnl:
        best_pnl = net_base
        best_name = "Baseline (no filter)"
        best_stats = stats_base

    print(f"\n  Best config: {best_name}")

    # Exit reasons
    print(f"\n  Exit Reasons (best config):")
    exit_counts = {}
    for t in best_stats.trades:
        r = t.exit_reason.value
        exit_counts[r] = exit_counts.get(r, 0) + 1
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        pct = count / best_stats.total_trades * 100 if best_stats.total_trades > 0 else 0
        print(f"    {reason:20s}: {count} ({pct:.1f}%)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "34_ml_v2d_results")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"ml_v2d_time_filter_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write(f"#34 ML-V2D: Time Filter + ML V2 Model D Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"ML Model: model_d.pkl (76 features, Test AUC 0.7339)\n")
        f.write(f"Base comparison: #31B (625 trades, 81.8% WR, $2,807)\n\n")

        f.write(f"=== BASELINE (V2D, no time filter) ===\n")
        f.write(f"  Trades: {stats_base.total_trades}, WR: {stats_base.win_rate:.1f}%, "
                f"PnL: ${net_base:,.2f}, DD: {stats_base.max_drawdown:.1f}%, "
                f"Sharpe: {stats_base.sharpe_ratio:.2f}, PF: {stats_base.profit_factor:.2f}\n\n")

        f.write(f"=== HOURLY ANALYSIS (WIB) ===\n")
        for h in sorted(hour_stats.keys()):
            s = hour_stats[h]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
            f.write(f"  {h:>2}:00 WIB  {s['trades']:>4} trades  {wr:>5.1f}% WR  ${s['pnl']:>8,.2f}  avg ${avg:>6,.2f}\n")

        f.write(f"\nWorst 2 hours: {sorted(worst_2_hours)}\n")
        f.write(f"Worst 3 hours: {sorted(worst_3_hours)}\n")

        f.write(f"\n=== DAY-OF-WEEK ANALYSIS ===\n")
        for d in sorted(day_stats.keys()):
            s = day_stats[d]
            wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
            f.write(f"  {DAY_NAMES[d]:>3}  {s['trades']:>4} trades  {wr:>5.1f}% WR  ${s['pnl']:>8,.2f}  avg ${avg:>6,.2f}\n")

        f.write(f"\nWorst day: {[DAY_NAMES[d] for d in sorted(worst_day)]}\n")

        f.write(f"\n=== FILTERED RESULTS ===\n")
        for cfg_name, stats, net_pnl, diff in all_results:
            f.write(f"  {cfg_name}: {stats.total_trades} trades, {stats.win_rate:.1f}% WR, "
                    f"${net_pnl:,.2f}, DD: {stats.max_drawdown:.1f}%, "
                    f"Sharpe: {stats.sharpe_ratio:.2f}, PF: {stats.profit_factor:.2f}, "
                    f"Blocked: {stats.time_filtered}, vs #31B: ${diff:+,.2f}\n")
        f.write(f"\nBest: {best_name}\n")
    print(f"  Log saved: {log_path}")

    try:
        from backtests.backtest_01_smc_only import generate_xlsx_report as gen_xlsx
        xlsx_path = os.path.join(output_dir, f"ml_v2d_time_filter_{timestamp}.xlsx")
        gen_xlsx(best_stats, xlsx_path, start_date, end_date)
        print(f"\n  Report saved: {xlsx_path}")
    except Exception as e:
        print(f"  [WARN] XLSX: {e}")

    mt5_conn.disconnect()

    print(f"\n{'=' * 70}")
    print(f"Output: {output_dir}")
    print(f"  Log: {os.path.basename(log_path)}")
    print("=" * 70)
    print("Backtest complete!")


if __name__ == "__main__":
    main()
