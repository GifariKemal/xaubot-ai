"""
Backtest #33 — Impulse Trail + Trail Tuning
=============================================
Base: #31B (H1 Price vs EMA20) — 625 trades, 81.8% WR, $2,807, Sharpe 3.97

Idea: Two-part optimization of trailing SL (47.8% of all exits):
1. IMPULSE DETECTION: When candle range > Nx ATR, tighten trail to lock profit fast
   ("Panic Trail" — secure profit from spikes before reversal)
2. TRAIL PARAMETER TUNING: Test wider trail_start from #22C data on #31B base

Configs:
  A: Impulse 1.5x ATR → tight trail 1.0x ATR (aggressive panic trail)
  B: Impulse 1.5x ATR → tight trail 1.5x ATR (moderate panic trail)
  C: Impulse 2.0x ATR → tight trail 1.0x ATR (only extreme impulse)
  D: Impulse 1.5x ATR → tight trail 1.0x + wider trail_start 5.0x (from #22C)
  E: Impulse 1.5x ATR → tight trail 1.0x + wider trail_start 5.0x + step 3.5x

Usage:
    python backtests/backtest_33_impulse_trail.py
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer, SMCSignal
from src.feature_eng import FeatureEngineer
from src.regime_detector import MarketRegimeDetector, MarketRegime
from src.ml_model import TradingModel
from src.config import get_config
from src.dynamic_confidence import DynamicConfidenceManager, create_dynamic_confidence, MarketQuality
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

WIB = ZoneInfo("Asia/Jakarta")


# ─── Enums & Dataclasses ──────────────────────────────────────

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
    impulse_triggered: bool = False

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
    impulse_trail_count: int = 0


# ─── Impulse Trail Backtest ──────────────────────────────────

class ImpulseTrailBacktest:
    """#31B base + impulse trail detection + trail parameter tuning."""

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
        # ═══ #33 IMPULSE TRAIL PARAMS ═══
        impulse_enabled: bool = False,
        impulse_threshold_mult: float = 1.5,   # Candle range > Nx ATR = impulse
        impulse_trail_mult: float = 1.0,        # Tightened trail step during impulse
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

        # #33 params
        self.impulse_enabled = impulse_enabled
        self.impulse_threshold_mult = impulse_threshold_mult
        self.impulse_trail_mult = impulse_trail_mult

        config = get_config()
        self.smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
        self.features = FeatureEngineer()
        self.dynamic_confidence = create_dynamic_confidence()

        self.ml_model = TradingModel(model_path="models/xgboost_model.pkl")
        try:
            self.ml_model.load()
            print("  ML model loaded")
        except Exception:
            print("  [WARN] ML model not loaded")

        self.regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
        try:
            self.regime_detector.load()
        except Exception:
            print("  [WARN] HMM model not loaded")

        self._ticket_counter = 2330000

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
        """#31B: H1 Price vs EMA20."""
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
        opens = df["open"].to_list()
        times = df["time"].to_list()

        atr = 12.0
        if "atr" in df.columns:
            atr_list = df["atr"].to_list()
            if entry_idx < len(atr_list) and atr_list[entry_idx] is not None:
                atr = atr_list[entry_idx]

        adaptive_breakeven_pips = atr * self.be_mult
        adaptive_trail_start_pips = atr * self.trail_start_mult
        adaptive_trail_step_pips = atr * self.trail_step_mult
        # #33: Impulse-tightened trail step
        impulse_trail_step_pips = atr * self.impulse_trail_mult
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
        impulse_triggered = False

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
            candle_open = opens[i]
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

            # ═══ #33 IMPULSE DETECTION ═══
            # Check if current candle range exceeds threshold
            candle_range = high - low
            # Get current ATR (use entry ATR as fallback)
            current_atr = atr
            if "atr" in df.columns and i < len(atr_list) and atr_list[i] is not None:
                current_atr = atr_list[i]

            is_impulse_bar = False
            if self.impulse_enabled and current_atr > 0:
                if candle_range > (current_atr * self.impulse_threshold_mult):
                    is_impulse_bar = True
                    if not impulse_triggered:
                        impulse_triggered = True

            # A.0 TP hit
            if direction == "BUY" and high >= take_profit:
                pips = (take_profit - entry_price) / 0.1
                return pips * pip_value * lot_size, pips, ExitReason.TAKE_PROFIT, i, take_profit, impulse_triggered
            elif direction == "SELL" and low <= take_profit:
                pips = (entry_price - take_profit) / 0.1
                return pips * pip_value * lot_size, pips, ExitReason.TAKE_PROFIT, i, take_profit, impulse_triggered

            # A.0b Trailing SL hit
            if breakeven_moved and current_sl > 0:
                if direction == "BUY" and low <= current_sl:
                    pips = (current_sl - entry_price) / 0.1
                    reason = ExitReason.TRAILING_SL if pip_profit_from_entry >= adaptive_trail_start_pips else ExitReason.BREAKEVEN_EXIT
                    return pips * pip_value * lot_size, pips, reason, i, current_sl, impulse_triggered
                elif direction == "SELL" and high >= current_sl:
                    pips = (entry_price - current_sl) / 0.1
                    reason = ExitReason.TRAILING_SL if pip_profit_from_entry >= adaptive_trail_start_pips else ExitReason.BREAKEVEN_EXIT
                    return pips * pip_value * lot_size, pips, reason, i, current_sl, impulse_triggered

            # A.1 Breakeven (#28B: Smart)
            if pip_profit_from_entry >= adaptive_breakeven_pips and not breakeven_moved:
                if direction == "BUY":
                    current_sl = entry_price + be_lock_distance
                else:
                    current_sl = entry_price - be_lock_distance
                breakeven_moved = True

            # A.2 Trailing SL (#33: impulse-aware)
            if pip_profit_from_entry >= adaptive_trail_start_pips:
                # #33: Use tighter trail during impulse bars
                if is_impulse_bar and self.impulse_enabled:
                    active_trail_step = impulse_trail_step_pips
                else:
                    active_trail_step = adaptive_trail_step_pips

                trail_distance = active_trail_step * 0.1
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
                    return current_profit, current_pips, ExitReason.PEAK_PROTECT, i, close, impulse_triggered

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
                    return current_profit, current_pips, ExitReason.MARKET_SIGNAL, i, close, impulse_triggered
                if urgency >= 7 and current_profit > 0:
                    return current_profit, current_pips, ExitReason.MARKET_SIGNAL, i, close, impulse_triggered

            # A.5 Weekend close
            if self._is_near_weekend_close(current_time):
                if current_profit > 0 or current_profit > -10:
                    return current_profit, current_pips, ExitReason.WEEKEND_CLOSE, i, close, impulse_triggered

            # B.1 Smart TP
            if current_profit >= 15:
                if current_profit >= 40:
                    return current_profit, current_pips, ExitReason.SMART_TP, i, close, impulse_triggered
                if current_profit >= 25 and momentum < -30:
                    return current_profit, current_pips, ExitReason.SMART_TP, i, close, impulse_triggered
                if peak_profit > 30 and current_profit < peak_profit * 0.6:
                    return current_profit, current_pips, ExitReason.PEAK_PROTECT, i, close, impulse_triggered
                if current_profit >= 20:
                    progress = (current_profit / target_tp_profit) * 100 if target_tp_profit > 0 else 0
                    progress_score = min(40, max(0, progress * 0.4))
                    momentum_score = ((momentum + 100) / 200) * 30
                    time_penalty = min(10, bars_since_entry / 4 * 2)
                    tp_probability = progress_score + momentum_score + 10 - time_penalty
                    if tp_probability < 25:
                        return current_profit, current_pips, ExitReason.SMART_TP, i, close, impulse_triggered

            # B.2 Smart Early Exit
            if 5 <= current_profit < 15:
                if momentum < -50 and cached_ml_confidence >= 0.65:
                    is_reversal = (direction == "BUY" and cached_ml_signal == "SELL") or (direction == "SELL" and cached_ml_signal == "BUY")
                    if is_reversal:
                        return current_profit, current_pips, ExitReason.EARLY_EXIT, i, close, impulse_triggered

            # B.3 Early cut
            if current_profit < 0:
                loss_percent_of_max = abs(current_profit) / self.max_loss_per_trade * 100
                if momentum < self.early_cut_momentum and loss_percent_of_max >= self.early_cut_loss_pct:
                    return current_profit, current_pips, ExitReason.EARLY_CUT, i, close, impulse_triggered

            # B.4 Trend Reversal
            is_ml_reversal = False
            if (direction == "BUY" and cached_ml_signal == "SELL" and cached_ml_confidence >= 0.75) or \
               (direction == "SELL" and cached_ml_signal == "BUY" and cached_ml_confidence >= 0.75):
                is_ml_reversal = True
                reversal_warnings += 1
            loss_moderate = abs(current_profit) > (self.max_loss_per_trade * 0.4)
            if is_ml_reversal and current_profit < -8 and loss_moderate:
                return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close, impulse_triggered
            if reversal_warnings >= 3 and current_profit < -10:
                return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close, impulse_triggered

            # B.5 Max loss
            if current_profit <= -(self.max_loss_per_trade * 0.50):
                htg = self._hours_to_golden(current_time)
                if htg <= 1 and htg > 0 and momentum > -40:
                    pass
                else:
                    return current_profit, current_pips, ExitReason.MAX_LOSS, i, close, impulse_triggered

            # B.6 Stall
            if len(profit_history) >= 10:
                recent_range = max(profit_history[-10:]) - min(profit_history[-10:])
                if recent_range < 3 and current_profit < -15:
                    stall_count += 1
                    if stall_count >= 5:
                        return current_profit, current_pips, ExitReason.STALL, i, close, impulse_triggered

            # B.7 Daily loss limit
            potential_daily_loss = daily_loss_so_far + abs(min(0, current_profit))
            if potential_daily_loss >= self.max_daily_loss_usd:
                return current_profit, current_pips, ExitReason.DAILY_LIMIT, i, close, impulse_triggered

            # C) Time-based
            if bars_since_entry >= 16 and current_profit < 5 and not profit_growing:
                if current_profit >= 0 or current_profit > -15:
                    return current_profit, current_pips, ExitReason.TIMEOUT, i, close, impulse_triggered
            if bars_since_entry >= 24 and (current_profit < 10 or not profit_growing):
                return current_profit, current_pips, ExitReason.TIMEOUT, i, close, impulse_triggered
            if bars_since_entry >= 32:
                return current_profit, current_pips, ExitReason.TIMEOUT, i, close, impulse_triggered

            # C.2 ATR trend reversal
            if bars_since_entry > 10:
                recent_closes = closes[i-5:i+1]
                mom = recent_closes[-1] - recent_closes[0]
                if (direction == "BUY" and mom < -reversal_momentum_threshold) or \
                   (direction == "SELL" and mom > reversal_momentum_threshold):
                    if current_profit < -min_loss_for_reversal_exit:
                        return current_profit, current_pips, ExitReason.TREND_REVERSAL, i, close, impulse_triggered

        final_idx = min(entry_idx + max_bars - 1, len(df) - 1)
        final_price = closes[final_idx]
        pips = ((final_price - entry_price) if direction == "BUY" else (entry_price - final_price)) / 0.1
        return pips * pip_value * lot_size, pips, ExitReason.TIMEOUT, final_idx, final_price, impulse_triggered

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

        times_m15 = df_m15["time"].to_list()
        times_h1 = df_h1["time"].to_list() if df_h1 is not None else []

        start_idx = next((i for i, t in enumerate(times_m15) if t >= start_date), 100) if start_date else 100
        end_idx = next((i for i, t in enumerate(times_m15) if t > end_date), len(df_m15) - 100) if end_date else len(df_m15) - 100

        last_trade_idx = -self.trade_cooldown_bars * 2

        print(f"  #33 impulse: {self.impulse_enabled}, threshold: {self.impulse_threshold_mult}x ATR, "
              f"tight trail: {self.impulse_trail_mult}x ATR, trail_start: {self.trail_start_mult}x, trail_step: {self.trail_step_mult}x")
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

            # Strict H1 filter
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

            profit, pips, exit_reason, exit_idx, exit_price, impulse_hit = self._simulate_trade_exit(
                df=df_m15, entry_idx=i, direction=smc_signal.signal_type,
                entry_price=entry_price, take_profit=take_profit_price,
                stop_loss=stop_loss_price, lot_size=lot_size,
                daily_loss_so_far=daily_loss, feature_cols=feature_cols,
            )

            self._ticket_counter += 1
            result = TradeResult.WIN if profit > 0 else (TradeResult.LOSS if profit < 0 else TradeResult.BREAKEVEN)

            if impulse_hit:
                stats.impulse_trail_count += 1

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
                impulse_triggered=impulse_hit,
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


# ─── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("XAUBOT AI — #33 Impulse Trail + Trail Tuning")
    print("Base: #31B (H1 Price vs EMA20) | Modified: Impulse-aware trailing SL")
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

    print("Calculating H1 indicators...")
    df_h1 = features.calculate_all(df_h1, include_ml_features=False)
    print("  All indicators calculated")

    baseline_31b_pnl = 2806.56

    # ═══ CONFIGS ═══
    configs = [
        ("A: Impulse 1.5x>1.0x trail", {
            "impulse_enabled": True,
            "impulse_threshold_mult": 1.5,
            "impulse_trail_mult": 1.0,
        }),
        ("B: Impulse 1.5x>1.5x trail", {
            "impulse_enabled": True,
            "impulse_threshold_mult": 1.5,
            "impulse_trail_mult": 1.5,
        }),
        ("C: Impulse 2.0x>1.0x trail", {
            "impulse_enabled": True,
            "impulse_threshold_mult": 2.0,
            "impulse_trail_mult": 1.0,
        }),
        ("D: A + wider start 5.0x", {
            "impulse_enabled": True,
            "impulse_threshold_mult": 1.5,
            "impulse_trail_mult": 1.0,
            "trail_start_mult": 5.0,
        }),
        ("E: A + start 5.0x step 3.5x", {
            "impulse_enabled": True,
            "impulse_threshold_mult": 1.5,
            "impulse_trail_mult": 1.0,
            "trail_start_mult": 5.0,
            "trail_step_mult": 3.5,
        }),
    ]

    all_results = []

    for cfg_name, cfg_params in configs:
        print(f"\n{'=' * 60}")
        print(f"  Config: {cfg_name}")

        bt = ImpulseTrailBacktest(**cfg_params)
        stats = bt.run(df_m15=df_m15, df_h1=df_h1, start_date=start_date, end_date=end_date, initial_capital=5000.0)
        net_pnl = stats.total_profit - stats.total_loss
        diff = net_pnl - baseline_31b_pnl

        buy_trades = [t for t in stats.trades if t.direction == "BUY"]
        sell_trades = [t for t in stats.trades if t.direction == "SELL"]
        buy_wins = sum(1 for t in buy_trades if t.result == TradeResult.WIN)
        sell_wins = sum(1 for t in sell_trades if t.result == TradeResult.WIN)
        buy_wr = buy_wins / len(buy_trades) * 100 if buy_trades else 0
        sell_wr = sell_wins / len(sell_trades) * 100 if sell_trades else 0
        buy_pnl = sum(t.profit_usd for t in buy_trades)
        sell_pnl = sum(t.profit_usd for t in sell_trades)

        # Impulse stats
        impulse_trades = [t for t in stats.trades if t.impulse_triggered]
        impulse_wins = sum(1 for t in impulse_trades if t.result == TradeResult.WIN)
        impulse_wr = impulse_wins / len(impulse_trades) * 100 if impulse_trades else 0
        impulse_pnl = sum(t.profit_usd for t in impulse_trades)

        print(f"\n  [{cfg_name}] Results:")
        print(f"    Trades: {stats.total_trades} | WR: {stats.win_rate:.1f}%")
        print(f"    Net PnL: ${net_pnl:,.2f} | PF: {stats.profit_factor:.2f}")
        print(f"    Max DD: {stats.max_drawdown:.1f}% | Sharpe: {stats.sharpe_ratio:.2f}")
        print(f"    BUY: {len(buy_trades)}, {buy_wr:.1f}% WR, ${buy_pnl:,.2f}")
        print(f"    SELL: {len(sell_trades)}, {sell_wr:.1f}% WR, ${sell_pnl:,.2f}")
        print(f"    Impulse trades: {len(impulse_trades)} ({impulse_wr:.1f}% WR, ${impulse_pnl:,.2f})")
        print(f"    vs #31B: ${diff:+,.2f}")

        all_results.append((cfg_name, stats, net_pnl, diff))

    # ═══ FINAL SUMMARY ═══
    print(f"\n{'=' * 70}")
    print("#33 IMPULSE TRAIL + TRAIL TUNING — ALL CONFIGURATIONS")
    print("=" * 70)

    print(f"\n  {'Config':<28} {'Trades':>6} {'WR':>6} {'Net PnL':>10} {'DD':>6} {'Sharpe':>7} {'PF':>5} {'Impulse':>8} {'vs #31B':>10}")
    print(f"  {'-' * 95}")
    print(f"  {'#24B (base)':<28} {'739':>6} {'80.4%':>6} {'$2,235':>10} {'3.4%':>6} {'2.87':>7} {'1.77':>5} {'—':>8} {'—':>10}")
    print(f"  {'#28B (smart BE)':<28} {'741':>6} {'79.8%':>6} {'$2,464':>10} {'3.5%':>6} {'3.23':>7} {'1.83':>5} {'—':>8} {'—':>10}")
    print(f"  {'#31B (H1 filter)':<28} {'625':>6} {'81.8%':>6} {'$2,807':>10} {'2.5%':>6} {'3.97':>7} {'2.19':>5} {'—':>8} {'—':>10}")
    for cfg_name, stats, net_pnl, diff in all_results:
        impulse_count = stats.impulse_trail_count
        print(f"  {cfg_name:<28} {stats.total_trades:>6} {stats.win_rate:>5.1f}% ${net_pnl:>9,.2f} {stats.max_drawdown:>5.1f}% {stats.sharpe_ratio:>7.2f} {stats.profit_factor:>5.2f} {impulse_count:>8} ${diff:>+9,.2f}")

    best_pnl = -999999
    best_name = ""
    best_stats = None
    for entry in all_results:
        if entry[2] > best_pnl:
            best_pnl = entry[2]
            best_name = entry[0]
            best_stats = entry[1]

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

    # Impulse analysis for best config
    print(f"\n  Impulse Analysis (best config):")
    impulse_trades = [t for t in best_stats.trades if t.impulse_triggered]
    non_impulse_trades = [t for t in best_stats.trades if not t.impulse_triggered]
    if impulse_trades:
        imp_wins = sum(1 for t in impulse_trades if t.result == TradeResult.WIN)
        imp_pnl = sum(t.profit_usd for t in impulse_trades)
        imp_avg = imp_pnl / len(impulse_trades)
        print(f"    Impulse trades: {len(impulse_trades)} ({imp_wins}/{len(impulse_trades)} wins = {imp_wins/len(impulse_trades)*100:.1f}% WR)")
        print(f"    Impulse PnL: ${imp_pnl:,.2f} (avg ${imp_avg:,.2f}/trade)")
    if non_impulse_trades:
        ni_wins = sum(1 for t in non_impulse_trades if t.result == TradeResult.WIN)
        ni_pnl = sum(t.profit_usd for t in non_impulse_trades)
        ni_avg = ni_pnl / len(non_impulse_trades)
        print(f"    Non-impulse trades: {len(non_impulse_trades)} ({ni_wins}/{len(non_impulse_trades)} wins = {ni_wins/len(non_impulse_trades)*100:.1f}% WR)")
        print(f"    Non-impulse PnL: ${ni_pnl:,.2f} (avg ${ni_avg:,.2f}/trade)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "33_impulse_trail_results")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"impulse_trail_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write(f"#33 Impulse Trail + Trail Tuning Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Base: #31B (625 trades, 81.8% WR, $2,807)\n\n")
        for cfg_name, stats, net_pnl, diff in all_results:
            imp_count = stats.impulse_trail_count
            f.write(f"  {cfg_name}: {stats.total_trades} trades, {stats.win_rate:.1f}% WR, "
                    f"${net_pnl:,.2f}, DD: {stats.max_drawdown:.1f}%, "
                    f"Sharpe: {stats.sharpe_ratio:.2f}, PF: {stats.profit_factor:.2f}, "
                    f"Impulse: {imp_count}, vs #31B: ${diff:+,.2f}\n")
        f.write(f"\nBest: {best_name}\n")
    print(f"  Log saved: {log_path}")

    try:
        from backtests.backtest_01_smc_only import generate_xlsx_report as gen_xlsx
        xlsx_path = os.path.join(output_dir, f"impulse_trail_{timestamp}.xlsx")
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
