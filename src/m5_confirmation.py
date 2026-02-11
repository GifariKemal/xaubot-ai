"""
M5 Confirmation System
======================
Fast confirmation using M5 timeframe for M15 trading.

Philosophy:
- M5 detects early trend changes (30-60 min faster than H1)
- SMC analysis on M5 shows micro-structures
- Prevents lagging H1 bias from blocking good M15 signals

Author: Claude Opus 4.6
Date: 2026-02-09
"""

import polars as pl
from typing import Literal, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class M5ConfirmationSignal:
    """M5 confirmation signal result."""
    signal: Literal["BUY", "SELL", "NEUTRAL"]
    confidence: float  # 0.0 to 1.0
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    smc_alignment: bool  # True if SMC structures align
    momentum_score: float  # -1.0 to +1.0
    details: Dict[str, any]


class M5ConfirmationAnalyzer:
    """
    Analyze M5 timeframe for fast confirmation of M15 signals.

    Uses:
    1. SMC structures (Order Blocks, FVG, BOS)
    2. EMA trend (EMA9 vs EMA21)
    3. Momentum (RSI, MACD)
    4. Candle structure
    """

    def __init__(self, smc_analyzer, feature_engineer):
        """
        Initialize M5 confirmation analyzer.

        Args:
            smc_analyzer: SMC analyzer instance (for M5 data)
            feature_engineer: Feature engineer instance
        """
        self.smc = smc_analyzer
        self.features = feature_engineer

    def analyze(
        self,
        df_m5: pl.DataFrame,
        m15_signal: str,
        m15_confidence: float
    ) -> M5ConfirmationSignal:
        """
        Analyze M5 timeframe to confirm M15 signal.

        Args:
            df_m5: M5 OHLCV data (at least 100 candles)
            m15_signal: M15 signal type ("BUY" or "SELL")
            m15_confidence: M15 signal confidence (0-1)

        Returns:
            M5ConfirmationSignal with recommendation
        """
        try:
            if len(df_m5) < 100:
                logger.warning(f"M5 data insufficient: {len(df_m5)} candles")
                return self._neutral_signal("Insufficient M5 data")

            # Calculate features and SMC on M5
            df_m5 = self.features.calculate_all(df_m5, include_ml_features=False)
            df_m5 = self.smc.calculate_all(df_m5)

            # Get latest values
            last = df_m5.row(-1, named=True)

            # === 1. EMA Trend Analysis ===
            ema_9 = last["ema_9"]
            ema_21 = last["ema_21"]
            price = last["close"]

            if ema_9 > ema_21 and price > ema_9:
                ema_trend = "BULLISH"
                ema_score = 1.0
            elif ema_9 < ema_21 and price < ema_9:
                ema_trend = "BEARISH"
                ema_score = -1.0
            else:
                ema_trend = "NEUTRAL"
                ema_score = 0.0

            # === 2. SMC Structure Analysis ===
            smc_bullish_score = 0.0
            smc_bearish_score = 0.0

            # Check for bullish order blocks
            if last.get("bullish_ob", False):
                smc_bullish_score += 0.3

            # Check for bearish order blocks
            if last.get("bearish_ob", False):
                smc_bearish_score += 0.3

            # Check for FVG (bullish)
            if last.get("bullish_fvg", False):
                smc_bullish_score += 0.2

            # Check for FVG (bearish)
            if last.get("bearish_fvg", False):
                smc_bearish_score += 0.2

            # Check for BOS (bullish)
            if last.get("bos_bullish", False):
                smc_bullish_score += 0.3

            # Check for BOS (bearish)
            if last.get("bos_bearish", False):
                smc_bearish_score += 0.3

            # Check for CHoCH
            if last.get("choch_bullish", False):
                smc_bullish_score += 0.2

            if last.get("choch_bearish", False):
                smc_bearish_score += 0.2

            smc_net_score = smc_bullish_score - smc_bearish_score

            # === 3. Momentum Indicators ===
            rsi = last.get("rsi", 50)
            macd_hist = last.get("macd_histogram", 0)

            # RSI momentum
            if rsi > 55:
                rsi_score = 0.5
            elif rsi < 45:
                rsi_score = -0.5
            else:
                rsi_score = 0.0

            # MACD momentum
            if macd_hist > 0:
                macd_score = 0.5
            elif macd_hist < 0:
                macd_score = -0.5
            else:
                macd_score = 0.0

            # === 4. Candle Structure (last 5 candles) ===
            last_5 = df_m5.tail(5)
            bullish_candles = sum(
                1 for row in last_5.iter_rows(named=True)
                if row["close"] > row["open"]
            )

            if bullish_candles >= 4:
                candle_score = 0.8
            elif bullish_candles >= 3:
                candle_score = 0.4
            elif bullish_candles <= 1:
                candle_score = -0.8
            elif bullish_candles <= 2:
                candle_score = -0.4
            else:
                candle_score = 0.0

            # === 5. Combined Momentum Score ===
            momentum_score = (
                ema_score * 0.35 +
                smc_net_score * 0.30 +
                rsi_score * 0.15 +
                macd_score * 0.10 +
                candle_score * 0.10
            )

            # === 6. Determine M5 Trend ===
            if momentum_score > 0.3:
                m5_trend = "BULLISH"
            elif momentum_score < -0.3:
                m5_trend = "BEARISH"
            else:
                m5_trend = "NEUTRAL"

            # === 7. Check Alignment with M15 ===
            if m15_signal == "BUY":
                if m5_trend == "BULLISH":
                    # Perfect alignment
                    signal = "BUY"
                    confidence = min(0.9, m15_confidence + 0.15)
                    smc_alignment = True
                elif m5_trend == "NEUTRAL":
                    # M5 neutral, allow M15
                    signal = "BUY"
                    confidence = m15_confidence
                    smc_alignment = False
                else:
                    # M5 conflicts (bearish)
                    signal = "NEUTRAL"
                    confidence = 0.3
                    smc_alignment = False

            elif m15_signal == "SELL":
                if m5_trend == "BEARISH":
                    # Perfect alignment
                    signal = "SELL"
                    confidence = min(0.9, m15_confidence + 0.15)
                    smc_alignment = True
                elif m5_trend == "NEUTRAL":
                    # M5 neutral, allow M15
                    signal = "SELL"
                    confidence = m15_confidence
                    smc_alignment = False
                else:
                    # M5 conflicts (bullish)
                    signal = "NEUTRAL"
                    confidence = 0.3
                    smc_alignment = False
            else:
                signal = "NEUTRAL"
                confidence = 0.5
                smc_alignment = False

            # === 8. Build Result ===
            details = {
                "ema_trend": ema_trend,
                "ema_score": ema_score,
                "smc_bullish": smc_bullish_score,
                "smc_bearish": smc_bearish_score,
                "smc_net": smc_net_score,
                "rsi": rsi,
                "rsi_score": rsi_score,
                "macd_histogram": macd_hist,
                "macd_score": macd_score,
                "bullish_candles": bullish_candles,
                "candle_score": candle_score,
                "momentum_score": momentum_score,
                "m5_trend": m5_trend,
                "m15_signal": m15_signal,
                "m15_confidence": m15_confidence,
            }

            return M5ConfirmationSignal(
                signal=signal,
                confidence=confidence,
                trend=m5_trend,
                smc_alignment=smc_alignment,
                momentum_score=momentum_score,
                details=details
            )

        except Exception as e:
            logger.error(f"M5 confirmation error: {e}")
            return self._neutral_signal(f"Error: {e}")

    def _neutral_signal(self, reason: str) -> M5ConfirmationSignal:
        """Return neutral signal with reason."""
        return M5ConfirmationSignal(
            signal="NEUTRAL",
            confidence=0.5,
            trend="NEUTRAL",
            smc_alignment=False,
            momentum_score=0.0,
            details={"reason": reason}
        )

    def get_strength(self, signal: M5ConfirmationSignal) -> str:
        """Get signal strength label."""
        conf = signal.confidence

        if conf >= 0.80:
            return "VERY_STRONG"
        elif conf >= 0.70:
            return "STRONG"
        elif conf >= 0.60:
            return "MODERATE"
        elif conf >= 0.50:
            return "WEAK"
        else:
            return "VERY_WEAK"


def get_m5_confirmation_summary(signal: M5ConfirmationSignal) -> str:
    """
    Get human-readable summary of M5 confirmation.

    Args:
        signal: M5 confirmation signal

    Returns:
        String summary for logging
    """
    details = signal.details

    summary = (
        f"M5 Confirmation: {signal.signal} "
        f"(Confidence: {signal.confidence:.0%}, Trend: {signal.trend})"
    )

    if signal.smc_alignment:
        summary += " | SMC ALIGNED ✓"

    summary += (
        f" | Momentum: {signal.momentum_score:+.2f} "
        f"| EMA: {details.get('ema_trend', 'N/A')} "
        f"| RSI: {details.get('rsi', 0):.0f}"
    )

    return summary
