"""
Dynamic Confidence System
=========================
Menyesuaikan confidence threshold berdasarkan kondisi market.

Prinsip:
- Market bagus (trending, session bagus) -> threshold lebih rendah (60%)
- Market jelek (choppy, low liquidity) -> threshold lebih tinggi (75%)
- Multiple konfirmasi -> threshold lebih rendah
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from loguru import logger


class MarketQuality(Enum):
    """Kualitas market untuk trading."""
    EXCELLENT = "excellent"   # Semua kondisi bagus
    GOOD = "good"            # Sebagian besar bagus
    MODERATE = "moderate"    # Biasa saja
    POOR = "poor"           # Kurang bagus
    AVOID = "avoid"         # Jangan trading


@dataclass
class MarketAnalysis:
    """Hasil analisis market."""
    quality: MarketQuality
    confidence_threshold: float
    reasons: list
    score: int  # 0-100


class DynamicConfidenceManager:
    """
    Manager untuk menentukan confidence threshold secara dinamis.

    Faktor yang dipertimbangkan:
    1. Session (London-NY overlap = terbaik)
    2. Regime (medium volatility = ideal)
    3. Trend clarity (trending > ranging)
    4. SMC confluence (ada OB/FVG = bonus)
    5. Spread (rendah = bagus)
    """

    def __init__(
        self,
        base_threshold: float = 0.65,
        min_threshold: float = 0.55,
        max_threshold: float = 0.80,
    ):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Track last analysis for logging
        self._last_quality = "moderate"
        self._last_score = 50
        self._last_threshold = base_threshold

    def analyze_market(
        self,
        session: str,
        regime: str,
        volatility: str,
        trend_direction: str,
        has_smc_signal: bool,
        spread: float = 0,
        ml_signal: str = "",
        ml_confidence: float = 0,
    ) -> MarketAnalysis:
        """
        Analisis kondisi market dan tentukan threshold yang tepat.

        Returns:
            MarketAnalysis dengan threshold yang disarankan
        """
        score = 50  # Start dari tengah
        reasons = []

        # 1. SESSION ANALYSIS (±20 points)
        session_lower = session.lower()
        if "overlap" in session_lower or "golden" in session_lower:
            score += 20
            reasons.append("[+] Session: London-NY Overlap (terbaik)")
        elif "london" in session_lower:
            score += 15
            reasons.append("[+] Session: London (bagus)")
        elif "new york" in session_lower or "ny" in session_lower:
            score += 10
            reasons.append("[+] Session: New York (bagus)")
        elif "asia" in session_lower or "tokyo" in session_lower:
            score += 0
            reasons.append("[!] Session: Asia (volatilitas rendah)")
        elif "closed" in session_lower or "weekend" in session_lower:
            score -= 30
            reasons.append("[X] Market closed/weekend")
        else:
            score += 5
            reasons.append(f"[i] Session: {session}")

        # 2. REGIME ANALYSIS (±15 points)
        regime_lower = regime.lower().replace(" ", "_")
        if regime_lower == "medium_volatility":
            score += 15
            reasons.append("[+] Regime: Medium volatility (ideal)")
        elif regime_lower == "low_volatility":
            score += 5
            reasons.append("[!] Regime: Low volatility (hati-hati ranging)")
        elif regime_lower == "high_volatility":
            score -= 5
            reasons.append("[!] Regime: High volatility (lot kecil!)")
        elif regime_lower == "crisis":
            score -= 25
            reasons.append("[X] Regime: Crisis (hindari trading)")

        # 3. VOLATILITY ANALYSIS (±10 points)
        vol_lower = volatility.lower()
        if vol_lower == "medium":
            score += 10
            reasons.append("[+] Volatility: Medium (ideal)")
        elif vol_lower == "low":
            score += 0
            reasons.append("[!] Volatility: Low (pergerakan kecil)")
        elif vol_lower == "high":
            score -= 5
            reasons.append("[!] Volatility: High")
        elif vol_lower == "extreme":
            score -= 10
            reasons.append("[!] Volatility: Extreme (hati-hati)")

        # 4. TREND CLARITY (±10 points)
        trend_lower = trend_direction.lower()
        if trend_lower in ["uptrend", "downtrend", "strong_up", "strong_down"]:
            score += 10
            reasons.append(f"[+] Trend: {trend_direction} (jelas)")
        elif trend_lower in ["neutral", "ranging", "sideways"]:
            score -= 5
            reasons.append("[!] Trend: Ranging/sideways")

        # 5. SMC CONFLUENCE (±10 points)
        if has_smc_signal:
            score += 10
            reasons.append("[+] SMC: Ada konfirmasi (OB/FVG/BOS)")

        # 6. ML ALIGNMENT (±5 points)
        if ml_confidence >= 0.70:
            score += 5
            reasons.append(f"[+] ML: High confidence ({ml_confidence:.0%})")
        elif ml_confidence >= 0.60:
            score += 2
            reasons.append(f"[i] ML: Moderate confidence ({ml_confidence:.0%})")

        # Clamp score
        score = max(0, min(100, score))

        # Determine quality and threshold - BALANCED SETTINGS for Active Trading
        # London/NY session should have reasonable opportunity to trade
        if score >= 80:
            quality = MarketQuality.EXCELLENT
            threshold = self.min_threshold  # 60% - kondisi terbaik
        elif score >= 65:
            quality = MarketQuality.GOOD
            threshold = 0.65  # 65% - kondisi bagus (turun dari 75%)
        elif score >= 50:
            quality = MarketQuality.MODERATE
            threshold = 0.70  # 70% - kondisi biasa (turun dari 80%)
        elif score >= 35:
            quality = MarketQuality.POOR
            threshold = 0.80  # 80% - kondisi kurang bagus (turun dari 85%)
        else:
            quality = MarketQuality.AVOID
            threshold = self.max_threshold  # 85% - hindari trading

        # Track for logging
        self._last_quality = quality.value
        self._last_score = score
        self._last_threshold = threshold

        return MarketAnalysis(
            quality=quality,
            confidence_threshold=threshold,
            reasons=reasons,
            score=score,
        )

    def get_entry_decision(
        self,
        ml_confidence: float,
        analysis: MarketAnalysis,
    ) -> Tuple[bool, str]:
        """
        Tentukan apakah boleh entry berdasarkan analisis.

        Returns:
            (can_entry, reason)
        """
        if analysis.quality == MarketQuality.AVOID:
            return False, f"Market quality: AVOID (score={analysis.score})"

        if ml_confidence >= analysis.confidence_threshold:
            return True, f"Entry OK: ML {ml_confidence:.0%} >= threshold {analysis.confidence_threshold:.0%} (score={analysis.score})"
        else:
            gap = analysis.confidence_threshold - ml_confidence
            return False, f"Wait: ML {ml_confidence:.0%} < threshold {analysis.confidence_threshold:.0%} (need +{gap:.0%})"

    def get_threshold_summary(self, analysis: MarketAnalysis) -> str:
        """Get summary string untuk logging."""
        return (
            f"Market: {analysis.quality.value.upper()} "
            f"(score={analysis.score}) -> "
            f"Threshold: {analysis.confidence_threshold:.0%}"
        )


def create_dynamic_confidence() -> DynamicConfidenceManager:
    """Create dynamic confidence manager - BALANCED (validated by backtest)."""
    return DynamicConfidenceManager(
        base_threshold=0.70,   # Default 70% - reasonable threshold
        min_threshold=0.60,    # Kondisi terbaik bisa turun ke 60%
        max_threshold=0.85,    # Kondisi jelek naik ke 85%
    )


if __name__ == "__main__":
    # Test
    manager = create_dynamic_confidence()

    print("=== Test 1: Kondisi Ideal ===")
    analysis = manager.analyze_market(
        session="London-NY Overlap (GOLDEN)",
        regime="medium_volatility",
        volatility="medium",
        trend_direction="UPTREND",
        has_smc_signal=True,
        ml_signal="BUY",
        ml_confidence=0.68,
    )
    print(f"Quality: {analysis.quality.value}")
    print(f"Score: {analysis.score}")
    print(f"Threshold: {analysis.confidence_threshold:.0%}")
    print("Reasons:")
    for r in analysis.reasons:
        print(f"  {r}")

    can_entry, reason = manager.get_entry_decision(0.68, analysis)
    print(f"\nCan Entry (68%): {can_entry} - {reason}")

    print("\n=== Test 2: Kondisi Jelek ===")
    analysis2 = manager.analyze_market(
        session="Asia (low liquidity)",
        regime="low_volatility",
        volatility="low",
        trend_direction="RANGING",
        has_smc_signal=False,
        ml_signal="BUY",
        ml_confidence=0.62,
    )
    print(f"Quality: {analysis2.quality.value}")
    print(f"Score: {analysis2.score}")
    print(f"Threshold: {analysis2.confidence_threshold:.0%}")

    can_entry2, reason2 = manager.get_entry_decision(0.62, analysis2)
    print(f"\nCan Entry (62%): {can_entry2} - {reason2}")
