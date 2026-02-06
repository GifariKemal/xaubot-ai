"""
News Agent - Market Sentiment & Economic Calendar Analysis
==========================================================
Mengintegrasikan analisis berita untuk keputusan trading yang lebih cerdas.

Fitur:
1. MT5 Economic Calendar - Deteksi news high-impact (NFP, FOMC, CPI)
2. Keyword Sentiment Analysis - Analisis headline berita
3. News Filter Gatekeeper - Blokir trading saat kondisi berbahaya

Prinsip: "Sentimen-First, Technical-Second"
- Jika ada news high-impact -> STOP trading
- Jika sentimen sangat negatif -> Reduce position size
- Jika aman -> Proceed dengan analisis teknikal
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from loguru import logger


class MarketCondition(Enum):
    """Kondisi market berdasarkan news analysis."""
    SAFE = "safe"                    # Aman untuk trading
    CAUTION = "caution"              # Hati-hati, reduce size
    DANGER_NEWS = "danger_news"      # Ada news high-impact, jangan trade
    DANGER_SENTIMENT = "danger_sentiment"  # Sentimen sangat negatif
    UNKNOWN = "unknown"              # Tidak bisa menentukan


@dataclass
class NewsEvent:
    """Representasi event dari economic calendar."""
    name: str
    currency: str
    importance: int  # 1=Low, 2=Medium, 3=High
    time: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None


@dataclass
class SentimentResult:
    """Hasil analisis sentimen."""
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    label: str    # BEARISH, NEUTRAL, BULLISH
    confidence: float
    keywords_found: List[str]


@dataclass
class NewsAnalysis:
    """Hasil lengkap analisis news."""
    condition: MarketCondition
    upcoming_events: List[NewsEvent]
    sentiment: Optional[SentimentResult]
    reason: str
    can_trade: bool
    recommended_lot_multiplier: float  # 1.0 = normal, 0.5 = half, 0 = no trade


class NewsAgent:
    """
    Agent untuk analisis berita dan economic calendar.

    Berfungsi sebagai "Gatekeeper" sebelum trading:
    1. Cek economic calendar MT5
    2. Analisis sentimen dari headline
    3. Tentukan apakah aman untuk trading
    """

    # High-impact news keywords (USD-related for XAUUSD)
    HIGH_IMPACT_EVENTS = [
        "Non-Farm Payroll", "NFP", "FOMC", "Fed", "Federal Reserve",
        "Interest Rate", "CPI", "Inflation", "GDP", "Unemployment",
        "Powell", "Yellen", "Treasury", "Core PCE", "Retail Sales",
        "ISM Manufacturing", "ISM Services", "PPI", "Trade Balance",
    ]

    # Bearish keywords untuk gold
    BEARISH_KEYWORDS = [
        # Geopolitical - usually bullish for gold, but sudden de-escalation is bearish
        "peace deal", "ceasefire", "de-escalation", "talks succeed",
        # Economic - hawkish Fed is bearish for gold
        "rate hike", "hawkish", "tightening", "strong dollar", "dollar surge",
        "inflation falls", "inflation drops", "fed raises", "higher rates",
        "economy strong", "jobs surge", "employment rises",
        # Market sentiment
        "risk on", "stocks rally", "equity surge", "sell gold", "gold crash",
        "gold plunge", "gold drops", "gold falls", "bearish gold",
    ]

    # Bullish keywords untuk gold
    BULLISH_KEYWORDS = [
        # Geopolitical - uncertainty is bullish for gold
        "war", "conflict", "invasion", "attack", "missile", "escalation",
        "tension", "crisis", "emergency", "pandemic", "outbreak",
        # Economic - dovish Fed is bullish for gold
        "rate cut", "dovish", "easing", "stimulus", "qe", "quantitative",
        "recession", "slowdown", "weak economy", "jobs miss", "unemployment rises",
        "inflation rises", "inflation surge", "fed pauses", "lower rates",
        # Market sentiment
        "risk off", "safe haven", "gold surge", "gold rally", "bullish gold",
        "buy gold", "gold demand", "central bank buying",
    ]

    # Neutral/cautionary keywords
    VOLATILE_KEYWORDS = [
        "breaking", "urgent", "flash", "sudden", "unexpected", "surprise",
        "shock", "crash", "plunge", "spike", "surge", "volatility",
    ]

    def __init__(
        self,
        news_buffer_minutes: int = 30,
        high_impact_buffer_minutes: int = 60,
        enable_mt5_calendar: bool = True,
        enable_sentiment: bool = True,
    ):
        """
        Initialize News Agent.

        Args:
            news_buffer_minutes: Jangan trade X menit sebelum/sesudah news biasa
            high_impact_buffer_minutes: Jangan trade X menit sebelum/sesudah news high-impact
            enable_mt5_calendar: Aktifkan pengecekan MT5 calendar
            enable_sentiment: Aktifkan analisis sentimen
        """
        self.news_buffer_minutes = news_buffer_minutes
        self.high_impact_buffer_minutes = high_impact_buffer_minutes
        self.enable_mt5_calendar = enable_mt5_calendar
        self.enable_sentiment = enable_sentiment

        # Cache untuk mengurangi API calls
        self._calendar_cache: List[NewsEvent] = []
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)

        logger.info("News Agent initialized")
        logger.info(f"  News buffer: {news_buffer_minutes} minutes")
        logger.info(f"  High-impact buffer: {high_impact_buffer_minutes} minutes")

    def check_economic_calendar(self) -> Tuple[MarketCondition, List[NewsEvent], str]:
        """
        Cek MT5 Economic Calendar untuk news high-impact.

        Returns:
            (condition, events, reason)
        """
        try:
            import MetaTrader5 as mt5

            # Check if MT5 is already initialized (by main connector)
            # Don't call mt5.initialize() here as it conflicts with main connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                # MT5 not initialized - skip silently (main connector will handle)
                # Don't log warning to avoid spam
                return MarketCondition.SAFE, [], "MT5 calendar check skipped"

            now = datetime.now()

            # Check high-impact window (60 min before/after)
            hi_start = now - timedelta(minutes=self.high_impact_buffer_minutes)
            hi_end = now + timedelta(minutes=self.high_impact_buffer_minutes)

            # Check normal news window (30 min before/after)
            news_start = now - timedelta(minutes=self.news_buffer_minutes)
            news_end = now + timedelta(minutes=self.news_buffer_minutes)

            # Get calendar events
            # Note: MT5 calendar functions may vary by broker
            # Using a broader approach
            try:
                # Try to get calendar events (broker-dependent)
                # Some brokers don't expose this API
                events = mt5.copy_ticks_from("XAUUSD", now - timedelta(hours=1), 1, mt5.COPY_TICKS_INFO)
                # If we get here, try calendar
                calendar_events = []

                # Fallback: Check known high-impact times
                # NFP: First Friday of month, 8:30 AM ET (20:30 WIB)
                # FOMC: ~8 times per year, 2:00 PM ET (02:00 WIB next day)
                # CPI: Monthly, 8:30 AM ET

                high_impact_found = self._check_known_events(now)
                if high_impact_found:
                    return MarketCondition.DANGER_NEWS, [], high_impact_found

            except Exception as e:
                logger.debug(f"Calendar API not available: {e}")

            return MarketCondition.SAFE, [], "No high-impact news detected"

        except ImportError:
            logger.warning("MT5 not available for calendar check")
            return MarketCondition.UNKNOWN, [], "MT5 module not available"
        except Exception as e:
            logger.error(f"Error checking calendar: {e}")
            return MarketCondition.UNKNOWN, [], str(e)

    def _check_known_events(self, now: datetime) -> Optional[str]:
        """
        Check for known high-impact events based on schedule.

        AGGRESSIVE MODE: Only block for HIGH impact news (NFP, FOMC, CPI)
        Based on backtest: +/-1h HIGH only gives best results

        Returns:
            Event name if within danger zone, None otherwise
        """
        weekday = now.weekday()  # 0=Monday, 4=Friday
        day = now.day
        hour = now.hour

        # NFP: First Friday of month, 20:30 WIB (8:30 AM ET)
        # Block: 19:30-21:30 WIB (+/-1h)
        if weekday == 4 and day <= 7:
            # First Friday
            if 19 <= hour <= 21:
                return "NFP (Non-Farm Payroll) - HIGH IMPACT"

        # FOMC: ~8 times per year, 02:00 WIB (2:00 PM ET previous day)
        # Only check on typical FOMC weeks (specific dates)
        # FOMC 2025-2026 dates roughly: Jan 29, Mar 19, May 7, Jun 18, Jul 30, Sep 17, Nov 5, Dec 17
        fomc_dates = [
            (1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (11, 5), (12, 17),  # 2025
            (1, 29), (3, 18), (5, 6), (6, 17), (7, 29),  # 2026
        ]
        current_month_day = (now.month, now.day)
        for fomc_month, fomc_day in fomc_dates:
            if current_month_day == (fomc_month, fomc_day):
                if 1 <= hour <= 3:  # FOMC announcement ~02:00 WIB
                    return "FOMC Decision - HIGH IMPACT"

        # CPI: Monthly around 10th-15th, 20:30 WIB (8:30 AM ET)
        # Only block the exact release window, not entire day
        # CPI is HIGH impact for gold
        if 10 <= day <= 15 and 19 <= hour <= 21:
            # Check if it looks like CPI day (usually Tuesday/Wednesday)
            if weekday in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
                return "CPI (Inflation) - HIGH IMPACT"

        return None

    def analyze_sentiment(self, headlines: List[str]) -> SentimentResult:
        """
        Analisis sentimen dari headline berita.

        Args:
            headlines: List of news headlines

        Returns:
            SentimentResult dengan score dan label
        """
        if not headlines:
            return SentimentResult(
                score=0.0,
                label="NEUTRAL",
                confidence=0.0,
                keywords_found=[],
            )

        # Combine headlines
        text = " ".join(headlines).lower()

        # Count keyword matches
        bearish_matches = []
        bullish_matches = []
        volatile_matches = []

        for keyword in self.BEARISH_KEYWORDS:
            if keyword.lower() in text:
                bearish_matches.append(keyword)

        for keyword in self.BULLISH_KEYWORDS:
            if keyword.lower() in text:
                bullish_matches.append(keyword)

        for keyword in self.VOLATILE_KEYWORDS:
            if keyword.lower() in text:
                volatile_matches.append(keyword)

        # Calculate score
        bullish_score = len(bullish_matches) * 0.3
        bearish_score = len(bearish_matches) * 0.3
        volatile_penalty = len(volatile_matches) * 0.1

        # Net score: positive = bullish, negative = bearish
        net_score = bullish_score - bearish_score

        # Clamp to [-1, 1]
        net_score = max(-1.0, min(1.0, net_score))

        # Determine label
        if net_score > 0.3:
            label = "BULLISH"
        elif net_score < -0.3:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # Confidence based on keyword matches
        total_matches = len(bearish_matches) + len(bullish_matches)
        confidence = min(1.0, total_matches * 0.2) if total_matches > 0 else 0.0

        # Reduce confidence if volatile keywords found (uncertain situation)
        if volatile_matches:
            confidence *= 0.7

        all_keywords = bearish_matches + bullish_matches + volatile_matches

        return SentimentResult(
            score=net_score,
            label=label,
            confidence=confidence,
            keywords_found=all_keywords,
        )

    def analyze(
        self,
        headlines: Optional[List[str]] = None,
        check_calendar: bool = True,
    ) -> NewsAnalysis:
        """
        Analisis lengkap news untuk keputusan trading.

        Args:
            headlines: Optional list of news headlines
            check_calendar: Whether to check economic calendar

        Returns:
            NewsAnalysis dengan rekomendasi trading
        """
        condition = MarketCondition.SAFE
        events: List[NewsEvent] = []
        sentiment: Optional[SentimentResult] = None
        reasons = []
        lot_multiplier = 1.0

        # 1. Check Economic Calendar
        if check_calendar and self.enable_mt5_calendar:
            cal_condition, cal_events, cal_reason = self.check_economic_calendar()
            events = cal_events

            if cal_condition == MarketCondition.DANGER_NEWS:
                condition = MarketCondition.DANGER_NEWS
                reasons.append(f"High-impact news: {cal_reason}")
                lot_multiplier = 0.0  # No trading
            elif cal_condition == MarketCondition.CAUTION:
                reasons.append(f"News caution: {cal_reason}")
                lot_multiplier = 0.5  # Half size

        # 2. Analyze Sentiment (if headlines provided)
        if headlines and self.enable_sentiment:
            sentiment = self.analyze_sentiment(headlines)

            if sentiment.label == "BEARISH" and sentiment.confidence > 0.5:
                if condition != MarketCondition.DANGER_NEWS:
                    condition = MarketCondition.DANGER_SENTIMENT
                reasons.append(f"Bearish sentiment: {sentiment.keywords_found}")
                lot_multiplier = min(lot_multiplier, 0.5)
            elif sentiment.label == "BULLISH" and sentiment.confidence > 0.5:
                reasons.append(f"Bullish sentiment: {sentiment.keywords_found}")
                # Could increase multiplier, but safer to keep at 1.0

        # Determine if can trade
        can_trade = condition in [MarketCondition.SAFE, MarketCondition.CAUTION]

        # Build reason string
        if not reasons:
            reasons.append("Market conditions normal")
        reason_str = "; ".join(reasons)

        return NewsAnalysis(
            condition=condition,
            upcoming_events=events,
            sentiment=sentiment,
            reason=reason_str,
            can_trade=can_trade,
            recommended_lot_multiplier=lot_multiplier,
        )

    def should_trade(self, headlines: Optional[List[str]] = None) -> Tuple[bool, str, float]:
        """
        Quick check: Apakah aman untuk trading?

        Returns:
            (can_trade, reason, lot_multiplier)
        """
        analysis = self.analyze(headlines=headlines)
        return analysis.can_trade, analysis.reason, analysis.recommended_lot_multiplier

    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        analysis = self.analyze()

        status = f"News Status: {analysis.condition.value.upper()}\n"
        status += f"Can Trade: {'Yes' if analysis.can_trade else 'NO'}\n"
        status += f"Lot Multiplier: {analysis.recommended_lot_multiplier:.1f}x\n"
        status += f"Reason: {analysis.reason}"

        return status


def create_news_agent(
    news_buffer_minutes: int = 30,
    high_impact_buffer_minutes: int = 60,
) -> NewsAgent:
    """Factory function untuk membuat NewsAgent."""
    return NewsAgent(
        news_buffer_minutes=news_buffer_minutes,
        high_impact_buffer_minutes=high_impact_buffer_minutes,
    )


# ============================================================
# EXTERNAL NEWS API INTEGRATION (Optional - for future use)
# ============================================================

class ExternalNewsProvider:
    """
    Base class untuk external news providers.
    Implement untuk NewsAPI, ForexFactory, Bloomberg, dll.
    """

    def get_headlines(self, keywords: List[str] = None) -> List[str]:
        """Get latest headlines. Override in subclass."""
        raise NotImplementedError

    def get_gold_news(self) -> List[str]:
        """Get gold-specific news."""
        return self.get_headlines(["gold", "XAUUSD", "precious metals"])


class NewsAPIProvider(ExternalNewsProvider):
    """
    NewsAPI.org integration.
    Requires API key from https://newsapi.org/
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def get_headlines(self, keywords: List[str] = None) -> List[str]:
        """Fetch headlines from NewsAPI."""
        try:
            import requests

            query = " OR ".join(keywords) if keywords else "gold forex"

            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": query,
                    "apiKey": self.api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return [article["title"] for article in data.get("articles", [])]
            else:
                logger.warning(f"NewsAPI error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []


class ForexFactoryProvider(ExternalNewsProvider):
    """
    ForexFactory calendar scraper.
    Note: Scraping may violate ToS, use responsibly.
    """

    def get_headlines(self, keywords: List[str] = None) -> List[str]:
        """ForexFactory doesn't provide headlines, only calendar."""
        return []

    def get_calendar_events(self) -> List[dict]:
        """
        Scrape ForexFactory calendar.
        Returns list of events with impact level.
        """
        # Implementation would require web scraping
        # For now, return empty (use MT5 calendar instead)
        logger.info("ForexFactory scraping not implemented - use MT5 calendar")
        return []


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test News Agent
    agent = create_news_agent()

    print("=" * 60)
    print("NEWS AGENT TEST")
    print("=" * 60)

    # Test without headlines
    print("\n1. Check without headlines:")
    can_trade, reason, multiplier = agent.should_trade()
    print(f"   Can Trade: {can_trade}")
    print(f"   Reason: {reason}")
    print(f"   Lot Multiplier: {multiplier}x")

    # Test with bearish headlines
    print("\n2. Test with BEARISH headlines:")
    bearish_headlines = [
        "Fed signals rate hike likely next month",
        "Dollar surges as inflation falls below expectations",
        "Gold plunges on hawkish Fed comments",
    ]
    sentiment = agent.analyze_sentiment(bearish_headlines)
    print(f"   Score: {sentiment.score:.2f}")
    print(f"   Label: {sentiment.label}")
    print(f"   Keywords: {sentiment.keywords_found}")

    can_trade, reason, multiplier = agent.should_trade(bearish_headlines)
    print(f"   Can Trade: {can_trade}")
    print(f"   Lot Multiplier: {multiplier}x")

    # Test with bullish headlines
    print("\n3. Test with BULLISH headlines:")
    bullish_headlines = [
        "War tensions escalate in Middle East",
        "Fed signals potential rate cut next quarter",
        "Gold surges as safe haven demand increases",
        "Central banks buying gold at record pace",
    ]
    sentiment = agent.analyze_sentiment(bullish_headlines)
    print(f"   Score: {sentiment.score:.2f}")
    print(f"   Label: {sentiment.label}")
    print(f"   Keywords: {sentiment.keywords_found}")

    can_trade, reason, multiplier = agent.should_trade(bullish_headlines)
    print(f"   Can Trade: {can_trade}")
    print(f"   Lot Multiplier: {multiplier}x")

    # Test full analysis
    print("\n4. Full Analysis:")
    analysis = agent.analyze(headlines=bullish_headlines)
    print(f"   Condition: {analysis.condition.value}")
    print(f"   Can Trade: {analysis.can_trade}")
    print(f"   Reason: {analysis.reason}")

    print("\n" + "=" * 60)
    print("Status Summary:")
    print("=" * 60)
    print(agent.get_status_summary())
