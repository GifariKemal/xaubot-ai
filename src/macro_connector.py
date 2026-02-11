"""
Macro Data Connector for Gold Trading
======================================
Fetches macro-economic data that influences XAUUSD (Gold).

Key Gold Drivers:
1. US Dollar Index (DXY) - 80% inverse correlation with gold
2. Real Yields (10Y TIPS) - Opportunity cost of holding gold
3. VIX (Fear Index) - Risk-on/risk-off sentiment
4. Fed Funds Rate - Interest rate expectations
5. Geopolitical Risk Index - Safe-haven demand

Data Sources:
- Yahoo Finance (DXY, VIX)
- FRED API (Fed Funds, Real Yields, CPI)
- Free APIs (no paid subscriptions required)

Author: AI Assistant (Phase 9 - FinceptTerminal Enhancement)
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import os


class MacroDataConnector:
    """
    Fetches macro-economic data for gold trading decisions.

    Provides real-time macro context to enhance entry/exit filters.
    """

    def __init__(self):
        """Initialize macro data connector."""
        # FRED API key (optional, has free tier)
        self.fred_api_key = os.getenv("FRED_API_KEY", "")

        # Cache macro data (update every 4 hours)
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 4 * 3600  # 4 hours

    async def _get_cached_or_fetch(
        self,
        key: str,
        fetch_func
    ) -> Optional[float]:
        """
        Get cached value or fetch new data.

        Args:
            key: Cache key
            fetch_func: Async function to fetch data

        Returns:
            Cached or fresh data
        """
        now = datetime.now().timestamp()

        # Return cached if valid
        if key in self.cache and key in self.cache_expiry:
            if now < self.cache_expiry[key]:
                return self.cache[key]

        # Fetch fresh data
        try:
            value = await fetch_func()
            if value is not None:
                self.cache[key] = value
                self.cache_expiry[key] = now + self.cache_duration
            return value
        except Exception as e:
            logger.warning(f"Failed to fetch {key}: {e}")
            # Return cached even if expired (stale data better than none)
            return self.cache.get(key)

    async def get_dxy_index(self) -> Optional[float]:
        """
        Get US Dollar Index (DXY).

        DXY measures USD strength vs basket of currencies.
        Gold has ~80% inverse correlation with DXY.

        Returns:
            DXY current value (~100-110 typical range)
        """
        async def fetch():
            # Use Yahoo Finance API (free)
            url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
            params = {"interval": "1d", "range": "1d"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                        return float(quote)
            return None

        return await self._get_cached_or_fetch("dxy", fetch)

    async def get_vix_index(self) -> Optional[float]:
        """
        Get VIX (CBOE Volatility Index).

        VIX is the "fear gauge" - measures S&P 500 implied volatility.
        High VIX = risk-off = gold bullish (safe haven)
        Low VIX = risk-on = gold neutral/bearish

        Returns:
            VIX current value (~10-30 typical, >40 = crisis)
        """
        async def fetch():
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
            params = {"interval": "1d", "range": "1d"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                        return float(quote)
            return None

        return await self._get_cached_or_fetch("vix", fetch)

    async def get_real_yields(self) -> Optional[float]:
        """
        Get 10-Year Real Yields (TIPS).

        Real yields = opportunity cost of holding gold (non-yielding asset).
        High real yields = bearish for gold
        Low/negative real yields = bullish for gold

        Returns:
            10Y TIPS yield (% per year, can be negative)
        """
        async def fetch():
            if not self.fred_api_key:
                logger.debug("FRED_API_KEY not set, skipping real yields")
                return None

            # FRED series: DFII10 (10-Year Treasury Inflation-Indexed Security)
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "DFII10",
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get("observations", [])
                        if observations:
                            value = observations[0].get("value")
                            if value != ".":
                                return float(value)
            return None

        return await self._get_cached_or_fetch("real_yields", fetch)

    async def get_fed_funds_rate(self) -> Optional[float]:
        """
        Get Federal Funds Effective Rate.

        Fed rate = cost of money = major gold driver.
        Higher rates = higher opportunity cost = bearish gold
        Lower rates = cheaper money = bullish gold

        Returns:
            Fed Funds rate (% per year)
        """
        async def fetch():
            if not self.fred_api_key:
                logger.debug("FRED_API_KEY not set, skipping fed funds")
                return None

            # FRED series: FEDFUNDS
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get("observations", [])
                        if observations:
                            value = observations[0].get("value")
                            if value != ".":
                                return float(value)
            return None

        return await self._get_cached_or_fetch("fed_funds", fetch)

    async def get_gold_etf_flows(self) -> Optional[float]:
        """
        Get GLD ETF holdings (proxy for institutional gold demand).

        GLD = SPDR Gold Trust, largest gold ETF.
        Rising holdings = institutional accumulation = bullish
        Falling holdings = institutional distribution = bearish

        Returns:
            GLD holdings in tonnes (approximate)
        """
        async def fetch():
            # GLD reports holdings on their website
            # For now, return None (requires web scraping or paid API)
            # TODO: Implement GLD holdings scraper or use Polygon.io
            return None

        return await self._get_cached_or_fetch("gld_flows", fetch)

    async def calculate_macro_score(self) -> Tuple[float, Dict]:
        """
        Calculate composite macro score for gold (0-1).

        Combines all macro factors into single score:
        - 0.0-0.3: Bearish macro environment
        - 0.3-0.7: Neutral
        - 0.7-1.0: Bullish macro environment

        Returns:
            (macro_score, components_dict)
        """
        # Fetch all macro data concurrently
        results = await asyncio.gather(
            self.get_dxy_index(),
            self.get_vix_index(),
            self.get_real_yields(),
            self.get_fed_funds_rate(),
            return_exceptions=True
        )

        dxy, vix, real_yields, fed_funds = results

        components = {
            "dxy": dxy,
            "vix": vix,
            "real_yields": real_yields,
            "fed_funds": fed_funds,
        }

        # Calculate individual scores (0-1)
        scores = []
        weights = []

        # DXY: Inverse correlation (lower DXY = higher gold)
        if dxy is not None:
            # DXY range ~95-115, normalize
            # Score: 1.0 if DXY=95, 0.0 if DXY=115
            dxy_score = (115 - dxy) / 20  # Inverted
            dxy_score = max(0, min(1, dxy_score))
            scores.append(dxy_score)
            weights.append(0.35)  # 35% weight (strongest factor)

        # VIX: Direct correlation (higher VIX = risk-off = gold bullish)
        if vix is not None:
            # VIX range ~10-50, normalize
            # Score: 0.0 if VIX=10, 1.0 if VIX=40+
            vix_score = (vix - 10) / 30
            vix_score = max(0, min(1, vix_score))
            scores.append(vix_score)
            weights.append(0.25)  # 25% weight

        # Real Yields: Inverse correlation (lower yields = gold bullish)
        if real_yields is not None:
            # Real yields range ~-1% to 3%, normalize
            # Score: 1.0 if yields=-1%, 0.0 if yields=3%
            yields_score = (3 - real_yields) / 4  # Inverted
            yields_score = max(0, min(1, yields_score))
            scores.append(yields_score)
            weights.append(0.30)  # 30% weight

        # Fed Funds: Inverse correlation (lower rates = gold bullish)
        if fed_funds is not None:
            # Fed Funds range ~0-6%, normalize
            # Score: 1.0 if rate=0%, 0.0 if rate=6%
            fed_score = (6 - fed_funds) / 6  # Inverted
            fed_score = max(0, min(1, fed_score))
            scores.append(fed_score)
            weights.append(0.10)  # 10% weight

        # Calculate weighted average
        if len(scores) == 0:
            logger.warning("No macro data available, returning neutral score")
            return 0.5, components

        total_weight = sum(weights[:len(scores)])
        weighted_sum = sum(s * w for s, w in zip(scores, weights[:len(scores)]))
        macro_score = weighted_sum / total_weight

        components["macro_score"] = macro_score
        components["dxy_score"] = scores[0] if len(scores) > 0 else None
        components["vix_score"] = scores[1] if len(scores) > 1 else None
        components["yields_score"] = scores[2] if len(scores) > 2 else None
        components["fed_score"] = scores[3] if len(scores) > 3 else None

        return macro_score, components

    async def get_macro_context(self) -> str:
        """
        Get human-readable macro context summary.

        Returns:
            Formatted string with macro analysis
        """
        macro_score, components = await self.calculate_macro_score()

        # Determine regime
        if macro_score < 0.3:
            regime = "[WARNING] BEARISH"
            color = "red"
        elif macro_score < 0.7:
            regime = "⚖️ NEUTRAL"
            color = "yellow"
        else:
            regime = "✅ BULLISH"
            color = "green"

        # Format components
        dxy = components.get("dxy", "N/A")
        vix = components.get("vix", "N/A")
        yields = components.get("real_yields", "N/A")
        fed = components.get("fed_funds", "N/A")

        dxy_str = f"{dxy:.2f}" if isinstance(dxy, float) else dxy
        vix_str = f"{vix:.1f}" if isinstance(vix, float) else vix
        yields_str = f"{yields:.2f}%" if isinstance(yields, float) else yields
        fed_str = f"{fed:.2f}%" if isinstance(fed, float) else fed

        summary = f"""
🌍 MACRO CONTEXT FOR GOLD
{'=' * 40}
Macro Score: {macro_score:.2f} {regime}

📊 Components:
  DXY (USD Index): {dxy_str}
  VIX (Fear Gauge): {vix_str}
  Real Yields: {yields_str}
  Fed Funds Rate: {fed_str}

💡 Interpretation:
  • DXY ↓ = Gold ↑ (inverse correlation)
  • VIX ↑ = Gold ↑ (risk-off flows)
  • Yields ↓ = Gold ↑ (lower opportunity cost)
  • Fed Rate ↓ = Gold ↑ (cheaper money)
{'=' * 40}
"""
        return summary


# Convenience function
async def get_quick_macro_score() -> float:
    """Quick macro score calculation."""
    connector = MacroDataConnector()
    score, _ = await connector.calculate_macro_score()
    return score


if __name__ == "__main__":
    # Example usage
    async def test():
        connector = MacroDataConnector()

        # Test individual metrics
        dxy = await connector.get_dxy_index()
        vix = await connector.get_vix_index()
        print(f"DXY: {dxy}")
        print(f"VIX: {vix}")

        # Test macro score
        score, components = await connector.calculate_macro_score()
        print(f"\nMacro Score: {score:.2f}")
        print(f"Components: {components}")

        # Test summary
        summary = await connector.get_macro_context()
        print(summary)

    asyncio.run(test())
