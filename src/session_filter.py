"""
Trading Session Filter
======================
Filter trades based on market sessions and optimal trading hours.
Timezone: WIB (Waktu Indonesia Barat) - GMT+7 for Batam/Jakarta.

Optimal Trading Hours for XAUUSD:
- London-NY Overlap: 20:00 - 00:00 WIB (BEST)
- London Session: 15:00 - 00:00 WIB
- NY Session: 20:00 - 05:00 WIB

Dangerous Zones:
- Rollover/Spread Wide: 04:00 - 06:00 WIB
- Low Liquidity: 00:00 - 04:00 WIB
- Friday Close: After 23:00 WIB Friday
"""

from datetime import datetime, time, timedelta
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import pytz


class TradingSession(Enum):
    """Market trading sessions."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_TOKYO_LONDON = "tokyo_london_overlap"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OFF_HOURS = "off_hours"


@dataclass
class SessionConfig:
    """Session trading configuration."""
    name: str
    start_hour: int  # WIB
    start_minute: int
    end_hour: int    # WIB
    end_minute: int
    volatility: str  # "low", "medium", "high", "extreme"
    allow_trading: bool
    position_size_multiplier: float


class SessionFilter:
    """
    Trading session filter for optimal trading hours.

    Configured for XAUUSD aggressive trading during London/NY overlap.
    All times in WIB (GMT+7).
    """

    def __init__(
        self,
        timezone: str = "Asia/Jakarta",  # WIB
        aggressive_mode: bool = True,     # Focus on high volatility
    ):
        self.tz = pytz.timezone(timezone)
        self.aggressive_mode = aggressive_mode

        # Define trading windows (WIB)
        self.sessions = {
            # Main sessions
            TradingSession.SYDNEY: SessionConfig(
                name="Sydney",
                start_hour=6, start_minute=0,  # Start after rollover (skip 04:00-06:00)
                end_hour=13, end_minute=0,
                volatility="low",
                allow_trading=True,  # ENABLED - backtest shows $5,934 profit!
                position_size_multiplier=0.5,  # HALF lot size for safety
            ),
            TradingSession.TOKYO: SessionConfig(
                name="Tokyo",
                start_hour=7, start_minute=0,
                end_hour=16, end_minute=0,
                volatility="medium",
                allow_trading=True,
                position_size_multiplier=0.7,
            ),
            TradingSession.LONDON: SessionConfig(
                name="London",
                start_hour=15, start_minute=0,
                end_hour=23, end_minute=59,
                volatility="high",
                allow_trading=True,
                position_size_multiplier=1.0,
            ),
            TradingSession.NEW_YORK: SessionConfig(
                name="New York",
                start_hour=20, start_minute=0,
                end_hour=23, end_minute=59,  # NY continues past midnight
                volatility="extreme",
                allow_trading=True,
                position_size_multiplier=1.0,
            ),
            # Overlap sessions (BEST TIMES)
            TradingSession.OVERLAP_TOKYO_LONDON: SessionConfig(
                name="Tokyo-London Overlap",
                start_hour=15, start_minute=0,
                end_hour=16, end_minute=0,
                volatility="high",
                allow_trading=True,
                position_size_multiplier=1.0,
            ),
            TradingSession.OVERLAP_LONDON_NY: SessionConfig(
                name="London-NY Overlap (GOLDEN)",
                start_hour=20, start_minute=0,
                end_hour=23, end_minute=59,
                volatility="extreme",
                allow_trading=True,
                position_size_multiplier=1.2,  # Boost during golden hours
            ),
        }

        # Danger zones (WIB)
        self.danger_zones = [
            # Rollover - spread extremely wide
            {"name": "Rollover", "start": (4, 0), "end": (6, 0), "reason": "Spread melebar saat rollover"},
            # Low liquidity
            {"name": "Dead Zone", "start": (0, 0), "end": (4, 0), "reason": "Likuiditas rendah, spread tinggi"},
        ]

        # High impact news times to avoid (typical release times in WIB)
        self.news_blackout_times = [
            # NFP - First Friday of month
            {"event": "NFP", "hour": 19, "minute": 30, "buffer_before": 15, "buffer_after": 30},
            # Fed Interest Rate
            {"event": "FOMC", "hour": 1, "minute": 0, "buffer_before": 15, "buffer_after": 45},
            # US CPI
            {"event": "CPI", "hour": 19, "minute": 30, "buffer_before": 15, "buffer_after": 30},
        ]

    def get_current_time_wib(self) -> datetime:
        """Get current time in WIB."""
        return datetime.now(self.tz)

    def get_current_session(self) -> Tuple[TradingSession, SessionConfig]:
        """
        Get the current trading session.

        Returns highest priority session if multiple overlap.
        Priority: Overlap > London/NY > Tokyo > Sydney > Off Hours
        """
        now = self.get_current_time_wib()
        hour = now.hour
        minute = now.minute
        current_time = hour * 60 + minute

        # Check overlaps first (highest priority)
        if 20 * 60 <= current_time <= 24 * 60:  # 20:00 - 00:00
            return TradingSession.OVERLAP_LONDON_NY, self.sessions[TradingSession.OVERLAP_LONDON_NY]

        if 15 * 60 <= current_time <= 16 * 60:  # 15:00 - 16:00
            return TradingSession.OVERLAP_TOKYO_LONDON, self.sessions[TradingSession.OVERLAP_TOKYO_LONDON]

        # Check main sessions
        for session, config in self.sessions.items():
            if session in [TradingSession.OVERLAP_LONDON_NY, TradingSession.OVERLAP_TOKYO_LONDON]:
                continue

            start = config.start_hour * 60 + config.start_minute
            end = config.end_hour * 60 + config.end_minute

            if start <= current_time <= end:
                return session, config

        # Off hours
        return TradingSession.OFF_HOURS, SessionConfig(
            name="Off Hours",
            start_hour=0, start_minute=0,
            end_hour=0, end_minute=0,
            volatility="low",
            allow_trading=False,
            position_size_multiplier=0.0,
        )

    def is_danger_zone(self) -> Tuple[bool, str]:
        """Check if current time is in a danger zone."""
        now = self.get_current_time_wib()
        hour = now.hour
        minute = now.minute
        current_time = hour * 60 + minute

        for zone in self.danger_zones:
            start = zone["start"][0] * 60 + zone["start"][1]
            end = zone["end"][0] * 60 + zone["end"][1]

            if start <= current_time < end:
                return True, zone["reason"]

        return False, ""

    def is_friday_close(self) -> bool:
        """Check if approaching Friday market close."""
        now = self.get_current_time_wib()
        # Friday = 4 (Monday=0)
        if now.weekday() == 4 and now.hour >= 23:
            return True
        return False

    def is_weekend(self) -> bool:
        """Check if market is closed (weekend)."""
        now = self.get_current_time_wib()
        weekday = now.weekday()

        # Saturday full day
        if weekday == 5:
            return True
        # Sunday until 04:00 WIB Monday
        if weekday == 6:
            return True
        # Saturday early morning (before market close at 05:00)
        if weekday == 5 and now.hour < 5:
            return False  # Market still open

        return False

    def can_trade(self) -> Tuple[bool, str, float]:
        """
        Check if trading is allowed right now.

        Returns:
            Tuple of (can_trade, reason, position_multiplier)
        """
        now = self.get_current_time_wib()

        # Check weekend
        if self.is_weekend():
            return False, "Market tutup (weekend)", 0.0

        # Check Friday close
        if self.is_friday_close():
            return False, "Mendekati penutupan Jumat - hindari gap weekend", 0.0

        # Check danger zones
        is_danger, danger_reason = self.is_danger_zone()
        if is_danger:
            return False, f"Zona bahaya: {danger_reason}", 0.0

        # Get current session
        session, config = self.get_current_session()

        if not config.allow_trading:
            return False, f"Trading tidak diizinkan saat {config.name}", 0.0

        # In aggressive mode, allow high volatility + Sydney (proven profitable)
        if self.aggressive_mode:
            # Sydney session is ALLOWED - backtest shows 62% WR, $5,934 profit
            if session == TradingSession.SYDNEY:
                return True, f"Trading OK - {config.name} (SAFE MODE: 0.5x lot)", config.position_size_multiplier
            # Other low volatility sessions not allowed
            if config.volatility not in ["high", "extreme"]:
                return False, f"Mode agresif: tunggu sesi {config.name} (volatilitas {config.volatility})", config.position_size_multiplier

        return True, f"Trading OK - {config.name} ({config.volatility} volatility)", config.position_size_multiplier

    def get_next_trading_window(self) -> Dict:
        """Get when the next optimal trading window starts."""
        now = self.get_current_time_wib()
        current_hour = now.hour

        # Find next London-NY overlap
        if current_hour < 20:
            # Today at 20:00
            next_window = now.replace(hour=20, minute=0, second=0, microsecond=0)
            hours_until = 20 - current_hour
        else:
            # Tomorrow at 20:00
            next_window = (now + timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
            hours_until = 24 - current_hour + 20

        return {
            "next_window": next_window.strftime("%Y-%m-%d %H:%M WIB"),
            "hours_until": hours_until,
            "session": "London-NY Overlap",
            "is_weekend": self.is_weekend(),
        }

    def get_status_report(self) -> Dict:
        """Get comprehensive trading session status."""
        now = self.get_current_time_wib()
        session, config = self.get_current_session()
        can_trade, reason, multiplier = self.can_trade()
        is_danger, danger_reason = self.is_danger_zone()

        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S WIB"),
            "day_of_week": now.strftime("%A"),
            "current_session": config.name,
            "volatility": config.volatility,
            "can_trade": can_trade,
            "reason": reason,
            "position_multiplier": multiplier,
            "is_danger_zone": is_danger,
            "danger_reason": danger_reason,
            "is_friday_close": self.is_friday_close(),
            "is_weekend": self.is_weekend(),
            "next_window": self.get_next_trading_window(),
        }


# Convenience function
def create_wib_session_filter(aggressive: bool = True) -> SessionFilter:
    """Create session filter for WIB timezone."""
    return SessionFilter(
        timezone="Asia/Jakarta",
        aggressive_mode=aggressive,
    )


if __name__ == "__main__":
    # Test session filter
    sf = create_wib_session_filter(aggressive=True)

    print("\n" + "=" * 60)
    print("TRADING SESSION STATUS")
    print("=" * 60)

    status = sf.get_status_report()
    for key, value in status.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    can_trade, reason, multiplier = sf.can_trade()
    print(f"Can Trade: {can_trade}")
    print(f"Reason: {reason}")
    print(f"Position Multiplier: {multiplier}")
