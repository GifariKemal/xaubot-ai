"""
Filter Configuration Manager
=============================
Load and save entry filter enable/disable states from data/filter_config.json.

Usage:
    from src.filter_config import FilterConfigManager

    config = FilterConfigManager()
    if config.is_enabled("h1_bias"):
        # Apply H1 bias filter
        pass
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger

WIB = ZoneInfo("Asia/Jakarta")


class FilterConfigManager:
    """Manage entry filter enable/disable configuration."""

    def __init__(self, config_path: str = "data/filter_config.json"):
        """Initialize filter config manager."""
        self.config_path = Path(config_path)
        self.filters: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load filter config from JSON file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Filter config not found at {self.config_path}, using defaults (all enabled)")
                self._init_defaults()
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.filters = data.get("filters", {})
            self.metadata = data.get("metadata", {})

            enabled_count = sum(1 for f in self.filters.values() if f.get("enabled", True))
            total_count = len(self.filters)
            logger.info(f"Filter config loaded: {enabled_count}/{total_count} filters enabled")

        except Exception as e:
            logger.error(f"Failed to load filter config: {e}")
            self._init_defaults()

    def save(self) -> None:
        """Save current filter config to JSON file."""
        try:
            self.metadata["updated_at"] = datetime.now(WIB).isoformat()

            data = {
                "filters": self.filters,
                "metadata": self.metadata
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Filter config saved to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save filter config: {e}")

    def is_enabled(self, filter_key: str) -> bool:
        """
        Check if a filter is enabled.

        Args:
            filter_key: Filter key (e.g., "h1_bias", "ml_confidence")

        Returns:
            True if enabled, False if disabled or not found
        """
        if filter_key not in self.filters:
            # Unknown filter — default to enabled for safety
            return True

        return self.filters[filter_key].get("enabled", True)

    def set_enabled(self, filter_key: str, enabled: bool) -> None:
        """
        Enable or disable a filter.

        Args:
            filter_key: Filter key
            enabled: True to enable, False to disable
        """
        if filter_key not in self.filters:
            logger.warning(f"Unknown filter key: {filter_key}")
            return

        self.filters[filter_key]["enabled"] = enabled
        logger.info(f"Filter '{filter_key}' {'enabled' if enabled else 'disabled'}")

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all filter configurations."""
        return self.filters

    def update_all(self, new_config: Dict[str, bool]) -> None:
        """
        Update multiple filters at once.

        Args:
            new_config: Dict mapping filter_key -> enabled (bool)
        """
        for filter_key, enabled in new_config.items():
            if filter_key in self.filters:
                self.filters[filter_key]["enabled"] = enabled

        self.save()

        enabled_count = sum(1 for f in self.filters.values() if f.get("enabled", True))
        logger.info(f"Filter config updated: {enabled_count}/{len(self.filters)} enabled")

    def _init_defaults(self) -> None:
        """Initialize default filter config (all enabled)."""
        self.filters = {
            "flash_crash_guard": {"enabled": True, "name": "Flash Crash Guard", "description": "Block entries during extreme price movements"},
            "regime_filter": {"enabled": True, "name": "Regime Filter", "description": "Filter based on HMM regime detection"},
            "risk_check": {"enabled": True, "name": "Risk Check", "description": "Daily/total loss limits validation"},
            "session_filter": {"enabled": True, "name": "Session Filter", "description": "Trading session validation"},
            "spread_check": {"enabled": True, "name": "Spread Check", "description": "Block entries when spread too wide"},
            "h1_bias": {"enabled": True, "name": "H1 Bias Filter", "description": "Multi-timeframe H1 EMA20 alignment"},
            "ml_confidence": {"enabled": True, "name": "ML Confidence", "description": "XGBoost confidence threshold"},
            "signal_combination": {"enabled": True, "name": "Signal Combination", "description": "SMC + ML signal agreement"},
            "cooldown": {"enabled": True, "name": "Cooldown Period", "description": "Minimum time between trades"},
            "time_filter": {"enabled": True, "name": "Time Filter", "description": "Block specific hours"},
            "market_close_guard": {"enabled": True, "name": "Market Close Guard", "description": "Block near market close"}
        }
        self.metadata = {
            "updated_at": datetime.now(WIB).isoformat(),
            "version": "1.0"
        }
        self.save()
