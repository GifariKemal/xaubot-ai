"""
Database Module for Trading Bot
===============================
PostgreSQL integration for trade logging, training history, and analytics.

Usage:
    from src.db import get_db, init_db, TradeRepository

    # Initialize database
    if init_db():
        db = get_db()

        # Use repository
        repo = TradeRepository(db)
        repo.insert_trade(trade_data)
"""

from .connection import DatabaseConnection, get_db, init_db
from .repository import (
    TradeRepository,
    TrainingRepository,
    SignalRepository,
    MarketSnapshotRepository,
    BotStatusRepository,
    DailySummaryRepository,
)

__all__ = [
    # Connection
    "DatabaseConnection",
    "get_db",
    "init_db",
    # Repositories
    "TradeRepository",
    "TrainingRepository",
    "SignalRepository",
    "MarketSnapshotRepository",
    "BotStatusRepository",
    "DailySummaryRepository",
]
