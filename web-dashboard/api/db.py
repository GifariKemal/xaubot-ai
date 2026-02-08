"""
Database connection pool for Trading Bot API.
Uses psycopg2 with a simple connection pool.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_pool: Optional[pool.SimpleConnectionPool] = None


def get_db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "trading_db"),
        "user": os.getenv("DB_USER", "trading_bot"),
        "password": os.getenv("DB_PASSWORD", "trading_bot_2026"),
    }


def init_pool(minconn: int = 1, maxconn: int = 5):
    """Initialize connection pool."""
    global _pool
    if _pool is not None:
        return
    try:
        config = get_db_config()
        _pool = pool.SimpleConnectionPool(minconn, maxconn, **config)
        logger.info("Database pool initialized: %s@%s:%s/%s", config["user"], config["host"], config["port"], config["dbname"])
    except Exception as e:
        logger.warning("Could not initialize DB pool: %s", e)
        _pool = None


def close_pool():
    """Close all pool connections."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("Database pool closed")


@contextmanager
def get_conn():
    """Get a connection from the pool (context manager)."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


@contextmanager
def get_cursor(commit: bool = False):
    """Get a dict cursor from the pool."""
    with get_conn() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def query(sql: str, params: tuple = (), one: bool = False):
    """Execute a query and return results as list of dicts."""
    try:
        with get_cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if one:
                return dict(rows[0]) if rows else None
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("DB query error: %s", e)
        return None if one else []


def is_available() -> bool:
    """Check if DB is available."""
    return _pool is not None
