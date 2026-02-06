"""
Database Connection Module
==========================
PostgreSQL connection management with connection pooling.

Features:
- Connection pooling for performance
- Auto-reconnect on failure
- Context manager support
- Thread-safe operations
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading

import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import connection as PgConnection
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class DatabaseConnection:
    """
    PostgreSQL database connection manager with connection pooling.

    Thread-safe singleton pattern for efficient connection reuse.
    """

    _instance: Optional["DatabaseConnection"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """
        Initialize database connection.

        Args:
            host: Database host (default: from env)
            port: Database port (default: 5432)
            database: Database name (default: from env)
            user: Database user (default: from env)
            password: Database password (default: from env)
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        # Only initialize once (singleton)
        if self._initialized:
            return

        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME", "trading_db")
        self.user = user or os.getenv("DB_USER", "trading_bot")
        self.password = password or os.getenv("DB_PASSWORD", "trading_bot_2026")

        self.min_connections = min_connections
        self.max_connections = max_connections

        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._connected = False

        self._initialized = True

    def connect(self) -> bool:
        """
        Initialize connection pool.

        Returns:
            True if connection successful
        """
        if self._connected and self._pool:
            return True

        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
            )

            # Test connection
            conn = self._pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._pool.putconn(conn)

            self._connected = True
            logger.info(f"Database connected: {self.database}@{self.host}:{self.port}")
            return True

        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            self._connected = False
            logger.info("Database disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._pool is not None

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).

        Usage:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM trades")
        """
        if not self.is_connected:
            self.connect()

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Get a cursor directly (context manager).

        Args:
            cursor_factory: Custom cursor factory (e.g., RealDictCursor)

        Usage:
            with db.get_cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM trades")
                rows = cur.fetchall()
        """
        with self.get_connection() as conn:
            cursor_factory = cursor_factory or extras.RealDictCursor
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur

    def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Execute a query.

        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results

        Returns:
            List of dicts if fetch=True, None otherwise
        """
        with self.get_cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None

    def execute_many(
        self,
        query: str,
        params_list: List[tuple],
    ) -> int:
        """
        Execute a query multiple times.

        Args:
            query: SQL query
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.get_cursor() as cur:
            cur.executemany(query, params_list)
            return cur.rowcount

    def insert_returning(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[Dict]:
        """
        Execute INSERT ... RETURNING and return the inserted row.

        Args:
            query: INSERT query with RETURNING clause
            params: Query parameters

        Returns:
            Inserted row as dict
        """
        with self.get_cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def get_status(self) -> Dict[str, Any]:
        """Get database connection status."""
        status = {
            "connected": self.is_connected,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "pool_min": self.min_connections,
            "pool_max": self.max_connections,
        }

        if self.is_connected and self._pool:
            # Get pool stats (approximate)
            try:
                with self.get_cursor() as cur:
                    cur.execute("SELECT count(*) FROM trades")
                    result = cur.fetchone()
                    status["total_trades"] = result["count"] if result else 0
            except:
                status["total_trades"] = "N/A"

        return status


# Global instance
_db_instance: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """
    Get or create global database instance.

    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


def init_db() -> bool:
    """
    Initialize database connection.

    Returns:
        True if successful
    """
    db = get_db()
    return db.connect()


if __name__ == "__main__":
    # Test connection
    print("Testing database connection...")

    db = get_db()
    if db.connect():
        print(f"Connected to {db.database}")

        # Test query
        with db.get_cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"PostgreSQL version: {version['version']}")

        # Test status
        status = db.get_status()
        print(f"Status: {status}")

        db.disconnect()
        print("Disconnected")
    else:
        print("Connection failed!")
