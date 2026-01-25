"""
TimescaleDB Data Loader

Loads OHLCV data from the TimescaleDB ohlcv_1min continuous aggregate.
"""

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from ..config import settings

logger = logging.getLogger(__name__)


class TimescaleLoader:
    """Load historical OHLCV data from TimescaleDB."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize loader with database connection parameters."""
        self.host = host or settings.market_data_db_host
        self.port = port or settings.market_data_db_port
        self.user = user or settings.market_data_db_user
        self.password = password or settings.market_data_db_password
        self.database = database or settings.market_data_db_name

    def _get_connection(self):
        """Create database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def load(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "WPM")
            start_date: Start date for data
            end_date: End date for data (default: today)

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
            Index is datetime (timezone-aware UTC)
        """
        if end_date is None:
            end_date = date.today()

        logger.info(f"Loading {symbol} data from {start_date} to {end_date}")

        # Aggregate 1-minute bars to daily OHLCV
        query = """
            SELECT
                time_bucket('1 day', time) as datetime,
                (array_agg(open ORDER BY time ASC))[1] as open,
                MAX(high) as high,
                MIN(low) as low,
                (array_agg(close ORDER BY time DESC))[1] as close,
                SUM(volume) as volume
            FROM ohlcv_1min
            WHERE symbol = %s
              AND time >= %s
              AND time <= %s
            GROUP BY time_bucket('1 day', time)
            ORDER BY datetime ASC;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (symbol, start_date, end_date))
                    rows = cur.fetchall()

            if not rows:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(rows)

            # Ensure datetime column is proper datetime type
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df.set_index("datetime", inplace=True)

            # Ensure numeric types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            logger.info(f"Loaded {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            raise

    def get_available_symbols(self) -> list:
        """Get list of symbols available in the database."""
        query = """
            SELECT DISTINCT symbol
            FROM ohlcv_1min
            ORDER BY symbol;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def get_date_range(self, symbol: str) -> tuple:
        """Get available date range for a symbol."""
        query = """
            SELECT
                MIN(time) as min_date,
                MAX(time) as max_date,
                COUNT(*) as bar_count
            FROM ohlcv_1min
            WHERE symbol = %s;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (symbol,))
                    row = cur.fetchone()

            if row and row["min_date"]:
                return (row["min_date"], row["max_date"], row["bar_count"])
            return (None, None, 0)

        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return (None, None, 0)
