"""
Data cache for validation pipeline.

Loads OHLCV data + indicators once per symbol/timeframe, reuses across
all configs. With fork-based multiprocessing, child workers inherit
the cached DataFrames via copy-on-write (zero extra memory).
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...data import TimescaleLoader
from ...indicators import calculate_indicators
from ...validation.lookahead import run_integrity_checks

logger = logging.getLogger(__name__)


class DataCache:
    """Pre-loads and caches DataFrames with indicators.

    Usage:
        cache = DataCache()
        cache.preload("AEM", date(2021,1,1), date(2025,12,31), exit_tf="5min")
        df_daily = cache.get_daily("AEM")
        df_5min = cache.get_intraday("AEM")
    """

    def __init__(self):
        self._loader = TimescaleLoader()
        self._daily: Dict[str, pd.DataFrame] = {}
        self._intraday: Dict[str, pd.DataFrame] = {}

    def preload(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        exit_timeframe: str = "daily",
    ) -> None:
        """Load daily + optional intraday data for a symbol.

        Runs integrity checks and computes indicators. Result is cached.
        """
        if symbol in self._daily:
            return  # already loaded

        logger.info(f"DataCache: loading {symbol} daily data")
        df_daily = self._loader.load(symbol, start_date, end_date, timeframe="daily")
        if df_daily.empty:
            raise ValueError(f"No daily data found for {symbol}")

        # Integrity checks on daily data
        integrity = run_integrity_checks(df_daily, end_date=end_date)
        if not integrity.all_passed:
            failures = "; ".join(f.detail for f in integrity.failures)
            raise ValueError(f"Data integrity check failed for {symbol}: {failures}")

        # Compute indicators on daily
        df_daily = calculate_indicators(df_daily)
        self._daily[symbol] = df_daily
        logger.info(f"DataCache: {symbol} daily cached ({len(df_daily)} bars)")

        # Load intraday if multi-TF
        if exit_timeframe != "daily":
            logger.info(f"DataCache: loading {symbol} {exit_timeframe} data")
            df_intraday = self._loader.load(symbol, start_date, end_date, timeframe=exit_timeframe)
            if df_intraday.empty:
                raise ValueError(f"No {exit_timeframe} data found for {symbol}")

            # Integrity checks — skip gap check for intraday (overnight/weekend gaps expected)
            integrity = run_integrity_checks(df_intraday, end_date=end_date, skip_gap_check=True)
            if not integrity.all_passed:
                failures = "; ".join(f.detail for f in integrity.failures)
                raise ValueError(f"Intraday integrity check failed for {symbol}: {failures}")

            # Oscillators only on intraday (no SMAs)
            df_intraday = calculate_indicators(df_intraday, sma_periods=[])
            self._intraday[symbol] = df_intraday
            logger.info(f"DataCache: {symbol} {exit_timeframe} cached ({len(df_intraday)} bars)")

    def get_daily(self, symbol: str) -> pd.DataFrame:
        """Get cached daily DataFrame (with indicators)."""
        if symbol not in self._daily:
            raise KeyError(f"No cached daily data for {symbol}. Call preload() first.")
        return self._daily[symbol]

    def get_intraday(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached intraday DataFrame (with indicators), or None."""
        return self._intraday.get(symbol)

    def get_daily_slice(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """Get a date-sliced view of the daily data (for walk-backward windows)."""
        df = self.get_daily(symbol)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        return df[(df.index >= start_ts) & (df.index <= end_ts)]

    def get_intraday_slice(
        self, symbol: str, start: date, end: date
    ) -> Optional[pd.DataFrame]:
        """Get a date-sliced view of the intraday data."""
        df = self.get_intraday(symbol)
        if df is None:
            return None
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        return df[(df.index >= start_ts) & (df.index <= end_ts)]

    def clear(self, symbol: str = None):
        """Clear cached data. If symbol given, clear only that symbol."""
        if symbol:
            self._daily.pop(symbol, None)
            self._intraday.pop(symbol, None)
        else:
            self._daily.clear()
            self._intraday.clear()

    @property
    def loaded_symbols(self) -> List[str]:
        return list(self._daily.keys())

    def memory_usage_mb(self) -> float:
        """Estimate memory usage of cached data in MB."""
        total = 0
        for df in self._daily.values():
            total += df.memory_usage(deep=True).sum()
        for df in self._intraday.values():
            total += df.memory_usage(deep=True).sum()
        return total / 1024 / 1024
