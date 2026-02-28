"""Lookahead and survivorship bias detection for backtesting integrity.

Validates data integrity before and during backtests to catch common
sources of forward-looking bias that inflate backtest results.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheck:
    """Result of a single data integrity check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class IntegrityReport:
    """Aggregated results from all data integrity checks."""

    checks: List[IntegrityCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[IntegrityCheck]:
        return [c for c in self.checks if not c.passed]


def check_timestamp_monotonic(df: pd.DataFrame) -> IntegrityCheck:
    """Verify timestamps are strictly increasing with no duplicates.

    Duplicate or out-of-order timestamps can cause indicators to be
    computed on the wrong data, silently corrupting signals.
    """
    name = "timestamp_monotonic"

    if len(df) < 2:
        return IntegrityCheck(name=name, passed=True, detail="Too few rows to check")

    if not df.index.is_monotonic_increasing:
        # Find first violation
        diffs = df.index.to_series().diff()
        bad = diffs[diffs <= pd.Timedelta(0)]
        if not bad.empty:
            first_bad = bad.index[0]
            return IntegrityCheck(
                name=name, passed=False,
                detail=f"Non-monotonic timestamp at {first_bad}",
            )

    dupes = df.index.duplicated()
    if dupes.any():
        first_dupe = df.index[dupes][0]
        count = int(dupes.sum())
        return IntegrityCheck(
            name=name, passed=False,
            detail=f"{count} duplicate timestamp(s), first at {first_dupe}",
        )

    return IntegrityCheck(name=name, passed=True)


def check_no_future_data(
    df: pd.DataFrame, end_date: date
) -> IntegrityCheck:
    """Verify no data points extend beyond the requested end date.

    Future data in the DataFrame means indicators are computed using
    information that wasn't available at the time of the signal.
    """
    name = "no_future_data"

    if len(df) == 0:
        return IntegrityCheck(name=name, passed=True, detail="Empty DataFrame")

    max_ts = df.index.max()
    max_date = max_ts.date() if hasattr(max_ts, "date") else max_ts

    if max_date > end_date:
        return IntegrityCheck(
            name=name, passed=False,
            detail=f"Data extends to {max_date}, beyond end_date {end_date}",
        )

    return IntegrityCheck(name=name, passed=True)


def check_no_gaps(
    df: pd.DataFrame, max_gap_multiple: float = 5.0
) -> IntegrityCheck:
    """Detect suspicious gaps in the time series.

    Large gaps can indicate missing data or survivorship bias (stock
    delisted/relisted). Uses median interval as baseline.
    """
    name = "no_large_gaps"

    if len(df) < 3:
        return IntegrityCheck(name=name, passed=True, detail="Too few rows to check")

    diffs = df.index.to_series().diff().dropna()
    median_diff = diffs.median()

    if median_diff == pd.Timedelta(0):
        return IntegrityCheck(name=name, passed=True, detail="Zero median interval")

    threshold = median_diff * max_gap_multiple
    large_gaps = diffs[diffs > threshold]

    if not large_gaps.empty:
        worst = large_gaps.idxmax()
        gap_size = large_gaps.max()
        return IntegrityCheck(
            name=name, passed=False,
            detail=(
                f"{len(large_gaps)} gap(s) > {max_gap_multiple}x median interval. "
                f"Largest: {gap_size} at {worst}"
            ),
        )

    return IntegrityCheck(name=name, passed=True)


def check_indicator_completeness(
    df: pd.DataFrame, indicator_columns: Optional[List[str]] = None
) -> IntegrityCheck:
    """Verify indicator columns have no NaN or Inf after warmup.

    NaN/Inf in indicators after the warmup period means something went
    wrong with calculation — possibly from data gaps or corrupt input.
    """
    name = "indicator_completeness"

    if indicator_columns is None:
        # Auto-detect indicator columns (non-OHLCV)
        ohlcv = {"open", "high", "low", "close", "volume"}
        indicator_columns = [c for c in df.columns if c.lower() not in ohlcv]

    if not indicator_columns:
        return IntegrityCheck(name=name, passed=True, detail="No indicator columns found")

    problems = []
    for col in indicator_columns:
        if col not in df.columns:
            continue

        series = df[col]
        nan_count = int(series.isna().sum())
        if nan_count > 0:
            problems.append(f"{col}: {nan_count} NaN")

        inf_count = int(np.isinf(series.replace([np.nan], [0.0])).sum())
        if inf_count > 0:
            problems.append(f"{col}: {inf_count} Inf")

    if problems:
        return IntegrityCheck(
            name=name, passed=False,
            detail=f"Bad indicator values: {'; '.join(problems[:5])}",
        )

    return IntegrityCheck(name=name, passed=True)


def check_multiframe_alignment(
    df_daily: pd.DataFrame, df_intraday: pd.DataFrame
) -> IntegrityCheck:
    """Verify daily and intraday feeds cover the same date range.

    If the daily feed extends beyond the intraday feed, the strategy
    can see tomorrow's SMA while making decisions on today's price —
    classic lookahead bias.
    """
    name = "multiframe_alignment"

    if len(df_daily) == 0 or len(df_intraday) == 0:
        return IntegrityCheck(name=name, passed=True, detail="Empty feed(s)")

    daily_max = df_daily.index.max().date() if hasattr(df_daily.index.max(), "date") else df_daily.index.max()
    intraday_max = df_intraday.index.max().date() if hasattr(df_intraday.index.max(), "date") else df_intraday.index.max()

    if daily_max > intraday_max:
        return IntegrityCheck(
            name=name, passed=False,
            detail=(
                f"Daily feed ({daily_max}) extends beyond intraday feed "
                f"({intraday_max}) — daily indicators may see future data"
            ),
        )

    daily_min = df_daily.index.min().date() if hasattr(df_daily.index.min(), "date") else df_daily.index.min()
    intraday_min = df_intraday.index.min().date() if hasattr(df_intraday.index.min(), "date") else df_intraday.index.min()

    if daily_min > intraday_min:
        return IntegrityCheck(
            name=name, passed=False,
            detail=(
                f"Intraday feed starts ({intraday_min}) before daily feed "
                f"({daily_min}) — early bars have no daily indicators"
            ),
        )

    return IntegrityCheck(name=name, passed=True)


def check_price_sanity(df: pd.DataFrame) -> IntegrityCheck:
    """Detect obviously corrupt price data.

    Catches zero/negative prices, extreme single-bar moves (>50%),
    and close prices outside high/low range — all signs of bad data
    that would make backtest results meaningless.
    """
    name = "price_sanity"

    if len(df) == 0:
        return IntegrityCheck(name=name, passed=True, detail="Empty DataFrame")

    problems = []

    # Check for zero or negative prices
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            bad = (df[col] <= 0).sum()
            if bad > 0:
                problems.append(f"{bad} non-positive {col} values")

    # Check close within high/low bounds
    if all(c in df.columns for c in ["close", "high", "low"]):
        above_high = (df["close"] > df["high"] * 1.001).sum()  # 0.1% tolerance
        below_low = (df["close"] < df["low"] * 0.999).sum()
        if above_high > 0:
            problems.append(f"{above_high} bars where close > high")
        if below_low > 0:
            problems.append(f"{below_low} bars where close < low")

    # Check for extreme single-bar moves (>50%)
    if "close" in df.columns and len(df) > 1:
        pct_change = df["close"].pct_change().abs()
        extreme = (pct_change > 0.5).sum()
        if extreme > 0:
            worst_idx = pct_change.idxmax()
            worst_val = pct_change.max()
            problems.append(
                f"{extreme} bar(s) with >50% move, worst: {worst_val:.1%} at {worst_idx}"
            )

    if problems:
        return IntegrityCheck(
            name=name, passed=False,
            detail="; ".join(problems),
        )

    return IntegrityCheck(name=name, passed=True)


def run_integrity_checks(
    df: pd.DataFrame,
    end_date: Optional[date] = None,
    df_secondary: Optional[pd.DataFrame] = None,
) -> IntegrityReport:
    """Run all applicable data integrity checks.

    Args:
        df: Primary data feed (DataFrame with DatetimeIndex).
        end_date: Expected end date for future-data check.
        df_secondary: Secondary feed for multi-timeframe alignment check.

    Returns:
        IntegrityReport with results of all checks.
    """
    report = IntegrityReport()

    report.checks.append(check_timestamp_monotonic(df))
    report.checks.append(check_price_sanity(df))

    if end_date is not None:
        report.checks.append(check_no_future_data(df, end_date))

    report.checks.append(check_no_gaps(df))
    report.checks.append(check_indicator_completeness(df))

    if df_secondary is not None:
        report.checks.append(check_multiframe_alignment(df, df_secondary))

    return report
