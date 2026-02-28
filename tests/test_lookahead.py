"""Tests for lookahead and data integrity validation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtesting.validation.lookahead import (
    IntegrityReport,
    check_indicator_completeness,
    check_multiframe_alignment,
    check_no_future_data,
    check_no_gaps,
    check_price_sanity,
    check_timestamp_monotonic,
    run_integrity_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(dates, close=None, **extra_cols):
    """Build a simple OHLCV DataFrame with DatetimeIndex."""
    idx = pd.to_datetime(dates).tz_localize("UTC")
    n = len(dates)
    close = close if close is not None else [100.0 + i for i in range(n)]
    data = {
        "open": close,
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1_000_000] * n,
    }
    data.update(extra_cols)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# check_timestamp_monotonic
# ---------------------------------------------------------------------------


class TestTimestampMonotonic:
    def test_valid_monotonic(self):
        df = _make_df(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = check_timestamp_monotonic(df)
        assert result.passed is True

    def test_duplicate_timestamps(self):
        df = _make_df(["2023-01-01", "2023-01-01", "2023-01-03"])
        result = check_timestamp_monotonic(df)
        assert result.passed is False
        assert "duplicate" in result.detail.lower()

    def test_non_monotonic(self):
        df = _make_df(["2023-01-03", "2023-01-01", "2023-01-02"])
        result = check_timestamp_monotonic(df)
        assert result.passed is False
        assert "non-monotonic" in result.detail.lower()

    def test_single_row(self):
        df = _make_df(["2023-01-01"])
        result = check_timestamp_monotonic(df)
        assert result.passed is True

    def test_empty_df(self):
        df = pd.DataFrame()
        result = check_timestamp_monotonic(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_no_future_data
# ---------------------------------------------------------------------------


class TestNoFutureData:
    def test_data_within_bounds(self):
        df = _make_df(["2023-01-01", "2023-06-15", "2023-12-31"])
        result = check_no_future_data(df, end_date=date(2023, 12, 31))
        assert result.passed is True

    def test_data_exceeds_end_date(self):
        df = _make_df(["2023-01-01", "2024-01-15"])
        result = check_no_future_data(df, end_date=date(2023, 12, 31))
        assert result.passed is False
        assert "2024-01-15" in result.detail

    def test_exactly_on_end_date(self):
        df = _make_df(["2023-12-31"])
        result = check_no_future_data(df, end_date=date(2023, 12, 31))
        assert result.passed is True

    def test_empty_df(self):
        df = _make_df([])
        result = check_no_future_data(df, end_date=date(2023, 12, 31))
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_no_gaps
# ---------------------------------------------------------------------------


class TestNoGaps:
    def test_uniform_spacing(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = _make_df(dates.strftime("%Y-%m-%d").tolist())
        result = check_no_gaps(df)
        assert result.passed is True

    def test_large_gap_detected(self):
        dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-06-01"]
        df = _make_df(dates)
        result = check_no_gaps(df)
        assert result.passed is False
        assert "gap" in result.detail.lower()

    def test_too_few_rows(self):
        df = _make_df(["2023-01-01", "2023-01-02"])
        result = check_no_gaps(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_indicator_completeness
# ---------------------------------------------------------------------------


class TestIndicatorCompleteness:
    def test_clean_indicators(self):
        df = _make_df(
            ["2023-01-01", "2023-01-02"],
            SMA_20=[100.0, 101.0],
            RSI_14=[55.0, 60.0],
        )
        result = check_indicator_completeness(df)
        assert result.passed is True

    def test_nan_in_indicator(self):
        df = _make_df(
            ["2023-01-01", "2023-01-02"],
            SMA_20=[100.0, float("nan")],
        )
        result = check_indicator_completeness(df)
        assert result.passed is False
        assert "SMA_20" in result.detail

    def test_inf_in_indicator(self):
        df = _make_df(
            ["2023-01-01", "2023-01-02"],
            SMA_20=[100.0, float("inf")],
        )
        result = check_indicator_completeness(df)
        assert result.passed is False
        assert "Inf" in result.detail

    def test_explicit_columns(self):
        df = _make_df(
            ["2023-01-01", "2023-01-02"],
            SMA_20=[100.0, float("nan")],
            RSI_14=[55.0, 60.0],
        )
        # Only check RSI — should pass even though SMA has NaN
        result = check_indicator_completeness(df, indicator_columns=["RSI_14"])
        assert result.passed is True

    def test_no_indicator_columns(self):
        df = _make_df(["2023-01-01"])
        result = check_indicator_completeness(df, indicator_columns=[])
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_multiframe_alignment
# ---------------------------------------------------------------------------


class TestMultiframeAlignment:
    def test_aligned_feeds(self):
        daily = _make_df(["2023-01-01", "2023-01-02", "2023-01-03"])
        intraday = _make_df(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = check_multiframe_alignment(daily, intraday)
        assert result.passed is True

    def test_daily_extends_beyond_intraday(self):
        daily = _make_df(["2023-01-01", "2023-01-02", "2023-01-05"])
        intraday = _make_df(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = check_multiframe_alignment(daily, intraday)
        assert result.passed is False
        assert "extends beyond" in result.detail

    def test_intraday_starts_before_daily(self):
        daily = _make_df(["2023-01-03", "2023-01-04"])
        intraday = _make_df(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
        result = check_multiframe_alignment(daily, intraday)
        assert result.passed is False
        assert "before daily" in result.detail

    def test_empty_feeds(self):
        daily = _make_df([])
        intraday = _make_df(["2023-01-01"])
        result = check_multiframe_alignment(daily, intraday)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_price_sanity
# ---------------------------------------------------------------------------


class TestPriceSanity:
    def test_valid_prices(self):
        df = _make_df(["2023-01-01", "2023-01-02"])
        result = check_price_sanity(df)
        assert result.passed is True

    def test_zero_price(self):
        df = _make_df(["2023-01-01", "2023-01-02"], close=[0.0, 100.0])
        result = check_price_sanity(df)
        assert result.passed is False
        assert "non-positive" in result.detail

    def test_negative_price(self):
        df = _make_df(["2023-01-01", "2023-01-02"], close=[-5.0, 100.0])
        result = check_price_sanity(df)
        assert result.passed is False

    def test_close_above_high(self):
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [105.0],  # close > high
                "volume": [1_000_000],
            },
            index=pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
        )
        result = check_price_sanity(df)
        assert result.passed is False
        assert "close > high" in result.detail

    def test_extreme_move(self):
        df = _make_df(["2023-01-01", "2023-01-02"], close=[100.0, 200.0])
        result = check_price_sanity(df)
        assert result.passed is False
        assert "50%" in result.detail

    def test_empty_df(self):
        df = _make_df([])
        result = check_price_sanity(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# run_integrity_checks (integration)
# ---------------------------------------------------------------------------


class TestRunIntegrityChecks:
    def test_clean_data_all_pass(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = _make_df(dates.strftime("%Y-%m-%d").tolist())
        report = run_integrity_checks(df, end_date=date(2023, 12, 31))
        assert report.all_passed is True
        assert len(report.failures) == 0

    def test_multiple_failures(self):
        # Non-monotonic + future data + NaN indicator
        df = _make_df(
            ["2023-01-03", "2023-01-01", "2024-06-01"],
            SMA_20=[100.0, float("nan"), 102.0],
        )
        report = run_integrity_checks(df, end_date=date(2023, 12, 31))
        assert report.all_passed is False
        assert len(report.failures) >= 2

    def test_multiframe_check_included_when_secondary(self):
        daily = _make_df(["2023-01-01", "2023-01-02", "2023-01-05"])
        intraday = _make_df(["2023-01-01", "2023-01-02", "2023-01-03"])
        report = run_integrity_checks(daily, df_secondary=intraday)
        names = [c.name for c in report.checks]
        assert "multiframe_alignment" in names

    def test_multiframe_check_skipped_without_secondary(self):
        df = _make_df(["2023-01-01", "2023-01-02"])
        report = run_integrity_checks(df)
        names = [c.name for c in report.checks]
        assert "multiframe_alignment" not in names

    def test_report_failure_list(self):
        df = _make_df(["2023-01-01", "2024-06-01"])
        report = run_integrity_checks(df, end_date=date(2023, 12, 31))
        failed_names = [f.name for f in report.failures]
        assert "no_future_data" in failed_names
