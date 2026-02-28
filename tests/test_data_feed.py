"""Tests for backtesting.engine.data_feed — PandasDataWithIndicators, create_data_feed."""

import pandas as pd
import numpy as np
import pytest

from backtesting.engine.data_feed import PandasDataWithIndicators, create_data_feed


def _ohlcv_df(n=10):
    """Create a minimal OHLCV DataFrame with DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    data = {
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# PandasDataWithIndicators — class structure
# ---------------------------------------------------------------------------


class TestPandasDataWithIndicatorsStructure:
    def test_indicator_lines_defined(self):
        all_lines = PandasDataWithIndicators.lines._getlines()
        expected_indicator_lines = [
            "rsi_14", "sma_20", "sma_50", "sma_200",
            "macd", "macd_signal", "macd_histogram",
            "bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent",
            "atr_14", "volume_sma_20",
        ]
        for line in expected_indicator_lines:
            assert line in all_lines

    def test_total_line_count(self):
        # 14 custom indicator lines + 7 base OHLCV lines = 21 total
        all_lines = PandasDataWithIndicators.lines._getlines()
        assert len(all_lines) == 21

    def test_subclass_of_pandasdata(self):
        import backtrader as bt
        assert issubclass(PandasDataWithIndicators, bt.feeds.PandasData)


# ---------------------------------------------------------------------------
# create_data_feed — validation
# ---------------------------------------------------------------------------


class TestCreateDataFeed:
    def test_valid_dataframe(self):
        df = _ohlcv_df()
        feed = create_data_feed(df, name="TEST")
        assert isinstance(feed, PandasDataWithIndicators)

    def test_non_datetime_index_raises(self):
        df = _ohlcv_df()
        df = df.reset_index(drop=True)  # Replace DatetimeIndex with RangeIndex
        with pytest.raises(ValueError, match="DatetimeIndex"):
            create_data_feed(df)

    def test_name_passed_through(self):
        df = _ohlcv_df()
        feed = create_data_feed(df, name="MY_SYMBOL")
        assert feed.p.name == "MY_SYMBOL"

    def test_none_name(self):
        df = _ohlcv_df()
        feed = create_data_feed(df, name=None)
        assert feed.p.name is None

    def test_empty_dataframe_with_datetime_index(self):
        """Empty DF with DatetimeIndex should not raise on creation."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="date"),
        )
        feed = create_data_feed(df)
        assert isinstance(feed, PandasDataWithIndicators)

    def test_integer_index_raises(self):
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]},
            index=[0],
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            create_data_feed(df)
