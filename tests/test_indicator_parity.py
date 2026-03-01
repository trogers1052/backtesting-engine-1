"""Indicator parity test: backtest bridge vs analytics-service.

Verifies that the backtesting pandas_ta_bridge produces the same
indicator columns with identical parameters as the live
analytics-service.  Both use pandas-ta under the hood, so exact
numerical parity is guaranteed when the call signatures match.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.indicators.pandas_ta_bridge import calculate_indicators


def _make_ohlcv(bars: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data with enough bars for all indicator warmup.

    Uses a trending price series to ensure MACD and other trend indicators
    can compute valid values (pure random walk can cause pandas-ta to
    return None for some indicators).
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=bars, freq="B", tz="UTC")
    # Start at 50, add a mild uptrend + noise so indicators converge
    trend = np.linspace(0, 20, bars)
    noise = np.cumsum(np.random.randn(bars) * 0.3)
    close = 50 + trend + noise
    close = np.maximum(close, 5.0)  # floor at $5
    high = close * (1 + np.abs(np.random.randn(bars) * 0.015))
    low = close * (1 - np.abs(np.random.randn(bars) * 0.015))
    opn = (high + low) / 2 + np.random.randn(bars) * 0.1
    volume = np.random.randint(100_000, 1_000_000, size=bars).astype(float)

    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# --- Column-level parity ---

# Expected columns that analytics-service publishes, mapped to the
# exact pandas-ta call parameters used in analytics/indicators.py.
EXPECTED_COLUMNS = [
    "RSI_14",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HISTOGRAM",
    "BB_LOWER",
    "BB_MID",
    "BB_UPPER",
    "BB_BANDWIDTH",
    "BB_PERCENT",
    "ATR_14",
    "STOCH_K",
    "STOCH_D",
    "ADX_14",
    "DMP_14",
    "DMN_14",
    "EMA_9",
    "EMA_21",
    "volume_sma_20",
]


class TestIndicatorParity:
    """Verify backtesting bridge matches analytics-service indicators."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_ohlcv()
        self.result = calculate_indicators(self.df)

    def test_all_expected_columns_present(self):
        """Every indicator the analytics-service publishes must exist."""
        missing = [c for c in EXPECTED_COLUMNS if c not in self.result.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_nan_after_warmup(self):
        """After warmup rows are dropped, no NaN should remain in indicators."""
        for col in EXPECTED_COLUMNS:
            nan_count = int(self.result[col].isna().sum())
            assert nan_count == 0, f"{col} has {nan_count} NaN values after warmup"

    def test_no_inf_values(self):
        """No Inf values in any indicator column."""
        for col in EXPECTED_COLUMNS:
            inf_count = int(np.isinf(self.result[col]).sum())
            assert inf_count == 0, f"{col} has {inf_count} Inf values"

    def test_rsi_range(self):
        """RSI must be in [0, 100]."""
        rsi = self.result["RSI_14"]
        assert rsi.min() >= 0, f"RSI min {rsi.min()} < 0"
        assert rsi.max() <= 100, f"RSI max {rsi.max()} > 100"

    def test_stochastic_range(self):
        """Stochastic K and D must be in [0, 100]."""
        for col in ["STOCH_K", "STOCH_D"]:
            vals = self.result[col]
            assert vals.min() >= 0, f"{col} min {vals.min()} < 0"
            assert vals.max() <= 100, f"{col} max {vals.max()} > 100"

    def test_adx_range(self):
        """ADX must be in [0, 100]."""
        adx = self.result["ADX_14"]
        assert adx.min() >= 0, f"ADX min {adx.min()} < 0"
        assert adx.max() <= 100, f"ADX max {adx.max()} > 100"

    def test_bollinger_band_ordering(self):
        """BB_LOWER <= BB_MID <= BB_UPPER for every row."""
        assert (self.result["BB_LOWER"] <= self.result["BB_MID"] + 1e-9).all()
        assert (self.result["BB_MID"] <= self.result["BB_UPPER"] + 1e-9).all()

    def test_sma_responsiveness(self):
        """SMA_20 reacts faster to recent price changes than SMA_200."""
        # SMA_20 should be closer to current price than SMA_200
        last_close = self.result["close"].iloc[-1]
        sma20_diff = abs(self.result["SMA_20"].iloc[-1] - last_close)
        sma200_diff = abs(self.result["SMA_200"].iloc[-1] - last_close)
        assert sma20_diff < sma200_diff

    def test_atr_positive(self):
        """ATR must be strictly positive."""
        assert (self.result["ATR_14"] > 0).all()

    def test_volume_sma_positive(self):
        """Volume SMA must be positive."""
        assert (self.result["volume_sma_20"] > 0).all()

    def test_ema_values_present(self):
        """EMA_9 and EMA_21 have valid positive values."""
        assert (self.result["EMA_9"] > 0).all()
        assert (self.result["EMA_21"] > 0).all()

    def test_macd_histogram_equals_diff(self):
        """MACD_HISTOGRAM should equal MACD - MACD_SIGNAL."""
        diff = self.result["MACD"] - self.result["MACD_SIGNAL"]
        np.testing.assert_allclose(
            self.result["MACD_HISTOGRAM"].values,
            diff.values,
            atol=1e-10,
        )

    def test_warmup_rows_dropped(self):
        """calculate_indicators drops rows where longest SMA is NaN."""
        # We passed 300 bars; SMA_200 needs 200 bars warmup
        # So result should have roughly 100 bars (300 - 200 + 1)
        assert len(self.result) < len(self.df)
        assert len(self.result) > 50  # sanity: not all dropped

    def test_default_params_match_analytics_service(self):
        """Verify default config params match analytics-service defaults."""
        from backtesting.config import settings

        # These must match analytics-service/analytics/config.py
        assert settings.rsi_period == 14
        assert settings.macd_fast == 12
        assert settings.macd_slow == 26
        assert settings.macd_signal == 9
        assert settings.sma_periods == [20, 50, 200]
        assert settings.bb_period == 20
        assert settings.atr_period == 14
