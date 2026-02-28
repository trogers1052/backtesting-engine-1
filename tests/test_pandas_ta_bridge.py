"""Tests for backtesting.indicators.pandas_ta_bridge — get_required_warmup_bars."""

import pytest

from backtesting.indicators.pandas_ta_bridge import get_required_warmup_bars


class TestGetRequiredWarmupBars:
    def test_default_periods(self):
        """Default SMA periods are [20, 50, 200] → max = 200 + 10 = 210."""
        result = get_required_warmup_bars()
        assert result == 210

    def test_custom_periods(self):
        result = get_required_warmup_bars([10, 30])
        assert result == 40  # 30 + 10

    def test_single_period(self):
        result = get_required_warmup_bars([50])
        assert result == 60

    def test_large_period(self):
        result = get_required_warmup_bars([500])
        assert result == 510
