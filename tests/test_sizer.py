"""Tests for backtesting.engine.sizer — all 5 sizer classes."""

from unittest.mock import MagicMock, PropertyMock, patch

import backtrader as bt
import pytest

from backtesting.engine.sizer import (
    CompoundingSizer,
    FixedCashSizer,
    FixedPercentSizer,
    PercentSizer,
    RiskBasedSizer,
)


def _make_sizer(sizer_cls, params=None, cash=100_000, price=50.0,
                position_size=100, portfolio_value=None, strategy=None):
    """Create a sizer with mocked broker, data, and comminfo.

    Returns (sizer, comminfo, cash, data, True).
    """
    sizer = sizer_cls.__new__(sizer_cls)
    # Apply params manually
    p = {}
    if hasattr(sizer_cls, "params"):
        for pname, pval in sizer_cls.params._getitems():
            p[pname] = pval
    if params:
        p.update(params)
    sizer.p = sizer.params = type("Params", (), p)()

    # Mock broker
    sizer.broker = MagicMock()
    position_mock = MagicMock()
    position_mock.size = position_size
    sizer.broker.getposition.return_value = position_mock
    sizer.broker.getvalue.return_value = portfolio_value or cash

    # Mock data
    data = MagicMock()
    data.close = [price]

    # Mock strategy (for RiskBasedSizer)
    if strategy:
        sizer.strategy = strategy
    else:
        sizer.strategy = MagicMock()

    comminfo = MagicMock()
    return sizer, comminfo, cash, data


# ---------------------------------------------------------------------------
# PercentSizer
# ---------------------------------------------------------------------------


class TestPercentSizer:
    def test_buy_normal(self):
        sizer, ci, cash, data = _make_sizer(PercentSizer, price=50.0, cash=100_000)
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # 100_000 * 0.95 / 50 = 1900
        assert size == 1900

    def test_buy_custom_percent(self):
        sizer, ci, cash, data = _make_sizer(
            PercentSizer, params={"percents": 50}, price=100.0, cash=100_000,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # 100_000 * 0.50 / 100 = 500
        assert size == 500

    def test_buy_zero_price(self):
        sizer, ci, cash, data = _make_sizer(PercentSizer, price=0.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_buy_negative_price(self):
        sizer, ci, cash, data = _make_sizer(PercentSizer, price=-10.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_sell_returns_full_position(self):
        sizer, ci, cash, data = _make_sizer(PercentSizer, position_size=250)
        size = sizer._getsizing(ci, cash, data, isbuy=False)
        assert size == 250

    def test_buy_insufficient_cash(self):
        sizer, ci, cash, data = _make_sizer(PercentSizer, price=200_000.0, cash=100)
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 0


# ---------------------------------------------------------------------------
# CompoundingSizer
# ---------------------------------------------------------------------------


class TestCompoundingSizer:
    def test_buy_uses_portfolio_value(self):
        sizer, ci, cash, data = _make_sizer(
            CompoundingSizer, price=50.0, cash=100_000, portfolio_value=120_000,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # available = min(120_000 * 0.95, 100_000 * 0.99) = min(114_000, 99_000) = 99_000
        # size = 99_000 / 50 = 1980
        assert size == 1980

    def test_buy_cash_constrained(self):
        """When cash < portfolio value, cash limits the size."""
        sizer, ci, cash, data = _make_sizer(
            CompoundingSizer, price=50.0, cash=50_000, portfolio_value=200_000,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # available = min(200_000 * 0.95, 50_000 * 0.99) = min(190_000, 49_500) = 49_500
        # 49_500 / 50 = 990
        assert size == 990

    def test_buy_zero_price(self):
        sizer, ci, cash, data = _make_sizer(CompoundingSizer, price=0.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_sell_returns_full_position(self):
        sizer, ci, cash, data = _make_sizer(CompoundingSizer, position_size=300)
        size = sizer._getsizing(ci, cash, data, isbuy=False)
        assert size == 300


# ---------------------------------------------------------------------------
# FixedPercentSizer
# ---------------------------------------------------------------------------


class TestFixedPercentSizer:
    def test_buy_uses_initial_capital(self):
        sizer, ci, cash, data = _make_sizer(
            FixedPercentSizer,
            params={"percents": 95, "initial_capital": 100_000},
            price=50.0, cash=200_000,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # available = min(100_000 * 0.95, 200_000 * 0.99) = min(95_000, 198_000) = 95_000
        # 95_000 / 50 = 1900
        assert size == 1900

    def test_buy_cash_limited(self):
        sizer, ci, cash, data = _make_sizer(
            FixedPercentSizer,
            params={"percents": 95, "initial_capital": 100_000},
            price=50.0, cash=40_000,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # available = min(95_000, 39_600) = 39_600
        # 39_600 / 50 = 792
        assert size == 792

    def test_buy_zero_price(self):
        sizer, ci, cash, data = _make_sizer(FixedPercentSizer, price=0.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_sell_returns_full_position(self):
        sizer, ci, cash, data = _make_sizer(FixedPercentSizer, position_size=500)
        assert sizer._getsizing(ci, cash, data, isbuy=False) == 500


# ---------------------------------------------------------------------------
# RiskBasedSizer
# ---------------------------------------------------------------------------


class TestRiskBasedSizer:
    def test_buy_with_valid_stop(self):
        strategy = MagicMock()
        strategy._current_stop_price = 45.0
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer,
            params={"risk_pct": 5.0, "max_position_pct": 20.0},
            price=50.0, cash=100_000, portfolio_value=100_000,
            strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # risk_per_share = 50 - 45 = 5
        # max_dollar_risk = 100_000 * 0.05 = 5_000
        # max_shares_by_risk = 5_000 / 5 = 1_000
        # max_position_value = 100_000 * 0.20 = 20_000
        # max_shares_by_position = 20_000 / 50 = 400
        # max_shares_by_cash = 100_000 * 0.99 / 50 = 1_980
        # size = min(1_000, 400, 1_980) = 400
        assert size == 400

    def test_buy_risk_limited(self):
        strategy = MagicMock()
        strategy._current_stop_price = 49.0  # Tight stop
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer,
            params={"risk_pct": 1.0, "max_position_pct": 50.0},
            price=50.0, cash=100_000, portfolio_value=100_000,
            strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # risk_per_share = 50 - 49 = 1
        # max_dollar_risk = 100_000 * 0.01 = 1_000
        # max_shares_by_risk = 1_000 / 1 = 1_000
        # max_shares_by_position = 100_000 * 0.50 / 50 = 1_000
        # max_shares_by_cash = 99_000 / 50 = 1_980
        # size = min(1_000, 1_000, 1_980) = 1_000
        assert size == 1000

    def test_buy_no_stop_falls_back(self):
        strategy = MagicMock()
        strategy._current_stop_price = None
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=50.0, cash=100_000, strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # Fallback: 100_000 * 0.95 / 50 = 1_900
        assert size == 1900

    def test_buy_stop_above_price_falls_back(self):
        strategy = MagicMock()
        strategy._current_stop_price = 60.0  # Above entry price
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=50.0, cash=100_000, strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 1900  # Fallback

    def test_buy_stop_zero_falls_back(self):
        strategy = MagicMock()
        strategy._current_stop_price = 0.0
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=50.0, cash=100_000, strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 1900  # Fallback

    def test_buy_stop_equal_price_falls_back(self):
        strategy = MagicMock()
        strategy._current_stop_price = 50.0  # Equal to entry
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=50.0, cash=100_000, strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 1900  # Fallback

    def test_buy_zero_price(self):
        strategy = MagicMock()
        strategy._current_stop_price = 45.0
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=0.0, strategy=strategy,
        )
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_sell_returns_full_position(self):
        sizer, ci, cash, data = _make_sizer(RiskBasedSizer, position_size=150)
        assert sizer._getsizing(ci, cash, data, isbuy=False) == 150

    def test_no_stop_attr_falls_back(self):
        """Strategy without _current_stop_price attribute → fallback."""
        strategy = MagicMock(spec=[])  # No attributes
        sizer, ci, cash, data = _make_sizer(
            RiskBasedSizer, price=50.0, cash=100_000, strategy=strategy,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 1900


# ---------------------------------------------------------------------------
# FixedCashSizer
# ---------------------------------------------------------------------------


class TestFixedCashSizer:
    def test_buy_default_cash(self):
        sizer, ci, cash, data = _make_sizer(
            FixedCashSizer, params={"cash": 10_000}, price=50.0,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # 10_000 / 50 = 200
        assert size == 200

    def test_buy_custom_cash(self):
        sizer, ci, cash, data = _make_sizer(
            FixedCashSizer, params={"cash": 5_000}, price=25.0,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 200

    def test_buy_zero_price(self):
        sizer, ci, cash, data = _make_sizer(FixedCashSizer, price=0.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_buy_negative_price(self):
        sizer, ci, cash, data = _make_sizer(FixedCashSizer, price=-10.0)
        assert sizer._getsizing(ci, cash, data, isbuy=True) == 0

    def test_sell_returns_full_position(self):
        sizer, ci, cash, data = _make_sizer(FixedCashSizer, position_size=75)
        assert sizer._getsizing(ci, cash, data, isbuy=False) == 75

    def test_buy_expensive_stock_rounds_down(self):
        sizer, ci, cash, data = _make_sizer(
            FixedCashSizer, params={"cash": 10_000}, price=3_333.0,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        # 10_000 / 3_333 = 3.0003 → int = 3
        assert size == 3

    def test_buy_very_expensive_stock_zero(self):
        sizer, ci, cash, data = _make_sizer(
            FixedCashSizer, params={"cash": 100}, price=200.0,
        )
        size = sizer._getsizing(ci, cash, data, isbuy=True)
        assert size == 0
