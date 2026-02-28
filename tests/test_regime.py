"""Tests for regime-stratified analysis."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.regime import (
    RegimeAnalysisResult,
    RegimeClassifier,
    RegimeMetrics,
    _lookup_regime,
    analyze_by_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regime_df(dates, regimes):
    """Build a regime DataFrame with datetime index and 'regime' column."""
    idx = pd.to_datetime(dates).tz_localize("UTC")
    return pd.DataFrame({"regime": regimes}, index=idx)


def _make_spy_df_with_indicators(dates, close, sma_50, sma_200):
    """Build a DataFrame that mimics calculate_indicators output."""
    idx = pd.to_datetime(dates).tz_localize("UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000] * len(dates),
            "SMA_50": sma_50,
            "SMA_200": sma_200,
        },
        index=idx,
    )


def _make_result(trades, symbol="TEST"):
    """Build a minimal BacktestResult from a trade list."""
    wins = [t for t in trades if (t.get("profit_pct") or 0) > 0]
    return BacktestResult(
        symbol=symbol,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        strategy_name="test",
        initial_cash=100_000,
        final_value=110_000,
        total_return=0.10,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(trades) - len(wins),
        win_rate=len(wins) / max(len(trades), 1),
        trades=trades,
    )


# ---------------------------------------------------------------------------
# _lookup_regime
# ---------------------------------------------------------------------------


class TestLookupRegime:
    def test_exact_date_match(self):
        df = _make_regime_df(["2023-03-15", "2023-03-16"], ["bull", "bear"])
        assert _lookup_regime(df, "2023-03-15") == "bull"
        assert _lookup_regime(df, "2023-03-16") == "bear"

    def test_fallback_to_prior_date(self):
        """Weekend should fall back to Friday."""
        df = _make_regime_df(["2023-03-10", "2023-03-13"], ["bull", "chop"])
        # 2023-03-11 is Saturday — should fall back to 2023-03-10 (Friday)
        assert _lookup_regime(df, "2023-03-11") == "bull"

    def test_date_before_all_data_returns_unknown(self):
        df = _make_regime_df(["2023-06-01", "2023-06-02"], ["bull", "bear"])
        assert _lookup_regime(df, "2023-01-01") == "unknown"

    def test_accepts_date_object(self):
        df = _make_regime_df(["2023-03-15"], ["chop"])
        assert _lookup_regime(df, date(2023, 3, 15)) == "chop"

    def test_accepts_datetime_object(self):
        df = _make_regime_df(["2023-03-15"], ["bull"])
        assert _lookup_regime(df, datetime(2023, 3, 15, 9, 30)) == "bull"

    def test_accepts_string_with_time(self):
        df = _make_regime_df(["2023-03-15"], ["bear"])
        assert _lookup_regime(df, "2023-03-15T09:30:00") == "bear"


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------


def _stub_loader(spy_df, vix_df=None):
    """Create a mock loader returning real DataFrames per symbol.

    SPY calls get a non-empty raw df (so df.empty is False).
    VIX calls get vix_df if provided, otherwise empty (no VIX overlay).
    """
    raw_spy = pd.DataFrame(
        {"close": [100.0]},
        index=pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
    )
    empty_df = pd.DataFrame()

    def _load(symbol, *args, **kwargs):
        if symbol == "VIX":
            return vix_df if vix_df is not None else empty_df
        return raw_spy

    mock_loader = MagicMock()
    mock_loader.load.side_effect = _load
    return mock_loader


class TestRegimeClassifier:
    @patch("backtesting.validation.regime.calculate_indicators")
    def test_bull_classification(self, mock_calc):
        """close > SMA_200 and SMA_50 > SMA_200 → bull."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "bull"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_bear_classification(self, mock_calc):
        """close < SMA_200 and SMA_50 < SMA_200 → bear."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[350.0], sma_50=[380.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "bear"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_chop_classification(self, mock_calc):
        """Mixed conditions → chop (default)."""
        mock_loader = _stub_loader(None)

        # close > SMA_200 but SMA_50 < SMA_200 → not bull, not bear → chop
        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[410.0], sma_50=[390.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "chop"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_cache_hit(self, mock_calc):
        """Second call with same dates should use cache, not reload data."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        r1 = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))
        first_call_count = mock_loader.load.call_count  # SPY + VIX calls
        r2 = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        # No additional load calls on cache hit
        assert mock_loader.load.call_count == first_call_count
        assert r1 is r2  # Same cached object

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_empty_data_raises(self, mock_calc):
        """Empty SPY data should raise ValueError."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = pd.DataFrame()

        classifier = RegimeClassifier(mock_loader)

        with pytest.raises(ValueError, match="No SPY data"):
            classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_multiple_days_mixed_regimes(self, mock_calc):
        """Multiple days with different regimes."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-01-01", "2023-06-01", "2023-09-01"],
            close=[450.0, 350.0, 410.0],
            sma_50=[430.0, 380.0, 390.0],
            sma_200=[400.0, 400.0, 400.0],
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        regimes = result["regime"].tolist()
        assert regimes == ["bull", "bear", "chop"]


# ---------------------------------------------------------------------------
# RegimeAnalysisResult.regime_dependent
# ---------------------------------------------------------------------------


class TestRegimeDependent:
    def test_all_profit_from_one_regime(self):
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=10, winning_trades=8,
                win_rate=0.8, total_return=50.0, avg_trade_return=5.0,
                sharpe_ratio=1.5, profit_factor=4.0,
            ),
            "bear": RegimeMetrics(
                regime="bear", total_trades=5, winning_trades=1,
                win_rate=0.2, total_return=-10.0, avg_trade_return=-2.0,
                sharpe_ratio=-0.5, profit_factor=0.3,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        # 100% of profit from bull → regime_dependent
        assert r.regime_dependent is True

    def test_balanced_profit(self):
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=10, winning_trades=7,
                win_rate=0.7, total_return=20.0, avg_trade_return=2.0,
                sharpe_ratio=1.0, profit_factor=2.0,
            ),
            "chop": RegimeMetrics(
                regime="chop", total_trades=8, winning_trades=5,
                win_rate=0.625, total_return=15.0, avg_trade_return=1.875,
                sharpe_ratio=0.8, profit_factor=1.5,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        # bull: 20/35 = 57%, chop: 15/35 = 43% → not regime dependent
        assert r.regime_dependent is False

    def test_no_profit_not_dependent(self):
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=5, winning_trades=1,
                win_rate=0.2, total_return=-5.0, avg_trade_return=-1.0,
                sharpe_ratio=-0.5, profit_factor=0.3,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        assert r.regime_dependent is False

    def test_empty_metrics_not_dependent(self):
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics={})
        assert r.regime_dependent is False


# ---------------------------------------------------------------------------
# analyze_by_regime (integration with mocked loader)
# ---------------------------------------------------------------------------


class TestAnalyzeByRegime:
    @patch("backtesting.validation.regime.calculate_indicators")
    def test_trades_grouped_by_regime(self, mock_calc):
        """Trades should be assigned to correct regimes based on entry date."""
        mock_loader = _stub_loader(None)

        # Bull on 2023-03-15, Bear on 2023-09-15
        spy_df = _make_spy_df_with_indicators(
            ["2023-03-15", "2023-09-15"],
            close=[450.0, 350.0],
            sma_50=[430.0, 380.0],
            sma_200=[400.0, 400.0],
        )
        mock_calc.return_value = spy_df

        trades = [
            {"entry_date": "2023-03-15", "profit_pct": 5.0},
            {"entry_date": "2023-03-15", "profit_pct": 3.0},
            {"entry_date": "2023-09-15", "profit_pct": -2.0},
        ]
        result = _make_result(trades)
        analysis = analyze_by_regime(result, mock_loader)

        assert "bull" in analysis.regime_metrics
        assert "bear" in analysis.regime_metrics
        assert analysis.regime_metrics["bull"].total_trades == 2
        assert analysis.regime_metrics["bear"].total_trades == 1

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_metrics_calculated_correctly(self, mock_calc):
        """Verify per-regime metric calculations."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-03-15"],
            close=[450.0],
            sma_50=[430.0],
            sma_200=[400.0],
        )
        mock_calc.return_value = spy_df

        trades = [
            {"entry_date": "2023-03-15", "profit_pct": 8.0},
            {"entry_date": "2023-03-15", "profit_pct": -3.0},
            {"entry_date": "2023-03-15", "profit_pct": 5.0},
        ]
        result = _make_result(trades)
        analysis = analyze_by_regime(result, mock_loader)

        m = analysis.regime_metrics["bull"]
        assert m.total_trades == 3
        assert m.winning_trades == 2
        assert abs(m.win_rate - 2 / 3) < 1e-9
        assert abs(m.total_return - 10.0) < 1e-9
        assert abs(m.avg_trade_return - 10.0 / 3) < 1e-9
        # profit_factor = gross_profit / gross_loss = 13.0 / 3.0
        assert abs(m.profit_factor - 13.0 / 3.0) < 0.01

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_no_trades_returns_empty_metrics(self, mock_calc):
        """Result with no trades should produce empty regime_metrics."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        result = _make_result([])
        analysis = analyze_by_regime(result, mock_loader)

        assert len(analysis.regime_metrics) == 0

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_trades_missing_entry_date_skipped(self, mock_calc):
        """Trades without entry_date should be silently skipped."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-03-15"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        trades = [
            {"entry_date": "2023-03-15", "profit_pct": 5.0},
            {"profit_pct": 3.0},  # no entry_date
        ]
        result = _make_result(trades)
        analysis = analyze_by_regime(result, mock_loader)

        assert analysis.regime_metrics["bull"].total_trades == 1

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_symbol_propagated(self, mock_calc):
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        result = _make_result([], symbol="PPLT")
        analysis = analyze_by_regime(result, mock_loader)
        assert analysis.symbol == "PPLT"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_all_losses_profit_factor_capped(self, mock_calc):
        """When gross_loss > 0 but gross_profit = 0, profit_factor should be 0."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-03-15"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        trades = [
            {"entry_date": "2023-03-15", "profit_pct": -5.0},
            {"entry_date": "2023-03-15", "profit_pct": -3.0},
        ]
        result = _make_result(trades)
        analysis = analyze_by_regime(result, mock_loader)

        m = analysis.regime_metrics["bull"]
        assert m.profit_factor == 0.0
        assert m.winning_trades == 0

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_all_wins_profit_factor_max(self, mock_calc):
        """When gross_loss = 0, profit_factor should be capped at 999.99."""
        mock_loader = _stub_loader(None)

        spy_df = _make_spy_df_with_indicators(
            ["2023-03-15"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        trades = [
            {"entry_date": "2023-03-15", "profit_pct": 5.0},
            {"entry_date": "2023-03-15", "profit_pct": 3.0},
        ]
        result = _make_result(trades)
        analysis = analyze_by_regime(result, mock_loader)

        m = analysis.regime_metrics["bull"]
        assert m.profit_factor == 999.99


# ---------------------------------------------------------------------------
# VIX overlay tests
# ---------------------------------------------------------------------------


def _make_vix_df(dates, vix_close):
    """Build a VIX DataFrame with datetime index and close column."""
    idx = pd.to_datetime(dates).tz_localize("UTC")
    return pd.DataFrame({"close": vix_close}, index=idx)


class TestVixOverlay:
    @patch("backtesting.validation.regime.calculate_indicators")
    def test_crisis_overrides_bull(self, mock_calc):
        """VIX > 35 should override bull to crisis."""
        vix_df = _make_vix_df(["2023-06-01"], [40.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "crisis"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_crisis_overrides_bear(self, mock_calc):
        """VIX > 35 should override bear to crisis."""
        vix_df = _make_vix_df(["2023-06-01"], [50.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[350.0], sma_50=[380.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "crisis"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_volatile_overrides_chop(self, mock_calc):
        """VIX > 25 (but < 35) with chop → volatile."""
        vix_df = _make_vix_df(["2023-06-01"], [28.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[410.0], sma_50=[390.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "volatile"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_volatile_overrides_bear(self, mock_calc):
        """VIX > 25 (but < 35) with bear → volatile."""
        vix_df = _make_vix_df(["2023-06-01"], [30.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[350.0], sma_50=[380.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "volatile"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_bull_preserved_with_elevated_vix(self, mock_calc):
        """VIX > 25 should NOT override bull — strong trend trumps vol."""
        vix_df = _make_vix_df(["2023-06-01"], [28.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "bull"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_low_vix_no_override(self, mock_calc):
        """VIX < 25 should not change any regime."""
        vix_df = _make_vix_df(["2023-06-01"], [15.0])
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[410.0], sma_50=[390.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        # chop stays chop with low VIX
        assert result.loc[result.index[0], "regime"] == "chop"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_vix_data_unavailable_falls_back(self, mock_calc):
        """Missing VIX data should gracefully fall back to SPY-only."""
        mock_loader = _stub_loader(None)  # No vix_df → empty VIX

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        # Falls back to SPY-only: bull
        assert result.loc[result.index[0], "regime"] == "bull"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_vix_loader_exception_falls_back(self, mock_calc):
        """Exception loading VIX should gracefully fall back to SPY-only."""
        mock_loader = MagicMock()
        raw_spy = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime(["2023-01-01"]).tz_localize("UTC"),
        )

        def _load(symbol, *args, **kwargs):
            if symbol == "VIX":
                raise Exception("VIX not in database")
            return raw_spy

        mock_loader.load.side_effect = _load

        spy_df = _make_spy_df_with_indicators(
            ["2023-06-01"], close=[450.0], sma_50=[430.0], sma_200=[400.0]
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        assert result.loc[result.index[0], "regime"] == "bull"

    @patch("backtesting.validation.regime.calculate_indicators")
    def test_mixed_vix_regimes(self, mock_calc):
        """Multiple days with varying VIX levels."""
        vix_df = _make_vix_df(
            ["2023-01-01", "2023-06-01", "2023-09-01"],
            [15.0, 30.0, 40.0],
        )
        mock_loader = _stub_loader(None, vix_df=vix_df)

        spy_df = _make_spy_df_with_indicators(
            ["2023-01-01", "2023-06-01", "2023-09-01"],
            close=[450.0, 410.0, 410.0],
            sma_50=[430.0, 390.0, 390.0],
            sma_200=[400.0, 400.0, 400.0],
        )
        mock_calc.return_value = spy_df

        classifier = RegimeClassifier(mock_loader)
        result = classifier.get_regimes(date(2023, 1, 1), date(2023, 12, 31))

        regimes = result["regime"].tolist()
        # Day 1: bull (VIX=15, low)
        # Day 2: volatile (chop + VIX=30)
        # Day 3: crisis (VIX=40)
        assert regimes == ["bull", "volatile", "crisis"]
