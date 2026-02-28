"""Tests for bootstrap statistical significance analysis."""

from datetime import date

import numpy as np
import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.bootstrap import (
    BootstrapResult,
    bootstrap_analysis,
    calculate_trade_sharpe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# calculate_trade_sharpe
# ---------------------------------------------------------------------------


class TestCalculateTradeSharpe:
    def test_fewer_than_2_trades(self):
        assert calculate_trade_sharpe(np.array([5.0])) == 0.0
        assert calculate_trade_sharpe(np.array([])) == 0.0

    def test_zero_std(self):
        """All identical returns → std=0 → Sharpe=0."""
        # Use 0.0 which is exactly representable in float (avoids FP precision noise)
        assert calculate_trade_sharpe(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_known_values(self):
        """Manually verify the Sharpe formula."""
        pnl = np.array([10.0, -2.0, 8.0, -1.0, 6.0])
        returns = pnl / 100.0  # [0.10, -0.02, 0.08, -0.01, 0.06]
        rf_per_trade = 0.02 / 52

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        expected = (mean_ret - rf_per_trade) / std_ret * np.sqrt(52)

        result = calculate_trade_sharpe(pnl, risk_free_rate=0.02)
        assert abs(result - expected) < 1e-9

    def test_negative_returns_negative_sharpe(self):
        """Mostly losing trades → negative Sharpe."""
        pnl = np.array([-5.0, -3.0, -8.0, 1.0, -6.0])
        result = calculate_trade_sharpe(pnl)
        assert result < 0

    def test_positive_returns_positive_sharpe(self):
        """Mostly winning trades → positive Sharpe."""
        pnl = np.array([5.0, 3.0, 8.0, -1.0, 6.0])
        result = calculate_trade_sharpe(pnl)
        assert result > 0


# ---------------------------------------------------------------------------
# BootstrapResult properties
# ---------------------------------------------------------------------------


class TestBootstrapResultProperties:
    def test_no_edge_sharpe_when_ci_includes_zero(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=20, n_bootstrap=1000,
            sharpe_point=0.5, sharpe_ci_lower=-0.2, sharpe_ci_upper=1.2,
            win_rate_point=0.6, win_rate_ci_lower=0.55, win_rate_ci_upper=0.65,
        )
        assert r.no_edge_sharpe is True

    def test_edge_sharpe_when_ci_above_zero(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=20, n_bootstrap=1000,
            sharpe_point=1.5, sharpe_ci_lower=0.8, sharpe_ci_upper=2.2,
            win_rate_point=0.6, win_rate_ci_lower=0.55, win_rate_ci_upper=0.65,
        )
        assert r.no_edge_sharpe is False

    def test_no_edge_wr_when_ci_includes_fifty(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=20, n_bootstrap=1000,
            sharpe_point=0.5, sharpe_ci_lower=0.1, sharpe_ci_upper=1.0,
            win_rate_point=0.55, win_rate_ci_lower=0.45, win_rate_ci_upper=0.65,
        )
        assert r.no_edge_wr is True

    def test_edge_wr_when_ci_above_fifty(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=20, n_bootstrap=1000,
            sharpe_point=0.5, sharpe_ci_lower=0.1, sharpe_ci_upper=1.0,
            win_rate_point=0.70, win_rate_ci_lower=0.55, win_rate_ci_upper=0.85,
        )
        assert r.no_edge_wr is False


# ---------------------------------------------------------------------------
# bootstrap_analysis
# ---------------------------------------------------------------------------


class TestBootstrapAnalysis:
    def test_fewer_than_2_trades_raises(self):
        result = _make_result([{"profit_pct": 5.0}])
        with pytest.raises(ValueError, match="at least 2 trades"):
            bootstrap_analysis(result)

    def test_no_trades_raises(self):
        result = _make_result([])
        with pytest.raises(ValueError, match="at least 2 trades"):
            bootstrap_analysis(result)

    def test_reproducibility_with_seed(self):
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0, -1.0, 6.0, 3.0]]
        result = _make_result(trades)

        r1 = bootstrap_analysis(result, n_bootstrap=500, random_seed=42)
        r2 = bootstrap_analysis(result, n_bootstrap=500, random_seed=42)

        assert r1.sharpe_ci_lower == r2.sharpe_ci_lower
        assert r1.sharpe_ci_upper == r2.sharpe_ci_upper
        assert r1.win_rate_ci_lower == r2.win_rate_ci_lower

    def test_strong_edge_significant(self):
        """Many winning trades → CI should not include zero/50%."""
        trades = [{"profit_pct": p} for p in
                  [8.0, 6.0, 10.0, 7.0, 5.0, 9.0, 4.0, 11.0, 6.0, 8.0,
                   7.0, 5.0, 9.0, 12.0, 6.0, 8.0, 10.0, 7.0, 5.0, 9.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=5000, random_seed=42)

        assert r.no_edge_sharpe is False
        assert r.no_edge_wr is False

    def test_random_trades_no_edge(self):
        """50/50 win/loss of similar magnitude → likely no significant edge."""
        np.random.seed(99)
        pnls = np.random.normal(0, 5, 30).tolist()
        trades = [{"profit_pct": p} for p in pnls]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=5000, random_seed=42)

        # With random noise around zero, Sharpe CI should include zero
        assert r.no_edge_sharpe is True

    def test_none_profit_pct_filtered(self):
        """Trades with None profit_pct should be skipped."""
        trades = [
            {"profit_pct": 5.0},
            {"profit_pct": None},
            {"profit_pct": 3.0},
            {"profit_pct": -2.0},
        ]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=100, random_seed=42)
        assert r.n_trades == 3  # None filtered out

    def test_ci_bounds_ordering(self):
        """Lower CI < point estimate < upper CI (approximately)."""
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0, -1.0, 6.0, 3.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=5000, random_seed=42)

        assert r.sharpe_ci_lower <= r.sharpe_ci_upper
        assert r.win_rate_ci_lower <= r.win_rate_ci_upper

    def test_symbol_propagated(self):
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0]]
        result = _make_result(trades, symbol="PPLT")
        r = bootstrap_analysis(result, n_bootstrap=100, random_seed=42)
        assert r.symbol == "PPLT"

    def test_n_bootstrap_propagated(self):
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=200, random_seed=42)
        assert r.n_bootstrap == 200

    def test_p_value_strong_edge_below_005(self):
        """Strong positive edge → p-value should be well below 0.05."""
        trades = [{"profit_pct": p} for p in
                  [8.0, 6.0, 10.0, 7.0, 5.0, 9.0, 4.0, 11.0, 6.0, 8.0,
                   7.0, 5.0, 9.0, 12.0, 6.0, 8.0, 10.0, 7.0, 5.0, 9.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=5000, random_seed=42)

        assert r.p_value < 0.05

    def test_p_value_no_edge_above_005(self):
        """Random noise around zero → p-value should be above 0.05."""
        np.random.seed(99)
        pnls = np.random.normal(0, 5, 30).tolist()
        trades = [{"profit_pct": p} for p in pnls]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=5000, random_seed=42)

        assert r.p_value > 0.05

    def test_p_value_between_0_and_1(self):
        """p-value must always be in [0, 1]."""
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0, -1.0, 6.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=1000, random_seed=42)

        assert 0.0 <= r.p_value <= 1.0

    def test_p_value_reproducible_with_seed(self):
        """Same seed → same p-value."""
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0, -1.0, 6.0, 3.0]]
        result = _make_result(trades)

        r1 = bootstrap_analysis(result, n_bootstrap=500, random_seed=42)
        r2 = bootstrap_analysis(result, n_bootstrap=500, random_seed=42)

        assert r1.p_value == r2.p_value

    def test_p_value_default_zero(self):
        """BootstrapResult default p_value is 0.0 when not provided."""
        r = BootstrapResult(
            symbol="TEST", n_trades=10, n_bootstrap=1000,
            sharpe_point=1.0, sharpe_ci_lower=0.5, sharpe_ci_upper=1.5,
            win_rate_point=0.6, win_rate_ci_lower=0.55, win_rate_ci_upper=0.65,
        )
        assert r.p_value == 0.0

    def test_vectorized_no_warnings(self):
        """Vectorized bootstrap should not emit divide-by-zero warnings."""
        import warnings
        trades = [{"profit_pct": p} for p in [5.0, 5.0, 5.0]]
        result = _make_result(trades)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bootstrap_analysis(result, n_bootstrap=100, random_seed=42)

    def test_vectorized_matches_scalar(self):
        """Vectorized Sharpe results should match the scalar function."""
        trades = [{"profit_pct": p} for p in [5.0, -2.0, 8.0, -1.0, 6.0, 3.0]]
        result = _make_result(trades)
        r = bootstrap_analysis(result, n_bootstrap=500, random_seed=42)

        # Point estimate should match direct call
        trade_pnl = np.array([5.0, -2.0, 8.0, -1.0, 6.0, 3.0])
        expected_sharpe = calculate_trade_sharpe(trade_pnl)
        assert abs(r.sharpe_point - expected_sharpe) < 1e-9
