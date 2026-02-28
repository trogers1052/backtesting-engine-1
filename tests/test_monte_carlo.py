"""Tests for Monte Carlo simulation analysis."""

from datetime import date

import numpy as np
import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.monte_carlo import (
    MonteCarloResult,
    monte_carlo_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(trades, symbol="TEST", initial_cash=100_000):
    """Build a minimal BacktestResult from a trade list."""
    wins = [t for t in trades if (t.get("profit_pct") or 0) > 0]
    return BacktestResult(
        symbol=symbol,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        strategy_name="test",
        initial_cash=initial_cash,
        final_value=initial_cash,
        total_return=0.0,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(trades) - len(wins),
        win_rate=len(wins) / max(len(trades), 1),
        trades=trades,
    )


def _trades_from_pnl(pnl_list):
    """Create trade dicts from a list of P&L percentages."""
    return [{"profit_pct": p} for p in pnl_list]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMonteCarloEdgeCases:
    def test_fewer_than_2_trades_raises(self):
        result = _make_result([{"profit_pct": 5.0}])
        with pytest.raises(ValueError, match="at least 2 trades"):
            monte_carlo_analysis(result)

    def test_no_trades_raises(self):
        result = _make_result([])
        with pytest.raises(ValueError, match="at least 2 trades"):
            monte_carlo_analysis(result)

    def test_trades_with_none_profit_skipped(self):
        """Trades missing profit_pct are filtered out."""
        trades = [
            {"profit_pct": 5.0},
            {"profit_pct": None},
            {"profit_pct": -3.0},
        ]
        result = _make_result(trades)
        mc = monte_carlo_analysis(result, n_simulations=100)
        assert mc.n_trades == 2  # None trade filtered out

    def test_exactly_2_trades(self):
        """Minimum valid input: 2 trades."""
        trades = _trades_from_pnl([10.0, -5.0])
        result = _make_result(trades)
        mc = monte_carlo_analysis(result, n_simulations=100)
        assert mc.n_trades == 2
        assert mc.n_simulations == 100


# ---------------------------------------------------------------------------
# All-winners scenario
# ---------------------------------------------------------------------------


class TestMonteCarloAllWinners:
    def test_no_ruin(self):
        """All winning trades should never hit ruin."""
        trades = _trades_from_pnl([5.0, 7.0, 3.0, 6.0, 4.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        assert mc.ruin_probability == 0.0
        assert mc.survival_rate == 1.0

    def test_all_equity_above_initial(self):
        """All percentiles should be above initial cash."""
        trades = _trades_from_pnl([5.0, 7.0, 3.0, 6.0, 4.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        assert mc.equity_p5 > 1000
        assert mc.equity_median > 1000

    def test_zero_drawdown_impossible_with_positive(self):
        """Drawdown can be zero if equity never dips."""
        trades = _trades_from_pnl([5.0, 7.0, 3.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        # All trades are positive, so drawdown should be 0
        assert mc.drawdown_median == 0.0
        assert mc.drawdown_worst == 0.0


# ---------------------------------------------------------------------------
# All-losers scenario
# ---------------------------------------------------------------------------


class TestMonteCarloAllLosers:
    def test_high_ruin_probability(self):
        """All losing trades with large losses should have high ruin risk."""
        # 5 trades of -20% each: 1000 * 0.8^5 = $327.68 < $500 threshold
        trades = _trades_from_pnl([-20.0, -20.0, -20.0, -20.0, -20.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(
            result, n_simulations=1000, initial_cash=1000, ruin_threshold_pct=0.5
        )
        # All orderings produce same result (all same trades)
        assert mc.ruin_probability == 1.0

    def test_equity_below_initial(self):
        """All percentiles should be below initial cash."""
        trades = _trades_from_pnl([-5.0, -3.0, -7.0, -4.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        assert mc.equity_p95 < 1000
        assert mc.equity_median < 1000


# ---------------------------------------------------------------------------
# Mixed trades
# ---------------------------------------------------------------------------


class TestMonteCarloMixed:
    def test_percentile_ordering(self):
        """Percentiles must be monotonically increasing."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0, -3.0, 8.0, -2.0, 6.0, -4.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=5000, initial_cash=1000)
        assert mc.equity_p5 <= mc.equity_p25
        assert mc.equity_p25 <= mc.equity_median
        assert mc.equity_median <= mc.equity_p75
        assert mc.equity_p75 <= mc.equity_p95

    def test_drawdown_ordering(self):
        """Drawdown percentiles: median <= p95 <= worst."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0, -3.0, 8.0, -2.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=5000, initial_cash=1000)
        assert mc.drawdown_median <= mc.drawdown_p95
        assert mc.drawdown_p95 <= mc.drawdown_worst

    def test_ruin_between_0_and_1(self):
        """Ruin probability is always in [0, 1]."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0, -3.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        assert 0.0 <= mc.ruin_probability <= 1.0

    def test_survival_rate_complement(self):
        """survival_rate = 1 - ruin_probability."""
        trades = _trades_from_pnl([10.0, -15.0, 7.0, -10.0, 5.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000)
        assert abs(mc.survival_rate - (1.0 - mc.ruin_probability)) < 1e-10


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestMonteCarloReproducibility:
    def test_same_seed_same_result(self):
        """Same random seed produces identical results."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0, -3.0, 8.0])
        result = _make_result(trades, initial_cash=1000)

        mc1 = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000, random_seed=42)
        mc2 = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000, random_seed=42)

        assert mc1.equity_median == mc2.equity_median
        assert mc1.drawdown_p95 == mc2.drawdown_p95
        assert mc1.ruin_probability == mc2.ruin_probability

    def test_different_seed_different_result(self):
        """Different seeds generally produce different results (not guaranteed but very likely)."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0, -3.0, 8.0, -2.0, 4.0, -6.0])
        result = _make_result(trades, initial_cash=1000)

        mc1 = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000, random_seed=42)
        mc2 = monte_carlo_analysis(result, n_simulations=1000, initial_cash=1000, random_seed=99)

        # With 8 different trades and different seeds, percentiles should differ
        # (extremely unlikely to be identical)
        assert mc1.equity_median != mc2.equity_median


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------


class TestMonteCarloParameters:
    def test_custom_initial_cash(self):
        """Initial cash parameter is respected."""
        trades = _trades_from_pnl([10.0, -5.0, 7.0])
        result = _make_result(trades, initial_cash=100_000)
        mc = monte_carlo_analysis(result, n_simulations=100, initial_cash=888.80)
        assert mc.initial_cash == 888.80

    def test_custom_ruin_threshold(self):
        """Ruin threshold adjusts based on parameter."""
        trades = _trades_from_pnl([-10.0, -10.0, -10.0, -10.0])
        result = _make_result(trades, initial_cash=1000)

        # 50% threshold: ruin at $500
        mc_50 = monte_carlo_analysis(
            result, n_simulations=100, initial_cash=1000, ruin_threshold_pct=0.5
        )
        # 20% threshold: ruin at $200
        mc_20 = monte_carlo_analysis(
            result, n_simulations=100, initial_cash=1000, ruin_threshold_pct=0.2
        )
        assert mc_50.ruin_threshold == 500.0
        assert mc_20.ruin_threshold == 200.0
        # Lower threshold → less ruin
        assert mc_20.ruin_probability <= mc_50.ruin_probability

    def test_defaults_to_backtest_initial_cash(self):
        """When initial_cash not provided, uses backtest result value."""
        trades = _trades_from_pnl([5.0, -3.0])
        result = _make_result(trades, initial_cash=888.80)
        mc = monte_carlo_analysis(result, n_simulations=100)
        assert mc.initial_cash == 888.80


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestMonteCarloResultDataclass:
    def test_fields_populated(self):
        """All fields are populated with reasonable values."""
        trades = _trades_from_pnl([5.0, -3.0, 7.0, -2.0])
        result = _make_result(trades, initial_cash=1000)
        mc = monte_carlo_analysis(result, n_simulations=100, initial_cash=1000)

        assert mc.symbol == "TEST"
        assert mc.n_simulations == 100
        assert mc.n_trades == 4
        assert mc.initial_cash == 1000
        assert mc.equity_p5 > 0
        assert mc.equity_median > 0
        assert mc.drawdown_median >= 0
        assert mc.drawdown_worst >= 0
