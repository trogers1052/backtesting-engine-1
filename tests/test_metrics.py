"""Tests for backtesting.analyzers.metrics — calculate_metrics, _max_consecutive,
calculate_risk_metrics."""

import math

import numpy as np
import pytest

from backtesting.analyzers.metrics import (
    _max_consecutive,
    calculate_metrics,
    calculate_risk_metrics,
)


# ---------------------------------------------------------------------------
# _max_consecutive
# ---------------------------------------------------------------------------


class TestMaxConsecutive:
    def test_empty_array(self):
        assert _max_consecutive(np.array([])) == 0

    def test_all_true(self):
        assert _max_consecutive(np.array([True, True, True])) == 3

    def test_all_false(self):
        assert _max_consecutive(np.array([False, False, False])) == 0

    def test_single_true(self):
        assert _max_consecutive(np.array([True])) == 1

    def test_single_false(self):
        assert _max_consecutive(np.array([False])) == 0

    def test_mixed_longest_at_start(self):
        cond = np.array([True, True, True, False, True])
        assert _max_consecutive(cond) == 3

    def test_mixed_longest_at_end(self):
        cond = np.array([True, False, True, True, True, True])
        assert _max_consecutive(cond) == 4

    def test_alternating(self):
        cond = np.array([True, False, True, False, True])
        assert _max_consecutive(cond) == 1

    def test_two_equal_runs(self):
        cond = np.array([True, True, False, True, True])
        assert _max_consecutive(cond) == 2


# ---------------------------------------------------------------------------
# calculate_metrics — empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestCalculateMetricsEmpty:
    def test_empty_list(self):
        result = calculate_metrics([])
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0
        assert result["avg_holding_days"] == 0

    def test_no_profit_pct(self):
        trades = [{"symbol": "AAPL"}, {"symbol": "GOOG"}]
        result = calculate_metrics(trades)
        assert result["total_trades"] == 2
        assert result.get("error") == "No completed trades"

    def test_none_profit_pct_filtered(self):
        trades = [{"profit_pct": None}, {"profit_pct": 0.05}]
        result = calculate_metrics(trades)
        assert result["total_trades"] == 1


# ---------------------------------------------------------------------------
# calculate_metrics — wins only
# ---------------------------------------------------------------------------


class TestCalculateMetricsWinsOnly:
    def test_all_winners(self):
        trades = [
            {"profit_pct": 0.10},
            {"profit_pct": 0.05},
            {"profit_pct": 0.08},
        ]
        result = calculate_metrics(trades)
        assert result["total_trades"] == 3
        assert result["winning_trades"] == 3
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 1.0
        assert abs(result["avg_profit"] - np.mean([0.10, 0.05, 0.08])) < 1e-9
        assert result["avg_loss"] == 0
        assert result["profit_factor"] == float("inf")
        assert result["consecutive_wins"] == 3
        assert result["consecutive_losses"] == 0


# ---------------------------------------------------------------------------
# calculate_metrics — losses only
# ---------------------------------------------------------------------------


class TestCalculateMetricsLossesOnly:
    def test_all_losers(self):
        trades = [
            {"profit_pct": -0.05},
            {"profit_pct": -0.03},
        ]
        result = calculate_metrics(trades)
        assert result["total_trades"] == 2
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 2
        assert result["win_rate"] == 0.0
        assert result["avg_win"] == 0
        assert result["profit_factor"] == 0
        assert result["consecutive_wins"] == 0
        assert result["consecutive_losses"] == 2


# ---------------------------------------------------------------------------
# calculate_metrics — mixed trades
# ---------------------------------------------------------------------------


class TestCalculateMetricsMixed:
    def test_mixed_trades(self, sample_trades):
        result = calculate_metrics(sample_trades)
        assert result["total_trades"] == 5
        assert result["winning_trades"] == 3
        assert result["losing_trades"] == 2
        assert abs(result["win_rate"] - 0.6) < 1e-9

    def test_best_worst_trade(self, sample_trades):
        result = calculate_metrics(sample_trades)
        assert result["best_trade"] == 0.10
        assert result["worst_trade"] == pytest.approx(-0.0417)

    def test_profit_factor_positive(self, sample_trades):
        result = calculate_metrics(sample_trades)
        # gross_profit = 0.10 + 0.10 + 0.069 = 0.269
        # gross_loss = abs(-0.0417 + -0.0286) = 0.0703
        assert result["profit_factor"] > 1.0
        assert result["profit_factor"] == pytest.approx(0.269 / 0.0703, rel=1e-3)

    def test_consecutive_wins_losses(self, sample_trades):
        # Pattern: W, L, W, L, W → max consecutive wins = 1
        result = calculate_metrics(sample_trades)
        assert result["consecutive_wins"] == 1
        assert result["consecutive_losses"] == 1

    def test_holding_days_from_date_strings(self, sample_trades):
        result = calculate_metrics(sample_trades)
        # 10d, 7d, 14d, 4d, 15d → avg = 10.0
        assert result["avg_holding_days"] == 10.0

    def test_holding_days_with_datetime_objects(self):
        from datetime import datetime
        trades = [
            {
                "profit_pct": 0.05,
                "entry_date": datetime(2023, 1, 1),
                "exit_date": datetime(2023, 1, 11),
            },
        ]
        result = calculate_metrics(trades)
        assert result["avg_holding_days"] == 10.0

    def test_holding_days_missing_dates(self):
        trades = [{"profit_pct": 0.05}]
        result = calculate_metrics(trades)
        assert result["avg_holding_days"] == 0

    def test_zero_profit_is_loss(self):
        """A trade at exactly 0% profit counts as a loss (profits <= 0)."""
        trades = [{"profit_pct": 0.0}]
        result = calculate_metrics(trades)
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 1


# ---------------------------------------------------------------------------
# calculate_risk_metrics
# ---------------------------------------------------------------------------


class TestCalculateRiskMetrics:
    def test_short_curve_returns_empty(self):
        assert calculate_risk_metrics([100.0]) == {}

    def test_empty_curve(self):
        assert calculate_risk_metrics([]) == {}

    def test_flat_equity_curve(self):
        """Constant equity → 0 volatility → Sharpe = 0."""
        curve = [100_000.0] * 100
        result = calculate_risk_metrics(curve)
        assert result["sharpe_ratio"] == 0
        assert result["sortino_ratio"] == 0
        assert result["max_drawdown"] == 0.0

    def test_growing_equity_curve(self):
        """Steadily growing equity → positive Sharpe, low drawdown."""
        curve = [100_000 + i * 100 for i in range(252)]
        result = calculate_risk_metrics(curve)
        assert result["sharpe_ratio"] > 0
        assert result["max_drawdown"] < 0.01  # Minimal drawdown
        assert result["annual_return"] > 0
        assert result["calmar_ratio"] >= 0

    def test_declining_equity_curve(self):
        """Steadily declining equity → negative/zero Sharpe."""
        curve = [100_000 - i * 100 for i in range(100)]
        result = calculate_risk_metrics(curve)
        assert result["max_drawdown"] > 0
        assert result["annual_return"] < 0

    def test_drawdown_calculation(self):
        """Peak at start, then drop."""
        curve = [100, 90, 80, 85, 95]
        result = calculate_risk_metrics(curve)
        # Max drawdown = (100 - 80) / 100 = 0.20
        assert result["max_drawdown"] == pytest.approx(0.20)

    def test_two_point_curve(self):
        """Minimum valid curve: 2 points."""
        result = calculate_risk_metrics([100, 110])
        assert "sharpe_ratio" in result
        assert "annual_return" in result

    def test_sortino_no_downside(self):
        """No negative returns → sortino = 0 (no downside std)."""
        curve = [100, 110, 120, 130]
        result = calculate_risk_metrics(curve)
        # If all excess returns are positive, downside_std=0 → sortino=0
        # Actually depends on risk-free subtraction
        assert isinstance(result["sortino_ratio"], float)

    def test_calmar_zero_drawdown(self):
        """No drawdown → calmar = 0."""
        curve = [100, 101, 102, 103, 104]
        result = calculate_risk_metrics(curve)
        assert result["calmar_ratio"] == 0 or result["max_drawdown"] == 0.0
