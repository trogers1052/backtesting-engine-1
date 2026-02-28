"""Tests for backtesting.reporting.console_report — print_report, print_multi_report."""

from datetime import date
from io import StringIO
from unittest.mock import patch

import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.reporting.console_report import print_report, print_multi_report


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_basic_report_runs_without_error(self, sample_result):
        """Smoke test — print_report should not raise."""
        print_report(sample_result)

    def test_show_trades_false(self, sample_result):
        """Should not print individual trade details."""
        print_report(sample_result, show_trades=False)

    def test_show_trades_true(self, sample_result):
        """Should print individual trades and summary."""
        print_report(sample_result, show_trades=True, trade_limit=5)

    def test_zero_trades(self):
        r = BacktestResult(
            symbol="EMPTY", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0,
        )
        print_report(r)  # Should not raise

    def test_negative_return(self):
        r = BacktestResult(
            symbol="LOSS", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=85_000,
            total_return=-0.15, total_trades=10, winning_trades=3, losing_trades=7,
            win_rate=0.30, avg_win=500.0, avg_loss=-1200.0,
            profit_factor=0.59, sharpe_ratio=-0.5,
            max_drawdown=15_000, max_drawdown_pct=15.0,
        )
        print_report(r)

    def test_infinity_profit_factor(self):
        r = BacktestResult(
            symbol="INF", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=120_000,
            total_return=0.20, total_trades=5, winning_trades=5, losing_trades=0,
            win_rate=1.0, avg_win=4000.0, profit_factor=float("inf"),
            sharpe_ratio=2.5, max_drawdown=1000, max_drawdown_pct=1.0,
        )
        print_report(r)

    def test_none_optional_fields(self):
        r = BacktestResult(
            symbol="MIN", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=1, winning_trades=0, losing_trades=1,
            win_rate=0.0,
        )
        print_report(r)  # avg_win, avg_loss, sharpe, etc. all None

    def test_show_trades_with_none_trades(self):
        r = BacktestResult(
            symbol="X", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, trades=None,
        )
        # trades is None → show_trades=True should still not raise
        print_report(r, show_trades=True)

    def test_show_trades_empty_list(self):
        r = BacktestResult(
            symbol="X", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, trades=[],
        )
        print_report(r, show_trades=True)

    def test_high_drawdown_report(self):
        r = BacktestResult(
            symbol="DD", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=70_000,
            total_return=-0.30, total_trades=20, winning_trades=5, losing_trades=15,
            win_rate=0.25, max_drawdown=30_000, max_drawdown_pct=30.0,
        )
        print_report(r)


# ---------------------------------------------------------------------------
# print_multi_report
# ---------------------------------------------------------------------------


class TestPrintMultiReport:
    def test_multiple_symbols(self, sample_result):
        r2 = BacktestResult(
            symbol="GOOG", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=105_000,
            total_return=0.05, total_trades=8, winning_trades=5, losing_trades=3,
            win_rate=0.625, sharpe_ratio=0.8, max_drawdown_pct=3.2,
        )
        results = {"AAPL": sample_result, "GOOG": r2}
        print_multi_report(results)  # Smoke test

    def test_single_symbol(self, sample_result):
        print_multi_report({"AAPL": sample_result})

    def test_no_sharpe_ratio(self):
        r = BacktestResult(
            symbol="X", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0,
        )
        print_multi_report({"X": r})

    def test_negative_returns_colored_red(self, sample_result):
        r_loss = BacktestResult(
            symbol="LOSS", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=90_000,
            total_return=-0.10, total_trades=5, winning_trades=1, losing_trades=4,
            win_rate=0.20, sharpe_ratio=-0.3, max_drawdown_pct=12.0,
        )
        results = {"AAPL": sample_result, "LOSS": r_loss}
        print_multi_report(results)  # Should not raise
