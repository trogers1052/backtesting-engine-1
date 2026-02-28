"""Tests for backtesting.engine.backtrader_runner — BacktestResult dataclass,
BacktraderRunner init, run_multiple error handling."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from backtesting.engine.backtrader_runner import BacktestResult, BacktraderRunner


# ---------------------------------------------------------------------------
# BacktestResult dataclass
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_required_fields(self):
        r = BacktestResult(
            symbol="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            strategy_name="test",
            initial_cash=100_000.0,
            final_value=110_000.0,
            total_return=0.10,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.60,
        )
        assert r.symbol == "AAPL"
        assert r.total_return == 0.10

    def test_optional_fields_default_none(self):
        r = BacktestResult(
            symbol="X", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="t", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0,
        )
        assert r.avg_win is None
        assert r.avg_loss is None
        assert r.profit_factor is None
        assert r.sharpe_ratio is None
        assert r.max_drawdown is None
        assert r.max_drawdown_pct is None
        assert r.trades is None

    def test_trades_list(self, sample_result):
        assert isinstance(sample_result.trades, list)
        assert len(sample_result.trades) == 2


# ---------------------------------------------------------------------------
# BacktraderRunner — init defaults
# ---------------------------------------------------------------------------


class TestBacktraderRunnerInit:
    @patch("backtesting.engine.backtrader_runner.TimescaleLoader")
    def test_default_values(self, mock_loader):
        runner = BacktraderRunner()
        assert runner.initial_cash == 100_000.0
        assert runner.commission == 0.001
        assert runner.compound is True
        assert runner.sizing_mode == "percent"

    @patch("backtesting.engine.backtrader_runner.TimescaleLoader")
    def test_custom_values(self, mock_loader):
        runner = BacktraderRunner(
            initial_cash=50_000,
            commission=0.002,
            compound=False,
            sizing_mode="risk_based",
            risk_pct=2.0,
            max_position_pct=10.0,
        )
        assert runner.initial_cash == 50_000
        assert runner.commission == 0.002
        assert runner.compound is False
        assert runner.sizing_mode == "risk_based"
        assert runner.risk_pct == 2.0
        assert runner.max_position_pct == 10.0


# ---------------------------------------------------------------------------
# BacktraderRunner — run_multiple error handling
# ---------------------------------------------------------------------------


class TestRunMultiple:
    @patch("backtesting.engine.backtrader_runner.TimescaleLoader")
    def test_error_in_one_symbol_continues(self, mock_loader):
        runner = BacktraderRunner()

        call_count = 0
        def mock_run(symbol, **kwargs):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise ValueError("No data for BAD")
            return BacktestResult(
                symbol=symbol, start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31), strategy_name="test",
                initial_cash=100_000, final_value=110_000, total_return=0.10,
                total_trades=5, winning_trades=3, losing_trades=2, win_rate=0.6,
            )

        runner.run = mock_run
        results = runner.run_multiple(["AAPL", "BAD", "GOOG"])
        assert "AAPL" in results
        assert "BAD" not in results
        assert "GOOG" in results
        assert call_count == 3

    @patch("backtesting.engine.backtrader_runner.TimescaleLoader")
    def test_empty_symbols(self, mock_loader):
        runner = BacktraderRunner()
        results = runner.run_multiple([])
        assert results == {}

    @patch("backtesting.engine.backtrader_runner.TimescaleLoader")
    def test_all_symbols_fail(self, mock_loader):
        runner = BacktraderRunner()
        runner.run = MagicMock(side_effect=Exception("boom"))
        results = runner.run_multiple(["A", "B", "C"])
        assert results == {}
