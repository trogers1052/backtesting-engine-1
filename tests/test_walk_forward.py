"""Tests for walk-forward validation."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(symbol="TEST", sharpe=1.5, **overrides):
    """Build a minimal BacktestResult with a given Sharpe."""
    defaults = dict(
        symbol=symbol,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        strategy_name="test",
        initial_cash=100_000,
        final_value=110_000,
        total_return=0.10,
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        win_rate=0.60,
        sharpe_ratio=sharpe,
        trades=[],
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _make_window(train_sharpe, test_sharpe, window_num=1):
    """Build a WalkForwardWindow with given Sharpe ratios."""
    return WalkForwardWindow(
        window_num=window_num,
        train_start=date(2020, 1, 1),
        train_end=date(2021, 12, 31),
        test_start=date(2022, 1, 1),
        test_end=date(2022, 12, 31),
        train_result=_make_result(sharpe=train_sharpe),
        test_result=_make_result(sharpe=test_sharpe),
    )


# ---------------------------------------------------------------------------
# WalkForwardWindow.is_overfit
# ---------------------------------------------------------------------------


class TestWalkForwardWindowIsOverfit:
    def test_train_sharpe_none(self):
        w = _make_window(train_sharpe=None, test_sharpe=1.0)
        assert w.is_overfit is True

    def test_test_sharpe_none(self):
        w = _make_window(train_sharpe=1.5, test_sharpe=None)
        assert w.is_overfit is True

    def test_both_sharpe_none(self):
        w = _make_window(train_sharpe=None, test_sharpe=None)
        assert w.is_overfit is True

    def test_train_sharpe_zero(self):
        w = _make_window(train_sharpe=0.0, test_sharpe=0.5)
        assert w.is_overfit is True

    def test_train_sharpe_negative(self):
        w = _make_window(train_sharpe=-0.5, test_sharpe=0.5)
        assert w.is_overfit is True

    def test_overfit_ratio_below_50pct(self):
        """Test Sharpe < 50% of train Sharpe → overfit."""
        w = _make_window(train_sharpe=2.0, test_sharpe=0.8)
        assert w.is_overfit is True  # 0.8 < 1.0 (50% of 2.0)

    def test_valid_ratio_at_50pct(self):
        """Test Sharpe == 50% of train Sharpe → valid."""
        w = _make_window(train_sharpe=2.0, test_sharpe=1.0)
        assert w.is_overfit is False

    def test_valid_ratio_above_50pct(self):
        """Test Sharpe > 50% of train Sharpe → valid."""
        w = _make_window(train_sharpe=2.0, test_sharpe=1.5)
        assert w.is_overfit is False

    def test_test_sharpe_exceeds_train(self):
        """Test Sharpe > train Sharpe → definitely valid."""
        w = _make_window(train_sharpe=1.0, test_sharpe=1.5)
        assert w.is_overfit is False


# ---------------------------------------------------------------------------
# WalkForwardResult properties
# ---------------------------------------------------------------------------


class TestWalkForwardResult:
    def test_empty_windows_no_overfit(self):
        r = WalkForwardResult(symbol="TEST", windows=[])
        assert r.overall_overfit is False
        assert r.overfit_count == 0

    def test_all_valid_no_overfit(self):
        windows = [
            _make_window(2.0, 1.5, window_num=1),
            _make_window(1.8, 1.2, window_num=2),
        ]
        r = WalkForwardResult(symbol="TEST", windows=windows)
        assert r.overall_overfit is False
        assert r.overfit_count == 0

    def test_one_overfit_flagged(self):
        windows = [
            _make_window(2.0, 1.5, window_num=1),  # valid
            _make_window(2.0, 0.5, window_num=2),  # overfit
        ]
        r = WalkForwardResult(symbol="TEST", windows=windows)
        assert r.overall_overfit is True
        assert r.overfit_count == 1

    def test_all_overfit(self):
        windows = [
            _make_window(2.0, 0.3, window_num=1),
            _make_window(2.0, 0.5, window_num=2),
        ]
        r = WalkForwardResult(symbol="TEST", windows=windows)
        assert r.overall_overfit is True
        assert r.overfit_count == 2


# ---------------------------------------------------------------------------
# WalkForwardValidator.validate_simple
# ---------------------------------------------------------------------------


class TestValidateSimple:
    def test_date_split_70_30(self):
        """Default 70/30 split should calculate correct date boundaries."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple("AAPL", start, end)

        assert len(result.windows) == 1
        assert result.symbol == "AAPL"

        # Check runner was called twice (train + test)
        assert mock_runner.run.call_count == 2

        # Verify date split: 70% of ~1461 days = ~1023 days
        train_call = mock_runner.run.call_args_list[0]
        test_call = mock_runner.run.call_args_list[1]

        assert train_call.kwargs["symbol"] == "AAPL"
        assert train_call.kwargs["start_date"] == start

        total_days = (end - start).days
        expected_train_end = start + timedelta(days=int(total_days * 0.7))
        assert train_call.kwargs["end_date"] == expected_train_end

        expected_test_start = expected_train_end + timedelta(days=1)
        assert test_call.kwargs["start_date"] == expected_test_start
        assert test_call.kwargs["end_date"] == end

    def test_custom_train_pct(self):
        """80/20 split with custom train_pct."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2024, 1, 1)

        result = validator.validate_simple("AAPL", start, end, train_pct=0.8)

        train_call = mock_runner.run.call_args_list[0]
        total_days = (end - start).days
        expected_train_end = start + timedelta(days=int(total_days * 0.8))
        assert train_call.kwargs["end_date"] == expected_train_end

    def test_run_kwargs_passed_through(self):
        """Extra kwargs should be forwarded to runner.run()."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        validator.validate_simple(
            "AAPL", date(2020, 1, 1), date(2023, 12, 31),
            strategy_name="custom_strat", cash=50000,
        )

        for call in mock_runner.run.call_args_list:
            assert call.kwargs["strategy_name"] == "custom_strat"
            assert call.kwargs["cash"] == 50000

    def test_window_contains_results(self):
        """Window should contain train and test BacktestResult."""
        train_result = _make_result(sharpe=2.0)
        test_result = _make_result(sharpe=1.0)
        mock_runner = MagicMock()
        mock_runner.run.side_effect = [train_result, test_result]

        validator = WalkForwardValidator(mock_runner)
        result = validator.validate_simple("AAPL", date(2020, 1, 1), date(2023, 12, 31))

        window = result.windows[0]
        assert window.train_result.sharpe_ratio == 2.0
        assert window.test_result.sharpe_ratio == 1.0
        assert window.window_num == 1


# ---------------------------------------------------------------------------
# WalkForwardValidator.validate_rolling
# ---------------------------------------------------------------------------


class TestValidateRolling:
    def test_generates_multiple_windows(self):
        """Rolling validation should produce multiple windows."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        # 5 years of data, 2-year train, 1-year test, 1-year step
        result = validator.validate_rolling(
            "AAPL",
            date(2018, 1, 1),
            date(2023, 12, 31),
            train_days=730,
            test_days=365,
            step_days=365,
        )

        # Should produce at least 2 windows
        assert len(result.windows) >= 2
        assert result.symbol == "AAPL"

        # Each window gets 2 runner.run() calls
        assert mock_runner.run.call_count == len(result.windows) * 2

    def test_windows_numbered_sequentially(self):
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        result = validator.validate_rolling(
            "AAPL", date(2018, 1, 1), date(2023, 12, 31),
            train_days=730, test_days=365, step_days=365,
        )

        for i, w in enumerate(result.windows, 1):
            assert w.window_num == i

    def test_stops_when_test_exceeds_end(self):
        """Should not create a window if test_end > end_date."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        # Only 3 years of data with 2-year train + 1-year test = exactly 1 window
        result = validator.validate_rolling(
            "AAPL", date(2020, 1, 1), date(2023, 1, 1),
            train_days=730, test_days=365, step_days=365,
        )

        assert len(result.windows) == 1

    def test_empty_when_period_too_short(self):
        """No windows if the date range is too short for even one window."""
        mock_runner = MagicMock()
        validator = WalkForwardValidator(mock_runner)

        result = validator.validate_rolling(
            "AAPL", date(2023, 1, 1), date(2023, 6, 1),
            train_days=730, test_days=365, step_days=365,
        )

        assert len(result.windows) == 0
        assert mock_runner.run.call_count == 0

    def test_run_kwargs_passed_through(self):
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        validator.validate_rolling(
            "AAPL", date(2018, 1, 1), date(2023, 12, 31),
            train_days=730, test_days=365, step_days=365,
            strategy_name="custom",
        )

        for call in mock_runner.run.call_args_list:
            assert call.kwargs["strategy_name"] == "custom"


# ---------------------------------------------------------------------------
# Embargo period tests
# ---------------------------------------------------------------------------


class TestEmbargoSimple:
    def test_embargo_shifts_test_start(self):
        """Embargo days should create a gap between train_end and test_start."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, embargo_days=10,
        )

        train_call = mock_runner.run.call_args_list[0]
        test_call = mock_runner.run.call_args_list[1]

        train_end = train_call.kwargs["end_date"]
        test_start = test_call.kwargs["start_date"]

        # Gap should be embargo_days + 1 (the original +1 day)
        gap = (test_start - train_end).days
        assert gap == 11  # 1 (original) + 10 (embargo)

    def test_zero_embargo_is_default_behavior(self):
        """embargo_days=0 should match original behavior (1-day gap)."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, embargo_days=0,
        )

        train_call = mock_runner.run.call_args_list[0]
        test_call = mock_runner.run.call_args_list[1]

        train_end = train_call.kwargs["end_date"]
        test_start = test_call.kwargs["start_date"]

        assert (test_start - train_end).days == 1

    def test_window_dates_reflect_embargo(self):
        """The WalkForwardWindow should store the actual test_start with embargo."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, embargo_days=5,
        )

        window = result.windows[0]
        gap = (window.test_start - window.train_end).days
        assert gap == 6  # 1 + 5


class TestEmbargoRolling:
    def test_embargo_shifts_rolling_test_start(self):
        """Embargo should create gap in rolling windows too."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        result = validator.validate_rolling(
            "AAPL",
            date(2018, 1, 1),
            date(2023, 12, 31),
            train_days=730,
            test_days=365,
            step_days=365,
            embargo_days=10,
        )

        assert len(result.windows) >= 1

        for w in result.windows:
            gap = (w.test_start - w.train_end).days
            assert gap == 11  # 1 + 10

    def test_embargo_reduces_available_windows(self):
        """Large embargo on tight date range should produce fewer windows."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)

        # Without embargo: fits 1 window (730 train + 365 test = 1095 days)
        r_no_embargo = validator.validate_rolling(
            "AAPL", date(2020, 1, 1), date(2023, 1, 1),
            train_days=730, test_days=365, step_days=365,
            embargo_days=0,
        )

        mock_runner.reset_mock()

        # With large embargo: may not fit even 1 window
        r_embargo = validator.validate_rolling(
            "AAPL", date(2020, 1, 1), date(2023, 1, 1),
            train_days=730, test_days=365, step_days=365,
            embargo_days=30,
        )

        assert len(r_embargo.windows) <= len(r_no_embargo.windows)


# ---------------------------------------------------------------------------
# Purge period tests
# ---------------------------------------------------------------------------


class TestPurgeSimple:
    def test_purge_trims_training_end(self):
        """Purge days should shorten the training period from the split point."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, purge_days=10,
        )

        train_call = mock_runner.run.call_args_list[0]
        test_call = mock_runner.run.call_args_list[1]

        train_end = train_call.kwargs["end_date"]
        test_start = test_call.kwargs["start_date"]

        # Split point is at 70% of total days
        total_days = (end - start).days
        split_date = start + timedelta(days=int(total_days * 0.7))

        # Train end should be 10 days before split
        assert train_end == split_date - timedelta(days=10)
        # Test start should be 1 day after split (no embargo)
        assert test_start == split_date + timedelta(days=1)

    def test_purge_creates_gap_between_train_and_test(self):
        """Total gap = purge_days + 1 + embargo_days."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, purge_days=10, embargo_days=5,
        )

        window = result.windows[0]
        total_gap = (window.test_start - window.train_end).days
        assert total_gap == 16  # 10 (purge) + 1 (base) + 5 (embargo)

    def test_zero_purge_matches_default(self):
        """purge_days=0 should not affect training end date."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)

        result = validator.validate_simple(
            "AAPL", start, end, purge_days=0, embargo_days=0,
        )

        train_call = mock_runner.run.call_args_list[0]
        test_call = mock_runner.run.call_args_list[1]

        total_days = (end - start).days
        expected_split = start + timedelta(days=int(total_days * 0.7))

        assert train_call.kwargs["end_date"] == expected_split
        assert test_call.kwargs["start_date"] == expected_split + timedelta(days=1)


class TestPurgeRolling:
    def test_purge_trims_rolling_training(self):
        """Purge should shorten training in rolling windows."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        result = validator.validate_rolling(
            "AAPL",
            date(2018, 1, 1),
            date(2023, 12, 31),
            train_days=730,
            test_days=365,
            step_days=365,
            purge_days=10,
        )

        assert len(result.windows) >= 1

        for w in result.windows:
            # Total gap from train_end to test_start = purge + 1 (no embargo)
            gap = (w.test_start - w.train_end).days
            assert gap == 11  # 10 (purge) + 1 (base)

    def test_purge_and_embargo_combined_rolling(self):
        """Both purge and embargo applied in rolling windows."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)
        result = validator.validate_rolling(
            "AAPL",
            date(2018, 1, 1),
            date(2023, 12, 31),
            train_days=730,
            test_days=365,
            step_days=365,
            purge_days=10,
            embargo_days=5,
        )

        assert len(result.windows) >= 1

        for w in result.windows:
            gap = (w.test_start - w.train_end).days
            assert gap == 16  # 10 (purge) + 1 (base) + 5 (embargo)

    def test_large_purge_reduces_windows(self):
        """Large purge on tight range should reduce or eliminate windows."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()

        validator = WalkForwardValidator(mock_runner)

        r_no_purge = validator.validate_rolling(
            "AAPL", date(2020, 1, 1), date(2023, 1, 1),
            train_days=730, test_days=365, step_days=365,
            purge_days=0,
        )

        mock_runner.reset_mock()

        r_purge = validator.validate_rolling(
            "AAPL", date(2020, 1, 1), date(2023, 1, 1),
            train_days=730, test_days=365, step_days=365,
            purge_days=60,
        )

        assert len(r_purge.windows) <= len(r_no_purge.windows)
