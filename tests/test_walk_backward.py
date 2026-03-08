"""Tests for hybrid walk-backward validation."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.walk_backward import (
    DEFAULT_REGIME_WINDOWS,
    RegimeWindow,
    WalkBackwardResult,
    WalkBackwardValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(symbol="TEST", total_return=0.10, sharpe=1.5, total_trades=20, **overrides):
    """Build a minimal BacktestResult."""
    defaults = dict(
        symbol=symbol,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        strategy_name="test",
        initial_cash=100_000,
        final_value=100_000 * (1 + total_return),
        total_return=total_return,
        total_trades=total_trades,
        winning_trades=int(total_trades * 0.6),
        losing_trades=total_trades - int(total_trades * 0.6),
        win_rate=0.60,
        sharpe_ratio=sharpe,
        trades=[],
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _make_regime_window(label="Test Regime", passed=True, total_return=0.05, total_trades=10):
    """Build a RegimeWindow."""
    return RegimeWindow(
        label=label,
        start=date(2022, 1, 1),
        end=date(2022, 12, 31),
        expected_regime="bear",
        actual_dominant_regime="bear",
        result=_make_result(total_return=total_return, total_trades=total_trades),
        passed=passed,
        reason="test",
    )


# ---------------------------------------------------------------------------
# WalkBackwardResult properties
# ---------------------------------------------------------------------------


class TestWalkBackwardResult:
    def test_robust_verdict(self):
        """3+ regimes passed + holdout passed = ROBUST."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=0.05),
            regime_windows=[
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=False),
            ],
            min_regimes_pass=3,
        )
        assert result.overall_verdict == "ROBUST"
        assert result.is_valid is True
        assert result.regimes_passed == 3

    def test_regime_dependent_verdict(self):
        """1-2 regimes passed = REGIME_DEPENDENT."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=0.02),
            regime_windows=[
                _make_regime_window(passed=True),
                _make_regime_window(passed=False),
                _make_regime_window(passed=False),
                _make_regime_window(passed=False),
            ],
            min_regimes_pass=3,
        )
        assert result.overall_verdict == "REGIME_DEPENDENT"
        assert result.is_valid is False

    def test_fragile_no_regimes(self):
        """0 regimes passed = FRAGILE."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=0.02),
            regime_windows=[
                _make_regime_window(passed=False),
                _make_regime_window(passed=False),
            ],
            min_regimes_pass=3,
        )
        assert result.overall_verdict == "FRAGILE"
        assert result.is_valid is False

    def test_fragile_holdout_fails(self):
        """Holdout failure = FRAGILE regardless of regime results."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=-0.15),
            regime_windows=[
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
            ],
            min_regimes_pass=3,
        )
        assert result.overall_verdict == "FRAGILE"
        assert result.holdout_passed is False

    def test_holdout_breakeven_passes(self):
        """Holdout at -1% (threshold) still passes."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=-0.01),
            regime_windows=[
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
            ],
            min_regimes_pass=3,
        )
        assert result.holdout_passed is True
        assert result.overall_verdict == "ROBUST"

    def test_empty_regime_windows(self):
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=0.05),
            regime_windows=[],
            min_regimes_pass=3,
        )
        assert result.regimes_passed == 0
        assert result.regimes_total == 0
        assert result.overall_verdict == "FRAGILE"

    def test_custom_min_regimes(self):
        """With min_regimes_pass=2, only 2 passed should be ROBUST."""
        result = WalkBackwardResult(
            symbol="TEST",
            tune_start=date(2024, 1, 1),
            tune_end=date(2024, 12, 31),
            tune_result=_make_result(),
            holdout_start=date(2025, 1, 1),
            holdout_end=date(2025, 3, 1),
            holdout_result=_make_result(total_return=0.03),
            regime_windows=[
                _make_regime_window(passed=True),
                _make_regime_window(passed=True),
                _make_regime_window(passed=False),
            ],
            min_regimes_pass=2,
        )
        assert result.overall_verdict == "ROBUST"


# ---------------------------------------------------------------------------
# RegimeWindow dataclass
# ---------------------------------------------------------------------------


class TestRegimeWindow:
    def test_properties(self):
        w = _make_regime_window(total_return=0.12, total_trades=15)
        assert w.total_return == 0.12
        assert w.total_trades == 15
        assert w.sharpe == 1.5


# ---------------------------------------------------------------------------
# WalkBackwardValidator.validate
# ---------------------------------------------------------------------------


class TestWalkBackwardValidate:
    def _make_validator(self):
        mock_runner = MagicMock()
        mock_runner.run.return_value = _make_result()
        mock_runner.loader = MagicMock()
        return WalkBackwardValidator(mock_runner), mock_runner

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_basic_flow(self, mock_classifier_cls):
        """Validator runs tune, holdout, and regime windows."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock()
        mock_classifier.get_regimes.return_value.empty = False
        mock_classifier.get_regimes.return_value.__getitem__ = MagicMock(
            return_value=MagicMock(value_counts=MagicMock(return_value=MagicMock(index=["bear"])))
        )
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=date(2026, 3, 1),
            tune_months=12,
            holdout_months=2,
        )

        assert result.symbol == "AAPL"
        assert result.tune_result is not None
        assert result.holdout_result is not None
        # tune + holdout + regime windows
        assert mock_runner.run.call_count >= 2

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_tune_period_dates(self, mock_classifier_cls):
        """Tune period should be tune_months before holdout."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()
        data_end = date(2026, 3, 1)

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=data_end,
            tune_months=12,
            holdout_months=2,
        )

        # Holdout should be last 2 months
        expected_holdout_start = data_end - timedelta(days=60)
        assert result.holdout_end == data_end
        assert result.holdout_start == expected_holdout_start

        # Tune should be 12 months before holdout
        expected_tune_end = expected_holdout_start - timedelta(days=1)
        expected_tune_start = expected_tune_end - timedelta(days=360)
        assert result.tune_end == expected_tune_end
        assert result.tune_start == expected_tune_start

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_tune_clamped_to_data_start(self, mock_classifier_cls):
        """Tune period clamped if data doesn't go back far enough."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()
        data_start = date(2025, 6, 1)  # Very recent start
        data_end = date(2026, 3, 1)

        result = validator.validate(
            symbol="AAPL",
            data_start=data_start,
            data_end=data_end,
            tune_months=12,
            holdout_months=2,
        )

        assert result.tune_start == data_start

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_skips_windows_before_data_start(self, mock_classifier_cls):
        """Regime windows before data_start should be skipped."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        # Data starts in 2023 — should skip 2020, 2021, 2022 windows
        result = validator.validate(
            symbol="AAPL",
            data_start=date(2023, 6, 1),
            data_end=date(2026, 3, 1),
            tune_months=6,
            holdout_months=2,
        )

        # Only 2023-2024 window should remain (others before data_start)
        for w in result.regime_windows:
            assert w.start >= date(2023, 1, 3)

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_run_kwargs_forwarded(self, mock_classifier_cls):
        """Extra kwargs should be forwarded to runner.run()."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=date(2026, 3, 1),
            tune_months=12,
            holdout_months=2,
            profit_target=0.10,
            stop_loss=0.03,
        )

        for call in mock_runner.run.call_args_list:
            assert call.kwargs.get("profit_target") == 0.10
            assert call.kwargs.get("stop_loss") == 0.03

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_custom_regime_windows(self, mock_classifier_cls):
        """Custom regime windows should be used instead of defaults."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        custom_windows = [
            {
                "label": "Custom Bull",
                "start": date(2021, 6, 1),
                "end": date(2021, 12, 31),
                "expected_regime": "bull",
            },
        ]

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=date(2026, 3, 1),
            regime_windows=custom_windows,
        )

        assert len(result.regime_windows) == 1
        assert result.regime_windows[0].label == "Custom Bull"

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_regime_error_handled(self, mock_classifier_cls):
        """Errors running regime backtests should be caught gracefully."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        # First two calls succeed (tune + holdout), then fail
        mock_runner.run.side_effect = [
            _make_result(),  # tune
            _make_result(),  # holdout
            Exception("No data"),  # first regime window
            _make_result(),  # second regime window
            _make_result(),  # third regime window
            _make_result(),  # fourth regime window
        ]

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=date(2026, 3, 1),
        )

        # The failed window should be marked as not passed
        failed = [w for w in result.regime_windows if not w.passed and "Error" in w.reason]
        assert len(failed) >= 1

    @patch("backtesting.validation.walk_backward.RegimeClassifier")
    def test_no_trades_regime_fails(self, mock_classifier_cls):
        """Regime windows with 0 trades should fail even if return >= 0."""
        mock_classifier = MagicMock()
        mock_classifier.get_regimes.return_value = MagicMock(empty=True)
        mock_classifier_cls.return_value = mock_classifier

        validator, mock_runner = self._make_validator()

        no_trade_result = _make_result(total_return=0.0, total_trades=0)
        mock_runner.run.side_effect = [
            _make_result(),  # tune
            _make_result(),  # holdout
            no_trade_result,  # regime window with no trades
        ]

        custom_windows = [
            {"label": "Test", "start": date(2022, 1, 1), "end": date(2022, 6, 1), "expected_regime": "bear"},
        ]

        result = validator.validate(
            symbol="AAPL",
            data_start=date(2019, 1, 1),
            data_end=date(2026, 3, 1),
            regime_windows=custom_windows,
        )

        assert result.regime_windows[0].passed is False
        assert "No trades" in result.regime_windows[0].reason


# ---------------------------------------------------------------------------
# Default regime windows sanity checks
# ---------------------------------------------------------------------------


class TestDefaultRegimeWindows:
    def test_has_four_windows(self):
        assert len(DEFAULT_REGIME_WINDOWS) == 4

    def test_windows_have_required_keys(self):
        for w in DEFAULT_REGIME_WINDOWS:
            assert "label" in w
            assert "start" in w
            assert "end" in w
            assert "expected_regime" in w
            assert isinstance(w["start"], date)
            assert isinstance(w["end"], date)

    def test_windows_start_before_end(self):
        for w in DEFAULT_REGIME_WINDOWS:
            assert w["start"] < w["end"], f"{w['label']}: start >= end"
