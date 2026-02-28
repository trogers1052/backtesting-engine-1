"""Smoke tests for validation report printing functions.

These verify the Rich-based report functions don't crash with valid
and edge-case inputs. They do not assert exact output formatting.
"""

from datetime import date

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.bootstrap import BootstrapResult
from backtesting.validation.regime import RegimeAnalysisResult, RegimeMetrics
from backtesting.validation.report import (
    print_bootstrap_report,
    print_regime_report,
    print_walk_forward_report,
)
from backtesting.validation.walk_forward import WalkForwardResult, WalkForwardWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(sharpe=1.5):
    return BacktestResult(
        symbol="TEST",
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


# ---------------------------------------------------------------------------
# print_walk_forward_report
# ---------------------------------------------------------------------------


class TestPrintWalkForwardReport:
    def test_valid_result(self):
        window = WalkForwardWindow(
            window_num=1,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            train_result=_make_result(sharpe=2.0),
            test_result=_make_result(sharpe=1.5),
        )
        wf_result = WalkForwardResult(symbol="TEST", windows=[window])
        print_walk_forward_report(wf_result)  # Should not raise

    def test_empty_windows(self):
        wf_result = WalkForwardResult(symbol="TEST", windows=[])
        print_walk_forward_report(wf_result)  # Should not raise

    def test_overfit_window(self):
        window = WalkForwardWindow(
            window_num=1,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            train_result=_make_result(sharpe=2.0),
            test_result=_make_result(sharpe=0.3),
        )
        wf_result = WalkForwardResult(symbol="TEST", windows=[window])
        print_walk_forward_report(wf_result)  # Should not raise

    def test_none_sharpe_values(self):
        window = WalkForwardWindow(
            window_num=1,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            train_result=_make_result(sharpe=None),
            test_result=_make_result(sharpe=None),
        )
        wf_result = WalkForwardResult(symbol="TEST", windows=[window])
        print_walk_forward_report(wf_result)  # Should not raise

    def test_negative_train_sharpe(self):
        window = WalkForwardWindow(
            window_num=1,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            train_result=_make_result(sharpe=-0.5),
            test_result=_make_result(sharpe=0.5),
        )
        wf_result = WalkForwardResult(symbol="TEST", windows=[window])
        print_walk_forward_report(wf_result)  # Should not raise


# ---------------------------------------------------------------------------
# print_bootstrap_report
# ---------------------------------------------------------------------------


class TestPrintBootstrapReport:
    def test_significant_result(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=30, n_bootstrap=10000,
            sharpe_point=1.5, sharpe_ci_lower=0.8, sharpe_ci_upper=2.2,
            win_rate_point=0.65, win_rate_ci_lower=0.55, win_rate_ci_upper=0.75,
            p_value=0.001,
        )
        print_bootstrap_report(r)  # Should not raise

    def test_no_edge_result(self):
        r = BootstrapResult(
            symbol="TEST", n_trades=10, n_bootstrap=1000,
            sharpe_point=0.2, sharpe_ci_lower=-0.5, sharpe_ci_upper=0.9,
            win_rate_point=0.52, win_rate_ci_lower=0.35, win_rate_ci_upper=0.70,
            p_value=0.42,
        )
        print_bootstrap_report(r)  # Should not raise

    def test_p_value_displayed(self):
        """Report should handle p_value at significance boundary."""
        r = BootstrapResult(
            symbol="TEST", n_trades=20, n_bootstrap=5000,
            sharpe_point=0.8, sharpe_ci_lower=0.01, sharpe_ci_upper=1.6,
            win_rate_point=0.58, win_rate_ci_lower=0.51, win_rate_ci_upper=0.65,
            p_value=0.049,
        )
        print_bootstrap_report(r)  # Should not raise


# ---------------------------------------------------------------------------
# print_regime_report
# ---------------------------------------------------------------------------


class TestPrintRegimeReport:
    def test_all_three_regimes(self):
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=10, winning_trades=7,
                win_rate=0.7, total_return=30.0, avg_trade_return=3.0,
                sharpe_ratio=1.2, profit_factor=3.5,
            ),
            "bear": RegimeMetrics(
                regime="bear", total_trades=5, winning_trades=1,
                win_rate=0.2, total_return=-8.0, avg_trade_return=-1.6,
                sharpe_ratio=-0.3, profit_factor=0.3,
            ),
            "chop": RegimeMetrics(
                regime="chop", total_trades=8, winning_trades=4,
                win_rate=0.5, total_return=5.0, avg_trade_return=0.625,
                sharpe_ratio=0.4, profit_factor=1.2,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        print_regime_report(r)  # Should not raise

    def test_empty_metrics(self):
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics={})
        print_regime_report(r)  # Should not raise

    def test_infinite_profit_factor(self):
        """Profit factor 999.99 should display as infinity symbol."""
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=5, winning_trades=5,
                win_rate=1.0, total_return=25.0, avg_trade_return=5.0,
                sharpe_ratio=2.0, profit_factor=999.99,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        print_regime_report(r)  # Should not raise

    def test_regime_dependent_warning(self):
        metrics = {
            "bull": RegimeMetrics(
                regime="bull", total_trades=10, winning_trades=8,
                win_rate=0.8, total_return=50.0, avg_trade_return=5.0,
                sharpe_ratio=1.5, profit_factor=4.0,
            ),
            "bear": RegimeMetrics(
                regime="bear", total_trades=3, winning_trades=1,
                win_rate=0.33, total_return=-5.0, avg_trade_return=-1.67,
                sharpe_ratio=-0.5, profit_factor=0.2,
            ),
        }
        r = RegimeAnalysisResult(symbol="TEST", regime_metrics=metrics)
        assert r.regime_dependent is True
        print_regime_report(r)  # Should not raise
