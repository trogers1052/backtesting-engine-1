"""Walk-forward validation for detecting overfit backtested parameters."""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List

from ..engine.backtrader_runner import BacktraderRunner, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test window with results."""

    window_num: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_result: BacktestResult
    test_result: BacktestResult

    @property
    def is_overfit(self) -> bool:
        """Test Sharpe < 50% of train Sharpe indicates overfit parameters."""
        train_sharpe = self.train_result.sharpe_ratio
        test_sharpe = self.test_result.sharpe_ratio

        # No Sharpe data = can't validate
        if train_sharpe is None or test_sharpe is None:
            return True

        # Train Sharpe <= 0 means no edge to begin with
        if train_sharpe <= 0:
            return True

        return test_sharpe < 0.5 * train_sharpe


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis across all windows."""

    symbol: str
    windows: List[WalkForwardWindow] = field(default_factory=list)

    @property
    def overall_overfit(self) -> bool:
        """Any window shows overfit."""
        return any(w.is_overfit for w in self.windows)

    @property
    def overfit_count(self) -> int:
        return sum(1 for w in self.windows if w.is_overfit)


class WalkForwardValidator:
    """Run walk-forward validation using the existing BacktraderRunner."""

    def __init__(self, runner: BacktraderRunner):
        self.runner = runner

    def validate_simple(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        train_pct: float = 0.7,
        **run_kwargs,
    ) -> WalkForwardResult:
        """Single train/test split by date.

        Args:
            symbol: Stock ticker.
            start_date: Full period start.
            end_date: Full period end.
            train_pct: Fraction of date range for training (default 0.7).
            **run_kwargs: Passed through to BacktraderRunner.run().

        Returns:
            WalkForwardResult with one window.
        """
        total_days = (end_date - start_date).days
        train_end = start_date + timedelta(days=int(total_days * train_pct))
        test_start = train_end + timedelta(days=1)

        logger.info(
            f"Walk-forward: train {start_date} to {train_end}, "
            f"test {test_start} to {end_date}"
        )

        train_result = self.runner.run(
            symbol=symbol,
            start_date=start_date,
            end_date=train_end,
            **run_kwargs,
        )

        test_result = self.runner.run(
            symbol=symbol,
            start_date=test_start,
            end_date=end_date,
            **run_kwargs,
        )

        window = WalkForwardWindow(
            window_num=1,
            train_start=start_date,
            train_end=train_end,
            test_start=test_start,
            test_end=end_date,
            train_result=train_result,
            test_result=test_result,
        )

        return WalkForwardResult(symbol=symbol, windows=[window])

    def validate_rolling(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        train_days: int = 730,
        test_days: int = 365,
        step_days: int = 365,
        **run_kwargs,
    ) -> WalkForwardResult:
        """Rolling walk-forward with multiple overlapping windows.

        Args:
            symbol: Stock ticker.
            start_date: Full period start.
            end_date: Full period end.
            train_days: Training window length in calendar days.
            test_days: Test window length in calendar days.
            step_days: Step forward between windows.
            **run_kwargs: Passed through to BacktraderRunner.run().

        Returns:
            WalkForwardResult with multiple windows.
        """
        windows = []
        window_num = 1
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)

            if test_end > end_date:
                break

            logger.info(
                f"Walk-forward window #{window_num}: "
                f"train {current_start} to {train_end}, "
                f"test {test_start} to {test_end}"
            )

            train_result = self.runner.run(
                symbol=symbol,
                start_date=current_start,
                end_date=train_end,
                **run_kwargs,
            )

            test_result = self.runner.run(
                symbol=symbol,
                start_date=test_start,
                end_date=test_end,
                **run_kwargs,
            )

            windows.append(
                WalkForwardWindow(
                    window_num=window_num,
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_result=train_result,
                    test_result=test_result,
                )
            )

            window_num += 1
            current_start += timedelta(days=step_days)

        return WalkForwardResult(symbol=symbol, windows=windows)
