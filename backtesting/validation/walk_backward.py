"""Hybrid walk-backward validation: tune on recent data, validate against historical regimes.

Philosophy: Traditional walk-forward tunes on old data and tests on new data.
This is backwards for active traders — you want rules tuned to the CURRENT
market structure, then validated that they don't blow up in other regimes.

Approach:
  1. TUNE on recent period (e.g., last 6-18 months)
  2. HOLDOUT the most recent 1-2 months (unseen recent data)
  3. WALK BACKWARD across labeled historical regimes
  4. VERDICT: rules must be profitable in at least N of M regimes
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..engine.backtrader_runner import BacktraderRunner, BacktestResult
from ..indicators.pandas_ta_bridge import get_required_warmup_bars
from .regime import RegimeClassifier

logger = logging.getLogger(__name__)


# Pre-defined historical regime windows (approximate calendar boundaries).
# These are the major regime shifts in recent US equity market history.
# Users can override with custom windows via the API.
DEFAULT_REGIME_WINDOWS = [
    {
        "label": "2022 Bear (Rate Hike)",
        "start": date(2022, 1, 3),
        "end": date(2022, 10, 14),
        "expected_regime": "bear",
    },
    {
        "label": "2021 Bull (Low Rate Momentum)",
        "start": date(2021, 1, 4),
        "end": date(2021, 12, 31),
        "expected_regime": "bull",
    },
    {
        "label": "2020 Crash + Recovery",
        "start": date(2020, 2, 19),
        "end": date(2020, 12, 31),
        "expected_regime": "crisis",
    },
    {
        "label": "2023-2024 Choppy Transition",
        "start": date(2023, 1, 3),
        "end": date(2024, 6, 30),
        "expected_regime": "chop",
    },
]


@dataclass
class RegimeWindow:
    """Result of testing rules against one historical regime."""

    label: str
    start: date
    end: date
    expected_regime: str
    actual_dominant_regime: str
    result: BacktestResult
    passed: bool
    reason: str

    @property
    def total_return(self) -> float:
        return self.result.total_return

    @property
    def sharpe(self) -> Optional[float]:
        return self.result.sharpe_ratio

    @property
    def total_trades(self) -> int:
        return self.result.total_trades


@dataclass
class WalkBackwardResult:
    """Complete hybrid walk-backward validation result."""

    symbol: str

    # Tune period
    tune_start: date
    tune_end: date
    tune_result: BacktestResult

    # Holdout period (most recent unseen data)
    holdout_start: date
    holdout_end: date
    holdout_result: BacktestResult

    # Historical regime windows walked backward
    regime_windows: List[RegimeWindow] = field(default_factory=list)

    # Pass/fail thresholds
    min_regimes_pass: int = 3

    @property
    def regimes_passed(self) -> int:
        return sum(1 for w in self.regime_windows if w.passed)

    @property
    def regimes_total(self) -> int:
        return len(self.regime_windows)

    @property
    def holdout_passed(self) -> bool:
        """Holdout passes if profitable or at least breakeven."""
        return self.holdout_result.total_return >= -0.01

    @property
    def overall_verdict(self) -> str:
        """ROBUST, FRAGILE, or REGIME_DEPENDENT."""
        if not self.holdout_passed:
            return "FRAGILE"
        if self.regimes_passed >= self.min_regimes_pass:
            return "ROBUST"
        if self.regimes_passed >= 1:
            return "REGIME_DEPENDENT"
        return "FRAGILE"

    @property
    def is_valid(self) -> bool:
        return self.overall_verdict == "ROBUST"


class WalkBackwardValidator:
    """Hybrid walk-backward validation: tune recent, validate historical."""

    def __init__(self, runner: BacktraderRunner):
        self.runner = runner

    def validate(
        self,
        symbol: str,
        data_start: date,
        data_end: date,
        tune_months: int = 12,
        holdout_months: int = 2,
        min_regimes_pass: int = 3,
        regime_windows: Optional[List[Dict]] = None,
        regime_pass_threshold: float = -0.01,
        **run_kwargs,
    ) -> WalkBackwardResult:
        """Run hybrid walk-backward validation.

        Args:
            symbol: Stock ticker.
            data_start: Earliest available data date.
            data_end: Latest available data date (usually today).
            tune_months: Months of recent data to tune on (default 12).
            holdout_months: Most recent months to hold out (default 2).
            min_regimes_pass: Minimum regime windows that must pass (default 3).
            regime_windows: Custom regime windows (default: 2020-2024 majors).
            regime_pass_threshold: Min total_return to pass a regime (default -0.01).
            **run_kwargs: Passed through to BacktraderRunner.run().

        Returns:
            WalkBackwardResult with tune, holdout, and per-regime results.
        """
        windows = regime_windows or DEFAULT_REGIME_WINDOWS

        # Calculate period boundaries working backward from data_end
        holdout_end = data_end
        holdout_start = data_end - timedelta(days=holdout_months * 30)
        tune_end = holdout_start - timedelta(days=1)
        tune_start = tune_end - timedelta(days=tune_months * 30)

        if tune_start < data_start:
            tune_start = data_start
            logger.warning(
                f"Tune period clamped to data_start {data_start} "
                f"(requested {tune_months} months but only "
                f"{(tune_end - data_start).days // 30} available)"
            )

        warmup_bars = get_required_warmup_bars()
        warmup_calendar_days = int(warmup_bars * 1.5)

        logger.info(
            f"Walk-backward validation for {symbol}:\n"
            f"  Tune:    {tune_start} to {tune_end} ({(tune_end - tune_start).days} days)\n"
            f"  Holdout: {holdout_start} to {holdout_end} ({(holdout_end - holdout_start).days} days)\n"
            f"  Regime windows: {len(windows)}"
        )

        # 1. TUNE: Run backtest on the tuning period
        logger.info(f"[1/3] Running TUNE period: {tune_start} to {tune_end}")
        tune_result = self._safe_run(
            symbol=symbol,
            start_date=tune_start,
            end_date=tune_end,
            **run_kwargs,
        )
        logger.info(
            f"  Tune result: {tune_result.total_return:.1%} return, "
            f"{tune_result.total_trades} trades, "
            f"Sharpe {tune_result.sharpe_ratio or 0:.2f}"
        )

        # 2. HOLDOUT: Run on the most recent unseen data
        logger.info(f"[2/3] Running HOLDOUT period: {holdout_start} to {holdout_end}")
        holdout_warmup = holdout_start - timedelta(days=warmup_calendar_days)
        holdout_result = self._safe_run(
            symbol=symbol,
            start_date=holdout_start,
            end_date=holdout_end,
            warmup_start=holdout_warmup,
            **run_kwargs,
        )
        logger.info(
            f"  Holdout result: {holdout_result.total_return:.1%} return, "
            f"{holdout_result.total_trades} trades"
        )

        # 3. WALK BACKWARD: Test against each historical regime window
        logger.info(f"[3/3] Walking backward across {len(windows)} regime windows")
        regime_results = []

        # Initialize regime classifier for dominant regime detection
        classifier = RegimeClassifier(self.runner.loader)

        for i, window in enumerate(windows):
            w_start = window["start"]
            w_end = window["end"]
            w_label = window["label"]
            w_expected = window.get("expected_regime", "unknown")

            # Skip windows that fall outside available data
            if w_start < data_start:
                logger.warning(
                    f"  Skipping '{w_label}': starts {w_start} before data_start {data_start}"
                )
                continue
            if w_end > tune_start:
                # Adjust window end to not overlap with tune period
                w_end = tune_start - timedelta(days=1)
                if w_end <= w_start:
                    logger.warning(
                        f"  Skipping '{w_label}': overlaps with tune period"
                    )
                    continue

            logger.info(f"  Regime {i+1}/{len(windows)}: {w_label} ({w_start} to {w_end})")

            warmup_start = w_start - timedelta(days=warmup_calendar_days)
            try:
                regime_result = self._safe_run(
                    symbol=symbol,
                    start_date=w_start,
                    end_date=w_end,
                    warmup_start=warmup_start,
                    **run_kwargs,
                )
            except Exception as e:
                logger.error(f"  Failed to run regime '{w_label}': {e}")
                regime_results.append(
                    RegimeWindow(
                        label=w_label,
                        start=w_start,
                        end=w_end,
                        expected_regime=w_expected,
                        actual_dominant_regime="error",
                        result=BacktestResult(
                            symbol=symbol,
                            start_date=w_start,
                            end_date=w_end,
                            strategy_name="error",
                            initial_cash=0,
                            final_value=0,
                            total_return=-1.0,
                            total_trades=0,
                            winning_trades=0,
                            losing_trades=0,
                            win_rate=0.0,
                            trades=[],
                        ),
                        passed=False,
                        reason=f"Error: {e}",
                    )
                )
                continue

            # Determine actual dominant regime via SPY classification
            actual_regime = self._get_dominant_regime(classifier, w_start, w_end)

            # Determine pass/fail
            passed = regime_result.total_return >= regime_pass_threshold
            if passed and regime_result.total_trades == 0:
                passed = False
                reason = "No trades generated"
            elif passed:
                reason = f"Return {regime_result.total_return:.1%} >= threshold"
            else:
                reason = f"Return {regime_result.total_return:.1%} below threshold"

            logger.info(
                f"    Result: {regime_result.total_return:.1%}, "
                f"{regime_result.total_trades} trades, "
                f"{'PASS' if passed else 'FAIL'}"
            )

            regime_results.append(
                RegimeWindow(
                    label=w_label,
                    start=w_start,
                    end=w_end,
                    expected_regime=w_expected,
                    actual_dominant_regime=actual_regime,
                    result=regime_result,
                    passed=passed,
                    reason=reason,
                )
            )

        result = WalkBackwardResult(
            symbol=symbol,
            tune_start=tune_start,
            tune_end=tune_end,
            tune_result=tune_result,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            holdout_result=holdout_result,
            regime_windows=regime_results,
            min_regimes_pass=min_regimes_pass,
        )

        logger.info(
            f"\nVerdict: {result.overall_verdict} "
            f"({result.regimes_passed}/{result.regimes_total} regimes passed, "
            f"holdout {'PASS' if result.holdout_passed else 'FAIL'})"
        )

        return result

    def _safe_run(self, symbol: str, start_date: date, end_date: date, **kwargs) -> BacktestResult:
        """Run backtest with fallback for analyzer errors (e.g. SharpeRatio division by zero)."""
        try:
            return self.runner.run(symbol=symbol, start_date=start_date, end_date=end_date, **kwargs)
        except (ZeroDivisionError, FloatingPointError) as e:
            logger.warning(f"Backtest analyzer error for {symbol} {start_date}-{end_date}: {e}")
            return BacktestResult(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy_name="error",
                initial_cash=self.runner.initial_cash,
                final_value=self.runner.initial_cash,
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                trades=[],
            )

    def _get_dominant_regime(
        self, classifier: RegimeClassifier, start: date, end: date
    ) -> str:
        """Get the most common regime label for a date range."""
        try:
            regime_df = classifier.get_regimes(start, end)
            if regime_df.empty:
                return "unknown"
            counts = regime_df["regime"].value_counts()
            return counts.index[0]
        except Exception:
            return "unknown"
