"""Bootstrap statistical significance tests for backtested strategies."""

from dataclasses import dataclass

import numpy as np

from ..engine.backtrader_runner import BacktestResult


@dataclass
class BootstrapResult:
    """Bootstrap analysis results with confidence intervals."""

    symbol: str
    n_trades: int
    n_bootstrap: int

    # Sharpe ratio
    sharpe_point: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    # Win rate
    win_rate_point: float
    win_rate_ci_lower: float
    win_rate_ci_upper: float

    @property
    def no_edge_sharpe(self) -> bool:
        """95% CI includes 0 — no evidence of positive risk-adjusted return."""
        return self.sharpe_ci_lower <= 0

    @property
    def no_edge_wr(self) -> bool:
        """95% CI includes 50% — no evidence of better-than-coin-flip win rate."""
        return self.win_rate_ci_lower <= 0.5


def calculate_trade_sharpe(
    trade_pnl_pct: np.ndarray, risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio from discrete trade returns.

    Uses trade-based Sharpe (not equity curve) which better captures
    strategy edge for infrequent traders (15-40 trades over 5 years).

    Annualizes assuming ~52 trades/year (conservative weekly estimate).
    """
    if len(trade_pnl_pct) < 2:
        return 0.0

    returns = trade_pnl_pct / 100.0
    trades_per_year = 52

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate / trades_per_year) / std_return * np.sqrt(
        trades_per_year
    )
    return float(sharpe)


def bootstrap_analysis(
    backtest_result: BacktestResult,
    n_bootstrap: int = 10000,
    risk_free_rate: float = 0.02,
    random_seed: int = 42,
) -> BootstrapResult:
    """Run bootstrap analysis on trade P&L series.

    Resamples the trade returns with replacement N times, computing
    Sharpe ratio and win rate for each sample to build confidence intervals.

    Optimized for Raspberry Pi with vectorized numpy operations.

    Args:
        backtest_result: Standard backtest output with trade list.
        n_bootstrap: Number of bootstrap samples (default 10,000).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        random_seed: For reproducibility.

    Returns:
        BootstrapResult with point estimates and 95% confidence intervals.

    Raises:
        ValueError: If fewer than 2 trades available.
    """
    np.random.seed(random_seed)

    trade_pnl = np.array(
        [
            t["profit_pct"]
            for t in backtest_result.trades
            if t.get("profit_pct") is not None
        ]
    )

    if len(trade_pnl) < 2:
        raise ValueError(f"Need at least 2 trades for bootstrap, got {len(trade_pnl)}")

    n_trades = len(trade_pnl)

    # Vectorized index generation — all bootstrap samples at once
    # Shape: (n_bootstrap, n_trades). Memory: ~2.4MB for 10k x 30
    random_indices = np.random.randint(0, n_trades, size=(n_bootstrap, n_trades))
    resampled_trades = trade_pnl[random_indices]

    sharpe_samples = np.zeros(n_bootstrap)
    win_rate_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = resampled_trades[i]
        sharpe_samples[i] = calculate_trade_sharpe(sample, risk_free_rate)
        win_rate_samples[i] = np.mean(sample > 0)

    sharpe_ci = np.percentile(sharpe_samples, [2.5, 97.5])
    wr_ci = np.percentile(win_rate_samples, [2.5, 97.5])

    return BootstrapResult(
        symbol=backtest_result.symbol,
        n_trades=n_trades,
        n_bootstrap=n_bootstrap,
        sharpe_point=calculate_trade_sharpe(trade_pnl, risk_free_rate),
        sharpe_ci_lower=float(sharpe_ci[0]),
        sharpe_ci_upper=float(sharpe_ci[1]),
        win_rate_point=float(np.mean(trade_pnl > 0)),
        win_rate_ci_lower=float(wr_ci[0]),
        win_rate_ci_upper=float(wr_ci[1]),
    )
