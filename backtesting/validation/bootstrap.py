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

    # p-value: fraction of bootstrap Sharpe samples <= 0
    p_value: float = 0.0

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
    trades_per_year = 52

    # Vectorized index generation — all bootstrap samples at once
    # Shape: (n_bootstrap, n_trades). Memory: ~2.4MB for 10k x 30
    random_indices = np.random.randint(0, n_trades, size=(n_bootstrap, n_trades))
    resampled_trades = trade_pnl[random_indices]

    # Fully vectorized win rate: fraction of positive trades per sample
    win_rate_samples = np.mean(resampled_trades > 0, axis=1)

    # Fully vectorized Sharpe: compute mean/std across all samples at once
    returns = resampled_trades / 100.0
    mean_returns = np.mean(returns, axis=1)
    std_returns = np.std(returns, ddof=1, axis=1)
    rf_per_trade = risk_free_rate / trades_per_year

    # Replace zero std with 1.0 to avoid division warning; result masked to 0.0
    safe_std = np.where(std_returns == 0, 1.0, std_returns)
    sharpe_samples = np.where(
        std_returns == 0,
        0.0,
        (mean_returns - rf_per_trade) / safe_std * np.sqrt(trades_per_year),
    )

    sharpe_ci = np.percentile(sharpe_samples, [2.5, 97.5])
    wr_ci = np.percentile(win_rate_samples, [2.5, 97.5])

    # p-value: fraction of bootstrap samples where Sharpe <= 0
    # Tests null hypothesis that strategy has no positive risk-adjusted edge
    p_value = float(np.mean(sharpe_samples <= 0))

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
        p_value=p_value,
    )
