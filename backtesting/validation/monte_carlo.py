"""Monte Carlo simulation for backtested strategies.

Shuffles trade ordering to measure path dependency, drawdown distribution,
and risk of ruin. Complements bootstrap (edge significance) by answering:
"Given my actual trades, how bad could it get with unlucky ordering?"
"""

from dataclasses import dataclass

import numpy as np

from ..engine.backtrader_runner import BacktestResult


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""

    symbol: str
    n_simulations: int
    n_trades: int
    initial_cash: float

    # Final equity percentiles
    equity_p5: float
    equity_p25: float
    equity_median: float
    equity_p75: float
    equity_p95: float

    # Max drawdown percentiles (positive percentages, e.g. 18.3 means -18.3%)
    drawdown_median: float
    drawdown_p95: float
    drawdown_worst: float

    # Risk of ruin
    ruin_probability: float
    ruin_threshold: float

    @property
    def survival_rate(self) -> float:
        """Fraction of simulations that stayed above ruin threshold."""
        return 1.0 - self.ruin_probability


def monte_carlo_analysis(
    backtest_result: BacktestResult,
    n_simulations: int = 10000,
    initial_cash: float = None,
    ruin_threshold_pct: float = 0.5,
    random_seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on trade P&L series.

    Randomly permutes the order of trades N times and simulates equity
    curves for each permutation. Measures the distribution of final equity
    and maximum drawdown across all paths.

    Args:
        backtest_result: Standard backtest output with trade list.
        n_simulations: Number of random permutations (default 10,000).
        initial_cash: Starting equity (default: backtest's initial_cash).
        ruin_threshold_pct: Ruin if equity drops below this fraction of
            initial cash (default 0.5 = 50%).
        random_seed: For reproducibility.

    Returns:
        MonteCarloResult with equity/drawdown distributions and ruin probability.

    Raises:
        ValueError: If fewer than 2 trades available.
    """
    rng = np.random.default_rng(random_seed)

    if initial_cash is None:
        initial_cash = backtest_result.initial_cash

    trade_pnl = np.array(
        [
            t["profit_pct"]
            for t in backtest_result.trades
            if t.get("profit_pct") is not None
        ]
    )

    if len(trade_pnl) < 2:
        raise ValueError(
            f"Need at least 2 trades for Monte Carlo, got {len(trade_pnl)}"
        )

    n_trades = len(trade_pnl)
    ruin_threshold = initial_cash * ruin_threshold_pct

    # Convert P&L percentages to return multipliers
    # e.g. +7% → 1.07, -5% → 0.95
    multipliers = 1.0 + trade_pnl / 100.0

    # Generate permutation indices: vectorized shuffle
    # For each simulation row, argsort of random values gives a permutation
    # Shape: (n_simulations, n_trades)
    random_keys = rng.random((n_simulations, n_trades))
    perm_indices = np.argsort(random_keys, axis=1)

    # Apply permutations to get shuffled multiplier sequences
    shuffled = multipliers[perm_indices]

    # Compute cumulative equity curves via cumprod
    # Shape: (n_simulations, n_trades) — each row is one equity path
    equity_curves = initial_cash * np.cumprod(shuffled, axis=1)

    # Final equity for each simulation
    final_equities = equity_curves[:, -1]

    # Max drawdown per simulation
    # Running maximum along each equity path
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns_pct = (running_max - equity_curves) / running_max * 100.0
    max_drawdowns = np.max(drawdowns_pct, axis=1)

    # Also check if equity dipped below ruin threshold at any point
    # Include initial cash as the starting point
    min_equities = np.min(equity_curves, axis=1)
    ruin_count = np.sum(min_equities < ruin_threshold)
    ruin_probability = float(ruin_count / n_simulations)

    # Compute percentiles
    equity_percentiles = np.percentile(final_equities, [5, 25, 50, 75, 95])
    dd_percentiles = np.percentile(max_drawdowns, [50, 95])

    return MonteCarloResult(
        symbol=backtest_result.symbol,
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_cash=initial_cash,
        equity_p5=float(equity_percentiles[0]),
        equity_p25=float(equity_percentiles[1]),
        equity_median=float(equity_percentiles[2]),
        equity_p75=float(equity_percentiles[3]),
        equity_p95=float(equity_percentiles[4]),
        drawdown_median=float(dd_percentiles[0]),
        drawdown_p95=float(dd_percentiles[1]),
        drawdown_worst=float(np.max(max_drawdowns)),
        ruin_probability=ruin_probability,
        ruin_threshold=ruin_threshold,
    )
