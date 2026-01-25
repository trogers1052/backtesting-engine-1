"""
Performance Metrics Calculation

Additional metrics beyond what backtrader provides.
"""

from typing import Dict, List, Optional
import numpy as np


def calculate_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate additional performance metrics from trade list.

    Args:
        trades: List of trade dictionaries with profit_pct field

    Returns:
        Dict with calculated metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_profit": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "profit_factor": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "avg_holding_days": 0,
        }

    # Extract profit percentages
    profits = [t["profit_pct"] for t in trades if t.get("profit_pct") is not None]

    if not profits:
        return {"total_trades": len(trades), "error": "No completed trades"}

    profits = np.array(profits)
    wins = profits[profits > 0]
    losses = profits[profits <= 0]

    # Basic stats
    total_trades = len(profits)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Averages
    avg_profit = float(np.mean(profits))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0

    # Best/Worst
    best_trade = float(np.max(profits))
    worst_trade = float(np.min(profits))

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Consecutive wins/losses
    consecutive_wins = _max_consecutive(profits > 0)
    consecutive_losses = _max_consecutive(profits <= 0)

    # Holding period (if dates available)
    holding_days = []
    for t in trades:
        entry = t.get("entry_date")
        exit_date = t.get("exit_date")
        if entry and exit_date:
            from datetime import datetime
            if isinstance(entry, str):
                entry = datetime.fromisoformat(entry)
            if isinstance(exit_date, str):
                exit_date = datetime.fromisoformat(exit_date)
            days = (exit_date - entry).days
            holding_days.append(days)

    avg_holding_days = np.mean(holding_days) if holding_days else 0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "profit_factor": profit_factor,
        "consecutive_wins": consecutive_wins,
        "consecutive_losses": consecutive_losses,
        "avg_holding_days": float(avg_holding_days),
    }


def _max_consecutive(condition: np.ndarray) -> int:
    """Find maximum consecutive True values in boolean array."""
    if len(condition) == 0:
        return 0

    max_count = 0
    current_count = 0

    for val in condition:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def calculate_risk_metrics(
    equity_curve: List[float],
    risk_free_rate: float = 0.02,
) -> Dict:
    """
    Calculate risk-adjusted metrics from equity curve.

    Args:
        equity_curve: List of portfolio values over time
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Dict with Sharpe, Sortino, Calmar ratios
    """
    if len(equity_curve) < 2:
        return {}

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Assume daily returns, 252 trading days
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    # Sharpe ratio (annualized)
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    # Sortino ratio (only downside deviation)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = float(np.max(drawdown))

    # Calmar ratio (return / max drawdown)
    total_return = (equity[-1] - equity[0]) / equity[0]
    years = len(equity) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "annual_return": float(annual_return),
    }
