"""
JSON Report Export

Export backtest results to JSON format.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Union

from ..engine.backtrader_runner import BacktestResult


def _serialize(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def export_json(
    result: Union[BacktestResult, Dict[str, BacktestResult]],
    output_path: Optional[str] = None,
) -> str:
    """
    Export backtest result(s) to JSON.

    Args:
        result: Single BacktestResult or dict of results
        output_path: Optional file path to write JSON

    Returns:
        JSON string
    """
    if isinstance(result, BacktestResult):
        data = _result_to_dict(result)
    else:
        data = {
            "multi_symbol": True,
            "symbols": list(result.keys()),
            "results": {sym: _result_to_dict(r) for sym, r in result.items()},
            "aggregate": _calculate_aggregate(result),
        }

    json_str = json.dumps(data, indent=2, default=_serialize)

    if output_path:
        Path(output_path).write_text(json_str)

    return json_str


def _result_to_dict(result: BacktestResult) -> Dict:
    """Convert BacktestResult to dictionary."""
    return {
        "symbol": result.symbol,
        "strategy": result.strategy_name,
        "period": {
            "start": result.start_date,
            "end": result.end_date,
        },
        "performance": {
            "initial_cash": result.initial_cash,
            "final_value": result.final_value,
            "total_return": result.total_return,
            "total_return_pct": f"{result.total_return:.2%}",
        },
        "trades": {
            "total": result.total_trades,
            "winners": result.winning_trades,
            "losers": result.losing_trades,
            "win_rate": result.win_rate,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
        },
        "risk_metrics": {
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
        },
        "trade_history": result.trades or [],
    }


def _calculate_aggregate(results: Dict[str, BacktestResult]) -> Dict:
    """Calculate aggregate statistics across multiple symbols."""
    if not results:
        return {}

    total_trades = sum(r.total_trades for r in results.values())
    total_wins = sum(r.winning_trades for r in results.values())
    returns = [r.total_return for r in results.values()]

    return {
        "total_symbols": len(results),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "overall_win_rate": total_wins / total_trades if total_trades > 0 else 0,
        "avg_return": sum(returns) / len(returns) if returns else 0,
        "best_symbol": max(results.items(), key=lambda x: x[1].total_return)[0] if results else None,
        "worst_symbol": min(results.items(), key=lambda x: x[1].total_return)[0] if results else None,
    }
