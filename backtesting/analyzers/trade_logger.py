"""
Trade Logger

Formats trade records for display.
"""

from typing import Dict, List


def format_trades(trades: List[Dict], limit: int = 10) -> str:
    """
    Format trade records for console display.

    Args:
        trades: List of trade dictionaries
        limit: Maximum number of trades to show (0 = all)

    Returns:
        Formatted string for console output
    """
    if not trades:
        return "No trades executed."

    lines = []
    display_trades = trades[-limit:] if limit > 0 else trades

    if limit > 0 and len(trades) > limit:
        lines.append(f"(Showing last {limit} of {len(trades)} trades)\n")

    for i, trade in enumerate(display_trades, 1):
        entry_date = trade.get("entry_date", "N/A")
        if isinstance(entry_date, str) and "T" in entry_date:
            entry_date = entry_date.split("T")[0]

        exit_date = trade.get("exit_date", "N/A")
        if isinstance(exit_date, str) and "T" in exit_date:
            exit_date = exit_date.split("T")[0]

        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        confidence = trade.get("entry_confidence", 0)
        profit_pct = trade.get("profit_pct", 0)
        entry_reason = trade.get("entry_reason", "N/A")
        exit_reason = trade.get("exit_reason", "N/A")
        rules = trade.get("rules_triggered", [])

        # Format profit with color indicator
        if profit_pct and profit_pct > 0:
            profit_str = f"+{profit_pct:.1%}"
        elif profit_pct:
            profit_str = f"{profit_pct:.1%}"
        else:
            profit_str = "N/A"

        lines.append(f"{i}. {entry_date}: BUY @ ${entry_price:.2f} (confidence: {confidence:.2f})")
        lines.append(f"   Reason: {entry_reason}")
        if rules:
            lines.append(f"   Rules: {', '.join(rules)}")
        lines.append(f"   {exit_date}: SELL @ ${exit_price:.2f} ({profit_str}) - {exit_reason}")
        lines.append("")

    return "\n".join(lines)


def format_trade_summary(trades: List[Dict]) -> Dict:
    """
    Create summary statistics from trades.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dict with summary statistics
    """
    if not trades:
        return {}

    profits = [t.get("profit_pct", 0) for t in trades if t.get("profit_pct") is not None]

    if not profits:
        return {}

    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    # Count exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade.get("exit_reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Count rules triggered
    rule_counts = {}
    for trade in trades:
        for rule in trade.get("rules_triggered", []):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    return {
        "total_trades": len(profits),
        "winners": len(wins),
        "losers": len(losses),
        "win_rate": len(wins) / len(profits) if profits else 0,
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "best_trade": max(profits) if profits else 0,
        "worst_trade": min(profits) if profits else 0,
        "exit_reasons": exit_reasons,
        "rule_contributions": rule_counts,
    }
