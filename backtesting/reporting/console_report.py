"""
Console Report

Pretty console output for backtest results using Rich.
"""

from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..engine.backtrader_runner import BacktestResult
from ..analyzers.trade_logger import format_trades, format_trade_summary

console = Console()


def print_report(
    result: BacktestResult,
    show_trades: bool = False,
    trade_limit: int = 10,
):
    """
    Print a formatted backtest report to the console.

    Args:
        result: BacktestResult from backtrader runner
        show_trades: Whether to show individual trade details
        trade_limit: Maximum number of trades to show
    """
    # Header
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Backtest Results: {result.symbol}[/bold blue]",
            border_style="blue",
        )
    )

    # Strategy info
    console.print(f"\n[bold]Strategy:[/bold] {result.strategy_name}")
    console.print(f"[bold]Period:[/bold] {result.start_date} to {result.end_date}")
    console.print()

    # Performance table
    perf_table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
    perf_table.add_column("Metric", style="dim")
    perf_table.add_column("Value", justify="right")

    # Color code return
    return_color = "green" if result.total_return >= 0 else "red"
    return_str = f"{result.total_return:+.1%}"

    perf_table.add_row("Initial Capital", f"${result.initial_cash:,.2f}")
    perf_table.add_row("Final Value", f"${result.final_value:,.2f}")
    perf_table.add_row("Total Return", f"[{return_color}]{return_str}[/{return_color}]")
    perf_table.add_row("", "")  # Spacer

    perf_table.add_row("Total Trades", str(result.total_trades))
    perf_table.add_row("Winning Trades", str(result.winning_trades))
    perf_table.add_row("Losing Trades", str(result.losing_trades))

    # Win rate with color
    win_color = "green" if result.win_rate >= 0.5 else "yellow"
    perf_table.add_row("Win Rate", f"[{win_color}]{result.win_rate:.1%}[/{win_color}]")

    if result.avg_win:
        perf_table.add_row("Avg Win", f"[green]+${result.avg_win:,.2f}[/green]")
    if result.avg_loss:
        perf_table.add_row("Avg Loss", f"[red]-${abs(result.avg_loss):,.2f}[/red]")

    perf_table.add_row("", "")  # Spacer

    if result.profit_factor:
        pf_color = "green" if result.profit_factor >= 1.5 else "yellow"
        pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
        perf_table.add_row("Profit Factor", f"[{pf_color}]{pf_str}[/{pf_color}]")

    if result.sharpe_ratio:
        sharpe_color = "green" if result.sharpe_ratio >= 1.0 else "yellow"
        perf_table.add_row("Sharpe Ratio", f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]")

    if result.max_drawdown_pct:
        dd_color = "red" if result.max_drawdown_pct > 20 else "yellow"
        perf_table.add_row("Max Drawdown", f"[{dd_color}]-{result.max_drawdown_pct:.1f}%[/{dd_color}]")

    console.print(perf_table)

    # Trade details
    if show_trades and result.trades:
        console.print()
        console.print("[bold]Recent Trades:[/bold]")
        console.print()

        trade_text = format_trades(result.trades, limit=trade_limit)
        console.print(trade_text)

        # Trade summary
        summary = format_trade_summary(result.trades)
        if summary.get("exit_reasons"):
            console.print("[bold]Exit Reasons:[/bold]")
            for reason, count in summary["exit_reasons"].items():
                console.print(f"  {reason}: {count}")

        if summary.get("rule_contributions"):
            console.print()
            console.print("[bold]Rule Contributions:[/bold]")
            for rule, count in sorted(
                summary["rule_contributions"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                console.print(f"  {rule}: {count} trades")

    console.print()


def print_multi_report(results: Dict[str, BacktestResult]):
    """
    Print a summary report for multiple symbols.

    Args:
        results: Dict mapping symbol to BacktestResult
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]Multi-Symbol Backtest Results[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    # Summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")

    for symbol, result in sorted(results.items()):
        # Color code values
        return_color = "green" if result.total_return >= 0 else "red"
        win_color = "green" if result.win_rate >= 0.5 else "yellow"
        sharpe_str = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A"

        table.add_row(
            symbol,
            str(result.total_trades),
            f"[{win_color}]{result.win_rate:.1%}[/{win_color}]",
            f"[{return_color}]{result.total_return:+.1%}[/{return_color}]",
            sharpe_str,
            f"-{result.max_drawdown_pct:.1f}%" if result.max_drawdown_pct else "N/A",
        )

    console.print(table)

    # Aggregate stats
    total_trades = sum(r.total_trades for r in results.values())
    total_wins = sum(r.winning_trades for r in results.values())
    avg_return = sum(r.total_return for r in results.values()) / len(results)

    console.print()
    console.print(f"[bold]Aggregate:[/bold]")
    console.print(f"  Total Trades: {total_trades}")
    console.print(f"  Overall Win Rate: {total_wins / total_trades:.1%}" if total_trades > 0 else "  No trades")
    console.print(f"  Average Return: {avg_return:+.1%}")
    console.print()
