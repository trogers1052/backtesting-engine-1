"""Rich console reports for validation results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .bootstrap import BootstrapResult
from .regime import RegimeAnalysisResult
from .walk_forward import WalkForwardResult

console = Console()


def print_walk_forward_report(wf_result: WalkForwardResult):
    """Print walk-forward validation results."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Walk-Forward Validation: {wf_result.symbol}[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Window", style="bold")
    table.add_column("Train Period")
    table.add_column("Test Period")
    table.add_column("Train Sharpe", justify="right")
    table.add_column("Test Sharpe", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Status", justify="center")

    for w in wf_result.windows:
        train_sharpe = w.train_result.sharpe_ratio or 0.0
        test_sharpe = w.test_result.sharpe_ratio or 0.0

        if train_sharpe > 0:
            ratio = test_sharpe / train_sharpe
            ratio_str = f"{ratio:.0%}"
        else:
            ratio_str = "N/A"

        status = "[red]OVERFIT[/red]" if w.is_overfit else "[green]VALID[/green]"

        table.add_row(
            f"#{w.window_num}",
            f"{w.train_start} to {w.train_end}",
            f"{w.test_start} to {w.test_end}",
            f"{train_sharpe:.2f}",
            f"{test_sharpe:.2f}",
            ratio_str,
            status,
        )

    console.print(table)

    if wf_result.overall_overfit:
        console.print(
            f"\n[bold red]WARNING: {wf_result.overfit_count}/{len(wf_result.windows)} "
            f"windows show overfit parameters[/bold red]"
        )
    else:
        console.print("\n[bold green]All windows pass walk-forward validation[/bold green]")

    console.print()


def print_bootstrap_report(bootstrap_result: BootstrapResult):
    """Print bootstrap significance test results."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Bootstrap Significance: {bootstrap_result.symbol}[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    console.print(
        f"  Trades: {bootstrap_result.n_trades}  |  "
        f"Samples: {bootstrap_result.n_bootstrap:,}"
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Point Est.", justify="right")
    table.add_column("95% CI Lower", justify="right")
    table.add_column("95% CI Upper", justify="right")
    table.add_column("Status", justify="center")

    # Sharpe row
    sharpe_status = (
        "[red]NO EDGE[/red]" if bootstrap_result.no_edge_sharpe
        else "[green]SIGNIFICANT[/green]"
    )
    table.add_row(
        "Sharpe Ratio",
        f"{bootstrap_result.sharpe_point:.2f}",
        f"{bootstrap_result.sharpe_ci_lower:.2f}",
        f"{bootstrap_result.sharpe_ci_upper:.2f}",
        sharpe_status,
    )

    # Win rate row
    wr_status = (
        "[red]NO EDGE[/red]" if bootstrap_result.no_edge_wr
        else "[green]SIGNIFICANT[/green]"
    )
    table.add_row(
        "Win Rate",
        f"{bootstrap_result.win_rate_point:.1%}",
        f"{bootstrap_result.win_rate_ci_lower:.1%}",
        f"{bootstrap_result.win_rate_ci_upper:.1%}",
        wr_status,
    )

    console.print(table)

    if bootstrap_result.no_edge_sharpe:
        console.print(
            "\n[bold red]WARNING: Sharpe 95% CI includes zero — "
            "no evidence of positive risk-adjusted return[/bold red]"
        )
    if bootstrap_result.no_edge_wr:
        console.print(
            "\n[bold red]WARNING: Win rate 95% CI includes 50% — "
            "no evidence of better-than-coin-flip performance[/bold red]"
        )
    if not bootstrap_result.no_edge_sharpe and not bootstrap_result.no_edge_wr:
        console.print(
            "\n[bold green]Both Sharpe and win rate are statistically significant[/bold green]"
        )

    console.print()


def print_regime_report(regime_result: RegimeAnalysisResult):
    """Print regime-stratified analysis results."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Regime Analysis: {regime_result.symbol}[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Regime", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Profit Factor", justify="right")

    for regime_name in ["bull", "bear", "chop"]:
        if regime_name not in regime_result.regime_metrics:
            table.add_row(regime_name.upper(), "0", "-", "-", "-", "-", "-")
            continue

        m = regime_result.regime_metrics[regime_name]
        wr_color = "green" if m.win_rate >= 0.5 else "yellow"
        ret_color = "green" if m.avg_trade_return > 0 else "red"
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 999 else "∞"

        table.add_row(
            regime_name.upper(),
            str(m.total_trades),
            f"[{wr_color}]{m.win_rate:.1%}[/{wr_color}]",
            f"[{ret_color}]{m.avg_trade_return:+.2f}%[/{ret_color}]",
            f"[{ret_color}]{m.total_return:+.1f}%[/{ret_color}]",
            f"{m.sharpe_ratio:.2f}",
            pf_str,
        )

    console.print(table)

    if regime_result.regime_dependent:
        console.print(
            "\n[bold red]WARNING: Strategy is regime-dependent — "
            ">70% of profit comes from one regime[/bold red]"
        )
    else:
        console.print(
            "\n[bold green]Profit distributed across regimes[/bold green]"
        )

    console.print()
