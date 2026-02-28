"""Rich console reports for validation results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .bootstrap import BootstrapResult
from .monte_carlo import MonteCarloResult
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

    p_color = "green" if bootstrap_result.p_value < 0.05 else "red"
    console.print(
        f"  Trades: {bootstrap_result.n_trades}  |  "
        f"Samples: {bootstrap_result.n_bootstrap:,}  |  "
        f"p-value: [{p_color}]{bootstrap_result.p_value:.4f}[/{p_color}]"
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

    for regime_name in ["bull", "bear", "chop", "volatile", "crisis"]:
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


def print_monte_carlo_report(mc_result: MonteCarloResult):
    """Print Monte Carlo simulation results."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold blue]Monte Carlo Simulation: {mc_result.symbol} "
            f"({mc_result.n_simulations:,} paths)[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    console.print(
        f"  Trades: {mc_result.n_trades}  |  "
        f"Initial: ${mc_result.initial_cash:,.2f}  |  "
        f"Simulations: {mc_result.n_simulations:,}"
    )
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # Final equity distribution
    table.add_row("Final Equity", "", end_section=True)
    table.add_row("  5th percentile", f"${mc_result.equity_p5:,.2f}")
    table.add_row("  25th percentile", f"${mc_result.equity_p25:,.2f}")
    table.add_row(
        "  Median",
        f"[bold]${mc_result.equity_median:,.2f}[/bold]",
    )
    table.add_row("  75th percentile", f"${mc_result.equity_p75:,.2f}")
    table.add_row(
        "  95th percentile",
        f"${mc_result.equity_p95:,.2f}",
        end_section=True,
    )

    # Drawdown distribution
    table.add_row("Max Drawdown", "", end_section=True)
    dd_med_color = "green" if mc_result.drawdown_median < 20 else "yellow"
    dd_95_color = "yellow" if mc_result.drawdown_p95 < 35 else "red"
    dd_worst_color = "yellow" if mc_result.drawdown_worst < 50 else "red"
    table.add_row(
        "  Median",
        f"[{dd_med_color}]-{mc_result.drawdown_median:.1f}%[/{dd_med_color}]",
    )
    table.add_row(
        "  95th percentile",
        f"[{dd_95_color}]-{mc_result.drawdown_p95:.1f}%[/{dd_95_color}]",
    )
    table.add_row(
        "  Worst",
        f"[{dd_worst_color}]-{mc_result.drawdown_worst:.1f}%[/{dd_worst_color}]",
        end_section=True,
    )

    # Risk of ruin
    ruin_pct = mc_result.ruin_probability * 100
    if ruin_pct < 5:
        ruin_color = "green"
    elif ruin_pct < 15:
        ruin_color = "yellow"
    else:
        ruin_color = "red"

    table.add_row("Risk of Ruin", "", end_section=True)
    table.add_row(
        f"  P(equity < ${mc_result.ruin_threshold:,.0f})",
        f"[{ruin_color}]{ruin_pct:.1f}%[/{ruin_color}]",
    )
    table.add_row(
        "  Survival rate",
        f"[{ruin_color}]{mc_result.survival_rate:.1%}[/{ruin_color}]",
    )

    console.print(table)

    if ruin_pct < 5:
        console.print(
            f"\n[bold green]Strategy survives in {mc_result.survival_rate:.1%} "
            f"of trade orderings[/bold green]"
        )
    elif ruin_pct < 15:
        console.print(
            f"\n[bold yellow]WARNING: {ruin_pct:.1f}% ruin probability — "
            f"consider tighter stops or smaller position sizes[/bold yellow]"
        )
    else:
        console.print(
            f"\n[bold red]DANGER: {ruin_pct:.1f}% ruin probability — "
            f"strategy is not safe for this account size[/bold red]"
        )

    console.print()
