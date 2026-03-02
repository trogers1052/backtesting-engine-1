#!/usr/bin/env python3
"""
CCJ Validate-Then-Tune Optimization

Validates the current CCJ baseline rules through 4 statistical gates,
diagnoses any failures, and makes targeted parameter adjustments.

Usage:
    python validate_ccj.py
"""

import logging
import os
import sys
import time
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ensure decision-engine is importable
de_path = os.path.expanduser("~/Projects/decision-engine")
if de_path not in sys.path:
    sys.path.insert(0, de_path)
os.environ.setdefault("DECISION_ENGINE_PATH", de_path)

from backtesting.engine import BacktraderRunner
from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.walk_forward import WalkForwardValidator, WalkForwardResult
from backtesting.validation.bootstrap import bootstrap_analysis, BootstrapResult
from backtesting.validation.monte_carlo import monte_carlo_analysis, MonteCarloResult
from backtesting.validation.regime import analyze_by_regime, RegimeAnalysisResult
from backtesting.validation.report import (
    print_walk_forward_report,
    print_bootstrap_report,
    print_monte_carlo_report,
    print_regime_report,
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# ============================================================================
# Configuration
# ============================================================================

SYMBOL = "CCJ"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 2, 28)
INITIAL_CASH = 1000

# Current baseline from decision-engine rules.yaml
BASELINE = dict(
    rules=["trend_continuation", "seasonality", "death_cross"],
    min_confidence=0.65,
    profit_target=0.10,
    stop_loss=1.0,
    max_loss_pct=6.0,
    cooldown_bars=7,
    timeframe="daily",
    exit_timeframe="5min",
)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class GateResult:
    """Result of a single validation gate."""

    name: str
    passed: bool
    detail: str
    data: object = None


@dataclass
class ValidationReport:
    """Full validation report for one config."""

    label: str
    kwargs: Dict
    backtest: Optional[BacktestResult] = None
    gates: List[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def pass_count(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def failed_gates(self) -> List[str]:
        return [g.name for g in self.gates if not g.passed]


# ============================================================================
# Validation runner
# ============================================================================


def run_full_validation(
    runner: BacktraderRunner,
    label: str,
    run_kwargs: Dict,
    print_reports: bool = True,
) -> ValidationReport:
    """Run all 4 validation gates on a config. Returns ValidationReport."""
    report = ValidationReport(label=label, kwargs=run_kwargs.copy())

    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit(f"[bold]Validating: {label}[/bold]", border_style="cyan"))

    # --- Full-period backtest (needed for bootstrap, MC, regime) ---
    console.print("  Running full-period backtest...")
    t0 = time.time()
    try:
        result = runner.run(
            symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE, **run_kwargs
        )
    except Exception as e:
        console.print(f"  [red]Backtest failed: {e}[/red]")
        report.gates = [
            GateResult(g, False, f"Backtest failed: {e}")
            for g in ["Walk-Forward", "Bootstrap", "Monte Carlo", "Regime"]
        ]
        return report

    elapsed = time.time() - t0
    report.backtest = result
    console.print(
        f"  Done in {elapsed:.0f}s — {result.total_trades} trades, "
        f"WR={result.win_rate:.1%}, Return={result.total_return:+.1%}, "
        f"Sharpe={result.sharpe_ratio or 0:.2f}, "
        f"PF={result.profit_factor or 0:.2f}, "
        f"DD=-{result.max_drawdown_pct or 0:.1f}%"
    )

    if result.total_trades < 2:
        console.print("  [red]Too few trades (<2) — all gates fail[/red]")
        report.gates = [
            GateResult(g, False, "Too few trades")
            for g in ["Walk-Forward", "Bootstrap", "Monte Carlo", "Regime"]
        ]
        return report

    # --- Gate 1: Walk-Forward ---
    console.print("\n  [cyan]Gate 1: Walk-Forward Validation[/cyan]")
    t0 = time.time()
    try:
        validator = WalkForwardValidator(runner)
        wf_result = validator.validate_simple(
            SYMBOL,
            START_DATE,
            END_DATE,
            train_pct=0.7,
            embargo_days=5,
            purge_days=10,
            **run_kwargs,
        )
        elapsed = time.time() - t0

        wf_passed = not wf_result.overall_overfit
        if wf_result.windows:
            w = wf_result.windows[0]
            train_s = w.train_result.sharpe_ratio or 0
            test_s = w.test_result.sharpe_ratio or 0
            ratio = test_s / train_s if train_s > 0 else 0
            wf_detail = (
                f"Train Sharpe={train_s:.2f}, Test Sharpe={test_s:.2f}, "
                f"Ratio={ratio:.0%} (need >=50%)"
            )
        else:
            wf_detail = "No windows generated"
            wf_passed = False

        report.gates.append(GateResult("Walk-Forward", wf_passed, wf_detail, wf_result))
        status = "[green]PASS[/green]" if wf_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {wf_detail} ({elapsed:.0f}s)")
        if print_reports:
            print_walk_forward_report(wf_result)
    except Exception as e:
        report.gates.append(GateResult("Walk-Forward", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # --- Gate 2: Bootstrap ---
    console.print("\n  [cyan]Gate 2: Bootstrap Significance (10,000 samples)[/cyan]")
    t0 = time.time()
    try:
        bs_result = bootstrap_analysis(result, n_bootstrap=10000)
        elapsed = time.time() - t0

        bs_passed = bs_result.p_value < 0.05 and not bs_result.no_edge_sharpe
        bs_detail = (
            f"p={bs_result.p_value:.4f}, "
            f"Sharpe CI=[{bs_result.sharpe_ci_lower:.2f}, "
            f"{bs_result.sharpe_ci_upper:.2f}], "
            f"WR CI=[{bs_result.win_rate_ci_lower:.1%}, "
            f"{bs_result.win_rate_ci_upper:.1%}]"
        )

        report.gates.append(GateResult("Bootstrap", bs_passed, bs_detail, bs_result))
        status = "[green]PASS[/green]" if bs_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {bs_detail} ({elapsed:.1f}s)")
        if print_reports:
            print_bootstrap_report(bs_result)
    except Exception as e:
        report.gates.append(GateResult("Bootstrap", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # --- Gate 3: Monte Carlo ---
    console.print("\n  [cyan]Gate 3: Monte Carlo (10,000 simulations)[/cyan]")
    t0 = time.time()
    try:
        mc_result = monte_carlo_analysis(
            result, n_simulations=10000, initial_cash=INITIAL_CASH
        )
        elapsed = time.time() - t0

        mc_passed = mc_result.ruin_probability < 0.10 and mc_result.drawdown_p95 < 40
        mc_detail = (
            f"Ruin={mc_result.ruin_probability:.1%}, "
            f"P95 DD=-{mc_result.drawdown_p95:.1f}%, "
            f"Median equity=${mc_result.equity_median:,.0f}, "
            f"Survival={mc_result.survival_rate:.1%}"
        )

        report.gates.append(GateResult("Monte Carlo", mc_passed, mc_detail, mc_result))
        status = "[green]PASS[/green]" if mc_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {mc_detail} ({elapsed:.1f}s)")
        if print_reports:
            print_monte_carlo_report(mc_result)
    except Exception as e:
        report.gates.append(GateResult("Monte Carlo", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # --- Gate 4: Regime Analysis ---
    console.print("\n  [cyan]Gate 4: Regime Analysis (SPY-based)[/cyan]")
    t0 = time.time()
    try:
        regime_result = analyze_by_regime(result, runner.loader)
        elapsed = time.time() - t0

        regime_passed = not regime_result.regime_dependent
        parts = []
        for rname in ["bull", "bear", "chop", "volatile", "crisis"]:
            if rname in regime_result.regime_metrics:
                m = regime_result.regime_metrics[rname]
                parts.append(f"{rname}:{m.total_trades}t/{m.total_return * 100:+.1f}%")
        regime_detail = ", ".join(parts) if parts else "No trades classified"

        report.gates.append(
            GateResult("Regime", regime_passed, regime_detail, regime_result)
        )
        status = "[green]PASS[/green]" if regime_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {regime_detail} ({elapsed:.0f}s)")
        if print_reports:
            print_regime_report(regime_result)
    except Exception as e:
        report.gates.append(GateResult("Regime", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # --- Summary ---
    console.print(f"\n  [bold]{'=' * 50}[/bold]")
    console.print(f"  [bold]Verdict: {report.pass_count}/4 gates passed[/bold]")
    if report.all_passed:
        console.print(
            "  [bold green]ALL GATES PASSED — CONFIG IS VALIDATED[/bold green]"
        )
    else:
        console.print(
            f"  [bold red]Failed gates: {', '.join(report.failed_gates)}[/bold red]"
        )

    return report


# ============================================================================
# Diagnosis and tuning
# ============================================================================


def generate_tune_configs(
    failed_gates: List[str], base_kwargs: Dict
) -> List[tuple]:
    """Generate targeted tuning configs based on which gates failed."""
    configs = []

    if "Walk-Forward" in failed_gates:
        # Params overfit to full period — loosen PT, try adaptive stops
        for pt in [0.08, 0.12, 0.15, 0.20]:
            configs.append((
                f"WF tune: PT={pt:.0%}",
                {**base_kwargs, "profit_target": pt},
            ))
        for atr in [2.0, 2.5, 3.0]:
            configs.append((
                f"WF tune: ATR {atr}x",
                {**base_kwargs, "stop_mode": "atr", "atr_multiplier": atr},
            ))
        configs.append((
            "WF tune: PT=12% + ATR 2.5x",
            {**base_kwargs, "profit_target": 0.12, "stop_mode": "atr", "atr_multiplier": 2.5},
        ))
        configs.append((
            "WF tune: PT=15% + ATR 2.0x",
            {**base_kwargs, "profit_target": 0.15, "stop_mode": "atr", "atr_multiplier": 2.0},
        ))

    if "Bootstrap" in failed_gates:
        # Not enough statistical edge — need more trades
        for mc in [0.50, 0.55, 0.60]:
            configs.append((
                f"BS tune: confidence={mc}",
                {**base_kwargs, "min_confidence": mc},
            ))
        for cb in [3, 5]:
            configs.append((
                f"BS tune: cooldown={cb}",
                {**base_kwargs, "cooldown_bars": cb},
            ))
        configs.append((
            "BS tune: conf=0.55 + cooldown=5",
            {**base_kwargs, "min_confidence": 0.55, "cooldown_bars": 5},
        ))

    if "Monte Carlo" in failed_gates:
        # Ruin risk too high — tighten risk controls
        for ml in [4.0, 5.0]:
            configs.append((
                f"MC tune: max_loss={ml}%",
                {**base_kwargs, "max_loss_pct": ml},
            ))
        for atr in [2.0, 2.5]:
            configs.append((
                f"MC tune: ATR {atr}x stops",
                {**base_kwargs, "stop_mode": "atr", "atr_multiplier": atr},
            ))
        configs.append((
            "MC tune: ML=4% + ATR 2.0x",
            {**base_kwargs, "max_loss_pct": 4.0, "stop_mode": "atr", "atr_multiplier": 2.0},
        ))

    if "Regime" in failed_gates:
        # Edge concentrated in one regime — add trend filters
        configs.append((
            "Regime tune: + trend_alignment",
            {**base_kwargs, "rules": base_kwargs["rules"] + ["trend_alignment"]},
        ))
        configs.append((
            "Regime tune: + golden_cross",
            {**base_kwargs, "rules": base_kwargs["rules"] + ["golden_cross"]},
        ))
        configs.append((
            "Regime tune: + trend_align + golden_cross",
            {
                **base_kwargs,
                "rules": base_kwargs["rules"] + ["trend_alignment", "golden_cross"],
            },
        ))
        configs.append((
            "Regime tune: higher confidence=0.70",
            {**base_kwargs, "min_confidence": 0.70},
        ))

    # Deduplicate by kwarg content
    seen = set()
    unique = []
    for label, kwargs in configs:
        key = str(sorted((k, str(v)) for k, v in kwargs.items()))
        if key not in seen:
            seen.add(key)
            unique.append((label, kwargs))

    return unique


# ============================================================================
# Report writer
# ============================================================================


def _format_kwargs(kw: Dict) -> str:
    """Format kwargs into a readable string."""
    parts = [f"Rules: {', '.join(kw.get('rules', []))}"]
    parts.append(f"PT={kw.get('profit_target', 0.10):.0%}")
    parts.append(f"Conf={kw.get('min_confidence', 0.65)}")
    parts.append(f"ML={kw.get('max_loss_pct', 6.0)}%")
    parts.append(f"Cooldown={kw.get('cooldown_bars', 7)}")
    if kw.get("stop_mode") == "atr":
        parts.append(f"ATR={kw.get('atr_multiplier', 2.0)}x")
    return ", ".join(parts)


def _write_config_section(f, report: ValidationReport):
    """Write a config section to the markdown file."""
    f.write(f"### {report.label}\n\n")

    kw = report.kwargs
    f.write(f"- **Rules:** `{', '.join(kw.get('rules', []))}`\n")
    f.write(f"- **Profit Target:** {kw.get('profit_target', 0.10):.0%}\n")
    f.write(f"- **Min Confidence:** {kw.get('min_confidence', 0.65)}\n")
    f.write(f"- **Max Loss:** {kw.get('max_loss_pct', 6.0)}%\n")
    f.write(f"- **Cooldown:** {kw.get('cooldown_bars', 7)} bars\n")
    if kw.get("stop_mode") == "atr":
        f.write(f"- **Stop Mode:** ATR x{kw.get('atr_multiplier', 2.0)}\n")
    f.write("\n")

    if report.backtest:
        b = report.backtest
        f.write(
            f"**Performance:** Return={b.total_return:+.1%}, "
            f"Trades={b.total_trades}, WR={b.win_rate:.1%}, "
            f"Sharpe={b.sharpe_ratio or 0:.2f}, PF={b.profit_factor or 0:.2f}, "
            f"DD=-{b.max_drawdown_pct or 0:.1f}%\n\n"
        )

    if report.gates:
        f.write("| Gate | Status | Detail |\n")
        f.write("|------|--------|--------|\n")
        for g in report.gates:
            status = "PASS" if g.passed else "FAIL"
            f.write(f"| {g.name} | {status} | {g.detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")

    f.write("---\n\n")


def write_markdown_report(
    baseline_report: ValidationReport,
    tune_reports: List[ValidationReport],
    final_report: Optional[ValidationReport],
):
    """Write ccj-validated.md with full validation evidence."""
    filepath = os.path.join(
        os.path.expanduser("~/Projects/backtesting-service"), "ccj-validated.md"
    )

    with open(filepath, "w") as f:
        f.write("# CCJ Validated Optimization Results\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**Initial Cash:** ${INITIAL_CASH:,}\n\n")

        # Baseline section
        f.write("## 1. Baseline Validation\n\n")
        _write_config_section(f, baseline_report)

        # Tuning section
        if tune_reports:
            f.write("## 2. Tuning Results\n\n")
            for tr in tune_reports:
                _write_config_section(f, tr)

        # Final recommendation
        f.write("## 3. Final Recommendation\n\n")

        if final_report and final_report.all_passed:
            _write_config_section(f, final_report)

            # CLI command
            kw = final_report.kwargs
            cmd_parts = [
                "python -m backtesting",
                f"--symbol {SYMBOL}",
                f"--start {START_DATE}",
                f"--end {END_DATE}",
                f"--rules {','.join(kw['rules'])}",
                f"--profit-target {kw.get('profit_target', 0.10)}",
                f"--min-confidence {kw.get('min_confidence', 0.65)}",
                f"--max-loss {kw.get('max_loss_pct', 6.0)}",
                f"--cooldown-bars {kw.get('cooldown_bars', 7)}",
                f"--timeframe {kw.get('timeframe', 'daily')}",
                f"--exit-timeframe {kw.get('exit_timeframe', '5min')}",
            ]
            if kw.get("stop_mode") == "atr":
                cmd_parts.append(f"--stop-mode atr")
                cmd_parts.append(
                    f"--atr-multiplier {kw.get('atr_multiplier', 2.0)}"
                )
            cmd_parts.append(
                "--walk-forward --bootstrap --monte-carlo --regime-analysis"
            )
            f.write(f"**Reproduce with CLI:**\n```\n{' '.join(cmd_parts)}\n```\n\n")

            # Comparison to baseline
            if baseline_report.backtest and final_report.backtest:
                b = baseline_report.backtest
                t = final_report.backtest
                f.write("### Baseline vs Tuned\n\n")
                f.write("| Metric | Baseline | Tuned | Delta |\n")
                f.write("|--------|----------|-------|-------|\n")
                f.write(
                    f"| Return | {b.total_return:+.1%} | {t.total_return:+.1%} "
                    f"| {t.total_return - b.total_return:+.1%} |\n"
                )
                f.write(
                    f"| Sharpe | {b.sharpe_ratio or 0:.2f} | {t.sharpe_ratio or 0:.2f} "
                    f"| {(t.sharpe_ratio or 0) - (b.sharpe_ratio or 0):+.2f} |\n"
                )
                f.write(
                    f"| Win Rate | {b.win_rate:.1%} | {t.win_rate:.1%} "
                    f"| {t.win_rate - b.win_rate:+.1%} |\n"
                )
                f.write(
                    f"| Max DD | -{b.max_drawdown_pct or 0:.1f}% "
                    f"| -{t.max_drawdown_pct or 0:.1f}% "
                    f"| {(b.max_drawdown_pct or 0) - (t.max_drawdown_pct or 0):+.1f}% |\n"
                )
                f.write(
                    f"| Trades | {b.total_trades} | {t.total_trades} "
                    f"| {t.total_trades - b.total_trades:+d} |\n"
                )
                f.write(
                    f"| PF | {b.profit_factor or 0:.2f} | {t.profit_factor or 0:.2f} "
                    f"| {(t.profit_factor or 0) - (b.profit_factor or 0):+.2f} |\n"
                )
                f.write(
                    f"| Validation | {baseline_report.pass_count}/4 "
                    f"| {final_report.pass_count}/4 | |\n"
                )
                f.write("\n")

        elif baseline_report.all_passed:
            f.write(
                "**Baseline passes all validation gates — no tuning needed.**\n\n"
            )
            _write_config_section(f, baseline_report)
        else:
            f.write("**No config passed all 4 validation gates.**\n\n")
            all_reports = [baseline_report] + tune_reports
            best = max(
                all_reports,
                key=lambda r: (
                    r.pass_count,
                    r.backtest.sharpe_ratio or 0 if r.backtest else 0,
                ),
            )
            f.write(
                f"Best available: **{best.label}** "
                f"({best.pass_count}/4 gates passed)\n\n"
            )
            _write_config_section(f, best)

    console.print(f"\n[bold]Report written to {filepath}[/bold]")


# ============================================================================
# Main
# ============================================================================


def main():
    total_start = time.time()

    console.print(
        Panel.fit(
            "[bold blue]CCJ Validate-Then-Tune Optimization[/bold blue]\n"
            f"Symbol: {SYMBOL} | Period: {START_DATE} to {END_DATE}\n"
            f"Baseline: {', '.join(BASELINE['rules'])}\n"
            f"Params: PT={BASELINE['profit_target']:.0%}, "
            f"Conf={BASELINE['min_confidence']}, "
            f"ML={BASELINE['max_loss_pct']}%, "
            f"Cooldown={BASELINE['cooldown_bars']}",
            border_style="blue",
        )
    )

    runner = BacktraderRunner(initial_cash=INITIAL_CASH)

    # ── Step 1: Validate baseline ──────────────────────────────────────────
    console.print("\n[bold blue]STEP 1: Validate Baseline[/bold blue]")
    baseline_report = run_full_validation(
        runner, "Baseline (rules.yaml)", BASELINE
    )

    if baseline_report.all_passed:
        console.print(
            "\n[bold green]Baseline passes all 4 validation gates! "
            "No tuning needed.[/bold green]"
        )
        write_markdown_report(baseline_report, [], baseline_report)
        elapsed = time.time() - total_start
        console.print(f"\nTotal time: {elapsed / 60:.1f} minutes")
        return

    # ── Step 2: Diagnose ───────────────────────────────────────────────────
    failed = baseline_report.failed_gates
    console.print(f"\n[bold blue]STEP 2: Diagnosis[/bold blue]")
    console.print(f"  Failed gates: [red]{', '.join(failed)}[/red]")

    tune_configs = generate_tune_configs(failed, BASELINE)
    console.print(f"  Generated {len(tune_configs)} targeted tuning configs")

    for label, _ in tune_configs:
        console.print(f"    - {label}")

    # ── Step 3: Quick screen tuning configs ────────────────────────────────
    console.print(
        f"\n[bold blue]STEP 3: Quick Screen "
        f"({len(tune_configs)} configs)[/bold blue]"
    )

    candidates = []
    for i, (label, kwargs) in enumerate(tune_configs, 1):
        console.print(f"\n  [{i}/{len(tune_configs)}] {label}...")
        try:
            result = runner.run(
                symbol=SYMBOL,
                start_date=START_DATE,
                end_date=END_DATE,
                **kwargs,
            )
            if result.total_trades >= 5:
                console.print(
                    f"    {result.total_trades} trades, "
                    f"WR={result.win_rate:.1%}, "
                    f"Return={result.total_return:+.1%}, "
                    f"Sharpe={result.sharpe_ratio or 0:.2f}"
                )
                candidates.append((label, kwargs, result))
            else:
                console.print(f"    Skipped ({result.total_trades} trades)")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    if not candidates:
        console.print("\n[red]No tuning configs produced viable results[/red]")
        write_markdown_report(baseline_report, [], None)
        elapsed = time.time() - total_start
        console.print(f"\nTotal time: {elapsed / 60:.1f} minutes")
        return

    # Rank by Sharpe, then profit factor
    candidates.sort(
        key=lambda c: (c[2].sharpe_ratio or 0, c[2].profit_factor or 0),
        reverse=True,
    )

    console.print(f"\n  Top candidates by Sharpe:")
    for i, (label, _, result) in enumerate(candidates[:5], 1):
        console.print(
            f"    #{i} {label}: Sharpe={result.sharpe_ratio or 0:.2f}, "
            f"Return={result.total_return:+.1%}, WR={result.win_rate:.1%}"
        )

    # ── Step 4: Full validation of top 3 ───────────────────────────────────
    n_validate = min(3, len(candidates))
    console.print(
        f"\n[bold blue]STEP 4: Full Validation of Top "
        f"{n_validate} Candidates[/bold blue]"
    )

    tune_reports = []
    best_report = None

    for label, kwargs, _ in candidates[:n_validate]:
        report = run_full_validation(runner, label, kwargs, print_reports=True)
        tune_reports.append(report)

        if report.all_passed:
            if best_report is None or (
                report.backtest
                and best_report.backtest
                and (report.backtest.sharpe_ratio or 0)
                > (best_report.backtest.sharpe_ratio or 0)
            ):
                best_report = report

    # If no tuned config passes all gates, pick the one with most passes
    if best_report is None:
        all_reports = [baseline_report] + tune_reports
        best_report = max(
            all_reports,
            key=lambda r: (
                r.pass_count,
                r.backtest.sharpe_ratio or 0 if r.backtest else 0,
            ),
        )
        console.print(
            f"\n[yellow]No config passed all 4 gates. "
            f"Best: {best_report.label} ({best_report.pass_count}/4)[/yellow]"
        )

    # ── Step 5: Write report ───────────────────────────────────────────────
    console.print(f"\n[bold blue]STEP 5: Writing Report[/bold blue]")
    final = best_report if best_report.all_passed else None
    write_markdown_report(
        baseline_report, tune_reports, final or best_report
    )

    elapsed = time.time() - total_start
    console.print(f"\n[bold]Total time: {elapsed / 60:.1f} minutes[/bold]")

    # Final summary table
    console.print()
    table = Table(
        title="Validation Summary",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Config", style="bold", max_width=45)
    table.add_column("WF", justify="center")
    table.add_column("BS", justify="center")
    table.add_column("MC", justify="center")
    table.add_column("Regime", justify="center")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")

    for r in [baseline_report] + tune_reports:
        gate_status = {}
        for g in r.gates:
            short = {
                "Walk-Forward": "WF",
                "Bootstrap": "BS",
                "Monte Carlo": "MC",
                "Regime": "Regime",
            }.get(g.name, g.name)
            gate_status[short] = (
                "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]"
            )

        sharpe_str = (
            f"{r.backtest.sharpe_ratio or 0:.2f}" if r.backtest else "-"
        )
        ret_str = f"{r.backtest.total_return:+.1%}" if r.backtest else "-"

        table.add_row(
            r.label[:45],
            gate_status.get("WF", "-"),
            gate_status.get("BS", "-"),
            gate_status.get("MC", "-"),
            gate_status.get("Regime", "-"),
            sharpe_str,
            ret_str,
        )

    console.print(table)


if __name__ == "__main__":
    main()
