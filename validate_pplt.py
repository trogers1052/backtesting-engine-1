#!/usr/bin/env python3
"""
PPLT (Platinum ETF) Validate-Then-Tune Optimization

Validates PPLT platinum ETF rules through 4 statistical gates,
diagnoses any failures, and makes targeted parameter adjustments.

PPLT is active Tier 1 (best performer): 59.4% WR, 0.81 Sharpe, 2.23 PF.
Physical platinum ETF — auto catalysts, hydrogen economy, supply deficit.
Current config: 7 rules, PT=6%, ML=4%, cooldown=3, conf=0.65.

Usage:
    python validate_pplt.py
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

SYMBOL = "PPLT"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 2, 28)
INITIAL_CASH = 1000

# Current rules.yaml config (7 rules, conservative compounder)
BASELINE = dict(
    rules=[
        "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
        "trend_alignment", "golden_cross", "trend_break_warning", "death_cross",
    ],
    min_confidence=0.65,
    profit_target=0.06,     # 6% — conservative compounder
    stop_loss=1.0,          # Disabled — max_loss_pct is the effective stop
    max_loss_pct=4.0,       # 4% tight stop
    cooldown_bars=3,
    timeframe="daily",
    exit_timeframe="5min",
)

# Alternative baselines to compare
ALT_BASELINES = [
    (
        "Alt A: CCJ-style (3 rules, 10%/6%)",
        dict(
            rules=["trend_continuation", "seasonality", "death_cross"],
            min_confidence=0.65,
            profit_target=0.10,
            stop_loss=1.0,
            max_loss_pct=6.0,
            cooldown_bars=7,
            timeframe="daily",
            exit_timeframe="5min",
        ),
    ),
    (
        "Alt B: Wider PT (7 rules, 10%/4%)",
        dict(
            rules=[
                "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
                "trend_alignment", "golden_cross", "trend_break_warning", "death_cross",
            ],
            min_confidence=0.65,
            profit_target=0.10,
            stop_loss=1.0,
            max_loss_pct=4.0,
            cooldown_bars=3,
            timeframe="daily",
            exit_timeframe="5min",
        ),
    ),
    (
        "Alt C: Lower confidence (7 rules, 6%/4%, conf=0.50)",
        dict(
            rules=[
                "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
                "trend_alignment", "golden_cross", "trend_break_warning", "death_cross",
            ],
            min_confidence=0.50,
            profit_target=0.06,
            stop_loss=1.0,
            max_loss_pct=4.0,
            cooldown_bars=3,
            timeframe="daily",
            exit_timeframe="5min",
        ),
    ),
    (
        "Alt D: Core only (3 rules, 6%/4%)",
        dict(
            rules=["enhanced_buy_dip", "momentum_reversal", "trend_continuation"],
            min_confidence=0.65,
            profit_target=0.06,
            stop_loss=1.0,
            max_loss_pct=4.0,
            cooldown_bars=5,
            timeframe="daily",
            exit_timeframe="5min",
        ),
    ),
]


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    data: object = None


@dataclass
class ValidationReport:
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
    """Run all 4 validation gates on a config."""
    report = ValidationReport(label=label, kwargs=run_kwargs.copy())

    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit(f"[bold]Validating: {label}[/bold]", border_style="cyan"))

    # --- Full-period backtest ---
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
            SYMBOL, START_DATE, END_DATE,
            train_pct=0.7, embargo_days=5, purge_days=10,
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
        # Try different profit targets
        for pt in [0.04, 0.08, 0.10, 0.12]:
            configs.append((
                f"WF tune: PT={pt:.0%}",
                {**base_kwargs, "profit_target": pt},
            ))
        # ATR-based stops
        for atr in [2.0, 2.5, 3.0]:
            configs.append((
                f"WF tune: ATR {atr}x",
                {**base_kwargs, "stop_mode": "atr", "atr_multiplier": atr},
            ))

    if "Bootstrap" in failed_gates:
        for mc in [0.50, 0.55, 0.60, 0.70]:
            configs.append((
                f"BS tune: confidence={mc}",
                {**base_kwargs, "min_confidence": mc},
            ))
        for cb in [5, 7]:
            configs.append((
                f"BS tune: cooldown={cb}",
                {**base_kwargs, "cooldown_bars": cb},
            ))

    if "Monte Carlo" in failed_gates:
        for ml in [3.0, 5.0, 6.0]:
            configs.append((
                f"MC tune: max_loss={ml}%",
                {**base_kwargs, "max_loss_pct": ml},
            ))
        for atr in [1.5, 2.0]:
            configs.append((
                f"MC tune: ATR {atr}x stops",
                {**base_kwargs, "stop_mode": "atr", "atr_multiplier": atr},
            ))

    if "Regime" in failed_gates:
        # Try adding/removing rules for regime diversity
        configs.append((
            "Regime tune: + seasonality",
            {**base_kwargs, "rules": base_kwargs["rules"] + ["seasonality"]},
        ))
        configs.append((
            "Regime tune: + commodity_breakout",
            {**base_kwargs, "rules": base_kwargs["rules"] + ["commodity_breakout"]},
        ))
        configs.append((
            "Regime tune: higher confidence=0.70",
            {**base_kwargs, "min_confidence": 0.70},
        ))
        configs.append((
            "Regime tune: - golden_cross (5 rules)",
            {**base_kwargs, "rules": [r for r in base_kwargs["rules"] if r != "golden_cross"]},
        ))

    # Deduplicate
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


def _write_config_section(f, report: ValidationReport):
    kw = report.kwargs
    f.write(f"### {report.label}\n\n")
    f.write(f"- **Rules:** `{', '.join(kw.get('rules', []))}`\n")
    f.write(f"- **Profit Target:** {kw.get('profit_target', 0.06):.0%}\n")
    f.write(f"- **Min Confidence:** {kw.get('min_confidence', 0.65)}\n")
    f.write(f"- **Max Loss:** {kw.get('max_loss_pct', 4.0)}%\n")
    f.write(f"- **Cooldown:** {kw.get('cooldown_bars', 3)} bars\n")
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
            status = "**PASS**" if g.passed else "FAIL"
            f.write(f"| {g.name} | {status} | {g.detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")

    f.write("---\n\n")


def write_markdown_report(
    baseline_report: ValidationReport,
    alt_reports: List[ValidationReport],
    tune_reports: List[ValidationReport],
    best_report: Optional[ValidationReport],
    total_elapsed: float,
):
    filepath = os.path.join(
        os.path.expanduser("~/Projects/backtesting-service"), "pplt-validated.md"
    )

    with open(filepath, "w") as f:
        f.write("# PPLT (Platinum ETF) Validated Optimization Results\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**Initial Cash:** ${INITIAL_CASH:,}\n")
        f.write(f"**Timeframe:** Daily entries + 5-minute exits (multi-TF hybrid)\n")
        f.write(f"**Validation Runtime:** {total_elapsed / 60:.1f} minutes\n")
        f.write(f"**Prior Status:** Active Tier 1 (59.4% WR, 0.81 Sharpe, 2.23 PF)\n\n")
        f.write("---\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("Same validate-then-tune approach as CCJ/HL/UUUU validations:\n\n")
        f.write("1. **Screen Baselines** — Test the current rules.yaml config + alternatives\n")
        f.write("2. **Validate Best** — Run through all 4 statistical validation gates\n")
        f.write("3. **Diagnose** — Identify which gates pass/fail and why\n")
        f.write("4. **Targeted Tune** — Sweep only the parameters that address failing gates\n")
        f.write("5. **Re-validate** — Confirm tuned configs through all 4 gates\n\n")
        f.write("### Validation Gates\n\n")
        f.write("| Gate | Method | Pass Criteria | Purpose |\n")
        f.write("|------|--------|---------------|---------|\n")
        f.write("| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting |\n")
        f.write("| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance |\n")
        f.write("| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures worst-case risk |\n")
        f.write("| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects regime dependency |\n\n")
        f.write("---\n\n")

        # Baseline screening
        f.write("## 1. Baseline Screening\n\n")
        f.write("PPLT is the current Tier 1 compounder — high win rate, tight stops, frequent trades.\n\n")
        f.write("| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |\n")
        f.write("|--------|--------|----------|--------|--------|-----|--------|\n")

        all_screened = [baseline_report] + alt_reports
        for r in all_screened:
            if r.backtest:
                b = r.backtest
                f.write(
                    f"| {r.label} | {b.total_trades} | {b.win_rate:.1%} | "
                    f"{b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} | "
                    f"{b.profit_factor or 0:.2f} | -{b.max_drawdown_pct or 0:.1f}% |\n"
                )
            else:
                f.write(f"| {r.label} | - | - | - | - | - | - |\n")
        f.write("\n")

        validated = best_report or baseline_report
        f.write(f"**Best baseline selected for validation: {validated.label}**\n\n")
        f.write("---\n\n")

        # Full validation
        f.write("## 2. Full Validation\n\n")
        _write_config_section(f, validated)

        # Tuning
        if tune_reports:
            f.write("## 3. Tuning Results\n\n")

            # Quick screen table
            screened = [tr for tr in tune_reports if tr.backtest]
            if screened:
                f.write("### Quick Screen\n\n")
                f.write("| Config | Trades | Win Rate | Return | Sharpe |\n")
                f.write("|--------|--------|----------|--------|--------|\n")
                for tr in screened:
                    b = tr.backtest
                    f.write(
                        f"| {tr.label} | {b.total_trades} | {b.win_rate:.1%} | "
                        f"{b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} |\n"
                    )
                f.write("\n")

            # Full validation details
            validated_tunes = [tr for tr in tune_reports if tr.gates]
            if validated_tunes:
                f.write("### Full Validation of Top Candidates\n\n")
                for tr in validated_tunes:
                    _write_config_section(f, tr)

        # Summary table
        f.write("## 4. Summary Table\n\n")
        f.write("| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |\n")
        f.write("|--------|-----|-----|-----|--------|--------|--------|--------|\n")

        all_validated = [r for r in [validated] + tune_reports if r.gates]
        for r in all_validated:
            gate_status = {}
            for g in r.gates:
                short = {
                    "Walk-Forward": "WF", "Bootstrap": "BS",
                    "Monte Carlo": "MC", "Regime": "Regime",
                }.get(g.name, g.name)
                gate_status[short] = "**PASS**" if g.passed else "FAIL"

            sharpe = f"{r.backtest.sharpe_ratio or 0:.2f}" if r.backtest else "-"
            ret = f"{r.backtest.total_return:+.1%}" if r.backtest else "-"
            trades = str(r.backtest.total_trades) if r.backtest else "-"

            label = r.label
            if r == best_report:
                label = f"**{label}**"
                sharpe = f"**{sharpe}**"
                ret = f"**{ret}**"

            f.write(
                f"| {label} | {gate_status.get('WF', '-')} | "
                f"{gate_status.get('BS', '-')} | {gate_status.get('MC', '-')} | "
                f"{gate_status.get('Regime', '-')} | {sharpe} | {ret} | {trades} |\n"
            )
        f.write("\n---\n\n")

        # Final recommendation
        f.write("## 5. Final Recommendation\n\n")

        if best_report and best_report.all_passed:
            f.write(f"**PPLT validated.** The {best_report.label} config passes all 4 validation gates.\n\n")
            _write_config_section(f, best_report)

            kw = best_report.kwargs
            f.write("### Production Configuration\n\n```\n")
            f.write(f"Rules: {', '.join(kw['rules'])}\n")
            f.write(f"Profit Target: {kw.get('profit_target', 0.06):.0%}\n")
            f.write(f"Min Confidence: {kw.get('min_confidence', 0.65)}\n")
            f.write(f"Max Loss: {kw.get('max_loss_pct', 4.0)}%\n")
            f.write(f"Cooldown: {kw.get('cooldown_bars', 3)} bars\n")
            f.write(f"Timeframe: daily (entries), 5min (exits)\n")
            f.write("```\n\n")

        elif best_report and best_report.pass_count >= 3:
            f.write(
                f"**PPLT partially validates.** "
                f"Best config: {best_report.label} ({best_report.pass_count}/4 gates).\n\n"
            )
            _write_config_section(f, best_report)
            f.write("### Deployment Recommendation\n\n")
            f.write("- Keep as Tier 1 with regime-aware deployment\n")
            f.write("- Reduce size in non-bull markets if Regime gate fails\n")
            f.write("- Re-validate after 6 months of additional data\n\n")

        elif best_report and best_report.pass_count >= 2:
            f.write(
                f"**PPLT shows limited edge.** "
                f"Best config: {best_report.label} ({best_report.pass_count}/4 gates).\n\n"
            )
            _write_config_section(f, best_report)
            f.write("### Deployment Recommendation\n\n")
            f.write("- Consider downgrading from Tier 1 to Tier 2\n")
            f.write("- Only trade during validated regime conditions\n")
            f.write("- Reduce position sizing\n\n")

        else:
            f.write(
                f"**PPLT should be downgraded or removed.** "
                f"Best config passes only {best_report.pass_count if best_report else 0}/4 gates.\n\n"
            )
            if best_report:
                _write_config_section(f, best_report)

    console.print(f"\n[bold]Report written to {filepath}[/bold]")


# ============================================================================
# Main
# ============================================================================


def main():
    total_start = time.time()

    console.print(
        Panel.fit(
            "[bold blue]PPLT (Platinum ETF) Validate-Then-Tune Optimization[/bold blue]\n"
            f"Symbol: {SYMBOL} | Period: {START_DATE} to {END_DATE}\n"
            f"Current status: Active Tier 1 (59.4% WR, 0.81 Sharpe, 2.23 PF)\n"
            f"Baseline: {', '.join(BASELINE['rules'][:3])}... ({len(BASELINE['rules'])} rules)\n"
            f"Params: PT={BASELINE['profit_target']:.0%}, "
            f"Conf={BASELINE['min_confidence']}, "
            f"ML={BASELINE['max_loss_pct']}%, "
            f"Cooldown={BASELINE['cooldown_bars']}",
            border_style="blue",
        )
    )

    runner = BacktraderRunner(initial_cash=INITIAL_CASH)

    # ── Step 1: Screen baseline + alternatives ─────────────────────────────
    console.print("\n[bold blue]STEP 1: Screen Baseline + Alternatives[/bold blue]")

    console.print(f"\n  Screening primary baseline (rules.yaml 7-rule config)...")
    try:
        baseline_bt = runner.run(
            symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE, **BASELINE
        )
        console.print(
            f"  Primary: {baseline_bt.total_trades} trades, "
            f"WR={baseline_bt.win_rate:.1%}, "
            f"Return={baseline_bt.total_return:+.1%}, "
            f"Sharpe={baseline_bt.sharpe_ratio or 0:.2f}"
        )
    except Exception as e:
        console.print(f"  [red]Primary baseline failed: {e}[/red]")
        baseline_bt = None

    alt_results = []
    for alt_label, alt_kwargs in ALT_BASELINES:
        console.print(f"\n  Screening {alt_label}...")
        try:
            alt_bt = runner.run(
                symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE, **alt_kwargs
            )
            console.print(
                f"  {alt_bt.total_trades} trades, "
                f"WR={alt_bt.win_rate:.1%}, "
                f"Return={alt_bt.total_return:+.1%}, "
                f"Sharpe={alt_bt.sharpe_ratio or 0:.2f}"
            )
            alt_results.append((alt_label, alt_kwargs, alt_bt))
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            alt_results.append((alt_label, alt_kwargs, None))

    # Pick best by Sharpe (min 5 trades)
    all_candidates = []
    if baseline_bt and baseline_bt.total_trades >= 5:
        all_candidates.append((
            f"rules.yaml baseline (7 rules, 6%/4%)",
            BASELINE, baseline_bt,
        ))
    for alt_label, alt_kwargs, alt_bt in alt_results:
        if alt_bt and alt_bt.total_trades >= 5:
            all_candidates.append((alt_label, alt_kwargs, alt_bt))

    if not all_candidates:
        console.print("\n[red]No baseline produced enough trades.[/red]")
        baseline_report = ValidationReport(
            label="rules.yaml baseline (7 rules, 6%/4%)", kwargs=BASELINE,
        )
        elapsed = time.time() - total_start
        write_markdown_report(baseline_report, [], [], None, elapsed)
        console.print(f"\nTotal time: {elapsed / 60:.1f} minutes")
        return

    all_candidates.sort(
        key=lambda c: (c[2].sharpe_ratio or 0, c[2].profit_factor or 0),
        reverse=True,
    )

    console.print(f"\n  [bold]Ranked baselines by Sharpe:[/bold]")
    for i, (label, _, bt) in enumerate(all_candidates, 1):
        console.print(
            f"    #{i} {label}: Sharpe={bt.sharpe_ratio or 0:.2f}, "
            f"Return={bt.total_return:+.1%}, WR={bt.win_rate:.1%}, "
            f"Trades={bt.total_trades}"
        )

    best_label, best_kwargs, best_bt = all_candidates[0]
    console.print(f"\n  [bold green]Selected for validation: {best_label}[/bold green]")

    # Build report objects
    alt_reports = []
    for alt_label, alt_kwargs, alt_bt in alt_results:
        r = ValidationReport(label=alt_label, kwargs=alt_kwargs)
        if alt_bt:
            r.backtest = alt_bt
        alt_reports.append(r)

    baseline_report = ValidationReport(
        label="rules.yaml baseline (7 rules, 6%/4%)", kwargs=BASELINE,
    )
    if baseline_bt:
        baseline_report.backtest = baseline_bt

    # ── Step 2: Full validation ────────────────────────────────────────────
    console.print(f"\n[bold blue]STEP 2: Full Validation of Best Baseline[/bold blue]")
    validated_report = run_full_validation(runner, best_label, best_kwargs)

    if validated_report.all_passed:
        console.print(
            "\n[bold green]Best baseline passes all 4 gates! No tuning needed.[/bold green]"
        )
        elapsed = time.time() - total_start
        write_markdown_report(
            baseline_report, alt_reports, [], validated_report, elapsed
        )
        console.print(f"\nTotal time: {elapsed / 60:.1f} minutes")
        return

    # ── Step 3: Diagnose ───────────────────────────────────────────────────
    failed = validated_report.failed_gates
    console.print(f"\n[bold blue]STEP 3: Diagnosis[/bold blue]")
    console.print(f"  Failed gates: [red]{', '.join(failed)}[/red]")

    tune_configs = generate_tune_configs(failed, best_kwargs)
    console.print(f"  Generated {len(tune_configs)} targeted tuning configs")
    for label, _ in tune_configs:
        console.print(f"    - {label}")

    # ── Step 4: Quick screen ───────────────────────────────────────────────
    console.print(
        f"\n[bold blue]STEP 4: Quick Screen ({len(tune_configs)} configs)[/bold blue]"
    )

    tune_candidates = []
    for i, (label, kwargs) in enumerate(tune_configs, 1):
        console.print(f"\n  [{i}/{len(tune_configs)}] {label}...")
        try:
            result = runner.run(
                symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE,
                **kwargs,
            )
            if result.total_trades >= 5:
                console.print(
                    f"    {result.total_trades} trades, "
                    f"WR={result.win_rate:.1%}, "
                    f"Return={result.total_return:+.1%}, "
                    f"Sharpe={result.sharpe_ratio or 0:.2f}"
                )
                tune_candidates.append((label, kwargs, result))
            else:
                console.print(f"    Skipped ({result.total_trades} trades)")
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")

    tune_reports = []

    if not tune_candidates:
        console.print("\n[red]No tuning configs produced viable results[/red]")
    else:
        tune_candidates.sort(
            key=lambda c: (c[2].sharpe_ratio or 0, c[2].profit_factor or 0),
            reverse=True,
        )

        console.print(f"\n  Top candidates by Sharpe:")
        for i, (label, _, result) in enumerate(tune_candidates[:5], 1):
            console.print(
                f"    #{i} {label}: Sharpe={result.sharpe_ratio or 0:.2f}, "
                f"Return={result.total_return:+.1%}, WR={result.win_rate:.1%}"
            )

        # ── Step 5: Full validation of top 3 ──────────────────────────────
        n_validate = min(3, len(tune_candidates))
        console.print(
            f"\n[bold blue]STEP 5: Full Validation of Top "
            f"{n_validate} Candidates[/bold blue]"
        )

        for label, kwargs, _ in tune_candidates[:n_validate]:
            report = run_full_validation(runner, label, kwargs, print_reports=True)
            tune_reports.append(report)

    # Find overall best
    all_reports = [validated_report] + tune_reports
    best_overall = max(
        all_reports,
        key=lambda r: (
            r.pass_count,
            r.backtest.sharpe_ratio or 0 if r.backtest else 0,
        ),
    )

    if best_overall.all_passed:
        console.print(
            f"\n[bold green]Found validated config: {best_overall.label}[/bold green]"
        )
    else:
        console.print(
            f"\n[yellow]No config passed all 4 gates. "
            f"Best: {best_overall.label} ({best_overall.pass_count}/4)[/yellow]"
        )

    # ── Step 6: Write report ──────────────────────────────────────────────
    console.print(f"\n[bold blue]STEP 6: Writing Report[/bold blue]")
    elapsed = time.time() - total_start
    write_markdown_report(
        baseline_report, alt_reports, tune_reports, best_overall, elapsed,
    )

    console.print(f"\n[bold]Total time: {elapsed / 60:.1f} minutes[/bold]")

    # Final summary table
    console.print()
    table = Table(
        title="PPLT Validation Summary",
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

    for r in [validated_report] + tune_reports:
        gate_status = {}
        for g in r.gates:
            short = {
                "Walk-Forward": "WF", "Bootstrap": "BS",
                "Monte Carlo": "MC", "Regime": "Regime",
            }.get(g.name, g.name)
            gate_status[short] = (
                "[green]PASS[/green]" if g.passed else "[red]FAIL[/red]"
            )

        sharpe_str = f"{r.backtest.sharpe_ratio or 0:.2f}" if r.backtest else "-"
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
