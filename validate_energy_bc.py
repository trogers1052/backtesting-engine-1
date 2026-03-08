#!/usr/bin/env python3
"""
Energy B/C Tier — Hybrid Walk-Backward Validation

Replaces traditional walk-forward with the hybrid approach:
  1. TUNE on recent 6 months (rules calibrated to current market)
  2. HOLDOUT last 2 months (unseen forward-looking test)
  3. WALK BACKWARD across historical regimes (2020-2024)
  4. Bootstrap, Monte Carlo, Regime gates unchanged

B Tier: CVX, KMI, OKE, COP
C Tier: PSX, FANG, NEP

Usage:
    python validate_energy_bc.py CVX        # Single stock
    python validate_energy_bc.py ALL        # All 7 stocks
"""

import logging
import os
import sys
import time
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional

de_path = os.path.expanduser("~/Projects/decision-engine")
if de_path not in sys.path:
    sys.path.insert(0, de_path)
os.environ.setdefault("DECISION_ENGINE_PATH", de_path)

from backtesting.engine import BacktraderRunner
from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.validation.walk_backward import WalkBackwardValidator, WalkBackwardResult
from backtesting.validation.bootstrap import bootstrap_analysis, BootstrapResult
from backtesting.validation.monte_carlo import monte_carlo_analysis, MonteCarloResult
from backtesting.validation.regime import analyze_by_regime, RegimeAnalysisResult
from backtesting.validation.report import (
    print_walk_backward_report,
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

START_DATE = date(2019, 1, 1)   # Earliest data for regime walk-back
END_DATE = date(2026, 3, 7)
INITIAL_CASH = 1000

# Walk-backward settings
TUNE_MONTHS = 6       # Tune on most recent 6 months
HOLDOUT_MONTHS = 2    # Hold out last 2 months as forward test
MIN_REGIMES_PASS = 3  # Must pass 3 of 4 historical regimes

SYMBOL_INFO = {
    # B Tier (ranks 4-7)
    "CVX": {
        "name": "Chevron",
        "description": "Integrated oil major — Permian, LNG, refining, strongest balance sheet",
        "tier": "B Tier — Integrated",
        "bear_beta": "0.70",
        "sub_sector": "integrated",
    },
    "KMI": {
        "name": "Kinder Morgan",
        "description": "Largest US natural gas pipeline network, 90% fee-based revenue",
        "tier": "B Tier — Midstream",
        "bear_beta": "0.50-0.60",
        "sub_sector": "midstream",
    },
    "OKE": {
        "name": "ONEOK",
        "description": "NGL pipelines and processing, post-Magellan diversification",
        "tier": "B Tier — Midstream",
        "bear_beta": "0.60-0.70",
        "sub_sector": "midstream",
    },
    "COP": {
        "name": "ConocoPhillips",
        "description": "Lowest-cost independent E&P — Permian, Eagle Ford, Bakken, Alaska",
        "tier": "B Tier — Upstream E&P",
        "bear_beta": "0.90-1.00",
        "sub_sector": "upstream",
    },
    # C Tier (ranks 8-10)
    "PSX": {
        "name": "Phillips 66",
        "description": "Refining + midstream + chemicals, margins can widen in volatility",
        "tier": "C Tier — Refining",
        "bear_beta": "0.80",
        "sub_sector": "refining",
    },
    "FANG": {
        "name": "Diamondback Energy",
        "description": "Lowest-cost Permian E&P, recovery play, survives anything",
        "tier": "C Tier — Upstream E&P",
        "bear_beta": "1.20-1.40",
        "sub_sector": "upstream",
    },
    "NEP": {
        "name": "NextEra Energy Partners",
        "description": "Contracted renewables portfolio, contrarian deep-value play",
        "tier": "C Tier — Renewables",
        "bear_beta": "0.40-0.50",
        "sub_sector": "renewables",
    },
}

# ============================================================================
# Rule sets
# ============================================================================

GENERAL_RULES_10 = [
    "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
    "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
    "golden_cross", "trend_break_warning", "death_cross", "seasonality",
]

ENERGY_RULES_14 = GENERAL_RULES_10 + [
    "energy_momentum", "energy_mean_reversion",
    "energy_seasonality", "midstream_yield_reversion",
]

LEAN_RULES_3 = ["trend_continuation", "seasonality", "death_cross"]
ENERGY_LEAN_MOMENTUM = ["energy_momentum", "energy_seasonality", "death_cross"]
ENERGY_LEAN_REVERSION = ["energy_mean_reversion", "energy_seasonality", "death_cross"]
MIDSTREAM_LEAN = ["midstream_yield_reversion", "energy_seasonality", "death_cross"]

SECTOR_RULES = {
    "integrated": ENERGY_LEAN_REVERSION,
    "upstream": ENERGY_LEAN_MOMENTUM,
    "midstream": MIDSTREAM_LEAN,
    "refining": ENERGY_LEAN_REVERSION,
    "renewables": LEAN_RULES_3,  # No energy-specific rules for renewables
}

BASELINE = dict(
    rules=LEAN_RULES_3,
    min_confidence=0.50,
    profit_target=0.10,
    stop_loss=1.0,
    max_loss_pct=5.0,
    cooldown_bars=5,
    timeframe="daily",
    exit_timeframe="daily",
)


def get_baselines(symbol: str) -> list:
    """Get baseline configs for a symbol."""
    sub = SYMBOL_INFO[symbol]["sub_sector"]
    sector_rules = SECTOR_RULES.get(sub, ENERGY_LEAN_MOMENTUM)

    baselines = [
        ("Full general (10 rules, 10%/5%)", dict(
            rules=GENERAL_RULES_10, min_confidence=0.50,
            profit_target=0.10, stop_loss=1.0, max_loss_pct=5.0,
            cooldown_bars=3, timeframe="daily", exit_timeframe="daily",
        )),
        ("Tighter stops (3 rules, 10%/4%)", dict(
            rules=LEAN_RULES_3, min_confidence=0.50,
            profit_target=0.10, stop_loss=1.0, max_loss_pct=4.0,
            cooldown_bars=5, timeframe="daily", exit_timeframe="daily",
        )),
        ("Wider PT (3 rules, 12%/5%)", dict(
            rules=LEAN_RULES_3, min_confidence=0.50,
            profit_target=0.12, stop_loss=1.0, max_loss_pct=5.0,
            cooldown_bars=5, timeframe="daily", exit_timeframe="daily",
        )),
        (f"Energy rules (14, 10%/5%)", dict(
            rules=ENERGY_RULES_14, min_confidence=0.50,
            profit_target=0.10, stop_loss=1.0, max_loss_pct=5.0,
            cooldown_bars=3, timeframe="daily", exit_timeframe="daily",
        )),
        (f"Sector rules ({sub})", dict(
            rules=sector_rules, min_confidence=0.50,
            profit_target=0.10, stop_loss=1.0, max_loss_pct=5.0,
            cooldown_bars=5, timeframe="daily", exit_timeframe="daily",
        )),
        ("Sector rules + 8% PT", dict(
            rules=sector_rules, min_confidence=0.50,
            profit_target=0.08, stop_loss=1.0, max_loss_pct=4.0,
            cooldown_bars=5, timeframe="daily", exit_timeframe="daily",
        )),
    ]

    if sub in ("upstream", "refining"):
        baselines.append(("Sector + wider stops 6%", dict(
            rules=sector_rules, min_confidence=0.50,
            profit_target=0.10, stop_loss=1.0, max_loss_pct=6.0,
            cooldown_bars=5, timeframe="daily", exit_timeframe="daily",
        )))

    return baselines


def get_tune_configs(failed_gates: List[str], base_kwargs: Dict, symbol: str) -> List[tuple]:
    """Generate targeted tuning configs based on failed gates."""
    configs = []
    base_rules = base_kwargs.get("rules", [])
    sub = SYMBOL_INFO[symbol]["sub_sector"]
    sector_rules = SECTOR_RULES.get(sub, ENERGY_LEAN_MOMENTUM)

    if "Walk-Backward" in failed_gates:
        # Walk-backward fails = rules overfit to recent data, don't survive history
        # Try: broader rules, different PT/stop combos, ATR stops
        for pt in [0.08, 0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((f"WB tune: PT={pt:.0%}", {**base_kwargs, "profit_target": pt}))
        for ml in [3.0, 4.0, 6.0]:
            if ml != base_kwargs.get("max_loss_pct"):
                configs.append((f"WB tune: stop={ml}%", {**base_kwargs, "max_loss_pct": ml}))
        configs.append(("WB tune: ATR x2.5", {
            **base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.5,
            "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0,
        }))
        if base_rules != ENERGY_RULES_14:
            configs.append(("WB tune: energy 14 rules", {**base_kwargs, "rules": ENERGY_RULES_14, "cooldown_bars": 3}))
        if base_rules != sector_rules:
            configs.append((f"WB tune: {sub} sector rules", {**base_kwargs, "rules": sector_rules}))

    if "Bootstrap" in failed_gates:
        for mc in [0.40, 0.45, 0.55]:
            if mc != base_kwargs.get("min_confidence"):
                configs.append((f"BS tune: conf={mc}", {**base_kwargs, "min_confidence": mc}))
        configs.append(("BS tune: energy 14 rules", {**base_kwargs, "rules": ENERGY_RULES_14, "cooldown_bars": 3}))
        if "energy_momentum" not in base_rules:
            configs.append(("BS tune: +energy_momentum", {**base_kwargs, "rules": base_rules + ["energy_momentum"]}))

    if "Monte Carlo" in failed_gates:
        for ml in [3.0, 4.0]:
            if ml != base_kwargs.get("max_loss_pct"):
                configs.append((f"MC tune: stop={ml}%", {**base_kwargs, "max_loss_pct": ml}))
        configs.append(("MC tune: ATR x2.0", {
            **base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.0,
            "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0,
        }))

    if "Regime" in failed_gates:
        if "energy_momentum" not in base_rules:
            configs.append(("Reg tune: +energy_momentum", {**base_kwargs, "rules": base_rules + ["energy_momentum"]}))
        if "energy_mean_reversion" not in base_rules:
            configs.append(("Reg tune: +mean_rev", {**base_kwargs, "rules": base_rules + ["energy_mean_reversion"]}))
        configs.append(("Reg tune: energy 14 rules", {**base_kwargs, "rules": ENERGY_RULES_14, "cooldown_bars": 3}))
        for pt in [0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((f"Reg tune: PT={pt:.0%}", {**base_kwargs, "profit_target": pt}))

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
# Validation runner (Walk-Backward replaces Walk-Forward)
# ============================================================================

def run_full_validation(
    runner: BacktraderRunner,
    symbol: str,
    label: str,
    run_kwargs: Dict,
    print_reports: bool = True,
) -> ValidationReport:
    """Run all 4 validation gates: Walk-Backward, Bootstrap, Monte Carlo, Regime."""
    report = ValidationReport(label=label, kwargs=run_kwargs.copy())

    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit(f"[bold]Validating: {label}[/bold]", border_style="cyan"))

    # --- Full-period backtest ---
    console.print("  Running full-period backtest...")
    t0 = time.time()
    try:
        result = runner.run(
            symbol=symbol, start_date=START_DATE, end_date=END_DATE, **run_kwargs
        )
    except Exception as e:
        console.print(f"  [red]Backtest failed: {e}[/red]")
        report.gates = [
            GateResult(g, False, f"Backtest failed: {e}")
            for g in ["Walk-Backward", "Bootstrap", "Monte Carlo", "Regime"]
        ]
        return report

    elapsed = time.time() - t0
    report.backtest = result
    console.print(
        f"  Done in {elapsed:.0f}s -- {result.total_trades} trades, "
        f"WR={result.win_rate:.1%}, Return={result.total_return:+.1%}, "
        f"Sharpe={result.sharpe_ratio or 0:.2f}, "
        f"PF={result.profit_factor or 0:.2f}, "
        f"DD=-{result.max_drawdown_pct or 0:.1f}%"
    )

    if result.total_trades < 2:
        console.print("  [red]Too few trades (<2) -- all gates fail[/red]")
        report.gates = [
            GateResult(g, False, "Too few trades")
            for g in ["Walk-Backward", "Bootstrap", "Monte Carlo", "Regime"]
        ]
        return report

    # --- Gate 1: Walk-Backward (REPLACES Walk-Forward) ---
    console.print(f"\n  [cyan]Gate 1: Walk-Backward (tune {TUNE_MONTHS}mo, holdout {HOLDOUT_MONTHS}mo, walk back)[/cyan]")
    t0 = time.time()
    try:
        wb_validator = WalkBackwardValidator(runner)
        wb_result = wb_validator.validate(
            symbol=symbol,
            data_start=START_DATE,
            data_end=END_DATE,
            tune_months=TUNE_MONTHS,
            holdout_months=HOLDOUT_MONTHS,
            min_regimes_pass=MIN_REGIMES_PASS,
            **run_kwargs,
        )
        elapsed = time.time() - t0

        wb_passed = wb_result.is_valid
        verdict = wb_result.overall_verdict
        wb_detail = (
            f"Verdict={verdict}, "
            f"Holdout={'PASS' if wb_result.holdout_passed else 'FAIL'} "
            f"({wb_result.holdout_result.total_return:+.1%}), "
            f"Regimes={wb_result.regimes_passed}/{wb_result.regimes_total} "
            f"(need {MIN_REGIMES_PASS})"
        )

        # Add per-regime detail
        regime_parts = []
        for w in wb_result.regime_windows:
            status = "PASS" if w.passed else "FAIL"
            regime_parts.append(f"{w.label}: {w.total_return:+.1%} [{status}]")
        if regime_parts:
            wb_detail += " | " + ", ".join(regime_parts)

        report.gates.append(GateResult("Walk-Backward", wb_passed, wb_detail, wb_result))
        status = "[green]PASS[/green]" if wb_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {verdict} ({elapsed:.0f}s)")
        console.print(f"    Tune: {wb_result.tune_result.total_return:+.1%} ({wb_result.tune_result.total_trades} trades)")
        console.print(f"    Holdout: {wb_result.holdout_result.total_return:+.1%} ({wb_result.holdout_result.total_trades} trades)")
        for w in wb_result.regime_windows:
            s = "[green]PASS[/green]" if w.passed else "[red]FAIL[/red]"
            console.print(f"    {w.label}: {w.total_return:+.1%} ({w.total_trades}t) {s}")

        if print_reports:
            print_walk_backward_report(wb_result)
    except Exception as e:
        report.gates.append(GateResult("Walk-Backward", False, f"Error: {e}"))
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
            f"Sharpe CI=[{bs_result.sharpe_ci_lower:.2f}, {bs_result.sharpe_ci_upper:.2f}], "
            f"WR CI=[{bs_result.win_rate_ci_lower:.1%}, {bs_result.win_rate_ci_upper:.1%}]"
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
        mc_result = monte_carlo_analysis(result, n_simulations=10000, initial_cash=INITIAL_CASH)
        elapsed = time.time() - t0
        mc_passed = mc_result.ruin_probability < 0.10 and mc_result.drawdown_p95 < 40
        mc_detail = (
            f"Ruin={mc_result.ruin_probability:.1%}, "
            f"P95 DD=-{mc_result.drawdown_p95:.1f}%, "
            f"Median=${mc_result.equity_median:,.0f}, "
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
        report.gates.append(GateResult("Regime", regime_passed, regime_detail, regime_result))
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
        console.print("  [bold green]ALL GATES PASSED[/bold green]")
    else:
        console.print(f"  [bold red]Failed: {', '.join(report.failed_gates)}[/bold red]")

    return report


# ============================================================================
# Report writer
# ============================================================================

def _write_config_section(f, report: ValidationReport):
    kw = report.kwargs
    f.write(f"### {report.label}\n\n")
    f.write(f"- **Rules:** `{', '.join(kw.get('rules', []))}`\n")
    f.write(f"- **Profit Target:** {kw.get('profit_target', 0.10):.0%}\n")
    f.write(f"- **Min Confidence:** {kw.get('min_confidence', 0.50)}\n")
    f.write(f"- **Max Loss:** {kw.get('max_loss_pct', 5.0)}%\n")
    f.write(f"- **Cooldown:** {kw.get('cooldown_bars', 5)} bars\n")
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
            # Truncate long walk-backward details for table readability
            detail = g.detail
            if len(detail) > 120:
                detail = detail[:117] + "..."
            f.write(f"| {g.name} | {status} | {detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")

        # Write detailed walk-backward breakdown if present
        for g in report.gates:
            if g.name == "Walk-Backward" and g.data and isinstance(g.data, WalkBackwardResult):
                wb = g.data
                f.write("#### Walk-Backward Detail\n\n")
                f.write(f"- **Tune period:** {wb.tune_start} to {wb.tune_end} "
                        f"({wb.tune_result.total_return:+.1%}, {wb.tune_result.total_trades} trades)\n")
                f.write(f"- **Holdout (forward):** {wb.holdout_start} to {wb.holdout_end} "
                        f"({wb.holdout_result.total_return:+.1%}, {wb.holdout_result.total_trades} trades) "
                        f"{'**PASS**' if wb.holdout_passed else 'FAIL'}\n")
                f.write(f"- **Regimes passed:** {wb.regimes_passed}/{wb.regimes_total} (need {wb.min_regimes_pass})\n\n")

                if wb.regime_windows:
                    f.write("| Regime Window | Period | Return | Trades | Dominant | Status |\n")
                    f.write("|---------------|--------|--------|--------|----------|--------|\n")
                    for w in wb.regime_windows:
                        s = "**PASS**" if w.passed else "FAIL"
                        f.write(f"| {w.label} | {w.start} to {w.end} | {w.total_return:+.1%} | "
                                f"{w.total_trades} | {w.actual_dominant_regime} | {s} |\n")
                    f.write("\n")

                f.write(f"**Verdict: {wb.overall_verdict}**\n\n")

    f.write("---\n\n")


def write_report(
    symbol: str,
    info: dict,
    baseline_report: ValidationReport,
    alt_reports: List[ValidationReport],
    tune_reports: List[ValidationReport],
    best_report: Optional[ValidationReport],
    total_elapsed: float,
):
    filepath = os.path.join(
        os.path.expanduser("~/Projects/backtesting-service"),
        f"{symbol.lower()}-validated.md",
    )

    with open(filepath, "w") as f:
        f.write(f"# {symbol} ({info['name']}) Validated Optimization Results\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**Initial Cash:** ${INITIAL_CASH:,}\n")
        f.write(f"**Validation:** Hybrid Walk-Backward (tune {TUNE_MONTHS}mo + holdout {HOLDOUT_MONTHS}mo + historical regimes)\n")
        f.write(f"**Runtime:** {total_elapsed / 60:.1f} minutes\n")
        f.write(f"**Category:** {info['tier']}\n")
        f.write(f"**Bear Beta:** {info['bear_beta']}\n\n")
        f.write("---\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("**Hybrid Walk-Backward Validation** — tune rules on recent market structure, "
                "then validate they survive historical regimes.\n\n")
        f.write("### Validation Gates\n\n")
        f.write("| Gate | Method | Pass Criteria | Purpose |\n")
        f.write("|------|--------|---------------|---------|\n")
        f.write(f"| Walk-Backward | Tune on last {TUNE_MONTHS}mo, holdout {HOLDOUT_MONTHS}mo, "
                f"walk back through 2020-2024 | Holdout profitable + {MIN_REGIMES_PASS}/4 regimes pass | "
                "Rules work in current AND historical markets |\n")
        f.write("| Bootstrap | 10,000 resamples | p < 0.05 AND Sharpe CI excludes zero | Statistical significance |\n")
        f.write("| Monte Carlo | 10,000 trade-order permutations | Ruin < 10% AND P95 DD < 40% | Worst-case risk |\n")
        f.write("| Regime | SPY SMA_50/SMA_200 + VIX | No regime >70% of profit | Not regime-dependent |\n\n")
        f.write("### Historical Regime Windows\n\n")
        f.write("| Window | Period | Expected |\n")
        f.write("|--------|--------|----------|\n")
        f.write("| 2020 Crash + Recovery | Feb 2020 - Dec 2020 | Crisis/Bull |\n")
        f.write("| 2021 Bull | Jan 2021 - Dec 2021 | Bull |\n")
        f.write("| 2022 Bear (Rate Hike) | Jan 2022 - Oct 2022 | Bear |\n")
        f.write("| 2023-2024 Chop | Jan 2023 - Jun 2024 | Chop |\n\n")
        f.write("---\n\n")

        # Baseline screening
        f.write("## 1. Baseline Screening\n\n")
        f.write(f"{symbol} -- {info['description']}. {info['tier']}.\n\n")

        all_screens = [("Lean 3 rules (10%/5%)", baseline_report)]
        baselines = get_baselines(symbol)
        for (name, _), r in zip(baselines, alt_reports):
            all_screens.append((name, r))

        f.write("| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |\n")
        f.write("|--------|--------|----------|--------|--------|-----|--------|\n")
        for name, r in all_screens:
            if r.backtest:
                b = r.backtest
                f.write(
                    f"| {name} | {b.total_trades} | {b.win_rate:.1%} | "
                    f"{b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} | "
                    f"{b.profit_factor or 0:.2f} | -{b.max_drawdown_pct or 0:.1f}% |\n"
                )
            else:
                f.write(f"| {name} | - | - | - | - | - | - |\n")

        all_reports = [baseline_report] + alt_reports
        valid = [r for r in all_reports if r.backtest and r.backtest.total_trades >= 5]
        if valid:
            best_screen = max(valid, key=lambda r: r.backtest.sharpe_ratio or 0)
            f.write(f"\n**Best baseline: {best_screen.label}**\n\n")
        f.write("---\n\n")

        # Full validation
        f.write("## 2. Full Validation (Walk-Backward)\n\n")
        if best_report:
            _write_config_section(f, best_report)
        else:
            _write_config_section(f, baseline_report)

        # Tuning
        f.write("## 3. Tuning Results\n\n")
        if tune_reports:
            quick_only = [r for r in tune_reports if not r.gates]
            validated = [r for r in tune_reports if r.gates]

            if quick_only:
                f.write("### Quick Screen\n\n")
                f.write("| Config | Trades | Win Rate | Return | Sharpe |\n")
                f.write("|--------|--------|----------|--------|--------|\n")
                for r in quick_only:
                    if r.backtest:
                        b = r.backtest
                        f.write(
                            f"| {r.label} | {b.total_trades} | {b.win_rate:.1%} | "
                            f"{b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} |\n"
                        )
                f.write("\n")

            if validated:
                f.write("### Full Validation of Top Candidates\n\n")
                for r in validated:
                    _write_config_section(f, r)
        else:
            f.write("No tuning needed -- baseline validates.\n\n---\n\n")

        # Summary
        f.write("## 4. Summary Table\n\n")
        f.write("| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |\n")
        f.write("|--------|-----|-----|-----|--------|--------|--------|--------|\n")
        all_validated = []
        if best_report and best_report.gates:
            all_validated.append(best_report)
        all_validated.extend([r for r in tune_reports if r.gates])
        all_validated.sort(
            key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
            reverse=True,
        )
        for i, r in enumerate(all_validated):
            bold = "**" if i == 0 else ""
            gates_map = {g.name: g for g in r.gates}
            wb = "**PASS**" if gates_map.get("Walk-Backward", GateResult("", False, "")).passed else "FAIL"
            bs = "**PASS**" if gates_map.get("Bootstrap", GateResult("", False, "")).passed else "FAIL"
            mc = "**PASS**" if gates_map.get("Monte Carlo", GateResult("", False, "")).passed else "FAIL"
            rg = "**PASS**" if gates_map.get("Regime", GateResult("", False, "")).passed else "FAIL"
            b = r.backtest
            if b:
                f.write(
                    f"| {bold}{r.label}{bold} | {wb} | {bs} | {mc} | {rg} | "
                    f"{bold}{b.sharpe_ratio or 0:.2f}{bold} | "
                    f"{bold}{b.total_return:+.1%}{bold} | {b.total_trades} |\n"
                )
        f.write("\n---\n\n")

        # Recommendation
        f.write("## 5. Final Recommendation\n\n")
        final = all_validated[0] if all_validated else baseline_report
        status_word = "fully validates" if final.all_passed else "partially validates"
        f.write(f"**{symbol} {status_word}.** Best config: {final.label} ({final.pass_count}/4 gates).\n\n")
        _write_config_section(f, final)

        f.write("### Deployment Recommendation\n\n")
        if final.pass_count >= 3:
            f.write("- Deploy with standard sizing\n")
        elif final.pass_count >= 2:
            f.write("- Conditional deployment with regime restrictions or reduced sizing\n")
        else:
            f.write("- Consider blacklisting or significant restrictions\n")
        f.write("- Monitor failing gate(s) in live trading\n")
        f.write("- Re-validate after 6 months of additional data\n\n")

    console.print(f"\n  Report: {filepath}")
    return filepath


# ============================================================================
# Main per-symbol runner
# ============================================================================

def run_symbol(symbol: str):
    info = SYMBOL_INFO[symbol]
    total_start = time.time()
    runner = BacktraderRunner(initial_cash=INITIAL_CASH)

    console.print(Panel.fit(
        f"[bold cyan]{symbol} -- Hybrid Walk-Backward Validation[/bold cyan]\n"
        f"{info['name']} -- {info['description']}\n"
        f"{info['tier']} | Bear Beta: {info['bear_beta']}",
        border_style="cyan",
    ))

    # Step 1: Screen baselines
    console.print("\n[bold]STEP 1: Screen Baselines[/bold]")
    console.print(f"\n  Running baseline: {symbol} lean 3 rules...")
    t0 = time.time()
    bl_result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **BASELINE)
    console.print(
        f"  Baseline: {bl_result.total_trades}t, WR={bl_result.win_rate:.1%}, "
        f"Return={bl_result.total_return:+.1%}, Sharpe={bl_result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)"
    )
    baseline_report = ValidationReport(label="Lean 3 rules (10%/5%)", kwargs=BASELINE.copy(), backtest=bl_result)

    alt_reports = []
    for alt_name, alt_kwargs in get_baselines(symbol):
        console.print(f"  Running {alt_name}...")
        t0 = time.time()
        try:
            alt_result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **alt_kwargs)
            console.print(
                f"    {alt_result.total_trades}t, WR={alt_result.win_rate:.1%}, "
                f"Return={alt_result.total_return:+.1%}, Sharpe={alt_result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)"
            )
            alt_reports.append(ValidationReport(label=alt_name, kwargs=alt_kwargs.copy(), backtest=alt_result))
        except Exception as e:
            console.print(f"    [red]Failed: {e}[/red]")
            alt_reports.append(ValidationReport(label=alt_name, kwargs=alt_kwargs.copy()))

    # Pick best by Sharpe (minimum 5 trades)
    all_screens = [baseline_report] + alt_reports
    valid_screens = [r for r in all_screens if r.backtest and r.backtest.total_trades >= 5]
    if valid_screens:
        best_screen = max(valid_screens, key=lambda r: r.backtest.sharpe_ratio or 0)
    else:
        best_screen = baseline_report

    console.print(f"\n  [bold green]Best: {best_screen.label} (Sharpe={best_screen.backtest.sharpe_ratio or 0:.2f})[/bold green]")

    # Step 2: Full validation with walk-backward
    console.print("\n[bold]STEP 2: Full Validation (Walk-Backward)[/bold]")
    best_report = run_full_validation(runner, symbol, best_screen.label, best_screen.kwargs, print_reports=True)

    # Step 3: Diagnose and tune
    console.print("\n[bold]STEP 3: Diagnose & Tune[/bold]")
    tune_reports = []
    if best_report.failed_gates:
        console.print(f"  Failed gates: {', '.join(best_report.failed_gates)}")
        tune_configs = get_tune_configs(best_report.failed_gates, best_screen.kwargs, symbol)
        console.print(f"  Generated {len(tune_configs)} tuning configs")

        # Quick screen
        screen_results = []
        for label, kwargs in tune_configs:
            console.print(f"  Screening: {label}...")
            t0 = time.time()
            try:
                result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **kwargs)
                console.print(
                    f"    {result.total_trades}t, WR={result.win_rate:.1%}, "
                    f"Return={result.total_return:+.1%}, Sharpe={result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)"
                )
                screen_results.append((label, kwargs, result))
            except Exception as e:
                console.print(f"    [red]Failed: {e}[/red]")

        # Top 3 by Sharpe for full validation
        screen_results.sort(
            key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
            reverse=True,
        )
        top_n = screen_results[:3]

        console.print(f"\n  [bold]Top {len(top_n)} for full validation:[/bold]")
        for label, _, result in top_n:
            console.print(f"    {label}: Sharpe={result.sharpe_ratio or 0:.2f}, Return={result.total_return:+.1%}")

        for label, kwargs, _ in top_n:
            report = run_full_validation(runner, symbol, label, kwargs, print_reports=True)
            tune_reports.append(report)

        # Add screen-only results
        for label, kwargs, result in screen_results:
            if not any(r.label == label for r in tune_reports):
                tune_reports.append(ValidationReport(label=label, kwargs=kwargs, backtest=result))
    else:
        console.print("  [green]All gates passed! No tuning needed.[/green]")

    # Step 4: Write report
    total_elapsed = time.time() - total_start
    console.print(f"\n[bold]STEP 4: Write Report[/bold]")
    write_report(symbol, info, baseline_report, alt_reports, tune_reports, best_report, total_elapsed)

    # Final summary
    all_final = [best_report] + [r for r in tune_reports if r.gates]
    all_final.sort(
        key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
        reverse=True,
    )
    winner = all_final[0] if all_final else best_report

    console.print(f"\n{'=' * 70}")
    console.print(f"[bold]{symbol} done in {total_elapsed / 60:.1f} min[/bold]")
    console.print(
        f"  Best: {winner.label} ({winner.pass_count}/4 gates, "
        f"Sharpe={winner.backtest.sharpe_ratio or 0:.2f}, "
        f"Return={winner.backtest.total_return:+.1%})"
    )

    return winner


# ============================================================================
# Batch comparison matrix
# ============================================================================

def write_comparison_matrix(results: Dict[str, ValidationReport]):
    filepath = os.path.join(
        os.path.expanduser("~/Projects/backtesting-service"),
        "energy-bc-comparison.md",
    )

    with open(filepath, "w") as f:
        f.write("# Energy B/C Tier -- Hybrid Walk-Backward Comparison Matrix\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Validation:** Walk-Backward (tune {TUNE_MONTHS}mo, holdout {HOLDOUT_MONTHS}mo, 4 historical regimes)\n\n")
        f.write("---\n\n")

        f.write("## Comparison Matrix\n\n")
        f.write("| Symbol | Tier | Bear Beta | Best Config | Gates | WB | BS | MC | Reg | Sharpe | Return | Trades | WR |\n")
        f.write("|--------|------|-----------|-------------|-------|----|----|----|-----|--------|--------|--------|----|\n")

        for sym in ["CVX", "KMI", "OKE", "COP", "PSX", "FANG", "NEP"]:
            if sym not in results:
                continue
            r = results[sym]
            info = SYMBOL_INFO[sym]
            gates_map = {g.name: g for g in r.gates}

            wb = "PASS" if gates_map.get("Walk-Backward", GateResult("", False, "")).passed else "FAIL"
            bs = "PASS" if gates_map.get("Bootstrap", GateResult("", False, "")).passed else "FAIL"
            mc = "PASS" if gates_map.get("Monte Carlo", GateResult("", False, "")).passed else "FAIL"
            rg = "PASS" if gates_map.get("Regime", GateResult("", False, "")).passed else "FAIL"

            b = r.backtest
            if b:
                f.write(
                    f"| **{sym}** | {info['tier'].split(' -- ')[0]} | {info['bear_beta']} | "
                    f"{r.label} | {r.pass_count}/4 | {wb} | {bs} | {mc} | {rg} | "
                    f"{b.sharpe_ratio or 0:.2f} | {b.total_return:+.1%} | {b.total_trades} | {b.win_rate:.0%} |\n"
                )

        f.write("\n---\n\n")

        # Walk-backward detail per stock
        f.write("## Walk-Backward Detail\n\n")
        for sym in ["CVX", "KMI", "OKE", "COP", "PSX", "FANG", "NEP"]:
            if sym not in results:
                continue
            r = results[sym]
            wb_gate = next((g for g in r.gates if g.name == "Walk-Backward"), None)
            if not wb_gate or not wb_gate.data:
                continue
            wb = wb_gate.data

            f.write(f"### {sym} ({SYMBOL_INFO[sym]['name']})\n\n")
            f.write(f"- Tune: {wb.tune_result.total_return:+.1%} ({wb.tune_result.total_trades}t)\n")
            f.write(f"- Holdout: {wb.holdout_result.total_return:+.1%} ({wb.holdout_result.total_trades}t) "
                    f"{'**PASS**' if wb.holdout_passed else 'FAIL'}\n")
            f.write(f"- Regimes: {wb.regimes_passed}/{wb.regimes_total}\n")
            f.write(f"- Verdict: **{wb.overall_verdict}**\n\n")

            if wb.regime_windows:
                f.write("| Regime | Return | Trades | Status |\n")
                f.write("|--------|--------|--------|--------|\n")
                for w in wb.regime_windows:
                    s = "**PASS**" if w.passed else "FAIL"
                    f.write(f"| {w.label} | {w.total_return:+.1%} | {w.total_trades} | {s} |\n")
                f.write("\n")

        f.write("---\n\n")
        f.write("## Key Findings\n\n")
        f.write("_To be filled after analysis._\n")

    console.print(f"\nComparison matrix: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_energy_bc.py <SYMBOL|ALL>")
        print(f"  B Tier: CVX, KMI, OKE, COP")
        print(f"  C Tier: PSX, FANG, NEP")
        print(f"  ALL: Run all 7 stocks")
        sys.exit(1)

    arg = sys.argv[1].upper()

    if arg == "ALL":
        symbols = ["CVX", "KMI", "OKE", "COP", "PSX", "FANG", "NEP"]
    elif arg in SYMBOL_INFO:
        symbols = [arg]
    else:
        print(f"Unknown: {arg}")
        print(f"Available: {', '.join(SYMBOL_INFO.keys())}, ALL")
        sys.exit(1)

    total_start = time.time()
    results = {}

    for i, sym in enumerate(symbols):
        console.print(f"\n{'#' * 70}")
        console.print(f"[bold]Stock {i+1}/{len(symbols)}: {sym}[/bold]")
        console.print(f"{'#' * 70}")
        try:
            winner = run_symbol(sym)
            results[sym] = winner
        except Exception as e:
            console.print(f"[red]FAILED {sym}: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Write comparison matrix if batch
    if len(symbols) > 1 and results:
        write_comparison_matrix(results)

    total = time.time() - total_start
    console.print(f"\n{'=' * 70}")
    console.print(f"[bold]Total: {len(results)}/{len(symbols)} stocks in {total / 60:.1f} min[/bold]")
    for sym, r in results.items():
        console.print(f"  {sym}: {r.pass_count}/4 gates, Sharpe={r.backtest.sharpe_ratio or 0:.2f}")


if __name__ == "__main__":
    main()
