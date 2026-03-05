#!/usr/bin/env python3
"""
Utilities / Power — Validate-Then-Tune Optimization

New batch: regulated utilities, yieldcos, nuclear/AI power, water, sector ETFs.
Per-stock rule recommendations based on sub-sector research:

- Regulated (SO, D, NEE): mean-reverting, rate-sensitive, RSI 38, tight 7-8% PT
- Water (AWK): ultra-defensive, negative beta, extremely tight RSI range (40)
- Yieldcos (BEP, CWEN): high yield, extreme rate sensitivity, RSI 33
- Nuclear/AI power (CEG, VST): momentum stocks disguised as utilities, high beta
- Sector ETFs (XLU, VPU): diversified, mean-reverting, best position sizing

Key sector insights:
- Utilities are the MOST interest-rate-sensitive equity sector
- Traditional utilities rarely hit RSI 30 — thresholds must be higher (35-42)
- Bollinger Bands + ADX < 20 is the best entry combo for regulated utilities
- SMA_200 is the absolute floor — never buy below it
- Nuclear/AI power (CEG, VST) are NOT traditional utilities — use momentum rules
- Tighter profit targets (7-8%) match the available range for low-vol names

Usage:
    python validate_utilities.py NEE
    python validate_utilities.py SO
    python validate_utilities.py BEP
    python validate_utilities.py AWK
    python validate_utilities.py CEG
    python validate_utilities.py CWEN
    python validate_utilities.py D
    python validate_utilities.py VPU
    python validate_utilities.py VST
    python validate_utilities.py XLU
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

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# ============================================================================
# Configuration
# ============================================================================

START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 2, 28)
INITIAL_CASH = 1000

SYMBOL_INFO = {
    "NEE": {
        "name": "NextEra Energy",
        "description": "Largest US utility, 70% regulated (FP&L) + 30% renewables — very rate-sensitive",
        "tier": "Regulated utility (renewable hybrid)",
    },
    "SO": {
        "name": "Southern Company",
        "description": "Traditional regulated utility (electric + gas) — classic defensive, beta 0.45",
        "tier": "Regulated utility",
    },
    "BEP": {
        "name": "Brookfield Renewable Partners",
        "description": "Pure-play global renewable energy (hydro/wind/solar) — yieldco, 5%+ dividend",
        "tier": "Clean energy yieldco",
    },
    "AWK": {
        "name": "American Water Works",
        "description": "Largest US water utility — negative beta, ultra-defensive, 17yr dividend streak",
        "tier": "Water utility",
    },
    "CEG": {
        "name": "Constellation Energy",
        "description": "Largest US nuclear fleet (21 reactors) — AI data center play, beta 1.1-1.6",
        "tier": "Nuclear / AI power (MOMENTUM)",
    },
    "CWEN": {
        "name": "Clearway Energy",
        "description": "Clean energy yieldco (12+ GW wind/solar/storage) — bond-like cash flows, 5%+ yield",
        "tier": "Clean energy yieldco",
    },
    "D": {
        "name": "Dominion Energy",
        "description": "Large regulated utility (electric + gas) — beta 0.70, frozen dividend risk",
        "tier": "Regulated utility",
    },
    "VPU": {
        "name": "Vanguard Utilities ETF",
        "description": "Broad utilities ETF — broader than XLU, includes small/mid-cap utilities",
        "tier": "Utility sector ETF",
    },
    "VST": {
        "name": "Vistra Corp",
        "description": "Nuclear + natural gas power — AI data center proxy, beta 1.4+, extremely volatile",
        "tier": "Nuclear / AI power (MOMENTUM)",
    },
    "XLU": {
        "name": "Utilities Select Sector SPDR",
        "description": "Large-cap utilities ETF — benchmark, mean-reverting, excellent position sizing",
        "tier": "Utility sector ETF",
    },
}

# ============================================================================
# Rule Sets
# ============================================================================

GENERAL_RULES_10 = [
    "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
    "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
    "golden_cross", "trend_break_warning", "death_cross", "seasonality",
]

LEAN_RULES_3 = ["trend_continuation", "seasonality", "death_cross"]

# Utility-specific rules
UTILITY_RULES_13 = GENERAL_RULES_10 + [
    "utility_mean_reversion", "utility_rate_reversion", "utility_seasonality",
]

# Lean sets by sub-sector character
UTILITY_LEAN_REVERSION = ["utility_mean_reversion", "utility_seasonality", "death_cross"]
UTILITY_LEAN_PULLBACK = ["utility_rate_reversion", "utility_seasonality", "death_cross"]
UTILITY_LEAN_BALANCED = ["utility_mean_reversion", "utility_rate_reversion", "utility_seasonality", "death_cross"]

# Nuclear/AI power stocks should use momentum rules, NOT utility rules
NUCLEAR_LEAN_MOMENTUM = ["trend_continuation", "momentum_reversal", "death_cross"]
NUCLEAR_LEAN_ENERGY = ["energy_momentum", "energy_seasonality", "death_cross"]

# Per-stock recommendations based on research
STOCK_RULE_RECOMMENDATIONS = {
    # Regulated utilities — mean-reverting, low vol
    "NEE": UTILITY_LEAN_BALANCED,     # Hybrid regulated + renewable, both patterns apply
    "SO": UTILITY_LEAN_REVERSION,     # Classic mean-reverter, beta 0.45
    "D": UTILITY_LEAN_REVERSION,      # Regulated, higher risk (div frozen)

    # Water — ultra-defensive
    "AWK": UTILITY_LEAN_REVERSION,    # Negative beta, very tight range

    # Yieldcos — rate-sensitive, higher beta than regulated
    "BEP": UTILITY_LEAN_BALANCED,     # Global renewable, higher beta than regulated
    "CWEN": UTILITY_LEAN_REVERSION,   # Very stable cash flows, bond-like

    # Nuclear/AI power — momentum, NOT utility mean reversion
    "CEG": NUCLEAR_LEAN_MOMENTUM,     # Beta 1.1-1.6, trades like tech
    "VST": NUCLEAR_LEAN_MOMENTUM,     # Beta 1.4+, extremely volatile

    # Sector ETFs
    "XLU": UTILITY_LEAN_BALANCED,     # Diversified, mean-reverting
    "VPU": UTILITY_LEAN_BALANCED,     # Broader than XLU
}

STOCK_SUBSECTOR = {
    "NEE": "regulated", "SO": "regulated", "D": "regulated",
    "AWK": "water",
    "BEP": "yieldco", "CWEN": "yieldco",
    "CEG": "nuclear_power", "VST": "nuclear_power",
    "XLU": "utility_etf", "VPU": "utility_etf",
}

# ============================================================================
# Baseline configs
# ============================================================================

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


def make_multitf(kwargs: dict) -> dict:
    return {**kwargs, "exit_timeframe": "5min"}


def get_alt_baselines(symbol: str) -> list:
    recommended_rules = STOCK_RULE_RECOMMENDATIONS.get(symbol, UTILITY_LEAN_BALANCED)
    sub_sector = STOCK_SUBSECTOR.get(symbol, "regulated")

    baselines = [
        (
            "Alt A: Full general rules (10 rules, 10%/5%)",
            dict(rules=GENERAL_RULES_10, min_confidence=0.50, profit_target=0.10,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=3,
                 timeframe="daily", exit_timeframe="daily"),
        ),
        (
            "Alt B: Tighter stops (3 rules, 10%/4%)",
            dict(rules=LEAN_RULES_3, min_confidence=0.50, profit_target=0.10,
                 stop_loss=1.0, max_loss_pct=4.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ),
        (
            "Alt C: Wider PT (3 rules, 12%/5%)",
            dict(rules=LEAN_RULES_3, min_confidence=0.50, profit_target=0.12,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ),
        (
            f"Alt D: Utility rules ({len(UTILITY_RULES_13)} rules, 10%/5%)",
            dict(rules=UTILITY_RULES_13, min_confidence=0.50, profit_target=0.10,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=3,
                 timeframe="daily", exit_timeframe="daily"),
        ),
        (
            f"Alt E: {sub_sector} lean ({len(recommended_rules)} rules, 10%/5%)",
            dict(rules=recommended_rules, min_confidence=0.50, profit_target=0.10,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ),
    ]

    # Sub-sector-specific Alt F variants
    if sub_sector == "regulated":
        # Tight PT (7-8%) matches low-vol utility range
        baselines.append((
            "Alt F: Regulated tight (7%/4%, conf=0.55, cooldown=7)",
            dict(rules=UTILITY_LEAN_REVERSION, min_confidence=0.55, profit_target=0.07,
                 stop_loss=1.0, max_loss_pct=4.0, cooldown_bars=7,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt G: Regulated moderate (8%/4%)",
            dict(rules=UTILITY_LEAN_BALANCED, min_confidence=0.50, profit_target=0.08,
                 stop_loss=1.0, max_loss_pct=4.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))
    elif sub_sector == "water":
        # Ultra-tight for ultra-defensive stock
        baselines.append((
            "Alt F: Water ultra-tight (6%/3%, conf=0.55)",
            dict(rules=UTILITY_LEAN_REVERSION, min_confidence=0.55, profit_target=0.06,
                 stop_loss=1.0, max_loss_pct=3.0, cooldown_bars=7,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt G: Water moderate (8%/4%)",
            dict(rules=UTILITY_LEAN_BALANCED, min_confidence=0.50, profit_target=0.08,
                 stop_loss=1.0, max_loss_pct=4.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))
    elif sub_sector == "yieldco":
        # Yieldcos: wider swings than regulated, use midstream-like params
        baselines.append((
            "Alt F: Yieldco reversion (8%/5%, conf=0.50)",
            dict(rules=UTILITY_LEAN_REVERSION, min_confidence=0.50, profit_target=0.08,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt G: Yieldco + midstream rule (10%/5%)",
            dict(rules=["utility_mean_reversion", "midstream_yield_reversion",
                         "utility_seasonality", "death_cross"],
                 min_confidence=0.50, profit_target=0.10,
                 stop_loss=1.0, max_loss_pct=5.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))
    elif sub_sector == "nuclear_power":
        # Momentum stocks — wide PT, wider stops
        baselines.append((
            "Alt F: Nuclear momentum (15%/8%)",
            dict(rules=NUCLEAR_LEAN_MOMENTUM, min_confidence=0.50, profit_target=0.15,
                 stop_loss=1.0, max_loss_pct=8.0, cooldown_bars=3,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt G: Nuclear energy rules (12%/7%)",
            dict(rules=NUCLEAR_LEAN_ENERGY, min_confidence=0.50, profit_target=0.12,
                 stop_loss=1.0, max_loss_pct=7.0, cooldown_bars=3,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt H: Nuclear wide (20%/10%)",
            dict(rules=NUCLEAR_LEAN_MOMENTUM, min_confidence=0.50, profit_target=0.20,
                 stop_loss=1.0, max_loss_pct=10.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))
    elif sub_sector == "utility_etf":
        # ETFs: tighter params, more trades expected
        baselines.append((
            "Alt F: ETF tight (6%/3%, conf=0.55)",
            dict(rules=UTILITY_LEAN_REVERSION, min_confidence=0.55, profit_target=0.06,
                 stop_loss=1.0, max_loss_pct=3.0, cooldown_bars=3,
                 timeframe="daily", exit_timeframe="daily"),
        ))
        baselines.append((
            "Alt G: ETF moderate (8%/4%)",
            dict(rules=UTILITY_LEAN_BALANCED, min_confidence=0.50, profit_target=0.08,
                 stop_loss=1.0, max_loss_pct=4.0, cooldown_bars=5,
                 timeframe="daily", exit_timeframe="daily"),
        ))

    return baselines


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

def run_full_validation(runner, symbol, label, run_kwargs, print_reports=True):
    report = ValidationReport(label=label, kwargs=run_kwargs.copy())
    console.print(f"\n{'=' * 70}")
    console.print(Panel.fit(f"[bold]Validating: {label}[/bold]", border_style="cyan"))

    console.print("  Running full-period backtest...")
    t0 = time.time()
    try:
        result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **run_kwargs)
    except Exception as e:
        console.print(f"  [red]Backtest failed: {e}[/red]")
        report.gates = [GateResult(g, False, f"Backtest failed: {e}")
                        for g in ["Walk-Forward", "Bootstrap", "Monte Carlo", "Regime"]]
        return report

    elapsed = time.time() - t0
    report.backtest = result
    console.print(f"  Done in {elapsed:.0f}s — {result.total_trades} trades, "
                  f"WR={result.win_rate:.1%}, Return={result.total_return:+.1%}, "
                  f"Sharpe={result.sharpe_ratio or 0:.2f}, PF={result.profit_factor or 0:.2f}, "
                  f"DD=-{result.max_drawdown_pct or 0:.1f}%")

    if result.total_trades < 2:
        console.print("  [red]Too few trades (<2) — all gates fail[/red]")
        report.gates = [GateResult(g, False, "Too few trades")
                        for g in ["Walk-Forward", "Bootstrap", "Monte Carlo", "Regime"]]
        return report

    # Gate 1: Walk-Forward
    console.print("\n  [cyan]Gate 1: Walk-Forward Validation[/cyan]")
    t0 = time.time()
    try:
        validator = WalkForwardValidator(runner)
        wf_result = validator.validate_simple(symbol, START_DATE, END_DATE,
                                              train_pct=0.7, embargo_days=5, purge_days=10, **run_kwargs)
        elapsed = time.time() - t0
        wf_passed = not wf_result.overall_overfit
        if wf_result.windows:
            w = wf_result.windows[0]
            train_s = w.train_result.sharpe_ratio or 0
            test_s = w.test_result.sharpe_ratio or 0
            ratio = test_s / train_s if train_s > 0 else 0
            wf_detail = f"Train Sharpe={train_s:.2f}, Test Sharpe={test_s:.2f}, Ratio={ratio:.0%} (need >=50%)"
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

    # Gate 2: Bootstrap
    console.print("\n  [cyan]Gate 2: Bootstrap Significance (10,000 samples)[/cyan]")
    t0 = time.time()
    try:
        bs_result = bootstrap_analysis(result, n_bootstrap=10000)
        elapsed = time.time() - t0
        bs_passed = bs_result.p_value < 0.05 and not bs_result.no_edge_sharpe
        bs_detail = (f"p={bs_result.p_value:.4f}, Sharpe CI=[{bs_result.sharpe_ci_lower:.2f}, "
                     f"{bs_result.sharpe_ci_upper:.2f}], WR CI=[{bs_result.win_rate_ci_lower:.1%}, "
                     f"{bs_result.win_rate_ci_upper:.1%}]")
        report.gates.append(GateResult("Bootstrap", bs_passed, bs_detail, bs_result))
        status = "[green]PASS[/green]" if bs_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {bs_detail} ({elapsed:.1f}s)")
        if print_reports:
            print_bootstrap_report(bs_result)
    except Exception as e:
        report.gates.append(GateResult("Bootstrap", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # Gate 3: Monte Carlo
    console.print("\n  [cyan]Gate 3: Monte Carlo (10,000 simulations)[/cyan]")
    t0 = time.time()
    try:
        mc_result = monte_carlo_analysis(result, n_simulations=10000, initial_cash=INITIAL_CASH)
        elapsed = time.time() - t0
        mc_passed = mc_result.ruin_probability < 0.10 and mc_result.drawdown_p95 < 40
        mc_detail = (f"Ruin={mc_result.ruin_probability:.1%}, P95 DD=-{mc_result.drawdown_p95:.1f}%, "
                     f"Median equity=${mc_result.equity_median:,.0f}, Survival={mc_result.survival_rate:.1%}")
        report.gates.append(GateResult("Monte Carlo", mc_passed, mc_detail, mc_result))
        status = "[green]PASS[/green]" if mc_passed else "[red]FAIL[/red]"
        console.print(f"  {status}: {mc_detail} ({elapsed:.1f}s)")
        if print_reports:
            print_monte_carlo_report(mc_result)
    except Exception as e:
        report.gates.append(GateResult("Monte Carlo", False, f"Error: {e}"))
        console.print(f"  [red]ERROR: {e}[/red]")

    # Gate 4: Regime
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

    console.print(f"\n  [bold]{'=' * 50}[/bold]")
    console.print(f"  [bold]Verdict: {report.pass_count}/4 gates passed[/bold]")
    if report.all_passed:
        console.print("  [bold green]ALL GATES PASSED — CONFIG IS VALIDATED[/bold green]")
    else:
        console.print(f"  [bold red]Failed gates: {', '.join(report.failed_gates)}[/bold red]")

    return report


# ============================================================================
# Diagnosis and tuning
# ============================================================================

def generate_tune_configs(failed_gates, base_kwargs, symbol=""):
    configs = []
    base_rules = base_kwargs.get("rules", [])
    recommended = STOCK_RULE_RECOMMENDATIONS.get(symbol, UTILITY_LEAN_BALANCED)
    sub_sector = STOCK_SUBSECTOR.get(symbol, "regulated")

    if "Walk-Forward" in failed_gates:
        for pt in [0.07, 0.08, 0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((f"WF tune: PT={int(pt*100)}%", {**base_kwargs, "profit_target": pt}))
        for mc in [0.45, 0.55, 0.60, 0.65]:
            if mc != base_kwargs.get("min_confidence"):
                configs.append((f"WF tune: conf={mc}", {**base_kwargs, "min_confidence": mc}))
        for cb in [3, 7]:
            if cb != base_kwargs.get("cooldown_bars"):
                configs.append((f"WF tune: cooldown={cb}", {**base_kwargs, "cooldown_bars": cb}))
        configs.append(("WF tune: ATR stops x2.5",
                        {**base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.5,
                         "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0}))
        if "utility_mean_reversion" not in base_rules:
            configs.append(("WF tune: + utility_mean_reversion",
                            {**base_kwargs, "rules": base_rules + ["utility_mean_reversion"]}))
        if "utility_rate_reversion" not in base_rules:
            configs.append(("WF tune: + utility_rate_reversion",
                            {**base_kwargs, "rules": base_rules + ["utility_rate_reversion"]}))

    if "Bootstrap" in failed_gates:
        for mc in [0.40, 0.45, 0.55]:
            if mc != base_kwargs.get("min_confidence"):
                configs.append((f"BS tune: conf={mc}", {**base_kwargs, "min_confidence": mc}))
        for cb in [3, 7]:
            if cb != base_kwargs.get("cooldown_bars"):
                configs.append((f"BS tune: cooldown={cb}", {**base_kwargs, "cooldown_bars": cb}))
        configs.append(("BS tune: full rules (10)",
                        {**base_kwargs, "rules": GENERAL_RULES_10, "min_confidence": 0.50, "cooldown_bars": 3}))
        configs.append(("BS tune: utility rules (13)",
                        {**base_kwargs, "rules": UTILITY_RULES_13, "min_confidence": 0.50, "cooldown_bars": 3}))
        if recommended != base_rules:
            configs.append((f"BS tune: {sub_sector} rules",
                            {**base_kwargs, "rules": recommended, "min_confidence": 0.50}))

    if "Monte Carlo" in failed_gates:
        for ml in [3.0, 4.0]:
            if ml != base_kwargs.get("max_loss_pct"):
                configs.append((f"MC tune: max_loss={ml}%", {**base_kwargs, "max_loss_pct": ml}))
        configs.append(("MC tune: ATR stops x2.0",
                        {**base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.0,
                         "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0}))

    if "Regime" in failed_gates:
        if "utility_mean_reversion" not in base_rules:
            configs.append(("Regime tune: + utility_mean_reversion",
                            {**base_kwargs, "rules": base_rules + ["utility_mean_reversion"]}))
        if "utility_rate_reversion" not in base_rules:
            configs.append(("Regime tune: + utility_rate_reversion",
                            {**base_kwargs, "rules": base_rules + ["utility_rate_reversion"]}))
        configs.append(("Regime tune: utility rules (13)",
                        {**base_kwargs, "rules": UTILITY_RULES_13, "min_confidence": 0.50, "cooldown_bars": 3}))
        configs.append(("Regime tune: full rules (10)",
                        {**base_kwargs, "rules": GENERAL_RULES_10, "min_confidence": 0.50, "cooldown_bars": 3}))
        configs.append(("Regime tune: conf=0.65", {**base_kwargs, "min_confidence": 0.65}))
        for pt in [0.08, 0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((f"Regime tune: PT={int(pt*100)}%", {**base_kwargs, "profit_target": pt}))
        if base_kwargs.get("max_loss_pct", 5.0) > 4.0:
            configs.append(("Regime tune: tighter stop 4%", {**base_kwargs, "max_loss_pct": 4.0}))

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

def _write_config_section(f, report):
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
        f.write(f"**Performance:** Return={b.total_return:+.1%}, Trades={b.total_trades}, "
                f"WR={b.win_rate:.1%}, Sharpe={b.sharpe_ratio or 0:.2f}, "
                f"PF={b.profit_factor or 0:.2f}, DD=-{b.max_drawdown_pct or 0:.1f}%\n\n")
    if report.gates:
        f.write("| Gate | Status | Detail |\n|------|--------|--------|\n")
        for g in report.gates:
            status = "**PASS**" if g.passed else "FAIL"
            f.write(f"| {g.name} | {status} | {g.detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")
    f.write("---\n\n")


def write_markdown_report(symbol, info, baseline_report, alt_reports, tune_reports, best_report, total_elapsed):
    filepath = os.path.join(os.path.expanduser("~/Projects/backtesting-service"),
                            f"{symbol.lower()}-validated.md")
    with open(filepath, "w") as f:
        f.write(f"# {symbol} ({info['name']}) Validated Optimization Results\n\n")
        f.write(f"**Date:** {date.today()}\n**Period:** {START_DATE} to {END_DATE}\n")
        f.write(f"**Initial Cash:** ${INITIAL_CASH:,}\n**Timeframe:** Daily-only screening + multi-TF re-validation\n")
        f.write(f"**Validation Runtime:** {total_elapsed / 60:.1f} minutes\n**Category:** {info['tier']}\n\n---\n\n")

        f.write("## Methodology\n\nValidate-then-tune approach with daily-only screening for speed, multi-TF re-validation for final config.\n\n")
        f.write("### Validation Gates\n\n| Gate | Method | Pass Criteria | Purpose |\n|------|--------|---------------|---------|\n")
        f.write("| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting |\n")
        f.write("| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance |\n")
        f.write("| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures worst-case risk |\n")
        f.write("| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects regime dependency |\n\n---\n\n")

        f.write("## 1. Baseline Screening\n\n")
        f.write(f"{symbol} — {info['description']}. {info['tier']}.\n\n")
        alt_baselines = get_alt_baselines(symbol)
        all_screens = [("Lean 3 rules baseline (10%/5%, conf=0.50)", baseline_report)] + [
            (name, r) for name, r in zip([a[0] for a in alt_baselines], alt_reports)]
        f.write("| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |\n|--------|--------|----------|--------|--------|-----|--------|\n")
        for name, r in all_screens:
            if r.backtest:
                b = r.backtest
                f.write(f"| {name} | {b.total_trades} | {b.win_rate:.1%} | {b.total_return:+.1%} | "
                        f"{b.sharpe_ratio or 0:.2f} | {b.profit_factor or 0:.2f} | -{b.max_drawdown_pct or 0:.1f}% |\n")
            else:
                f.write(f"| {name} | - | - | - | - | - | - |\n")

        all_reports = [baseline_report] + alt_reports
        valid = [r for r in all_reports if r.backtest and r.backtest.total_trades >= 10]
        if valid:
            best_screen = max(valid, key=lambda r: r.backtest.sharpe_ratio or 0)
            f.write(f"\n**Best baseline selected for validation: {best_screen.label}**\n\n")
        else:
            f.write(f"\n**Best baseline selected for validation: {baseline_report.label}**\n\n")
        f.write("---\n\n")

        f.write("## 2. Full Validation\n\n")
        _write_config_section(f, best_report if best_report else baseline_report)

        f.write("## 3. Tuning Results\n\n")
        if tune_reports:
            f.write("### Quick Screen\n\n| Config | Trades | Win Rate | Return | Sharpe |\n|--------|--------|----------|--------|--------|\n")
            for r in tune_reports:
                if r.backtest:
                    b = r.backtest
                    f.write(f"| {r.label} | {b.total_trades} | {b.win_rate:.1%} | {b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} |\n")
            f.write("\n### Full Validation of Top Candidates\n\n")
            for r in tune_reports:
                if r.gates:
                    _write_config_section(f, r)
        else:
            f.write("No tuning needed — baseline validates.\n\n---\n\n")

        f.write("## 4. Summary Table\n\n| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |\n|--------|-----|-----|-----|--------|--------|--------|--------|\n")
        all_validated = []
        if best_report and best_report.gates:
            all_validated.append(best_report)
        all_validated.extend([r for r in tune_reports if r.gates])
        all_validated.sort(key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0), reverse=True)
        for i, r in enumerate(all_validated):
            bold = "**" if i == 0 else ""
            gates_map = {g.name: g for g in r.gates}
            wf = "**PASS**" if gates_map.get("Walk-Forward", GateResult("", False, "")).passed else "FAIL"
            bs = "**PASS**" if gates_map.get("Bootstrap", GateResult("", False, "")).passed else "FAIL"
            mc = "**PASS**" if gates_map.get("Monte Carlo", GateResult("", False, "")).passed else "FAIL"
            rg = "**PASS**" if gates_map.get("Regime", GateResult("", False, "")).passed else "FAIL"
            b = r.backtest
            if b:
                f.write(f"| {bold}{r.label}{bold} | {wf} | {bs} | {mc} | {rg} | "
                        f"{bold}{b.sharpe_ratio or 0:.2f}{bold} | {bold}{b.total_return:+.1%}{bold} | {b.total_trades} |\n")
        f.write("\n---\n\n")

        f.write("## 5. Final Recommendation\n\n")
        final = all_validated[0] if all_validated else baseline_report
        status_word = "fully validates" if final.all_passed else "partially validates"
        f.write(f"**{symbol} {status_word}.** Best config: {final.label} ({final.pass_count}/4 gates).\n\n")
        _write_config_section(f, final)
        f.write("### Deployment Recommendation\n\n")
        if final.pass_count >= 3:
            f.write("- Full deployment with no restrictions\n")
        elif final.pass_count >= 2:
            f.write("- Conditional deployment with regime restrictions and/or reduced sizing\n")
        else:
            f.write("- Consider blacklisting or significant restrictions\n")
        f.write("- Monitor the failing gate(s) in live trading\n- Re-validate after 6 months of additional data\n\n")

    console.print(f"\n  Report written to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_utilities.py <SYMBOL>")
        print(f"  Symbols: {', '.join(SYMBOL_INFO.keys())}")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    if symbol not in SYMBOL_INFO:
        print(f"Unknown symbol: {symbol}")
        print(f"Available: {', '.join(SYMBOL_INFO.keys())}")
        sys.exit(1)

    info = SYMBOL_INFO[symbol]
    total_start = time.time()
    runner = BacktraderRunner(initial_cash=INITIAL_CASH)

    console.print(Panel.fit(
        f"[bold cyan]{symbol} — Validate-Then-Tune[/bold cyan]\n"
        f"{info['name']} — {info['description']}\nCategory: {info['tier']}",
        border_style="cyan"))

    # Step 1: Screen baselines
    console.print("\n[bold]STEP 1: Screen Baselines[/bold]")
    console.print(f"\n  Running baseline: {symbol} (3 rules, 10%/5%, conf=0.50)...")
    t0 = time.time()
    bl_result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **BASELINE)
    console.print(f"  Baseline: {bl_result.total_trades} trades, WR={bl_result.win_rate:.1%}, "
                  f"Return={bl_result.total_return:+.1%}, Sharpe={bl_result.sharpe_ratio or 0:.2f}, "
                  f"PF={bl_result.profit_factor or 0:.2f}, DD=-{bl_result.max_drawdown_pct or 0:.1f}% ({time.time()-t0:.0f}s)")
    baseline_report = ValidationReport(label="Lean 3 rules baseline (10%/5%, conf=0.50)",
                                       kwargs=BASELINE.copy(), backtest=bl_result)

    alt_reports = []
    for alt_name, alt_kwargs in get_alt_baselines(symbol):
        console.print(f"\n  Running {alt_name}...")
        t0 = time.time()
        try:
            alt_result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **alt_kwargs)
            console.print(f"  {alt_name}: {alt_result.total_trades} trades, WR={alt_result.win_rate:.1%}, "
                          f"Return={alt_result.total_return:+.1%}, Sharpe={alt_result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)")
            alt_reports.append(ValidationReport(label=alt_name, kwargs=alt_kwargs.copy(), backtest=alt_result))
        except Exception as e:
            console.print(f"  [red]{alt_name} failed: {e}[/red]")
            alt_reports.append(ValidationReport(label=alt_name, kwargs=alt_kwargs.copy()))

    all_screens = [baseline_report] + alt_reports
    valid_screens = [r for r in all_screens if r.backtest and r.backtest.total_trades >= 10]
    if valid_screens:
        best_screen = max(valid_screens, key=lambda r: r.backtest.sharpe_ratio or 0)
    else:
        valid_screens_5 = [r for r in all_screens if r.backtest and r.backtest.total_trades >= 5]
        best_screen = max(valid_screens_5, key=lambda r: r.backtest.sharpe_ratio or 0) if valid_screens_5 else baseline_report

    console.print(f"\n  [bold green]Best by Sharpe: {best_screen.label} "
                  f"(Sharpe={best_screen.backtest.sharpe_ratio or 0:.2f})[/bold green]")

    # Step 2: Full validation
    console.print("\n[bold]STEP 2: Full Validation of Best Baseline (daily-only)[/bold]")
    best_report = run_full_validation(runner, symbol, best_screen.label, best_screen.kwargs, print_reports=True)

    # Step 3: Diagnose and tune
    console.print("\n[bold]STEP 3: Diagnose & Targeted Tune[/bold]")
    tune_reports = []
    if best_report.failed_gates:
        console.print(f"  Failed gates: {', '.join(best_report.failed_gates)}")
        tune_configs = generate_tune_configs(best_report.failed_gates, best_screen.kwargs, symbol=symbol)
        console.print(f"  Generated {len(tune_configs)} tuning configs")

        screen_results = []
        for label, kwargs in tune_configs:
            console.print(f"\n  Screening: {label}...")
            t0 = time.time()
            try:
                result = runner.run(symbol=symbol, start_date=START_DATE, end_date=END_DATE, **kwargs)
                console.print(f"  {result.total_trades} trades, WR={result.win_rate:.1%}, "
                              f"Return={result.total_return:+.1%}, Sharpe={result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)")
                screen_results.append((label, kwargs, result))
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/red]")

        screen_results.sort(key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999, reverse=True)
        top_n = screen_results[:3]
        console.print(f"\n  [bold]Top {len(top_n)} candidates for full validation:[/bold]")
        for label, kwargs, result in top_n:
            console.print(f"    {label}: Sharpe={result.sharpe_ratio or 0:.2f}, Return={result.total_return:+.1%}")

        for label, kwargs, _ in top_n:
            report = run_full_validation(runner, symbol, label, kwargs, print_reports=True)
            tune_reports.append(report)

        for label, kwargs, result in screen_results:
            if not any(r.label == label for r in tune_reports):
                tune_reports.append(ValidationReport(label=label, kwargs=kwargs, backtest=result))
    else:
        console.print("  [green]All gates passed! No tuning needed.[/green]")

    # Step 4: Multi-TF re-validation
    all_candidates = [best_report] + [r for r in tune_reports if r.gates]
    all_candidates.sort(key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0), reverse=True)
    winner = all_candidates[0] if all_candidates else best_report

    if winner.pass_count >= 2:
        console.print("\n[bold]STEP 4: Multi-TF Re-validation (daily entries + 5min exits)[/bold]")
        console.print(f"  Re-validating winner: {winner.label}")
        mtf_kwargs = make_multitf(winner.kwargs)
        mtf_report = run_full_validation(runner, symbol, f"{winner.label} [multi-TF]", mtf_kwargs, print_reports=True)
        if mtf_report.backtest and mtf_report.backtest.total_trades >= 2:
            tune_reports.append(mtf_report)
            console.print(f"\n  Multi-TF result: {mtf_report.pass_count}/4 gates, "
                          f"Sharpe={mtf_report.backtest.sharpe_ratio or 0:.2f}")
        else:
            console.print("  [yellow]Multi-TF produced too few trades — using daily-only results[/yellow]")
    else:
        console.print("\n  [yellow]Winner has <2 gates — skipping multi-TF re-validation[/yellow]")

    # Step 5: Write report
    total_elapsed = time.time() - total_start
    console.print(f"\n[bold]STEP 5: Write Report[/bold]")
    write_markdown_report(symbol, info, baseline_report, alt_reports, tune_reports, best_report, total_elapsed)

    console.print(f"\n{'=' * 70}")
    console.print(f"[bold]Total runtime: {total_elapsed / 60:.1f} minutes[/bold]")
    all_final = [best_report] + [r for r in tune_reports if r.gates]
    if all_final:
        all_final.sort(key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0), reverse=True)
        winner = all_final[0]
        console.print(f"\n[bold green]BEST: {winner.label}[/bold green]"
                      f"\n  Gates: {winner.pass_count}/4\n  Sharpe: {winner.backtest.sharpe_ratio or 0:.2f}"
                      f"\n  Return: {winner.backtest.total_return:+.1%}\n  Trades: {winner.backtest.total_trades}"
                      f"\n  WR: {winner.backtest.win_rate:.1%}")


if __name__ == "__main__":
    main()
