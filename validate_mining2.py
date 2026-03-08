#!/usr/bin/env python3
"""
Mining/Materials Extras — Validate-Then-Tune Optimization

Second batch of mining/materials stocks: copper miners, gold/silver ETFs & miners,
uranium ETF, rare earth ETF, and oilfield services.

Per-stock rule recommendations based on sub-sector research:
- Copper (FCX, COPX): momentum-driven, high beta, ADX confirmation essential
- Gold ETF (GLD): low-vol commodity, tight PT (6-8%), too expensive for small accounts
- Gold miners (GDX, AEM): leveraged gold, GDX/GLD ratio mean-reversion is primary edge
- Silver miners (SIL, PAAS): amplified silver, -83% max DD, strict risk controls
- Uranium ETF (URA): CCJ-proven rules transfer directly (trend+seasonality+death_cross)
- Rare earth (REMX): MP-proven approach, spike trading, volume breakout
- Oilfield services (FET): NOT mining — uses energy rules, capex cycle-driven

Usage:
    python validate_mining2.py FCX
    python validate_mining2.py GDX
    python validate_mining2.py GLD
    python validate_mining2.py COPX
    python validate_mining2.py SIL
    python validate_mining2.py PAAS
    python validate_mining2.py AEM
    python validate_mining2.py URA
    python validate_mining2.py REMX
    python validate_mining2.py FET
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

START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 2, 28)
INITIAL_CASH = 1000

# Symbol descriptions
SYMBOL_INFO = {
    "FCX": {
        "name": "Freeport-McMoRan",
        "description": "Largest publicly traded copper miner — Grasberg mine, gold/moly byproducts",
        "tier": "Large-cap copper miner",
    },
    "COPX": {
        "name": "Global X Copper Miners ETF",
        "description": "Basket of ~30 global copper miners — diversified copper exposure",
        "tier": "Copper miners ETF",
    },
    "GDX": {
        "name": "VanEck Gold Miners ETF",
        "description": "Basket of ~50 senior gold miners — 2-3x leveraged to gold price",
        "tier": "Gold miners ETF",
    },
    "GLD": {
        "name": "SPDR Gold Trust",
        "description": "Physical gold commodity ETF — tracks gold spot price directly",
        "tier": "Gold commodity ETF (WARNING: ~$250/share, too expensive for $1,000 account)",
    },
    "SIL": {
        "name": "Global X Silver Miners ETF",
        "description": "Basket of ~30 silver miners — amplified silver exposure, -83% max DD historically",
        "tier": "Silver miners ETF (HIGH RISK)",
    },
    "PAAS": {
        "name": "Pan American Silver",
        "description": "Largest primary silver producer in Americas — also gold, zinc, lead",
        "tier": "Large-cap silver miner",
    },
    "AEM": {
        "name": "Agnico Eagle Mines",
        "description": "Senior gold producer — tier-1 jurisdictions (Canada, Australia, Finland), low-cost",
        "tier": "Large-cap gold miner",
    },
    "URA": {
        "name": "Global X Uranium ETF",
        "description": "Basket of uranium miners and nuclear component manufacturers",
        "tier": "Uranium ETF",
    },
    "REMX": {
        "name": "VanEck Rare Earth/Strategic Metals ETF",
        "description": "Rare earth and strategic metals — geopolitical spike trading, China supply risk",
        "tier": "Rare earth ETF",
    },
    "FET": {
        "name": "Forum Energy Technologies",
        "description": "Oilfield services equipment — capex cycle play, NOT mining",
        "tier": "Small-cap oilfield services",
    },
}

# ============================================================================
# Rule Sets
# ============================================================================

# General rules (no sector-specific rules)
GENERAL_RULES_10 = [
    "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
    "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
    "golden_cross", "trend_break_warning", "death_cross", "seasonality",
]

# Mining-specific rules
MINING_RULES = [
    "commodity_breakout", "miner_metal_ratio", "dollar_weakness",
    "seasonality", "volume_breakout",
]

# Full mining set: general + mining-specific
FULL_MINING_14 = GENERAL_RULES_10 + [
    "commodity_breakout", "miner_metal_ratio", "volume_breakout", "dollar_weakness",
]

# Lean baselines
LEAN_RULES_3 = ["trend_continuation", "seasonality", "death_cross"]

# Mining lean sets by sub-sector
MINING_LEAN_MOMENTUM = ["commodity_breakout", "trend_continuation", "seasonality", "death_cross"]
MINING_LEAN_RATIO = ["miner_metal_ratio", "trend_continuation", "seasonality", "death_cross"]
MINING_LEAN_VOLUME = ["volume_breakout", "trend_continuation", "seasonality", "death_cross"]

# Energy rules for FET (oilfield services — NOT mining)
ENERGY_LEAN_MOMENTUM = ["energy_momentum", "energy_seasonality", "death_cross"]

# Per-stock recommended rule sets based on research
STOCK_RULE_RECOMMENDATIONS = {
    # Copper miners: momentum-driven, high beta, need ADX/golden cross
    "FCX": MINING_LEAN_MOMENTUM,
    "COPX": MINING_LEAN_MOMENTUM,

    # Gold miners: GDX/GLD ratio mean-reversion + breakout
    "GDX": MINING_LEAN_RATIO,
    "AEM": MINING_LEAN_MOMENTUM,  # Senior producer, trends more smoothly

    # Gold commodity: tight params, low vol
    "GLD": LEAN_RULES_3,  # Low vol, tight PT — generic rules work

    # Silver miners: high risk, volume confirmation critical
    "SIL": MINING_LEAN_VOLUME,
    "PAAS": MINING_LEAN_VOLUME,

    # Uranium: CCJ-proven rules
    "URA": LEAN_RULES_3,  # trend_continuation + seasonality + death_cross

    # Rare earth: spike trading, volume breakout
    "REMX": MINING_LEAN_VOLUME,

    # Oilfield services: energy rules, NOT mining
    "FET": ENERGY_LEAN_MOMENTUM,
}

# Sub-sector classification
STOCK_SUBSECTOR = {
    "FCX": "copper", "COPX": "copper",
    "GDX": "gold_miner", "AEM": "gold_miner",
    "GLD": "gold_commodity",
    "SIL": "silver_miner", "PAAS": "silver_miner",
    "URA": "uranium",
    "REMX": "rare_earth",
    "FET": "oilfield_services",
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
    """Convert a daily-only config to multi-TF (daily entries + 5min exits)."""
    return {**kwargs, "exit_timeframe": "5min"}


def get_alt_baselines(symbol: str) -> list:
    """Get alternative baselines including stock-specific mining rules."""
    recommended_rules = STOCK_RULE_RECOMMENDATIONS.get(symbol, MINING_LEAN_MOMENTUM)
    sub_sector = STOCK_SUBSECTOR.get(symbol, "copper")

    baselines = [
        (
            "Alt A: Full general rules (10 rules, 10%/5%)",
            dict(
                rules=GENERAL_RULES_10,
                min_confidence=0.50,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=3,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ),
        (
            "Alt B: Tighter stops (3 rules, 10%/4%)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.50,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=4.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ),
        (
            "Alt C: Wider PT (3 rules, 12%/5%)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.50,
                profit_target=0.12,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ),
        (
            f"Alt D: Full mining rules ({len(FULL_MINING_14)} rules, 10%/5%)",
            dict(
                rules=FULL_MINING_14,
                min_confidence=0.50,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=3,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ),
        (
            f"Alt E: {sub_sector} rules ({len(recommended_rules)} rules, 10%/5%)",
            dict(
                rules=recommended_rules,
                min_confidence=0.50,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ),
    ]

    # Sub-sector-specific Alt F variants
    if sub_sector == "copper":
        # Copper: wider PT (momentum runs), wider stops (volatile)
        baselines.append((
            "Alt F: Copper momentum (12%/6%)",
            dict(
                rules=MINING_LEAN_MOMENTUM,
                min_confidence=0.50,
                profit_target=0.12,
                stop_loss=1.0,
                max_loss_pct=6.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "gold_commodity":
        # GLD: tight PT and tight stops (low volatility)
        baselines.append((
            "Alt F: Gold commodity tight (6%/3%)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.55,
                profit_target=0.06,
                stop_loss=1.0,
                max_loss_pct=3.0,
                cooldown_bars=3,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
        baselines.append((
            "Alt G: Gold commodity (8%/4%)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.50,
                profit_target=0.08,
                stop_loss=1.0,
                max_loss_pct=4.0,
                cooldown_bars=3,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "gold_miner":
        # Gold miners: ratio mean-reversion + wider PT
        baselines.append((
            "Alt F: Gold miner ratio (12%/5%)",
            dict(
                rules=MINING_LEAN_RATIO,
                min_confidence=0.50,
                profit_target=0.12,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "silver_miner":
        # Silver miners: tight max_loss (catastrophic DD history), volume confirmation
        baselines.append((
            "Alt F: Silver miner strict risk (10%/4%)",
            dict(
                rules=MINING_LEAN_VOLUME,
                min_confidence=0.55,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=4.0,
                cooldown_bars=7,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
        baselines.append((
            "Alt G: Silver miner wider PT (12%/4%)",
            dict(
                rules=MINING_LEAN_VOLUME,
                min_confidence=0.50,
                profit_target=0.12,
                stop_loss=1.0,
                max_loss_pct=4.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "uranium":
        # URA: CCJ-proven config
        baselines.append((
            "Alt F: CCJ-proven (10%/6%, conf=0.65, cd=7)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.65,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=6.0,
                cooldown_bars=7,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "rare_earth":
        # REMX: wider PT for spike trading, volume breakout
        baselines.append((
            "Alt F: Rare earth wide PT (15%/5%)",
            dict(
                rules=MINING_LEAN_VOLUME,
                min_confidence=0.50,
                profit_target=0.15,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
        baselines.append((
            "Alt G: MP-style lean (15%/6%, conf=0.50)",
            dict(
                rules=LEAN_RULES_3,
                min_confidence=0.50,
                profit_target=0.15,
                stop_loss=1.0,
                max_loss_pct=6.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
        ))
    elif sub_sector == "oilfield_services":
        # FET: energy momentum, wider stops for small-cap vol
        baselines.append((
            "Alt F: OFS momentum (12%/6%)",
            dict(
                rules=ENERGY_LEAN_MOMENTUM,
                min_confidence=0.50,
                profit_target=0.12,
                stop_loss=1.0,
                max_loss_pct=6.0,
                cooldown_bars=5,
                timeframe="daily",
                exit_timeframe="daily",
            ),
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


def run_full_validation(
    runner: BacktraderRunner,
    symbol: str,
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
            symbol=symbol, start_date=START_DATE, end_date=END_DATE, **run_kwargs
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
            symbol, START_DATE, END_DATE,
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
    failed_gates: List[str], base_kwargs: Dict, symbol: str = ""
) -> List[tuple]:
    """Generate targeted tuning configs based on which gates failed.

    Mining-specific: includes commodity_breakout, miner_metal_ratio,
    volume_breakout, and sub-sector-specific rules.
    """
    configs = []
    base_rules = base_kwargs.get("rules", [])
    recommended = STOCK_RULE_RECOMMENDATIONS.get(symbol, MINING_LEAN_MOMENTUM)
    sub_sector = STOCK_SUBSECTOR.get(symbol, "copper")

    if "Walk-Forward" in failed_gates:
        for pt in [0.08, 0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((
                    f"WF tune: PT={pt:.0%}",
                    {**base_kwargs, "profit_target": pt},
                ))
        for mc in [0.45, 0.55, 0.60, 0.65]:
            if mc != base_kwargs.get("min_confidence"):
                configs.append((
                    f"WF tune: conf={mc}",
                    {**base_kwargs, "min_confidence": mc},
                ))
        for cb in [3, 7]:
            if cb != base_kwargs.get("cooldown_bars"):
                configs.append((
                    f"WF tune: cooldown={cb}",
                    {**base_kwargs, "cooldown_bars": cb},
                ))
        # ATR stops (mining stocks need adaptive stops)
        configs.append((
            "WF tune: ATR stops x2.5",
            {**base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.5,
             "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0},
        ))
        # Mining-specific rules
        if "commodity_breakout" not in base_rules:
            configs.append((
                "WF tune: + commodity_breakout",
                {**base_kwargs, "rules": base_rules + ["commodity_breakout"]},
            ))
        if "miner_metal_ratio" not in base_rules:
            configs.append((
                "WF tune: + miner_metal_ratio",
                {**base_kwargs, "rules": base_rules + ["miner_metal_ratio"]},
            ))

    if "Bootstrap" in failed_gates:
        for mc in [0.40, 0.45, 0.55]:
            if mc != base_kwargs.get("min_confidence"):
                configs.append((
                    f"BS tune: conf={mc}",
                    {**base_kwargs, "min_confidence": mc},
                ))
        for cb in [3, 7]:
            if cb != base_kwargs.get("cooldown_bars"):
                configs.append((
                    f"BS tune: cooldown={cb}",
                    {**base_kwargs, "cooldown_bars": cb},
                ))
        # Full general rule set for more trade signals
        configs.append((
            "BS tune: full rules (10)",
            {**base_kwargs, "rules": GENERAL_RULES_10, "min_confidence": 0.50, "cooldown_bars": 3},
        ))
        # Full mining rules (14)
        configs.append((
            "BS tune: full mining rules (14)",
            {**base_kwargs, "rules": FULL_MINING_14, "min_confidence": 0.50, "cooldown_bars": 3},
        ))
        # Stock-specific recommended rules
        if recommended != base_rules:
            configs.append((
                f"BS tune: {sub_sector} rules",
                {**base_kwargs, "rules": recommended, "min_confidence": 0.50},
            ))
        # Volume breakout (more signals for thin-trade stocks)
        if "volume_breakout" not in base_rules:
            configs.append((
                "BS tune: + volume_breakout",
                {**base_kwargs, "rules": base_rules + ["volume_breakout"]},
            ))

    if "Monte Carlo" in failed_gates:
        for ml in [3.0, 4.0]:
            if ml != base_kwargs.get("max_loss_pct"):
                configs.append((
                    f"MC tune: max_loss={ml}%",
                    {**base_kwargs, "max_loss_pct": ml},
                ))
        # ATR stops for better risk management
        configs.append((
            "MC tune: ATR stops x2.0",
            {**base_kwargs, "stop_mode": "atr", "atr_multiplier": 2.0,
             "atr_stop_min_pct": 3.0, "atr_stop_max_pct": 15.0},
        ))

    if "Regime" in failed_gates:
        # Mining-specific rules for regime diversification
        if "commodity_breakout" not in base_rules:
            configs.append((
                "Regime tune: + commodity_breakout",
                {**base_kwargs, "rules": base_rules + ["commodity_breakout"]},
            ))
        if "miner_metal_ratio" not in base_rules:
            configs.append((
                "Regime tune: + miner_metal_ratio",
                {**base_kwargs, "rules": base_rules + ["miner_metal_ratio"]},
            ))
        if "volume_breakout" not in base_rules:
            configs.append((
                "Regime tune: + volume_breakout",
                {**base_kwargs, "rules": base_rules + ["volume_breakout"]},
            ))
        # Full mining rules
        configs.append((
            "Regime tune: full mining rules (14)",
            {**base_kwargs, "rules": FULL_MINING_14, "min_confidence": 0.50, "cooldown_bars": 3},
        ))
        # Full general rules
        configs.append((
            "Regime tune: full rules (10)",
            {**base_kwargs, "rules": GENERAL_RULES_10, "min_confidence": 0.50, "cooldown_bars": 3},
        ))
        # Higher confidence
        configs.append((
            "Regime tune: conf=0.65",
            {**base_kwargs, "min_confidence": 0.65},
        ))
        # Wider PT (let mining trends run)
        for pt in [0.12, 0.15]:
            if pt != base_kwargs.get("profit_target"):
                configs.append((
                    f"Regime tune: PT={pt:.0%}",
                    {**base_kwargs, "profit_target": pt},
                ))
        # Tighter stops
        if base_kwargs.get("max_loss_pct", 5.0) > 4.0:
            configs.append((
                "Regime tune: tighter stop 4%",
                {**base_kwargs, "max_loss_pct": 4.0},
            ))

    # FET-specific: try energy rules if mining rules don't work
    if symbol == "FET":
        energy_rules_14 = GENERAL_RULES_10 + [
            "energy_momentum", "energy_mean_reversion",
            "energy_seasonality", "midstream_yield_reversion",
        ]
        configs.append((
            "FET tune: energy rules (14)",
            {**base_kwargs, "rules": energy_rules_14, "min_confidence": 0.50, "cooldown_bars": 3},
        ))
        configs.append((
            "FET tune: energy momentum + wider PT (12%/6%)",
            {**base_kwargs, "rules": ENERGY_LEAN_MOMENTUM,
             "profit_target": 0.12, "max_loss_pct": 6.0},
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
            f.write(f"| {g.name} | {status} | {g.detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")

    f.write("---\n\n")


def write_markdown_report(
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
        f.write(f"**Timeframe:** Daily-only screening + multi-TF re-validation\n")
        f.write(f"**Validation Runtime:** {total_elapsed / 60:.1f} minutes\n")
        f.write(f"**Category:** {info['tier']}\n\n")
        f.write("---\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("Validate-then-tune approach with daily-only screening for speed, multi-TF re-validation for final config.\n\n")
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
        f.write(f"{symbol} — {info['description']}. {info['tier']}.\n\n")

        alt_baselines = get_alt_baselines(symbol)
        all_screens = [("Lean 3 rules baseline (10%/5%, conf=0.50)", baseline_report)] + [
            (name, r) for name, r in zip(
                [a[0] for a in alt_baselines], alt_reports
            )
        ]
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
        valid = [r for r in all_reports if r.backtest and r.backtest.total_trades >= 10]
        if valid:
            best_screen = max(valid, key=lambda r: r.backtest.sharpe_ratio or 0)
            f.write(f"\n**Best baseline selected for validation: {best_screen.label}**\n\n")
        else:
            f.write(f"\n**Best baseline selected for validation: {baseline_report.label}**\n\n")

        f.write("---\n\n")

        # Full validation of best
        f.write("## 2. Full Validation\n\n")
        if best_report:
            _write_config_section(f, best_report)
        else:
            _write_config_section(f, baseline_report)

        # Tuning results
        f.write("## 3. Tuning Results\n\n")
        if tune_reports:
            f.write("### Quick Screen\n\n")
            f.write("| Config | Trades | Win Rate | Return | Sharpe |\n")
            f.write("|--------|--------|----------|--------|--------|\n")
            for r in tune_reports:
                if r.backtest:
                    b = r.backtest
                    f.write(
                        f"| {r.label} | {b.total_trades} | {b.win_rate:.1%} | "
                        f"{b.total_return:+.1%} | {b.sharpe_ratio or 0:.2f} |\n"
                    )
            f.write("\n")

            f.write("### Full Validation of Top Candidates\n\n")
            validated_tunes = [r for r in tune_reports if r.gates]
            for r in validated_tunes:
                _write_config_section(f, r)
        else:
            f.write("No tuning needed — baseline validates.\n\n")
            f.write("---\n\n")

        # Summary table
        f.write("## 4. Summary Table\n\n")
        f.write("| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |\n")
        f.write("|--------|-----|-----|-----|--------|--------|--------|--------|\n")
        all_validated = []
        if best_report and best_report.gates:
            all_validated.append(best_report)
        validated_tunes = [r for r in tune_reports if r.gates]
        all_validated.extend(validated_tunes)

        all_validated.sort(
            key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
            reverse=True,
        )
        for i, r in enumerate(all_validated):
            bold = "**" if i == 0 else ""
            gates_map = {g.name: g for g in r.gates}
            wf = "**PASS**" if gates_map.get("Walk-Forward", GateResult("", False, "")).passed else "FAIL"
            bs = "**PASS**" if gates_map.get("Bootstrap", GateResult("", False, "")).passed else "FAIL"
            mc = "**PASS**" if gates_map.get("Monte Carlo", GateResult("", False, "")).passed else "FAIL"
            rg = "**PASS**" if gates_map.get("Regime", GateResult("", False, "")).passed else "FAIL"
            b = r.backtest
            if b:
                f.write(
                    f"| {bold}{r.label}{bold} | {wf} | {bs} | {mc} | {rg} | "
                    f"{bold}{b.sharpe_ratio or 0:.2f}{bold} | "
                    f"{bold}{b.total_return:+.1%}{bold} | {b.total_trades} |\n"
                )

        f.write("\n---\n\n")

        # Final recommendation
        f.write("## 5. Final Recommendation\n\n")
        final = all_validated[0] if all_validated else baseline_report
        status_word = "fully validates" if final.all_passed else "partially validates"
        f.write(
            f"**{symbol} {status_word}.** "
            f"Best config: {final.label} ({final.pass_count}/4 gates).\n\n"
        )
        _write_config_section(f, final)

        # Deployment recommendation
        f.write("### Deployment Recommendation\n\n")
        if final.pass_count >= 3:
            f.write("- Full deployment with no restrictions\n")
        elif final.pass_count >= 2:
            f.write("- Conditional deployment with regime restrictions and/or reduced sizing\n")
        else:
            f.write("- Consider blacklisting or significant restrictions\n")
        f.write("- Monitor the failing gate(s) in live trading\n")
        f.write("- Re-validate after 6 months of additional data\n\n")

    console.print(f"\n  Report written to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_mining2.py <SYMBOL>")
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
        f"{info['name']} — {info['description']}\n"
        f"Category: {info['tier']}",
        border_style="cyan",
    ))

    # ── Step 1: Screen baselines ──────────────────────────────────────────
    console.print("\n[bold]STEP 1: Screen Baselines[/bold]")

    console.print(f"\n  Running baseline: {symbol} (3 rules, 10%/5%, conf=0.50)...")
    t0 = time.time()
    bl_result = runner.run(
        symbol=symbol, start_date=START_DATE, end_date=END_DATE, **BASELINE
    )
    console.print(
        f"  Baseline: {bl_result.total_trades} trades, "
        f"WR={bl_result.win_rate:.1%}, Return={bl_result.total_return:+.1%}, "
        f"Sharpe={bl_result.sharpe_ratio or 0:.2f}, PF={bl_result.profit_factor or 0:.2f}, "
        f"DD=-{bl_result.max_drawdown_pct or 0:.1f}% ({time.time()-t0:.0f}s)"
    )
    baseline_report = ValidationReport(
        label="Lean 3 rules baseline (10%/5%, conf=0.50)",
        kwargs=BASELINE.copy(),
        backtest=bl_result,
    )

    alt_reports = []
    alt_baselines = get_alt_baselines(symbol)
    for alt_name, alt_kwargs in alt_baselines:
        console.print(f"\n  Running {alt_name}...")
        t0 = time.time()
        try:
            alt_result = runner.run(
                symbol=symbol, start_date=START_DATE, end_date=END_DATE, **alt_kwargs
            )
            console.print(
                f"  {alt_name}: {alt_result.total_trades} trades, "
                f"WR={alt_result.win_rate:.1%}, Return={alt_result.total_return:+.1%}, "
                f"Sharpe={alt_result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)"
            )
            alt_reports.append(
                ValidationReport(label=alt_name, kwargs=alt_kwargs.copy(), backtest=alt_result)
            )
        except Exception as e:
            console.print(f"  [red]{alt_name} failed: {e}[/red]")
            alt_reports.append(ValidationReport(label=alt_name, kwargs=alt_kwargs.copy()))

    # Pick best by Sharpe
    all_screens = [baseline_report] + alt_reports
    valid_screens = [
        r for r in all_screens if r.backtest and r.backtest.total_trades >= 10
    ]
    if valid_screens:
        best_screen = max(valid_screens, key=lambda r: r.backtest.sharpe_ratio or 0)
    else:
        valid_screens_5 = [
            r for r in all_screens if r.backtest and r.backtest.total_trades >= 5
        ]
        if valid_screens_5:
            best_screen = max(valid_screens_5, key=lambda r: r.backtest.sharpe_ratio or 0)
        else:
            best_screen = baseline_report

    console.print(
        f"\n  [bold green]Best by Sharpe: {best_screen.label} "
        f"(Sharpe={best_screen.backtest.sharpe_ratio or 0:.2f})[/bold green]"
    )

    # ── Step 2: Full validation of best (daily-only for speed) ───────────
    console.print("\n[bold]STEP 2: Full Validation of Best Baseline (daily-only)[/bold]")
    best_report = run_full_validation(
        runner, symbol, best_screen.label, best_screen.kwargs, print_reports=True
    )

    # ── Step 3: Diagnose and tune ─────────────────────────────────────────
    console.print("\n[bold]STEP 3: Diagnose & Targeted Tune[/bold]")

    tune_reports = []
    if best_report.failed_gates:
        console.print(f"  Failed gates: {', '.join(best_report.failed_gates)}")
        console.print("  Generating targeted tuning configs...")

        tune_configs = generate_tune_configs(
            best_report.failed_gates, best_screen.kwargs, symbol=symbol
        )
        console.print(f"  Generated {len(tune_configs)} tuning configs")

        # Quick screen: run backtests only
        screen_results = []
        for label, kwargs in tune_configs:
            console.print(f"\n  Screening: {label}...")
            t0 = time.time()
            try:
                result = runner.run(
                    symbol=symbol, start_date=START_DATE, end_date=END_DATE, **kwargs
                )
                console.print(
                    f"  {result.total_trades} trades, WR={result.win_rate:.1%}, "
                    f"Return={result.total_return:+.1%}, "
                    f"Sharpe={result.sharpe_ratio or 0:.2f} ({time.time()-t0:.0f}s)"
                )
                screen_results.append((label, kwargs, result))
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/red]")

        # Rank by Sharpe, take top 3 for full validation
        screen_results.sort(
            key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
            reverse=True,
        )
        top_n = screen_results[:3]

        console.print(f"\n  [bold]Top {len(top_n)} candidates for full validation:[/bold]")
        for label, kwargs, result in top_n:
            console.print(
                f"    {label}: Sharpe={result.sharpe_ratio or 0:.2f}, "
                f"Return={result.total_return:+.1%}, WR={result.win_rate:.1%}"
            )

        # Full validation of top candidates
        for label, kwargs, _ in top_n:
            report = run_full_validation(runner, symbol, label, kwargs, print_reports=True)
            tune_reports.append(report)

        # Also add quick-screen-only results for the report
        for label, kwargs, result in screen_results:
            if not any(r.label == label for r in tune_reports):
                tune_reports.append(
                    ValidationReport(label=label, kwargs=kwargs, backtest=result)
                )
    else:
        console.print("  [green]All gates passed! No tuning needed.[/green]")

    # ── Step 4: Multi-TF re-validation of winner ─────────────────────────
    all_candidates = [best_report] + [r for r in tune_reports if r.gates]
    all_candidates.sort(
        key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
        reverse=True,
    )
    winner = all_candidates[0] if all_candidates else best_report

    if winner.pass_count >= 2:
        console.print("\n[bold]STEP 4: Multi-TF Re-validation (daily entries + 5min exits)[/bold]")
        console.print(f"  Re-validating winner: {winner.label}")
        mtf_kwargs = make_multitf(winner.kwargs)
        mtf_report = run_full_validation(
            runner, symbol, f"{winner.label} [multi-TF]", mtf_kwargs, print_reports=True
        )
        if mtf_report.backtest and mtf_report.backtest.total_trades >= 2:
            tune_reports.append(mtf_report)
            console.print(
                f"\n  Multi-TF result: {mtf_report.pass_count}/4 gates, "
                f"Sharpe={mtf_report.backtest.sharpe_ratio or 0:.2f}"
            )
        else:
            console.print("  [yellow]Multi-TF produced too few trades — using daily-only results[/yellow]")
    else:
        console.print("\n  [yellow]Winner has <2 gates — skipping multi-TF re-validation[/yellow]")

    # ── Step 5: Write report ──────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    console.print(f"\n[bold]STEP 5: Write Report[/bold]")
    write_markdown_report(
        symbol, info, baseline_report, alt_reports, tune_reports, best_report, total_elapsed
    )

    # ── Final summary ─────────────────────────────────────────────────────
    console.print(f"\n{'=' * 70}")
    console.print(
        f"[bold]Total runtime: {total_elapsed / 60:.1f} minutes[/bold]"
    )

    all_final = [best_report] + [r for r in tune_reports if r.gates]
    if all_final:
        all_final.sort(
            key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
            reverse=True,
        )
        winner = all_final[0]
        console.print(
            f"\n[bold green]BEST: {winner.label}[/bold green]"
            f"\n  Gates: {winner.pass_count}/4"
            f"\n  Sharpe: {winner.backtest.sharpe_ratio or 0:.2f}"
            f"\n  Return: {winner.backtest.total_return:+.1%}"
            f"\n  Trades: {winner.backtest.total_trades}"
            f"\n  WR: {winner.backtest.win_rate:.1%}"
        )


if __name__ == "__main__":
    main()
