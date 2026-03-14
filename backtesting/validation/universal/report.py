"""
Report generation for universal validation.

Writes per-symbol reports and master comparison matrix.
"""
import os
from datetime import date
from typing import Dict, List, Optional

from ..walk_backward import WalkBackwardResult
from .config import get_symbol_sector
from .models import GateResult, SymbolResult, ValidationReport


def _write_config_section(f, report: ValidationReport, min_regimes: int = 2):
    """Write one validated config section."""
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
            detail = g.detail[:117] + "..." if len(g.detail) > 120 else g.detail
            f.write(f"| {g.name} | {status} | {detail} |\n")
        f.write(f"\n**Result: {report.pass_count}/4 gates passed**\n\n")

        # Walk-backward detail
        for g in report.gates:
            if g.name == "Walk-Backward" and g.data and isinstance(g.data, WalkBackwardResult):
                wb = g.data
                f.write("#### Walk-Backward Detail\n\n")
                f.write(f"- **Tune:** {wb.tune_start} to {wb.tune_end} "
                        f"({wb.tune_result.total_return:+.1%}, {wb.tune_result.total_trades}t)\n")
                f.write(f"- **Holdout:** {wb.holdout_start} to {wb.holdout_end} "
                        f"({wb.holdout_result.total_return:+.1%}, {wb.holdout_result.total_trades}t) "
                        f"{'**PASS**' if wb.holdout_passed else 'FAIL'}\n")
                f.write(f"- **Regimes:** {wb.regimes_passed}/{wb.regimes_total} (need {min_regimes})\n\n")

                if wb.regime_windows:
                    f.write("| Regime | Period | Return | Trades | Status |\n")
                    f.write("|--------|--------|--------|--------|--------|\n")
                    for w in wb.regime_windows:
                        s = "**PASS**" if w.passed else "FAIL"
                        f.write(f"| {w.label} | {w.start}->{w.end} | {w.total_return:+.1%} | "
                                f"{w.total_trades} | {s} |\n")
                    f.write("\n")
    f.write("---\n\n")


def write_symbol_report(sr: SymbolResult, output_dir: str):
    """Write per-symbol validation report."""
    info = get_symbol_sector(sr.symbol)
    filepath = os.path.join(output_dir, f"{sr.symbol.lower()}-validated.md")

    with open(filepath, "w") as f:
        f.write(f"# {sr.symbol} ({info['name']}) — Universal Validation\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Sector:** {info['sector']} / {info['sub_sector']}\n")
        f.write(f"**Tier:** {info['tier']}\n")
        f.write(f"**Runtime:** {sr.elapsed / 60:.1f} minutes\n")
        f.write(f"**Recommendation:** {sr.recommendation.upper()}\n\n---\n\n")

        # Individual rules
        f.write("## 1. Individual Rule Screen\n\n")
        f.write("| Rule | Trades | WR | Return | Sharpe | PF |\n")
        f.write("|------|--------|-----|--------|--------|-----|\n")
        for name, _, r in sr.individual_results:
            f.write(
                f"| {name} | {r.total_trades} | {r.win_rate:.0%} | "
                f"{r.total_return:+.1%} | {r.sharpe_ratio or 0:.2f} | "
                f"{r.profit_factor or 0:.2f} |\n"
            )
        f.write("\n---\n\n")

        # Combos
        f.write("## 2. Combo Screen\n\n")
        f.write("| Combo | Trades | WR | Return | Sharpe | PF |\n")
        f.write("|-------|--------|-----|--------|--------|-----|\n")
        for name, _, r in sr.combo_results:
            f.write(
                f"| {name} | {r.total_trades} | {r.win_rate:.0%} | "
                f"{r.total_return:+.1%} | {r.sharpe_ratio or 0:.2f} | "
                f"{r.profit_factor or 0:.2f} |\n"
            )
        f.write("\n---\n\n")

        # Sweep top 10
        f.write("## 3. Parameter Sweep -- Top 10\n\n")
        f.write("| Rank | Config | Trades | WR | Return | Sharpe | PF |\n")
        f.write("|------|--------|--------|-----|--------|--------|-----|\n")
        for i, (label, _, r) in enumerate(sr.sweep_results[:10], 1):
            if r.total_trades >= 5:
                f.write(
                    f"| {i} | {label} | {r.total_trades} | {r.win_rate:.0%} | "
                    f"{r.total_return:+.1%} | {r.sharpe_ratio or 0:.2f} | "
                    f"{r.profit_factor or 0:.2f} |\n"
                )
        f.write("\n---\n\n")

        # Validation results
        f.write("## 4. Full Validation\n\n")
        for vr in sr.validated_reports:
            _write_config_section(f, vr)

    return filepath


def write_master_report(
    all_results: List[SymbolResult],
    total_elapsed: float,
    output_dir: str,
    deployed_configs: Optional[Dict] = None,
):
    """Write master comparison report across all symbols."""
    filepath = os.path.join(output_dir, f"universal-validation-{date.today()}.md")

    with open(filepath, "w") as f:
        f.write("# Universal Validation Report\n\n")
        f.write(f"**Date:** {date.today()}\n")
        f.write(f"**Symbols:** {len(all_results)}\n")
        f.write(f"**Runtime:** {total_elapsed / 60:.1f} minutes\n\n---\n\n")

        # Master comparison table
        f.write("## Results Matrix\n\n")
        f.write("| Symbol | Sector | Best Config | Gates | WB | BS | MC | Reg | Sharpe | Return | Trades | WR | Action |\n")
        f.write("|--------|--------|-------------|-------|----|----|----|-----|--------|--------|--------|----|--------|\n")

        for sr in all_results:
            info = get_symbol_sector(sr.symbol)
            winner = sr.validated_reports[0] if sr.validated_reports else None
            if winner and winner.backtest:
                gates_map = {g.name: g for g in winner.gates}
                wb = "**PASS**" if gates_map.get("Walk-Backward", GateResult("", False, "")).passed else "FAIL"
                bs = "**PASS**" if gates_map.get("Bootstrap", GateResult("", False, "")).passed else "FAIL"
                mc = "**PASS**" if gates_map.get("Monte Carlo", GateResult("", False, "")).passed else "FAIL"
                rg = "**PASS**" if gates_map.get("Regime", GateResult("", False, "")).passed else "FAIL"
                b = winner.backtest
                f.write(
                    f"| **{sr.symbol}** | {info['sector']} | {winner.label} | "
                    f"{winner.pass_count}/4 | {wb} | {bs} | {mc} | {rg} | "
                    f"{b.sharpe_ratio or 0:.2f} | {b.total_return:+.1%} | "
                    f"{b.total_trades} | {b.win_rate:.0%} | {sr.recommendation} |\n"
                )
            else:
                f.write(
                    f"| **{sr.symbol}** | {info['sector']} | No valid config | 0/4 | "
                    f"FAIL | FAIL | FAIL | FAIL | - | - | 0 | - | skip |\n"
                )

        f.write("\n---\n\n")

        # Comparison vs deployed (if available)
        if deployed_configs:
            f.write("## vs Currently Deployed\n\n")
            f.write("| Symbol | Deployed Gates | Deployed Sharpe | New Gates | New Sharpe | Delta | Action |\n")
            f.write("|--------|---------------|-----------------|-----------|------------|-------|--------|\n")

            for sr in all_results:
                winner = sr.validated_reports[0] if sr.validated_reports else None
                deployed = deployed_configs.get(sr.symbol)

                new_gates = winner.pass_count if winner else 0
                new_sharpe = winner.backtest.sharpe_ratio or 0 if winner and winner.backtest else 0

                if deployed:
                    d_gates = deployed.get("gates", "?")
                    d_sharpe = deployed.get("sharpe", "?")
                    delta = "improved" if new_gates > (d_gates if isinstance(d_gates, int) else 0) else "same/worse"
                    f.write(f"| {sr.symbol} | {d_gates} | {d_sharpe} | {new_gates}/4 | {new_sharpe:.2f} | {delta} | {sr.recommendation} |\n")
                else:
                    f.write(f"| {sr.symbol} | NEW | - | {new_gates}/4 | {new_sharpe:.2f} | new | {sr.recommendation} |\n")

            f.write("\n---\n\n")

        # Recommendations summary
        deploy = [sr for sr in all_results if sr.recommendation == "deploy"]
        conditional = [sr for sr in all_results if sr.recommendation == "conditional"]
        skip = [sr for sr in all_results if sr.recommendation in ("skip", "")]

        if deploy:
            f.write("## Deploy (3-4/4 gates)\n\n")
            for sr in deploy:
                w = sr.validated_reports[0]
                f.write(f"- **{sr.symbol}**: {w.pass_count}/4 gates, {w.label}\n")
            f.write("\n")

        if conditional:
            f.write("## Conditional (2/4 gates)\n\n")
            for sr in conditional:
                w = sr.validated_reports[0]
                f.write(f"- **{sr.symbol}**: {w.pass_count}/4 gates, {w.label}\n")
            f.write("\n")

        if skip:
            f.write("## Skip (<2/4 gates)\n\n")
            for sr in skip:
                if sr.validated_reports:
                    w = sr.validated_reports[0]
                    f.write(f"- **{sr.symbol}**: {w.pass_count}/4 gates\n")
                else:
                    f.write(f"- **{sr.symbol}**: No valid configs found\n")
            f.write("\n")

    return filepath
