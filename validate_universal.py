#!/usr/bin/env python3
"""
Universal Stock Validation

Validates any stock through the full 4-gate pipeline:
  1. Individual rule screen (sector-specific + generic)
  2. Combo screen (sector-specific combos)
  3. Parameter sweep on top combos
  4. Full 4-gate validation (Walk-Backward, Bootstrap, Monte Carlo, Regime)

Reads active tickers from rules.yaml. Maps each to its sector automatically.
Generates per-symbol reports and master comparison.

Usage:
    python validate_universal.py                        # All active tickers
    python validate_universal.py AEM OKE PSX            # Specific symbols
    python validate_universal.py --sector energy        # All energy stocks
    python validate_universal.py --sector mining        # All mining stocks
    python validate_universal.py --resume               # Resume interrupted run
    python validate_universal.py --list                 # List known symbols + sectors
"""
import argparse
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

# Path setup — same as validate_energy_wb.py
sys.path.insert(0, str(Path(__file__).parent))
de_path = os.path.expanduser("~/Projects/decision-engine")
if de_path not in sys.path:
    sys.path.insert(0, de_path)
os.environ.setdefault("DECISION_ENGINE_PATH", de_path)

from backtesting.validation.universal.config import (
    SECTOR_REGISTRY,
    SYMBOL_SECTORS,
    get_sector_config,
    get_symbol_sector,
)

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Paths
RULES_YAML = Path(__file__).parent.parent / "decision-engine" / "config" / "rules.yaml"
OUTPUT_DIR = Path(__file__).parent
# Regime windows per sector — must match the original validation scripts exactly.
# The key difference: 2021 was a gold/mining bear but an energy/industrial bull.
_COMMON_2022_BEAR = {
    "label": "2022 Bear (Rate Hike)",
    "start": date(2022, 1, 3),
    "end": date(2022, 10, 14),
    "expected_regime": "bear",
}
_COMMON_CHOP = {
    "label": "2023-2024 Choppy Transition",
    "start": date(2023, 1, 3),
    "end": date(2024, 6, 30),
    "expected_regime": "chop",
}

SECTOR_REGIME_WINDOWS = {
    # Gold miners: 2022 bear + choppy transition
    # Note: 2021 Gold Bear and early 2022 Bear windows removed — insufficient
    # data depth after indicator warmup for most mining symbols (PPLT data
    # starts Dec 2021, AEM similar). Keeping 2022-H2 bear and choppy windows.
    "mining": [
        {
            "label": "2022-H2 Bear (Rate Hike Peak)",
            "start": date(2022, 6, 1),
            "end": date(2022, 10, 14),
            "expected_regime": "bear",
        },
        _COMMON_CHOP,
    ],
    # Energy: 2021 was a strong recovery (oil rally post-COVID)
    "energy": [
        _COMMON_2022_BEAR,
        {
            "label": "2021 Recovery (Post-COVID Oil Rally)",
            "start": date(2021, 1, 4),
            "end": date(2021, 12, 31),
            "expected_regime": "bull",
        },
        _COMMON_CHOP,
    ],
    # Defense: 2021 was a lagging bull (defense underperformed, ITA +9.4% vs SPY +19.7%)
    # 2022 was counter-cyclical bull (ITA +10% while SPY -14.7%)
    "defense": [
        {
            "label": "2022 Counter-Cyclical (Defense outperformed bear by +24.6%)",
            "start": date(2022, 1, 3),
            "end": date(2022, 10, 14),
            "expected_regime": "bull",
        },
        {
            "label": "2021 Lagging Bull (Defense +9.4% vs SPY +19.7%)",
            "start": date(2021, 1, 4),
            "end": date(2021, 12, 31),
            "expected_regime": "bull",
        },
        _COMMON_CHOP,
    ],
    # Default for all other sectors: 2021 was a bull market
    "_default": [
        _COMMON_2022_BEAR,
        {
            "label": "2021 Bull (Low Rate Momentum)",
            "start": date(2021, 1, 4),
            "end": date(2021, 12, 31),
            "expected_regime": "bull",
        },
        _COMMON_CHOP,
    ],
}


def load_active_tickers() -> dict:
    """Load active tickers from rules.yaml."""
    if not RULES_YAML.exists():
        console.print(f"[red]rules.yaml not found at {RULES_YAML}[/red]")
        return {}
    with open(RULES_YAML) as f:
        config = yaml.safe_load(f)
    return config.get("active_tickers", {})


def get_symbols_for_args(args) -> list:
    """Resolve CLI args to list of symbols."""
    if args.list:
        return []

    if args.symbols:
        return [s.upper() for s in args.symbols]

    if args.sector:
        sector = args.sector.lower()
        return [sym for sym, info in SYMBOL_SECTORS.items() if info["sector"] == sector]

    # Default: all active tickers from rules.yaml
    active = load_active_tickers()
    return list(active.keys())


def show_symbol_list():
    """Display all known symbols with sectors."""
    table = Table(title="Known Symbols")
    table.add_column("Symbol", style="cyan")
    table.add_column("Name")
    table.add_column("Sector", style="green")
    table.add_column("Sub-sector")
    table.add_column("Tier")

    active = load_active_tickers()

    for sym, info in sorted(SYMBOL_SECTORS.items()):
        deployed = " [bold green]*[/bold green]" if sym in active else ""
        table.add_row(
            sym + deployed,
            info["name"],
            info["sector"],
            info["sub_sector"],
            info["tier"],
        )

    console.print(table)
    console.print(f"\n[green]*[/green] = currently deployed in rules.yaml ({len(active)} tickers)")
    console.print(f"\nSectors: {', '.join(sorted(SECTOR_REGISTRY.keys()))}")
    console.print("\nUsage:")
    console.print("  python validate_universal.py AEM OKE      # specific symbols")
    console.print("  python validate_universal.py --sector energy  # all energy stocks")
    console.print("  python validate_universal.py              # all active tickers")


def main():
    parser = argparse.ArgumentParser(description="Universal Stock Validation")
    parser.add_argument("symbols", nargs="*", help="Symbols to validate")
    parser.add_argument("--sector", help="Validate all stocks in a sector")
    parser.add_argument("--list", action="store_true", help="List known symbols")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--full", action="store_true", help="Force full pipeline (ignore saved profiles)")
    parser.add_argument("--retune", action="store_true", help="Re-sweep params on known-good combos (skip discovery)")
    parser.add_argument("--revalidate", action="store_true", help="Re-run 4-gate validation on exact winning params")
    parser.add_argument("--list-profiles", action="store_true", help="List symbols with saved profiles")
    parser.add_argument("--start-date", default="2021-01-01", help="Backtest start date")
    parser.add_argument("--cash", type=float, default=1000, help="Initial cash")
    parser.add_argument("--exit-tf", default="daily", help="Exit timeframe (daily or 5min for multi-TF)")
    parser.add_argument("--top-combos", type=int, default=3, help="Top combos to sweep")
    parser.add_argument("--top-validate", type=int, default=5, help="Top configs to validate")
    parser.add_argument("--workers", type=int, default=6, help="Parallel symbol workers (0=serial)")
    args = parser.parse_args()

    if args.list:
        show_symbol_list()
        return

    if args.list_profiles:
        from backtesting.validation.universal.profiles import ProfileManager
        profiles = ProfileManager()
        rows = profiles.list_profiles()
        if not rows:
            console.print("[yellow]No saved profiles.[/yellow]")
            return
        table = Table(title="Saved Profiles")
        table.add_column("Symbol", style="cyan")
        table.add_column("Sector")
        table.add_column("Gates", justify="center")
        table.add_column("Sharpe", justify="right")
        table.add_column("Action", style="bold")
        table.add_column("V", justify="right", header_style="green")
        table.add_column("JV", justify="right", header_style="yellow")
        table.add_column("Updated")
        for r in rows:
            action_color = {"deploy": "green", "conditional": "yellow", "skip": "red"}.get(r["recommendation"] or "", "white")
            table.add_row(
                r["symbol"], r["sector"],
                f"{r['best_gates']}/{r.get('total_gates', r['best_gates'])}",
                f"{r['best_sharpe'] or 0:.2f}",
                f"[{action_color}]{r['recommendation'] or '-'}[/{action_color}]",
                str(r["varsity_count"]),
                str(r["jv_count"]),
                r["last_updated"].strftime("%Y-%m-%d") if r["last_updated"] else "-",
            )
        console.print(table)
        profiles.close()
        return

    symbols = get_symbols_for_args(args)
    if not symbols:
        console.print("[red]No symbols to validate. Use --list to see available symbols.[/red]")
        return

    # Heavy imports — only loaded when actually running validation
    from backtesting.engine import BacktraderRunner
    from backtesting.validation.universal.checkpoint import CheckpointManager
    from backtesting.validation.universal.data_cache import DataCache
    from backtesting.validation.universal.parallel import parallel_validate_symbols
    from backtesting.validation.universal.pipeline import UniversalValidator
    from backtesting.validation.universal.profiles import ProfileManager
    from backtesting.validation.universal.report import write_master_report, write_symbol_report

    # Checkpoint support
    checkpoint = CheckpointManager()
    run_id = checkpoint.get_run_id()

    if args.resume:
        completed = checkpoint.get_completed_symbols(run_id)
        if completed:
            console.print(f"[yellow]Resuming run {run_id}. Already completed: {', '.join(completed)}[/yellow]")
            symbols = [s for s in symbols if s not in completed]
            if not symbols:
                console.print("[green]All symbols already completed![/green]")
                return

    start_date = date.fromisoformat(args.start_date)
    end_date = date.today()

    exit_tf = args.exit_tf
    multi_tf = exit_tf != "daily"
    num_workers = args.workers

    console.print(f"\n[bold]Universal Validation[/bold]")
    console.print(f"  Symbols: {', '.join(symbols)} ({len(symbols)} total)")
    console.print(f"  Period: {start_date} to {end_date}")
    console.print(f"  Cash: ${args.cash:,.0f}")
    console.print(f"  Exit TF: {exit_tf}{' (multi-timeframe)' if multi_tf else ''}")
    console.print(f"  Workers: {num_workers if num_workers > 0 else 'serial'}")
    console.print(f"  Run ID: {run_id}")

    # Show sector breakdown
    sector_counts = {}
    for sym in symbols:
        info = get_symbol_sector(sym)
        sector_counts[info["sector"]] = sector_counts.get(info["sector"], 0) + 1
    console.print(f"  Sectors: {', '.join(f'{s}({c})' for s, c in sorted(sector_counts.items()))}")

    profiles = ProfileManager()

    # Build work items — resolve mode for each symbol before forking
    work_items = []
    for symbol in symbols:
        sym_info = get_symbol_sector(symbol)
        sym_sector = sym_info["sector"]
        regime_windows = SECTOR_REGIME_WINDOWS.get(sym_sector, SECTOR_REGIME_WINDOWS["_default"])

        known_combos = None
        winning_configs = None

        if args.full:
            mode = "full"
        elif args.revalidate:
            winning_configs = profiles.get_winning_configs(symbol)
            mode = "revalidate" if winning_configs else "full"
        elif args.retune:
            known_combos = profiles.get_known_combos(symbol)
            mode = "retune" if known_combos else "full"
        else:
            if profiles.has_profile(symbol) and not profiles.is_stale(symbol, sym_sector):
                known_combos = profiles.get_known_combos(symbol)
                mode = "retune" if known_combos else "full"
            else:
                mode = "full"

        work_items.append({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": args.cash,
            "exit_timeframe": exit_tf,
            "top_combos": args.top_combos,
            "top_validate": args.top_validate,
            "regime_windows": regime_windows,
            "mode": mode,
            "known_combos": known_combos,
            "winning_configs": winning_configs,
        })

    mode_counts = {}
    for w in work_items:
        mode_counts[w["mode"]] = mode_counts.get(w["mode"], 0) + 1
    console.print(f"  Modes: {', '.join(f'{m}({c})' for m, c in sorted(mode_counts.items()))}")

    all_results = []
    total_start = time.time()

    if num_workers > 0 and len(symbols) > 1:
        # ── Parallel: run symbols across cores ──
        console.print(f"\n[bold]Running {len(symbols)} symbols across {num_workers} workers...[/bold]")

        parallel_results = parallel_validate_symbols(
            work_items, max_workers=num_workers,
        )

        for symbol, result in parallel_results:
            if result is None:
                console.print(f"[red]{symbol} FAILED[/red]")
                continue

            all_results.append(result)
            winner = result.validated_reports[0] if result.validated_reports else None
            if winner and winner.backtest:
                console.print(
                    f"  [cyan]{symbol}[/cyan]: {winner.pass_count}/4 gates, "
                    f"Sharpe={winner.backtest.sharpe_ratio or 0:.2f}, "
                    f"Return={winner.backtest.total_return:+.1%}, "
                    f"{result.elapsed / 60:.0f}m → [bold]{result.recommendation}[/bold]"
                )
            else:
                console.print(f"  [cyan]{symbol}[/cyan]: 0/4 gates, {result.elapsed / 60:.0f}m → skip")

            # Save profile + checkpoint (runs in main process after worker returns)
            try:
                mode = next(w["mode"] for w in work_items if w["symbol"] == symbol)
                profiles.save_profile(result, run_mode=mode)
            except Exception:
                pass

            try:
                summary = {
                    "symbol": symbol,
                    "recommendation": result.recommendation,
                    "elapsed": result.elapsed,
                    "best_gates": winner.pass_count if winner else 0,
                    "total_gates": len(winner.gates) if winner else 0,
                    "best_label": winner.label if winner else "none",
                    "best_sharpe": (
                        winner.backtest.sharpe_ratio
                        if winner and winner.backtest else 0
                    ),
                }
                checkpoint.save_symbol_complete(run_id, symbol, summary)
            except Exception:
                pass
    else:
        # ── Serial: one symbol at a time (vectorized discovery + backtrader validation) ──
        runner = BacktraderRunner(initial_cash=args.cash)
        data_cache = DataCache()

        for i, item in enumerate(work_items, 1):
            symbol = item["symbol"]
            console.print(f"\n{'=' * 70}")
            console.print(f"Symbol {i}/{len(symbols)}: {symbol}")
            console.print(f"{'=' * 70}")

            try:
                validator = UniversalValidator(
                    runner=runner,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=args.cash,
                    exit_timeframe=exit_tf,
                    top_combos=args.top_combos,
                    top_validate=args.top_validate,
                    regime_windows=item["regime_windows"],
                    data_cache=data_cache,
                )

                mode = item["mode"]
                if mode == "full":
                    console.print(f"  Mode: [bold]FULL[/bold] (complete pipeline)")
                    result = validator.validate_symbol(symbol)
                elif mode == "retune":
                    console.print(f"  Mode: [yellow]RETUNE[/yellow] (re-sweep {len(item['known_combos'])} combos)")
                    result = validator.validate_symbol_retune(symbol, item["known_combos"])
                elif mode == "revalidate":
                    console.print(f"  Mode: [green]REVALIDATE[/green] (4-gate only on {len(item['winning_configs'])} configs)")
                    result = validator.validate_symbol_revalidate(symbol, item["winning_configs"])

                all_results.append(result)
                runner.clear_indicator_cache()
                data_cache.clear(symbol)  # free memory before next symbol
                profiles.save_profile(result, run_mode=mode)

                summary = {
                    "symbol": symbol,
                    "recommendation": result.recommendation,
                    "elapsed": result.elapsed,
                    "best_gates": result.validated_reports[0].pass_count if result.validated_reports else 0,
                    "best_label": result.validated_reports[0].label if result.validated_reports else "none",
                    "best_sharpe": (
                        result.validated_reports[0].backtest.sharpe_ratio
                        if result.validated_reports and result.validated_reports[0].backtest
                        else 0
                    ),
                }
                checkpoint.save_symbol_complete(run_id, symbol, summary)

            except Exception as e:
                console.print(f"[red]{symbol} FAILED: {e}[/red]")
                import traceback
                traceback.print_exc()

    profiles.close()

    total_elapsed = time.time() - total_start

    if not all_results:
        console.print("[red]No results to report.[/red]")
        return

    # Write reports
    console.print(f"\n{'=' * 70}")
    console.print("[bold]Writing Reports[/bold]")

    output_dir = str(OUTPUT_DIR)
    for sr in all_results:
        path = write_symbol_report(sr, output_dir)
        console.print(f"  {path}")

    master_path = write_master_report(all_results, total_elapsed, output_dir)
    console.print(f"  [bold]{master_path}[/bold]")

    # Summary table
    console.print(f"\n{'=' * 70}")
    table = Table(title="Validation Summary")
    table.add_column("Symbol", style="cyan")
    table.add_column("Sector")
    table.add_column("Gates", justify="center")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Action", style="bold")

    for sr in sorted(all_results, key=lambda x: (
        x.validated_reports[0].pass_count if x.validated_reports else 0,
        x.validated_reports[0].backtest.sharpe_ratio or 0 if x.validated_reports and x.validated_reports[0].backtest else 0,
    ), reverse=True):
        winner = sr.validated_reports[0] if sr.validated_reports else None
        if winner and winner.backtest:
            action_color = {"deploy": "green", "conditional": "yellow", "skip": "red"}.get(sr.recommendation, "white")
            table.add_row(
                sr.symbol, sr.sector,
                f"{winner.pass_count}/{len(winner.gates)}",
                f"{winner.backtest.sharpe_ratio or 0:.2f}",
                f"{winner.backtest.total_return:+.1%}",
                f"{sr.elapsed / 60:.0f}m",
                f"[{action_color}]{sr.recommendation}[/{action_color}]",
            )
        else:
            table.add_row(sr.symbol, sr.sector, "0/?", "-", "-", f"{sr.elapsed / 60:.0f}m", "[red]skip[/red]")

    console.print(table)
    console.print(f"\n[bold]Total runtime: {total_elapsed / 60:.1f} minutes[/bold]")


if __name__ == "__main__":
    main()
