"""
Universal validation pipeline.

Runs the full 4-step validation for any stock:
  1. Individual rule screen
  2. Combo screen
  3. Parameter sweep on top combos
  4. Full 4-gate validation (Walk-Backward, Bootstrap, Monte Carlo, Regime)
"""
import logging
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...engine import BacktraderRunner
from ..bootstrap import bootstrap_analysis
from ..monte_carlo import monte_carlo_analysis
from ..regime import analyze_by_regime
from ..walk_backward import WalkBackwardValidator, WalkBackwardResult

from .data_cache import DataCache
from .vectorized import VectorizedEngine

from .config import (
    COARSE_CONFIDENCE,
    COARSE_COOLDOWN,
    COARSE_MAX_LOSS,
    COARSE_PROFIT_TARGET,
    CONFIDENCE_VALUES,
    COOLDOWN_VALUES,
    MAX_LOSS_VALUES,
    MODIFIER_RULES,
    PROFIT_TARGET_VALUES,
    SectorConfig,
    get_sector_config,
    get_symbol_sector,
)
from .parallel import parallel_sweep
from .models import GateResult, SymbolResult, ValidationReport

logger = logging.getLogger(__name__)
console = Console()

# Walk-backward defaults
DEFAULT_TUNE_MONTHS = 6
DEFAULT_HOLDOUT_MONTHS = 2
DEFAULT_MIN_REGIMES_PASS = 2


class UniversalValidator:
    """Runs the full validation pipeline for any stock."""

    def __init__(
        self,
        runner: BacktraderRunner,
        start_date: date = date(2021, 1, 1),
        end_date: Optional[date] = None,
        initial_cash: float = 1000,
        exit_timeframe: str = "daily",
        top_combos: int = 3,
        top_validate: int = 5,
        regime_windows: Optional[List[Dict]] = None,
        serial_sweep: bool = False,
        data_cache: Optional[DataCache] = None,
    ):
        self.runner = runner
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.initial_cash = initial_cash
        self.top_combos = top_combos
        self.top_validate = top_validate
        self.regime_windows = regime_windows  # None = use defaults from walk_backward.py
        self.serial_sweep = serial_sweep  # True when running inside a symbol-level worker
        self.data_cache = data_cache
        self._vec_engine: Optional[VectorizedEngine] = None
        self._vec_symbol: Optional[str] = None
        # Common kwargs passed to runner.run() — match existing validate scripts
        self.common_kwargs = dict(
            timeframe="daily",
            exit_timeframe=exit_timeframe,
        )

    def _ensure_vec_engine(self, symbol: str):
        """Create VectorizedEngine for a symbol from cached data.

        Preloads data into DataCache if needed. Falls back to backtrader
        if data loading fails.
        """
        if self.data_cache is None:
            return
        if self._vec_engine is not None and self._vec_symbol == symbol:
            return  # already created for this symbol

        exit_tf = self.common_kwargs.get("exit_timeframe", "daily")
        try:
            self.data_cache.preload(
                symbol, self.start_date, self.end_date,
                exit_timeframe=exit_tf,
            )
            self._vec_engine = VectorizedEngine(
                df_daily=self.data_cache.get_daily(symbol),
                df_intraday=self.data_cache.get_intraday(symbol),
                initial_cash=self.initial_cash,
            )
            self._vec_symbol = symbol
            console.print(f"  [green]Using vectorized engine[/green]")
        except Exception as e:
            logger.warning(f"Vectorized engine unavailable for {symbol}: {e}")
            console.print(f"  [yellow]Vectorized engine unavailable: {e}, using backtrader[/yellow]")
            self._vec_engine = None
            self._vec_symbol = None

    def _run_backtest(self, symbol: str, kwargs: dict):
        """Run single config — vectorized if available, else backtrader."""
        if self._vec_engine and self._vec_symbol == symbol:
            return self._vec_engine.run(symbol=symbol, **kwargs)
        return self.runner.run(
            symbol=symbol, start_date=self.start_date,
            end_date=self.end_date, **kwargs,
        )

    def validate_symbol(self, symbol: str) -> SymbolResult:
        """Full pipeline for one symbol."""
        info = get_symbol_sector(symbol)
        sector_config = get_sector_config(info["sector"])
        sym_start = time.time()
        self._ensure_vec_engine(symbol)

        console.print(f"\n{'=' * 70}")
        console.print(Panel.fit(
            f"[bold cyan]{symbol} — {info['name']}[/bold cyan]\n"
            f"Sector: {info['sector']} / {info['sub_sector']} | {info['tier']}",
            border_style="cyan",
        ))

        sr = SymbolResult(symbol=symbol, sector=info["sector"])

        # Step 1: Individual rules
        sr.individual_results = self._screen_individual_rules(symbol, sector_config)

        # Step 2: Combos
        sr.combo_results = self._screen_combos(symbol, sector_config)

        # Step 3: Parameter sweep on top combos
        valid_combos = [(n, k, r) for n, k, r in sr.combo_results if r.total_trades >= 5]
        valid_combos.sort(key=lambda x: x[2].sharpe_ratio or 0, reverse=True)
        top = valid_combos[:self.top_combos]

        if not top:
            console.print(f"  [red]{symbol}: No combos with 5+ trades — skipping sweep[/red]")
            sr.elapsed = time.time() - sym_start
            sr.recommendation = "skip"
            return sr

        sr.sweep_results = self._sweep_params(symbol, top)

        # Step 4: Full validation on top unique configs
        console.print(f"\n  [bold]STEP 4: Full 4-Gate Validation ({symbol})[/bold]")
        valid_sweep = [(l, k, r) for l, k, r in sr.sweep_results if r.total_trades >= 5]
        valid_sweep.sort(key=lambda x: x[2].sharpe_ratio or 0, reverse=True)

        seen_keys = set()
        unique_top = []
        for label, kwargs, result in valid_sweep:
            key = (result.total_trades, round(result.total_return, 4), round(result.sharpe_ratio or 0, 4))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_top.append((label, kwargs, result))
            if len(unique_top) >= self.top_validate:
                break

        for label, kwargs, _ in unique_top:
            report = self._run_full_validation(symbol, label, kwargs)
            sr.validated_reports.append(report)

        self._set_recommendation(sr, sym_start)
        return sr

    def _set_recommendation(self, sr: SymbolResult, sym_start: float):
        """Set recommendation based on gate results and record elapsed time."""
        sr.validated_reports.sort(
            key=lambda r: (r.pass_count, r.backtest.sharpe_ratio or 0 if r.backtest else 0),
            reverse=True,
        )
        sr.elapsed = time.time() - sym_start
        winner = sr.validated_reports[0] if sr.validated_reports else None
        if winner and winner.backtest:
            n_gates = len(winner.gates)
            console.print(
                f"\n  [bold]{sr.symbol} best: {winner.label} ({winner.pass_count}/{n_gates} gates, "
                f"Sharpe={winner.backtest.sharpe_ratio or 0:.2f}, "
                f"Return={winner.backtest.total_return:+.1%})[/bold]"
            )
            # Deploy if passing majority of gates (>= 60%)
            pass_pct = winner.pass_count / n_gates if n_gates > 0 else 0
            if pass_pct >= 0.6:
                sr.recommendation = "deploy"
            elif pass_pct >= 0.4:
                sr.recommendation = "conditional"
            else:
                sr.recommendation = "skip"
        else:
            sr.recommendation = "skip"
        console.print(f"  {sr.symbol} completed in {sr.elapsed / 60:.1f} min")

    def validate_symbol_retune(
        self, symbol: str, known_combos: List[Tuple[str, dict]]
    ) -> SymbolResult:
        """Run Steps 3+4 only using known combos from a profile.

        Skips individual rule screen and combo screen.
        """
        info = get_symbol_sector(symbol)
        sym_start = time.time()
        self._ensure_vec_engine(symbol)

        console.print(f"\n{'=' * 70}")
        console.print(Panel.fit(
            f"[bold cyan]{symbol} — {info['name']}[/bold cyan]\n"
            f"Sector: {info['sector']} / {info['sub_sector']} | [yellow]RETUNE[/yellow]",
            border_style="yellow",
        ))

        sr = SymbolResult(symbol=symbol, sector=info["sector"])

        # Run each known combo once to get a baseline result for the sweep
        top = []
        for combo_name, base_kwargs in known_combos:
            rules = base_kwargs["rules"]
            kwargs = dict(
                rules=rules,
                min_confidence=0.50,
                profit_target=0.10,
                stop_loss=1.0,
                max_loss_pct=5.0,
                cooldown_bars=5,
                **self.common_kwargs,
            )
            try:
                result = self._run_backtest(symbol, kwargs)
                if result.total_trades >= 5:
                    top.append((combo_name, kwargs, result))
            except Exception:
                pass

        if not top:
            console.print(f"  [red]{symbol}: No known combos produced 5+ trades[/red]")
            sr.elapsed = time.time() - sym_start
            sr.recommendation = "skip"
            return sr

        # Step 3: Sweep
        sr.sweep_results = self._sweep_params(symbol, top)

        # Step 4: Validate top unique configs
        console.print(f"\n  [bold]STEP 4: Full 4-Gate Validation ({symbol})[/bold]")
        valid_sweep = [(l, k, r) for l, k, r in sr.sweep_results if r.total_trades >= 5]
        valid_sweep.sort(key=lambda x: x[2].sharpe_ratio or 0, reverse=True)

        seen_keys = set()
        unique_top = []
        for label, kwargs, result in valid_sweep:
            key = (result.total_trades, round(result.total_return, 4), round(result.sharpe_ratio or 0, 4))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_top.append((label, kwargs, result))
            if len(unique_top) >= self.top_validate:
                break

        for label, kwargs, _ in unique_top:
            report = self._run_full_validation(symbol, label, kwargs)
            sr.validated_reports.append(report)

        self._set_recommendation(sr, sym_start)
        return sr

    def validate_symbol_revalidate(
        self, symbol: str, winning_configs: List[Tuple[str, dict]]
    ) -> SymbolResult:
        """Run Step 4 only on exact kwargs from a profile.

        Skips individual rule screen, combo screen, and parameter sweep.
        """
        info = get_symbol_sector(symbol)
        sym_start = time.time()

        console.print(f"\n{'=' * 70}")
        console.print(Panel.fit(
            f"[bold cyan]{symbol} — {info['name']}[/bold cyan]\n"
            f"Sector: {info['sector']} / {info['sub_sector']} | [green]REVALIDATE[/green]",
            border_style="green",
        ))

        sr = SymbolResult(symbol=symbol, sector=info["sector"])

        console.print(f"\n  [bold]STEP 4: Full 4-Gate Validation ({symbol})[/bold]")
        for label, kwargs in winning_configs:
            report = self._run_full_validation(symbol, label, kwargs)
            sr.validated_reports.append(report)

        self._set_recommendation(sr, sym_start)
        return sr

    # ── Step 1: Individual rule screen ───────────────────────────────────

    def _screen_individual_rules(
        self, symbol: str, sector_config: SectorConfig
    ) -> List[tuple]:
        console.print(f"\n  [bold]STEP 1: Individual Rule Screen ({symbol})[/bold]")

        default_kwargs = dict(
            min_confidence=0.50,
            profit_target=0.10,
            stop_loss=1.0,
            max_loss_pct=5.0,
            cooldown_bars=5,
            **self.common_kwargs,
        )

        results = []
        table = Table(title=f"{symbol} — Individual Rules")
        table.add_column("Rule", style="cyan", min_width=25)
        table.add_column("Trades", justify="right")
        table.add_column("WR", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("PF", justify="right")
        table.add_column("Grade", justify="center")

        for rule_name in sector_config.individual_rules:
            try:
                kwargs = {**default_kwargs, "rules": [rule_name]}
                result = self._run_backtest(symbol, kwargs)
                sharpe = result.sharpe_ratio or 0
                if result.total_trades < 3:
                    grade = "[dim]F (no trades)[/dim]"
                elif sharpe >= 0.8 and result.total_return > 0:
                    grade = "[bold green]A[/bold green]"
                elif sharpe >= 0.4 and result.total_return > 0:
                    grade = "[green]B[/green]"
                elif result.total_return > 0:
                    grade = "[yellow]C[/yellow]"
                else:
                    grade = "[red]F[/red]"

                table.add_row(
                    rule_name, str(result.total_trades),
                    f"{result.win_rate:.0%}", f"{result.total_return:+.1%}",
                    f"{sharpe:.2f}", f"{result.profit_factor or 0:.2f}", grade,
                )
                results.append((rule_name, kwargs, result))
            except Exception:
                table.add_row(rule_name, "-", "-", "-", "-", "-", "[red]ERR[/red]")

        console.print(table)
        results.sort(key=lambda x: x[2].sharpe_ratio or 0, reverse=True)
        return results

    # ── Step 2: Combo screen ─────────────────────────────────────────────

    def _screen_combos(
        self, symbol: str, sector_config: SectorConfig
    ) -> List[tuple]:
        console.print(f"\n  [bold]STEP 2: Combo Screen ({symbol})[/bold]")

        default_kwargs = dict(
            min_confidence=0.50,
            profit_target=0.10,
            stop_loss=1.0,
            max_loss_pct=5.0,
            cooldown_bars=5,
            **self.common_kwargs,
        )

        results = []
        table = Table(title=f"{symbol} — Rule Combos")
        table.add_column("Combo", style="cyan", min_width=20)
        table.add_column("# Rules", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("WR", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("PF", justify="right")

        for combo_name, rules in sector_config.rule_combos.items():
            try:
                kwargs = {**default_kwargs, "rules": rules}
                result = self._run_backtest(symbol, kwargs)
                table.add_row(
                    combo_name, str(len(rules)), str(result.total_trades),
                    f"{result.win_rate:.0%}", f"{result.total_return:+.1%}",
                    f"{result.sharpe_ratio or 0:.2f}", f"{result.profit_factor or 0:.2f}",
                )
                results.append((combo_name, kwargs, result))
            except Exception:
                table.add_row(combo_name, str(len(rules)), "-", "-", "-", "-", "-")

        console.print(table)
        results.sort(
            key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
            reverse=True,
        )
        return results

    # ── Step 3: Two-phase adaptive parameter sweep ──────────────────────

    @staticmethod
    def _neighbor_values(center, all_values):
        """Get center and its immediate neighbors from the full grid."""
        if center not in all_values:
            return [center]
        idx = all_values.index(center)
        neighbors = set()
        if idx > 0:
            neighbors.add(all_values[idx - 1])
        neighbors.add(center)
        if idx < len(all_values) - 1:
            neighbors.add(all_values[idx + 1])
        return sorted(neighbors)

    def _sweep_params(
        self, symbol: str, top_combos: List[tuple]
    ) -> List[tuple]:
        console.print(f"\n  [bold]STEP 3: Two-Phase Parameter Sweep ({symbol})[/bold]")

        all_results = []
        sweep_count = 0
        # Vectorized engine is fast enough for serial — skip process pool overhead
        use_parallel = not self.serial_sweep and self._vec_engine is None

        for combo_name, base_kwargs, _ in top_combos:
            rules = base_kwargs["rules"]

            # ── Phase 1: Coarse grid (3×3×3×2 = 54 configs) ──
            coarse_configs = []
            for conf in COARSE_CONFIDENCE:
                for pt in COARSE_PROFIT_TARGET:
                    for ml in COARSE_MAX_LOSS:
                        for cd in COARSE_COOLDOWN:
                            kwargs = dict(
                                rules=rules,
                                min_confidence=conf,
                                profit_target=pt,
                                stop_loss=1.0,
                                max_loss_pct=ml,
                                cooldown_bars=cd,
                                **self.common_kwargs,
                            )
                            label = f"{combo_name} c={conf} pt={pt:.0%} ml={ml}% cd={cd}"
                            coarse_configs.append((label, kwargs))

            console.print(f"    {combo_name}: Phase 1 coarse ({len(coarse_configs)} configs)")

            if use_parallel:
                try:
                    coarse_results = parallel_sweep(
                        coarse_configs, symbol, self.start_date, self.end_date,
                        initial_cash=self.initial_cash, max_workers=6,
                    )
                except Exception as e:
                    console.print(f"    [yellow]Parallel sweep failed ({e}), falling back to serial[/yellow]")
                    use_parallel = False
                    coarse_results = self._serial_sweep(symbol, coarse_configs)
            else:
                coarse_results = self._serial_sweep(symbol, coarse_configs)

            sweep_count += len(coarse_results)

            # Sort coarse results by Sharpe (min 5 trades)
            coarse_results.sort(
                key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
                reverse=True,
            )

            # ── Phase 2: Fine-tune top 3 coarse winners ──
            top_coarse = [
                r for r in coarse_results
                if r[2].total_trades >= 5
            ][:3]

            if top_coarse:
                fine_configs = []
                seen_keys = set()  # avoid duplicate configs
                for _, winner_kwargs, _ in top_coarse:
                    conf_neighbors = self._neighbor_values(winner_kwargs["min_confidence"], CONFIDENCE_VALUES)
                    pt_neighbors = self._neighbor_values(winner_kwargs["profit_target"], PROFIT_TARGET_VALUES)
                    ml_neighbors = self._neighbor_values(winner_kwargs["max_loss_pct"], MAX_LOSS_VALUES)
                    cd_neighbors = self._neighbor_values(winner_kwargs["cooldown_bars"], COOLDOWN_VALUES)

                    for conf in conf_neighbors:
                        for pt in pt_neighbors:
                            for ml in ml_neighbors:
                                for cd in cd_neighbors:
                                    key = (conf, pt, ml, cd)
                                    if key in seen_keys:
                                        continue
                                    seen_keys.add(key)
                                    kwargs = dict(
                                        rules=rules,
                                        min_confidence=conf,
                                        profit_target=pt,
                                        stop_loss=1.0,
                                        max_loss_pct=ml,
                                        cooldown_bars=cd,
                                        **self.common_kwargs,
                                    )
                                    label = f"{combo_name} c={conf} pt={pt:.0%} ml={ml}% cd={cd}"
                                    fine_configs.append((label, kwargs))

                console.print(f"    {combo_name}: Phase 2 fine-tune ({len(fine_configs)} configs around top 3)")

                if use_parallel:
                    try:
                        fine_results = parallel_sweep(
                            fine_configs, symbol, self.start_date, self.end_date,
                            initial_cash=self.initial_cash, max_workers=6,
                        )
                    except Exception:
                        fine_results = self._serial_sweep(symbol, fine_configs)
                else:
                    fine_results = self._serial_sweep(symbol, fine_configs)

                sweep_count += len(fine_results)
                coarse_results.extend(fine_results)

            # Log top 3 for this combo
            coarse_results.sort(
                key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
                reverse=True,
            )
            for label, _, r in coarse_results[:3]:
                if r.total_trades >= 5:
                    console.print(
                        f"      {label}: Sharpe={r.sharpe_ratio or 0:.2f}, "
                        f"Return={r.total_return:+.1%}, {r.total_trades}t"
                    )
            all_results.extend(coarse_results)

        console.print(f"    Total configs swept: {sweep_count}")

        # ATR stops on top 5 (skip in vectorized mode — ATR logic not vectorized)
        if self._vec_engine is None:
            all_results.sort(
                key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
                reverse=True,
            )
            atr_configs = []
            for label, kwargs, _ in all_results[:5]:
                for atr_mult in [2.0, 2.5, 3.0]:
                    atr_kwargs = {
                        **kwargs,
                        "stop_mode": "atr",
                        "atr_multiplier": atr_mult,
                        "atr_stop_min_pct": 3.0,
                        "atr_stop_max_pct": 15.0,
                    }
                    atr_label = f"{label} ATR={atr_mult}"
                    atr_configs.append((atr_label, atr_kwargs))

            if atr_configs:
                if use_parallel:
                    try:
                        atr_results = parallel_sweep(
                            atr_configs, symbol, self.start_date, self.end_date,
                            initial_cash=self.initial_cash, max_workers=6,
                        )
                    except Exception:
                        atr_results = self._serial_sweep(symbol, atr_configs)
                else:
                    atr_results = self._serial_sweep(symbol, atr_configs)
                all_results.extend(atr_results)

        all_results.sort(
            key=lambda x: x[2].sharpe_ratio or 0 if x[2].total_trades >= 5 else -999,
            reverse=True,
        )
        return all_results

    def _serial_sweep(self, symbol: str, configs: List[tuple]) -> List[tuple]:
        """Serial sweep — uses vectorized engine if available."""
        results = []
        for label, kwargs in configs:
            try:
                result = self._run_backtest(symbol, kwargs)
                results.append((label, kwargs, result))
            except Exception:
                pass
        return results

    # ── Step 4: Full 4-gate validation ───────────────────────────────────

    def _run_full_validation(
        self, symbol: str, label: str, run_kwargs: Dict
    ) -> ValidationReport:
        report = ValidationReport(label=label, kwargs=run_kwargs.copy())
        console.print(f"\n    [cyan]Validating: {label}[/cyan]")

        # Full-period backtest
        try:
            result = self.runner.run(
                symbol=symbol, start_date=self.start_date, end_date=self.end_date, **run_kwargs
            )
        except Exception as e:
            console.print(f"    [red]Backtest failed: {e}[/red]")
            report.gates = [
                GateResult(g, False, f"Backtest failed: {e}")
                for g in ["Walk-Backward", "Bootstrap", "Monte Carlo", "Regime"]
            ]
            return report

        report.backtest = result
        console.print(
            f"    {result.total_trades}t, WR={result.win_rate:.0%}, "
            f"Return={result.total_return:+.1%}, Sharpe={result.sharpe_ratio or 0:.2f}, "
            f"PF={result.profit_factor or 0:.2f}, DD=-{result.max_drawdown_pct or 0:.1f}%"
        )

        if result.total_trades < 2:
            # Build regime gate names from config
            regime_gate_names = []
            if self.regime_windows:
                regime_gate_names = [f"WB:{rw['label']}" for rw in self.regime_windows]
            else:
                regime_gate_names = ["WB:regime"]
            report.gates = [
                GateResult(g, False, "Too few trades")
                for g in regime_gate_names + ["Bootstrap", "Monte Carlo", "Regime"]
            ]
            return report

        # Walk-Backward regime gates — each regime window is its own gate
        # A bear gate tests survival, a chop gate tests sideways profitability
        try:
            wb_validator = WalkBackwardValidator(self.runner)
            wb_kwargs = dict(
                symbol=symbol,
                data_start=self.start_date,
                data_end=self.end_date,
                tune_months=DEFAULT_TUNE_MONTHS,
                holdout_months=DEFAULT_HOLDOUT_MONTHS,
                min_regimes_pass=DEFAULT_MIN_REGIMES_PASS,
                **run_kwargs,
            )
            if self.regime_windows:
                wb_kwargs["regime_windows"] = self.regime_windows
            wb_result = wb_validator.validate(**wb_kwargs)

            for w in wb_result.regime_windows:
                gate_name = f"WB:{w.label}"
                ws = "[green]PASS[/green]" if w.passed else "[red]FAIL[/red]"
                detail = f"{w.total_return:+.1%} ({w.total_trades}t)"
                report.gates.append(GateResult(gate_name, w.passed, detail, w))
                console.print(f"      {gate_name}: {ws} ({detail})")
        except Exception as e:
            # If WB fails entirely, add one failed gate per expected window
            if self.regime_windows:
                for rw in self.regime_windows:
                    report.gates.append(GateResult(f"WB:{rw['label']}", False, f"Error: {e}"))
            else:
                report.gates.append(GateResult("WB:regime", False, f"Error: {e}"))

        # Gate 2: Bootstrap
        try:
            bs_result = bootstrap_analysis(result, n_bootstrap=10000)
            bs_passed = bs_result.p_value < 0.05 and not bs_result.no_edge_sharpe
            bs_detail = (
                f"p={bs_result.p_value:.4f}, "
                f"Sharpe CI=[{bs_result.sharpe_ci_lower:.2f}, {bs_result.sharpe_ci_upper:.2f}]"
            )
            report.gates.append(GateResult("Bootstrap", bs_passed, bs_detail, bs_result))
            s = "[green]PASS[/green]" if bs_passed else "[red]FAIL[/red]"
            console.print(f"      BS: {s} (p={bs_result.p_value:.4f})")
        except Exception as e:
            report.gates.append(GateResult("Bootstrap", False, f"Error: {e}"))

        # Gate 3: Monte Carlo
        try:
            mc_result = monte_carlo_analysis(result, n_simulations=10000, initial_cash=self.initial_cash)
            mc_passed = mc_result.ruin_probability < 0.10 and mc_result.drawdown_p95 < 40
            mc_detail = (
                f"Ruin={mc_result.ruin_probability:.1%}, "
                f"P95 DD=-{mc_result.drawdown_p95:.1f}%"
            )
            report.gates.append(GateResult("Monte Carlo", mc_passed, mc_detail, mc_result))
            s = "[green]PASS[/green]" if mc_passed else "[red]FAIL[/red]"
            console.print(f"      MC: {s} (Ruin={mc_result.ruin_probability:.1%}, DD P95=-{mc_result.drawdown_p95:.1f}%)")
        except Exception as e:
            report.gates.append(GateResult("Monte Carlo", False, f"Error: {e}"))

        # Gate 4: Regime Analysis
        try:
            regime_result = analyze_by_regime(result, self.runner.loader)
            regime_passed = not regime_result.regime_dependent
            parts = []
            for rname in ["bull", "bear", "chop", "volatile", "crisis"]:
                if rname in regime_result.regime_metrics:
                    m = regime_result.regime_metrics[rname]
                    parts.append(f"{rname}:{m.total_trades}t/{m.total_return * 100:+.1f}%")
            regime_detail = ", ".join(parts) if parts else "No trades classified"
            report.gates.append(GateResult("Regime", regime_passed, regime_detail, regime_result))
            s = "[green]PASS[/green]" if regime_passed else "[red]FAIL[/red]"
            console.print(f"      Reg: {s} ({regime_detail})")
        except Exception as e:
            report.gates.append(GateResult("Regime", False, f"Error: {e}"))

        console.print(f"      [bold]Result: {report.pass_count}/{len(report.gates)} gates[/bold]")
        return report
