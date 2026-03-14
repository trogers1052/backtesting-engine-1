"""
Multiprocessing support for validation pipeline.

Two levels of parallelism (never nested):
  1. Symbol-level: run N symbols simultaneously, each on its own core.
     Each symbol runs its full pipeline serially (including sweep).
  2. Sweep-level: run N configs in parallel for one symbol's param sweep.
     Used when running symbols serially.

Uses ProcessPoolExecutor with fork context so workers inherit the parent's
loaded modules and state. Each worker gets its own BacktraderRunner (with
its own DB connection and indicator cache).
"""
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level runner for worker processes (set by initializer)
_worker_runner = None


def _init_worker(initial_cash: float):
    """Initialize a BacktraderRunner in each worker process.

    With fork context, modules are already imported — just create a fresh
    runner with its own DB connection.
    """
    global _worker_runner
    from backtesting.engine import BacktraderRunner
    _worker_runner = BacktraderRunner(initial_cash=initial_cash)


def _run_one_config(args: tuple) -> Optional[tuple]:
    """Run a single backtest config in a worker process.

    Args is a tuple of (label, symbol, start_date_iso, end_date_iso, kwargs).
    Returns (label, kwargs, result) or None on failure.
    """
    label, symbol, start_date_iso, end_date_iso, kwargs = args
    global _worker_runner
    try:
        result = _worker_runner.run(
            symbol=symbol,
            start_date=date.fromisoformat(start_date_iso),
            end_date=date.fromisoformat(end_date_iso),
            **kwargs,
        )
        return (label, kwargs, result)
    except Exception:
        return None


def parallel_sweep(
    configs: List[tuple],
    symbol: str,
    start_date: date,
    end_date: date,
    initial_cash: float,
    max_workers: int = 6,
) -> List[tuple]:
    """Run multiple backtest configs in parallel.

    Args:
        configs: List of (label, kwargs) tuples
        symbol: Symbol to backtest
        start_date: Start date
        end_date: End date
        initial_cash: Initial cash for each runner
        max_workers: Number of parallel workers

    Returns:
        List of (label, kwargs, result) tuples for successful runs
    """
    if not configs:
        return []

    # Pack args for pickling (dates as ISO strings)
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()
    work_items = [
        (label, symbol, start_iso, end_iso, kwargs)
        for label, kwargs in configs
    ]

    # Use fork context to inherit loaded modules (avoids re-import overhead).
    # Each worker still creates its own BacktraderRunner + DB connection.
    fork_ctx = multiprocessing.get_context("fork")

    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(initial_cash,),
        mp_context=fork_ctx,
    ) as executor:
        futures = {
            executor.submit(_run_one_config, item): item[0]
            for item in work_items
        }
        for future in as_completed(futures):
            try:
                out = future.result()
                if out is not None:
                    results.append(out)
            except Exception as e:
                logger.warning(f"Worker error: {e}")

    return results


# ── Symbol-level parallelism ──────────────────────────────────────────


def _run_one_symbol(args: tuple):
    """Run the full validation pipeline for one symbol in a worker process.

    Returns (symbol, SymbolResult) or (symbol, None) on failure.
    """
    symbol, start_date_iso, end_date_iso, initial_cash, exit_tf, \
        top_combos, top_validate, regime_windows, mode, known_combos, winning_configs = args

    try:
        from backtesting.engine import BacktraderRunner
        from backtesting.validation.universal.data_cache import DataCache
        from backtesting.validation.universal.pipeline import UniversalValidator

        runner = BacktraderRunner(initial_cash=initial_cash)
        data_cache = DataCache()
        validator = UniversalValidator(
            runner=runner,
            start_date=date.fromisoformat(start_date_iso),
            end_date=date.fromisoformat(end_date_iso),
            initial_cash=initial_cash,
            exit_timeframe=exit_tf,
            top_combos=top_combos,
            top_validate=top_validate,
            regime_windows=regime_windows,
            serial_sweep=True,  # No nested multiprocessing
            data_cache=data_cache,
        )

        if mode == "full":
            result = validator.validate_symbol(symbol)
        elif mode == "retune":
            result = validator.validate_symbol_retune(symbol, known_combos)
        elif mode == "revalidate":
            result = validator.validate_symbol_revalidate(symbol, winning_configs)
        else:
            result = validator.validate_symbol(symbol)

        # Free memory before returning
        runner.clear_indicator_cache()
        return (symbol, result)
    except Exception as e:
        logger.error(f"Symbol worker failed for {symbol}: {e}")
        return (symbol, None)


def parallel_validate_symbols(
    work_items: List[Dict],
    max_workers: int = 6,
) -> List[tuple]:
    """Run full validation pipeline for multiple symbols in parallel.

    Args:
        work_items: List of dicts with keys:
            symbol, start_date, end_date, initial_cash, exit_timeframe,
            top_combos, top_validate, regime_windows, mode,
            known_combos (optional), winning_configs (optional)
        max_workers: Number of parallel symbol workers

    Returns:
        List of (symbol, SymbolResult) tuples
    """
    if not work_items:
        return []

    # Pack args as tuples for pickling
    packed = []
    for item in work_items:
        packed.append((
            item["symbol"],
            item["start_date"].isoformat(),
            item["end_date"].isoformat(),
            item["initial_cash"],
            item["exit_timeframe"],
            item["top_combos"],
            item["top_validate"],
            item["regime_windows"],
            item["mode"],
            item.get("known_combos"),
            item.get("winning_configs"),
        ))

    fork_ctx = multiprocessing.get_context("fork")

    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=fork_ctx,
    ) as executor:
        futures = {
            executor.submit(_run_one_symbol, item): item[0]
            for item in packed
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                out = future.result()
                results.append(out)
                if out[1] is not None:
                    logger.info(f"{symbol}: completed ({out[1].recommendation})")
                else:
                    logger.warning(f"{symbol}: failed")
            except Exception as e:
                logger.error(f"{symbol}: worker exception: {e}")
                results.append((symbol, None))

    return results
