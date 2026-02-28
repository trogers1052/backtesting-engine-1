"""
Backtesting Service - Main Entry Point

Tests trading strategies against historical data from TimescaleDB.

Usage:
    # Basic usage
    python -m backtesting --symbol WPM --start 2021-01-01

    # Multiple symbols
    python -m backtesting --symbol WPM,GOLD,NEM --start 2021-01-01

    # Custom rules and parameters
    python -m backtesting --symbol WPM \\
        --rules buy_dip_in_uptrend,strong_buy_signal \\
        --profit-target 0.10 \\
        --stop-loss 0.03 \\
        --min-confidence 0.7

    # Show trade details
    python -m backtesting --symbol WPM --show-trades

    # Export to JSON
    python -m backtesting --symbol WPM --output results.json

    # List available rules
    python -m backtesting --list-rules
"""

import argparse
import logging
import signal
import sys
from datetime import date, datetime
from typing import List

from .config import settings
from .engine import BacktraderRunner
from .strategies import list_available_rules
from .reporting import print_report, export_json
from .reporting.console_report import print_multi_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level flag for cooperative shutdown during long-running backtests.
_shutdown_requested = False


def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols."""
    return [s.strip().upper() for s in symbols_str.split(",")]


def parse_rules(rules_str: str) -> List[str]:
    """Parse comma-separated rule names."""
    return [r.strip().lower() for r in rules_str.split(",")]


def _handle_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        # Second signal: force exit immediately
        logger.warning(f"Received {sig_name} again, forcing exit")
        sys.exit(1)
    logger.info(f"Received {sig_name}, finishing current backtest then exiting...")
    _shutdown_requested = True


def main():
    """Main entry point for the backtesting service."""
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    parser = argparse.ArgumentParser(
        description="Backtest trading strategies against historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol WPM --start 2021-01-01
  %(prog)s --symbol WPM,GOLD --rules buy_dip_in_uptrend,strong_buy_signal
  %(prog)s --list-rules
        """,
    )

    # Symbol and date arguments
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Stock symbol(s) to test (comma-separated, e.g., WPM,GOLD,NEM)",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=parse_date(settings.default_start_date),
        help=f"Start date (YYYY-MM-DD, default: {settings.default_start_date})",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=date.today(),
        help="End date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default=settings.default_timeframe,
        choices=["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"],
        help=f"Bar timeframe (default: {settings.default_timeframe})",
    )
    parser.add_argument(
        "--exit-timeframe",
        type=str,
        default=settings.default_exit_timeframe,
        choices=["1min", "5min", "15min", "30min", "1hour"],
        help="Intraday timeframe for exits/entries (enables multi-timeframe mode)",
    )

    # Strategy arguments
    parser.add_argument(
        "--rules", "-r",
        type=str,
        default="buy_dip_in_uptrend",
        help="Rule names to use (comma-separated, default: buy_dip_in_uptrend)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=settings.default_min_confidence,
        help=f"Minimum confidence threshold (default: {settings.default_min_confidence})",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=settings.default_profit_target,
        help=f"Profit target percentage (default: {settings.default_profit_target})",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=settings.default_stop_loss,
        help=f"Stop loss percentage (default: {settings.default_stop_loss})",
    )
    parser.add_argument(
        "--require-consensus",
        action="store_true",
        help="Require multiple rules to agree for signals",
    )

    # Scale-in (averaging down) arguments
    parser.add_argument(
        "--allow-scale-in",
        action="store_true",
        help="Allow averaging down on deeper dips (use with average_down rule)",
    )
    parser.add_argument(
        "--max-scale-ins",
        type=int,
        default=2,
        help="Maximum number of scale-ins per position (default: 2)",
    )
    parser.add_argument(
        "--scale-in-size",
        type=float,
        default=0.5,
        help="Scale-in size relative to initial position (default: 0.5 = half)",
    )

    # Risk-based sizing arguments
    parser.add_argument(
        "--sizing-mode",
        type=str,
        choices=["percent", "risk_based"],
        default=settings.default_sizing_mode,
        help=f"Position sizing mode (default: {settings.default_sizing_mode})",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=settings.default_risk_pct,
        help=f"Risk %% per trade for risk_based mode (default: {settings.default_risk_pct})",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=settings.default_max_position_pct,
        help=f"Max position %% for risk_based mode (default: {settings.default_max_position_pct})",
    )
    parser.add_argument(
        "--stop-mode",
        type=str,
        choices=["fixed", "atr"],
        default=settings.default_stop_mode,
        help=f"Stop loss calculation mode (default: {settings.default_stop_mode})",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=settings.default_atr_multiplier,
        help=f"ATR multiplier for stop calculation (default: {settings.default_atr_multiplier})",
    )

    # Trade filter arguments
    parser.add_argument(
        "--max-extension",
        type=float,
        default=settings.default_max_price_extension_pct,
        help=f"Max price extension above SMA_20 %% to allow entry (default: {settings.default_max_price_extension_pct})",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=settings.default_cooldown_bars,
        help=f"Bars to wait after exit before re-entry (default: {settings.default_cooldown_bars})",
    )
    parser.add_argument(
        "--max-trend-spread",
        type=float,
        default=settings.default_max_trend_spread_pct,
        help=f"Max SMA_20/SMA_50 spread %% (default: {settings.default_max_trend_spread_pct})",
    )

    parser.add_argument(
        "--max-loss",
        type=float,
        default=settings.default_max_loss_pct,
        help=f"Max loss %% per trade before forced exit (default: {settings.default_max_loss_pct})",
    )

    # Capital arguments
    parser.add_argument(
        "--cash",
        type=float,
        default=settings.default_initial_cash,
        help=f"Initial cash (default: ${settings.default_initial_cash:,.0f})",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=settings.default_commission,
        help=f"Commission rate (default: {settings.default_commission})",
    )
    parser.add_argument(
        "--compound",
        action="store_true",
        default=True,
        help="Reinvest profits into larger positions (default: enabled)",
    )
    parser.add_argument(
        "--no-compound",
        action="store_true",
        help="Use fixed position sizes based on initial capital (no reinvestment)",
    )

    # Output arguments
    parser.add_argument(
        "--show-trades", "-t",
        action="store_true",
        help="Show individual trade details",
    )
    parser.add_argument(
        "--trade-limit",
        type=int,
        default=10,
        help="Maximum trades to show (default: 10, 0 = all)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )

    # Info arguments
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List available trading rules",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List symbols available in database",
    )

    # Validation arguments
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable walk-forward validation (single symbol only)",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.7,
        help="Train split percentage for walk-forward (default: 0.7)",
    )
    parser.add_argument(
        "--rolling-wf",
        action="store_true",
        help="Use rolling walk-forward with multiple windows",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run bootstrap significance tests on trade P&L",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples (default: 10000)",
    )
    parser.add_argument(
        "--regime-analysis",
        action="store_true",
        help="Analyze results by market regime (bull/bear/chop using SPY)",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="Gap days between train/test in walk-forward to prevent leakage (default: 5)",
    )
    parser.add_argument(
        "--purge-days",
        type=int,
        default=10,
        help="Days to trim from end of training to prevent trade overlap with test (default: 10)",
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list_rules:
        print("\nAvailable Trading Rules:")
        print("=" * 60)
        for name, description in list_available_rules().items():
            print(f"  {name}")
            print(f"      {description}")
        print()
        return 0

    if args.list_symbols:
        from .data import TimescaleLoader
        loader = TimescaleLoader()
        symbols = loader.get_available_symbols()
        print("\nAvailable Symbols in Database:")
        print("=" * 40)
        for sym in symbols:
            min_date, max_date, count = loader.get_date_range(sym)
            if min_date:
                print(f"  {sym}: {min_date.date()} to {max_date.date()} ({count} bars)")
            else:
                print(f"  {sym}: No data")
        print()
        return 0

    # Validate required arguments
    if not args.symbol:
        parser.error("--symbol is required (use --list-symbols to see available)")

    # Parse arguments
    symbols = parse_symbols(args.symbol)
    rules = parse_rules(args.rules)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Determine compounding mode
    compound = not args.no_compound  # Default is to compound

    # Create runner
    runner = BacktraderRunner(
        initial_cash=args.cash,
        commission=args.commission,
        compound=compound,
        sizing_mode=args.sizing_mode,
        risk_pct=args.risk_pct,
        max_position_pct=args.max_position_pct,
    )

    # Common run kwargs shared across single and multi-symbol backtests
    run_kwargs = dict(
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        rules=rules,
        min_confidence=args.min_confidence,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        require_consensus=args.require_consensus,
        allow_scale_in=args.allow_scale_in,
        max_scale_ins=args.max_scale_ins,
        scale_in_size=args.scale_in_size,
        stop_mode=args.stop_mode,
        atr_multiplier=args.atr_multiplier,
        max_price_extension_pct=args.max_extension,
        cooldown_bars=args.cooldown_bars,
        max_trend_spread_pct=args.max_trend_spread,
        max_loss_pct=args.max_loss,
        exit_timeframe=args.exit_timeframe,
    )

    # Run backtest(s)
    try:
        if len(symbols) == 1:
            # Single symbol
            result = runner.run(symbol=symbols[0], **run_kwargs)

            # Output
            if not args.quiet:
                print_report(
                    result,
                    show_trades=args.show_trades,
                    trade_limit=args.trade_limit,
                )

            if args.output:
                export_json(result, args.output)
                print(f"Results exported to {args.output}")

            # Validation analyses (single symbol only)
            if args.walk_forward:
                from .validation import WalkForwardValidator
                from .validation.report import print_walk_forward_report

                # Strip date keys — walk-forward sets its own date ranges
                wf_kwargs = {
                    k: v for k, v in run_kwargs.items()
                    if k not in ("start_date", "end_date")
                }
                validator = WalkForwardValidator(runner)
                if args.rolling_wf:
                    wf_result = validator.validate_rolling(
                        symbols[0], args.start, args.end,
                        train_days=730, test_days=365, step_days=365,
                        embargo_days=args.embargo_days,
                        purge_days=args.purge_days,
                        **wf_kwargs,
                    )
                else:
                    wf_result = validator.validate_simple(
                        symbols[0], args.start, args.end,
                        train_pct=args.train_pct,
                        embargo_days=args.embargo_days,
                        purge_days=args.purge_days,
                        **wf_kwargs,
                    )
                print_walk_forward_report(wf_result)

            if args.bootstrap and result.total_trades >= 2:
                from .validation import bootstrap_analysis
                from .validation.report import print_bootstrap_report

                bs_result = bootstrap_analysis(
                    result, n_bootstrap=args.n_bootstrap,
                )
                print_bootstrap_report(bs_result)

            if args.regime_analysis:
                from .validation import analyze_by_regime
                from .validation.report import print_regime_report

                regime_result = analyze_by_regime(result, runner.loader)
                print_regime_report(regime_result)

        else:
            # Multiple symbols — run individually so we can check for
            # shutdown between symbols for cooperative cancellation.
            results = {}
            for symbol in symbols:
                if _shutdown_requested:
                    logger.info(
                        f"Shutdown requested, skipping remaining symbols "
                        f"({len(symbols) - len(results)} of {len(symbols)} left)"
                    )
                    break
                try:
                    results[symbol] = runner.run(symbol=symbol, **run_kwargs)
                except Exception as e:
                    logger.error(f"Failed to backtest {symbol}: {e}")

            # Output whatever results we collected
            if results:
                if not args.quiet:
                    print_multi_report(results)

                    if args.show_trades:
                        for sym, res in results.items():
                            print(f"\n{'=' * 60}")
                            print(f"Trades for {sym}:")
                            print_report(res, show_trades=True, trade_limit=args.trade_limit)

                if args.output:
                    export_json(results, args.output)
                    print(f"Results exported to {args.output}")
            elif not _shutdown_requested:
                logger.warning("No backtest results produced")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

    if _shutdown_requested:
        logger.info("Backtesting service shut down cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
