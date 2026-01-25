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


def main():
    """Main entry point for the backtesting service."""
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

    # Create runner
    runner = BacktraderRunner(
        initial_cash=args.cash,
        commission=args.commission,
    )

    # Run backtest(s)
    try:
        if len(symbols) == 1:
            # Single symbol
            result = runner.run(
                symbol=symbols[0],
                start_date=args.start,
                end_date=args.end,
                rules=rules,
                min_confidence=args.min_confidence,
                profit_target=args.profit_target,
                stop_loss=args.stop_loss,
                require_consensus=args.require_consensus,
            )

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

        else:
            # Multiple symbols
            results = runner.run_multiple(
                symbols=symbols,
                start_date=args.start,
                end_date=args.end,
                rules=rules,
                min_confidence=args.min_confidence,
                profit_target=args.profit_target,
                stop_loss=args.stop_loss,
                require_consensus=args.require_consensus,
            )

            # Output
            if not args.quiet:
                print_multi_report(results)

                if args.show_trades:
                    for symbol, result in results.items():
                        print(f"\n{'=' * 60}")
                        print(f"Trades for {symbol}:")
                        print_report(result, show_trades=True, trade_limit=args.trade_limit)

            if args.output:
                export_json(results, args.output)
                print(f"Results exported to {args.output}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
