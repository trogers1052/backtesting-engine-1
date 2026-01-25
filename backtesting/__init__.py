"""
Backtesting Service - Strategy Testing Engine

Test decision-engine trading rules against historical data from TimescaleDB.

Usage:
    from backtesting import BacktraderRunner

    runner = BacktraderRunner()
    result = runner.run(
        symbol="WPM",
        start_date=date(2021, 1, 1),
        rules=["buy_dip_in_uptrend"],
    )
    print(f"Win rate: {result.win_rate:.1%}")
"""

__version__ = "0.1.0"

from .config import settings
from .engine import BacktraderRunner
from .strategies import DecisionEngineStrategy, create_strategy, list_available_rules
from .data import TimescaleLoader
from .reporting import print_report, export_json

__all__ = [
    "settings",
    "BacktraderRunner",
    "DecisionEngineStrategy",
    "create_strategy",
    "list_available_rules",
    "TimescaleLoader",
    "print_report",
    "export_json",
]
