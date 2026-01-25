"""Reporting modules for backtesting results."""

from .console_report import print_report
from .json_report import export_json

__all__ = ["print_report", "export_json"]
