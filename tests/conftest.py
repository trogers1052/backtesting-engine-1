"""Shared fixtures for backtesting-service tests."""

import os
import sys
import types
from datetime import date
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Mock pandas_ta before any backtesting imports (not available on Python <3.12)
if "pandas_ta" not in sys.modules:
    _pta = types.ModuleType("pandas_ta")
    _pta.rsi = MagicMock(return_value=None)
    _pta.sma = MagicMock(return_value=None)
    _pta.macd = MagicMock(return_value=None)
    _pta.bbands = MagicMock(return_value=None)
    _pta.atr = MagicMock(return_value=None)
    sys.modules["pandas_ta"] = _pta

# Ensure decision-engine is importable
de_path = os.environ.get(
    "DECISION_ENGINE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "decision-engine"),
)
de_path = os.path.abspath(de_path)
if de_path not in sys.path and os.path.isdir(de_path):
    sys.path.insert(0, de_path)


from backtesting.config import Settings
from backtesting.engine.backtrader_runner import BacktestResult


@pytest.fixture
def settings():
    """Default Settings instance (uses env defaults)."""
    return Settings()


@pytest.fixture
def sample_result():
    """A representative BacktestResult for report / serialization tests."""
    return BacktestResult(
        symbol="AAPL",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        strategy_name="buy_dip_in_uptrend",
        initial_cash=100_000.0,
        final_value=112_345.67,
        total_return=0.1234567,
        total_trades=25,
        winning_trades=15,
        losing_trades=10,
        win_rate=0.60,
        avg_win=1500.0,
        avg_loss=-800.0,
        profit_factor=2.81,
        sharpe_ratio=1.45,
        max_drawdown=5432.10,
        max_drawdown_pct=5.4,
        trades=[
            {
                "entry_date": "2023-03-15T09:30:00",
                "entry_price": 150.0,
                "entry_reason": "RSI dip in uptrend",
                "entry_confidence": 0.72,
                "exit_date": "2023-03-28T15:30:00",
                "exit_price": 162.0,
                "exit_reason": "profit_target",
                "profit_pct": 0.08,
                "rules_triggered": ["buy_dip_in_uptrend"],
            },
            {
                "entry_date": "2023-06-01T09:30:00",
                "entry_price": 170.0,
                "entry_reason": "Strong buy signal",
                "entry_confidence": 0.85,
                "exit_date": "2023-06-10T15:30:00",
                "exit_price": 165.0,
                "exit_reason": "stop_loss",
                "profit_pct": -0.0294,
                "rules_triggered": ["strong_buy_signal", "rsi_oversold"],
            },
        ],
    )


@pytest.fixture
def sample_trades():
    """A list of trade dicts suitable for metrics / trade_logger tests."""
    return [
        {
            "entry_date": "2023-01-10",
            "exit_date": "2023-01-20",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "entry_confidence": 0.70,
            "entry_reason": "RSI dip",
            "exit_reason": "profit_target",
            "profit_pct": 0.10,
            "rules_triggered": ["buy_dip_in_uptrend"],
        },
        {
            "entry_date": "2023-02-05",
            "exit_date": "2023-02-12",
            "entry_price": 120.0,
            "exit_price": 115.0,
            "entry_confidence": 0.65,
            "entry_reason": "MACD crossover",
            "exit_reason": "stop_loss",
            "profit_pct": -0.0417,
            "rules_triggered": ["macd_bullish_crossover"],
        },
        {
            "entry_date": "2023-03-01",
            "exit_date": "2023-03-15",
            "entry_price": 130.0,
            "exit_price": 143.0,
            "entry_confidence": 0.80,
            "entry_reason": "Strong buy",
            "exit_reason": "profit_target",
            "profit_pct": 0.10,
            "rules_triggered": ["strong_buy_signal", "buy_dip_in_uptrend"],
        },
        {
            "entry_date": "2023-04-01",
            "exit_date": "2023-04-05",
            "entry_price": 140.0,
            "exit_price": 136.0,
            "entry_confidence": 0.60,
            "entry_reason": "RSI oversold",
            "exit_reason": "stop_loss",
            "profit_pct": -0.0286,
            "rules_triggered": ["rsi_oversold"],
        },
        {
            "entry_date": "2023-05-10",
            "exit_date": "2023-05-25",
            "entry_price": 145.0,
            "exit_price": 155.0,
            "entry_confidence": 0.75,
            "entry_reason": "Trend continuation",
            "exit_reason": "profit_target",
            "profit_pct": 0.069,
            "rules_triggered": ["trend_continuation"],
        },
    ]
