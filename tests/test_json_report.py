"""Tests for backtesting.reporting.json_report — _serialize, _result_to_dict,
_calculate_aggregate, export_json."""

import json
import math
from datetime import date, datetime

import pytest

from backtesting.engine.backtrader_runner import BacktestResult
from backtesting.reporting.json_report import (
    _calculate_aggregate,
    _result_to_dict,
    _serialize,
    export_json,
)


# ---------------------------------------------------------------------------
# _serialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_datetime(self):
        dt = datetime(2023, 6, 15, 10, 30, 0)
        assert _serialize(dt) == "2023-06-15T10:30:00"

    def test_date(self):
        d = date(2023, 6, 15)
        assert _serialize(d) == "2023-06-15"

    def test_positive_infinity(self):
        assert _serialize(float("inf")) == "inf"

    def test_negative_infinity(self):
        assert _serialize(float("-inf")) == "-inf"

    def test_normal_float_raises(self):
        with pytest.raises(TypeError, match="not serializable"):
            _serialize(42.5)

    def test_string_raises(self):
        with pytest.raises(TypeError, match="not serializable"):
            _serialize("hello")

    def test_none_raises(self):
        with pytest.raises(TypeError, match="not serializable"):
            _serialize(None)

    def test_list_raises(self):
        with pytest.raises(TypeError, match="not serializable"):
            _serialize([1, 2, 3])


# ---------------------------------------------------------------------------
# _result_to_dict
# ---------------------------------------------------------------------------


class TestResultToDict:
    def test_structure(self, sample_result):
        d = _result_to_dict(sample_result)
        assert d["symbol"] == "AAPL"
        assert d["strategy"] == "buy_dip_in_uptrend"
        assert "period" in d
        assert "performance" in d
        assert "trades" in d
        assert "risk_metrics" in d
        assert "trade_history" in d

    def test_period_fields(self, sample_result):
        d = _result_to_dict(sample_result)
        assert d["period"]["start"] == date(2023, 1, 1)
        assert d["period"]["end"] == date(2023, 12, 31)

    def test_performance_fields(self, sample_result):
        d = _result_to_dict(sample_result)
        perf = d["performance"]
        assert perf["initial_cash"] == 100_000.0
        assert perf["final_value"] == 112_345.67
        assert perf["total_return"] == pytest.approx(0.1234567)
        assert perf["total_return_pct"] == "12.35%"

    def test_trades_fields(self, sample_result):
        d = _result_to_dict(sample_result)
        trades = d["trades"]
        assert trades["total"] == 25
        assert trades["winners"] == 15
        assert trades["losers"] == 10
        assert trades["win_rate"] == 0.60
        assert trades["avg_win"] == 1500.0
        assert trades["avg_loss"] == -800.0

    def test_risk_metrics_fields(self, sample_result):
        d = _result_to_dict(sample_result)
        risk = d["risk_metrics"]
        assert risk["profit_factor"] == 2.81
        assert risk["sharpe_ratio"] == 1.45
        assert risk["max_drawdown"] == 5432.10
        assert risk["max_drawdown_pct"] == 5.4

    def test_trade_history(self, sample_result):
        d = _result_to_dict(sample_result)
        assert len(d["trade_history"]) == 2
        assert d["trade_history"][0]["entry_price"] == 150.0

    def test_none_trades(self):
        r = BacktestResult(
            symbol="X", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, trades=None,
        )
        d = _result_to_dict(r)
        assert d["trade_history"] == []


# ---------------------------------------------------------------------------
# _calculate_aggregate
# ---------------------------------------------------------------------------


class TestCalculateAggregate:
    def test_empty_results(self):
        assert _calculate_aggregate({}) == {}

    def test_single_result(self, sample_result):
        results = {"AAPL": sample_result}
        agg = _calculate_aggregate(results)
        assert agg["total_symbols"] == 1
        assert agg["total_trades"] == 25
        assert agg["total_wins"] == 15
        assert agg["overall_win_rate"] == pytest.approx(0.60)
        assert agg["avg_return"] == pytest.approx(0.1234567)
        assert agg["best_symbol"] == "AAPL"
        assert agg["worst_symbol"] == "AAPL"

    def test_multiple_results(self, sample_result):
        r2 = BacktestResult(
            symbol="GOOG", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=95_000,
            total_return=-0.05, total_trades=10, winning_trades=3, losing_trades=7,
            win_rate=0.30,
        )
        results = {"AAPL": sample_result, "GOOG": r2}
        agg = _calculate_aggregate(results)
        assert agg["total_symbols"] == 2
        assert agg["total_trades"] == 35  # 25 + 10
        assert agg["total_wins"] == 18  # 15 + 3
        assert agg["overall_win_rate"] == pytest.approx(18 / 35)
        assert agg["best_symbol"] == "AAPL"
        assert agg["worst_symbol"] == "GOOG"

    def test_no_trades_win_rate_zero(self):
        r = BacktestResult(
            symbol="EMPTY", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=100_000,
            total_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0,
        )
        agg = _calculate_aggregate({"EMPTY": r})
        assert agg["overall_win_rate"] == 0


# ---------------------------------------------------------------------------
# export_json — single result
# ---------------------------------------------------------------------------


class TestExportJsonSingle:
    def test_returns_valid_json(self, sample_result):
        json_str = export_json(sample_result)
        data = json.loads(json_str)
        assert data["symbol"] == "AAPL"
        assert "performance" in data

    def test_dates_serialized(self, sample_result):
        json_str = export_json(sample_result)
        data = json.loads(json_str)
        # date objects get serialized via _serialize
        assert data["period"]["start"] == "2023-01-01"
        assert data["period"]["end"] == "2023-12-31"

    def test_infinity_serialized(self):
        """float('inf') in JSON → Python json.dumps outputs 'Infinity' (non-standard).
        json.loads reads it back as float inf."""
        r = BacktestResult(
            symbol="TEST", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=110_000,
            total_return=0.10, total_trades=5, winning_trades=5, losing_trades=0,
            win_rate=1.0, profit_factor=float("inf"),
        )
        json_str = export_json(r)
        # Python json.dumps outputs Infinity for float("inf"), json.loads reads it back
        data = json.loads(json_str)
        assert data["risk_metrics"]["profit_factor"] == float("inf")

    def test_write_to_file(self, sample_result, tmp_path):
        out_path = str(tmp_path / "report.json")
        json_str = export_json(sample_result, output_path=out_path)
        assert (tmp_path / "report.json").exists()
        file_content = (tmp_path / "report.json").read_text()
        assert file_content == json_str

    def test_no_file_when_path_none(self, sample_result, tmp_path):
        json_str = export_json(sample_result, output_path=None)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# export_json — multi-symbol
# ---------------------------------------------------------------------------


class TestExportJsonMulti:
    def test_multi_symbol_structure(self, sample_result):
        r2 = BacktestResult(
            symbol="GOOG", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31),
            strategy_name="test", initial_cash=100_000, final_value=105_000,
            total_return=0.05, total_trades=8, winning_trades=5, losing_trades=3,
            win_rate=0.625,
        )
        results = {"AAPL": sample_result, "GOOG": r2}
        json_str = export_json(results)
        data = json.loads(json_str)
        assert data["multi_symbol"] is True
        assert set(data["symbols"]) == {"AAPL", "GOOG"}
        assert "AAPL" in data["results"]
        assert "GOOG" in data["results"]
        assert "aggregate" in data
        assert data["aggregate"]["total_symbols"] == 2

    def test_multi_symbol_to_file(self, sample_result, tmp_path):
        results = {"AAPL": sample_result}
        out_path = str(tmp_path / "multi.json")
        export_json(results, output_path=out_path)
        assert (tmp_path / "multi.json").exists()
        data = json.loads((tmp_path / "multi.json").read_text())
        assert data["multi_symbol"] is True
