"""Tests for backtesting.analyzers.trade_logger — format_trades, format_trade_summary."""

import pytest

from backtesting.analyzers.trade_logger import format_trades, format_trade_summary


# ---------------------------------------------------------------------------
# format_trades
# ---------------------------------------------------------------------------


class TestFormatTradesEmpty:
    def test_empty_list(self):
        assert format_trades([]) == "No trades executed."

    def test_no_trades_string(self):
        result = format_trades([])
        assert "No trades" in result


class TestFormatTradesSingle:
    def test_single_trade_output(self):
        trades = [
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
            }
        ]
        result = format_trades(trades)
        assert "BUY @ $100.00" in result
        assert "SELL @ $110.00" in result
        assert "+10.0%" in result
        assert "RSI dip" in result
        assert "buy_dip_in_uptrend" in result

    def test_negative_profit(self):
        trades = [
            {
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-15",
                "entry_price": 100.0,
                "exit_price": 95.0,
                "entry_confidence": 0.60,
                "entry_reason": "Test",
                "exit_reason": "stop_loss",
                "profit_pct": -0.05,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades)
        assert "-5.0%" in result

    def test_zero_profit(self):
        trades = [
            {
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-15",
                "entry_price": 100.0,
                "exit_price": 100.0,
                "entry_confidence": 0.50,
                "entry_reason": "Test",
                "exit_reason": "manual",
                "profit_pct": 0,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades)
        assert "N/A" in result  # 0 is falsy → N/A branch

    def test_none_profit(self):
        trades = [
            {
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-15",
                "entry_price": 100.0,
                "exit_price": 100.0,
                "entry_confidence": 0.50,
                "entry_reason": "Test",
                "exit_reason": "manual",
                "profit_pct": None,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades)
        assert "N/A" in result


class TestFormatTradesLimiting:
    def test_limit_shows_last_n(self):
        trades = [
            {
                "entry_date": f"2023-01-{10+i:02d}",
                "exit_date": f"2023-01-{15+i:02d}",
                "entry_price": 100.0 + i,
                "exit_price": 105.0 + i,
                "entry_confidence": 0.65,
                "entry_reason": f"Trade {i}",
                "exit_reason": "profit_target",
                "profit_pct": 0.05,
                "rules_triggered": [],
            }
            for i in range(15)
        ]
        result = format_trades(trades, limit=5)
        assert "Showing last 5 of 15 trades" in result
        # Should contain trade 10-14 (last 5), not trade 0
        assert "Trade 10" in result
        assert "Trade 0" not in result

    def test_limit_zero_shows_all(self):
        trades = [
            {
                "entry_date": f"2023-01-{10+i:02d}",
                "exit_date": f"2023-01-{15+i:02d}",
                "entry_price": 100.0,
                "exit_price": 105.0,
                "entry_confidence": 0.65,
                "entry_reason": f"Trade {i}",
                "exit_reason": "profit_target",
                "profit_pct": 0.05,
                "rules_triggered": [],
            }
            for i in range(15)
        ]
        result = format_trades(trades, limit=0)
        assert "Showing last" not in result
        # All 15 trades present
        for i in range(15):
            assert f"Trade {i}" in result

    def test_limit_larger_than_list(self):
        trades = [
            {
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-20",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_confidence": 0.70,
                "entry_reason": "Test",
                "exit_reason": "profit_target",
                "profit_pct": 0.10,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades, limit=10)
        assert "Showing last" not in result


class TestFormatTradesDateFormatting:
    def test_iso_datetime_truncated(self):
        trades = [
            {
                "entry_date": "2023-01-10T09:30:00",
                "exit_date": "2023-01-20T15:30:00",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_confidence": 0.70,
                "entry_reason": "Test",
                "exit_reason": "profit_target",
                "profit_pct": 0.10,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades)
        # T should be split out — only date portion shown
        assert "2023-01-10" in result
        assert "T09:30" not in result

    def test_missing_fields_default_to_na(self):
        trades = [
            {
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_confidence": 0.70,
                "profit_pct": 0.05,
            }
        ]
        result = format_trades(trades)
        assert "N/A" in result  # missing dates, reason

    def test_no_rules_triggered(self):
        trades = [
            {
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-20",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "entry_confidence": 0.70,
                "entry_reason": "Manual",
                "exit_reason": "profit_target",
                "profit_pct": 0.10,
                "rules_triggered": [],
            }
        ]
        result = format_trades(trades)
        assert "Rules:" not in result


# ---------------------------------------------------------------------------
# format_trade_summary
# ---------------------------------------------------------------------------


class TestFormatTradeSummaryEmpty:
    def test_empty_list(self):
        assert format_trade_summary([]) == {}

    def test_no_profit_pct(self):
        trades = [{"symbol": "AAPL"}]
        assert format_trade_summary(trades) == {}


class TestFormatTradeSummaryStats:
    def test_basic_stats(self, sample_trades):
        result = format_trade_summary(sample_trades)
        assert result["total_trades"] == 5
        assert result["winners"] == 3
        assert result["losers"] == 2
        assert abs(result["win_rate"] - 0.6) < 1e-9

    def test_avg_win_loss(self, sample_trades):
        result = format_trade_summary(sample_trades)
        assert result["avg_win"] > 0
        assert result["avg_loss"] < 0

    def test_best_worst_trade(self, sample_trades):
        result = format_trade_summary(sample_trades)
        assert result["best_trade"] == 0.10
        assert result["worst_trade"] == pytest.approx(-0.0417)


class TestFormatTradeSummaryExitReasons:
    def test_exit_reason_counts(self, sample_trades):
        result = format_trade_summary(sample_trades)
        assert result["exit_reasons"]["profit_target"] == 3
        assert result["exit_reasons"]["stop_loss"] == 2

    def test_missing_exit_reason_defaults_to_unknown(self):
        trades = [{"profit_pct": 0.05}]
        result = format_trade_summary(trades)
        assert result["exit_reasons"]["Unknown"] == 1


class TestFormatTradeSummaryRuleContributions:
    def test_rule_counts(self, sample_trades):
        result = format_trade_summary(sample_trades)
        rules = result["rule_contributions"]
        assert rules["buy_dip_in_uptrend"] == 2  # Appears in trades 0 and 2
        assert rules["strong_buy_signal"] == 1
        assert rules["macd_bullish_crossover"] == 1
        assert rules["rsi_oversold"] == 1
        assert rules["trend_continuation"] == 1

    def test_no_rules_gives_empty_dict(self):
        trades = [{"profit_pct": 0.05, "rules_triggered": []}]
        result = format_trade_summary(trades)
        assert result["rule_contributions"] == {}

    def test_missing_rules_key(self):
        trades = [{"profit_pct": 0.05}]
        result = format_trade_summary(trades)
        assert result["rule_contributions"] == {}
