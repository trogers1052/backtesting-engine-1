"""Tests for backtesting.config — Settings defaults, db_url, env overrides."""

import os

import pytest

from backtesting.config import Settings


class TestSettingsDefaults:
    def test_default_db_host(self):
        s = Settings()
        assert s.market_data_db_host == "localhost"

    def test_default_db_port(self):
        s = Settings()
        assert s.market_data_db_port == 5433

    def test_default_db_user(self):
        s = Settings()
        assert s.market_data_db_user == "ingestor"

    def test_default_db_name(self):
        s = Settings()
        assert s.market_data_db_name == "stock_db"

    def test_default_timeframe(self):
        s = Settings()
        assert s.default_timeframe == "daily"

    def test_default_exit_timeframe_none(self):
        s = Settings()
        assert s.default_exit_timeframe is None

    def test_default_start_date(self):
        s = Settings()
        assert s.default_start_date == "2021-01-01"

    def test_default_initial_cash(self):
        s = Settings()
        assert s.default_initial_cash == 100_000.0

    def test_default_commission(self):
        s = Settings()
        assert s.default_commission == 0.001

    def test_default_profit_target(self):
        s = Settings()
        assert s.default_profit_target == 0.07

    def test_default_stop_loss(self):
        s = Settings()
        assert s.default_stop_loss == 0.05

    def test_default_min_confidence(self):
        s = Settings()
        assert s.default_min_confidence == 0.6

    def test_default_position_size_pct(self):
        s = Settings()
        assert s.default_position_size_pct == 0.95

    def test_default_sizing_mode(self):
        s = Settings()
        assert s.default_sizing_mode == "percent"

    def test_default_risk_pct(self):
        s = Settings()
        assert s.default_risk_pct == 5.0

    def test_default_max_position_pct(self):
        s = Settings()
        assert s.default_max_position_pct == 20.0

    def test_default_stop_mode(self):
        s = Settings()
        assert s.default_stop_mode == "fixed"

    def test_default_atr_multiplier(self):
        s = Settings()
        assert s.default_atr_multiplier == 2.0

    def test_default_atr_stop_min_pct(self):
        s = Settings()
        assert s.default_atr_stop_min_pct == 3.0

    def test_default_atr_stop_max_pct(self):
        s = Settings()
        assert s.default_atr_stop_max_pct == 15.0

    def test_default_max_price_extension_pct(self):
        s = Settings()
        assert s.default_max_price_extension_pct == 15.0

    def test_default_cooldown_bars(self):
        s = Settings()
        assert s.default_cooldown_bars == 5

    def test_default_max_trend_spread_pct(self):
        s = Settings()
        assert s.default_max_trend_spread_pct == 20.0

    def test_default_max_loss_pct(self):
        s = Settings()
        assert s.default_max_loss_pct == 5.0

    def test_indicator_periods(self):
        s = Settings()
        assert s.rsi_period == 14
        assert s.sma_periods == [20, 50, 200]
        assert s.macd_fast == 12
        assert s.macd_slow == 26
        assert s.macd_signal == 9
        assert s.bb_period == 20
        assert s.atr_period == 14


class TestMarketDataDbUrl:
    def test_default_url(self):
        s = Settings()
        assert s.market_data_db_url == (
            "postgresql://ingestor:ingestor@localhost:5433/stock_db"
        )

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_DB_HOST", "pi.local")
        monkeypatch.setenv("MARKET_DATA_DB_PORT", "5432")
        monkeypatch.setenv("MARKET_DATA_DB_USER", "admin")
        monkeypatch.setenv("MARKET_DATA_DB_PASSWORD", "secret")
        monkeypatch.setenv("MARKET_DATA_DB_NAME", "trading_db")
        s = Settings()
        assert s.market_data_db_url == (
            "postgresql://admin:secret@pi.local:5432/trading_db"
        )


class TestSettingsEnvOverrides:
    def test_override_initial_cash(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_INITIAL_CASH", "50000")
        s = Settings()
        assert s.default_initial_cash == 50_000.0

    def test_override_profit_target(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_PROFIT_TARGET", "0.10")
        s = Settings()
        assert s.default_profit_target == 0.10

    def test_override_stop_loss(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_STOP_LOSS", "0.03")
        s = Settings()
        assert s.default_stop_loss == 0.03

    def test_override_timeframe(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TIMEFRAME", "1min")
        s = Settings()
        assert s.default_timeframe == "1min"

    def test_extra_fields_ignored(self, monkeypatch):
        monkeypatch.setenv("SOME_RANDOM_FIELD", "whatever")
        s = Settings()
        assert not hasattr(s, "some_random_field")
