"""
Backtesting Service Configuration

Uses pydantic-settings for environment variable management.
"""

from datetime import date
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backtesting service configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Market Data Database (TimescaleDB)
    market_data_db_host: str = "localhost"
    market_data_db_port: int = 5433
    market_data_db_user: str = "ingestor"
    market_data_db_password: str = "ingestor"
    market_data_db_name: str = "stock_db"

    # Default backtest parameters
    default_timeframe: str = "daily"  # 1min, 5min, 15min, 30min, 1hour, 4hour, daily
    default_start_date: str = "2021-01-01"
    default_end_date: Optional[str] = None  # None = today

    # Strategy defaults
    default_initial_cash: float = 100000.0
    default_commission: float = 0.001  # 0.1% commission
    default_profit_target: float = 0.07  # 7%
    default_stop_loss: float = 0.05  # 5%
    default_min_confidence: float = 0.6

    # Position sizing
    default_position_size_pct: float = 0.95  # Use 95% of available cash
    default_sizing_mode: str = "percent"  # "percent" or "risk_based"
    default_risk_pct: float = 5.0  # % of portfolio risked per trade
    default_max_position_pct: float = 20.0  # Max % of portfolio in one position
    default_stop_mode: str = "fixed"  # "fixed" or "atr"
    default_atr_multiplier: float = 2.0  # ATR multiplier for stop calculation
    default_atr_stop_min_pct: float = 3.0  # Minimum stop distance %
    default_atr_stop_max_pct: float = 15.0  # Maximum stop distance %
    default_max_price_extension_pct: float = 15.0  # Skip buy if price > X% above SMA_20
    default_cooldown_bars: int = 5  # Wait N bars after exit before re-entering
    default_max_trend_spread_pct: float = 20.0  # Skip buy if SMA_20/SMA_50 spread > X%
    default_max_loss_pct: float = 5.0  # Force exit if trade down > X% (primary downside protection)

    # Indicator periods (match analytics-service)
    rsi_period: int = 14
    sma_periods: List[int] = [20, 50, 200]
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    atr_period: int = 14

    @property
    def market_data_db_url(self) -> str:
        """PostgreSQL connection URL for market data."""
        return (
            f"postgresql://{self.market_data_db_user}:{self.market_data_db_password}"
            f"@{self.market_data_db_host}:{self.market_data_db_port}/{self.market_data_db_name}"
        )


# Global settings instance
settings = Settings()
