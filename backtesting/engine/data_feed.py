"""
Custom Backtrader DataFeed with Indicators

Extends PandasData to include pre-calculated indicator lines.
"""

import backtrader as bt
import pandas as pd


class PandasDataWithIndicators(bt.feeds.PandasData):
    """
    Custom DataFeed that includes pre-calculated indicators as additional lines.

    This allows the strategy to access indicators directly from the data feed
    rather than calculating them in backtrader (ensures consistency with
    analytics-service calculations).
    """

    # Add indicator lines beyond standard OHLCV
    lines = (
        "rsi_14",
        "sma_20",
        "sma_50",
        "sma_200",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_lower",
        "bb_mid",
        "bb_upper",
        "atr_14",
    )

    # Map DataFrame columns to lines
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        # Indicator mappings
        ("rsi_14", "RSI_14"),
        ("sma_20", "SMA_20"),
        ("sma_50", "SMA_50"),
        ("sma_200", "SMA_200"),
        ("macd", "MACD"),
        ("macd_signal", "MACD_SIGNAL"),
        ("macd_histogram", "MACD_HISTOGRAM"),
        ("bb_lower", "BB_LOWER"),
        ("bb_mid", "BB_MID"),
        ("bb_upper", "BB_UPPER"),
        ("atr_14", "ATR_14"),
    )


def create_data_feed(
    df: pd.DataFrame,
    name: str = None,
) -> PandasDataWithIndicators:
    """
    Create a backtrader DataFeed from a pandas DataFrame with indicators.

    Args:
        df: DataFrame with OHLCV and indicator columns
        name: Optional name for the data feed

    Returns:
        PandasDataWithIndicators instance
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Create the data feed
    data = PandasDataWithIndicators(
        dataname=df,
        name=name,
        datetime=None,  # Use index
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,  # Not used
    )

    return data
