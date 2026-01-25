"""
Pandas-TA Indicator Bridge

Calculates technical indicators using pandas-ta to match the analytics-service.
"""

import logging
from typing import List

import pandas as pd
import pandas_ta as ta

from ..config import settings

logger = logging.getLogger(__name__)


def calculate_indicators(
    df: pd.DataFrame,
    rsi_period: int = None,
    sma_periods: List[int] = None,
    macd_fast: int = None,
    macd_slow: int = None,
    macd_signal: int = None,
    bb_period: int = None,
    atr_period: int = None,
) -> pd.DataFrame:
    """
    Calculate technical indicators on OHLCV data.

    Matches the indicators calculated by analytics-service.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        *_period: Override default periods from settings

    Returns:
        DataFrame with additional indicator columns
    """
    # Use defaults from settings if not provided
    rsi_period = rsi_period or settings.rsi_period
    sma_periods = sma_periods or settings.sma_periods
    macd_fast = macd_fast or settings.macd_fast
    macd_slow = macd_slow or settings.macd_slow
    macd_signal = macd_signal or settings.macd_signal
    bb_period = bb_period or settings.bb_period
    atr_period = atr_period or settings.atr_period

    result = df.copy()

    # RSI
    logger.debug(f"Calculating RSI({rsi_period})")
    result[f"RSI_{rsi_period}"] = ta.rsi(result["close"], length=rsi_period)

    # SMAs
    for period in sma_periods:
        logger.debug(f"Calculating SMA({period})")
        result[f"SMA_{period}"] = ta.sma(result["close"], length=period)

    # MACD
    logger.debug(f"Calculating MACD({macd_fast},{macd_slow},{macd_signal})")
    macd_result = ta.macd(
        result["close"],
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
    )
    if macd_result is not None and not macd_result.empty:
        result["MACD"] = macd_result.iloc[:, 0]  # MACD line
        result["MACD_HISTOGRAM"] = macd_result.iloc[:, 1]  # Histogram
        result["MACD_SIGNAL"] = macd_result.iloc[:, 2]  # Signal line

    # Bollinger Bands
    logger.debug(f"Calculating Bollinger Bands({bb_period})")
    bb_result = ta.bbands(result["close"], length=bb_period)
    if bb_result is not None and not bb_result.empty:
        result["BB_LOWER"] = bb_result.iloc[:, 0]
        result["BB_MID"] = bb_result.iloc[:, 1]
        result["BB_UPPER"] = bb_result.iloc[:, 2]
        result["BB_BANDWIDTH"] = bb_result.iloc[:, 3] if bb_result.shape[1] > 3 else None
        result["BB_PERCENT"] = bb_result.iloc[:, 4] if bb_result.shape[1] > 4 else None

    # ATR
    logger.debug(f"Calculating ATR({atr_period})")
    result[f"ATR_{atr_period}"] = ta.atr(
        result["high"],
        result["low"],
        result["close"],
        length=atr_period,
    )

    # Drop rows with NaN indicators (warm-up period)
    initial_rows = len(result)
    result = result.dropna(subset=[f"SMA_{max(sma_periods)}"])
    dropped = initial_rows - len(result)
    if dropped > 0:
        logger.debug(f"Dropped {dropped} rows during indicator warm-up")

    logger.info(f"Calculated indicators: {len(result)} bars with valid data")
    return result


def get_required_warmup_bars(sma_periods: List[int] = None) -> int:
    """Get number of bars needed for indicator warm-up."""
    periods = sma_periods or settings.sma_periods
    return max(periods) + 10  # Extra buffer
