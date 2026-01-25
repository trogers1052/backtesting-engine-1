"""
Technical Analysis Indicator Bridge

Calculates technical indicators using the 'ta' library to match the analytics-service.
"""

import logging
from typing import List

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

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
    rsi = RSIIndicator(close=result["close"], window=rsi_period)
    result[f"RSI_{rsi_period}"] = rsi.rsi()

    # SMAs
    for period in sma_periods:
        logger.debug(f"Calculating SMA({period})")
        sma = SMAIndicator(close=result["close"], window=period)
        result[f"SMA_{period}"] = sma.sma_indicator()

    # MACD
    logger.debug(f"Calculating MACD({macd_fast},{macd_slow},{macd_signal})")
    macd = MACD(
        close=result["close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    result["MACD"] = macd.macd()
    result["MACD_SIGNAL"] = macd.macd_signal()
    result["MACD_HISTOGRAM"] = macd.macd_diff()

    # Bollinger Bands
    logger.debug(f"Calculating Bollinger Bands({bb_period})")
    bb = BollingerBands(close=result["close"], window=bb_period, window_dev=2)
    result["BB_LOWER"] = bb.bollinger_lband()
    result["BB_MID"] = bb.bollinger_mavg()
    result["BB_UPPER"] = bb.bollinger_hband()
    result["BB_BANDWIDTH"] = bb.bollinger_wband()
    result["BB_PERCENT"] = bb.bollinger_pband()

    # ATR
    logger.debug(f"Calculating ATR({atr_period})")
    atr = AverageTrueRange(
        high=result["high"],
        low=result["low"],
        close=result["close"],
        window=atr_period,
    )
    result[f"ATR_{atr_period}"] = atr.average_true_range()

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
