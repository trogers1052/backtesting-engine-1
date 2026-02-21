"""
Technical Analysis Indicator Bridge

Calculates technical indicators using pandas-ta — the same library and
identical call signatures as the live analytics-service.  This ensures
that backtest results are computed with exactly the same algorithms,
not just the same names.
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

    Uses pandas-ta to exactly match the analytics-service implementation.

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
    macd_df = ta.macd(
        result["close"],
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
    )
    # pandas-ta names MACD columns dynamically: MACD_f_s_sig, MACDs_f_s_sig, MACDh_f_s_sig
    result["MACD"] = macd_df[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
    result["MACD_SIGNAL"] = macd_df[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
    result["MACD_HISTOGRAM"] = macd_df[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]

    # Bollinger Bands
    logger.debug(f"Calculating Bollinger Bands({bb_period})")
    bb_df = ta.bbands(result["close"], length=bb_period, std=2)
    # pandas-ta names BB columns dynamically (e.g., BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
    lower_col = [c for c in bb_df.columns if c.startswith("BBL_")][0]
    mid_col = [c for c in bb_df.columns if c.startswith("BBM_")][0]
    upper_col = [c for c in bb_df.columns if c.startswith("BBU_")][0]
    bw_col = [c for c in bb_df.columns if c.startswith("BBB_")][0]
    pct_col = [c for c in bb_df.columns if c.startswith("BBP_")][0]
    result["BB_LOWER"] = bb_df[lower_col]
    result["BB_MID"] = bb_df[mid_col]
    result["BB_UPPER"] = bb_df[upper_col]
    result["BB_BANDWIDTH"] = bb_df[bw_col]
    result["BB_PERCENT"] = bb_df[pct_col]

    # ATR
    logger.debug(f"Calculating ATR({atr_period})")
    result[f"ATR_{atr_period}"] = ta.atr(
        high=result["high"],
        low=result["low"],
        close=result["close"],
        length=atr_period,
    )

    # Volume SMA (for volume confirmation in enhanced rules)
    logger.debug("Calculating Volume SMA(20)")
    result["volume_sma_20"] = ta.sma(result["volume"].astype(float), length=20)

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
