"""
Vectorized backtesting engine for fast parameter sweeps.

Instead of iterating 300K bars per config through backtrader's Python loop,
this engine:
  1. Pre-computes entry signals for a rule combo (vectorized, once)
  2. Simulates trades with numpy — loops over trades (~50), not bars (~300K)
  3. Returns metrics compatible with BacktestResult

Used for Steps 1-3 (discovery). Step 4 (validation) still uses backtrader
for full-fidelity walk-backward, bootstrap, monte carlo analysis.

~100-600x faster than backtrader for parameter sweeps.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """Lightweight result matching BacktestResult interface."""
    symbol: str
    start_date: date
    end_date: date
    strategy_name: str
    initial_cash: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    trades: List[Dict] = None


# ── Rule signal computation (vectorized) ──────────────────────────────

def _compute_rule_confidence(
    rule_name: str, df: pd.DataFrame, symbol: str = ""
) -> pd.Series:
    """Compute per-bar confidence for a single rule.

    Returns a Series of confidence values (0.0 where rule doesn't fire).
    Uses the daily indicator DataFrame.
    """
    n = len(df)
    conf = pd.Series(0.0, index=df.index)

    rsi = df.get("RSI_14")
    sma20 = df.get("SMA_20")
    sma50 = df.get("SMA_50")
    sma200 = df.get("SMA_200")
    macd = df.get("MACD")
    macd_sig = df.get("MACD_SIGNAL")
    macd_hist = df.get("MACD_HISTOGRAM")
    close = df["close"]
    volume = df.get("volume")
    vol_sma = df.get("volume_sma_20")
    atr = df.get("ATR_14")
    bb_lower = df.get("BB_LOWER")
    bb_upper = df.get("BB_UPPER")
    bb_pct = df.get("BB_PERCENT")
    adx = df.get("ADX_14")
    ema9 = df.get("EMA_9")
    ema21 = df.get("EMA_21")
    stoch_k = df.get("STOCH_K")
    stoch_d = df.get("STOCH_D")

    # Helper: safe division
    def safe_spread(a, b):
        """(a - b) / b * 100, safe for zero/NaN."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (a - b) / b * 100
        return result.fillna(0)

    # Helper: volume ratio (reused across rules)
    def vol_ratio_series():
        if vol_sma is not None and volume is not None:
            vr = volume / vol_sma.replace(0, np.nan)
            return vr.fillna(0)
        return pd.Series(1.0, index=df.index)

    vol_r = vol_ratio_series()

    # Helper: distance from a reference price as %
    def dist_pct(price, ref):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (price - ref) / ref * 100
        return result.fillna(0) if isinstance(result, pd.Series) else result

    # ── RSI Rules ──
    if rule_name == "rsi_oversold":
        # RSIOversoldRule: RSI < 30, confidence 0.5 at 30, 0.9 at 20
        mask = rsi < 30
        conf[mask] = 0.5 + (30 - rsi[mask]) / 25
        conf = conf.clip(0, 0.9)

    elif rule_name == "rsi_overbought":
        # RSIOverboughtRule: SELL if RSI >= 80 (-0.85), WATCH if 70-80
        mask_sell = rsi >= 80
        mask_watch = (rsi > 70) & (rsi < 80)
        conf[mask_sell] = -0.85
        conf[mask_watch] = -(0.4 + (rsi[mask_watch] - 70) / 33.3)
        conf = conf.clip(-0.85, 0)

    elif rule_name == "rsi_approaching_oversold":
        # RSIApproachingOversoldRule: WATCH, RSI 30-40, flat 0.40
        mask = (rsi >= 30) & (rsi <= 40)
        conf[mask] = 0.4

    # ── MACD Rules ──
    elif rule_name == "macd_bullish_crossover":
        # MACDBullishCrossoverRule: MACD > signal
        # Fresh (0 < hist < 0.1): 0.65 + |hist|*3 (max 0.85)
        # Old (hist >= 0.1): 0.50
        mask = macd > macd_sig
        hist_fresh = (macd_hist > 0) & (macd_hist < 0.1)
        conf[mask & hist_fresh] = (0.65 + macd_hist[mask & hist_fresh].abs() * 3).clip(0, 0.85)
        conf[mask & ~hist_fresh] = 0.50

    elif rule_name == "macd_bearish_crossover":
        # MACDBearishCrossoverRule: MACD < signal, WATCH signal
        # Fresh (-0.1 < hist < 0): -0.60 - |hist|*3 (max -0.80)
        # Old (hist <= -0.1): -0.50
        mask = macd < macd_sig
        hist_fresh = (macd_hist < 0) & (macd_hist > -0.1)
        conf[mask & hist_fresh] = -(0.60 + macd_hist[mask & hist_fresh].abs() * 3).clip(0, 0.80)
        conf[mask & ~hist_fresh] = -0.50

    elif rule_name == "macd_momentum":
        # MACDMomentumRule: |hist| > 0.05
        # Positive hist: BUY, min(0.40 + hist*2, 0.70)
        # Negative hist: WATCH (negative confidence)
        mask_buy = macd_hist > 0.05
        mask_sell = macd_hist < -0.05
        conf[mask_buy] = np.minimum(0.40 + macd_hist[mask_buy] * 2, 0.70)
        conf[mask_sell] = -np.minimum(0.40 + macd_hist[mask_sell].abs() * 2, 0.70)

    # ── Trend Rules ──
    elif rule_name == "weekly_uptrend":
        # WeeklyUptrendRule: WATCH signal, SMA_20 > SMA_50
        mask = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread >= 2)] = 0.85
        conf[mask & (spread >= 1) & (spread < 2)] = 0.70
        conf[mask & (spread < 1)] = 0.55

    elif rule_name == "monthly_uptrend":
        # MonthlyUptrendRule: WATCH signal, SMA_50 > SMA_200
        mask = sma50 > sma200
        spread = safe_spread(sma50, sma200)
        conf[mask & (spread >= 5)] = 0.85
        conf[mask & (spread >= 2) & (spread < 5)] = 0.70
        conf[mask & (spread < 2)] = 0.55

    elif rule_name == "trend_alignment":
        # FullTrendAlignmentRule: SMA_20 > SMA_50 > SMA_200
        # Confidence: 0.6 + total_spread/30, capped 0.95
        # Volume gate: < 0.5x = no signal, < 0.8x = -0.10 penalty
        aligned = (sma20 > sma50) & (sma50 > sma200)
        spread_20_50 = safe_spread(sma20, sma50)
        spread_50_200 = safe_spread(sma50, sma200)
        total_spread = spread_20_50 + spread_50_200
        conf[aligned] = (0.6 + total_spread[aligned] / 30).clip(0.5, 0.95)
        conf[aligned & (vol_r < 0.5)] = 0.0
        conf[aligned & (vol_r >= 0.5) & (vol_r < 0.8)] -= 0.10
        conf = conf.clip(0, 0.95)

    elif rule_name == "golden_cross":
        # GoldenCrossRule: SMA_50 > SMA_200
        # Fresh (spread < 1%): 0.75, Old (spread >= 1%): 0.50
        mask = (sma50 > sma200)
        spread = safe_spread(sma50, sma200)
        conf[mask & (spread < 1.0)] = 0.75
        conf[mask & (spread >= 1.0)] = 0.50

    elif rule_name == "death_cross":
        # DeathCrossRule: SELL, SMA_50 < SMA_200
        # Fresh (spread < 1%): -0.75, Old: -0.60
        mask = (sma50 < sma200)
        spread = safe_spread(sma200, sma50)
        conf[mask & (spread < 1.0)] = -0.75
        conf[mask & (spread >= 1.0)] = -0.60

    elif rule_name == "trend_break_warning":
        # TrendBreakWarningRule: SELL, SMA_20 < SMA_50
        # Fresh (spread < 0.5%): -0.70, Old: -0.60
        mask = (sma20 < sma50)
        spread = safe_spread(sma50, sma20)
        conf[mask & (spread < 0.5)] = -0.70
        conf[mask & (spread >= 0.5)] = -0.60

    # ── Composite Rules ──
    elif rule_name == "buy_dip_in_uptrend":
        # BuyDipInUptrendRule: SMA_20 > SMA_50, RSI < 40
        uptrend = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        dip = rsi < 40
        mask = uptrend & dip
        conf[mask & (rsi < 30)] = 0.85
        conf[mask & (rsi >= 30) & (rsi < 35)] = 0.70
        conf[mask & (rsi >= 35) & (rsi < 40)] = 0.55
        conf[mask & (spread >= 2)] += 0.10
        conf[mask & (spread >= 1) & (spread < 2)] += 0.05
        conf = conf.clip(0, 0.95)

    elif rule_name == "strong_buy_signal":
        # StrongBuySignalRule: full alignment + RSI < 35
        full_align = (sma20 > sma50) & (sma50 > sma200)
        dip = rsi < 35
        mask = full_align & dip
        conf[mask & (rsi < 25)] = 0.90
        conf[mask & (rsi >= 25) & (rsi < 30)] = 0.80
        conf[mask & (rsi >= 30) & (rsi < 35)] = 0.70
        spread_20_50 = safe_spread(sma20, sma50)
        spread_50_200 = safe_spread(sma50, sma200)
        total_spread = spread_20_50 + spread_50_200
        conf[mask] += (total_spread[mask] / 50).clip(0, 0.10)
        conf = conf.clip(0, 0.98)

    elif rule_name == "rsi_macd_confluence":
        # RSIAndMACDConfluenceRule: RSI < 35 AND MACD > signal
        mask = (rsi < 35) & (macd > macd_sig)
        conf[mask] = 0.70
        conf[mask & (rsi < 30)] += 0.15
        conf[mask & (rsi >= 30) & (rsi < 33)] += 0.10
        conf[mask & (macd_hist > 0.05)] += 0.05
        conf = conf.clip(0, 0.95)

    elif rule_name == "dip_recovery":
        # TrendDipRecoveryRule: uptrend + RSI 30-45
        uptrend = sma20 > sma50
        recovery = (rsi >= 30) & (rsi <= 45)
        mask = uptrend & recovery
        conf[mask] = (0.55 + (45 - rsi[mask]) / 30).clip(0, 0.75)

    # ── Enhanced Rules ──
    elif rule_name == "enhanced_buy_dip":
        # EnhancedBuyDipRule: uptrend + spread >= 1.5% + RSI < 35 + close > SMA_200 + vol >= 0.8x
        uptrend = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        above_200 = close > sma200
        dip = rsi < 35
        mask = uptrend & (spread >= 1.5) & dip & above_200 & (vol_r >= 0.8)
        rsi_score = pd.Series(0.0, index=df.index)
        rsi_score[rsi < 30] = 0.40
        rsi_score[(rsi >= 30) & (rsi < 33)] = 0.30
        rsi_score[(rsi >= 33) & (rsi < 35)] = 0.20
        trend_score = pd.Series(0.0, index=df.index)
        trend_score[spread >= 3] = 0.25
        trend_score[(spread >= 2) & (spread < 3)] = 0.20
        trend_score[(spread >= 1.5) & (spread < 2)] = 0.15
        align_bonus = pd.Series(0.0, index=df.index)
        align_bonus[sma50 > sma200] = 0.15
        vol_bonus = pd.Series(0.0, index=df.index)
        vol_bonus[vol_r >= 1.5] = 0.10
        vol_bonus[(vol_r >= 1.2) & (vol_r < 1.5)] = 0.07
        vol_bonus[(vol_r >= 1.0) & (vol_r < 1.2)] = 0.05
        total = rsi_score + trend_score + align_bonus + vol_bonus
        conf[mask] = total[mask].clip(0.50, 0.95)

    elif rule_name == "momentum_reversal":
        # MomentumReversalRule: golden cross + vol >= 0.5x + RSI 30-40 + MACD > signal
        # Additional: weak reversal gate (hist <= 0.05 AND vol < 1.0)
        golden = sma50 > sma200
        vol_ok = vol_r >= 0.5
        rsi_recovery = (rsi >= 30) & (rsi <= 40)
        macd_bull = macd > macd_sig
        weak_reversal = (macd_hist <= 0.05) & (vol_r < 1.0)
        mask = golden & vol_ok & rsi_recovery & macd_bull & ~weak_reversal
        conf[mask] = 0.55
        conf[mask & (sma20 > sma50)] += 0.15
        conf[mask & (macd_hist > 0.10)] += 0.10
        conf[mask & (macd_hist > 0.05) & (macd_hist <= 0.10)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi <= 40)] += 0.05
        conf[mask & (vol_r >= 1.2)] += 0.05
        conf = conf.clip(0, 0.90)

    elif rule_name == "trend_continuation":
        # TrendContinuationRule: full alignment + within ±2% of SMA_20 + RSI 35-60 + vol >= 0.5x
        full_align = (sma20 > sma50) & (sma50 > sma200)
        at_support = (dist_pct(close, sma20).abs()) <= 2.0
        rsi_mod = (rsi >= 35) & (rsi <= 60)
        vol_ok = vol_r >= 0.5
        mask = full_align & at_support & rsi_mod & vol_ok
        conf[mask] = 0.60
        spread20_50 = safe_spread(sma20, sma50)
        spread50_200 = safe_spread(sma50, sma200)
        conf[mask & (spread20_50 > 3)] += 0.10
        conf[mask & (spread50_200 > 5)] += 0.10
        dist_from_sma = dist_pct(close, sma20).abs()
        conf[mask & (dist_from_sma < 0.5)] += 0.05
        conf[mask & (vol_r >= 1.0)] += 0.05
        conf[mask & (vol_r < 0.8)] -= 0.05
        conf = conf.clip(0, 0.85)

    # ── Mining Rules ──
    elif rule_name == "commodity_breakout":
        # CommodityBreakoutRule: uptrend + spread >= 1% + breakout >= 2% above SMA_20
        # + RSI 45-75 + vol >= 0.5x
        uptrend = (sma20 > sma50)
        trend_strength = safe_spread(sma20, sma50)
        breakout_pct = safe_spread(close, sma20)
        mask = uptrend & (trend_strength >= 1.0) & (breakout_pct >= 2.0) & (rsi >= 45) & (rsi <= 75) & (vol_r >= 0.5)
        conf[mask] = 0.55
        conf[mask & (breakout_pct > 4.0)] += 0.15
        conf[mask & (breakout_pct > 3.0) & (breakout_pct <= 4.0)] += 0.10
        conf[mask & (breakout_pct > 2.0) & (breakout_pct <= 3.0)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.10
        conf[mask & (vol_r > 1.2) & (vol_r <= 1.5)] += 0.05
        conf[mask & (rsi >= 55) & (rsi <= 65)] += 0.05
        conf = conf.clip(0, 0.85)

    elif rule_name == "miner_metal_ratio":
        # Simplified proxy: RSI < 35 + golden cross
        mask = (rsi < 35) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "dollar_weakness":
        # Simplified proxy: MACD bullish + uptrend
        mask = (macd > macd_sig) & (sma20 > sma50)
        conf[mask] = 0.55

    elif rule_name == "seasonality":
        # Mining SeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Gold/silver: strong=[1,2,8,9,11,12], weak=[5,6]
        month = df.index.month
        strong_months = month.isin([1, 2, 8, 9, 11, 12])
        weak = month.isin([5, 6])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong_months] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "volume_breakout":
        # VolumeBreakoutRule: vol >= 1.5x + close > SMA_20
        mask = (vol_r >= 1.5) & (close > sma20)
        conf[mask] = 0.60
        conf[mask & (vol_r >= 2.0)] = 0.70

    # ── Energy Rules ──
    elif rule_name == "energy_momentum":
        # EnergyMomentumRule: golden cross + close > SMA_50 + ADX >= 25 + MACD_HIST > 0
        # + RSI 45-75 + vol >= 0.8x
        mask = ((sma50 > sma200) & (close > sma50) & (adx >= 25) &
                (macd_hist > 0) & (rsi >= 45) & (rsi <= 75) & (vol_r >= 0.8))
        conf[mask] = 0.55
        conf[mask & (adx > 35)] += 0.10
        conf[mask & (adx > 30) & (adx <= 35)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.10
        conf[mask & (vol_r > 1.2) & (vol_r <= 1.5)] += 0.05
        conf[mask & (rsi >= 50) & (rsi <= 65)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "energy_mean_reversion":
        # EnergyMeanReversionRule: close >= SMA_200 - 3% + RSI >= 15
        # + need >= 2 of: RSI<30, BB%<0.10, Stoch_K<20 & Stoch_D<20
        # + ADX<30 or not(close<SMA_50 & MACD_HIST<0) + vol >= 0.5x
        above_support = close >= (sma200 * 0.97) if sma200 is not None else True
        rsi_ok = (rsi >= 15) if rsi is not None else True
        os_rsi = (rsi < 30) if rsi is not None else False
        os_bb = (bb_pct < 0.10) if bb_pct is not None else False
        os_stoch = ((stoch_k < 20) & (stoch_d < 20)) if stoch_k is not None and stoch_d is not None else False
        oversold_count = os_rsi.astype(int) + os_bb.astype(int) + os_stoch.astype(int)
        not_freefall = ~((adx > 30) & (close < sma50) & (macd_hist < 0)) if adx is not None else True
        mask = above_support & rsi_ok & (oversold_count >= 2) & not_freefall & (vol_r >= 0.5)
        conf[mask] = 0.55
        conf[mask & (oversold_count >= 3)] += 0.10
        conf[mask & (oversold_count == 2)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.10
        conf[mask & (vol_r > 1.2) & (vol_r <= 1.5)] += 0.05
        conf[mask & (close > sma200)] += 0.05
        conf[mask & (rsi < 22)] += 0.05
        conf[mask & (rsi >= 22) & (rsi < 25)] += 0.03
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "energy_seasonality":
        # EnergySeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED (no new entries).
        # Strong months vary by subsector; using integrated/upstream defaults.
        month = df.index.month
        strong = month.isin([1, 2, 5, 6, 7, 8, 9, 11, 12])
        weak = month.isin([3, 4, 10])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "midstream_yield_reversion":
        # MidstreamYieldReversionRule: discount from SMA_200 >= 5%
        # + (RSI < 35 or BB% < 0.15) + not freefall + near SMA_20 + vol >= 0.5x
        discount = safe_spread(sma200, close)  # how far below SMA_200
        near_sma20 = (dist_pct(close, sma20).abs()) <= 3.0
        os_check = (rsi < 35) | ((bb_pct < 0.15) if bb_pct is not None else False)
        not_freefall = ~((adx > 30) & (close < sma50)) if adx is not None else True
        mask = (discount >= 5.0) & os_check & not_freefall & near_sma20 & (vol_r >= 0.5)
        conf[mask] = 0.55
        conf[mask & (discount > 10)] += 0.15
        conf[mask & (discount > 8) & (discount <= 10)] += 0.10
        conf[mask & (discount > 6) & (discount <= 8)] += 0.05
        conf[mask & (rsi < 25)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Defense Rules ──
    elif rule_name == "defense_momentum":
        # DefenseMomentumRule: golden cross + close > SMA_50 + close > SMA_200
        # + uptrend + ADX >= 20 + MACD_HIST > 0 + RSI 40-72 + vol >= 0.7x
        mask = ((sma50 > sma200) & (close > sma50) & (close > sma200) &
                (sma20 > sma50) & (adx >= 20) &
                (macd_hist > 0) & (rsi >= 40) & (rsi <= 72) & (vol_r >= 0.7))
        conf[mask] = 0.55
        conf[mask & (adx > 30)] += 0.10
        conf[mask & (adx > 25) & (adx <= 30)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.10
        conf[mask & (vol_r > 1.2) & (vol_r <= 1.5)] += 0.05
        conf[mask & (rsi >= 45) & (rsi <= 60)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "defense_mean_reversion":
        # DefenseMeanReversionRule: close > SMA_200 + ADX < 25 + BB% < 0.20
        # + RSI 20-42 + vol >= 0.5x
        bb_ok = (bb_pct < 0.20) if bb_pct is not None else True
        mask = ((close > sma200) & (adx < 25) & bb_ok &
                (rsi >= 20) & (rsi <= 42) & (vol_r >= 0.5))
        conf[mask] = 0.58
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.07
        conf[mask & (bb_pct >= 0.05) & (bb_pct < 0.10)] += 0.04
        conf[mask & (rsi < 30)] += 0.10
        conf[mask & (rsi >= 30) & (rsi < 35)] += 0.05
        conf[mask & (adx < 15)] += 0.05
        conf[mask & (sma50 > sma200)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "defense_budget_cycle":
        # DefenseBudgetCycleRule: golden cross + close > SMA_200 + budget months
        # + RSI 20-72 + vol >= 0.5x + uptrend
        month = df.index.month
        budget_months = month.isin([9, 10, 11, 12, 1, 2, 3])
        mask = (budget_months & (sma50 > sma200) & (close > sma200) &
                (rsi >= 20) & (rsi <= 72) & (vol_r >= 0.5) & (sma20 > sma50))
        conf[mask] = 0.55
        conf[mask & month.isin([10, 11, 12, 1, 2])] += 0.10
        conf[mask & month.isin([7, 8, 9])] += 0.05
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "defense_counter_cyclical":
        # DefenseCounterCyclicalRule: close > SMA_50 + uptrend + RSI 40-70 + vol >= 0.5x
        mask = ((close > sma50) & (sma20 > sma50) & (rsi >= 40) & (rsi <= 70) & (vol_r >= 0.5))
        conf[mask] = 0.55
        above_200_pct = safe_spread(close, sma200)
        conf[mask & (above_200_pct > 10)] += 0.10
        conf[mask & (above_200_pct > 5) & (above_200_pct <= 10)] += 0.05
        conf[mask & (rsi >= 45) & (rsi <= 60)] += 0.05
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf[mask & (adx > 20)] += 0.05
        conf[mask & (vol_r > 1.3)] += 0.07
        conf[mask & (vol_r > 1.1) & (vol_r <= 1.3)] += 0.03
        conf = conf.clip(0.40, 0.85)

    # ── Industrial Rules ──
    elif rule_name == "industrial_mean_reversion":
        # IndustrialMeanReversionRule: close > SMA_200 + golden cross + ADX < 22
        # + BB% < 0.15 + RSI 22-42 + vol >= 0.5x
        bb_ok = (bb_pct < 0.15) if bb_pct is not None else True
        mask = ((close > sma200) & (sma50 > sma200) & (adx < 22) & bb_ok &
                (rsi >= 22) & (rsi <= 42) & (vol_r >= 0.5))
        conf[mask] = 0.58
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.07
        conf[mask & (bb_pct >= 0.05) & (bb_pct < 0.10)] += 0.03
        conf[mask & (rsi < 30)] += 0.10
        conf[mask & (rsi >= 30) & (rsi < 35)] += 0.05
        conf[mask & (adx < 15)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "industrial_pullback":
        # IndustrialPullbackRule: golden cross + within -4% to +2.5% of SMA_50
        # + RSI <= 45 + MACD_HIST >= -0.5
        dist_50 = dist_pct(close, sma50)
        macd_ok = (macd_hist >= -0.5) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (dist_50 >= -4.0) & (dist_50 <= 2.5) &
                (rsi <= 45) & macd_ok)
        conf[mask] = 0.55
        conf[mask & (dist_50.abs() < 1)] += 0.10
        conf[mask & (dist_50.abs() >= 1) & (dist_50.abs() < 2)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi < 40)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (adx < 22)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "industrial_seasonality":
        # IndustrialSeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Strong=[10,11,12,1], Weak=[5,6]
        month = df.index.month
        strong = month.isin([10, 11, 12, 1])
        weak = month.isin([5, 6])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Tech Rules ──
    elif rule_name == "tech_ema_pullback":
        # TechEMAPullbackRule: EMA_21 > SMA_50 > SMA_200 + close > SMA_50
        # + within -3% to +1.5% of EMA_21 + RSI 38-65 + vol >= 0.7x
        ema_align = (ema21 > sma50) & (sma50 > sma200) if ema21 is not None else False
        above_50 = close > sma50
        dist_ema = dist_pct(close, ema21) if ema21 is not None else pd.Series(99, index=df.index)
        mask = (ema_align & above_50 & (dist_ema >= -3.0) & (dist_ema <= 1.5) &
                (rsi >= 38) & (rsi <= 65) & (vol_r >= 0.7))
        conf[mask] = 0.55
        conf[mask & (dist_ema.abs() < 0.5)] += 0.15
        conf[mask & (dist_ema.abs() >= 0.5) & (dist_ema.abs() < 1.0)] += 0.10
        conf[mask & (dist_ema.abs() >= 1.0) & (dist_ema.abs() < 1.5)] += 0.05
        conf[mask & (rsi >= 45) & (rsi <= 55)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0, 0.85)

    elif rule_name == "tech_mean_reversion":
        # TechMeanReversionRule: need >= 2 of: RSI<30, BB%<0.10, Stoch_K<20
        # + near SMA_50 or SMA_200
        os_rsi = (rsi < 30) if rsi is not None else False
        os_bb = (bb_pct < 0.10) if bb_pct is not None else False
        os_stoch = (stoch_k < 20) if stoch_k is not None else False
        oversold_count = os_rsi.astype(int) + os_bb.astype(int) + os_stoch.astype(int)
        near_sma50 = (dist_pct(close, sma50).abs()) <= 3.0
        near_sma200 = (dist_pct(close, sma200).abs()) <= 5.0 if sma200 is not None else False
        near_support = near_sma50 | near_sma200
        mask = (oversold_count >= 2) & near_support & (rsi >= 10)
        conf[mask] = 0.55
        conf[mask & (oversold_count >= 3)] += 0.15
        conf[mask & (oversold_count == 2)] += 0.10
        conf[mask & near_sma200] += 0.10
        conf = conf.clip(0, 0.85)

    elif rule_name == "tech_seasonality":
        # TechSeasonalityRule: uptrend + RSI 30-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Using semi_equip defaults: strong=[1,2,3,10,11,12], weak=[5,6,7]
        month = df.index.month
        strong = month.isin([1, 2, 3, 10, 11, 12])
        weak = month.isin([5, 6, 7])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 30) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "semi_cycle":
        # SemiCycleRule: only for semi_equip/memory subsectors
        # golden cross + ADX >= 15 + near SMA_50 or EMA_21
        # + close > SMA_200 + RSI 25-45 (semi_equip threshold ~35)
        # + vol >= 0.8x + MACD_HIST >= -1.0
        rsi_threshold = 35  # semi_equip default
        near_ema21 = (dist_pct(close, ema21).abs()) <= 2.0 if ema21 is not None else False
        near_sma50_semi = (dist_pct(close, sma50).abs()) <= 3.0
        near_support = near_ema21 | near_sma50_semi
        macd_ok = (macd_hist >= -1.0) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (adx >= 15) & near_support &
                (close > sma200) & (rsi >= rsi_threshold - 10) &
                (rsi <= rsi_threshold + 15) &
                (vol_r >= 0.8) & macd_ok)
        conf[mask] = 0.55
        if ema21 is not None:
            ema_dist = dist_pct(close, ema21).abs()
            conf[mask & near_ema21 & (ema_dist < 1.0)] += 0.10
        sma50_dist = dist_pct(close, sma50).abs()
        conf[mask & near_sma50_semi & (sma50_dist < 1.5)] += 0.10
        conf[mask & (rsi < rsi_threshold)] += 0.10
        conf[mask & (rsi >= rsi_threshold) & (rsi < rsi_threshold + 5)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (vol_r > 1.5)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Financial Rules ──
    elif rule_name == "financial_mean_reversion":
        # FinancialMeanReversionRule: golden cross + within 3% of SMA_200
        # + ADX < 25 + BB% < 0.10 + RSI 28-42 + vol >= 0.5x
        bb_ok = (bb_pct < 0.10) if bb_pct is not None else False
        near_200 = (dist_pct(close, sma200).abs()) <= 3.0 if sma200 is not None else True
        adx_ok = (adx < 25) if adx is not None else True
        mask = ((sma50 > sma200) & near_200 & adx_ok & bb_ok &
                (rsi >= 28) & (rsi <= 42) & (vol_r >= 0.5))
        conf[mask] = 0.60
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.05
        conf[mask & (rsi < 32)] += 0.10
        conf[mask & (rsi >= 32) & (rsi < 35)] += 0.05
        conf[mask & (adx < 15)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "financial_pullback":
        # FinancialPullbackRule: golden cross + close > SMA_200 + RSI 30-60
        # + MACD_HIST >= -0.5 + vol >= 0.6x
        macd_ok = (macd_hist >= -0.5) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (close > sma200) &
                (rsi >= 30) & (rsi <= 60) & macd_ok & (vol_r >= 0.6))
        conf[mask] = 0.55
        conf[mask & (rsi >= 38) & (rsi <= 50)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (vol_r > 1.3)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "financial_seasonality":
        # FinancialSeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Strong=[10,11,12,1], Weak=[5,6]
        month = df.index.month
        strong = month.isin([10, 11, 12, 1])
        weak = month.isin([5, 6])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Utility Rules ──
    elif rule_name == "utility_mean_reversion":
        # UtilityMeanReversionRule: close > SMA_200 + golden cross + ADX < 20
        # + BB% < 0.15 + RSI 25-42 + vol >= 0.5x (excludes nuclear_power)
        bb_ok = (bb_pct < 0.15) if bb_pct is not None else True
        mask = ((close > sma200) & (sma50 > sma200) & (adx < 20) & bb_ok &
                (rsi >= 25) & (rsi <= 42) & (vol_r >= 0.5))
        conf[mask] = 0.60
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.07
        conf[mask & (bb_pct >= 0.05) & (bb_pct < 0.10)] += 0.03
        conf[mask & (rsi < 30)] += 0.10
        conf[mask & (rsi >= 30) & (rsi < 35)] += 0.05
        conf[mask & (adx < 12)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "utility_rate_reversion":
        # UtilityRateReversionRule: golden cross + within -3% to +2.5% of SMA_50
        # + RSI <= 45 + MACD_HIST >= -0.5
        dist_50 = dist_pct(close, sma50)
        macd_ok = (macd_hist >= -0.5) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (dist_50 >= -3.0) & (dist_50 <= 2.5) &
                (rsi <= 45) & macd_ok)
        conf[mask] = 0.55
        conf[mask & (dist_50.abs() < 1)] += 0.10
        conf[mask & (dist_50.abs() >= 1) & (dist_50.abs() < 2)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi < 40)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "utility_seasonality":
        # UtilitySeasonalityRule: uptrend + RSI 25-65
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Using nuclear_power defaults: strong=[1,2,10,11,12], weak=[5,6,9]
        # Regulated defaults: strong=[3,7,10], weak=[2,9]
        # Compromise: strong=[1,2,3,7,10,11,12], weak=[5,6,9]
        month = df.index.month
        strong = month.isin([1, 2, 3, 7, 10, 11, 12])
        weak = month.isin([5, 6, 9])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 65) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 1.5)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "nuclear_power_momentum":
        # NuclearPowerMomentumRule: EMA_21 > SMA_50 > SMA_200 + close > SMA_50
        # + ADX >= 20 + within -6% to +3% of EMA_21 + RSI 30-70 + vol >= 0.6x
        # + (MACD > signal or MACD_HIST > -0.5)
        ema_align = (ema21 > sma50) & (sma50 > sma200) if ema21 is not None else False
        dist_ema = dist_pct(close, ema21) if ema21 is not None else pd.Series(99, index=df.index)
        macd_ok = (macd > macd_sig) | (macd_hist > -0.5) if macd is not None else True
        adx_ok = (adx >= 20) if adx is not None else True
        mask = (ema_align & (close > sma50) & adx_ok &
                (dist_ema >= -6.0) & (dist_ema <= 3.0) &
                (rsi >= 30) & (rsi <= 70) & (vol_r >= 0.6) & macd_ok)
        conf[mask] = 0.55
        conf[mask & (dist_ema.abs() < 1)] += 0.10
        conf[mask & (dist_ema.abs() >= 1) & (dist_ema.abs() < 2)] += 0.05
        conf[mask & (macd > macd_sig)] += 0.05
        conf[mask & (adx > 30)] += 0.05
        conf[mask & (rsi >= 40) & (rsi <= 55)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 15)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Consumer Staples Rules ──
    elif rule_name == "consumer_staples_mean_reversion":
        # ConsumerStaplesMeanReversionRule: close > SMA_200 + golden cross + ADX < 18
        # + BB% < 0.15 + RSI 25-45 + vol >= 0.5x
        bb_ok = (bb_pct < 0.15) if bb_pct is not None else True
        mask = ((close > sma200) & (sma50 > sma200) & (adx < 18) & bb_ok &
                (rsi >= 25) & (rsi <= 45) & (vol_r >= 0.5))
        conf[mask] = 0.60
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.07
        conf[mask & (bb_pct >= 0.05) & (bb_pct < 0.10)] += 0.03
        conf[mask & (rsi < 30)] += 0.10
        conf[mask & (rsi >= 30) & (rsi < 35)] += 0.05
        conf[mask & (adx < 12)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "consumer_staples_pullback":
        # ConsumerStaplesPullbackRule: golden cross + within -3% to +2% of SMA_50
        # + RSI <= 45 + MACD_HIST >= -0.3
        dist_50 = dist_pct(close, sma50)
        macd_ok = (macd_hist >= -0.3) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (dist_50 >= -3.0) & (dist_50 <= 2.0) &
                (rsi <= 45) & macd_ok)
        conf[mask] = 0.55
        conf[mask & (dist_50.abs() < 1)] += 0.10
        conf[mask & (dist_50.abs() >= 1) & (dist_50.abs() < 2)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi < 40)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "consumer_staples_seasonality":
        # ConsumerStaplesSeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Strong=[9,10,11,12], Weak=[3,4,5]
        month = df.index.month
        strong = month.isin([9, 10, 11, 12])
        weak = month.isin([3, 4, 5])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    # ── Healthcare Rules ──
    elif rule_name == "healthcare_mean_reversion":
        # HealthcareMeanReversionRule: close > SMA_200 + golden cross + ADX < 20
        # + BB% < 0.15 + RSI 22-42 + vol >= 0.5x
        bb_ok = (bb_pct < 0.15) if bb_pct is not None else True
        mask = ((close > sma200) & (sma50 > sma200) & (adx < 20) & bb_ok &
                (rsi >= 22) & (rsi <= 42) & (vol_r >= 0.5))
        conf[mask] = 0.60
        conf[mask & (bb_pct < 0.0)] += 0.10
        conf[mask & (bb_pct >= 0.0) & (bb_pct < 0.05)] += 0.07
        conf[mask & (bb_pct >= 0.05) & (bb_pct < 0.10)] += 0.03
        conf[mask & (rsi < 28)] += 0.10
        conf[mask & (rsi >= 28) & (rsi < 35)] += 0.05
        conf[mask & (adx < 12)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf[mask & (vol_r > 1.2)] += 0.05
        conf = conf.clip(0.40, 0.90)

    elif rule_name == "healthcare_pullback":
        # HealthcarePullbackRule: golden cross + within -3% to +2.5% of SMA_50
        # + RSI <= 45 + MACD_HIST >= -0.5
        dist_50 = dist_pct(close, sma50)
        macd_ok = (macd_hist >= -0.5) if macd_hist is not None else True
        mask = ((sma50 > sma200) & (dist_50 >= -3.0) & (dist_50 <= 2.5) &
                (rsi <= 45) & macd_ok)
        conf[mask] = 0.55
        conf[mask & (dist_50.abs() < 1)] += 0.10
        conf[mask & (dist_50.abs() >= 1) & (dist_50.abs() < 2)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi < 40)] += 0.05
        conf[mask & (macd_hist > 0)] += 0.05
        conf[mask & (sma20 > sma50)] += 0.05
        conf = conf.clip(0.40, 0.85)

    elif rule_name == "healthcare_seasonality":
        # HealthcareSeasonalityRule: uptrend + RSI 25-70
        # Fires on strong + neutral months. Weak months BLOCKED.
        # Strong=[10,11,1,4], Weak=[6,7,8]
        month = df.index.month
        strong = month.isin([10, 11, 1, 4])
        weak = month.isin([6, 7, 8])
        uptrend = sma20 > sma50
        mask = uptrend & (rsi >= 25) & (rsi <= 70) & ~weak
        conf[mask] = 0.55  # base for neutral months
        conf[mask & strong] += 0.10  # strong month boost
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread > 2)] += 0.05
        conf = conf.clip(0.40, 0.85)

    else:
        logger.warning(f"Vectorized rule not implemented: {rule_name}, returning zero confidence")

    return conf


def compute_entry_signals(
    df_daily: pd.DataFrame,
    rules: List[str],
    min_confidence: float,
    max_price_extension_pct: float = 15.0,
    min_price_extension_pct: float = -3.0,
    max_trend_spread_pct: float = 20.0,
    symbol: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute entry signals for a rule combo on daily data.

    Returns:
        (signal_mask, avg_confidence) — boolean array + confidence array
        Both indexed to df_daily rows.
    """
    n = len(df_daily)

    # Compute per-rule confidence
    rule_confs = []
    for rule_name in rules:
        rc = _compute_rule_confidence(rule_name, df_daily, symbol)
        rule_confs.append(rc.values)

    rule_confs = np.array(rule_confs)  # shape: (n_rules, n_bars)

    # BUY signals: positive confidence
    buy_mask = rule_confs > 0  # (n_rules, n_bars)
    n_buy = buy_mask.sum(axis=0)  # how many rules fired per bar

    # Average confidence across triggered buy rules
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_sum = np.where(buy_mask, rule_confs, 0).sum(axis=0)
        avg_conf = np.where(n_buy > 0, buy_sum / n_buy, 0)

    # Entry condition: avg confidence >= min_confidence and at least 1 rule fired
    signal = (avg_conf >= min_confidence) & (n_buy > 0)

    # ── Pre-buy filters (same as strategy's next()) ──
    sma20 = df_daily["SMA_20"].values
    sma50 = df_daily["SMA_50"].values
    close = df_daily["close"].values

    # Filter 1: Price extension — skip if too far above/below SMA_20
    # Upside: max_price_extension_pct (default 15%)
    # Downside: min_price_extension_pct (default -15%, wider than backtrader's
    # -3% because we use daily closes, not intraday prices)
    with np.errstate(divide="ignore", invalid="ignore"):
        extension = np.where(sma20 > 0, (close - sma20) / sma20 * 100, 0)
    signal &= (extension <= max_price_extension_pct) & (extension >= min_price_extension_pct)

    # Filter 2: Trend maturity — skip if SMA_20/SMA_50 spread too wide
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_spread = np.where(sma50 > 0, (sma20 - sma50) / sma50 * 100, 0)
    signal &= (trend_spread <= max_trend_spread_pct)

    # Filter 3: Warmup — match backtrader's 200-bar warmup (no signals in first 200 bars)
    warmup_bars = 200
    if n > warmup_bars:
        signal[:warmup_bars] = False

    return signal, avg_conf


# ── Trade simulation (loops over trades, not bars) ────────────────────

def simulate_trades(
    signal_bars: np.ndarray,
    avg_conf: np.ndarray,
    df_exit: pd.DataFrame,
    df_daily: pd.DataFrame,
    profit_target: float,
    max_loss_pct: float,
    cooldown_bars: int,
    initial_cash: float,
    is_multi_tf: bool,
) -> VectorResult:
    """Simulate trades given entry signals and exit conditions.

    For multi-TF: entries on daily signal bars, exits scan 5-min bars.
    For daily-only: entries and exits both on daily bars.

    Args:
        signal_bars: boolean array on daily index — where entries can occur
        avg_conf: confidence array on daily index
        df_exit: DataFrame to scan for exits (5-min for multi-TF, daily otherwise)
        df_daily: Daily DataFrame (for date mapping)
        profit_target: e.g., 0.07 for 7%
        max_loss_pct: e.g., 5.0 for 5%
        cooldown_bars: daily bars to skip after exit
        initial_cash: starting capital
        is_multi_tf: whether df_exit is intraday
    """
    daily_dates = df_daily.index
    daily_close = df_daily["close"].values
    exit_close = df_exit["close"].values
    exit_index = df_exit.index
    n_daily = len(daily_dates)

    trades = []
    cash = initial_cash
    position_value = 0.0
    peak_value = initial_cash
    max_dd = 0.0
    max_dd_pct = 0.0

    bar = 0
    consecutive_pt = 0  # consecutive profit target exits

    while bar < n_daily:
        if not signal_bars[bar]:
            bar += 1
            continue

        # Entry
        entry_price = daily_close[bar]
        entry_date = daily_dates[bar]
        shares = int(cash * 0.95 / entry_price) if entry_price > 0 else 0
        if shares == 0:
            bar += 1
            continue

        target_price = entry_price * (1 + profit_target)
        stop_price = entry_price * (1 - max_loss_pct / 100)

        # Find exit
        exit_price = None
        exit_date = None
        exit_reason = None

        if is_multi_tf:
            # Scan 5-min bars starting from the NEXT daily bar's date
            # (don't exit on entry day for multi-TF)
            if bar + 1 < n_daily:
                next_day = daily_dates[bar + 1]
            else:
                next_day = daily_dates[bar] + pd.Timedelta(days=1)

            # Find 5-min bars from next_day onward
            exit_start_idx = exit_index.searchsorted(next_day, side="left")

            for ei in range(exit_start_idx, len(exit_close)):
                price = exit_close[ei]
                if price >= target_price:
                    exit_price = target_price  # fill at target
                    exit_date = exit_index[ei]
                    exit_reason = "Profit target"
                    break
                elif price <= stop_price:
                    exit_price = stop_price  # fill at stop
                    exit_date = exit_index[ei]
                    exit_reason = f"Max loss cap ({max_loss_pct}%)"
                    break
        else:
            # Daily-only: scan subsequent daily bars
            for di in range(bar + 1, n_daily):
                price = daily_close[di]
                if price >= target_price:
                    exit_price = target_price
                    exit_date = daily_dates[di]
                    exit_reason = "Profit target"
                    break
                elif price <= stop_price:
                    exit_price = stop_price
                    exit_date = daily_dates[di]
                    exit_reason = f"Max loss cap ({max_loss_pct}%)"
                    break

        # If no exit found, close at last bar
        if exit_price is None:
            if is_multi_tf and len(exit_close) > 0:
                exit_price = exit_close[-1]
                exit_date = exit_index[-1]
            else:
                exit_price = daily_close[-1]
                exit_date = daily_dates[-1]
            exit_reason = "End of backtest"

        # Record trade
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_dollar = shares * (exit_price - entry_price)
        cash += pnl_dollar  # simplified: all-in, all-out

        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "profit_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "shares": shares,
        })

        # Track drawdown
        current_value = cash
        if current_value > peak_value:
            peak_value = current_value
        dd = peak_value - current_value
        dd_pct = dd / peak_value * 100 if peak_value > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # Consecutive profit target tracking (for doubled cooldown)
        if exit_reason == "Profit target":
            consecutive_pt += 1
        else:
            consecutive_pt = 0

        # Advance past exit date + cooldown
        if exit_date is not None:
            exit_day = exit_date.date() if hasattr(exit_date, "date") else exit_date
            # Find the daily bar corresponding to exit
            exit_daily_bar = np.searchsorted(
                daily_dates.values, np.datetime64(exit_day), side="right"
            )
            effective_cooldown = cooldown_bars * 2 if consecutive_pt >= 2 else cooldown_bars
            bar = exit_daily_bar + effective_cooldown
        else:
            bar += 1

    # Compute metrics
    total_trades = len(trades)
    if total_trades == 0:
        return VectorResult(
            symbol="", start_date=date.today(), end_date=date.today(),
            strategy_name="", initial_cash=initial_cash,
            final_value=initial_cash, total_return=0, total_trades=0,
            winning_trades=0, losing_trades=0, win_rate=0,
        )

    wins = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = np.mean([t["pnl_dollar"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_dollar"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl_dollar"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_dollar"] for t in losses)) if losses else 0
    profit_factor = min(gross_profit / gross_loss, 999.99) if gross_loss > 0 else (999.99 if gross_profit > 0 else 0)

    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash

    # Sharpe ratio (annualized from trade returns)
    if total_trades >= 2:
        trade_returns = np.array([t["profit_pct"] for t in trades])
        mean_r = trade_returns.mean()
        std_r = trade_returns.std()
        if std_r > 0:
            # Annualize: assume ~50 trades/year as baseline
            trades_per_year = min(total_trades * 252 / max((daily_dates[-1] - daily_dates[0]).days, 1), 252)
            sharpe = (mean_r / std_r) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
    else:
        sharpe = 0

    return VectorResult(
        symbol="",
        start_date=daily_dates[0].date() if hasattr(daily_dates[0], "date") else daily_dates[0],
        end_date=daily_dates[-1].date() if hasattr(daily_dates[-1], "date") else daily_dates[-1],
        strategy_name="vectorized",
        initial_cash=initial_cash,
        final_value=final_value,
        total_return=total_return,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        trades=[{k: (v.isoformat() if hasattr(v, "isoformat") else v)
                 for k, v in t.items()} for t in trades],
    )


# ── Public API ────────────────────────────────────────────────────────

class VectorizedEngine:
    """Fast vectorized backtesting for parameter sweeps.

    Usage:
        engine = VectorizedEngine(df_daily, df_5min, initial_cash=1000)
        result = engine.run(rules=["buy_dip_in_uptrend", "rsi_oversold"],
                           min_confidence=0.5, profit_target=0.07,
                           max_loss_pct=5.0, cooldown_bars=5)
    """

    def __init__(
        self,
        df_daily: pd.DataFrame,
        df_intraday: Optional[pd.DataFrame] = None,
        initial_cash: float = 1000,
    ):
        self.df_daily = df_daily
        self.df_intraday = df_intraday
        self.initial_cash = initial_cash
        self.is_multi_tf = df_intraday is not None

        # Pre-compute date arrays for fast lookups
        self._daily_dates = df_daily.index
        if self.is_multi_tf:
            self._intraday_dates = df_intraday.index

    def run(
        self,
        symbol: str = "",
        rules: List[str] = None,
        min_confidence: float = 0.5,
        profit_target: float = 0.07,
        max_loss_pct: float = 5.0,
        cooldown_bars: int = 5,
        max_price_extension_pct: float = 15.0,
        max_trend_spread_pct: float = 20.0,
        **ignored_kwargs,
    ) -> VectorResult:
        """Run a single vectorized backtest.

        Accepts and ignores kwargs not relevant to vectorized mode
        (stop_loss, stop_mode, atr_*, etc.) for interface compatibility.
        """
        if rules is None:
            rules = ["buy_dip_in_uptrend"]

        # Step 1: Compute entry signals on daily data
        signal_bars, avg_conf = compute_entry_signals(
            self.df_daily, rules, min_confidence,
            max_price_extension_pct=max_price_extension_pct,
            max_trend_spread_pct=max_trend_spread_pct,
            symbol=symbol,
        )

        # Step 2: Simulate trades
        df_exit = self.df_intraday if self.is_multi_tf else self.df_daily
        result = simulate_trades(
            signal_bars, avg_conf, df_exit, self.df_daily,
            profit_target, max_loss_pct, cooldown_bars,
            self.initial_cash, self.is_multi_tf,
        )
        result.symbol = symbol
        result.strategy_name = ", ".join(rules)
        return result
