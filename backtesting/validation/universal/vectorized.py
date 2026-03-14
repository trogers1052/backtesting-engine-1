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

    # ── RSI Rules ──
    if rule_name == "rsi_oversold":
        mask = rsi < 30
        conf[mask] = 0.5 + (30 - rsi[mask]) / 25  # 0.5 at RSI=30, 0.9 at RSI=20
        conf = conf.clip(0, 0.9)

    elif rule_name == "rsi_overbought":
        # SELL signal — return negative confidence to distinguish
        mask = rsi > 70
        conf[mask] = -(0.4 + (rsi[mask] - 70) / 33.3)
        conf = conf.clip(-0.85, 0)

    elif rule_name == "rsi_approaching_oversold":
        mask = (rsi >= 30) & (rsi <= 40)
        conf[mask] = 0.4  # WATCH signal

    # ── MACD Rules ──
    elif rule_name == "macd_bullish_crossover":
        mask = macd > macd_sig
        hist_fresh = macd_hist.abs() < 0.1
        conf[mask & hist_fresh] = 0.65
        conf[mask & ~hist_fresh] = 0.5

    elif rule_name == "macd_bearish_crossover":
        mask = macd < macd_sig
        conf[mask] = -0.5  # SELL/WATCH

    elif rule_name == "macd_momentum":
        mask = macd_hist > 0.05
        conf[mask] = np.minimum(0.5 + macd_hist[mask] * 2, 0.7)

    # ── Trend Rules ──
    elif rule_name == "weekly_uptrend":
        mask = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        conf[mask] = 0.55
        conf[mask & (spread >= 1)] = 0.7
        conf[mask & (spread >= 2)] = 0.85

    elif rule_name == "monthly_uptrend":
        mask = sma50 > sma200
        spread = safe_spread(sma50, sma200)
        conf[mask] = 0.55
        conf[mask & (spread >= 1)] = 0.7
        conf[mask & (spread >= 2)] = 0.85

    elif rule_name == "trend_alignment":
        mask = (sma20 > sma50) & (sma50 > sma200)
        conf[mask] = 0.75

    elif rule_name == "golden_cross":
        mask = (sma50 > sma200)
        conf[mask] = 0.65

    elif rule_name == "death_cross":
        mask = (sma50 < sma200)
        conf[mask] = -0.6

    elif rule_name == "trend_break_warning":
        mask = (sma20 < sma50)
        conf[mask] = -0.55

    # ── Composite Rules ──
    elif rule_name == "buy_dip_in_uptrend":
        uptrend = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        dip = rsi < 40
        mask = uptrend & dip
        conf[mask & (rsi < 30)] = 0.85
        conf[mask & (rsi >= 30) & (rsi < 35)] = 0.70
        conf[mask & (rsi >= 35) & (rsi < 40)] = 0.55
        # Trend spread bonus
        conf[mask & (spread >= 2)] += 0.10
        conf[mask & (spread >= 1) & (spread < 2)] += 0.05
        conf = conf.clip(0, 0.95)

    elif rule_name == "strong_buy_signal":
        full_align = (sma20 > sma50) & (sma50 > sma200)
        dip = rsi < 35
        mask = full_align & dip
        conf[mask & (rsi < 25)] = 0.90
        conf[mask & (rsi >= 25) & (rsi < 30)] = 0.80
        conf[mask & (rsi >= 30) & (rsi < 35)] = 0.70
        # Trend strength bonus
        spread = safe_spread(sma20, sma50)
        conf[mask & (spread >= 2)] += 0.10
        conf = conf.clip(0, 0.98)

    elif rule_name == "rsi_macd_confluence":
        mask = (rsi < 35) & (macd > macd_sig)
        conf[mask] = 0.70
        conf[mask & (rsi < 30)] += 0.15
        conf[mask & (rsi >= 30) & (rsi < 33)] += 0.10
        conf[mask & (macd_hist > 0.05)] += 0.05
        conf = conf.clip(0, 0.95)

    elif rule_name == "dip_recovery":
        uptrend = sma20 > sma50
        recovery = (rsi >= 30) & (rsi <= 45)
        mask = uptrend & recovery
        conf[mask] = 0.55 + (45 - rsi[mask]) / 30
        conf = conf.clip(0, 0.75)

    # ── Enhanced Rules ──
    elif rule_name == "enhanced_buy_dip":
        uptrend = sma20 > sma50
        spread = safe_spread(sma20, sma50)
        above_200 = close > sma200
        dip = rsi < 35
        mask = uptrend & (spread >= 1.5) & dip & above_200
        # RSI score
        rsi_score = pd.Series(0.0, index=df.index)
        rsi_score[rsi < 30] = 0.40
        rsi_score[(rsi >= 30) & (rsi < 33)] = 0.30
        rsi_score[(rsi >= 33) & (rsi < 35)] = 0.20
        # Trend score
        trend_score = pd.Series(0.0, index=df.index)
        trend_score[spread >= 3] = 0.25
        trend_score[(spread >= 2) & (spread < 3)] = 0.20
        trend_score[(spread >= 1.5) & (spread < 2)] = 0.15
        # Alignment bonus
        align_bonus = pd.Series(0.0, index=df.index)
        align_bonus[sma50 > sma200] = 0.15
        total = rsi_score + trend_score + align_bonus
        conf[mask] = total[mask].clip(0.5, 0.95)

    elif rule_name == "momentum_reversal":
        golden = sma50 > sma200
        vol_ok = volume >= (vol_sma * 0.5) if vol_sma is not None else True
        rsi_recovery = (rsi >= 30) & (rsi <= 40)
        macd_bull = macd > macd_sig
        mask = golden & vol_ok & rsi_recovery & macd_bull
        conf[mask] = 0.55
        conf[mask & (sma20 > sma50)] += 0.15
        conf[mask & (macd_hist > 0.10)] += 0.10
        conf[mask & (macd_hist > 0.05) & (macd_hist <= 0.10)] += 0.05
        conf[mask & (rsi < 35)] += 0.10
        conf[mask & (rsi >= 35) & (rsi <= 40)] += 0.05
        conf = conf.clip(0, 0.90)

    elif rule_name == "trend_continuation":
        full_align = (sma20 > sma50) & (sma50 > sma200)
        at_support = ((close - sma20).abs() / sma20 * 100) <= 2
        rsi_mod = (rsi >= 35) & (rsi <= 60)
        vol_ok = volume >= (vol_sma * 0.5) if vol_sma is not None else True
        mask = full_align & at_support & rsi_mod & vol_ok
        conf[mask] = 0.60
        spread20_50 = safe_spread(sma20, sma50)
        spread50_200 = safe_spread(sma50, sma200)
        conf[mask & (spread20_50 > 3)] += 0.10
        conf[mask & (spread50_200 > 5)] += 0.10
        conf = conf.clip(0, 0.85)

    # ── Mining Rules ──
    elif rule_name == "commodity_breakout":
        sma_ref = sma50 if sma50 is not None else sma20
        mask = (close > sma_ref * 1.02) & (volume > vol_sma * 1.2) if vol_sma is not None else (close > sma_ref * 1.02)
        conf[mask] = 0.65

    elif rule_name == "miner_metal_ratio":
        # Simplified: use RSI oversold as proxy (actual ratio needs commodity data)
        mask = (rsi < 35) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "dollar_weakness":
        # Simplified: bullish when MACD bullish + uptrend (actual needs DXY data)
        mask = (macd > macd_sig) & (sma20 > sma50)
        conf[mask] = 0.55

    elif rule_name == "seasonality":
        # Seasonal months — extract month from index
        month = df.index.month
        strong_months = month.isin([1, 2, 8, 9, 11, 12])  # Gold seasonal strength
        mask = strong_months & (rsi < 50)
        conf[mask] = 0.55
        conf[mask & (rsi < 40)] = 0.65

    elif rule_name == "volume_breakout":
        vol_ratio = volume / vol_sma if vol_sma is not None else pd.Series(1.0, index=df.index)
        mask = (vol_ratio >= 1.5) & (close > sma20)
        conf[mask] = 0.60
        conf[mask & (vol_ratio >= 2.0)] = 0.70

    # ── Energy Rules ──
    elif rule_name == "energy_momentum":
        mask = (adx > 20) & (rsi > 40) & (rsi < 70) & (macd > macd_sig)
        conf[mask] = 0.60

    elif rule_name == "energy_mean_reversion":
        mask = (rsi < 35) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 35)
        conf[mask] = 0.65

    elif rule_name == "energy_seasonality":
        month = df.index.month
        strong = month.isin([10, 11, 12, 1, 2])
        mask = strong & (rsi < 50)
        conf[mask] = 0.55

    elif rule_name == "midstream_yield_reversion":
        mask = (rsi < 35) & (sma50 > sma200)
        conf[mask] = 0.60

    # ── Defense Rules ──
    elif rule_name == "defense_momentum":
        mask = (adx > 20) & (sma20 > sma50) & (rsi > 40) & (rsi < 65)
        conf[mask] = 0.60

    elif rule_name == "defense_mean_reversion":
        mask = (rsi < 35) & (sma50 > sma200)
        conf[mask] = 0.65

    elif rule_name == "defense_budget_cycle":
        month = df.index.month
        budget_months = month.isin([9, 10, 11, 12, 1, 2, 3])
        mask = budget_months & (rsi < 45) & (sma20 > sma50)
        conf[mask] = 0.55

    elif rule_name == "defense_counter_cyclical":
        mask = (rsi < 40) & (sma50 > sma200)
        conf[mask] = 0.60

    # ── Industrial Rules ──
    elif rule_name == "industrial_mean_reversion":
        mask = (rsi < 35) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 35)
        conf[mask] = 0.65

    elif rule_name == "industrial_pullback":
        mask = (rsi < 40) & (sma20 > sma50) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "industrial_seasonality":
        month = df.index.month
        strong = month.isin([10, 11, 12, 1])
        mask = strong & (rsi < 50) & (sma20 > sma50)
        conf[mask] = 0.55

    # ── Tech Rules ──
    elif rule_name == "tech_ema_pullback":
        mask = (close <= ema21 * 1.01) & (ema9 > ema21) & (rsi < 45)
        conf[mask] = 0.60

    elif rule_name == "tech_mean_reversion":
        mask = (rsi < 30) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 30)
        conf[mask] = 0.70

    elif rule_name == "tech_seasonality":
        month = df.index.month
        strong = month.isin([10, 11, 1, 4])
        mask = strong & (rsi < 45)
        conf[mask] = 0.55

    elif rule_name == "semi_cycle":
        mask = (rsi < 35) & (sma50 > sma200) & (macd > macd_sig)
        conf[mask] = 0.65

    # ── Financial Rules ──
    elif rule_name == "financial_mean_reversion":
        bb_os = bb_pct < 0.10 if bb_pct is not None else pd.Series(False, index=df.index)
        mask = (rsi >= 28) & (rsi <= 42) & bb_os
        adx_ok = adx < 25 if adx is not None else True
        mask = mask & adx_ok
        conf[mask] = 0.65

    elif rule_name == "financial_pullback":
        mask = (rsi < 40) & (sma20 > sma50) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "financial_seasonality":
        month = df.index.month
        strong = month.isin([10, 11, 12, 1])
        mask = strong & (rsi < 50)
        conf[mask] = 0.55

    # ── Utility Rules ──
    elif rule_name == "utility_mean_reversion":
        mask = (rsi < 35) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 35)
        conf[mask] = 0.65

    elif rule_name == "utility_rate_reversion":
        mask = (rsi < 40) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "utility_seasonality":
        month = df.index.month
        strong = month.isin([5, 6, 7, 8, 9])
        mask = strong & (rsi < 50) & (sma20 > sma50)
        conf[mask] = 0.55

    elif rule_name == "nuclear_power_momentum":
        mask = (adx > 20) & (sma20 > sma50) & (rsi > 40) & (rsi < 65)
        conf[mask] = 0.60

    # ── Consumer Staples Rules ──
    elif rule_name == "consumer_staples_mean_reversion":
        mask = (rsi < 35) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 35)
        conf[mask] = 0.65

    elif rule_name == "consumer_staples_pullback":
        mask = (rsi < 40) & (sma20 > sma50) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "consumer_staples_seasonality":
        month = df.index.month
        strong = month.isin([9, 10, 11, 12])
        mask = strong & (rsi < 50)
        conf[mask] = 0.55

    # ── Healthcare Rules ──
    elif rule_name == "healthcare_mean_reversion":
        mask = (rsi < 35) & (close <= bb_lower * 1.02) if bb_lower is not None else (rsi < 35)
        conf[mask] = 0.65

    elif rule_name == "healthcare_pullback":
        mask = (rsi < 40) & (sma20 > sma50) & (sma50 > sma200)
        conf[mask] = 0.60

    elif rule_name == "healthcare_seasonality":
        month = df.index.month
        strong = month.isin([10, 11, 1, 4])
        mask = strong & (rsi < 50)
        conf[mask] = 0.55

    else:
        logger.warning(f"Vectorized rule not implemented: {rule_name}, returning zero confidence")

    return conf


def compute_entry_signals(
    df_daily: pd.DataFrame,
    rules: List[str],
    min_confidence: float,
    max_price_extension_pct: float = 15.0,
    min_price_extension_pct: float = -15.0,
    max_trend_spread_pct: float = 20.0,
    symbol: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute entry signals for a rule combo on daily data.

    The downside extension threshold is wider than backtrader's -3% because
    the vectorized engine uses daily closes, while backtrader checks intraday
    prices that can bounce closer to the SMA during the day. A -15% floor
    still filters extreme outliers while capturing valid dip-buy setups.

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
