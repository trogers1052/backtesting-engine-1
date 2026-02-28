"""Regime-stratified analysis: bull, bear, chop classification using SPY."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.timescale_loader import TimescaleLoader
from ..engine.backtrader_runner import BacktestResult
from ..indicators.pandas_ta_bridge import calculate_indicators
from .bootstrap import calculate_trade_sharpe

logger = logging.getLogger(__name__)


@dataclass
class RegimeMetrics:
    """Performance metrics for trades within one regime."""

    regime: str
    total_trades: int
    winning_trades: int
    win_rate: float
    total_return: float  # sum of all trade returns in this regime
    avg_trade_return: float
    sharpe_ratio: float
    profit_factor: float


@dataclass
class RegimeAnalysisResult:
    """Regime-stratified backtest results."""

    symbol: str
    regime_metrics: Dict[str, RegimeMetrics]

    @property
    def regime_dependent(self) -> bool:
        """Performance concentrated in one regime (>70% of profit)."""
        profits = {
            k: m.total_return
            for k, m in self.regime_metrics.items()
            if m.total_return > 0
        }
        total_profit = sum(profits.values())
        if total_profit <= 0:
            return False

        max_regime_profit = max(profits.values())
        return max_regime_profit / total_profit > 0.7


class RegimeClassifier:
    """Classify market regimes using SPY price vs SMA_50/SMA_200 + VIX thresholds."""

    # VIX thresholds aligned with CLAUDE.md regime framework
    VIX_VOLATILE = 25.0  # "Volatile Chop" — most signals disabled
    VIX_CRISIS = 35.0  # "Crisis / Risk-Off" — defensive only, kill switch

    def __init__(self, loader: TimescaleLoader, vix_symbol: str = "VIX"):
        self.loader = loader
        self.vix_symbol = vix_symbol
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_regimes(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Load SPY data and classify each day as bull/bear/chop/crisis.

        Uses SPY SMA_50/SMA_200 for trend classification, then overlays
        VIX thresholds when available:
          - VIX > 35 → "crisis" (overrides all other regimes)
          - VIX > 25 + not bull → "volatile" (high-vol chop)

        Returns DataFrame with datetime index and 'regime' column.
        """
        cache_key = f"SPY_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.info(f"Loading SPY data for regime classification: {start_date} to {end_date}")
        df = self.loader.load("SPY", start_date, end_date, timeframe="daily")

        if df.empty:
            raise ValueError("No SPY data available for regime classification")

        df = calculate_indicators(df)

        # Base classification from SPY trend structure
        conditions = [
            (df["close"] > df["SMA_200"]) & (df["SMA_50"] > df["SMA_200"]),
            (df["close"] < df["SMA_200"]) & (df["SMA_50"] < df["SMA_200"]),
        ]
        choices = ["bull", "bear"]
        df["regime"] = np.select(conditions, choices, default="chop")

        # Overlay VIX thresholds when data is available
        df = self._apply_vix_overlay(df, start_date, end_date)

        result = df[["regime"]].copy()
        self._cache[cache_key] = result
        return result

    def _apply_vix_overlay(
        self, df: pd.DataFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Refine regime classification using VIX fear gauge.

        VIX > 35: crisis (overrides everything)
        VIX > 25 and regime != bull: volatile
        """
        try:
            vix_df = self.loader.load(
                self.vix_symbol, start_date, end_date, timeframe="daily"
            )
        except Exception:
            logger.debug(f"Could not load {self.vix_symbol} data, skipping VIX overlay")
            return df

        if vix_df.empty:
            logger.debug(f"No {self.vix_symbol} data available, using SPY-only classification")
            return df

        # Align VIX close with SPY dates
        vix_close = vix_df["close"].reindex(df.index, method="ffill")

        # Crisis: VIX > 35 overrides all regimes
        crisis_mask = vix_close > self.VIX_CRISIS
        df.loc[crisis_mask, "regime"] = "crisis"

        # Volatile: VIX > 25 but not crisis, and not already bull
        volatile_mask = (
            (vix_close > self.VIX_VOLATILE)
            & ~crisis_mask
            & (df["regime"] != "bull")
        )
        df.loc[volatile_mask, "regime"] = "volatile"

        logger.info(
            f"VIX overlay applied: {crisis_mask.sum()} crisis days, "
            f"{volatile_mask.sum()} volatile days"
        )

        return df


def _lookup_regime(regime_df: pd.DataFrame, entry_date) -> str:
    """Find the regime for a given entry date."""
    if isinstance(entry_date, str):
        entry_date = pd.to_datetime(entry_date)
    elif isinstance(entry_date, date) and not isinstance(entry_date, datetime):
        entry_date = pd.to_datetime(entry_date)

    target = entry_date.date() if hasattr(entry_date, "date") else entry_date

    # Find matching date in regime index
    matches = regime_df.index[regime_df.index.date == target]
    if len(matches) > 0:
        return regime_df.loc[matches[0], "regime"]

    # Fallback: find nearest prior date
    prior = regime_df.index[regime_df.index.date <= target]
    if len(prior) > 0:
        return regime_df.loc[prior[-1], "regime"]

    return "unknown"


def analyze_by_regime(
    backtest_result: BacktestResult,
    loader: TimescaleLoader,
) -> RegimeAnalysisResult:
    """Stratify backtest results by market regime.

    Uses SPY price vs SMA_50/SMA_200 to classify each trade's entry
    date as bull, bear, or chop, then reports metrics per regime.

    Args:
        backtest_result: Standard backtest output.
        loader: TimescaleLoader for fetching SPY data.

    Returns:
        RegimeAnalysisResult with per-regime metrics.
    """
    classifier = RegimeClassifier(loader)
    regime_df = classifier.get_regimes(
        backtest_result.start_date,
        backtest_result.end_date,
    )

    # Group trades by regime (includes VIX-derived regimes)
    trades_by_regime: Dict[str, list] = {
        "bull": [],
        "bear": [],
        "chop": [],
        "volatile": [],
        "crisis": [],
    }

    for trade in backtest_result.trades:
        entry_date = trade.get("entry_date")
        if not entry_date:
            continue

        regime = _lookup_regime(regime_df, entry_date)
        if regime in trades_by_regime:
            trades_by_regime[regime].append(trade)

    # Calculate metrics per regime
    regime_metrics = {}

    for regime_name, trades in trades_by_regime.items():
        if not trades:
            continue

        pnl = np.array(
            [t["profit_pct"] for t in trades if t.get("profit_pct") is not None]
        )

        if len(pnl) == 0:
            continue

        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99

        sharpe = calculate_trade_sharpe(pnl) if len(pnl) >= 2 else 0.0

        regime_metrics[regime_name] = RegimeMetrics(
            regime=regime_name,
            total_trades=len(pnl),
            winning_trades=len(wins),
            win_rate=float(len(wins) / len(pnl)),
            total_return=float(np.sum(pnl)),
            avg_trade_return=float(np.mean(pnl)),
            sharpe_ratio=sharpe,
            profit_factor=min(profit_factor, 999.99),
        )

    return RegimeAnalysisResult(
        symbol=backtest_result.symbol,
        regime_metrics=regime_metrics,
    )
