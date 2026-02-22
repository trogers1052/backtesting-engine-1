"""
Backtrader Runner

Main orchestrator for running backtests with decision-engine rules.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Type

import backtrader as bt
import pandas as pd

from ..config import settings
from ..data import TimescaleLoader
from ..indicators import calculate_indicators
from ..strategies import DecisionEngineStrategy, create_strategy
from .data_feed import create_data_feed
from .sizer import PercentSizer, CompoundingSizer, FixedPercentSizer, RiskBasedSizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    symbol: str
    start_date: date
    end_date: date
    strategy_name: str

    # Performance metrics
    initial_cash: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Advanced metrics
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_pct: Optional[float] = None

    # Trade details
    trades: List[Dict] = None


class BacktraderRunner:
    """
    Orchestrates backtrader backtests with decision-engine rules.

    Usage:
        runner = BacktraderRunner()
        result = runner.run(
            symbol="WPM",
            start_date=date(2021, 1, 1),
            rules=["buy_dip_in_uptrend", "strong_buy_signal"],
        )
        print(result)
    """

    def __init__(
        self,
        initial_cash: float = None,
        commission: float = None,
        compound: bool = True,
        sizing_mode: str = None,
        risk_pct: float = None,
        max_position_pct: float = None,
    ):
        """Initialize the runner."""
        self.initial_cash = initial_cash if initial_cash is not None else settings.default_initial_cash
        self.commission = commission if commission is not None else settings.default_commission
        self.compound = compound
        self.sizing_mode = sizing_mode or settings.default_sizing_mode
        self.risk_pct = risk_pct if risk_pct is not None else settings.default_risk_pct
        self.max_position_pct = max_position_pct if max_position_pct is not None else settings.default_max_position_pct
        self.loader = TimescaleLoader()

    def run(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
        timeframe: str = None,
        rules: List[str] = None,
        min_confidence: float = None,
        profit_target: float = None,
        stop_loss: float = None,
        require_consensus: bool = False,
        rule_params: Dict[str, Dict] = None,
        allow_scale_in: bool = False,
        max_scale_ins: int = 2,
        scale_in_size: float = 0.5,
        stop_mode: str = None,
        atr_multiplier: float = None,
        atr_stop_min_pct: float = None,
        atr_stop_max_pct: float = None,
        max_price_extension_pct: float = None,
        cooldown_bars: int = None,
        max_trend_spread_pct: float = None,
        max_loss_pct: float = None,
    ) -> BacktestResult:
        """
        Run a backtest for a single symbol.

        Args:
            symbol: Stock symbol to test
            start_date: Start date (default from settings)
            end_date: End date (default today)
            rules: List of rule names to use
            min_confidence: Minimum confidence threshold
            profit_target: Profit target percentage
            stop_loss: Stop loss percentage
            require_consensus: Require multiple rules to agree
            rule_params: Parameter overrides per rule
            allow_scale_in: Allow averaging down on deeper dips
            max_scale_ins: Maximum number of scale-ins per position
            scale_in_size: Scale-in size relative to initial position

        Returns:
            BacktestResult with performance metrics
        """
        # Set defaults
        start_date = start_date or datetime.strptime(
            settings.default_start_date, "%Y-%m-%d"
        ).date()
        end_date = end_date or date.today()
        timeframe = timeframe or settings.default_timeframe
        rules = rules or ["buy_dip_in_uptrend"]
        min_confidence = min_confidence if min_confidence is not None else settings.default_min_confidence
        profit_target = profit_target if profit_target is not None else settings.default_profit_target
        stop_loss = stop_loss if stop_loss is not None else settings.default_stop_loss
        stop_mode = stop_mode or settings.default_stop_mode
        atr_multiplier = atr_multiplier if atr_multiplier is not None else settings.default_atr_multiplier
        atr_stop_min_pct = atr_stop_min_pct if atr_stop_min_pct is not None else settings.default_atr_stop_min_pct
        atr_stop_max_pct = atr_stop_max_pct if atr_stop_max_pct is not None else settings.default_atr_stop_max_pct
        max_price_extension_pct = max_price_extension_pct if max_price_extension_pct is not None else settings.default_max_price_extension_pct
        cooldown_bars = cooldown_bars if cooldown_bars is not None else settings.default_cooldown_bars
        max_trend_spread_pct = max_trend_spread_pct if max_trend_spread_pct is not None else settings.default_max_trend_spread_pct
        max_loss_pct = max_loss_pct if max_loss_pct is not None else settings.default_max_loss_pct

        logger.info(f"Running backtest for {symbol}")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Rules: {rules}")

        # Load data
        df = self.loader.load(symbol, start_date, end_date, timeframe=timeframe)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Calculate indicators
        df = calculate_indicators(df)
        logger.info(f"  Bars with indicators: {len(df)}")

        # Create cerebro engine
        cerebro = bt.Cerebro()

        # Add data feed
        data = create_data_feed(df, name=symbol)
        cerebro.adddata(data)

        # Create and add strategy
        strategy_class = create_strategy(
            rule_names=rules,
            min_confidence=min_confidence,
            require_consensus=require_consensus,
            profit_target=profit_target,
            stop_loss=stop_loss,
            rule_params=rule_params,
            allow_scale_in=allow_scale_in,
            max_scale_ins=max_scale_ins,
            scale_in_size=scale_in_size,
            stop_mode=stop_mode,
            atr_multiplier=atr_multiplier,
            atr_stop_min_pct=atr_stop_min_pct,
            atr_stop_max_pct=atr_stop_max_pct,
            max_price_extension_pct=max_price_extension_pct,
            cooldown_bars=cooldown_bars,
            max_trend_spread_pct=max_trend_spread_pct,
            max_loss_pct=max_loss_pct,
        )
        cerebro.addstrategy(strategy_class)

        # Configure broker
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Add sizer based on sizing mode
        if self.sizing_mode == "risk_based":
            cerebro.addsizer(
                RiskBasedSizer,
                risk_pct=self.risk_pct,
                max_position_pct=self.max_position_pct,
            )
            logger.info(
                f"  Position sizing: Risk-based ({self.risk_pct}% risk, "
                f"{self.max_position_pct}% max position, stop_mode={stop_mode})"
            )
        elif self.compound:
            cerebro.addsizer(CompoundingSizer, percents=95)
            logger.info("  Position sizing: Compounding (reinvesting profits)")
        else:
            cerebro.addsizer(FixedPercentSizer, percents=95, initial_capital=self.initial_cash)
            logger.info("  Position sizing: Fixed (no reinvestment)")

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # Run backtest
        logger.info("  Running backtest...")
        results = cerebro.run()
        strategy = results[0]

        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # Get trade analyzer results
        trade_analysis = strategy.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get("total", {}).get("total", 0)
        won = trade_analysis.get("won", {})
        lost = trade_analysis.get("lost", {})
        winning_trades = won.get("total", 0)
        losing_trades = lost.get("total", 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average win/loss
        avg_win = won.get("pnl", {}).get("average", 0) if winning_trades > 0 else 0
        avg_loss = lost.get("pnl", {}).get("average", 0) if losing_trades > 0 else 0

        # Profit factor
        gross_profit = won.get("pnl", {}).get("total", 0)
        gross_loss = abs(lost.get("pnl", {}).get("total", 0))
        if gross_loss > 0:
            profit_factor = min(gross_profit / gross_loss, 999.99)
        else:
            profit_factor = 999.99 if gross_profit > 0 else 0.0

        # Sharpe ratio
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get("sharperatio")

        # Drawdown
        dd_analysis = strategy.analyzers.drawdown.get_analysis()
        max_drawdown = dd_analysis.get("max", {}).get("moneydown", 0)
        max_drawdown_pct = dd_analysis.get("max", {}).get("drawdown", 0)

        # Build trade list from strategy
        trades = []
        for trade in strategy.trade_records:
            trades.append({
                "entry_date": trade.entry_date.isoformat() if trade.entry_date else None,
                "entry_price": trade.entry_price,
                "entry_reason": trade.entry_reason,
                "entry_confidence": trade.entry_confidence,
                "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "profit_pct": trade.profit_pct,
                "rules_triggered": trade.rules_triggered,
            })

        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_name=", ".join(rules),
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
        )

        logger.info(f"  Backtest complete: {total_trades} trades, {win_rate:.1%} win rate")
        return result

    def run_multiple(
        self,
        symbols: List[str],
        start_date: date = None,
        end_date: date = None,
        **kwargs,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtests for multiple symbols.

        Args:
            symbols: List of symbols to test
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments passed to run()

        Returns:
            Dict mapping symbol to BacktestResult
        """
        results = {}

        for symbol in symbols:
            try:
                result = self.run(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs,
                )
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to backtest {symbol}: {e}")

        return results
