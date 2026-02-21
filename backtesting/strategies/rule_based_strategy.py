"""
Decision Engine Strategy for Backtrader

Wraps decision-engine rules into a backtrader Strategy.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import backtrader as bt

# Import decision-engine components
import os
import sys
decision_engine_path = os.environ.get("DECISION_ENGINE_PATH", "/app")
if decision_engine_path not in sys.path:
    sys.path.insert(0, decision_engine_path)

from decision_engine.rules.base import Rule, SymbolContext, SignalType
from decision_engine.rules.registry import RuleRegistry

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    entry_date: datetime
    entry_price: float
    entry_reason: str
    entry_confidence: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    profit_pct: Optional[float] = None
    rules_triggered: List[str] = field(default_factory=list)


class DecisionEngineStrategy(bt.Strategy):
    """
    Backtrader strategy that uses decision-engine rules.

    This strategy evaluates rules from the decision-engine on each bar
    and generates buy/sell signals based on rule confidence.
    """

    params = (
        ("rules", []),  # List of Rule instances
        ("min_confidence", 0.6),
        ("require_consensus", False),
        ("profit_target", 0.07),  # 7%
        ("stop_loss", 0.05),  # 5%
        ("log_trades", True),
        ("allow_scale_in", False),  # Allow averaging down
        ("max_scale_ins", 2),  # Max number of scale-ins
        ("scale_in_size", 0.5),  # Size of scale-in relative to initial (0.5 = half size)
        ("warmup_bars", 200),
    )

    def __init__(self):
        """Initialize the strategy."""
        self.rules: List[Rule] = self.params.rules
        self.order = None
        self.entry_price = None  # Initial entry price
        self.avg_cost_basis = None  # Average cost across all buys
        self.entry_date = None
        self.entry_reason = None
        self.entry_confidence = None
        self.entry_rules = []
        self.scale_in_count = 0  # Number of scale-ins done
        self.total_shares = 0  # Total shares held
        self.total_cost = 0  # Total cost basis

        self.bar_count = 0

        # Trade history
        self.trade_records: List[TradeRecord] = []
        self.current_trade: Optional[TradeRecord] = None

        logger.info(
            f"DecisionEngineStrategy initialized with {len(self.rules)} rules, "
            f"min_confidence={self.params.min_confidence}, "
            f"allow_scale_in={self.params.allow_scale_in}"
        )
        for rule in self.rules:
            logger.info(f"  - {rule.name}: {rule.description}")

    def log(self, txt, dt=None):
        """Log a message with timestamp."""
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.log_trades:
            logger.info(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                executed_price = order.executed.price
                executed_size = order.executed.size

                # Track cost basis for averaging
                self.total_shares += executed_size
                self.total_cost += executed_price * executed_size
                self.avg_cost_basis = self.total_cost / self.total_shares if self.total_shares > 0 else executed_price

                is_scale_in = self.entry_price is not None

                if is_scale_in:
                    self.scale_in_count += 1
                    self.log(
                        f"SCALE-IN #{self.scale_in_count} @ ${executed_price:.2f} "
                        f"(avg cost: ${self.avg_cost_basis:.2f}, confidence: {self.entry_confidence:.2f})"
                    )
                else:
                    self.entry_price = executed_price
                    self.entry_date = self.datas[0].datetime.datetime(0)
                    self.log(
                        f"BUY EXECUTED @ ${executed_price:.2f} "
                        f"(confidence: {self.entry_confidence:.2f})"
                    )
                self.log(f"  Reason: {self.entry_reason}")

                # Start trade record (only for initial entry)
                if not is_scale_in:
                    self.current_trade = TradeRecord(
                        symbol=self.datas[0]._name or "UNKNOWN",
                        entry_date=self.entry_date,
                        entry_price=self.entry_price,
                        entry_reason=self.entry_reason,
                        entry_confidence=self.entry_confidence,
                        rules_triggered=self.entry_rules.copy(),
                    )
            else:
                exit_price = order.executed.price
                cost_basis = self.avg_cost_basis or self.entry_price
                profit_pct = (exit_price - cost_basis) / cost_basis if cost_basis else 0

                bt_pnl = getattr(order.executed, 'pnl', None)
                if bt_pnl is not None and cost_basis and self.total_shares > 0:
                    bt_profit_pct = bt_pnl / (cost_basis * self.total_shares)
                    if abs(profit_pct - bt_profit_pct) > 0.01:
                        logger.warning(
                            f"P&L drift detected: custom={profit_pct:+.2%}, "
                            f"backtrader={bt_profit_pct:+.2%}, diff={abs(profit_pct - bt_profit_pct):.2%}"
                        )
                        profit_pct = bt_profit_pct

                self.log(
                    f"SELL EXECUTED @ ${exit_price:.2f} "
                    f"(vs avg cost ${cost_basis:.2f}: {profit_pct:+.1%})"
                )
                if self.scale_in_count > 0:
                    self.log(f"  Position had {self.scale_in_count} scale-in(s)")

                # Complete trade record
                if self.current_trade:
                    self.current_trade.exit_date = self.datas[0].datetime.datetime(0)
                    self.current_trade.exit_price = exit_price
                    self.current_trade.profit_pct = profit_pct
                    self.trade_records.append(self.current_trade)
                    self.current_trade = None

                # Reset all position tracking
                self.entry_price = None
                self.entry_date = None
                self.avg_cost_basis = None
                self.scale_in_count = 0
                self.total_shares = 0
                self.total_cost = 0

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return

        self.log(f"TRADE PROFIT: ${trade.pnl:.2f} ({trade.pnlcomm:.2f} after commission)")

    def _build_context(self) -> SymbolContext:
        """Build SymbolContext from current bar data."""
        # Get indicator values from data feed lines
        indicators = {}

        # Map data feed lines to indicator names
        if hasattr(self.datas[0], "rsi_14") and len(self.datas[0].rsi_14) > 0:
            val = self.datas[0].rsi_14[0]
            if math.isfinite(val):
                indicators["RSI_14"] = val

        if hasattr(self.datas[0], "sma_20") and len(self.datas[0].sma_20) > 0:
            val = self.datas[0].sma_20[0]
            if math.isfinite(val):
                indicators["SMA_20"] = val

        if hasattr(self.datas[0], "sma_50") and len(self.datas[0].sma_50) > 0:
            val = self.datas[0].sma_50[0]
            if math.isfinite(val):
                indicators["SMA_50"] = val

        if hasattr(self.datas[0], "sma_200") and len(self.datas[0].sma_200) > 0:
            val = self.datas[0].sma_200[0]
            if math.isfinite(val):
                indicators["SMA_200"] = val

        if hasattr(self.datas[0], "macd") and len(self.datas[0].macd) > 0:
            val = self.datas[0].macd[0]
            if math.isfinite(val):
                indicators["MACD"] = val

        if hasattr(self.datas[0], "macd_signal") and len(self.datas[0].macd_signal) > 0:
            val = self.datas[0].macd_signal[0]
            if math.isfinite(val):
                indicators["MACD_SIGNAL"] = val

        if hasattr(self.datas[0], "atr_14") and len(self.datas[0].atr_14) > 0:
            val = self.datas[0].atr_14[0]
            if math.isfinite(val):
                indicators["ATR_14"] = val

        if hasattr(self.datas[0], "macd_histogram") and len(self.datas[0].macd_histogram) > 0:
            val = self.datas[0].macd_histogram[0]
            if math.isfinite(val):
                indicators["MACD_HISTOGRAM"] = val

        if hasattr(self.datas[0], "bb_lower") and len(self.datas[0].bb_lower) > 0:
            val = self.datas[0].bb_lower[0]
            if math.isfinite(val):
                indicators["BB_LOWER"] = val

        if hasattr(self.datas[0], "bb_mid") and len(self.datas[0].bb_mid) > 0:
            val = self.datas[0].bb_mid[0]
            if math.isfinite(val):
                indicators["BB_MID"] = val

        if hasattr(self.datas[0], "bb_upper") and len(self.datas[0].bb_upper) > 0:
            val = self.datas[0].bb_upper[0]
            if math.isfinite(val):
                indicators["BB_UPPER"] = val

        if hasattr(self.datas[0], "bb_bandwidth") and len(self.datas[0].bb_bandwidth) > 0:
            val = self.datas[0].bb_bandwidth[0]
            if math.isfinite(val):
                indicators["BB_BANDWIDTH"] = val

        if hasattr(self.datas[0], "bb_percent") and len(self.datas[0].bb_percent) > 0:
            val = self.datas[0].bb_percent[0]
            if math.isfinite(val):
                indicators["BB_PERCENT"] = val

        # Add close and volume for enhanced rules
        if len(self.datas[0].close) > 0:
            indicators["close"] = self.datas[0].close[0]
        if len(self.datas[0].volume) > 0:
            indicators["volume"] = self.datas[0].volume[0]

        if hasattr(self.datas[0], "volume_sma_20") and len(self.datas[0].volume_sma_20) > 0:
            val = self.datas[0].volume_sma_20[0]
            if math.isfinite(val):
                indicators["volume_sma_20"] = val

        # Current position
        position = None
        if self.position.size > 0:
            position = "long"

        # Build metadata for scale-in rules
        metadata = {}
        if self.entry_price is not None:
            metadata["entry_price"] = self.entry_price
            metadata["avg_cost_basis"] = self.avg_cost_basis
            metadata["scale_in_count"] = self.scale_in_count

        return SymbolContext(
            symbol=self.datas[0]._name or "UNKNOWN",
            indicators=indicators,
            timestamp=self.datas[0].datetime.datetime(0),
            current_position=position,
            metadata=metadata,
        )

    def _evaluate_rules(self, context: SymbolContext) -> tuple:
        """
        Evaluate all rules and return aggregated signal.

        Returns:
            (signal_type, confidence, reason, triggered_rules)
        """
        buy_results = []
        sell_results = []

        for rule in self.rules:
            if not rule.can_evaluate(context):
                continue

            result = rule.evaluate(context)

            if result.triggered:
                if result.signal == SignalType.BUY:
                    buy_results.append((rule, result))
                elif result.signal == SignalType.SELL:
                    sell_results.append((rule, result))

        # Process buy signals
        if buy_results:
            avg_confidence = sum(r[1].confidence for r in buy_results) / len(buy_results)

            if self.params.require_consensus and len(buy_results) < 2:
                return (None, 0, None, [])

            if avg_confidence >= self.params.min_confidence:
                # Build reason from highest confidence rule
                best_rule, best_result = max(buy_results, key=lambda x: x[1].confidence)
                reason = best_result.reasoning
                rules = [r[0].name for r in buy_results]
                return (SignalType.BUY, avg_confidence, reason, rules)

        # Process sell signals
        if sell_results:
            avg_confidence = sum(r[1].confidence for r in sell_results) / len(sell_results)

            if self.params.require_consensus and len(sell_results) < 2:
                return (None, 0, None, [])

            if avg_confidence >= self.params.min_confidence:
                best_rule, best_result = max(sell_results, key=lambda x: x[1].confidence)
                reason = best_result.reasoning
                rules = [r[0].name for r in sell_results]
                return (SignalType.SELL, avg_confidence, reason, rules)

        return (None, 0, None, [])

    def next(self):
        """Process each bar."""
        self.bar_count += 1

        if self.bar_count <= self.params.warmup_bars:
            return

        # Skip if we have a pending order
        if self.order:
            return

        current_price = self.datas[0].close[0]

        # Use average cost basis for exit calculations (if available)
        cost_basis = self.avg_cost_basis or self.entry_price

        # Check exit conditions if in position
        if self.position.size > 0 and cost_basis:
            # Check profit target (based on average cost)
            if current_price >= cost_basis * (1 + self.params.profit_target):
                self.log(f"PROFIT TARGET reached @ ${current_price:.2f} (avg cost: ${cost_basis:.2f})")
                if self.current_trade:
                    self.current_trade.exit_reason = "Profit target"
                self.order = self.sell()
                return

            # Check stop loss (based on average cost)
            if current_price <= cost_basis * (1 - self.params.stop_loss):
                self.log(f"STOP LOSS triggered @ ${current_price:.2f} (avg cost: ${cost_basis:.2f})")
                if self.current_trade:
                    self.current_trade.exit_reason = "Stop loss"
                self.order = self.sell()
                return

        # Build context and evaluate rules
        context = self._build_context()
        signal_type, confidence, reason, rules = self._evaluate_rules(context)

        # Execute signals
        if signal_type == SignalType.BUY:
            # New position
            if self.position.size == 0:
                self.entry_reason = reason
                self.entry_confidence = confidence
                self.entry_rules = rules
                self.order = self.buy()

            # Scale-in (average down)
            elif self.params.allow_scale_in:
                # Debug: log why scale-in might not happen
                if self.scale_in_count >= self.params.max_scale_ins:
                    logger.debug(f"Scale-in skipped: max scale-ins reached ({self.scale_in_count})")
                elif "Average Down" not in rules:
                    logger.debug(f"Scale-in skipped: Average Down not in triggered rules: {rules}")
                else:
                    self.entry_reason = reason
                    self.entry_confidence = confidence
                    # Calculate scale-in size
                    scale_size = max(1, int(self.position.size * self.params.scale_in_size))
                    self.log(f"SCALE-IN signal: adding {scale_size} shares @ ${current_price:.2f}")
                    self.order = self.buy(size=scale_size)

        elif signal_type == SignalType.SELL and self.position.size > 0:
            if self.current_trade:
                self.current_trade.exit_reason = reason
            self.order = self.sell()

    def stop(self):
        """Called when backtest ends."""
        # Close any open position
        if self.position.size > 0:
            self.log("Closing open position at end of backtest")
            if self.current_trade:
                self.current_trade.exit_reason = "End of backtest"
                self.current_trade.exit_date = self.datas[0].datetime.datetime(0)
                self.current_trade.exit_price = self.datas[0].close[0]
                cost_basis = self.avg_cost_basis if self.avg_cost_basis else self.current_trade.entry_price
                profit_pct = (
                    self.current_trade.exit_price - cost_basis
                ) / cost_basis
                self.current_trade.profit_pct = profit_pct
                self.trade_records.append(self.current_trade)

        # Log summary
        if self.trade_records:
            wins = sum(1 for t in self.trade_records if t.profit_pct and t.profit_pct > 0)
            total = len(self.trade_records)
            win_rate = wins / total if total > 0 else 0
            self.log(f"Total trades: {total}, Win rate: {win_rate:.1%}")
