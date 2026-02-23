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
        ("stop_mode", "fixed"),  # "fixed" or "atr"
        ("atr_multiplier", 2.0),  # ATR multiplier for stop calculation
        ("atr_stop_min_pct", 3.0),  # Minimum stop distance %
        ("atr_stop_max_pct", 15.0),  # Maximum stop distance %
        ("max_price_extension_pct", 15.0),  # Skip buy if price > X% above SMA_20
        ("cooldown_bars", 5),  # Wait N bars after exit before re-entering
        ("max_trend_spread_pct", 20.0),  # Skip buy if SMA_20/SMA_50 spread > X%
        ("max_loss_pct", 10.0),  # Force exit if trade down > X% (gap-down protection)
        ("exit_timeframe", None),  # Intraday timeframe for multi-TF mode
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

        # Bridge variable for RiskBasedSizer — set before buy(), read by sizer
        self._current_stop_price = None
        # Stored stop price for the current position (used for ATR-based exits)
        self._entry_stop_price = None
        # Cooldown tracking — bar number of last exit
        self._last_exit_bar = None

        # Multi-timeframe tracking
        self._last_daily_len = 0  # Detect new daily bars via len(datas[1])
        self._daily_bar_count = 0  # Count of daily bars for cooldown
        self._last_exit_daily_bar = None  # Daily-bar cooldown in multi-TF mode
        self._new_daily_bar = False  # True on the first 5-min bar of a new day

        # Post-profit cooldown: consecutive profit target exits double cooldown
        self._consecutive_profit_targets = 0

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

    def _is_multi_timeframe(self):
        """True when running with dual data feeds (intraday + daily)."""
        return len(self.datas) > 1 and self.params.exit_timeframe is not None

    @property
    def _daily_feed(self):
        """Data feed for daily-scale indicators (SMAs, BB, ATR)."""
        return self.datas[1] if self._is_multi_timeframe() else self.datas[0]

    def log(self, txt, dt=None):
        """Log a message with timestamp."""
        if self._is_multi_timeframe():
            dt = dt or self.datas[0].datetime.datetime(0)
        else:
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

                # Track consecutive profit target exits for extended cooldown
                exit_reason = self.trade_records[-1].exit_reason if self.trade_records else None
                if exit_reason == "Profit target":
                    self._consecutive_profit_targets += 1
                else:
                    self._consecutive_profit_targets = 0

                # Reset all position tracking
                self.entry_price = None
                self.entry_date = None
                self.avg_cost_basis = None
                self.scale_in_count = 0
                self.total_shares = 0
                self.total_cost = 0
                self._current_stop_price = None
                self._entry_stop_price = None
                self._last_exit_bar = self.bar_count
                self._last_exit_daily_bar = self._daily_bar_count

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return

        self.log(f"TRADE PROFIT: ${trade.pnl:.2f} ({trade.pnlcomm:.2f} after commission)")

    def _read_indicator(self, feed, attr):
        """Read a single indicator value from a data feed, or None if invalid."""
        if hasattr(feed, attr) and len(getattr(feed, attr)) > 0:
            val = getattr(feed, attr)[0]
            if math.isfinite(val):
                return val
        return None

    def _build_context(self) -> SymbolContext:
        """Build SymbolContext from current bar data.

        In multi-timeframe mode, ALL indicators come from the daily feed.
        Entry signals evaluate on daily bars only (daily RSI/MACD = meaningful).
        The 5-min feed is only used for intraday exit price checks (not context).
        """
        indicators = {}
        daily = self._daily_feed

        # --- ALL indicators from daily feed ---
        for attr, key in [
            ("sma_20", "SMA_20"),
            ("sma_50", "SMA_50"),
            ("sma_200", "SMA_200"),
            ("atr_14", "ATR_14"),
            ("bb_lower", "BB_LOWER"),
            ("bb_mid", "BB_MID"),
            ("bb_upper", "BB_UPPER"),
            ("bb_bandwidth", "BB_BANDWIDTH"),
            ("bb_percent", "BB_PERCENT"),
            ("volume_sma_20", "volume_sma_20"),
            ("rsi_14", "RSI_14"),
            ("macd", "MACD"),
            ("macd_signal", "MACD_SIGNAL"),
            ("macd_histogram", "MACD_HISTOGRAM"),
        ]:
            val = self._read_indicator(daily, attr)
            if val is not None:
                indicators[key] = val

        # Close and volume from daily feed (rules see end-of-day values)
        if len(daily.close) > 0:
            indicators["close"] = daily.close[0]
        if len(daily.volume) > 0:
            indicators["volume"] = daily.volume[0]

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

    def _calculate_stop_price(self, entry_price: float) -> float:
        """Calculate stop price based on stop_mode.

        For ATR mode, matches production TradePlanEngine logic:
        stop = entry - (ATR * multiplier) with min/max % guards.
        Uses daily ATR in multi-TF mode for proper daily-scale stops.
        """
        if self.params.stop_mode == "atr":
            atr = self._read_indicator(self._daily_feed, "atr_14")
            if atr is not None and atr > 0:
                raw_stop = entry_price - (atr * self.params.atr_multiplier)
                stop_pct = (entry_price - raw_stop) / entry_price * 100

                if stop_pct < self.params.atr_stop_min_pct:
                    # Stop too tight — widen to min + 1%
                    floor_pct = self.params.atr_stop_min_pct + 1.0
                    raw_stop = entry_price * (1 - floor_pct / 100)
                elif stop_pct > self.params.atr_stop_max_pct:
                    # Stop too wide — cap to 10%
                    raw_stop = entry_price * 0.90

                return raw_stop

        # Fixed mode (or ATR unavailable): use stop_loss param
        return entry_price * (1 - self.params.stop_loss)

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
        """Process each bar.

        In multi-TF mode: daily entries + 5-min exits.
        - Every 5-min bar: check protective exits (profit target, stop loss, max loss)
        - On new daily bar only: evaluate entry/sell rules using fully daily context
        This gives daily-quality entry signals + 5-min exit precision.
        """
        self.bar_count += 1

        # Track daily bar count for multi-TF cooldown
        self._new_daily_bar = False
        if self._is_multi_timeframe():
            daily_len = len(self.datas[1])
            if daily_len > self._last_daily_len:
                self._daily_bar_count += (daily_len - self._last_daily_len)
                self._last_daily_len = daily_len
                self._new_daily_bar = True

            # Warmup: wait for daily feed to have enough bars
            if daily_len < self.params.warmup_bars:
                return
        else:
            if self.bar_count <= self.params.warmup_bars:
                return

        # Skip if we have a pending order
        if self.order:
            return

        current_price = self.datas[0].close[0]

        # Use average cost basis for exit calculations (if available)
        cost_basis = self.avg_cost_basis or self.entry_price

        # === EXITS: Check every 5-min bar (intraday precision) ===
        if self.position.size > 0 and cost_basis:
            # Check profit target (based on average cost)
            if current_price >= cost_basis * (1 + self.params.profit_target):
                self.log(f"PROFIT TARGET reached @ ${current_price:.2f} (avg cost: ${cost_basis:.2f})")
                if self.current_trade:
                    self.current_trade.exit_reason = "Profit target"
                self.order = self.sell()
                return

            # Check stop loss
            if self._entry_stop_price is not None:
                # Use the ATR-based stop price calculated at entry
                if current_price <= self._entry_stop_price:
                    self.log(
                        f"STOP LOSS triggered @ ${current_price:.2f} "
                        f"(stop: ${self._entry_stop_price:.2f}, avg cost: ${cost_basis:.2f})"
                    )
                    if self.current_trade:
                        self.current_trade.exit_reason = "Stop loss"
                    self.order = self.sell()
                    return
            else:
                # Fallback: fixed % stop from average cost
                if current_price <= cost_basis * (1 - self.params.stop_loss):
                    self.log(f"STOP LOSS triggered @ ${current_price:.2f} (avg cost: ${cost_basis:.2f})")
                    if self.current_trade:
                        self.current_trade.exit_reason = "Stop loss"
                    self.order = self.sell()
                    return

            # Max loss cap — force exit if gap blew through stop
            loss_pct = (cost_basis - current_price) / cost_basis * 100
            if loss_pct >= self.params.max_loss_pct:
                self.log(
                    f"MAX LOSS CAP triggered @ ${current_price:.2f} "
                    f"(-{loss_pct:.1f}% vs avg cost ${cost_basis:.2f}, cap: {self.params.max_loss_pct}%)"
                )
                if self.current_trade:
                    self.current_trade.exit_reason = f"Max loss cap ({self.params.max_loss_pct}%)"
                self.order = self.sell()
                return

        # === ENTRIES + RULE-BASED SELLS: Daily bars only in multi-TF ===
        # In multi-TF mode, only evaluate rules when a new daily bar forms.
        # This gives daily-quality signals (daily RSI/MACD are meaningful)
        # while exits above still fire on every 5-min bar.
        if self._is_multi_timeframe() and not self._new_daily_bar:
            return  # Wait for next daily bar to evaluate rules

        # Pre-buy filters (only apply when not in a position)
        if self.position.size == 0:
            # Use daily feed for SMA-based filters
            daily = self._daily_feed
            sma20 = self._read_indicator(daily, "sma_20")

            # Filter 1: Price extension — skip if price too far above SMA_20
            if sma20 is not None and sma20 > 0:
                extension = (current_price - sma20) / sma20 * 100
                if extension > self.params.max_price_extension_pct:
                    return  # Price too extended above SMA_20

                # Filter 1b: Downside extension — skip if price >3% below SMA_20
                # Buying below the moving average is catching a falling knife
                if extension < -3.0:
                    return  # Price too far below SMA_20, falling knife

            # Filter 2: Cooldown — wait N bars after last exit
            # Double cooldown after 2+ consecutive profit targets to avoid
            # re-entering at the top after a streak of wins
            cooldown = self.params.cooldown_bars
            if self._consecutive_profit_targets >= 2:
                cooldown = cooldown * 2

            if self._is_multi_timeframe():
                # Cooldown in daily bars (not intraday bars)
                if self._last_exit_daily_bar is not None and cooldown > 0:
                    daily_bars_since_exit = self._daily_bar_count - self._last_exit_daily_bar
                    if daily_bars_since_exit <= cooldown:
                        return  # Still in cooldown after last exit
            else:
                if self._last_exit_bar is not None and cooldown > 0:
                    bars_since_exit = self.bar_count - self._last_exit_bar
                    if bars_since_exit <= cooldown:
                        return  # Still in cooldown after last exit

            # Filter 3: Trend maturity — skip if SMA_20/SMA_50 spread too wide
            sma50 = self._read_indicator(daily, "sma_50")
            if (sma20 is not None and sma20 > 0 and
                    sma50 is not None and sma50 > 0):
                trend_spread = (sma20 - sma50) / sma50 * 100
                if trend_spread > self.params.max_trend_spread_pct:
                    return  # Trend is late-stage, high reversion risk

        # Build context and evaluate rules (all daily indicators)
        context = self._build_context()
        signal_type, confidence, reason, rules = self._evaluate_rules(context)

        # Execute signals
        if signal_type == SignalType.BUY:
            # New position
            if self.position.size == 0:
                self.entry_reason = reason
                self.entry_confidence = confidence
                self.entry_rules = rules
                # Calculate stop price and set bridge variable for sizer
                self._current_stop_price = self._calculate_stop_price(current_price)
                self._entry_stop_price = self._current_stop_price
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
