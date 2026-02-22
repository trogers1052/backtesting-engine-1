"""
Position Sizers for Backtrader

Custom sizers to control position sizing.
"""

import backtrader as bt


class PercentSizer(bt.Sizer):
    """
    Size positions as a percentage of available cash.

    Parameters:
        percents: Percentage of cash to use (default: 95%)
    """

    params = (("percents", 95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # Return full position for sells
            position = self.broker.getposition(data)
            return position.size

        # Calculate size based on percentage of available cash
        price = data.close[0]
        if price <= 0:
            return 0

        available = cash * (self.params.percents / 100.0)
        size = int(available / price)
        if size <= 0:
            return 0

        return size


class CompoundingSizer(bt.Sizer):
    """
    Size positions as a percentage of TOTAL portfolio value (cash + positions).

    This ensures profits are automatically reinvested into larger positions.
    As your portfolio grows, position sizes grow proportionally.

    Parameters:
        percents: Percentage of portfolio to use (default: 95%)
    """

    params = (("percents", 95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # Return full position for sells
            position = self.broker.getposition(data)
            return position.size

        price = data.close[0]
        if price <= 0:
            return 0

        # Use total portfolio value (cash + unrealized P&L) for compounding
        portfolio_value = self.broker.getvalue()
        available = portfolio_value * (self.params.percents / 100.0)

        # Don't exceed available cash
        available = min(available, cash * 0.99)

        size = int(available / price)
        if size <= 0:
            return 0
        return size


class FixedPercentSizer(bt.Sizer):
    """
    Size positions as a percentage of INITIAL capital (no compounding).

    Position sizes stay constant regardless of profits/losses.
    Useful for comparing strategy performance without compounding effects.

    Parameters:
        percents: Percentage of initial capital to use (default: 95%)
        initial_capital: Starting capital (set by strategy)
    """

    params = (
        ("percents", 95),
        ("initial_capital", 100000),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            position = self.broker.getposition(data)
            return position.size

        price = data.close[0]
        if price <= 0:
            return 0

        # Always use initial capital for sizing (no compounding)
        available = self.params.initial_capital * (self.params.percents / 100.0)

        # Don't exceed available cash
        available = min(available, cash * 0.99)

        size = int(available / price)
        if size <= 0:
            return 0
        return size


class RiskBasedSizer(bt.Sizer):
    """
    Size positions based on risk per trade.

    Calculates position size so that if the stop loss is hit, the maximum loss
    equals risk_pct % of portfolio value. This matches production sizing logic.

    The strategy must set self._current_stop_price before calling buy().
    The sizer reads it via self.strategy._current_stop_price.

    Parameters:
        risk_pct: Max % of portfolio to risk per trade (default: 5%)
        max_position_pct: Max % of portfolio in a single position (default: 20%)
    """

    params = (
        ("risk_pct", 5.0),
        ("max_position_pct", 20.0),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            position = self.broker.getposition(data)
            return position.size

        price = data.close[0]
        if price <= 0:
            return 0

        # Read stop price from strategy bridge variable
        stop_price = getattr(self.strategy, "_current_stop_price", None)

        if stop_price is None or stop_price <= 0 or stop_price >= price:
            # Fallback: use 95% of cash if no valid stop price
            import logging
            logging.getLogger(__name__).warning(
                "RiskBasedSizer: no valid stop price on strategy, "
                "falling back to 95%% of cash"
            )
            available = cash * 0.95
            return int(available / price)

        portfolio_value = self.broker.getvalue()
        risk_per_share = price - stop_price

        # Max shares by risk budget
        max_dollar_risk = portfolio_value * (self.params.risk_pct / 100.0)
        max_shares_by_risk = int(max_dollar_risk / risk_per_share)

        # Max shares by position size limit
        max_position_value = portfolio_value * (self.params.max_position_pct / 100.0)
        max_shares_by_position = int(max_position_value / price)

        # Take the minimum of risk and position limits
        size = min(max_shares_by_risk, max_shares_by_position)

        # Don't exceed available cash
        max_shares_by_cash = int(cash * 0.99 / price)
        size = min(size, max_shares_by_cash)

        if size <= 0:
            return 0

        return size


class FixedCashSizer(bt.Sizer):
    """
    Size positions with a fixed cash amount.

    Parameters:
        cash: Fixed cash amount per trade
    """

    params = (("cash", 10000),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            position = self.broker.getposition(data)
            return position.size

        price = data.close[0]
        if price <= 0:
            return 0

        size = int(self.params.cash / price)
        if size <= 0:
            return 0
        return size
