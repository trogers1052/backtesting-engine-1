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
        return size
