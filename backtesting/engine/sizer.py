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
