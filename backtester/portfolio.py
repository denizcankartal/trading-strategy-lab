# portfolio management for positions, cash, and equity

import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class Position:
    # represents a position in the portfolio
    symbol: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp

    @property
    def cost_basis(self) -> float:
        # total cost of position
        return self.shares * self.entry_price

    def current_value(self, current_price: float) -> float:
        # current market value
        return self.shares * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        # unrealized profit/loss
        return self.current_value(current_price) - self.cost_basis

    def unrealized_pnl_pct(self, current_price: float) -> float:
        # unrealized profit/loss as percentage
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.cost_basis

# portfolio tracker for backtesting positions and cash
class Portfolio:

    def __init__(self, initial_cash: float = 100000):
        # initialize with starting cash balance
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}

        self.equity_history = []
        self.cash_history = []
        self.position_value_history = []
        self.dates = []

    @property
    def position_value(self) -> float:
        # total value of all positions at cost basis
        return sum(pos.cost_basis for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        # total portfolio value at cost basis
        return self.cash + self.position_value

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        # total portfolio value using current prices
        position_value = sum(
            pos.current_value(current_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def has_position(self, symbol: str) -> bool:
        # check if position exists and has shares
        return symbol in self.positions and self.positions[symbol].shares > 0

    def get_position(self, symbol: str) -> Optional[Position]:
        # get position or none if not exists
        return self.positions.get(symbol)

    def buy(
        self,
        symbol: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
        commission: float = 0.0
    ) -> bool:
        # buy shares, open or add to position
        cost = shares * price + commission

        if cost > self.cash:
            return False

        self.cash -= cost

        if symbol in self.positions:
            existing = self.positions[symbol]
            total_shares = existing.shares + shares
            total_cost = existing.cost_basis + (shares * price)
            avg_price = total_cost / total_shares

            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=existing.entry_date
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                entry_date=date
            )

        return True

    def sell(
        self,
        symbol: str,
        shares: Optional[float],
        price: float,
        date: pd.Timestamp,
        commission: float = 0.0
    ) -> Optional[float]:
        # sell shares, close or reduce position
        if not self.has_position(symbol):
            return None

        position = self.positions[symbol]
        shares_to_sell = shares if shares is not None else position.shares
        shares_to_sell = min(shares_to_sell, position.shares)

        proceeds = shares_to_sell * price - commission
        cost_basis_sold = shares_to_sell * position.entry_price
        realized_pnl = proceeds - cost_basis_sold

        self.cash += proceeds

        if shares_to_sell >= position.shares:
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=position.shares - shares_to_sell,
                entry_price=position.entry_price,
                entry_date=position.entry_date
            )

        return realized_pnl

    def update_history(self, date: pd.Timestamp, current_prices: Dict[str, float]):
        # record current portfolio state in history
        position_value = sum(
            pos.current_value(current_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )

        total_value = self.cash + position_value

        self.dates.append(date)
        self.equity_history.append(total_value)
        self.cash_history.append(self.cash)
        self.position_value_history.append(position_value)

    def get_equity_curve(self) -> pd.Series:
        # get equity curve as pandas series with dates as index
        if not self.dates:
            return pd.Series(dtype=float)

        return pd.Series(self.equity_history, index=self.dates, name='equity')

    def get_returns(self) -> pd.Series:
        # calculate period-over-period returns from equity curve
        equity = self.get_equity_curve()
        return equity.pct_change().dropna()

    def get_summary(self, current_prices: Dict[str, float]) -> dict:
        # get portfolio summary statistics
        total_value = self.get_total_value(current_prices)
        total_return = (total_value - self.initial_cash) / self.initial_cash

        position_summaries = []
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos.entry_price)
            position_summaries.append({
                'symbol': symbol,
                'shares': pos.shares,
                'entry_price': pos.entry_price,
                'current_price': current_price,
                'cost_basis': pos.cost_basis,
                'current_value': pos.current_value(current_price),
                'unrealized_pnl': pos.unrealized_pnl(current_price),
                'unrealized_pnl_pct': pos.unrealized_pnl_pct(current_price),
                'entry_date': pos.entry_date
            })

        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'position_value': sum(p['current_value'] for p in position_summaries),
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return,
            'num_positions': len(self.positions),
            'positions': position_summaries
        }

    def __repr__(self) -> str:
        # string representation of portfolio
        return (
            f"Portfolio(cash=${self.cash:,.2f}, "
            f"positions={len(self.positions)}, "
            f"value=${self.total_value:,.2f})"
        )
