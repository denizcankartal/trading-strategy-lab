# trade representation and tracking

import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class Trade:
    # represents a completed round-trip trade
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    side: str
    entry_commission: float = 0.0
    exit_commission: float = 0.0

    @property
    def holding_period(self) -> pd.Timedelta:
        # time between entry and exit
        return self.exit_date - self.entry_date

    @property
    def holding_days(self) -> int:
        # holding period in days
        return self.holding_period.days

    @property
    def entry_value(self) -> float:
        # total cost of entry with commission
        return self.shares * self.entry_price + self.entry_commission

    @property
    def exit_value(self) -> float:
        # total proceeds from exit minus commission
        return self.shares * self.exit_price - self.exit_commission

    @property
    def gross_pnl(self) -> float:
        # profit/loss before commissions
        if self.side == 'long':
            return self.shares * (self.exit_price - self.entry_price)
        else:
            return self.shares * (self.entry_price - self.exit_price)

    @property
    def commission_paid(self) -> float:
        # total commission paid
        return self.entry_commission + self.exit_commission

    @property
    def net_pnl(self) -> float:
        # profit/loss after commissions
        return self.gross_pnl - self.commission_paid

    @property
    def return_pct(self) -> float:
        # return as percentage of entry value
        if self.entry_value == 0:
            return 0.0
        return self.net_pnl / self.entry_value

    @property
    def is_winner(self) -> bool:
        # true if trade was profitable
        return self.net_pnl > 0

    def to_dict(self) -> dict:
        # convert trade to dictionary
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'holding_days': self.holding_days,
            'entry_value': self.entry_value,
            'exit_value': self.exit_value,
            'gross_pnl': self.gross_pnl,
            'commission_paid': self.commission_paid,
            'net_pnl': self.net_pnl,
            'return_pct': self.return_pct,
            'is_winner': self.is_winner
        }

    def __repr__(self) -> str:
        # string representation of trade
        return (
            f"Trade({self.symbol} {self.side.upper()}: "
            f"{self.entry_date.date()} -> {self.exit_date.date()}, "
            f"P&L: ${self.net_pnl:,.2f} ({self.return_pct:.2%}))"
        )

@dataclass
class BacktestResult:
    # results from a backtest run
    strategy_name: str
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict

    @property
    def total_return(self) -> float:
        # total return as decimal
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_return_pct(self) -> float:
        # total return as percentage
        return self.total_return

    @property
    def num_trades(self) -> int:
        # total number of trades
        return len(self.trades)

    @property
    def winning_trades(self) -> list[Trade]:
        # list of winning trades
        return [t for t in self.trades if t.is_winner]

    @property
    def losing_trades(self) -> list[Trade]:
        # list of losing trades
        return [t for t in self.trades if not t.is_winner]

    @property
    def num_winners(self) -> int:
        # number of winning trades
        return len(self.winning_trades)

    @property
    def num_losers(self) -> int:
        # number of losing trades
        return len(self.losing_trades)

    @property
    def win_rate(self) -> float:
        # win rate as decimal
        if self.num_trades == 0:
            return 0.0
        return self.num_winners / self.num_trades

    @property
    def avg_win(self) -> float:
        # average profit from winning trades
        if not self.winning_trades:
            return 0.0
        return sum(t.net_pnl for t in self.winning_trades) / len(self.winning_trades)

    @property
    def avg_loss(self) -> float:
        # average loss from losing trades
        if not self.losing_trades:
            return 0.0
        return sum(t.net_pnl for t in self.losing_trades) / len(self.losing_trades)

    @property
    def profit_factor(self) -> float:
        # profit factor ratio
        gross_wins = sum(t.net_pnl for t in self.winning_trades)
        gross_losses = abs(sum(t.net_pnl for t in self.losing_trades))

        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0

        return gross_wins / gross_losses

    @property
    def avg_holding_period(self) -> float:
        # average holding period in days
        if not self.trades:
            return 0.0
        return sum(t.holding_days for t in self.trades) / len(self.trades)

    def get_trades_df(self) -> pd.DataFrame:
        # get trades as dataframe with one row per trade
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.trades])

    def print_summary(self):
        # print formatted summary of backtest results
        print("=" * 60)
        print(f"backtest results: {self.strategy_name}")
        print("=" * 60)
        print(f"symbol:          {self.symbol}")
        print(f"period:          {self.start_date.date()} to {self.end_date.date()}")
        print(f"duration:        {(self.end_date - self.start_date).days} days")
        print()
        print("performance")
        print("-" * 60)
        print(f"initial capital: ${self.initial_capital:,.2f}")
        print(f"final capital:   ${self.final_capital:,.2f}")
        print(f"total return:    {self.total_return_pct:>8.2%}")
        print()
        print("metrics")
        print("-" * 60)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if 'ratio' in key or key == 'win_rate':
                    print(f"{key:20s}: {value:>8.2f}")
                else:
                    print(f"{key:20s}: {value:>8.2%}")
            else:
                print(f"{key:20s}: {value:>8}")
        print()
        print("trades")
        print("-" * 60)
        print(f"total trades:    {self.num_trades}")
        print(f"winners:         {self.num_winners} ({self.win_rate:.1%})")
        print(f"losers:          {self.num_losers}")
        print(f"profit factor:   {self.profit_factor:.2f}")
        print(f"avg win:         ${self.avg_win:,.2f}")
        print(f"avg loss:        ${self.avg_loss:,.2f}")
        print(f"avg hold period: {self.avg_holding_period:.1f} days")
        print("=" * 60)

    def __repr__(self) -> str:
        # string representation of backtest result
        return (
            f"BacktestResult({self.strategy_name}, "
            f"return={self.total_return_pct:.2%}, "
            f"trades={self.num_trades})"
        )
