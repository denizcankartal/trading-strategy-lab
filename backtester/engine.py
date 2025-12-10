import pandas as pd
from typing import Optional
from strategies.base import Strategy
from .portfolio import Portfolio
from .trade import Trade, BacktestResult
from metrics import calculate_all_metrics

# execute strategies on historical data with realistic costs
class Backtester:

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct

        self.portfolio: Optional[Portfolio] = None
        self.trades: list[Trade] = []
        self.open_trade_entry: dict = {}

    def calculate_shares(self, price: float, available_capital: float) -> float:
        # max shares affordable with commission
        if price <= 0 or available_capital <= 0:
            return 0
        max_shares = available_capital / (price * (1 + self.commission_pct))
        return int(max_shares)

    def calculate_commission(self, shares: float, price: float) -> float:
        # commission cost for trade
        trade_value = shares * price
        return trade_value * self.commission_pct

    def apply_slippage(self, price: float, side: str) -> float:
        # add slippage: pay more when buying, receive less when selling
        if side == 'buy':
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> BacktestResult:
        # run backtest on historical data
        if data.empty:
            raise ValueError("data cannot be empty")
        if 'close' not in data.columns:
            raise ValueError("data must contain 'close' column")

        self.portfolio = Portfolio(self.initial_capital)
        self.trades = []
        self.open_trade_entry = {}

        symbol = symbol or 'UNKNOWN'
        signals = strategy.generate_signals(data)

        for date, row in data.iterrows():
            current_price = row['close']
            signal = signals.loc[date] if date in signals.index else 0
            current_prices = {symbol: current_price}

            if signal == 1:
                self._process_buy_signal(symbol, date, current_price)
            elif signal == -1:
                self._process_sell_signal(symbol, date, current_price)

            self.portfolio.update_history(date, current_prices)

        # close remaining positions
        if self.portfolio.has_position(symbol):
            final_price = data.iloc[-1]['close']
            final_date = data.index[-1]
            self._process_sell_signal(symbol, final_date, final_price)

        returns = self.portfolio.get_returns()
        metrics = calculate_all_metrics(returns) if len(returns) > 0 else {}

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self.portfolio.total_value,
            equity_curve=self.portfolio.get_equity_curve(),
            trades=self.trades,
            metrics=metrics
        )

        return result

    def _process_buy_signal(self, symbol: str, date: pd.Timestamp, price: float):
        # process buy signal
        if self.portfolio.has_position(symbol):
            return

        execution_price = self.apply_slippage(price, 'buy')
        available_capital = self.portfolio.cash * self.position_size_pct
        shares = self.calculate_shares(execution_price, available_capital)

        if shares <= 0:
            return

        commission = self.calculate_commission(shares, execution_price)
        success = self.portfolio.buy(
            symbol=symbol,
            shares=shares,
            price=execution_price,
            date=date,
            commission=commission
        )

        if success:
            self.open_trade_entry[symbol] = {
                'entry_date': date,
                'entry_price': execution_price,
                'shares': shares,
                'entry_commission': commission
            }

    def _process_sell_signal(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float
    ):
        # process sell signal
        if not self.portfolio.has_position(symbol):
            return

        execution_price = self.apply_slippage(price, 'sell')
        position = self.portfolio.get_position(symbol)
        shares = position.shares
        commission = self.calculate_commission(shares, execution_price)

        realized_pnl = self.portfolio.sell(
            symbol=symbol,
            shares=None,
            price=execution_price,
            date=date,
            commission=commission
        )

        if symbol in self.open_trade_entry:
            entry_info = self.open_trade_entry[symbol]
            trade = Trade(
                symbol=symbol,
                entry_date=entry_info['entry_date'],
                entry_price=entry_info['entry_price'],
                exit_date=date,
                exit_price=execution_price,
                shares=shares,
                side='long',
                entry_commission=entry_info['entry_commission'],
                exit_commission=commission
            )
            self.trades.append(trade)
            del self.open_trade_entry[symbol]

    def run_multiple(
        self,
        strategies: list[Strategy],
        data: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> dict[str, BacktestResult]:
        # run backtest for multiple strategies
        results = {}
        for strategy in strategies:
            result = self.run(strategy, data, symbol)
            results[strategy.name] = result
        return results

    def walk_forward_analysis(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 63,
        symbol: Optional[str] = None
    ) -> list[BacktestResult]:
        # split data into rolling windows and backtest on each
        results = []
        start_idx = 0

        while start_idx + train_size + test_size <= len(data):
            train_end = start_idx + train_size
            test_end = train_end + test_size
            test_data = data.iloc[train_end:test_end]

            result = self.run(strategy, test_data, symbol)
            results.append(result)

            start_idx += step_size

        return results

# fetch data and run backtest in one call
def quick_backtest(
    strategy: Strategy,
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    initial_capital: float = 100000
) -> BacktestResult:
    from data import quick_load

    data = quick_load(symbol, start_date=start_date, end_date=end_date)
    backtester = Backtester(initial_capital=initial_capital)
    result = backtester.run(strategy, data, symbol)

    return result
