import pandas as pd
from .base import Strategy

# ma crossover strategy
# buy when fast ma crosses above slow ma
# sell when fast ma crosses below slow ma
class MovingAverageCrossover(Strategy):

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be positive")
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        super().__init__(
            name="Moving Average Crossover",
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        )
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # generate buy/sell signals based on ma crossover
        if 'close' not in data.columns:
            raise ValueError("data must contain 'close' column")

        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()

        signals = pd.Series(0, index=data.index)

        # create position series: 1 when fast > slow, -1 otherwise
        position = pd.Series(0, index=data.index)
        position[fast_ma > slow_ma] = 1
        position[fast_ma < slow_ma] = -1

        # convert positions to signals (only on changes)
        signals = position.diff()

        # normalize to -1, 0, 1
        signals[signals > 0] = 1
        signals[signals < 0] = -1

        return signals
