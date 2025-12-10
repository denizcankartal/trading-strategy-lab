from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

# base class for all trading strategies
class Strategy(ABC):

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # generate trading signals from market data
        # data: dataframe with ohlcv columns and datetime index
        # returns: series with 1 (buy), -1 (sell), 0 (hold)
        pass

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params_str})"
