# base data provider class

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class DataProvider(ABC):
    # base class for data providers
    # all providers must implement fetch method

    def __init__(self, name: str):
        # initialize data provider
        # name: readable name for provider
        self.name = name

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        # fetch historical market data for symbol
        # symbol: ticker symbol
        # start_date: start date in 'yyyy-mm-dd' format
        # end_date: end date in 'yyyy-mm-dd' format
        # interval: data interval
        # returns: dataframe with columns: ['open', 'high', 'low', 'close', 'volume']
        # raises: valueerror if symbol is invalid
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        # validate fetched data format
        # data: dataframe to validate
        # returns: true if valid, false otherwise
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # check if all required columns exist
        if not all(col in data.columns for col in required_columns):
            return False

        # check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            return False

        # check for no data
        if len(data) == 0:
            return False

        return True

    def __repr__(self) -> str:
        # string representation of data provider
        return f"{self.__class__.__name__}(name='{self.name}')"
