# data providers for fetching market data

from typing import Optional
import pandas as pd
import yfinance as yf
from .base import DataProvider

class YahooFinanceProvider(DataProvider):
    # yahoo finance data provider

    def __init__(self):
        # initialize yahoo finance provider
        super().__init__(name="Yahoo Finance")

    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        # fetch historical market data from yahoo finance
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )

            if data.empty:
                raise ValueError(
                    f"no data retrieved for symbol '{symbol}', "
                    f"check if symbol is valid"
                )

            data.columns = data.columns.str.lower()
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            data = data[required_columns]

            if not self.validate_data(data):
                raise ValueError(
                    f"Data validation failed for symbol '{symbol}'"
                )

            return data

        except Exception as e:
            raise ValueError(
                f"failed to fetch data for '{symbol}' from yahoo finance: {str(e)}"
            )

    def fetch_multiple(
        self,
        symbols: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> dict:
        # fetch data for multiple symbols
        results = {}

        for symbol in symbols:
            try:
                data = self.fetch(symbol, start_date, end_date, interval)
                results[symbol] = data
            except ValueError as e:
                print(f"warning: could not fetch {symbol}: {str(e)}")
                results[symbol] = None

        return results

    def get_info(self, symbol: str) -> dict:
        # get company info for symbol
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            raise ValueError(f"failed to get info for '{symbol}': {str(e)}")


class CSVDataProvider(DataProvider):
    # csv data provider for loading from files

    def __init__(self):
        # initialize csv data provider
        super().__init__(name="CSV File")

    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        # load historical data from csv file
        try:
            data = pd.read_csv(symbol)

            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns.str.lower()]

            if missing_columns:
                raise ValueError(
                    f"csv file missing required columns: {missing_columns}, "
                    f"required: {required_columns}"
                )

            data.columns = data.columns.str.lower()
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)

            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]

            data = data[['open', 'high', 'low', 'close', 'volume']]

            if not self.validate_data(data):
                raise ValueError("data validation failed")

            if data.empty:
                raise ValueError(
                    f"no data available for specified date range, "
                    f"start: {start_date}, end: {end_date}"
                )

            return data

        except FileNotFoundError:
            raise ValueError(f"csv file not found: {symbol}")
        except Exception as e:
            raise ValueError(f"failed to load csv file '{symbol}': {str(e)}")
