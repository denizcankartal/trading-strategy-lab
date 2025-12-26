# data providers for fetching crypto market data

from typing import Optional
import pandas as pd
import ccxt
from .base import DataProvider


class BinanceProvider(DataProvider):
    # binance crypto data provider using ccxt

    def __init__(self):
        super().__init__(name="Binance")
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        # fetch ohlcv data from binance
        # symbol: trading pair like 'BTC/USDT'
        # start_date: start date 'yyyy-mm-dd'
        # end_date: end date 'yyyy-mm-dd'
        # timeframe: '1m', '5m', '15m', '1h', '4h', '1d', '1w'
        # returns: dataframe with ohlcv columns and datetime index

        try:
            # convert dates to timestamps if provided
            since = None
            if start_date:
                since = self.exchange.parse8601(f"{start_date}T00:00:00Z")

            # fetch ohlcv data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=1000  # max per request
            )

            if not ohlcv:
                raise ValueError(f"no data retrieved for {symbol}")

            # convert to dataframe
            data = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data.index.name = 'date'

            # filter by end date if provided
            if end_date:
                end_ts = pd.Timestamp(end_date)
                data = data[data.index <= end_ts]

            # validate
            if not self.validate_data(data):
                raise ValueError(f"data validation failed for {symbol}")

            return data

        except ccxt.NetworkError as e:
            raise ValueError(f"network error fetching {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"exchange error fetching {symbol}: {str(e)}")
        except Exception as e:
            raise ValueError(f"failed to fetch {symbol}: {str(e)}")

    def get_available_symbols(self) -> list:
        # get list of available trading pairs
        try:
            markets = self.exchange.load_markets()
            return [symbol for symbol in markets.keys() if '/USDT' in symbol]
        except Exception as e:
            raise ValueError(f"failed to get symbols: {str(e)}")


# convenience function for quick data fetching
def fetch_crypto(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = '1d'
) -> pd.DataFrame:
    # fetch crypto data from binance
    # simple wrapper for quick usage
    provider = BinanceProvider()
    return provider.fetch(symbol, start_date, end_date, timeframe)
