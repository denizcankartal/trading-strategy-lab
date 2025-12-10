# data loading with caching support

import os
import pickle
from typing import Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from .base import DataProvider
from .providers import YahooFinanceProvider

class DataLoader:
    # data loader with disk caching to avoid repeated api calls

    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        cache_dir: str = '.data_cache'
    ):
        # initialize with optional provider and cache directory
        self.provider = provider or YahooFinanceProvider()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
        use_cache: bool = True,
        cache_expiry_days: int = 1
    ) -> pd.DataFrame:
        # load data with optional caching
        cache_file = self._get_cache_filename(symbol, start_date, end_date, interval)

        if use_cache and self._is_cache_valid(cache_file, cache_expiry_days):
            try:
                data = self._load_from_cache(cache_file)
                print(f"loaded {symbol} from cache")
                return data
            except Exception as e:
                print(f"cache load failed: {str(e)}, fetching fresh data...")

        print(f"fetching {symbol} from {self.provider.name}...")
        data = self.provider.fetch(symbol, start_date, end_date, interval)

        if use_cache:
            self._save_to_cache(data, cache_file)

        return data

    def load_multiple(
        self,
        symbols: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
        use_cache: bool = True
    ) -> dict:
        # load data for multiple symbols
        results = {}

        for symbol in symbols:
            try:
                data = self.load(symbol, start_date, end_date, interval, use_cache)
                results[symbol] = data
            except Exception as e:
                print(f"warning: failed to load {symbol}: {str(e)}")
                results[symbol] = None

        return results

    def clear_cache(self, symbol: Optional[str] = None):
        # clear cached data, optionally for specific symbol
        if symbol:
            pattern = f"{symbol}_*.pkl"
            for file in self.cache_dir.glob(pattern):
                file.unlink()
                print(f"cleared cache for {symbol}")
        else:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            print("cleared all cached data")

    def _get_cache_filename(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str
    ) -> Path:
        # generate cache filename based on parameters
        safe_symbol = symbol.replace('/', '_').replace('^', '_')
        parts = [safe_symbol]
        if start_date:
            parts.append(start_date)
        if end_date:
            parts.append(end_date)
        parts.append(interval)

        filename = '_'.join(parts) + '.pkl'
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_file: Path, expiry_days: int) -> bool:
        # check if cache file exists and is not expired
        if not cache_file.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(days=expiry_days)

    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        # load data from cache file
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path):
        # save data to cache file
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)


def quick_load(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    # quick data loading with caching
    loader = DataLoader()
    return loader.load(symbol, start_date, end_date, interval)
