# data module for fetching and preprocessing crypto market data

from .base import DataProvider
from .providers import BinanceProvider, fetch_crypto
from .preprocessing import (
    handle_missing_data,
    remove_outliers,
    add_returns,
    add_log_returns,
    normalize_volume,
    resample_data,
    align_data,
    split_train_test,
    create_features,
    validate_ohlc_logic
)

__all__ = [
    # base
    'DataProvider',

    # providers
    'BinanceProvider',
    'fetch_crypto',

    # preprocessing
    'handle_missing_data',
    'remove_outliers',
    'add_returns',
    'add_log_returns',
    'normalize_volume',
    'resample_data',
    'align_data',
    'split_train_test',
    'create_features',
    'validate_ohlc_logic'
]
