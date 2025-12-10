# data module for fetching, loading, and preprocessing market data

from .base import DataProvider
from .providers import YahooFinanceProvider, CSVDataProvider
from .loaders import DataLoader, quick_load
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
    # base classes
    'DataProvider',

    # providers
    'YahooFinanceProvider',
    'CSVDataProvider',

    # loaders
    'DataLoader',
    'quick_load',

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
