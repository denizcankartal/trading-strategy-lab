# data preprocessing and cleaning utilities

import pandas as pd
import numpy as np

def handle_missing_data(
    data: pd.DataFrame,
    method: str = 'ffill'
) -> pd.DataFrame:
    # handle missing data with forward fill, drop, or interpolation
    data = data.copy()

    if method == 'ffill':
        data = data.ffill()
    elif method == 'drop':
        data = data.dropna()
    elif method == 'interpolate':
        data = data.interpolate(method='linear')
    else:
        raise ValueError(f"unknown method: {method}")

    return data


def remove_outliers(
    data: pd.DataFrame,
    column: str = 'close',
    n_std: float = 5.0
) -> pd.DataFrame:
    # remove outliers based on standard deviation
    data = data.copy()

    mean = data[column].mean()
    std = data[column].std()

    if std == 0:
        return data

    z_scores = np.abs((data[column] - mean) / std)
    data = data[z_scores < n_std]

    return data


def add_returns(
    data: pd.DataFrame,
    periods: list = [1]
) -> pd.DataFrame:
    # add return columns to dataframe
    data = data.copy()

    for period in periods:
        col_name = f'returns_{period}d' if period != 1 else 'returns'
        data[col_name] = data['close'].pct_change(periods=period)

    return data


def add_log_returns(
    data: pd.DataFrame,
    periods: list = [1]
) -> pd.DataFrame:
    # add log return columns (additive returns)
    data = data.copy()

    for period in periods:
        col_name = f'log_returns_{period}d' if period != 1 else 'log_returns'
        price_ratio = data['close'] / data['close'].shift(period)
        # replace 0, inf, -inf with nan to avoid log(0) and division by zero
        price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
        data[col_name] = np.log(price_ratio)

    return data


def normalize_volume(data: pd.DataFrame) -> pd.DataFrame:
    # normalize volume using 20-day rolling average
    data = data.copy()
    avg_volume = data['volume'].rolling(window=20).mean()
    # replace 0 with nan to avoid division by zero
    data['normalized_volume'] = data['volume'] / avg_volume.replace(0, np.nan)

    return data


def resample_data(
    data: pd.DataFrame,
    freq: str = 'W'
) -> pd.DataFrame:
    # resample ohlcv data to different frequency
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have datetimeindex for resampling")

    resampled = data.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    resampled = resampled.dropna()

    return resampled


def align_data(
    data_dict: dict,
    method: str = 'inner'
) -> dict:
    # align multiple dataframes to same date index
    if not data_dict:
        return {}

    dfs = list(data_dict.values())
    symbols = list(data_dict.keys())

    if method == 'inner':
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)

        aligned = {
            symbol: data_dict[symbol].loc[common_index]
            for symbol in symbols
        }

    elif method == 'outer':
        all_dates = dfs[0].index
        for df in dfs[1:]:
            all_dates = all_dates.union(df.index)

        aligned = {
            symbol: data_dict[symbol].reindex(all_dates)
            for symbol in symbols
        }

    else:
        raise ValueError(f"unknown method: {method}")

    return aligned


def split_train_test(
    data: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    # split time series data maintaining temporal order
    # train on earlier data, test on later data
    split_idx = int(len(data) * (1 - test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    return train, test


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    # create common technical features: returns, moving averages, volatility
    data = data.copy()

    data = add_returns(data, periods=[1, 5, 20])

    for period in [10, 20, 50, 200]:
        data[f'sma_{period}'] = data['close'].rolling(window=period).mean()

    for period in [10, 20]:
        data[f'volatility_{period}'] = data['returns'].rolling(window=period).std()

    data = normalize_volume(data)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)

    return data


def validate_ohlc_logic(data: pd.DataFrame, fix: bool = False) -> pd.DataFrame:
    # validate ohlc logic and optionally fix invalid rows
    data = data.copy()

    invalid_high = (data['high'] < data['open']) | (data['high'] < data['close']) | (data['high'] < data['low'])
    invalid_low = (data['low'] > data['open']) | (data['low'] > data['close']) | (data['low'] > data['high'])
    invalid_volume = data['volume'] < 0

    total_invalid = invalid_high.sum() + invalid_low.sum() + invalid_volume.sum()

    if total_invalid > 0:
        if not fix:
            raise ValueError(
                f"found {total_invalid} rows with invalid ohlc logic, "
                f"set fix=true to attempt correction"
            )

        data.loc[invalid_high, 'high'] = data.loc[invalid_high, ['open', 'close', 'low']].max(axis=1)
        data.loc[invalid_low, 'low'] = data.loc[invalid_low, ['open', 'close', 'high']].min(axis=1)
        data.loc[invalid_volume, 'volume'] = 0

        print(f"fixed {total_invalid} invalid rows")

    return data
