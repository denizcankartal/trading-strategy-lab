import numpy as np
import pandas as pd
from typing import Union

def calculate_returns(prices: pd.Series) -> pd.Series:
    # period-over-period returns
    return prices.pct_change()

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    # cumulative returns from period returns
    return (1 + returns).cumprod() - 1

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    # sharpe ratio measuring excess return per unit of risk
    if returns.std() == 0:
        return 0.0

    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free_rate
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

    return sharpe

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    # sortino ratio penalizing only downside volatility
    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free_rate
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()

    return sortino

def calculate_max_drawdown(returns: pd.Series) -> float:
    # maximum drawdown from peak to trough
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()

def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    # calmar ratio measuring return relative to max drawdown
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    annual_return = (1 + returns.mean()) ** periods_per_year - 1

    return annual_return / abs(max_dd)

def calculate_win_rate(returns: pd.Series) -> float:
    # percentage of winning periods
    if len(returns) == 0:
        return 0.0

    winning_periods = (returns > 0).sum()
    total_periods = len(returns[returns != 0])

    if total_periods == 0:
        return 0.0

    return winning_periods / total_periods

def calculate_profit_factor(returns: pd.Series) -> float:
    # profit factor ratio of wins to losses
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss

def calculate_annual_return(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    # annualized return
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years == 0:
        return 0.0

    annual_return = (1 + total_return) ** (1 / years) - 1

    return annual_return

def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    # annualized volatility
    return returns.std() * np.sqrt(periods_per_year)

def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    # calculate all performance metrics at once
    return {
        'annual_return': calculate_annual_return(returns, periods_per_year),
        'volatility': calculate_volatility(returns, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'total_return': (1 + returns).prod() - 1,
        'num_periods': len(returns)
    }
