# visualization functions for strategy analysis and performance reporting

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_equity_curve(
    returns: pd.Series,
    title: str = "equity curve",
    benchmark_returns: Optional[pd.Series] = None,
    initial_capital: float = 100000,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    # plot equity curve from returns with optional benchmark
    fig, ax = plt.subplots(figsize=figsize)

    equity = initial_capital * (1 + returns).cumprod()
    ax.plot(equity.index, equity.values, linewidth=2, label='strategy', color='blue')

    if benchmark_returns is not None:
        benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity.values,
                linewidth=2, label='benchmark', color='gray', alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('date', fontsize=12)
    ax.set_ylabel('portfolio value ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    return fig

def plot_drawdown(
    returns: pd.Series,
    title: str = "drawdown chart",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    # plot drawdown chart showing peak-to-trough declines
    fig, ax = plt.subplots(figsize=figsize)

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    ax.fill_between(drawdown.index, drawdown.values, 0,
                     alpha=0.3, color='red', label='drawdown')
    ax.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('date', fontsize=12)
    ax.set_ylabel('drawdown', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()
    ax.plot(max_dd_idx, max_dd_value, 'ro', markersize=10,
            label=f'max dd: {max_dd_value:.2%}')
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()
    return fig

def plot_trades(
    data: pd.DataFrame,
    signals: pd.Series,
    title: str = "price chart with trade signals",
    figsize: Tuple[int, int] = (14, 7)
) -> plt.Figure:
    # plot price chart with buy/sell signals overlaid
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(data.index, data['close'], linewidth=1.5,
            label='close price', color='black', alpha=0.7)

    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]

    if len(buy_signals) > 0:
        ax.scatter(buy_signals.index,
                   data.loc[buy_signals.index, 'close'],
                   marker='^', color='green', s=100, alpha=0.8,
                   label=f'buy ({len(buy_signals)})', zorder=5)

    if len(sell_signals) > 0:
        ax.scatter(sell_signals.index,
                   data.loc[sell_signals.index, 'close'],
                   marker='v', color='red', s=100, alpha=0.8,
                   label=f'sell ({len(sell_signals)})', zorder=5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('date', fontsize=12)
    ax.set_ylabel('price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def plot_returns_distribution(
    returns: pd.Series,
    title: str = "returns distribution",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    # plot histogram of returns distribution with statistics
    fig, ax = plt.subplots(figsize=figsize)

    returns_clean = returns.dropna()
    n, bins, patches = ax.hist(returns_clean, bins=50, alpha=0.7,
                                color='blue', edgecolor='black')

    mean_return = returns_clean.mean()
    ax.axvline(mean_return, color='red', linestyle='--',
               linewidth=2, label=f'mean: {mean_return:.2%}')

    median_return = returns_clean.median()
    ax.axvline(median_return, color='green', linestyle='--',
               linewidth=2, label=f'median: {median_return:.2%}')

    stats_text = f"""
    mean: {mean_return:.2%}
    median: {median_return:.2%}
    std dev: {returns_clean.std():.2%}
    skewness: {returns_clean.skew():.2f}
    kurtosis: {returns_clean.kurtosis():.2f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('returns', fontsize=12)
    ax.set_ylabel('frequency', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    plt.tight_layout()
    return fig

def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    title: str = "rolling performance metrics",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    # plot rolling sharpe ratio and volatility over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    rolling_return = returns.rolling(window=window).mean() * window
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
    rolling_sharpe = rolling_return / rolling_vol

    ax1.plot(rolling_return.index, rolling_return.values,
             linewidth=2, color='blue')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title(f'{title} (window: {window} periods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('annualized return', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    ax2.plot(rolling_vol.index, rolling_vol.values,
             linewidth=2, color='orange')
    ax2.set_ylabel('annualized volatility', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
             linewidth=2, color='green')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('sharpe ratio', fontsize=11)
    ax3.set_xlabel('date', fontsize=12)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "monthly returns heatmap",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    # plot heatmap of monthly returns by year
    monthly_returns = (1 + returns).resample('M').prod() - 1

    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })

    pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')

    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'returns'},
                linewidths=0.5, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('month', fontsize=12)
    ax.set_ylabel('year', fontsize=12)

    plt.tight_layout()
    return fig

def create_performance_dashboard(
    data: pd.DataFrame,
    returns: pd.Series,
    signals: pd.Series,
    metrics: dict,
    title: str = "strategy performance dashboard",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    # create comprehensive performance dashboard with multiple plots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    equity = (1 + returns).cumprod()
    ax1.plot(equity.index, equity.values, linewidth=2, color='blue')
    ax1.set_title('equity curve', fontweight='bold')
    ax1.set_ylabel('cumulative returns')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}x'))

    ax2 = fig.add_subplot(gs[0, 1])
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')
    ax2.set_title('drawdown', fontweight='bold')
    ax2.set_ylabel('drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(data.index, data['close'], linewidth=1.5, color='black', alpha=0.7)
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]
    if len(buy_signals) > 0:
        ax3.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'],
                    marker='^', color='green', s=80, alpha=0.8, zorder=5)
    if len(sell_signals) > 0:
        ax3.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'],
                    marker='v', color='red', s=80, alpha=0.8, zorder=5)
    ax3.set_title('price chart with trade signals', fontweight='bold')
    ax3.set_ylabel('price ($)')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    returns_clean = returns.dropna()
    ax4.hist(returns_clean, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(returns_clean.mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_title('returns distribution', fontweight='bold')
    ax4.set_xlabel('returns')
    ax4.set_ylabel('frequency')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    metrics_text = [
        ['metric', 'value'],
        ['─' * 20, '─' * 15],
        ['annual return', f"{metrics.get('annual_return', 0):.2%}"],
        ['volatility', f"{metrics.get('volatility', 0):.2%}"],
        ['sharpe ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ['sortino ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
        ['max drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
        ['calmar ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
        ['win rate', f"{metrics.get('win_rate', 0):.2%}"],
        ['profit factor', f"{metrics.get('profit_factor', 0):.2f}"],
        ['total return', f"{metrics.get('total_return', 0):.2%}"],
        ['num periods', f"{metrics.get('num_periods', 0)}"]
    ]

    table = ax5.table(cellText=metrics_text, cellLoc='left', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax5.set_title('performance metrics', fontweight='bold', pad=20)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    return fig
