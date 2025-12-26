# complete workflow example: end-to-end crypto trading strategy research
# demonstrates workflow from data loading to backtesting to visualization
# typical quant trading workflow

from data import fetch_crypto, create_features, split_train_test
from strategies import MovingAverageCrossover
from backtester import Backtester, quick_backtest
from metrics import calculate_all_metrics
from visualization import (
    create_performance_dashboard,
    plot_equity_curve,
    plot_trades
)
import matplotlib.pyplot as plt


def complete_workflow():
    # complete workflow: data -> strategy -> backtest -> visualize
    print("=" * 70)
    print(" complete crypto trading strategy research workflow")
    print("=" * 70)
    print()

    # step 1: data acquisition
    print("step 1: data acquisition")
    print("-" * 70)

    symbol = 'BTC/USDT'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    print(f"loading {symbol} data from {start_date} to {end_date}...")
    data = fetch_crypto(symbol, start_date=start_date, end_date=end_date)

    print(f"loaded {len(data)} candles of data")
    print(f"  date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print()

    # step 2: feature engineering
    print("step 2: feature engineering")
    print("-" * 70)

    print("adding technical features (mas, volatility, etc.)...")
    data = create_features(data)

    print(f"added {len(data.columns) - 5} technical features")
    print(f"  features: {[col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']][:5]}...")
    print()

    # step 3: strategy development
    print("step 3: strategy development")
    print("-" * 70)

    # create strategy
    strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
    print(f"strategy: {strategy}")
    print(f"parameters: {strategy.parameters}")

    # generate signals
    signals = strategy.generate_signals(data)
    num_signals = (signals != 0).sum()
    num_buy = (signals == 1).sum()
    num_sell = (signals == -1).sum()

    print(f"generated {num_signals} total signals")
    print(f"  buy signals:  {num_buy}")
    print(f"  sell signals: {num_sell}")
    print()

    # step 4: train/test split
    print("step 4: train/test split")
    print("-" * 70)

    train_data, test_data = split_train_test(data, test_size=0.2)

    print(f"training set: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"test set:     {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")
    print()

    # step 5: backtest on training data
    print("step 5: backtest on training data")
    print("-" * 70)

    print("running backtest on training period...")
    backtester = Backtester(
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        position_size_pct=1.0
    )

    train_result = backtester.run(strategy, train_data, symbol=symbol)

    print(f"training backtest complete")
    print(f"  total return:  {train_result.total_return_pct:>8.2%}")
    print(f"  sharpe ratio:  {train_result.metrics.get('sharpe_ratio', 0):>8.2f}")
    print(f"  max drawdown:  {train_result.metrics.get('max_drawdown', 0):>8.2%}")
    print(f"  win rate:      {train_result.win_rate:>8.1%}")
    print(f"  total trades:  {train_result.num_trades:>8}")
    print()

    # step 6: validate on test data (out-of-sample)
    print("step 6: validate on test data (out-of-sample)")
    print("-" * 70)

    print("running backtest on test period...")
    test_result = backtester.run(strategy, test_data, symbol=symbol)

    print(f"test backtest complete")
    print(f"  total return:  {test_result.total_return_pct:>8.2%}")
    print(f"  sharpe ratio:  {test_result.metrics.get('sharpe_ratio', 0):>8.2f}")
    print(f"  max drawdown:  {test_result.metrics.get('max_drawdown', 0):>8.2%}")
    print(f"  win rate:      {test_result.win_rate:>8.1%}")
    print(f"  total trades:  {test_result.num_trades:>8}")
    print()

    # step 7: compare train vs test performance
    print("step 7: train vs test performance")
    print("-" * 70)

    print(f"{'metric':<20} {'training':<15} {'test':<15} {'difference'}")
    print("-" * 70)
    print(f"{'total return':<20} {train_result.total_return_pct:>13.2%}  "
          f"{test_result.total_return_pct:>13.2%}  "
          f"{test_result.total_return_pct - train_result.total_return_pct:>10.2%}")
    print(f"{'sharpe ratio':<20} {train_result.metrics.get('sharpe_ratio', 0):>13.2f}  "
          f"{test_result.metrics.get('sharpe_ratio', 0):>13.2f}  "
          f"{test_result.metrics.get('sharpe_ratio', 0) - train_result.metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"{'max drawdown':<20} {train_result.metrics.get('max_drawdown', 0):>13.2%}  "
          f"{test_result.metrics.get('max_drawdown', 0):>13.2%}  "
          f"{test_result.metrics.get('max_drawdown', 0) - train_result.metrics.get('max_drawdown', 0):>10.2%}")
    print(f"{'win rate':<20} {train_result.win_rate:>13.1%}  "
          f"{test_result.win_rate:>13.1%}  "
          f"{test_result.win_rate - train_result.win_rate:>10.1%}")

    # check for overfitting
    if test_result.total_return_pct < train_result.total_return_pct * 0.5:
        print("\nwarning: significant performance degradation in test set")
        print("strategy may be overfit to training data")
    else:
        print("\nperformance is consistent between train and test periods")

    print()

    # step 8: full period backtest
    print("step 8: full period backtest")
    print("-" * 70)

    print("running backtest on full dataset...")
    full_result = backtester.run(strategy, data, symbol=symbol)

    print(f"full backtest complete")
    full_result.print_summary()

    # step 9: visualization
    print("\n" + "=" * 70)
    print("step 9: visualization")
    print("-" * 70)

    # create dashboard
    print("\ngenerating performance dashboard...")
    returns = full_result.equity_curve.pct_change().dropna()
    signals = strategy.generate_signals(data)

    fig = create_performance_dashboard(
        data=data,
        returns=returns,
        signals=signals,
        metrics=full_result.metrics,
        title=f"{symbol} - {strategy.name} - complete analysis"
    )

    dashboard_file = 'complete_workflow_dashboard.png'
    fig.savefig(dashboard_file, dpi=150, bbox_inches='tight')
    print(f"saved: {dashboard_file}")
    plt.close(fig)

    # create equity curve with benchmark
    print("generating equity curve comparison...")
    spy_data = quick_load('SPY', start_date=start_date, end_date=end_date)
    spy_returns = spy_data['close'].pct_change().dropna()

    # align returns
    aligned_returns = returns.reindex(spy_returns.index).dropna()
    aligned_spy = spy_returns.reindex(returns.index).dropna()

    fig = plot_equity_curve(
        returns=aligned_returns,
        benchmark_returns=aligned_spy,
        title=f"{symbol} - {strategy.name} vs spy benchmark"
    )

    equity_file = 'complete_workflow_equity.png'
    fig.savefig(equity_file, dpi=150, bbox_inches='tight')
    print(f"saved: {equity_file}")
    plt.close(fig)

    # step 10: final analysis & decision
    print("\n" + "=" * 70)
    print("step 10: final analysis & decision")
    print("-" * 70)

    # calculate buy-and-hold benchmark
    bh_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]

    print("\nstrategy vs buy-and-hold:")
    print(f"  strategy return:     {full_result.total_return_pct:>8.2%}")
    print(f"  buy & hold return:   {bh_return:>8.2%}")
    print(f"  outperformance:      {full_result.total_return_pct - bh_return:>8.2%}")

    print(f"\nrisk metrics:")
    print(f"  sharpe ratio:        {full_result.metrics.get('sharpe_ratio', 0):>8.2f}")
    print(f"  max drawdown:        {full_result.metrics.get('max_drawdown', 0):>8.2%}")
    print(f"  calmar ratio:        {full_result.metrics.get('calmar_ratio', 0):>8.2f}")

    print(f"\ntrading activity:")
    print(f"  total trades:        {full_result.num_trades:>8}")
    print(f"  win rate:            {full_result.win_rate:>8.1%}")
    print(f"  profit factor:       {full_result.profit_factor:>8.2f}")
    print(f"  avg hold period:     {full_result.avg_holding_period:>8.1f} days")

    # decision criteria
    print("\n" + "-" * 70)
    print("decision criteria:")
    print("-" * 70)

    criteria_met = 0
    total_criteria = 5

    if full_result.total_return_pct > 0:
        print("positive returns")
        criteria_met += 1
    else:
        print("negative returns")

    if full_result.total_return_pct > bh_return:
        print("outperforms buy-and-hold")
        criteria_met += 1
    else:
        print("underperforms buy-and-hold")

    if full_result.metrics.get('sharpe_ratio', 0) > 1.0:
        print("sharpe ratio > 1.0")
        criteria_met += 1
    else:
        print("sharpe ratio < 1.0")

    if full_result.metrics.get('max_drawdown', 0) > -0.30:
        print("max drawdown < 30%")
        criteria_met += 1
    else:
        print("max drawdown > 30%")

    if full_result.win_rate > 0.45:
        print("win rate > 45%")
        criteria_met += 1
    else:
        print("win rate < 45%")

    print("-" * 70)
    print(f"criteria met: {criteria_met}/{total_criteria}")

    if criteria_met >= 4:
        print("\nrecommendation: strategy approved")
        print("strategy shows strong performance and meets most criteria")
        print("consider deploying with proper risk management")
    elif criteria_met >= 3:
        print("\nrecommendation: strategy needs improvement")
        print("strategy shows promise but needs refinement")
        print("consider parameter optimization or additional filters")
    else:
        print("\nrecommendation: strategy rejected")
        print("strategy does not meet minimum criteria")
        print("consider trying a different approach")

    print("\n" + "=" * 70)
    print("workflow complete")
    print("=" * 70)
    print(f"\ngenerated files:")
    print(f"  1. {dashboard_file}")
    print(f"  2. {equity_file}")
    print("\nnext steps:")
    print("  - review visualizations")
    print("  - analyze individual trades")
    print("  - consider parameter optimization")
    print("  - test on different symbols/markets")
    print("  - implement risk management improvements")
    print()


if __name__ == "__main__":
    try:
        complete_workflow()
    except Exception as e:
        print(f"\nerror: {str(e)}")
        print("\nnote: requires internet connection for data")
        print("run: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
