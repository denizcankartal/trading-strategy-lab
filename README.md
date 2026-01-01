## trading strategy research
framework for developing and backtesting trading strategies

1. fetch market data
2. add features
3. create strategy
4. split train/test
5. backtest on train
6. validate on test
7. analyze performance metrics
8. visualize results with charts
9. make go/no-go decision

## quick start

```bash
# run backtest
docker compose up

# or run in background
docker compose up -d
docker compose logs -f
```

## modules

* data - fetch market data
* strategies - create trading strategies by inheriting from base class
* backtester - execute strategies with realistic market conditions such as:
    - commission and slippage modeling
    - whole share constraints (no fractional)
    - cash constraints (can't trade without capital)
    - position tracking
    - full trade history

* metrics - calculate performance metrics
* visualization - create charts and dashboards

    available plots:
    - `plot_equity_curve()` - portfolio value over time
    - `plot_drawdown()` - underwater chart
    - `plot_trades()` - price chart with signals
    - `plot_returns_distribution()` - histogram
    - `plot_rolling_metrics()` - rolling sharpe/volatility
    - `plot_monthly_returns_heatmap()` - calendar view
    - `create_performance_dashboard()` - 6-panel summary

## key concepts

### realistic execution

the backtester models real market conditions:

**commission**: pay fee on each trade
- example: buy $10,000 worth @ 0.1% = $10 commission

**slippage**: pay more when buying, receive less when selling
- simulates bid-ask spread and market impact
- buy: price * (1 + slippage_pct)
- sell: price * (1 - slippage_pct)

**whole shares**: no fractional shares
- 100 shares, not 100.5 shares

**cash constraints**: can't buy without sufficient capital

### walk-forward analysis

prevents overfitting by testing across multiple time windows:

```python
# train on 1 year, test on 3 months, step forward 3 months
results = backtester.walk_forward_analysis(
    strategy, data,
    train_size=252,  # days
    test_size=63,
    step_size=63
)
```

this gives multiple out-of-sample test results to validate robustness

## performance tips

**start simple**: test basic strategies before complex ones

**validate out-of-sample**: always test on unseen data (walk-forward or train/test split)

**check transaction costs**: many strategies look good until you add realistic costs

**compare to benchmark**: beating buy-and-hold is hard

## troubleshooting

**"no data retrieved"**: check symbol format (BTC/USDT not BTCUSDT)

**"insufficient cash"**: reduce position_size_pct or increase initial_capital

**poor performance**: add transaction costs (commission_pct, slippage_pct)
