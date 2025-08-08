from backtester import Backtester
from strategies.macd_strategy import MACDStrategy
from performance_metrics import get_performance_metrics, create_performance_report
import pandas as pd

if __name__ == "__main__":
    # Define backtest parameters
    test_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    test_start_date = "2019-01-08"
    test_end_date = "2023-12-31"
    initial_cash = 100000.0
    transaction_cost_bps = 1.5 # More realistic transaction cost

    # Instantiate your strategy
    my_strategy = MACDStrategy()

    # Instantiate the backtester
    backtester = Backtester(
        strategy=my_strategy,
        tickers=test_tickers,
        start_date=test_start_date,
        end_date=test_end_date,
        initial_cash=initial_cash,
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=365  # Provide a one-year lookback period
    )

    # Run the backtest
    results = backtester.run()

    # Get and print performance metrics
    equity_curve = results['equity_curve']
    trades = results['trades']
    metrics = get_performance_metrics(equity_curve)

    print("\n--- Overall Backtest Performance ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Generate and show the interactive performance report
    if not equity_curve.empty:
        create_performance_report(equity_curve, trades, metrics)

