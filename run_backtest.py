from backtester import Backtester
from strategies.news_trading_strategy import NewsSentimentStrategy # Import the new strategy
from performance_metrics import get_performance_metrics, create_performance_report
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Define backtest parameters
    # Using Nifty 50 symbols as per the sentiment data
    test_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
        "HCLTECH.NS", "ASIANPAINT.NS", "WIPRO.NS", "AXISBANK.NS", "LT.NS",
        "MARUTI.NS", "ULTRACEMCO.NS", "DMART.NS", "BAJAJFINSV.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "INDUSINDBK.NS",
        "NESTLEIND.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS", "GRASIM.NS",
        "HINDALCO.NS", "DRREDDY.NS", "CIPLA.NS", "ADANIPORTS.NS", "SBILIFE.NS",
        "BAJAJ-AUTO.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "COALINDIA.NS",
        "IOC.NS", "BPCL.NS", "SHREECEM.NS", "DIVISLAB.NS", "UPL.NS", "TATAMOTORS.NS",
        "M&M.NS", "WIPRO.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"
    ]
    
    # Adjust start date to match sentiment data (2017-01-03 is the earliest in the CSV)
    test_start_date = "2017-01-03"
    test_end_date = "2021-04-30" # Sentiment data ends April 2021
    initial_cash = 100000.0
    transaction_cost_bps = 1.5
    lookback_days = 30 # Adjusted for sentiment strategy, which uses sentiment lag

    print("\nStarting backtest with parameters:")
    print(f"Tickers: {len(test_tickers)} symbols")
    print(f"Date range: {test_start_date} to {test_end_date}")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Transaction cost (bps): {transaction_cost_bps}")
    print(f"Lookback days: {lookback_days}")

    # Instantiate the NewsSentimentStrategy
    # You can adjust sentiment_decay_alpha, n_top_stocks, sentiment_lag here
    my_strategy = NewsSentimentStrategy(
        sentiment_decay_alpha=0.05, # Example value, can be optimized
        n_top_stocks=5,
        sentiment_lag=1
    )

    # Instantiate the backtester
    backtester = Backtester(
        strategy=my_strategy,
        tickers=test_tickers,
        start_date=test_start_date,
        end_date=test_end_date,
        initial_cash=initial_cash,
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=lookback_days
    )

    # Run the backtest
    results = backtester.run()

    # Get and print performance metrics
    equity_curve = results['equity_curve']
    trades = results['trades']
    metrics = get_performance_metrics(equity_curve)

    print("\n--- News Sentiment Strategy Backtest Performance ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Generate and show the interactive performance report
    if not equity_curve.empty:
        create_performance_report(equity_curve, trades, metrics)


