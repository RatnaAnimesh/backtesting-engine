# Event-Driven Backtesting Engine

This project is a powerful, event-driven backtesting engine designed for quantitative finance research. It allows you to test trading strategies against historical market data, providing a robust framework for strategy development and analysis.

## Key Features

- **Event-Driven Architecture:** The engine is built around a queue of events (Market, Signal, Order, Fill), which makes it flexible and realistic.
- **Strategy Abstraction:** Easily create and plug in new trading strategies by inheriting from a base strategy class.
- **Portfolio Management:** Tracks portfolio value, cash, positions, and performance metrics over time.
- **Realistic Costs:** Includes support for transaction costs to provide more accurate backtest results.
- **Performance Metrics:** Automatically calculates key performance indicators like Sharpe Ratio, CAGR, and Max Drawdown.

## Project Structure

```
/
├── backtester.py             # Core backtesting engine and Portfolio class
├── data_manager.py           # Handles downloading and caching of market data
├── performance_metrics.py    # Functions for calculating performance metrics
├── run_backtest.py           # Main script to configure and run a backtest
├── strategies/               # Directory for strategy implementations
│   ├── __init__.py
│   ├── base_strategy.py      # Abstract base class for all strategies
│   └── macd_strategy.py      # Example: MACD-based trading strategy
└── data/                     # Cached market data
    └── equities/
```

## How to Use

1.  **Configure Your Strategy:**
    - Open `run_backtest.py`.
    - Define the `tickers`, `start_date`, `end_date`, and other parameters for your backtest.
    - Instantiate the strategy you want to test (e.g., `MACDStrategy`).

2.  **Run the Backtest:**
    - Execute the `run_backtest.py` script from your terminal:
      ```bash
      python run_backtest.py
      ```

3.  **Analyze the Results:**
    - The backtest will print a summary of the performance metrics to the console.
    - An interactive performance report will be generated, allowing you to visualize the equity curve and trades over time.

## Creating a New Strategy

To create your own trading strategy:

1.  Create a new Python file in the `strategies/` directory (e.g., `my_strategy.py`).
2.  Import the `BaseStrategy` class: `from .base_strategy import BaseStrategy`
3.  Create a new class that inherits from `BaseStrategy`.
4.  Implement the `generate_signals` method. This method takes the current market data as input and should return a Pandas Series of target weights for each asset.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- yfinance
- plotly (for performance reports)

You can install them using pip:
```bash
pip install pandas numpy yfinance plotly
```
