import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from data_manager import get_multiple_historical_data

class Portfolio:
    """
    Manages the state of the portfolio during a backtest.
    """
    def __init__(self, initial_cash: float = 100000.0, transaction_cost_bps: float = 1.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = pd.Series(dtype=float) # Stores number of shares for each asset
        self.holdings_value = 0.0
        self.total_value = initial_cash
        self.transaction_cost_bps = transaction_cost_bps # Basis points (e.g., 1.0 for 0.01%)
        self.trades = [] # List to record trade details

    def _calculate_transaction_cost(self, value: float) -> float:
        """
        Calculates transaction cost for a given trade value.
        """
        return abs(value) * (self.transaction_cost_bps / 10000.0)

    def update_portfolio(self, current_prices: pd.Series):
        """
        Updates the portfolio's value based on current market prices.
        """
        # Remove positions with 0 shares
        self.positions = self.positions[self.positions != 0]

        if not self.positions.empty and not current_prices.empty:
            # Ensure prices align with positions
            aligned_prices = current_prices.reindex(self.positions.index).dropna()
            self.holdings_value = (self.positions.loc[aligned_prices.index] * aligned_prices).sum()
        else:
            self.holdings_value = 0.0

        self.total_value = self.cash + self.holdings_value

    def execute_trades(self, target_weights: pd.Series, current_prices: pd.Series, current_date: pd.Timestamp):
        """
        Executes trades to rebalance the portfolio to target weights.
        """
        if current_prices.empty or target_weights.empty:
            return

        # Ensure all potential tickers are in the positions Series
        for ticker in target_weights.index:
            if ticker not in self.positions:
                self.positions[ticker] = 0.0

        # Calculate current market value of holdings
        current_holdings_value = (self.positions.reindex(current_prices.index, fill_value=0) * current_prices).sum()
        # Total capital available for allocation (including cash)
        total_capital = self.cash + current_holdings_value

        # Calculate target value for each asset
        target_values = target_weights * total_capital

        # Calculate current value of each position
        current_values = self.positions.reindex(current_prices.index, fill_value=0) * current_prices

        # Calculate value difference (what needs to be bought/sold)
        value_diff = target_values - current_values

        # Sort trades to sell first (to free up cash) then buy
        trade_order = value_diff.sort_values(ascending=True) # Negative values (sells) first

        for ticker, value_change in trade_order.items():
            if abs(value_change) < 0.01: # Ignore very small changes
                continue

            if ticker not in current_prices or pd.isna(current_prices[ticker]) or current_prices[ticker] == 0:
                print(f"Warning: Cannot trade {ticker} on {current_date} as price is zero or not available.")
                continue

            # Calculate shares to trade
            shares_to_trade = value_change / current_prices[ticker]

            # Apply transaction costs
            cost = self._calculate_transaction_cost(value_change)

            # Execute trade
            self.cash -= cost # Deduct cost regardless of buy/sell

            if value_change > 0: # Buy
                # Ensure enough cash for buys
                if self.cash >= value_change:
                    self.positions[ticker] = self.positions.get(ticker, 0) + shares_to_trade
                    self.cash -= value_change
                    self.trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'type': 'BUY',
                        'shares': shares_to_trade,
                        'price': current_prices[ticker],
                        'value': value_change,
                        'cost': cost
                    })
                else:
                    # Adjust shares to trade based on available cash
                    affordable_value = self.cash
                    if affordable_value > 0:
                        affordable_shares = affordable_value / current_prices[ticker]
                        self.positions[ticker] = self.positions.get(ticker, 0) + affordable_shares
                        self.cash -= affordable_value
                        self.trades.append({
                            'date': current_date,
                            'ticker': ticker,
                            'type': 'BUY (partial)',
                            'shares': affordable_shares,
                            'price': current_prices[ticker],
                            'value': affordable_value,
                            'cost': self._calculate_transaction_cost(affordable_value)
                        })
                        print(f"Warning: Not enough cash to fully buy {ticker} on {current_date}. Bought {affordable_shares:.2f} shares.")
            else: # Sell
                # Ensure we have the position to sell
                if abs(shares_to_trade) > self.positions.get(ticker, 0):
                    print(f"Warning: Attempted to sell more {ticker} than held on {current_date}. Adjusting trade.")
                    shares_to_trade = -self.positions.get(ticker, 0)
                    value_change = shares_to_trade * current_prices[ticker]

                self.positions[ticker] += shares_to_trade # shares_to_trade is negative
                self.cash -= value_change # value_change is negative, so cash increases
                self.trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'type': 'SELL',
                    'shares': shares_to_trade,
                    'price': current_prices[ticker],
                    'value': value_change,
                    'cost': cost
                })

        # Clean up any tiny residual positions due to floating point arithmetic
        self.positions = self.positions[~np.isclose(self.positions, 0)]


class Backtester:
    """
    Core backtesting engine.
    """
    def __init__(self, strategy: BaseStrategy, tickers: list[str], start_date: str, end_date: str,
                 initial_cash: float = 100000.0, transaction_cost_bps: float = 1.0, lookback_days: int = 30):
        self.strategy = strategy
        self.tickers = tickers
        self.backtest_start_date = pd.to_datetime(start_date)
        self.backtest_end_date = pd.to_datetime(end_date)
        self.initial_cash = initial_cash
        self.transaction_cost_bps = transaction_cost_bps
        self.portfolio = Portfolio(initial_cash, transaction_cost_bps)
        self.equity_curve = pd.Series(dtype=float)

        # Determine the data loading start date based on the lookback period
        self.data_start_date = self.backtest_start_date - pd.Timedelta(days=lookback_days)

        # Load all data (including the lookback period)
        self.full_data = self._load_data(self.data_start_date.strftime('%Y-%m-%d'), self.backtest_end_date.strftime('%Y-%m-%d'))

        if self.full_data.empty:
            raise ValueError("No data loaded for backtesting. Check tickers and date range.")

        # Ensure full_data index is datetime and sorted
        self.full_data.index = pd.to_datetime(self.full_data.index)
        self.full_data = self.full_data.sort_index()

        # The actual data used for iterating through the backtest period
        self.backtest_data = self.full_data.loc[self.backtest_start_date:self.backtest_end_date]

        # Rebalancing dates (e.g., end of each month within the backtest period)
        self.rebalance_dates = pd.to_datetime(self.backtest_data.index.to_period('M').drop_duplicates().to_timestamp(how='end'))
        self.rebalance_dates = self.rebalance_dates[(self.rebalance_dates >= self.backtest_start_date) & (self.rebalance_dates <= self.backtest_end_date)]

        # Call strategy pre-run setup with all available data
        self.strategy.pre_run_setup(self.full_data)

    def _load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads all necessary historical data using the data_manager.
        """
        print(f"Loading data for tickers: {self.tickers} from {start_date} to {end_date}")
        return get_multiple_historical_data(self.tickers, start_date, end_date)

    def run(self) -> dict:
        """
        Runs the backtest simulation.
        """
        print("Starting backtest...")
        portfolio_values = []

        # Iterate through each day in the backtest period
        for i, current_date in enumerate(self.backtest_data.index):
            # Get prices for the current day
            current_prices = self.backtest_data.loc[current_date].dropna()

            if current_prices.empty:
                print(f"No price data for {current_date}. Skipping.")
                # On skip days, the portfolio value should be carried over
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])
                else:
                    portfolio_values.append(self.initial_cash)
                continue

            # Rebalance on specific dates
            if current_date in self.rebalance_dates:
                print(f"Rebalancing on {current_date}...")
                # Pass all historical data up to the current date for signal generation
                historical_data_for_strategy = self.full_data.loc[:current_date]
                target_weights = self.strategy.generate_signals(historical_data_for_strategy)

                if not target_weights.empty:
                    self.portfolio.execute_trades(target_weights, current_prices, current_date)
                else:
                    print(f"No signals generated for {current_date}.")

            # Update portfolio value at the end of each day
            self.portfolio.update_portfolio(current_prices)
            portfolio_values.append(self.portfolio.total_value)

        self.equity_curve = pd.Series(portfolio_values, index=self.backtest_data.index)
        # Normalize to initial cash, handle case where first value is zero
        if self.equity_curve.iloc[0] != 0:
            self.equity_curve = self.equity_curve / self.equity_curve.iloc[0] * self.initial_cash
        else:
            self.equity_curve = pd.Series(self.initial_cash, index=self.backtest_data.index)


        results = {
            'equity_curve': self.equity_curve,
            'trades': pd.DataFrame(self.portfolio.trades)
        }

        # Call strategy post-run analysis
        self.strategy.post_run_analysis(results)

        print("Backtest finished.")
        return results

# Example usage (for testing purposes, will be moved to run_backtest.py)
if __name__ == "__main__":
    class DummyStrategy(BaseStrategy):
        def __init__(self):
            super().__init__()
            print("Dummy Strategy Initialized")

        def generate_signals(self, current_data: pd.DataFrame) -> pd.Series:
            # Example: simple equal weight for first 3 assets if available
            if current_data.empty:
                return pd.Series(dtype=float)

            # Get the latest prices for all available assets
            latest_prices = current_data.iloc[-1].dropna()

            if latest_prices.empty:
                return pd.Series(dtype=float)

            # Simple equal weight for the first 3 available assets
            target_assets = latest_prices.index[:3]
            if len(target_assets) == 0:
                return pd.Series(dtype=float)

            weight_per_asset = 1.0 / len(target_assets)
            signals = pd.Series(weight_per_asset, index=target_assets)
            return signals

        def post_run_analysis(self, results: dict):
            print("Dummy Strategy Post-Run Analysis:")
            print(f"Final Equity: {results['equity_curve'].iloc[-1]:.2f}")
            # if not results['trades'].empty:
            #     print("Sample Trades:")
            #     print(results['trades'].head())

    # Make sure you have some data for these tickers locally or they will be fetched
    test_tickers = ["AAPL", "MSFT", "GOOG"]
    test_start_date = "2020-01-01"
    test_end_date = "2021-12-31"

    dummy_strategy = DummyStrategy()
    backtester = Backtester(
        strategy=dummy_strategy,
        tickers=test_tickers,
        start_date=test_start_date,
        end_date=test_end_date
    )

    results = backtester.run()
    print("\nBacktest Results:")
    print(results['equity_curve'].tail())
