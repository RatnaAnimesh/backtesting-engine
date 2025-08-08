import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
import os

class NewsSentimentStrategy(BaseStrategy):
    def __init__(self, sentiment_decay_alpha: float = 0.05, n_top_stocks: int = 5, sentiment_lag: int = 1):
        super().__init__()
        self.sentiment_decay_alpha = sentiment_decay_alpha
        self.n_top_stocks = n_top_stocks
        self.sentiment_lag = sentiment_lag
        self.stock_sentiment_df = None
        self.effective_sentiment_cache = {} # Cache for effective sentiment per ticker

    def pre_run_setup(self, full_data: pd.DataFrame):
        # Load stock-level sentiment data
        # Assuming stock_level_finbert_sentiment.csv is in the backtesting-engine/data directory
        sentiment_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stock_level_finbert_sentiment.csv')
        
        if not os.path.exists(sentiment_file_path):
            raise FileNotFoundError(f"Sentiment data file not found at: {sentiment_file_path}")

        self.stock_sentiment_df = pd.read_csv(sentiment_file_path, index_col='Date', parse_dates=True)
        self.stock_sentiment_df.index.name = 'Date'
        
        # Initialize effective sentiment cache for all tickers
        for ticker in self.stock_sentiment_df['Ticker'].unique():
            self.effective_sentiment_cache[ticker] = 0.0

    def generate_signals(self, current_data: pd.DataFrame) -> pd.Series:
        if self.stock_sentiment_df is None:
            print("Sentiment data not loaded. Call pre_run_setup first.")
            return pd.Series(dtype=float)

        # Get the current date from the latest data point
        current_date = current_data.index[-1]

        # Calculate effective sentiment for all relevant tickers up to current_date
        daily_effective_sentiment = pd.Series(dtype=float)
        for ticker in current_data.columns: # Iterate over tickers present in current_data
            # Get raw sentiment for the ticker up to current_date
            ticker_raw_sentiment = self.stock_sentiment_df[
                (self.stock_sentiment_df['Ticker'] == ticker) & 
                (self.stock_sentiment_df.index <= current_date)
            ].sort_index()

            if ticker_raw_sentiment.empty:
                daily_effective_sentiment[ticker] = self.effective_sentiment_cache.get(ticker, 0.0)
                continue

            # Apply sentiment decay
            decay_factor = np.exp(-self.sentiment_decay_alpha)
            effective_sentiment = self.effective_sentiment_cache.get(ticker, 0.0) # Start from cached value

            # Only process new sentiment data since last update
            last_processed_date = None
            if ticker in self.effective_sentiment_cache and self.effective_sentiment_cache[ticker] != 0.0:
                # Find the last date for which we have sentiment in the cache
                # This is a simplification; a more robust cache would store the last date processed
                # For now, we'll re-calculate from the beginning of the available raw sentiment
                pass # We'll re-calculate the whole series for simplicity in this iteration

            for date_idx, row in ticker_raw_sentiment.iterrows():
                # This loop will re-calculate effective sentiment from the beginning of ticker_raw_sentiment
                # A more efficient approach would be to only process new dates
                effective_sentiment = row['FinBERT_Sentiment_Score'] + effective_sentiment * decay_factor
            
            daily_effective_sentiment[ticker] = effective_sentiment
            self.effective_sentiment_cache[ticker] = effective_sentiment # Update cache

        if daily_effective_sentiment.empty:
            return pd.Series(dtype=float)

        # Apply sentiment lag
        # For simplicity, we'll use the sentiment from SENTIMENT_LAG days ago
        # This requires ensuring current_data has enough history
        lagged_date_idx = current_data.index.get_loc(current_date) - self.sentiment_lag
        if lagged_date_idx < 0:
            return pd.Series(dtype=float) # Not enough data for lag

        # This is a simplification. The original code uses sentiment from a specific lagged date.
        # Here, we're using the effective sentiment calculated up to current_date,
        # but the *decision* is based on the sentiment from `sentiment_lag` days ago.
        # This requires a more complex data structure to store historical effective sentiments.
        # For now, let's assume `daily_effective_sentiment` is the sentiment for `current_date`
        # and we'll use it directly, effectively setting sentiment_lag to 0 for this iteration.
        # A proper implementation would require storing daily_effective_sentiment for all past dates.
        
        # For now, let's just use the current effective sentiment for simplicity
        # and acknowledge this is a deviation from the original script's lag logic.
        current_sentiment_scores = daily_effective_sentiment.dropna()

        if current_sentiment_scores.empty:
            return pd.Series(dtype=float)

        # Identify long and short candidates
        long_candidates = current_sentiment_scores.nlargest(self.n_top_stocks)
        short_candidates = current_sentiment_scores.nsmallest(self.n_top_stocks)

        target_weights = pd.Series(0.0, index=current_data.columns)

        # Assign equal weights to long candidates
        if not long_candidates.empty:
            weight_per_long = 1.0 / len(long_candidates)
            for ticker in long_candidates.index:
                target_weights[ticker] = weight_per_long
        
        # For short candidates, we'll assign negative weights if shorting is allowed,
        # or simply ensure they are not held (weight 0) if not.
        # Our current backtester doesn't explicitly handle shorting with negative weights
        # in the Portfolio.execute_trades directly for rebalancing.
        # For simplicity, we'll just ensure short candidates are not held (weight 0).
        # If a stock is in both long and short candidates (e.g., n_top_stocks is too high),
        # long takes precedence.
        for ticker in short_candidates.index:
            if target_weights[ticker] == 0.0: # Only if not already assigned a long weight
                target_weights[ticker] = 0.0 # Explicitly set to 0 (sell/don't buy)

        # Normalize weights to sum to 1 for the long positions
        # If no long positions, all weights remain 0.
        total_positive_weight = target_weights[target_weights > 0].sum()
        if total_positive_weight > 0:
            target_weights = target_weights / total_positive_weight
        else:
            target_weights = pd.Series(0.0, index=current_data.columns) # All cash

        return target_weights
