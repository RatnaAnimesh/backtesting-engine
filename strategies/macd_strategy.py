
import pandas as pd
from .base_strategy import BaseStrategy

def calculate_macd(data: pd.Series, slow_period: int = 26, fast_period: int = 12, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculates MACD indicator for a given data series.
    """
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal})

class MACDStrategy(BaseStrategy):
    """
    A simple strategy based on the MACD indicator.
    - Buys when the MACD line crosses above the Signal line.
    - Sells when the MACD line crosses below the Signal line.
    """
    def __init__(self, slow_period: int = 26, fast_period: int = 12, signal_period: int = 9):
        super().__init__()
        self.slow_period = slow_period
        self.fast_period = fast_period
        self.signal_period = signal_period
        self.weights = pd.Series(dtype=float) # To keep track of current weights

    def generate_signals(self, current_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on the MACD state (above/below signal line).
        """
        if current_data.empty or len(current_data) < self.slow_period:
            return pd.Series(dtype=float)  # Not enough data

        target_weights = pd.Series(0.0, index=current_data.columns)

        for ticker in current_data.columns:
            if current_data[ticker].isnull().all() or len(current_data[ticker].dropna()) < self.slow_period:
                continue  # Skip if not enough data for this specific ticker

            # Calculate MACD
            macd_df = calculate_macd(current_data[ticker], self.slow_period, self.fast_period, self.signal_period)

            if macd_df.empty:
                continue

            # Get the most recent MACD and Signal value
            curr_macd = macd_df['MACD'].iloc[-1]
            curr_signal = macd_df['Signal'].iloc[-1]

            # Buy Signal: MACD is above Signal line (bullish)
            if curr_macd > curr_signal:
                target_weights[ticker] = 1.0  # Assign a target weight (pre-normalization)

            # Sell Signal: MACD is below Signal line (bearish)
            else:
                target_weights[ticker] = 0.0  # Exit position

        # Normalize weights so they sum to 1, maintaining the 0 weights
        total_positive_weight = target_weights.sum()
        if total_positive_weight > 0:
            self.weights = target_weights / total_positive_weight
        else:
            self.weights = target_weights # All weights are 0

        return self.weights

    def post_run_analysis(self, results: dict):
        print("\n--- MACD Strategy Backtest Summary ---")
        print(f"Final Equity: {results['equity_curve'].iloc[-1]:.2f}")
