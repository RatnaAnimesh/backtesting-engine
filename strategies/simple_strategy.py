import pandas as pd
from strategies.base_strategy import BaseStrategy

class SimpleStrategy(BaseStrategy):
    """
    A very basic strategy that allocates equal weight to the first 3 available assets.
    Used for testing the backtesting engine's core functionality.
    """
    def __init__(self):
        super().__init__()
        print("Simple Strategy Initialized")

    def generate_signals(self, current_data: pd.DataFrame) -> pd.Series:
        """
        Generates target portfolio weights.
        """
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
        print("\n--- Simple Strategy Backtest Summary ---")
        print(f"Final Equity: {results['equity_curve'].iloc[-1]:.2f}")
        # You can add more detailed analysis here later

