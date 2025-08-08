from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self):
        pass

    @abstractmethod
    def generate_signals(self, current_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals for the given data.

        Args:
            current_data (pd.DataFrame): Historical data up to the current date.

        Returns:
            pd.Series: A series with tickers as index and target weights as values.
        """
        raise NotImplementedError("Should implement generate_signals()")

    def pre_run_setup(self, full_data: pd.DataFrame):
        """
        Optional method to perform any setup before the backtest runs.
        This can be used for pre-calculating indicators or other data transformations.
        """
        pass

    def post_run_analysis(self, results: dict):
        """
        Optional method to perform any analysis after the backtest is complete.
        This can be used to print strategy-specific metrics or generate custom plots.
        """
        pass