import pandas as pd
import yfinance as yf
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'equities')

def _get_local_path(ticker: str) -> str:
    """Constructs the local file path for a given ticker."""
    return os.path.join(DATA_DIR, f"{ticker.upper()}.parquet")

def _fetch_and_save_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetches data from yfinance and saves it locally.
    The 'Close' price is used, as yfinance now defaults to auto-adjusting prices.
    """
    print(f"Fetching {ticker} data from yfinance for {start_date} to {end_date}...")
    try:
        # Fetch data using yfinance, with auto_adjust=True
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)

        if df.empty:
            print(f"No data returned from yfinance for {ticker}.")
            return pd.DataFrame()

        # Rename columns to be consistent (e.g., 'Adj Close' to 'adj_close')
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]

        # Ensure 'date' is datetime index and sorted
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Save to parquet
        local_path = _get_local_path(ticker)
        if os.path.exists(local_path):
            # If file exists, load existing, merge, and save
            existing_df = pd.read_parquet(local_path)
            combined_df = pd.concat([existing_df, df])
            # Remove duplicates based on the index, keeping the last entry
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
            combined_df.to_parquet(local_path)
            print(f"Updated local data for {ticker}.")
            return combined_df
        else:
            df.to_parquet(local_path)
            print(f"Saved new local data for {ticker}.")
            return df
    except Exception as e:
        print(f"Error fetching data for {ticker} from yfinance: {e}")
        return pd.DataFrame()

def get_historical_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Retrieves historical stock data for a given ticker, prioritizing local cache.
    Fetches from yfinance if data is missing or not fully available locally.
    """
    local_path = _get_local_path(ticker)
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Always fetch the full requested date range for simplicity and robustness
    full_data = _fetch_and_save_data(ticker, start_date, end_date)

    # Filter the data to the requested range before returning
    if not full_data.empty:
        if start_date:
            full_data = full_data[full_data.index >= pd.to_datetime(start_date)]
        if end_date:
            full_data = full_data[full_data.index <= pd.to_datetime(end_date)]

    return full_data

def get_multiple_historical_data(tickers: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Retrieves historical data for multiple tickers and combines them into a single DataFrame.
    The DataFrame will have a MultiIndex (Date, Ticker) or Ticker as columns.
    For now, let's return a DataFrame with tickers as columns and 'adj_close' as values.
    """
    all_data = {}
    for ticker in tickers:
        df = get_historical_data(ticker, start_date, end_date)
        print(f"Loaded data for {ticker}: {len(df)} rows") # Add this line
        if not df.empty:
            # We are primarily interested in 'close' now
            all_data[ticker] = df['close']
        else:
            print(f"Warning: No data available for {ticker} in the specified range.")

    if not all_data:
        print("Warning: No data was loaded for any ticker") # Add this line
        return pd.DataFrame()

    # Combine into a single DataFrame, aligning by date
    combined_df = pd.DataFrame(all_data)
    print(f"Combined data shape: {combined_df.shape}") # Add this line
    combined_df.index.name = 'date'
    return combined_df.sort_index()

# Example usage (for testing purposes, will be removed or put in a test file later)
if __name__ == "__main__":
    # Test single ticker
    print("--- Testing single ticker ---")
    aapl_data = get_historical_data("AAPL", start_date="2020-01-01", end_date="2021-12-31")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.head())
    print(aapl_data.tail())

    # Test fetching more data for the same ticker (should use cache and extend)
    print("\n--- Testing extending single ticker data ---")
    aapl_data_extended = get_historical_data("AAPL", start_date="2019-01-01", end_date="2022-12-31")
    print(f"AAPL extended data shape: {aapl_data_extended.shape}")
    print(aapl_data_extended.head())
    print(aapl_data_extended.tail())

    # Test multiple tickers
    print("\n--- Testing multiple tickers ---")
    tickers = ["MSFT", "GOOG", "AMZN"]
    multi_data = get_multiple_historical_data(tickers, start_date="2023-01-01", end_date="2023-12-31")
    print(f"Multi-ticker data shape: {multi_data.shape}")
    print(multi_data.head())
    print(multi_data.tail())

    # Test a ticker that might not exist or has limited data
    print("\n--- Testing non-existent/limited data ticker ---")
    non_existent_data = get_historical_data("NONEXISTENTSTOCK", start_date="2020-01-01", end_date="2020-01-31")
    print(f"Non-existent data shape: {non_existent_data.shape}")