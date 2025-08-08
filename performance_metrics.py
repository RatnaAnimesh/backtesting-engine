import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    Assumes daily returns.
    """
    excess_returns = returns - risk_free_rate / 252 # Assuming 252 trading days
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve.
    """
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_cagr(equity_curve: pd.Series) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    """
    if equity_curve.empty:
        return 0.0
    # Ensure index is datetime for date range calculation
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.to_datetime(equity_curve.index)

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

    if num_years <= 0:
        return 0.0

    return (end_value / start_value)**(1 / num_years) - 1

def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculates the annualized volatility of returns.
    Assumes daily returns.
    """
    return returns.std() * np.sqrt(252)

def get_performance_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates a comprehensive set of performance metrics.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Annualized Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0
        }

    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = calculate_cagr(equity_curve)
    annualized_volatility = calculate_volatility(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(equity_curve)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

def create_performance_report(equity_curve: pd.Series, trades: pd.DataFrame, metrics: dict):
    """
    Generates and displays an interactive HTML performance report with multiple plots.
    """
    pio.templates.default = "plotly_dark"

    # Create a figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'scatter', 'rowspan': 2}, {'type': 'table'}],
               [None, {'type': 'scatter'}],
               [{'type': 'histogram', 'colspan': 2}, None]],
        subplot_titles=('Equity Curve & Trades', 'Performance Metrics', 'Drawdown', 'Daily Returns Distribution'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 1. Equity Curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve,
        mode='lines',
        name='Equity',
        line=dict(color='cyan', width=2)
    ), row=1, col=1)

    # Add buy/sell markers from trades DataFrame
    if not trades.empty:
        buys = trades[trades['type'].str.contains('BUY')]
        sells = trades[trades['type'] == 'SELL']
        fig.add_trace(go.Scatter(
            x=buys['date'],
            y=equity_curve.loc[buys['date']],
            mode='markers',
            name='Buys',
            marker=dict(color='lime', size=8, symbol='triangle-up')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sells['date'],
            y=equity_curve.loc[sells['date']],
            mode='markers',
            name='Sells',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ), row=1, col=1)


    # 2. Performance Metrics Table
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.index.name = 'Metric'
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}") # Format values
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'],
                    fill_color='rgba(31, 119, 180, 0.8)',
                    align='left',
                    font=dict(color='white')),
        cells=dict(values=[metrics_df.index, metrics_df.Value],
                   fill_color='rgba(17, 17, 17, 0.8)',
                   align='left',
                   font=dict(color='white'))
    ), row=1, col=2)

    # 3. Drawdown Plot
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red', width=1)
    ), row=2, col=2)


    # 4. Daily Returns Histogram
    returns = equity_curve.pct_change().dropna()
    fig.add_trace(go.Histogram(
        x=returns,
        name='Daily Returns',
        marker_color='royalblue'
    ), row=3, col=1)


    # Update layout
    fig.update_layout(
        title_text='Strategy Performance Dashboard',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show the figure
    fig.show()


# Example Usage (for testing)
if __name__ == "__main__":
    # Create a dummy equity curve
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=252*3, freq='B'))
    # Simulate some returns (e.g., 10% annual return with 15% annual vol)
    np.random.seed(42)
    daily_returns = np.random.normal(0.10/252, 0.15/np.sqrt(252), len(dates))
    equity_curve_data = (1 + daily_returns).cumprod() * 100000 # Start with 100k
    equity_curve = pd.Series(equity_curve_data, index=dates)

    # Create dummy trades
    trade_dates = dates[::30]
    trades_list = []
    for i, date in enumerate(trade_dates):
        trade_type = 'BUY' if i % 2 == 0 else 'SELL'
        trades_list.append({'date': date, 'type': trade_type})
    trades_df = pd.DataFrame(trades_list)


    metrics = get_performance_metrics(equity_curve)
    print("Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    create_performance_report(equity_curve, trades_df, metrics)
