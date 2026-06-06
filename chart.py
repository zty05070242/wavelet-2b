import plotly.graph_objects as go
import pandas as pd


# Standard OHLCV + signal columns that every strategy produces.
# Anything else in the DataFrame is treated as an overlay indicator.
# Helper columns that are NOT on the price scale are excluded here so
# they don't get auto-plotted on the price axis and wreck the chart.
_BASE_COLS = {
    'open', 'high', 'low', 'close', 'volume', 'signal',
    'avg_volume', 'atr', 'stop_loss',
}


def plot_signals(strategy, save_path: str = 'signals_chart.html', show: bool = True):
    """
    Plot a candlestick chart with strategy overlays and buy/sell markers.

    Works with any strategy — auto-detects indicator columns
    (e.g. fast_ma, slow_ma, rsi_upper) and overlays them on the chart.

    Args:
        strategy: A Strategy instance that has already been run (signals generated).
        save_path: Where to save the HTML chart.
        show: If True, opens the chart in the browser.
    """
    signals = strategy.get_signals()
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=signals.index,
                                  open=signals['open'], high=signals['high'],
                                  low=signals['low'], close=signals['close'],
                                  name='Price'))

    # Auto-detect and plot indicator columns (anything not in _BASE_COLS)
    indicator_cols = [col for col in signals.columns if col not in _BASE_COLS]
    for col in indicator_cols:
        fig.add_trace(go.Scatter(x=signals.index, y=signals[col],
                                 name=col, line=dict(width=1)))

    # Buy / sell markers
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                             name='Buy', mode='markers',
                             marker=dict(symbol='triangle-up', size=8, color='green')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                             name='Sell', mode='markers',
                             marker=dict(symbol='triangle-down', size=8, color='red')))

    fig.update_layout(title=f'Signals — {strategy.name}',
                      xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      hovermode='x unified')

    fig.write_html(save_path)
    print(f"Chart saved to {save_path} — open it in your browser")

    if show:
        fig.show()
