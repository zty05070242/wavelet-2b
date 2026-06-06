import os
import sys
import numpy as np
import pandas as pd
import mplfinance as mpf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_historical_data
from strategy_folder.two_b import TwoB


def plot_twob(ticker: str, start: str, end: str, lookback: int = 200, swing_n: int = 5):
    # --- Load & generate signals ---
    df = load_historical_data(ticker, start, end)

    strategy = TwoB(lookback=lookback, swing_n=swing_n)
    strategy.set_data(df)
    signals = strategy.generate_signals()

    # --- Recompute swing levels so we can draw them on the chart ---
    n = swing_n
    swing_low_mask  = df['low']  == df['low'].rolling(window=2*n+1, center=True).min()
    swing_high_mask = df['high'] == df['high'].rolling(window=2*n+1, center=True).max()

    swing_low_vals  = df['low'].where(swing_low_mask).shift(n)
    swing_high_vals = df['high'].where(swing_high_mask).shift(n)

    def last_valid(x):
        valid = x[~np.isnan(x)]
        return valid[-1] if len(valid) > 0 else np.nan

    prior_swing_low  = swing_low_vals.rolling(window=lookback, min_periods=1).apply(last_valid, raw=True)
    prior_swing_high = swing_high_vals.rolling(window=lookback, min_periods=1).apply(last_valid, raw=True)

    prior_swing_low  = prior_swing_low.reindex(signals.index)
    prior_swing_high = prior_swing_high.reindex(signals.index)

    # --- Build plot DataFrame (mplfinance needs capitalised columns) ---
    plot_df = signals.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    # --- Signal markers ---
    # Place buy triangle just below the candle low, sell triangle just above the high
    buy_markers  = plot_df['Low'].where(signals['signal'] == 1)  * 0.995
    sell_markers = plot_df['High'].where(signals['signal'] == -1) * 1.005

    addplots = [
        mpf.make_addplot(prior_swing_low,  color='blue',  linestyle='--', width=0.8, alpha=0.5),
        mpf.make_addplot(prior_swing_high, color='orange', linestyle='--', width=0.8, alpha=0.5),
    ]
    if buy_markers.notna().any():
        addplots.append(mpf.make_addplot(buy_markers,  type='scatter', markersize=80, marker='^', color='lime'))
    if sell_markers.notna().any():
        addplots.append(mpf.make_addplot(sell_markers, type='scatter', markersize=80, marker='v', color='red'))

    buy_count  = int((signals['signal'] == 1).sum())
    sell_count = int((signals['signal'] == -1).sum())

    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title=f"{ticker} — 2B Reversal  |  lookback={lookback}, swing_n={swing_n}  |  Buys: {buy_count}  Sells: {sell_count}",
        addplot=addplots,
        volume=True,
        figsize=(18, 9),
        show_nontrading=False,
        warn_too_much_data=1000,
    )


if __name__ == "__main__":
    # Narrow the window to keep the chart readable — adjust as needed
    plot_twob("^GSPC", start="2023-01-01", end="2026-04-09")
