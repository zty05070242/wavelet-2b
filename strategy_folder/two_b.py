import numpy as np
import pandas as pd

from strategy_folder._strategy_base_class import Strategy


class TwoB(Strategy):
    """Mechanical 2B baseline using prior rolling highs and lows."""

    def __init__(self, lookback: int = 20, confirmation_days: int = 3):
        if lookback < 3:
            raise ValueError("lookback must be at least three bars")
        if not 1 <= confirmation_days <= 5:
            raise ValueError("confirmation_days must be between one and five")
        super().__init__(name=f"2B Rule (lookback={lookback}, confirm={confirmation_days})")
        self.lookback = lookback
        self.confirmation_days = confirmation_days

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded; call set_data() first")

        frame = self.data.copy()
        highs = frame["high"].to_numpy()
        lows = frame["low"].to_numpy()
        closes = frame["close"].to_numpy()
        signals = np.zeros(len(frame))

        prior_high = frame["high"].shift(1).rolling(self.lookback).max()
        prior_low = frame["low"].shift(1).rolling(self.lookback).min()
        frame["swing_high"] = prior_high
        frame["swing_low"] = prior_low

        for breakout in range(self.lookback, len(frame)):
            high_level = prior_high.iloc[breakout]
            low_level = prior_low.iloc[breakout]
            end = min(breakout + self.confirmation_days + 1, len(frame))

            if highs[breakout] > high_level:
                for confirmed in range(breakout, end):
                    if closes[confirmed] < high_level:
                        signals[confirmed] = -1
                        break

            if lows[breakout] < low_level:
                for confirmed in range(breakout, end):
                    if closes[confirmed] > low_level:
                        if signals[confirmed] == 0:
                            signals[confirmed] = 1
                        break

        frame["signal"] = signals
        frame = frame.dropna(subset=["swing_high", "swing_low"])
        self.data = frame
        self._signals_generated = True
        return frame


if __name__ == "__main__":
    from data_loader import load_historical_data

    strategy = TwoB()
    strategy.set_data(load_historical_data("SI=F", "2000-01-01", "2026-04-15"))
    output = strategy.generate_signals()
    print(output[["close", "swing_high", "swing_low", "signal"]].tail())
