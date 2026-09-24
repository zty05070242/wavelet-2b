import pandas as pd

from strategy_folder._strategy_base_class import Strategy
from strategy_folder.two_b_rule import two_b_signals


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
        prior_high = frame["high"].shift(1).rolling(self.lookback).max()
        prior_low = frame["low"].shift(1).rolling(self.lookback).min()
        frame["swing_high"] = prior_high
        frame["swing_low"] = prior_low
        frame["signal"] = two_b_signals(
            highs,
            lows,
            closes,
            prior_high.to_numpy(),
            prior_low.to_numpy(),
            self.confirmation_days,
        )
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
