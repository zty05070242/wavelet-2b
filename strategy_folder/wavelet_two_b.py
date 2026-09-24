import pandas as pd

from pivot_detector import causal_prominent_pivots
from strategy_folder._strategy_base_class import Strategy
from strategy_folder.two_b_rule import two_b_signals
from wavelet_denoiser import rolling_wavelet_denoise


class WaveletTwoB(Strategy):
    """2B rule using fixed-lag pivots from a trailing wavelet estimate."""

    def __init__(
        self,
        denoise_window: int = 128,
        wavelet: str = "db6",
        threshold_scale: float = 0.5,
        min_prominence_atr: float = 1.0,
        min_pivot_distance: int = 5,
        pivot_confirm_bars: int = 3,
        prominence_lookback: int = 20,
        confirmation_days: int = 3,
        atr_period: int = 14,
    ):
        if denoise_window < 16:
            raise ValueError("denoise_window too small for wavelet decomposition.")
        if confirmation_days < 1 or confirmation_days > 5:
            raise ValueError("confirmation_days must be between 1 and 5.")
        if pivot_confirm_bars < 1:
            raise ValueError("pivot_confirm_bars must be >= 1.")
        if min_pivot_distance < 1:
            raise ValueError("min_pivot_distance must be >= 1.")
        if prominence_lookback < min_pivot_distance:
            raise ValueError("prominence_lookback must be at least min_pivot_distance.")

        super().__init__(
            name=f"Wavelet-2B (win={denoise_window}, prom={min_prominence_atr}ATR)"
        )
        self.denoise_window = denoise_window
        self.wavelet = wavelet
        self.threshold_scale = threshold_scale
        self.min_prominence_atr = min_prominence_atr
        self.min_pivot_distance = min_pivot_distance
        self.pivot_confirm_bars = pivot_confirm_bars
        self.prominence_lookback = prominence_lookback
        self.confirmation_days = confirmation_days
        self.atr_period = atr_period

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        frame = self.data.copy()

        frame['wavelet_close'] = rolling_wavelet_denoise(
            frame['close'],
            window=self.denoise_window,
            wavelet=self.wavelet,
            mode="soft",
            threshold_scale=self.threshold_scale,
        )

        tr = pd.concat([
            frame['high'] - frame['low'],
            (frame['high'] - frame['close'].shift(1)).abs(),
            (frame['low'] - frame['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        frame['atr'] = tr.rolling(window=self.atr_period).mean()

        frame = frame.dropna(subset=['wavelet_close', 'atr']).copy()

        denoised = frame['wavelet_close'].to_numpy()
        atrs = frame['atr'].to_numpy()
        highs = frame['high'].to_numpy()
        lows = frame['low'].to_numpy()
        closes = frame['close'].to_numpy()
        active_swing_high, active_swing_low = causal_prominent_pivots(
            denoised,
            highs,
            lows,
            atrs,
            min_prominence_atr=self.min_prominence_atr,
            min_distance=self.min_pivot_distance,
            confirm_bars=self.pivot_confirm_bars,
            prominence_lookback=self.prominence_lookback,
        )

        frame['swing_high'] = active_swing_high
        frame['swing_low'] = active_swing_low

        frame['signal'] = two_b_signals(
            highs,
            lows,
            closes,
            active_swing_high,
            active_swing_low,
            self.confirmation_days,
        )
        self.data = frame
        self._signals_generated = True

        return frame


if __name__ == "__main__":
    from data_loader import load_historical_data
    from backtester import Backtester
    from chart import plot_signals

    strategy = WaveletTwoB(
        denoise_window=128,
        wavelet="db6",
        threshold_scale=0.5,
        min_prominence_atr=1.0,
        min_pivot_distance=5,
        pivot_confirm_bars=3,
        prominence_lookback=20,
        confirmation_days=3,
        atr_period=14,
    )
    df = load_historical_data("SI=F", "2000-01-01", "2026-04-15")
    Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001).run(
        df, strategy, verbose=True
    )
    plot_signals(strategy)
