from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
from wavelet_denoiser import wavelet_denoise, rolling_wavelet_denoise
import pandas as pd
import numpy as np


class WaveletMACrossover(Strategy):
    """
    MA crossover on wavelet-denoised close prices.

    Pipeline:
      1. Denoise the raw close price directly inside each rolling window.
         This gives a smooth local price track without the drift artifact
         that comes from denoising returns and then reconstructing via cumprod.
      2. Run the same fast/slow MA crossover logic as MovingAverageCrossover,
         but on the denoised close (wavelet_close) instead of raw close.

    Note on stationarity: the academic recommendation is to denoise returns
    (stationary) rather than price (non-stationary). However, for the
    rolling-window version, within any 252-bar window the price series is
    approximately locally stationary — the trend doesn't shift enough to
    violate the noise model materially. Denoising price directly avoids
    the compounding drift that plagues the returns-then-reconstruct approach.

    threshold_scale dials how aggressively to suppress detail:
      1.0 = Donoho-Johnstone universal threshold (strong suppression)
      0.5 = half the threshold (gentler — recommended for price series)

    `mode` parameter: 'rolling' (default, causal) or 'global' (look-ahead,
    for honesty-check comparisons only).
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        wavelet: str = "db6",
        mode: str = "rolling",
        rolling_window: int = 252,
        threshold_scale: float = 0.5,
    ):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be positive.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be shorter than slow_period.")
        if mode not in ("rolling", "global"):
            raise ValueError("mode must be 'rolling' or 'global'.")

        super().__init__(
            name=f"Wavelet({mode})+MA Crossover {fast_period}/{slow_period}"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.wavelet = wavelet
        self.mode = mode
        self.rolling_window = rolling_window
        self.threshold_scale = threshold_scale

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        # --- Step 1: denoise close price directly ---
        # rolling_wavelet_denoise() is causal: at each bar t it only sees the
        # prior `rolling_window` bars, so no look-ahead when mode='rolling'.
        if self.mode == "rolling":
            wavelet_close = rolling_wavelet_denoise(
                df['close'],
                window=self.rolling_window,
                wavelet=self.wavelet,
                mode="soft",
                threshold_scale=self.threshold_scale,
            )
        else:
            # Global: one-shot over the full series — uses future data.
            # Only for offline comparison (honesty check).
            wavelet_close = wavelet_denoise(
                df['close'],
                wavelet=self.wavelet,
                mode="soft",
                threshold_scale=self.threshold_scale,
            )

        df['wavelet_close'] = wavelet_close
        # Drop warm-up rows where the rolling window wasn't yet full
        df = df.dropna(subset=['wavelet_close']).copy()

        # --- Step 2: MA crossover on denoised close ---
        df['fast_ma'] = df['wavelet_close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['wavelet_close'].rolling(window=self.slow_period).mean()
        df['signal'] = 0.0

        # Same cross logic as MovingAverageCrossover
        df.loc[
            (df['fast_ma'] > df['slow_ma'])
            & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)),
            'signal'
        ] = 1
        df.loc[
            (df['fast_ma'] < df['slow_ma'])
            & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)),
            'signal'
        ] = -1

        df = df.dropna()

        # --- Step 3: wavelet-derived stop loss ---
        # The noise component is exactly what the wavelet removed: close - wavelet_close.
        # Its rolling std is the noise sigma in price units — the same sigma the
        # denoiser uses internally. Scaling by the same formula gives a SL width
        # that sits at the edge of the noise band: moves inside it are noise,
        # moves outside it are signal. Setting SL just beyond the noise floor
        # means we only exit when the signal actually reverses, not on noise.
        noise = df['close'] - df['wavelet_close']
        # min_periods=20 lets the estimate start as soon as there's enough data
        # rather than waiting for a full second rolling_window warm-up period.
        sigma_price = noise.rolling(window=self.rolling_window, min_periods=20).std()
        threshold_price = sigma_price * np.sqrt(2.0 * np.log(self.rolling_window)) * self.threshold_scale

        # Never tighten the SL below what the raw bar already gives.
        # On volatile bars the wavelet threshold can be smaller than the bar range,
        # so we always take the wider of the two levels.
        df['stop_loss'] = np.nan
        df.loc[df['signal'] == 1,  'stop_loss'] = np.minimum(
            df['wavelet_close'] - threshold_price, df['low'])
        df.loc[df['signal'] == -1, 'stop_loss'] = np.maximum(
            df['wavelet_close'] + threshold_price, df['high'])

        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = WaveletMACrossover(fast_period=20, slow_period=50, mode="rolling")
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2010-01-01", "2026-04-15")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'wavelet_close', 'fast_ma', 'slow_ma', 'signal']].tail(10))
    print(f"BUY  signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
