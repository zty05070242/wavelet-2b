from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
from wavelet_denoiser import wavelet_denoise, rolling_wavelet_denoise
from pykalman import KalmanFilter
import pandas as pd
import numpy as np


class WaveletKalmanCrossover(Strategy):
    """
    Kalman crossover on wavelet-denoised close prices.

    Pipeline:
      1. Denoise the raw close price directly inside each rolling window,
         producing a smooth local price track (wavelet_close).
      2. Run two Kalman filters (fast and slow) on wavelet_close — not on
         raw close — and cross them like a standard Kalman crossover.

    Stacking motivation (DSP analogy): the wavelet step is a broadband
    denoising front-end that removes intraday/tick noise embedded in daily
    bars. The Kalman filter then adaptively tracks the remaining trend.
    Think: LP-filter the signal before feeding it to a tracker.

    threshold_scale dials the wavelet aggressiveness:
      1.0 = universal threshold (strong suppression)
      0.5 = gentler, recommended for price series (default)
    """

    def __init__(
        self,
        fast_cov: float = 0.01,
        slow_cov: float = 0.001,
        wavelet: str = "db6",
        mode: str = "rolling",
        rolling_window: int = 252,
        threshold_scale: float = 0.5,
    ):
        if fast_cov <= slow_cov:
            raise ValueError(
                "fast_cov must be larger than slow_cov "
                "(higher covariance = more responsive filter)."
            )
        if mode not in ("rolling", "global"):
            raise ValueError("mode must be 'rolling' or 'global'.")

        super().__init__(
            name=f"Wavelet({mode})+Kalman Crossover {fast_cov}/{slow_cov}"
        )
        self.fast_cov = fast_cov
        self.slow_cov = slow_cov
        self.wavelet = wavelet
        self.mode = mode
        self.rolling_window = rolling_window
        self.threshold_scale = threshold_scale

    def _kalman_smooth(self, series: pd.Series, transition_covariance: float) -> pd.Series:
        """Causal 1D Kalman filter — no look-ahead (uses .filter(), not .smooth())."""
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=transition_covariance,
        )
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        # --- Step 1: denoise close price directly ---
        if self.mode == "rolling":
            wavelet_close = rolling_wavelet_denoise(
                df['close'],
                window=self.rolling_window,
                wavelet=self.wavelet,
                mode="soft",
                threshold_scale=self.threshold_scale,
            )
        else:
            # Global: uses future data — offline comparison only.
            wavelet_close = wavelet_denoise(
                df['close'],
                wavelet=self.wavelet,
                mode="soft",
                threshold_scale=self.threshold_scale,
            )

        df['wavelet_close'] = wavelet_close
        df = df.dropna(subset=['wavelet_close']).copy()

        # --- Step 2: Kalman crossover on denoised close ---
        df['kalman_fast'] = self._kalman_smooth(df['wavelet_close'], self.fast_cov)
        df['kalman_slow'] = self._kalman_smooth(df['wavelet_close'], self.slow_cov)
        df['signal'] = 0.0

        # Same cross logic as KalmanCrossover
        df.loc[
            (df['kalman_fast'] > df['kalman_slow'])
            & (df['kalman_fast'].shift(1) <= df['kalman_slow'].shift(1)),
            'signal'
        ] = 1
        df.loc[
            (df['kalman_fast'] < df['kalman_slow'])
            & (df['kalman_fast'].shift(1) >= df['kalman_slow'].shift(1)),
            'signal'
        ] = -1

        df = df.dropna()

        # --- Step 3: wavelet-derived stop loss ---
        # noise = what the wavelet removed. Its rolling std is the noise sigma
        # in price units — consistent with the denoiser's own sigma estimate.
        noise = df['close'] - df['wavelet_close']
        sigma_price = noise.rolling(window=self.rolling_window, min_periods=20).std()
        threshold_price = sigma_price * np.sqrt(2.0 * np.log(self.rolling_window)) * self.threshold_scale

        df['stop_loss'] = np.nan
        df.loc[df['signal'] == 1,  'stop_loss'] = np.minimum(
            df['wavelet_close'] - threshold_price, df['low'])
        df.loc[df['signal'] == -1, 'stop_loss'] = np.maximum(
            df['wavelet_close'] + threshold_price, df['high'])

        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = WaveletKalmanCrossover(fast_cov=0.01, slow_cov=0.001, mode="rolling")
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2010-01-01", "2026-04-15")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'wavelet_close', 'kalman_fast', 'kalman_slow', 'signal']].tail(10))
    print(f"BUY  signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
