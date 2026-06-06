from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
from pykalman import KalmanFilter
import pandas as pd


class KalmanCrossover(Strategy):
    """
    Two Kalman filters of different responsiveness, crossed like an MA crossover.

    transition_covariance controls how 'jumpy' the filter is:
      - Higher value  = filter reacts faster to price changes (fast line)
      - Lower value   = filter is smoother, lags more (slow line)

    The filter only ever sees data up to the current bar, so this is
    look-ahead-safe.
    """

    def __init__(self, fast_cov: float = 0.1, slow_cov: float = 0.001):
        if fast_cov <= slow_cov:
            raise ValueError("fast_cov must be larger than slow_cov "
                             "(higher covariance = more responsive filter).")

        super().__init__(name=f"Kalman Crossover {fast_cov}/{slow_cov}")
        self.fast_cov = fast_cov
        self.slow_cov = slow_cov

    def _kalman_smooth(self, series: pd.Series, transition_covariance: float) -> pd.Series:
        """Run a 1D Kalman filter over a price series."""
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=transition_covariance,
        )
        # .filter() is causal — each estimate uses only past + current data
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        df['kalman_fast'] = self._kalman_smooth(df['close'], self.fast_cov)
        df['kalman_slow'] = self._kalman_smooth(df['close'], self.slow_cov)
        df['signal'] = 0.0

        # Cross logic: same as MA crossover
        df.loc[(df['kalman_fast'] > df['kalman_slow']) &
               (df['kalman_fast'].shift(1) <= df['kalman_slow'].shift(1)), 'signal'] = 1
        df.loc[(df['kalman_fast'] < df['kalman_slow']) &
               (df['kalman_fast'].shift(1) >= df['kalman_slow'].shift(1)), 'signal'] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = KalmanCrossover(fast_cov=0.01, slow_cov=0.001)
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2000-01-01", "2026-04-12")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'kalman_fast', 'kalman_slow', 'signal']].tail(20))
    print(f"BUY signals : {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
