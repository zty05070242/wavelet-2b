from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
from pykalman import KalmanFilter
import pandas as pd


class KalmanMAHybrid(Strategy):
    """
    Hybrid crossover: fast Kalman filter vs slow moving average.

    Idea: the Kalman filter reacts quickly with less lag than an MA,
    while the slow MA gives a stable, smooth reference level.
    Crossover of the two generates signals.

    - fast_cov:    Kalman transition_covariance. Higher = more reactive.
    - slow_period: window (bars) for the slow moving average.
    """

    def __init__(self, fast_cov: float = 0.1, slow_period: int = 50):
        if fast_cov <= 0:
            raise ValueError("fast_cov must be positive.")
        if slow_period <= 1:
            raise ValueError("slow_period must be greater than 1.")

        super().__init__(name=f"Kalman/MA Hybrid cov={fast_cov} / MA{slow_period}")
        self.fast_cov = fast_cov
        self.slow_period = slow_period

    def _kalman_smooth(self, series: pd.Series, transition_covariance: float) -> pd.Series:
        """Causal 1D Kalman filter — no look-ahead."""
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

        df['kalman_fast'] = self._kalman_smooth(df['close'], self.fast_cov)
        df['slow_ma']     = df['close'].rolling(window=self.slow_period).mean()
        df['signal']      = 0.0

        # Cross logic: fast Kalman vs slow MA
        df.loc[(df['kalman_fast'] > df['slow_ma']) &
               (df['kalman_fast'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
        df.loc[(df['kalman_fast'] < df['slow_ma']) &
               (df['kalman_fast'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1

        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    strategy = KalmanMAHybrid(fast_cov=0.1, slow_period=50)
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("^GSPC", "2000-01-01", "2026-04-12")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars: {len(signals)}")
    print(signals[['close', 'kalman_fast', 'slow_ma', 'signal']].tail(20))
    print(f"BUY signals : {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")
