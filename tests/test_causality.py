import unittest

import numpy as np

from pivot_detector import causal_prominent_pivots
from regime_hmm import _decode_block_causally, _raw_features


class CausalityTests(unittest.TestCase):
    def test_regime_features_support_negative_futures_prices(self):
        import pandas as pd

        series = pd.Series([10.0] * 20 + [5.0, -2.0, 3.0] + [4.0] * 20)
        features = _raw_features(series)
        self.assertFalse(features.empty)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_pivot_history_is_prefix_invariant(self):
        values = np.array([10, 11, 13, 12, 11, 12, 14, 13, 10, 9, 11, 12, 10, 8], dtype=float)
        highs = values + 0.25
        lows = values - 0.25
        atr = np.full(len(values), 0.5)
        kwargs = dict(
            min_prominence_atr=1.0,
            min_distance=2,
            confirm_bars=2,
            prominence_lookback=4,
        )
        full_high, full_low = causal_prominent_pivots(values, highs, lows, atr, **kwargs)

        for length in range(4, len(values) + 1):
            prefix_high, prefix_low = causal_prominent_pivots(
                values[:length], highs[:length], lows[:length], atr[:length], **kwargs
            )
            np.testing.assert_allclose(prefix_high, full_high[:length], equal_nan=True)
            np.testing.assert_allclose(prefix_low, full_low[:length], equal_nan=True)

    def test_regime_block_is_decoded_one_prefix_at_a_time(self):
        class LastObservationModel:
            means_ = np.array([[-1.0], [1.0]])
            covars_ = np.array([[[1.0]], [[1.0]]])
            startprob_ = np.array([0.5, 0.5])
            transmat_ = np.array([[0.8, 0.2], [0.2, 0.8]])

        train = np.array([[-1.0], [-0.5]])
        first = _decode_block_causally(
            LastObservationModel(), train, np.array([[-0.2], [5.0]])
        )
        changed_future = _decode_block_causally(
            LastObservationModel(), train, np.array([[-0.2], [-5.0]])
        )
        self.assertEqual(first[0], changed_future[0])

    def test_complete_rolling_regime_labels_are_prefix_invariant(self):
        import pandas as pd
        from unittest.mock import patch
        from regime_hmm import rolling_causal_regimes

        class DeterministicModel:
            means_ = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]])
            covars_ = np.array([np.eye(2), np.eye(2), np.eye(2)])
            startprob_ = np.full(3, 1 / 3)
            transmat_ = np.full((3, 3), 0.1)
            np.fill_diagonal(transmat_, 0.8)

        rng = np.random.default_rng(11)
        price = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 180))),
            index=pd.date_range("2020-01-01", periods=180, freq="D"),
        )
        kwargs = dict(train_window=60, refit_every=15, n_states=3, random_state=42)
        with patch("regime_hmm._fit", return_value=DeterministicModel()):
            full = rolling_causal_regimes(price, **kwargs)
            for length in (120, 150, 180):
                prefix = rolling_causal_regimes(price.iloc[:length], **kwargs)
                pd.testing.assert_series_equal(
                    prefix,
                    full.loc[prefix.index],
                    check_names=True,
                )

    def test_complete_wavelet_strategy_is_prefix_invariant(self):
        try:
            import pandas as pd
            from strategy_folder.wavelet_two_b import WaveletTwoB
        except ImportError as error:
            self.skipTest(f"optional strategy dependency unavailable: {error}")

        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 1, 360))
        spread = rng.uniform(0.2, 1.0, len(close))
        frame = pd.DataFrame({
            "open": close + rng.normal(0, 0.2, len(close)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1_000,
        }, index=pd.date_range("2020-01-01", periods=len(close), freq="D"))

        def generate(data):
            strategy = WaveletTwoB(
                denoise_window=64,
                min_prominence_atr=0.5,
                min_pivot_distance=3,
                pivot_confirm_bars=2,
                prominence_lookback=12,
            )
            strategy.set_data(data)
            return strategy.generate_signals()

        full = generate(frame)
        for length in (180, 260, 360):
            prefix = generate(frame.iloc[:length])
            expected = full.loc[prefix.index, ["signal", "swing_high", "swing_low"]]
            np.testing.assert_allclose(
                prefix[["signal", "swing_high", "swing_low"]],
                expected,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
