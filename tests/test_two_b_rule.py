import unittest

import numpy as np

from strategy_folder.two_b_rule import two_b_signals


class TwoBRuleTests(unittest.TestCase):
    def test_breakouts_confirm_within_the_allowed_window(self):
        signals = two_b_signals(
            highs=np.array([11.0, 10.5, 8.5, 9.0]),
            lows=np.array([10.0, 9.0, 7.0, 8.0]),
            closes=np.array([10.5, 9.5, 7.5, 8.5]),
            swing_highs=np.array([10.0, np.nan, np.nan, np.nan]),
            swing_lows=np.array([np.nan, np.nan, 8.0, np.nan]),
            confirmation_days=1,
        )

        np.testing.assert_array_equal(signals, np.array([0.0, -1.0, 0.0, 1.0]))

    def test_short_signal_takes_precedence_when_both_confirm(self):
        signals = two_b_signals(
            highs=np.array([11.0]),
            lows=np.array([9.0]),
            closes=np.array([10.0]),
            swing_highs=np.array([10.5]),
            swing_lows=np.array([9.5]),
            confirmation_days=1,
        )

        np.testing.assert_array_equal(signals, np.array([-1.0]))


if __name__ == "__main__":
    unittest.main()
