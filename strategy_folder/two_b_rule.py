"""Shared failed-breakout rule used by the 2B strategies."""

import numpy as np


def two_b_signals(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
    confirmation_days: int,
) -> np.ndarray:
    """Return 2B signals for a supplied sequence of active swing levels.

    A high breakout followed by a close below its swing level produces a short
    signal. A low breakout followed by a close above its swing level produces
    a long signal. If both confirm on the same bar, the short takes precedence.
    """
    signals = np.zeros(len(closes))

    for breakout in range(len(closes)):
        high_level = swing_highs[breakout]
        low_level = swing_lows[breakout]
        confirmation_end = min(breakout + confirmation_days + 1, len(closes))

        if highs[breakout] > high_level:
            for confirmed in range(breakout, confirmation_end):
                if closes[confirmed] < high_level:
                    signals[confirmed] = -1.0
                    break

        if lows[breakout] < low_level:
            for confirmed in range(breakout, confirmation_end):
                if closes[confirmed] > low_level:
                    if signals[confirmed] == 0.0:
                        signals[confirmed] = 1.0
                    break

    return signals
