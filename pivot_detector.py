from __future__ import annotations

import numpy as np


def causal_prominent_pivots(
    values: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    *,
    min_prominence_atr: float,
    min_distance: int,
    confirm_bars: int,
    prominence_lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the latest confirmed high/low pivot available at each bar.

    A candidate at ``k`` is evaluated only at ``k + confirm_bars``. Its local
    shape and prominence use the history through that confirmation bar, never
    observations after it. Accepted pivots are not revised later.
    """
    arrays = [np.asarray(item, dtype=float) for item in (values, highs, lows, atr)]
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("values, highs, lows and atr must have equal lengths")
    if min_distance < 1 or confirm_bars < 1 or prominence_lookback < 1:
        raise ValueError("distance, confirmation and lookback must be positive")

    values, highs, lows, atr = arrays
    n = len(values)
    high_events: dict[int, float] = {}
    low_events: dict[int, float] = {}
    last_high_k = -min_distance
    last_low_k = -min_distance

    for k in range(1, max(n - confirm_bars, 1)):
        known_at = k + confirm_bars
        if known_at >= n or not np.isfinite(atr[k]):
            continue

        local_left = values[max(0, k - min_distance):k]
        right = values[k + 1:known_at + 1]
        prominence_left = values[max(0, k - prominence_lookback):k]
        if not len(local_left) or not len(right) or not len(prominence_left):
            continue

        is_peak = values[k] > np.max(local_left) and values[k] > np.max(right)
        peak_prominence = values[k] - max(np.min(prominence_left), np.min(right))
        if (
            is_peak
            and peak_prominence >= min_prominence_atr * atr[k]
            and k - last_high_k >= min_distance
        ):
            high_events[known_at] = highs[k]
            last_high_k = k

        is_trough = values[k] < np.min(local_left) and values[k] < np.min(right)
        trough_prominence = min(np.max(prominence_left), np.max(right)) - values[k]
        if (
            is_trough
            and trough_prominence >= min_prominence_atr * atr[k]
            and k - last_low_k >= min_distance
        ):
            low_events[known_at] = lows[k]
            last_low_k = k

    active_high = np.full(n, np.nan)
    active_low = np.full(n, np.nan)
    current_high = np.nan
    current_low = np.nan
    for t in range(n):
        current_high = high_events.get(t, current_high)
        current_low = low_events.get(t, current_low)
        active_high[t] = current_high
        active_low[t] = current_low

    return active_high, active_low
