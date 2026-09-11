"""Global and trailing-window wavelet denoisers."""

import numpy as np
import pandas as pd
import pywt


def _estimate_sigma(detail_coefficients: np.ndarray) -> float:
    return float(np.median(np.abs(detail_coefficients)) / 0.6745)


def _universal_threshold(length: int, sigma: float) -> float:
    return sigma * np.sqrt(2 * np.log(length))


def _denoise_array(
    values: np.ndarray,
    wavelet: str,
    level: int | None,
    mode: str,
    threshold_scale: float,
) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=True)
    if level is None:
        level = max(
            pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len),
            1,
        )
    coefficients = pywt.wavedec(values, wavelet, level=level)
    threshold = (
        _universal_threshold(len(values), _estimate_sigma(coefficients[-1]))
        * threshold_scale
    )
    filtered = [coefficients[0], *[
        pywt.threshold(detail, threshold, mode=mode)
        for detail in coefficients[1:]
    ]]
    return pywt.waverec(filtered, wavelet)[:len(values)]


def _validate(mode: str, threshold_scale: float) -> None:
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    if threshold_scale < 0:
        raise ValueError("threshold_scale cannot be negative")


def wavelet_denoise(
    series: pd.Series,
    wavelet: str = "db6",
    level: int | None = None,
    mode: str = "soft",
    threshold_scale: float = 1.0,
) -> pd.Series:
    """Denoise a complete series.

    Reconstruction uses observations on both sides of a timestamp. This function
    is suitable for descriptive analysis, not historical trading signals.
    """
    _validate(mode, threshold_scale)
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    clean = series.dropna()
    if len(clean) < 2:
        return series.copy()
    values = _denoise_array(clean.to_numpy(), wavelet, level, mode, threshold_scale)
    return pd.Series(values, index=clean.index, name=series.name)


def rolling_wavelet_denoise(
    series: pd.Series,
    window: int = 252,
    wavelet: str = "db6",
    mode: str = "soft",
    threshold_scale: float = 1.0,
) -> pd.Series:
    """Return the final reconstructed value from each trailing window."""
    _validate(mode, threshold_scale)
    if window < 8:
        raise ValueError("window must be at least eight samples")

    clean = series.dropna()
    values = clean.to_numpy()
    output = np.full(len(values), np.nan)
    level = max(
        pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len),
        1,
    )
    for end in range(window - 1, len(values)):
        start = end - window + 1
        output[end] = _denoise_array(
            values[start:end + 1], wavelet, level, mode, threshold_scale
        )[-1]
    return pd.Series(output, index=clean.index, name=series.name)


if __name__ == "__main__":
    from data_loader import load_historical_data

    prices = load_historical_data("^GSPC", "2020-01-01", "2026-04-15")
    returns = prices["close"].pct_change().dropna()
    comparison = pd.DataFrame({
        "raw": returns,
        "global": wavelet_denoise(returns),
        "rolling": rolling_wavelet_denoise(returns),
    })
    print(comparison.tail())
