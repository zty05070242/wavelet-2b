import numpy as np
import pandas as pd
import yfinance as yf


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a price frame and conservatively repair invalid bar ranges."""
    missing = set(OHLCV_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = data[OHLCV_COLUMNS].dropna().sort_index().copy()
    if frame.empty:
        raise ValueError("No complete OHLCV rows were returned")
    if frame.index.has_duplicates:
        raise ValueError("Price index contains duplicate timestamps")
    if not np.isfinite(frame[["open", "high", "low", "close"]]).all().all():
        raise ValueError("Prices must be finite")
    bad_high = frame["high"] < frame[["open", "close", "low"]].max(axis=1)
    bad_low = frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
    frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
    frame.attrs["repaired_high_rows"] = int(bad_high.sum())
    frame.attrs["repaired_low_rows"] = int(bad_low.sum())
    return frame


def load_historical_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    print(f"Loading {ticker}: {start_date} to {end_date} ({interval})")
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.lower)
    frame = validate_ohlcv(data)
    repaired = frame.attrs["repaired_high_rows"] + frame.attrs["repaired_low_rows"]
    print(f"Loaded {len(frame)} bars; repaired {repaired} invalid range fields")
    return frame


if __name__ == "__main__":
    print(load_historical_data("GC=F", "2020-01-01", "2021-01-01").tail())
