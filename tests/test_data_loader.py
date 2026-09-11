import unittest

import pandas as pd

try:
    from data_loader import validate_ohlcv
except ImportError:
    validate_ohlcv = None


@unittest.skipIf(validate_ohlcv is None, "yfinance is not installed")
class DataValidationTests(unittest.TestCase):
    def test_range_is_expanded_and_repair_is_reported(self):
        frame = pd.DataFrame({
            "open": [100.0, 100.0],
            "high": [99.0, 103.0],
            "low": [98.0, 101.0],
            "close": [102.0, 100.0],
            "volume": [1_000, 1_000],
        }, index=pd.date_range("2024-01-01", periods=2))

        cleaned = validate_ohlcv(frame)

        self.assertEqual(cleaned.iloc[0]["high"], 102.0)
        self.assertEqual(cleaned.iloc[1]["low"], 100.0)
        self.assertEqual(cleaned.attrs["repaired_high_rows"], 1)
        self.assertEqual(cleaned.attrs["repaired_low_rows"], 1)


if __name__ == "__main__":
    unittest.main()
