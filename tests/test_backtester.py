import unittest

import pandas as pd

from backtester import Backtester
from backtester_scaled import BacktesterScaled


class Replay:
    name = "test"

    def set_data(self, data):
        self.data = data.copy()

    def generate_signals(self):
        return self.data.copy()


def prices(rows):
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "signal", "stop_loss"])
    frame["volume"] = 1_000
    frame.index = pd.date_range("2024-01-01", periods=len(frame), freq="D")
    return frame


class BacktesterExecutionTests(unittest.TestCase):
    def test_entry_day_stop_is_not_skipped(self):
        frame = prices([
            [100, 101, 95, 100, 1, 98],
            [100, 103, 97, 102, 0, float("nan")],
            [102, 103, 101, 102, 0, float("nan")],
        ])
        result = Backtester(initial_balance=10_000, risk_pct=0.02).run(
            frame, Replay(), verbose=False
        )

        self.assertEqual(result["num_trades"], 1)
        self.assertEqual(result["trades"][0]["exit_reason"], "stop")
        self.assertEqual(result["final_balance"], 9_800)

    def test_stop_gap_fills_at_open_not_at_stop(self):
        frame = prices([
            [100, 101, 99, 100, 1, 95],
            [100, 102, 99, 100, 0, float("nan")],
            [90, 91, 85, 88, 0, float("nan")],
        ])
        result = Backtester(initial_balance=10_000, risk_pct=0.02).run(
            frame, Replay(), verbose=False
        )

        self.assertEqual(result["trades"][0]["exit_price"], 90)
        self.assertEqual(result["final_balance"], 9_600)

    def test_equity_is_marked_to_market_and_final_trade_is_recorded(self):
        frame = prices([
            [100, 101, 99, 100, 1, 90],
            [100, 106, 99, 105, 0, float("nan")],
            [105, 111, 104, 110, 0, float("nan")],
        ])
        result = Backtester(initial_balance=10_000, risk_pct=0.02).run(
            frame, Replay(), verbose=False
        )

        curve = pd.DataFrame(result["equity_curve"]).set_index("date")
        self.assertEqual(curve.iloc[1]["balance"], 10_100)
        self.assertEqual(result["num_trades"], 1)
        self.assertEqual(result["trades"][0]["exit_reason"], "end_of_data")
        self.assertEqual(result["final_balance"], 10_200)

    def test_drawdown_includes_an_unrealized_loss(self):
        frame = prices([
            [100, 101, 99, 100, 1, 50],
            [100, 101, 99, 100, 0, float("nan")],
            [90, 91, 60, 75, 0, float("nan")],
            [100, 101, 99, 100, 0, float("nan")],
        ])
        result = Backtester(initial_balance=10_000, risk_pct=0.02).run(
            frame, Replay(), verbose=False
        )

        self.assertEqual(result["max_drawdown_pct"], -1.0)
        self.assertEqual(result["final_balance"], 10_000)

    def test_contract_multiplier_integer_sizing_and_costs(self):
        frame = prices([
            [100, 101, 99, 100, 1, 99],
            [100, 101, 98, 100, 0, float("nan")],
        ])
        result = Backtester(
            initial_balance=10_000,
            risk_pct=0.02,
            contract_multiplier=100,
            commission_per_unit=2.50,
            integer_positions=True,
        ).run(frame, Replay(), verbose=False)

        self.assertEqual(result["trades"][0]["position_size"], 2)
        self.assertEqual(result["total_costs"], 10)
        self.assertEqual(result["final_balance"], 9_790)

    def test_signal_is_skipped_when_risk_budget_cannot_buy_one_contract(self):
        frame = prices([
            [100, 101, 99, 100, 1, 50],
            [100, 101, 99, 100, 0, float("nan")],
        ])
        result = Backtester(
            initial_balance=10_000,
            risk_pct=0.02,
            contract_multiplier=100,
            integer_positions=True,
        ).run(frame, Replay(), verbose=False)

        self.assertEqual(result["num_trades"], 0)
        self.assertEqual(result["final_balance"], 10_000)

    def test_negative_futures_price_uses_absolute_notional(self):
        engine = Backtester(initial_balance=10_000, risk_pct=0.02, slippage_pct=0.01)
        self.assertAlmostEqual(engine._fill_price(-10, 1), -9.9)
        self.assertAlmostEqual(engine._fill_price(-10, -1), -10.1)

        frame = prices([
            [1, 2, -1, 1, -1, 3],
            [-1, 0, -3, -2, 0, float("nan")],
            [-2, -1, -5, -4, 0, float("nan")],
        ])
        result = Backtester(initial_balance=10_000, risk_pct=0.02).run(
            frame, Replay(), verbose=False
        )
        self.assertEqual(result["num_trades"], 1)
        self.assertGreater(result["final_balance"], 10_000)

    def test_scaled_engine_records_end_of_data_tranches(self):
        frame = prices([
            [100, 101, 99, 100, 1, 90],
            [100, 102, 99, 101, 1, 90],
            [101, 103, 100, 102, 0, float("nan")],
        ])
        result = BacktesterScaled(
            initial_balance=10_000, risk_pct=0.02, max_tranches=3
        ).run(frame, Replay(), verbose=False)

        self.assertEqual(result["num_trades"], 2)
        self.assertTrue(all(t["exit_reason"] == "end_of_data" for t in result["trades"]))


if __name__ == "__main__":
    unittest.main()
