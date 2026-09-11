from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from backtest_metrics import calculate_metrics
from position_sizer import calculate_position_size


class Backtester:
    """Daily-bar, one-position backtester with mark-to-market accounting."""

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_pct: float = 0.02,
        slippage_pct: float = 0.0,
        *,
        contract_multiplier: float = 1.0,
        commission_per_unit: float = 0.0,
        tick_size: float | None = None,
        slippage_ticks: int = 0,
        max_leverage: float = 20.0,
        integer_positions: bool = False,
    ):
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not 0 < risk_pct <= 0.1:
            raise ValueError("risk_pct must be greater than zero and at most 0.1")
        if contract_multiplier <= 0 or max_leverage <= 0:
            raise ValueError("contract_multiplier and max_leverage must be positive")
        if slippage_pct < 0 or commission_per_unit < 0 or slippage_ticks < 0:
            raise ValueError("trading costs cannot be negative")
        if tick_size is not None and tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct
        self.contract_multiplier = contract_multiplier
        self.commission_per_unit = commission_per_unit
        self.tick_size = tick_size
        self.slippage_ticks = slippage_ticks
        self.max_leverage = max_leverage
        self.integer_positions = integer_positions
        self._reset()

    def _reset(self) -> None:
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.entry_date = None
        self.entry_balance = 0.0
        self.entry_cost = 0.0
        self.current_direction = 0
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    @property
    def trade_open(self) -> bool:
        return self.current_direction != 0

    def _fill_price(self, price: float, side: int) -> float:
        """Apply adverse percentage and tick slippage; side is +1 buy / -1 sell."""
        fill = price + side * abs(price) * self.slippage_pct
        if self.tick_size and self.slippage_ticks:
            fill += side * self.tick_size * self.slippage_ticks
            ticks = fill / self.tick_size
            fill = (
                math.ceil(ticks - 1e-12) * self.tick_size
                if side > 0
                else math.floor(ticks + 1e-12) * self.tick_size
            )
        return float(fill)

    @staticmethod
    def _valid_stop(direction: int, entry_price: float, stop_loss: float) -> bool:
        return stop_loss < entry_price if direction == 1 else stop_loss > entry_price

    def _open(self, date, open_price: float, direction: int, stop_loss: float) -> bool:
        if self.current_balance <= 0:
            return False
        entry_price = self._fill_price(open_price, direction)
        if entry_price == 0:
            return False
        if not self._valid_stop(direction, entry_price, stop_loss):
            return False

        sizing = calculate_position_size(
            account_balance=self.current_balance,
            risk_pct=self.risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            contract_multiplier=self.contract_multiplier,
            integer_positions=self.integer_positions,
        )
        max_units = (
            self.current_balance * self.max_leverage
            / (abs(entry_price) * self.contract_multiplier)
        )
        units = min(sizing["units_to_trade"], max_units)
        if self.integer_positions:
            units = math.floor(units)
        if units <= 0:
            return False

        self.position = units
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.entry_date = date
        self.entry_balance = self.current_balance
        self.entry_cost = units * self.commission_per_unit
        self.current_balance -= self.entry_cost
        self.current_direction = direction
        return True

    def _close(self, date, raw_exit_price: float, reason: str) -> None:
        direction = self.current_direction
        exit_price = self._fill_price(raw_exit_price, -direction)
        gross_pnl = (
            (exit_price - self.entry_price)
            * self.position
            * direction
            * self.contract_multiplier
        )
        exit_cost = self.position * self.commission_per_unit
        net_pnl = gross_pnl - self.entry_cost - exit_cost
        self.current_balance += gross_pnl - exit_cost

        self.trades.append({
            "entry_date": self.entry_date,
            "exit_date": date,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "position_size": self.position,
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(net_pnl / self.entry_balance * 100, 2),
            "costs": round(self.entry_cost + exit_cost, 2),
            "exit_reason": reason,
        })

        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.entry_date = None
        self.entry_balance = 0.0
        self.entry_cost = 0.0
        self.current_direction = 0

    def _stop_price(self, row: pd.Series) -> float | None:
        if not self.trade_open:
            return None
        if self.current_direction == 1 and row["low"] <= self.stop_loss:
            return min(float(row["open"]), self.stop_loss)
        if self.current_direction == -1 and row["high"] >= self.stop_loss:
            return max(float(row["open"]), self.stop_loss)
        return None

    def _record_equity(self, date, close_price: float) -> None:
        unrealized = 0.0
        if self.trade_open:
            unrealized = (
                (close_price - self.entry_price)
                * self.position
                * self.current_direction
                * self.contract_multiplier
            )
        point = {
            "date": date,
            "balance": self.current_balance + unrealized,
            "cash_balance": self.current_balance,
            "exposed": self.trade_open,
        }
        if self.equity_curve and self.equity_curve[-1]["date"] == date:
            self.equity_curve[-1] = point
        else:
            self.equity_curve.append(point)

    def run(self, data: pd.DataFrame, strategy, verbose: bool = True) -> Dict:
        self._reset()
        strategy.set_data(data)
        df = strategy.generate_signals()
        if df.empty:
            return calculate_metrics(
                self.initial_balance, self.current_balance, self.trades, self.equity_curve
            )

        pending_signal = 0
        pending_stop = 0.0
        has_custom_stop = "stop_loss" in df.columns

        for date, row in df.iterrows():
            # Decisions made on yesterday's close execute at today's open.
            if self.trade_open and pending_signal == -self.current_direction:
                self._close(date, float(row["open"]), "signal")

            if not self.trade_open and pending_signal:
                self._open(date, float(row["open"]), pending_signal, pending_stop)

            # A position opened today is exposed to today's complete price range.
            stop_price = self._stop_price(row)
            if stop_price is not None:
                self._close(date, stop_price, "stop")

            self._record_equity(date, float(row["close"]))

            pending_signal = int(row["signal"])
            if pending_signal == 1:
                pending_stop = (
                    float(row["stop_loss"])
                    if has_custom_stop and pd.notna(row["stop_loss"])
                    else float(row["low"])
                )
            elif pending_signal == -1:
                pending_stop = (
                    float(row["stop_loss"])
                    if has_custom_stop and pd.notna(row["stop_loss"])
                    else float(row["high"])
                )

        if self.trade_open:
            last_date = df.index[-1]
            self._close(last_date, float(df.iloc[-1]["close"]), "end_of_data")
            self._record_equity(last_date, float(df.iloc[-1]["close"]))

        metrics = calculate_metrics(
            self.initial_balance, self.current_balance, self.trades, self.equity_curve
        )
        if verbose:
            self._print_summary(strategy.name, metrics)
        return metrics

    @staticmethod
    def _print_summary(strategy_name: str, metrics: dict) -> None:
        print(f"Strategy      : {strategy_name}")
        print(f"Final balance : {metrics['final_balance']:,.2f}")
        print(f"Total return  : {metrics['total_return_pct']:.2f}%")
        print(f"CAGR          : {metrics['cagr_pct']:.2f}%")
        print(f"Sharpe        : {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")
        print(f"Exposure      : {metrics['exposure_pct']:.1f}%")
        print(f"Trades        : {metrics['num_trades']}")


if __name__ == "__main__":
    from data_loader import load_historical_data
    from strategy_folder.two_b import TwoB

    prices = load_historical_data("SI=F", "2000-01-01", "2026-04-15")
    Backtester(slippage_pct=0.0001).run(prices, TwoB(), verbose=True)
