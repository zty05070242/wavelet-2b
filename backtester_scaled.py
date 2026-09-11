from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from backtest_metrics import calculate_metrics
from position_sizer import calculate_position_size


class BacktesterScaled:
    """Daily-bar backtester that scales in and closes opposite tranches FIFO."""

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_pct: float = 0.02,
        slippage_pct: float = 0.0,
        max_tranches: int = 3,
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
        if max_tranches < 1:
            raise ValueError("max_tranches must be at least one")
        if contract_multiplier <= 0 or max_leverage <= 0:
            raise ValueError("contract_multiplier and max_leverage must be positive")
        if slippage_pct < 0 or commission_per_unit < 0 or slippage_ticks < 0:
            raise ValueError("trading costs cannot be negative")
        if tick_size is not None and tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct
        self.max_tranches = max_tranches
        self.tranche_risk_pct = risk_pct / max_tranches
        self.contract_multiplier = contract_multiplier
        self.commission_per_unit = commission_per_unit
        self.tick_size = tick_size
        self.slippage_ticks = slippage_ticks
        self.max_leverage = max_leverage
        self.integer_positions = integer_positions
        self._reset()

    def _reset(self) -> None:
        self.current_balance = self.initial_balance
        self.long_tranches: list[dict] = []
        self.short_tranches: list[dict] = []
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def _fill_price(self, price: float, side: int) -> float:
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

    def _gross_notional(self, price: float) -> float:
        units = sum(t["size"] for t in self.long_tranches + self.short_tranches)
        return units * abs(price) * self.contract_multiplier

    def _open_tranche(self, date, raw_price: float, direction: int, stop_loss: float) -> bool:
        if self.current_balance <= 0:
            return False
        entry_price = self._fill_price(raw_price, direction)
        if entry_price == 0:
            return False
        if not self._valid_stop(direction, entry_price, stop_loss):
            return False

        sizing = calculate_position_size(
            account_balance=self.current_balance,
            risk_pct=self.tranche_risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            contract_multiplier=self.contract_multiplier,
            integer_positions=self.integer_positions,
        )
        available_notional = max(
            self.current_balance * self.max_leverage - self._gross_notional(entry_price),
            0.0,
        )
        max_units = available_notional / (abs(entry_price) * self.contract_multiplier)
        units = min(sizing["units_to_trade"], max_units)
        if self.integer_positions:
            units = math.floor(units)
        if units <= 0:
            return False

        entry_cost = units * self.commission_per_unit
        tranche = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "size": units,
            "entry_date": date,
            "entry_balance": self.current_balance,
            "entry_cost": entry_cost,
        }
        self.current_balance -= entry_cost
        target = self.long_tranches if direction == 1 else self.short_tranches
        target.append(tranche)
        return True

    def _close_tranche(
        self,
        tranche: dict,
        date,
        raw_exit_price: float,
        direction: int,
        reason: str,
    ) -> None:
        exit_price = self._fill_price(raw_exit_price, -direction)
        gross_pnl = (
            (exit_price - tranche["entry_price"])
            * tranche["size"]
            * direction
            * self.contract_multiplier
        )
        exit_cost = tranche["size"] * self.commission_per_unit
        net_pnl = gross_pnl - tranche["entry_cost"] - exit_cost
        self.current_balance += gross_pnl - exit_cost
        self.trades.append({
            "entry_date": tranche["entry_date"],
            "exit_date": date,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(tranche["entry_price"], 6),
            "exit_price": round(exit_price, 6),
            "position_size": tranche["size"],
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(net_pnl / tranche["entry_balance"] * 100, 2),
            "costs": round(tranche["entry_cost"] + exit_cost, 2),
            "exit_reason": reason,
        })

    def _close_fifo(self, date, raw_price: float, direction: int) -> None:
        tranches = self.long_tranches if direction == 1 else self.short_tranches
        if tranches:
            self._close_tranche(tranches.pop(0), date, raw_price, direction, "signal")

    def _apply_stops(self, date, row: pd.Series) -> None:
        survivors = []
        for tranche in self.long_tranches:
            if row["low"] <= tranche["stop_loss"]:
                raw_exit = min(float(row["open"]), tranche["stop_loss"])
                self._close_tranche(tranche, date, raw_exit, 1, "stop")
            else:
                survivors.append(tranche)
        self.long_tranches = survivors

        survivors = []
        for tranche in self.short_tranches:
            if row["high"] >= tranche["stop_loss"]:
                raw_exit = max(float(row["open"]), tranche["stop_loss"])
                self._close_tranche(tranche, date, raw_exit, -1, "stop")
            else:
                survivors.append(tranche)
        self.short_tranches = survivors

    def _record_equity(self, date, close_price: float) -> None:
        unrealized = sum(
            (close_price - t["entry_price"]) * t["size"] * self.contract_multiplier
            for t in self.long_tranches
        ) + sum(
            (t["entry_price"] - close_price) * t["size"] * self.contract_multiplier
            for t in self.short_tranches
        )
        point = {
            "date": date,
            "balance": self.current_balance + unrealized,
            "cash_balance": self.current_balance,
            "exposed": bool(self.long_tranches or self.short_tranches),
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
            if pending_signal == 1:
                self._close_fifo(date, float(row["open"]), -1)
                if len(self.long_tranches) < self.max_tranches:
                    self._open_tranche(date, float(row["open"]), 1, pending_stop)
            elif pending_signal == -1:
                self._close_fifo(date, float(row["open"]), 1)
                if len(self.short_tranches) < self.max_tranches:
                    self._open_tranche(date, float(row["open"]), -1, pending_stop)

            self._apply_stops(date, row)
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

        last_date = df.index[-1]
        last_close = float(df.iloc[-1]["close"])
        for tranche in list(self.long_tranches):
            self._close_tranche(tranche, last_date, last_close, 1, "end_of_data")
        for tranche in list(self.short_tranches):
            self._close_tranche(tranche, last_date, last_close, -1, "end_of_data")
        self.long_tranches.clear()
        self.short_tranches.clear()
        self._record_equity(last_date, last_close)

        metrics = calculate_metrics(
            self.initial_balance, self.current_balance, self.trades, self.equity_curve
        )
        if verbose:
            print(f"Strategy      : {strategy.name}")
            print(f"Final balance : {metrics['final_balance']:,.2f}")
            print(f"Sharpe        : {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown  : {metrics['max_drawdown_pct']:.2f}%")
            print(f"Trades        : {metrics['num_trades']}")
        return metrics


if __name__ == "__main__":
    from data_loader import load_historical_data
    from strategy_folder.two_b import TwoB

    prices = load_historical_data("ZW=F", "2000-01-01", "2026-04-15")
    BacktesterScaled(slippage_pct=0.0001).run(prices, TwoB(), verbose=True)
