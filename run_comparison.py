"""Run the 2B and Wavelet-2B strategies on development and evaluation samples."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from backtester import Backtester
from backtester_scaled import BacktesterScaled
from data_loader import load_historical_data
from strategy_folder._strategy_base_class import Strategy
from strategy_folder.two_b import TwoB
from strategy_folder.wavelet_two_b import WaveletTwoB


START_DATE = "2000-01-01"
END_DATE = "2026-04-15"
EVALUATION_START = "2018-01-01"
INITIAL_BALANCE = 100_000.0
RISK_PCT = 0.02
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 1
MAX_LEVERAGE = 20.0
MAX_TRANCHES = 3
RESULTS_DIR = "results"
ENGINE_VERSION = "0.2.0"


@dataclass(frozen=True)
class FuturesSpec:
    name: str
    multiplier: float
    tick_size: float


# Standard US futures specifications. Yahoo's continuous series remains a
# research proxy; these values only make sizing and costs dimensionally correct.
INSTRUMENTS = {
    "GC=F": FuturesSpec("Gold", 100, 0.10),
    "SI=F": FuturesSpec("Silver", 5_000, 0.005),
    "CL=F": FuturesSpec("WTI Crude", 1_000, 0.01),
    "NG=F": FuturesSpec("Natural Gas", 10_000, 0.001),
    "HG=F": FuturesSpec("Copper", 25_000, 0.0005),
    "ZW=F": FuturesSpec("Wheat", 50, 0.25),
    "ZC=F": FuturesSpec("Corn", 50, 0.25),
    "ZS=F": FuturesSpec("Soybeans", 50, 0.25),
    "KC=F": FuturesSpec("Coffee", 375, 0.05),
    "LE=F": FuturesSpec("Live Cattle", 400, 0.025),
}

BACKTESTERS = [("", Backtester), (" (scaled)", BacktesterScaled)]


class _ReplayStrategy(Strategy):
    def __init__(self, name: str, signals: pd.DataFrame):
        super().__init__(name=name)
        self._cached = signals.copy()

    def generate_signals(self) -> pd.DataFrame:
        self.data = self._cached.copy()
        self._signals_generated = True
        return self.data


def _build_strategies() -> List[Tuple[str, Strategy]]:
    return [
        ("2B Rule", TwoB(lookback=20, confirmation_days=3)),
        ("Wavelet-2B", WaveletTwoB(
            denoise_window=128,
            min_prominence_atr=1.0,
            min_pivot_distance=5,
            pivot_confirm_bars=3,
            prominence_lookback=20,
            confirmation_days=3,
        )),
    ]


def _data_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _make_backtester(engine, spec: FuturesSpec):
    common = dict(
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        contract_multiplier=spec.multiplier,
        commission_per_unit=COMMISSION_PER_CONTRACT,
        tick_size=spec.tick_size,
        slippage_ticks=SLIPPAGE_TICKS,
        max_leverage=MAX_LEVERAGE,
        integer_positions=True,
    )
    if engine is BacktesterScaled:
        common["max_tranches"] = MAX_TRANCHES
    return engine(**common)


def _sample_frames(signals: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    split = pd.Timestamp(EVALUATION_START)
    return [
        ("development", signals.loc[signals.index < split]),
        ("evaluation", signals.loc[signals.index >= split]),
    ]


def _metrics_row(
    ticker: str,
    label: str,
    sample: str,
    frame: pd.DataFrame,
    metrics: dict,
    source_hash: str,
    spec: FuturesSpec,
    repaired_range_fields: int,
) -> dict:
    strategy_parameters = (
        "lookback=20;confirmation_days=3"
        if label.startswith("2B Rule")
        else (
            "denoise_window=128;min_prominence_atr=1.0;min_pivot_distance=5;"
            "pivot_confirm_bars=3;prominence_lookback=20;confirmation_days=3"
        )
    )
    return {
        "engine_version": ENGINE_VERSION,
        "ticker": ticker,
        "strategy": label,
        "strategy_parameters": strategy_parameters,
        "sample": sample,
        "start_date": frame.index.min().date().isoformat(),
        "end_date": frame.index.max().date().isoformat(),
        "source_sha256": source_hash,
        "repaired_range_fields": repaired_range_fields,
        "initial_balance": INITIAL_BALANCE,
        "risk_pct": RISK_PCT,
        "contract_multiplier": spec.multiplier,
        "tick_size": spec.tick_size,
        "commission_per_contract": COMMISSION_PER_CONTRACT,
        "slippage_ticks": SLIPPAGE_TICKS,
        "max_leverage": MAX_LEVERAGE,
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "total_return_pct": metrics["total_return_pct"],
        "exposure_pct": metrics["exposure_pct"],
        "num_trades": metrics["num_trades"],
        "profit_factor": metrics["profit_factor"],
        "total_costs": metrics["total_costs"],
    }


def _plot_equity_curves(ticker: str, curves: list[tuple[str, list]], save_path: str) -> None:
    figure = go.Figure()
    for label, curve in curves:
        equity = pd.DataFrame(curve)
        if not equity.empty:
            figure.add_trace(go.Scatter(
                x=equity["date"], y=equity["balance"], name=label, mode="lines"
            ))
    figure.update_layout(
        title=f"Evaluation-sample mark-to-market equity — {ticker}",
        xaxis_title="Date",
        yaxis_title="Account equity (USD)",
        hovermode="x unified",
    )
    figure.write_html(save_path)


def run_comparison(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or list(INSTRUMENTS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []

    for ticker in tickers:
        if ticker not in INSTRUMENTS:
            raise ValueError(f"No futures specification configured for {ticker}")
        spec = INSTRUMENTS[ticker]
        print(f"\n{ticker} — {spec.name}")
        prices = load_historical_data(ticker, START_DATE, END_DATE)
        source_hash = _data_hash(prices)
        repaired_range_fields = (
            prices.attrs.get("repaired_high_rows", 0)
            + prices.attrs.get("repaired_low_rows", 0)
        )
        evaluation_curves: list[tuple[str, list]] = []

        for strategy_label, strategy in _build_strategies():
            strategy.set_data(prices)
            signals = strategy.generate_signals()

            for sample_name, sample in _sample_frames(signals):
                if sample.empty:
                    continue
                for suffix, engine in BACKTESTERS:
                    label = strategy_label + suffix
                    replay = _ReplayStrategy(strategy.name, sample)
                    metrics = _make_backtester(engine, spec).run(sample, replay, verbose=False)
                    rows.append(_metrics_row(
                        ticker,
                        label,
                        sample_name,
                        sample,
                        metrics,
                        source_hash,
                        spec,
                        repaired_range_fields,
                    ))
                    if sample_name == "evaluation":
                        evaluation_curves.append((label, metrics["equity_curve"]))

        safe_ticker = ticker.replace("=", "_").replace("^", "")
        _plot_equity_curves(
            ticker,
            evaluation_curves,
            os.path.join(RESULTS_DIR, f"equity_{safe_ticker}.html"),
        )

    result = pd.DataFrame(rows)
    stamp = date.today().strftime("%Y%m%d")
    output = os.path.join(RESULTS_DIR, f"comparison_validated_{stamp}.csv")
    result.to_csv(output, index=False)
    print(f"\nSaved {output}")
    return result


if __name__ == "__main__":
    run_comparison()
