"""
run_comparison.py — 2B Rule vs Wavelet-2B across ten commodity futures, with
both the plain and scaled backtesters for each strategy (four rows per ticker).

For each ticker, runs:
  - 2B Rule          + Backtester
  - 2B Rule          + BacktesterScaled
  - Wavelet-2B       + Backtester
  - Wavelet-2B       + BacktesterScaled

Saves:
  - results/comparison_YYYYMMDD.csv — full table across every (ticker, run).
  - results/equity_<ticker>.html    — equity curves overlaid for all 4 runs.

Usage: python run_comparison.py
"""

from __future__ import annotations
import os
from datetime import date
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from data_loader import load_historical_data
from backtester import Backtester
from backtester_scaled import BacktesterScaled

from strategy_folder._strategy_base_class import Strategy
from strategy_folder.two_b import TwoB
from strategy_folder.wavelet_two_b import WaveletTwoB


# --- Configuration ----------------------------------------------------------

DEFAULT_TICKERS = [
    'GC=F',   # Gold
    'SI=F',   # Silver
    'CL=F',   # Crude Oil (WTI)
    'NG=F',   # Natural Gas
    'HG=F',   # Copper
    'ZW=F',   # Wheat
    'ZC=F',   # Corn
    'ZS=F',   # Soybeans
    'KC=F',   # Coffee
    'LE=F',   # Live Cattle
]
START_DATE      = '2000-01-01'
END_DATE        = '2026-04-15'

INITIAL_BALANCE = 10000.0
RISK_PCT        = 0.02
SLIPPAGE_PCT    = 0.0001

RESULTS_DIR     = 'results'

# Strategy params. Kept in one place so the README can reference them.
TWO_B_LOOKBACK       = 20
TWO_B_CONFIRM        = 3
W2B_DENOISE_WINDOW   = 128
W2B_PROMINENCE_ATR   = 1.0
W2B_PIVOT_CONFIRM    = 3

# Scaled backtester params.
MAX_TRANCHES         = 3

LBL_2B         = '2B Rule'
LBL_W2B        = 'Wavelet-2B'
SUFFIX_SCALED  = ' (scaled)'

# Backtester variants iterated for every strategy. Each entry pairs a label
# suffix with a factory that returns a fresh backtester instance.
BACKTESTER_VARIANTS = [
    ('', lambda: Backtester(
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        slippage_pct=SLIPPAGE_PCT,
    )),
    (SUFFIX_SCALED, lambda: BacktesterScaled(
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        slippage_pct=SLIPPAGE_PCT,
        max_tranches=MAX_TRANCHES,
    )),
]


def _build_strategies() -> List[Tuple[str, object]]:
    return [
        (LBL_2B,  TwoB(lookback=TWO_B_LOOKBACK, confirmation_days=TWO_B_CONFIRM)),
        (LBL_W2B, WaveletTwoB(
            denoise_window=W2B_DENOISE_WINDOW,
            min_prominence_atr=W2B_PROMINENCE_ATR,
            pivot_confirm_bars=W2B_PIVOT_CONFIRM,
            confirmation_days=TWO_B_CONFIRM,
        )),
    ]


class _ReplayStrategy(Strategy):
    """
    Wraps a pre-computed signals DataFrame so generate_signals() is essentially
    free. Used to avoid re-running the expensive Wavelet-2B pipeline once per
    backtester variant — we generate signals once per (ticker, strategy) and
    replay them through every backtester.
    """
    def __init__(self, name: str, signals_df: pd.DataFrame):
        super().__init__(name=name)
        self._cached = signals_df.copy()

    def generate_signals(self) -> pd.DataFrame:
        # set_data() (called by Backtester.run) overwrites self.data, so we
        # restore the cached signals here. The backtester then iterates this
        # DataFrame directly via the return value.
        self.data = self._cached.copy()
        self._signals_generated = True
        return self.data


def _metrics_row(ticker: str, label: str, metrics: dict) -> dict:
    """Flatten the metrics dict to the subset we want in the summary table."""
    return {
        'ticker':            ticker,
        'strategy':          label,
        'sharpe_ratio':      metrics['sharpe_ratio'],
        'max_drawdown_pct':  metrics['max_drawdown_pct'],
        'total_return_pct':  round(metrics['total_return_pct'], 2),
        'num_trades':        metrics['num_trades'],
        'win_rate_pct':      metrics['win_rate_pct'],
        'profit_factor':     metrics['profit_factor'],
        'expectancy':        metrics['expectancy'],
    }


def _plot_equity_curves(ticker: str, curves: List[Tuple[str, list]], save_path: str):
    """
    Overlay equity curves for all strategies on one figure.
    `curves` is a list of (label, equity_curve_list) tuples, where each entry
    in equity_curve_list is a {'date': ..., 'balance': ...} dict.
    """
    fig = go.Figure()
    for label, eq in curves:
        if not eq:
            continue
        eq_df = pd.DataFrame(eq)
        fig.add_trace(go.Scatter(
            x=eq_df['date'], y=eq_df['balance'],
            name=label, mode='lines', line=dict(width=1.5),
        ))
    fig.update_layout(
        title=f'Equity curves — {ticker} (initial £{INITIAL_BALANCE:,.0f})',
        xaxis_title='Date', yaxis_title='Balance (£)',
        hovermode='x unified',
    )
    fig.write_html(save_path)
    print(f"  saved equity plot -> {save_path}")


def run_comparison(tickers: List[str] = None,
                   start: str = START_DATE,
                   end: str = END_DATE) -> pd.DataFrame:
    """Main entry point. Returns the assembled results DataFrame."""
    tickers = tickers or DEFAULT_TICKERS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    # Preserve insertion order so the equity-plot legend matches the run order.
    full_labels = [
        s_lbl + bt_suffix
        for s_lbl, _ in _build_strategies()
        for bt_suffix, _ in BACKTESTER_VARIANTS
    ]

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        df = load_historical_data(ticker, start, end)

        results = {}
        for s_label, strat in _build_strategies():
            # Generate signals ONCE per (ticker, strategy) — wavelet denoise
            # is expensive, so we replay the same signal frame through every
            # backtester variant via _ReplayStrategy.
            print(f"  generating signals: {s_label}")
            strat.set_data(df)
            signals_df = strat.generate_signals()

            for bt_suffix, make_bt in BACKTESTER_VARIANTS:
                full_label = s_label + bt_suffix
                print(f"    running backtest: {full_label}")
                replay = _ReplayStrategy(strat.name, signals_df)
                metrics = make_bt().run(df, replay, verbose=False)
                results[full_label] = metrics
                rows.append(_metrics_row(ticker, full_label, metrics))

        equity_curves = [(lbl, results[lbl]['equity_curve']) for lbl in full_labels]
        _plot_equity_curves(
            ticker,
            equity_curves,
            os.path.join(RESULTS_DIR, f"equity_{ticker.replace('^', '').replace('=', '_')}.html"),
        )

    results_df = pd.DataFrame(rows)

    # --- Print per-ticker tables ---
    TICKER_NAMES = {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'CL=F': 'WTI Crude Oil',
        'NG=F': 'Natural Gas',
        'HG=F': 'Copper',
        'ZW=F': 'Wheat',
        'ZC=F': 'Corn',
        'ZS=F': 'Soybeans',
        'KC=F': 'Coffee',
        'LE=F': 'Live Cattle',
    }

    print("\n" + "=" * 72)
    print("RESULTS — 2B Rule vs Wavelet-2B (plain + scaled backtester)")
    print("=" * 72)
    for ticker in tickers:
        name = TICKER_NAMES.get(ticker, ticker)
        sub = results_df[results_df['ticker'] == ticker].drop(columns='ticker')
        print(f"\n{ticker} ({name}):")
        print(sub.to_string(index=False))

    # --- Save CSV ---
    stamp = date.today().strftime('%Y%m%d')
    csv_path = os.path.join(RESULTS_DIR, f'comparison_{stamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return results_df


if __name__ == "__main__":
    run_comparison()
