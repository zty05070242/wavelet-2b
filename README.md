# Algorithmic Trading Backtester — DSP-Enhanced 2B Rule

While reading Victor Sperandeo's *Trader Vic* books, I came across his 2B Rule
and wanted to see if I could express it in code. The obvious starting point was
a rolling maximum and minimum: if price breaks a recent extreme and then falls
back through it, trade the failed breakout.

That first version was far noisier than the pattern I could see on a chart. A
rolling maximum treats every spike as meaningful and forgets an older pivot as
soon as it leaves the window. Being a tonmeister, I wondered whether the signal
processing tools I use in audio could help. Stefan Jansen's *Machine Learning
for Algorithmic Trading* pointed me toward wavelet decomposition, which led to
the experiment in this repository:

```text
price → trailing wavelet denoise → confirmed prominent pivots → 2B rule
```

The result is not a magic filter. Wavelet-2B is a more selective version of the
same idea: it trades less, often changes the drawdown profile substantially, and
behaves differently across markets. That is useful in its own right because it
turns a visual concept—"this swing matters"—into something explicit and
testable.

## The experiment

Sperandeo's failed-breakout pattern is straightforward:

- **Short:** price breaks above a prior swing high, then closes back below it.
- **Long:** price breaks below a prior swing low, then closes back above it.

The difficult part is defining a swing. [`TwoB`](strategy_folder/two_b.py) uses
the highest high and lowest low of the preceding 20 bars. It is intentionally
simple and gives the wavelet version a mechanical baseline.

[`WaveletTwoB`](strategy_folder/wavelet_two_b.py) builds its reference levels in
four stages:

1. Denoise the close inside a trailing 128-bar window using a `db6` wavelet.
2. Find local highs and lows in the reconstructed series.
3. Wait for a fixed number of bars before accepting the pivot.
4. Require enough prominence relative to ATR, then use the original bar's high
   or low as the breakout level.

The 2B entry rule is unchanged. Only the source of the swing level differs.

### Causality

Both signal paths are designed to behave the same way in a backtest and at the
edge of a live chart.

- The wavelet estimate at time `t` uses only the trailing window ending at `t`.
- A pivot is accepted after its confirmation delay and is never revised later.
- Appending new data cannot change an already-generated signal.
- Orders generated at the close execute at the next bar's open.

The causality tests run the strategies on progressively longer prefixes of the
same price series and compare every overlapping output. See
[`tests/test_causality.py`](tests/test_causality.py).

## What the current run shows

The main comparison covers ten commodity futures. Results are split into a
development period before 2018 and an evaluation period from 2018 onward. The
table below shows the one-position engine on the evaluation period.

| Market | 2B Sharpe | Wavelet-2B Sharpe | 2B drawdown | Wavelet-2B drawdown |
|---|---:|---:|---:|---:|
| Gold | 0.34 | 0.02 | -42.84% | -33.46% |
| Silver | 0.97 | 0.21 | -39.98% | -34.81% |
| WTI Crude | -0.88 | -0.07 | -73.58% | -32.42% |
| Natural Gas | -0.56 | -0.62 | -81.13% | -52.74% |
| Copper | 0.01 | 0.16 | -57.83% | -53.05% |
| Wheat | 0.15 | 0.17 | -50.75% | -29.08% |
| Corn | -0.65 | -0.40 | -85.66% | -74.20% |
| Soybeans | -0.42 | -0.17 | -81.44% | -52.46% |
| Coffee | 0.05 | -0.29 | -45.77% | -65.77% |
| Live Cattle | -0.30 | -0.74 | -86.34% | -75.54% |

The wavelet detector traded less in every market and reduced drawdown in nine of
ten in this particular comparison. Sharpe improved in five of ten. Performance
remains market-dependent, so the interesting finding is selectivity and risk
shape rather than a universal return advantage.

The scaled engine tells a similar but less uniform story. Its full development
and evaluation output is in
[`results/comparison_validated_20260911.csv`](results/comparison_validated_20260911.csv).
Earlier CSVs are kept as an audit trail; they came from the first version of the
engine and are documented in [`results/README.md`](results/README.md).

## Backtest mechanics

There are two engines with the same strategy interface:

- [`Backtester`](backtester.py) holds one position at a time.
- [`BacktesterScaled`](backtester_scaled.py) opens up to three tranches and
  closes opposing tranches FIFO.

The comparison uses a $100,000 account and a 2% planned risk budget. Contract
multipliers, tick sizes and whole-contract sizing are configured per market.
Every fill pays $2.50 per contract plus one tick of adverse slippage.

Account equity is marked to market at every daily close. New positions are
exposed to the full range of their entry bar, stops that gap are filled at the
opening price, and end-of-data exits are included in both account P&L and the
trade log. Metrics include CAGR, annualized volatility, Sharpe, maximum
drawdown, exposure, profit factor and modeled trading costs.

Each output row also contains its exact dates, assumptions, engine version and a
SHA-256 fingerprint of the downloaded price frame.

## Regime analysis

[`regime_hmm.py`](regime_hmm.py) adds an optional three-state Gaussian HMM using
20-day normalized price change and realized volatility.

The states are called `low_vol`, `medium_vol` and `high_vol`. Those names are
deliberately literal; a medium-volatility state is not automatically a trend.
[`run_regime_analysis.py`](run_regime_analysis.py) provides:

- a retrospective per-regime breakdown of completed trades;
- an evaluation run that suppresses new signals during causally detected
  high-volatility periods.

The rolling version refits on a trailing five-year window and filters forward
one observation at a time. The full-series decoder remains available for
descriptive analysis only.

## What's in the repository

```text
backtester.py               one-position execution engine
backtester_scaled.py        FIFO tranche execution engine
backtest_metrics.py         mark-to-market performance metrics
data_loader.py              Yahoo download and OHLC validation
position_sizer.py           risk and futures contract sizing
pivot_detector.py           fixed-lag causal pivot detection
wavelet_denoiser.py         global and trailing wavelet transforms
regime_hmm.py               retrospective and rolling HMM labels
run_comparison.py           ten-market comparison harness
run_regime_analysis.py      optional regime overlay

strategy_folder/
    two_b.py                rolling high/low baseline
    wavelet_two_b.py        wavelet pivot version
    ma_cross.py             moving-average experiment
    kalman_cross.py         dual-Kalman experiment
    kalman_ma_hybrid.py     Kalman/MA experiment
    wavelet_ma_cross.py     wavelet/MA experiment
    wavelet_kalman_cross.py wavelet/Kalman experiment

tests/                      execution, accounting and causality checks
results/                    dated comparison output and provenance notes
```

## Running it

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m unittest discover -s tests -v
python3 run_comparison.py
python3 run_regime_analysis.py
```

`run_comparison.py` downloads the data, runs both strategies through both
engines, saves a dated CSV, and writes evaluation-period equity charts to
`results/`.

## Notes on the data

The source is Yahoo's daily continuous-futures history. It is convenient for
cross-market research but it is not an executable chain of individual contracts.
Roll construction, exchange margin changes, partial fills, limit moves and
market impact are outside the model.

Some Yahoo futures rows report a settlement close outside the session high/low.
The loader expands the range to include open and close, records the number of
repaired fields, and includes the repaired data in the source hash. This keeps
the process reproducible without silently allowing impossible OHLC bars.

The 2018 evaluation period is separated in code, although earlier iterations of
the project have already looked at those dates. The next clean test is therefore
a frozen configuration on new data or a prospective paper-trading period.

## Where I would take it next

- Plot parameter sensitivity rather than selecting a single best setting.
- Freeze one configuration before collecting new observations.
- Replace continuous Yahoo histories with explicit contract rolls.
- Test the pivot detector as a reusable component in other failed-breakout and
  support/resistance strategies.

The original question was whether signal processing could make a discretionary
chart concept precise enough to test. This repository now provides a useful
answering machine for that question—even when the answer differs by market.
