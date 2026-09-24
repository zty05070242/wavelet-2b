# Wavelet-2B: A DSP Approach to Failed-Breakout Detection

While reading Victor Sperandeo's *Trader Vic* books, I became interested in
expressing his 2B Rule in code. The pattern itself is clear: price breaks a prior
extreme, fails to hold it, and reverses. The difficulty appeared when I translated
that visual pattern into code. A human looking at a chart naturally filters out
small fluctuations and recognizes certain extremes as meaningful. A mechanical
rule sees every qualifying high or low literally, so it can fire on price moves
that are little more than noise.

That is a familiar problem from audio engineering. As a tonmeister, I use signal
processing techniques including Fourier analysis and filtering—to decompose a
signal, suppress unwanted components, and reconstruct a cleaner representation.
I wanted to see whether the same principle could give the 2B Rule a cleaner basis
for identifying market pivots. Stefan Jansen's *Machine Learning for Algorithmic
Trading* pointed me toward wavelet decomposition, which can reduce short-term 
noise while retaining the location of meaningful turning points in the price 
series. This led to the experiment in this repository:

```text
price → trailing wavelet denoise → confirmed prominent pivots → 2B rule
```

Wavelet-2B turns the visual judgment that "this swing matters" into a causal,
testable signal. In the current ten-market evaluation it traded less in every
market and reduced maximum drawdown in nine. The evidence supports a specific
conclusion: wavelet-confirmed pivots materially change trade selection and the
risk profile of the 2B Rule.

## From chart pattern to signal

Sperandeo's failed-breakout pattern has two cases:

- **Short:** price breaks above a prior swing high, then closes back below it.
- **Long:** price breaks below a prior swing low, then closes back above it.

The baseline [`TwoB`](strategy_folder/two_b.py) strategy defines the reference
level as the highest high or lowest low of the preceding 20 bars.

[`WaveletTwoB`](strategy_folder/wavelet_two_b.py) changes the definition of the
swing, while leaving the entry rule intact:

1. Denoise the close inside a trailing 128-bar window using a `db6` wavelet.
2. Identify local highs and lows in the reconstructed series.
3. Confirm each pivot after a fixed delay so it cannot be revised later.
4. Require prominence relative to ATR, then use the original bar's high or low
   as the breakout level.

Both strategies pass their swing levels to the same
[`two_b_signals`](strategy_folder/two_b_rule.py) function. This makes the
comparison focused: they trade the same failed-breakout rule, but disagree
about which prior extremes are meaningful.

## What the experiment found

The comparison covers ten commodity futures, using data before 2018 for
development and data from 2018 onward for evaluation.

| Evaluation finding | Result |
|---|---:|
| Markets with fewer Wavelet-2B trades | 10 of 10 |
| Markets with lower Wavelet-2B maximum drawdown | 9 of 10 |
| Median market-level drawdown reduction | 11.1 percentage points |

The strongest effect is selectivity. Wavelet confirmation filters out many of
the short-lived extremes accepted by a rolling lookback, producing a smaller
set of failed-breakout signals. Its effect on return varies by market, while the
change in drawdown is broader across this sample.

Full market-level results—including Sharpe, CAGR, exposure, trade count, profit
factor and modeled costs—are available in
[`results/comparison_validated_20260911.csv`](results/comparison_validated_20260911.csv).

## Research design

The signal and execution paths are causal:

- The wavelet estimate at time `t` uses the trailing window ending at `t`.
- Pivots become available only after their confirmation delay.
- Appending future observations does not alter earlier signals.
- Signals generated at the close execute at the next bar's open.

The prefix-invariance tests in
[`tests/test_causality.py`](tests/test_causality.py) verify these properties by
re-running the strategy on progressively longer versions of the same series.

The comparison starts with a $100,000 account and a 2% planned risk budget.
Sizing uses each market's contract multiplier, tick size and whole-contract
constraint. Every fill includes $2.50 per contract in commission and one tick
of adverse slippage.

Equity is marked to market at each daily close. Entry bars are exposed to their
full price range, stop gaps fill at the opening price, and end-of-data exits are
included in account P&L and the trade log. The execution and accounting tests
are in [`tests/test_backtester.py`](tests/test_backtester.py).

## Running the comparison

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m unittest discover -s tests -v
python3 run_comparison.py
```

`run_comparison.py` downloads the data, runs the baseline and wavelet strategies
through both execution engines, saves a dated CSV, and writes evaluation-period
equity charts to `results/`.

The main entry points are:

- [`run_comparison.py`](run_comparison.py) — ten-market experiment
- [`strategy_folder/wavelet_two_b.py`](strategy_folder/wavelet_two_b.py) —
  wavelet-derived swing levels
- [`strategy_folder/two_b_rule.py`](strategy_folder/two_b_rule.py) — shared 2B
  failed-breakout rule
- [`pivot_detector.py`](pivot_detector.py) — fixed-lag pivot confirmation
- [`backtester.py`](backtester.py) — one-position execution engine
- [`backtester_scaled.py`](backtester_scaled.py) — scaled-entry execution engine
- [`tests/`](tests/) — causality, accounting and data-quality checks

## Data scope

The experiment uses Yahoo's daily continuous-futures histories. They provide a
consistent cross-market research dataset, while contract-level roll execution,
exchange margin changes, partial fills, limit moves and market impact remain
outside the scope of this model.

Some rows contain settlement closes outside the reported session high or low.
The loader expands those ranges to include open and close, records the repaired
fields, and fingerprints the resulting price frame so each output can be tied
to its source data.

## Regime extension

[`regime_hmm.py`](regime_hmm.py) explores whether volatility regimes provide an
additional filter. It uses a three-state Gaussian HMM built from normalized
20-day price change and realized volatility. The rolling implementation refits
on a trailing five-year window and filters forward one observation at a time;
the full-series decoder is retained for retrospective analysis.

Run the extension with:

```bash
python3 run_regime_analysis.py
```

The next stage is to freeze the current specification and evaluate it
prospectively on newly collected data. The broader idea is also reusable beyond
the 2B Rule: fixed-lag, wavelet-confirmed pivots can serve as explicit reference
levels in other support, resistance and failed-breakout systems.

This project demonstrates how a signal-processing idea from audio can turn a
discretionary chart pattern into a causal, reproducible research process.
