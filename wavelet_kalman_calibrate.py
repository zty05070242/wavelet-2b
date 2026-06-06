"""
wavelet_kalman_calibrate.py

Uses a global (offline, look-ahead) wavelet spectral analysis of return data
to derive data-driven Q parameters for the Kalman crossover strategy, then
compares those parameters against the hand-tuned defaults.

ACADEMIC / OFFLINE USE ONLY
----------------------------
The wavelet decomposition here uses the full 2000-2026 series, so the derived
Q values are informed by future data relative to any point in the backtest.
This makes the calibration unsuitable for live trading as-is. For production
use, re-run this analysis on a rolling training window (e.g. calibrate on the
prior 5 years, re-calibrate annually).

The purpose here is to ask: is there a principled, data-driven Q that beats
the hand-picked one, even under these idealised conditions?

Method
------
1. Decompose daily returns via DWT (db6, full decomposition).
2. At each detail level j, band centre period ≈ 2^(j+0.5) bars.
3. Energy at level j: E_j = mean(cD_j²).
4. Noise floor: sigma estimated from finest detail band (MAD / 0.6745), then
   noise_energy = sigma².
5. SNR_j = E_j / noise_energy — levels where SNR > 1 carry real signal.
6. Pick the two highest-SNR levels (at least 1 octave apart) as the fast and
   slow signal bands.
7. Convert centre period T to Kalman Q via Q = 4 / (T² - 1).
   Derivation: for 1D random walk (Q, R=1), steady-state gain K = 2/(T+1),
   and Q = K²/(1-K) simplifies to 4/(T²-1).
"""

import numpy as np
import pandas as pd
import pywt

from data_loader import load_historical_data
from backtester import Backtester
from strategy_folder.kalman_cross import KalmanCrossover

TICKER  = "^GSPC"
START   = "2000-01-01"
END     = "2026-04-15"
WAVELET = "db6"

# Hand-tuned baseline used in existing backtests
BASELINE_FAST_COV = 0.01
BASELINE_SLOW_COV = 0.001


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def Q_to_period(Q: float) -> float:
    """Equivalent MA period for a given Kalman Q (R=1 model)."""
    K = (-Q + np.sqrt(Q**2 + 4 * Q)) / 2
    return 2 / K - 1


def period_to_Q(T: float) -> float:
    """Kalman Q for a desired equivalent MA period T (R=1 model)."""
    return 4.0 / (T**2 - 1)


def wavelet_spectral_analysis(returns: pd.Series, wavelet: str = "db6") -> pd.DataFrame:
    """
    Full-depth DWT of the return series; compute per-band energy and SNR.

    Returns a DataFrame with one row per detail level, sorted finest-first.
    Columns: level, period_lo, period_hi, period_centre, energy, snr, kalman_Q.
    """
    values = np.array(returns.dropna().values, dtype=np.float64, copy=True)
    n = len(values)
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    max_level = max(max_level, 1)

    coeffs = pywt.wavedec(values, wavelet, level=max_level)
    # coeffs[0]  = cA (approximation, coarsest)
    # coeffs[1:] = cD levels, finest last (level 1 = coeffs[-1])

    # Noise sigma from the finest detail band (level 1 = coeffs[-1])
    sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745
    noise_energy = sigma_noise ** 2

    rows = []
    n_detail = len(coeffs) - 1  # number of detail bands
    for idx, cD in enumerate(reversed(coeffs[1:]), start=1):
        # Level j: periods in [2^j, 2^(j+1)], centre at 2^(j+0.5)
        period_lo     = 2 ** idx
        period_hi     = 2 ** (idx + 1)
        period_centre = 2 ** (idx + 0.5)

        energy = float(np.mean(cD ** 2))
        snr    = energy / noise_energy if noise_energy > 0 else 0.0
        Q      = period_to_Q(period_centre) if period_centre > 1 else None

        rows.append({
            'level':          idx,
            'period_lo':      period_lo,
            'period_hi':      period_hi,
            'period_centre':  round(period_centre, 1),
            'energy':         round(energy, 8),
            'snr':            round(snr, 3),
            'kalman_Q':       round(Q, 8) if Q else None,
        })

    return pd.DataFrame(rows)


def pick_fast_slow_bands(spec: pd.DataFrame, min_level: int = 3) -> tuple:
    """
    From the spectral table, return (fast_band, slow_band) rows.

    Only considers levels >= min_level (ignores finest 2 levels which are
    noise-dominated by construction).  Picks the two highest-SNR levels that
    are at least 1 octave apart, then assigns the lower-level one as fast
    (higher frequency) and the higher-level one as slow (lower frequency).
    """
    candidates = spec[spec['level'] >= min_level].sort_values('snr', ascending=False)

    band_a = candidates.iloc[0]
    # Second band: highest SNR at least 1 octave away from band_a
    others = candidates[abs(candidates['level'] - band_a['level']) >= 1]
    if others.empty:
        # All candidates are at the same level — just take the next row
        others = candidates.iloc[1:]
    band_b = others.iloc[0]

    # Lower level = higher frequency = fast; higher level = lower frequency = slow
    if band_a['level'] < band_b['level']:
        return band_a, band_b
    else:
        return band_b, band_a


# ---------------------------------------------------------------------------
# Backtesting helpers
# ---------------------------------------------------------------------------

def run_kalman(fast_cov: float, slow_cov: float, df: pd.DataFrame, label: str) -> dict:
    strategy = KalmanCrossover(fast_cov=fast_cov, slow_cov=slow_cov)
    bt = Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001)
    return bt.run(df, strategy, verbose=False)


def print_metrics(label: str, fast_cov: float, slow_cov: float, m: dict):
    T_fast = Q_to_period(fast_cov)
    T_slow = Q_to_period(slow_cov)
    print(f"\n{label}")
    print(f"  fast_cov={fast_cov:.6f} (≈{T_fast:.0f}-bar period)  "
          f"slow_cov={slow_cov:.6f} (≈{T_slow:.0f}-bar period)")
    print(f"  {'Total return':20s}: {m['total_return_pct']:.2f}%")
    print(f"  {'Sharpe ratio':20s}: {m['sharpe_ratio']:.2f}")
    print(f"  {'Max drawdown':20s}: {m['max_drawdown_pct']:.2f}%")
    print(f"  {'Win rate':20s}: {m['win_rate_pct']:.1f}%")
    print(f"  {'Profit factor':20s}: {m['profit_factor']}")
    print(f"  {'Num trades':20s}: {m['num_trades']}")
    print(f"  {'Avg win':20s}: £{m['avg_win']:,.2f}")
    print(f"  {'Avg loss':20s}: £{m['avg_loss']:,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {TICKER} {START} → {END}")
    df      = load_historical_data(TICKER, START, END)
    returns = df['close'].pct_change().dropna()

    print(f"\n{'='*60}")
    print(f"WAVELET SPECTRAL ANALYSIS  ({WAVELET}, {len(returns)} bars)")
    print(f"{'='*60}")
    spec = wavelet_spectral_analysis(returns, WAVELET)
    print(spec.to_string(index=False))

    # Show equivalent Kalman periods for the baseline parameters
    print(f"\nBaseline hand-tuned Q values:")
    print(f"  fast_cov={BASELINE_FAST_COV}  → equiv. period ≈ {Q_to_period(BASELINE_FAST_COV):.0f} bars")
    print(f"  slow_cov={BASELINE_SLOW_COV} → equiv. period ≈ {Q_to_period(BASELINE_SLOW_COV):.0f} bars")

    print(f"\nTwo dominant signal bands (level ≥ 3, highest SNR, ≥1 octave apart):")
    fast_band, slow_band = pick_fast_slow_bands(spec, min_level=3)
    for band in [fast_band, slow_band]:
        print(f"  Level {int(band['level']):2d} | "
              f"period {int(band['period_lo']):4d}–{int(band['period_hi']):4d} bars | "
              f"centre {band['period_centre']:.1f} bars | "
              f"SNR={band['snr']:.3f} | "
              f"→ Q={band['kalman_Q']:.6f}")

    Q_fast_derived = fast_band['kalman_Q']
    Q_slow_derived = slow_band['kalman_Q']

    print(f"\nDerived Q:  fast={Q_fast_derived:.6f}  slow={Q_slow_derived:.6f}")

    print(f"\n{'='*60}")
    print("BACKTEST COMPARISON  (^GSPC 2000–2026, £10k, 2% risk, 1bp slip)")
    print(f"{'='*60}")

    m_baseline = run_kalman(BASELINE_FAST_COV, BASELINE_SLOW_COV, df, "baseline")
    m_derived  = run_kalman(Q_fast_derived,    Q_slow_derived,    df, "derived")

    print_metrics("Hand-tuned baseline", BASELINE_FAST_COV, BASELINE_SLOW_COV, m_baseline)
    print_metrics("Wavelet-derived Q",   Q_fast_derived,    Q_slow_derived,    m_derived)

    print(f"\n{'='*60}")
    print("DELTA (derived − baseline)")
    print(f"{'='*60}")
    for key, label in [
        ('total_return_pct', 'Total return %'),
        ('sharpe_ratio',     'Sharpe ratio'),
        ('max_drawdown_pct', 'Max drawdown %'),
        ('win_rate_pct',     'Win rate %'),
    ]:
        delta = m_derived[key] - m_baseline[key]
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:20s}: {sign}{delta:.2f}")

    winner = "Wavelet-derived" if m_derived['sharpe_ratio'] > m_baseline['sharpe_ratio'] else "Hand-tuned baseline"
    print(f"\nBetter Sharpe: {winner}")
