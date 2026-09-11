"""
Gaussian HMM regime classifier for daily price series.

Two modes:
  decode_regimes_full()    — fits on the full series; NOT causal; for post-hoc analysis only
  rolling_causal_regimes() — rolling refit on trailing window; causal; safe for signal gating

Features: 20-day normalized price change and realized volatility, z-scored within each fit window.
States are named only by the quantity used to order them: realized volatility.
"""

import numpy as np
import pandas as pd

RETURN_WINDOW = 20
VOL_WINDOW    = 20

_STATE_NAMES = {2: ['low_vol', 'high_vol'], 3: ['low_vol', 'medium_vol', 'high_vol']}


def _raw_features(series: pd.Series) -> pd.DataFrame:
    """Features that remain defined when a futures settlement crosses zero."""
    scale = series.abs().rolling(RETURN_WINDOW, min_periods=1).median().shift(1)
    scale = scale.replace(0, np.nan)
    normalized_change = series.diff() / scale
    roll_ret = normalized_change.rolling(RETURN_WINDOW).sum()
    roll_vol = normalized_change.rolling(VOL_WINDOW).std()
    return pd.DataFrame({'ret': roll_ret, 'vol': roll_vol}).dropna()


def _zscale(X_train: np.ndarray, X_apply: np.ndarray | None = None):
    """Z-score X_train; apply same shift+scale to X_apply when given."""
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    scaled = (X_train - mu) / std
    if X_apply is None:
        return scaled
    return scaled, (X_apply - mu) / std


def _state_map(model, n_states: int) -> dict[int, str]:
    """Sort HMM state indices by mean realized vol (col 1) ascending, assign names."""
    order = np.argsort(model.means_[:, 1])
    names = _STATE_NAMES.get(n_states, [f'state_{i}' for i in range(n_states)])
    return {int(order[i]): names[i] for i in range(n_states)}


def _fit(X: np.ndarray, n_states: int, random_state: int):
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states, covariance_type='full',
        n_iter=300, random_state=random_state,
    )
    model.fit(X)
    return model


def _gaussian_log_likelihood(model, observations: np.ndarray) -> np.ndarray:
    """Log emission probability for each observation and HMM state."""
    dimensions = observations.shape[1]
    output = np.empty((len(observations), len(model.means_)))
    constant = dimensions * np.log(2 * np.pi)
    for state, (mean, covariance) in enumerate(zip(model.means_, model.covars_)):
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("HMM covariance matrix is not positive definite")
        difference = observations - mean
        mahalanobis = np.einsum(
            "ij,jk,ik->i", difference, np.linalg.inv(covariance), difference
        )
        output[:, state] = -0.5 * (constant + log_determinant + mahalanobis)
    return output


def _decode_block_causally(model, train_X: np.ndarray, block_X: np.ndarray) -> np.ndarray:
    """Filter a block forward without the backward pass used by Viterbi smoothing."""
    observations = np.vstack([train_X, block_X])
    log_likelihood = _gaussian_log_likelihood(model, observations)
    likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))

    probabilities = model.startprob_ * likelihood[0]
    probabilities /= probabilities.sum()
    states = []
    for index in range(1, len(observations)):
        probabilities = (probabilities @ model.transmat_) * likelihood[index]
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            probabilities = np.full(len(probabilities), 1 / len(probabilities))
        else:
            probabilities /= total
        if index >= len(train_X):
            states.append(int(np.argmax(probabilities)))
    return np.asarray(states, dtype=int)


def decode_regimes_full(
    series: pd.Series,
    n_states: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit HMM on the full series, decode Viterbi path. NOT causal.
    At each bar the model has seen future data — use only for post-hoc analysis.
    """
    feat  = _raw_features(series)
    X     = _zscale(feat.values)
    model = _fit(X, n_states, random_state)
    raw   = model.predict(X)
    smap  = _state_map(model, n_states)
    labels = pd.Series([smap[s] for s in raw], index=feat.index, name='regime')
    return labels.reindex(series.index)      # NaN at leading bars before window fills


def rolling_causal_regimes(
    series: pd.Series,
    train_window: int = 1260,
    refit_every: int = 63,
    n_states: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """
    Causal rolling regime labels. Fits HMM on the trailing train_window bars every
    refit_every bars, then predicts forward. No future data leaks at any bar.

    train_window=1260 (~5 years) and refit_every=63 (~1 quarter) are the defaults.
    First train_window bars are NaN — the causal warmup period.
    """
    feat = _raw_features(series)
    n    = len(feat)
    out  = pd.Series(index=series.index, dtype=object, name='regime')

    i = train_window
    while i < n:
        end     = min(i + refit_every, n)
        train_X = feat.iloc[i - train_window : i].values
        block_X = feat.iloc[i : end].values

        train_Xs, block_Xs = _zscale(train_X, block_X)

        try:
            model = _fit(train_Xs, n_states, random_state)
        except Exception:
            i += refit_every
            continue

        # Re-decode the growing prefix for each date. Decoding the full quarter
        # in one call would let later observations alter earlier state labels.
        block_states = _decode_block_causally(model, train_Xs, block_Xs)
        smap         = _state_map(model, n_states)

        for j, idx in enumerate(feat.index[i : end]):
            out[idx] = smap[block_states[j]]

        i += refit_every

    return out


if __name__ == '__main__':
    from data_loader import load_historical_data

    for ticker in ['NG=F', 'GC=F', 'CL=F']:
        df = load_historical_data(ticker, '2000-01-01', '2026-04-15')
        reg = decode_regimes_full(df['close'])
        pct = reg.value_counts(normalize=True).mul(100).round(1)
        print(f"{ticker}: {pct.to_dict()}")
