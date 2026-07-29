'''Scale-free forecast accuracy metrics and forecast-comparison tests.

Design doc §8.3:
  Primary   : MASE   (scaled vs seasonal-naive, m = 52)
  Secondary : RMSSE  (root mean squared scaled error)
  Report    : MAE, RMSE, sMAPE (unstable near low weeks; flagged)
  Probabilistic : pinball loss + interval coverage
  Directional   : Pearson / Spearman of forecast vs actual
  Significance  : Diebold-Mariano per model pair

Leakage rule
The MASE / RMSSE denominator is the in-sample seasonal-naive error computed on
the training series only (y_train). Never compute the scale from the test
window. All functions here are pure NumPy and never fit anything on test data.

Conventions
y_true, y_pred : 1-D arrays of equal length (one crop, one horizon), OR
                 2-D arrays (samples, ...) metrics reduce over all elements
                 unless noted. NaNs are ignored pairwise.
'''

from __future__ import annotations

import numpy as np

SEASONAL_PERIOD = 52  # weeks; annual cycle used for the seasonal-naive scale


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Flatten and keep only positions finite in both arrays.'''
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def seasonal_naive_scale(y_train: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    '''In-sample mean absolute seasonal-naive error (MASE/RMSSE denominator).

    scale = mean(|y_t - y_{t-m}|) over the training series. Computed on
    y_train ONLY. Falls back to the m=1 naive scale, then to the series
    std, if the season-m differences are unavailable (very short series).
    '''
    y = np.asarray(y_train, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size > m:
        diffs = np.abs(y[m:] - y[:-m])
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size and diffs.mean() > 0:
            return float(diffs.mean())
    if y.size > 1:                                  # fall back to first-diff naive
        d1 = np.abs(np.diff(y))
        d1 = d1[np.isfinite(d1)]
        if d1.size and d1.mean() > 0:
            return float(d1.mean())
    s = float(np.std(y)) if y.size else 0.0         # last resort
    return s if s > 0 else 1.0


def squared_seasonal_naive_scale(y_train: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    '''In-sample mean squared seasonal-naive error (RMSSE denominator, squared).'''
    y = np.asarray(y_train, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size > m:
        diffs = (y[m:] - y[:-m]) ** 2
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size and diffs.mean() > 0:
            return float(diffs.mean())
    if y.size > 1:
        d1 = np.diff(y) ** 2
        d1 = d1[np.isfinite(d1)]
        if d1.size and d1.mean() > 0:
            return float(d1.mean())
    v = float(np.var(y)) if y.size else 0.0
    return v if v > 0 else 1.0


def mae(y_true, y_pred) -> float:
    a, b = _finite_pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else np.nan


def rmse(y_true, y_pred) -> float:
    a, b = _finite_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else np.nan


def smape(y_true, y_pred) -> float:
    '''Symmetric MAPE in [0, 200]. Unstable near low weeks; report only.'''
    a, b = _finite_pair(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    mask = denom > 0
    if not mask.any():
        return np.nan
    return float(np.mean(200.0 * np.abs(a[mask] - b[mask]) / denom[mask]))


def mase(y_true, y_pred, y_train, m: int = SEASONAL_PERIOD) -> float:
    '''Mean Absolute Scaled Error. Scale from y_train seasonal-naive.'''
    a, b = _finite_pair(y_true, y_pred)
    if not a.size:
        return np.nan
    scale = seasonal_naive_scale(y_train, m=m)
    return float(np.mean(np.abs(a - b)) / scale)


def rmsse(y_true, y_pred, y_train, m: int = SEASONAL_PERIOD) -> float:
    '''Root Mean Squared Scaled Error. Scale from y_train seasonal-naive.'''
    a, b = _finite_pair(y_true, y_pred)
    if not a.size:
        return np.nan
    scale2 = squared_seasonal_naive_scale(y_train, m=m)
    return float(np.sqrt(np.mean((a - b) ** 2) / scale2))


def pearson(y_true, y_pred) -> float:
    a, b = _finite_pair(y_true, y_pred)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(y_true, y_pred) -> float:
    a, b = _finite_pair(y_true, y_pred)
    if a.size < 2:
        return np.nan
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    '''Average-tie ranks (avoids a scipy dependency).'''
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # Average ties.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pinball_loss(y_true, y_pred_q, quantiles) -> float:
    '''Average pinball (quantile) loss over quantiles and samples.

    y_pred_q : array (..., Q); last axis indexes quantiles.
    y_true   : array (...) broadcastable to y_pred_q[..., 0].
    '''
    y = np.asarray(y_true, dtype=float)
    yq = np.asarray(y_pred_q, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if yq.shape[-1] != q.size:
        raise ValueError('last axis of y_pred_q must match len(quantiles)')
    y = y[..., None]
    diff = y - yq
    loss = np.maximum(q * diff, (q - 1.0) * diff)   # (..., Q)
    mask = np.isfinite(loss)
    return float(loss[mask].mean()) if mask.any() else np.nan


def interval_coverage(y_true, lower, upper) -> float:
    '''Fraction of observations falling within [lower, upper].'''
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return np.nan
    inside = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
    return float(inside.mean())


def diebold_mariano(y_true, pred_a, pred_b, h: int = 1, loss: str = 'squared'):
    '''Diebold-Mariano test that model A and B have equal expected loss.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a
    Newey-West style HAC variance with (h-1) lags for multi-step forecasts.

    Returns
    (dm_stat, p_value) : floats. p_value from a two-sided normal approx.
                         A negative dm_stat means model A has lower loss
                         (A is better). NaNs returned if insufficient data.
    '''
    a, _ = _finite_pair(y_true, pred_a)   # only to validate shapes
    yt = np.asarray(y_true, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(pa) & np.isfinite(pb)
    yt, pa, pb = yt[mask], pa[mask], pb[mask]
    n = yt.size
    if n < 3:
        return np.nan, np.nan

    if loss == 'squared':
        ea, eb = (yt - pa) ** 2, (yt - pb) ** 2
    elif loss == 'absolute':
        ea, eb = np.abs(yt - pa), np.abs(yt - pb)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = ea - eb
    d_bar = d.mean()

    # HAC (Newey-West) variance of the mean with h-1 lags.
    gamma0 = np.mean((d - d_bar) ** 2)
    var = gamma0
    for lag in range(1, h):
        if lag >= n:
            break
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        var += 2.0 * (1.0 - lag / h) * cov
    var = var / n
    if var <= 0:
        return np.nan, np.nan

    dm = d_bar / np.sqrt(var)
    # Harvey-Leybourne-Newbold correction.
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2.0 * (1.0 - _norm_cdf(abs(dm)))
    return float(dm), float(p)


def _norm_cdf(x: float) -> float:
    '''Standard-normal CDF via erf (math, no scipy).'''
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def evaluate_point(y_true, y_pred, y_train, m: int = SEASONAL_PERIOD) -> dict:
    '''Return all point + directional metrics as a dict.'''
    return {
        'MASE':     mase(y_true, y_pred, y_train, m=m),
        'RMSSE':    rmsse(y_true, y_pred, y_train, m=m),
        'MAE':      mae(y_true, y_pred),
        'RMSE':     rmse(y_true, y_pred),
        'sMAPE':    smape(y_true, y_pred),
        'Pearson':  pearson(y_true, y_pred),
        'Spearman': spearman(y_true, y_pred),
    }
