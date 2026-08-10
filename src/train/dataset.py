'''Turn the (N, T, C) feature tensor into supervised forecasting samples.

We use direct multi-horizon forecasting: a separate target column per horizon,
all predicted from the same input window ending at the origin. This matches
the rolling-origin evaluation in cv.py and avoids recursive error accumulation.

Sample definition
For an origin position o and input window L:

    inputs  X[:, o-L : o, :]        shape (N, L, C)   weeks [o-L, o)
    target  y[:, o+h-1]  for each h in horizons        shape (N, H)

The last observed week is o-1; the target for horizon h is week o+h-1.
Targets use the model-space (scaled) target channel so the loss is
scale-stable; predictions are inverted to raw space for metrics.

Leakage
This module only slices arrays by position. The feature tensor passed in must
already have been built with train-only scalers (see build_tensor.py); this
module never fits anything.
'''

from __future__ import annotations

import numpy as np


def make_samples(
    X:        np.ndarray,
    y:        np.ndarray,
    origins:  np.ndarray,
    horizons: tuple = (1, 4, 8, 13),
    window:   int = 52,
) -> dict:
    '''Build direct multi-horizon samples for the given forecast origins.

    Parameters
    X        : (N, T, C) feature tensor (already scaled).
    y        : (N, T) target matrix in MODEL space (scaled target channel).
    origins  : 1-D array of origin positions (each observes weeks [0, o)).
    horizons : forecast horizons.
    window   : input window length L (weeks).

    Returns
    dict with
      'X'        : (S, N, L, C) float32   input windows
      'y'        : (S, N, H)    float32   targets (model space)
      'origins'  : (S,) int      origin position for each sample
      'horizons' : tuple         echoed horizons
    Samples whose window would run off the left edge (o < window) or whose
    max-horizon target runs off the right edge are dropped.
    '''
    N, T, C = X.shape
    H = len(horizons)
    hmax = max(horizons)

    Xs, ys, keep = [], [], []
    for o in origins:
        o = int(o)
        if o - window < 0 or o + hmax - 1 >= T:
            continue
        Xs.append(X[:, o - window:o, :])                       # (N, L, C)
        tgt = np.stack([y[:, o + h - 1] for h in horizons], axis=1)  # (N, H)
        ys.append(tgt)
        keep.append(o)

    if not Xs:
        return {'X': np.zeros((0, N, window, C), np.float32),
                'y': np.zeros((0, N, H), np.float32),
                'origins': np.zeros(0, int), 'horizons': tuple(horizons)}

    return {
        'X':        np.asarray(Xs, dtype=np.float32),
        'y':        np.asarray(ys, dtype=np.float32),
        'origins':  np.asarray(keep, dtype=int),
        'horizons': tuple(horizons),
    }


def train_origins(train_end: int, window: int, horizons: tuple,
                  stride: int = 1) -> np.ndarray:
    '''Origins usable for TRAINING inside a fold (targets stay in-sample).

    Valid origins satisfy window <= o and o + max(h) - 1 < train_end,
    i.e. both the input window and every horizon target lie strictly within
    the training slice [0, train_end).
    '''
    hmax = max(horizons)
    lo = window
    hi = train_end - hmax + 1        # exclusive
    if hi <= lo:
        return np.zeros(0, dtype=int)
    return np.arange(lo, hi, stride, dtype=int)


def split_train_val(origins: np.ndarray, val_frac: float = 0.2) -> tuple:
    '''Chronological train/val split of training origins (val = most recent).'''
    n = len(origins)
    if n == 0:
        return origins, origins
    n_val = max(1, int(round(n * val_frac)))
    return origins[:-n_val], origins[-n_val:]
