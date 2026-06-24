"""lag_features.py
Causal lag channels for the Trends target series.

All shifts are forward-only (shift(k) uses data from t-k), so there is
zero leakage.  Call this inside a rolling-origin fold on the training
slice; the same lags are applied to the test slice at inference time.
"""

import pandas as pd

LAGS = [1, 2, 4, 8, 13, 26, 52]   # weeks; 52-week = annual lag (most important)


def make_lag_features(panel: pd.DataFrame,
                      lags: list = LAGS) -> pd.DataFrame:
    """Build causal lag columns for every crop.

    Parameters
    ----------
    panel : DataFrame, shape (T, N)
        Date-indexed Trends panel; columns = crop names.
    lags  : list of int
        Week offsets.  Default = LAGS.

    Returns
    -------
    DataFrame, shape (T, N * len(lags))
        Columns named  {crop}_lag{k}.
        First max(lags) rows will be NaN — handle downstream.
    """
    frames = []
    for lag in lags:
        shifted = panel.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in panel.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def lag_column_names(crops: list, lags: list = LAGS) -> list:
    """Return ordered list of lag column names (matches make_lag_features output)."""
    return [f"{c}_lag{k}" for k in lags for c in crops]
