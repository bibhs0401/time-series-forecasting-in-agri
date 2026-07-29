'''Causal lag channels for the Trends target series.

All shifts are forward-only (shift(k) uses data from t-k), so there is
no leakage into the future. Call this on the training slice inside each
rolling-origin fold; apply the same lags to the test slice at inference.
'''

import pandas as pd

LAGS = [1, 2, 4, 8, 13, 26, 52]   # weeks; 52 is the annual lag


def make_lag_features(panel: pd.DataFrame,
                      lags: list = LAGS) -> pd.DataFrame:
    '''Build causal lag columns for every crop.

    Parameters:
    panel : DataFrame, shape (T, N)
        Date-indexed Trends panel; columns are crop names.
    lags  : list of int
        Week offsets. Defaults to LAGS.

    Returns:
    DataFrame, shape (T, N * len(lags))
        Columns named {crop}_lag{k}.
        The first max(lags) rows will be NaN; handle that downstream.
    '''
    frames = []
    for lag in lags:
        shifted = panel.shift(lag)
        shifted.columns = [f'{c}_lag{lag}' for c in panel.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def lag_column_names(crops: list, lags: list = LAGS) -> list:
    '''Return ordered lag column names matching make_lag_features output.'''
    return [f'{c}_lag{k}' for k in lags for c in crops]
