'''STL remainder extraction for crop time series.

We use the STL remainder (not the raw series) when building graph edges.
Removing trend and seasonality first makes edges less likely to just
capture shared seasonality rather than a real link between crops.

Fit these decompositions on the training slice only.
'''

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

PERIOD = 52   # annual cycle in weeks


def compute_stl_remainders(panel: pd.DataFrame,
                           period: int = PERIOD,
                           robust: bool = True,
                           min_length: int = 104) -> pd.DataFrame:
    '''Compute STL remainders for every crop in the training panel.

    panel should be training data only. Short series (fewer than
    min_length non-missing points) or failed fits return all-NaN columns
    and should be skipped when building edges.
    '''
    remainders = {}

    for crop in panel.columns:
        series = panel[crop].dropna()

        if len(series) < min_length:
            warnings.warn(
                f"  STL: '{crop}' has only {len(series)} obs "
                f"(< {min_length}); remainder set to NaN.",
                RuntimeWarning,
            )
            remainders[crop] = pd.Series(np.nan, index=panel.index, name=crop)
            continue

        try:
            stl = STL(series, period=period, robust=robust)
            fit = stl.fit()
            remainders[crop] = pd.Series(
                fit.resid, index=series.index, name=crop
            )
        except Exception as e:
            warnings.warn(f"  STL: '{crop}' failed ({e}); remainder set to NaN.")
            remainders[crop] = pd.Series(np.nan, index=panel.index, name=crop)

    return pd.DataFrame(remainders).reindex(panel.index)


def z_normalize_remainders(remainders: pd.DataFrame) -> pd.DataFrame:
    '''Standardize each crop remainder to zero mean and unit variance.

    Fit mean/std on the training remainders, then reuse them for any
    held-out data.
    '''
    means = remainders.mean()
    stds  = remainders.std().replace(0, 1)   # avoid divide-by-zero
    return (remainders - means) / stds


def stl_components(series: pd.Series,
                   period: int = PERIOD,
                   robust: bool = True) -> pd.DataFrame:
    '''Return the full STL decomposition for one crop series.

    Columns: observed, trend, seasonal, resid.
    '''
    stl = STL(series.dropna(), period=period, robust=robust)
    fit = stl.fit()
    return pd.DataFrame({
        'observed': fit.observed,
        'trend':    fit.trend,
        'seasonal': fit.seasonal,
        'resid':    fit.resid,
    })


if __name__ == '__main__':
    from src.utils.io import read_panel

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    panel = panel[crops]

    print('Computing STL remainders on train fold')
    R = compute_stl_remainders(panel)
    Rz = z_normalize_remainders(R)
    print(f'Remainder shape:            {R.shape}')
    print(f'NaN crops: {R.columns[R.isna().all()].tolist()}')
    print(f'Z-normed std (should ≈ 1):  {Rz.std().describe()}')
