"""stl_decomp.py
STL (Seasonal-Trend decomposition using Loess) remainder extraction.

The STL REMAINDER — not the raw series — feeds graph edge construction
(Phase 4: graphical lasso, lagged cross-correlation, transfer entropy).
Removing trend + seasonal components prevents spurious edges that simply
reflect shared seasonality rather than direct conditional dependence.

IMPORTANT: always call compute_stl_remainders() on the TRAINING SLICE
of each rolling-origin fold.  Never fit on the full sample.
"""

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
    """Extract STL remainders for every crop in the training panel.

    Parameters
    ----------
    panel      : DataFrame (T, N), training slice only — NO test data.
    period     : seasonal period in weeks (default 52 = annual).
    robust     : use LOWESS robustness iterations (recommended for Trends data
                 which has occasional spikes).
    min_length : minimum non-NaN observations required; shorter series get
                 NaN remainders with a warning.

    Returns
    -------
    DataFrame (T, N) of STL remainders, same index as panel.
    Crops that fail decomposition return a NaN column — these should be
    excluded from edge construction for that fold.
    """
    remainders = {}

    for crop in panel.columns:
        series = panel[crop].dropna()

        if len(series) < min_length:
            warnings.warn(
                f"  STL: '{crop}' has only {len(series)} obs "
                f"(< {min_length}) — remainder set to NaN.",
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
            warnings.warn(f"  STL: '{crop}' failed — {e}.  Remainder set to NaN.")
            remainders[crop] = pd.Series(np.nan, index=panel.index, name=crop)

    return pd.DataFrame(remainders).reindex(panel.index)


def z_normalize_remainders(remainders: pd.DataFrame) -> pd.DataFrame:
    """Z-normalise each crop's remainder (zero mean, unit variance).

    Apply this AFTER compute_stl_remainders() and BEFORE passing residuals
    to graphical lasso / cross-correlation / transfer entropy so that
    high-variance crops don't dominate edge weights.

    Must be called on the training remainder only; apply the same
    (mean, std) to the held-out portion if needed.
    """
    means = remainders.mean()
    stds  = remainders.std().replace(0, 1)   # avoid divide-by-zero
    return (remainders - means) / stds


def stl_components(series: pd.Series,
                   period: int = PERIOD,
                   robust: bool = True) -> pd.DataFrame:
    """Return full STL decomposition for a single crop series.

    Useful for EDA and paper figures.  Returns DataFrame with columns
    [observed, trend, seasonal, resid].
    """
    stl = STL(series.dropna(), period=period, robust=robust)
    fit = stl.fit()
    return pd.DataFrame({
        "observed": fit.observed,
        "trend":    fit.trend,
        "seasonal": fit.seasonal,
        "resid":    fit.resid,
    })


if __name__ == "__main__":
    from src.utils.io import read_panel

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / "crop_list.csv")["crop"].tolist()
    panel = panel[crops]

    print("Computing STL remainders on full panel (demo only — use train fold in practice)…")
    R = compute_stl_remainders(panel)
    Rz = z_normalize_remainders(R)
    print(f"Remainder shape:            {R.shape}")
    print(f"NaN crops: {R.columns[R.isna().all()].tolist()}")
    print(f"Z-normed std (should ≈ 1):  {Rz.std().describe()}")
