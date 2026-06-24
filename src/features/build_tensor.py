"""build_tensor.py
Assemble the multi-view node feature tensor X[node, time, channel].

Feature channel groups
----------------------
  G0  target_lags   — Trends value (t) + lags {1,2,4,8,13,26,52}   shape (N, T, 8)
  G1  fourier       — sin/cos k=1..4 (shared, broadcast)            shape (N, T, 8)
  G2  season_flag   — per-crop in/out-of-season binary               shape (N, T, 1)
  G3  weather       — NOAA TMAX, TMIN, PRCP, GDD (shared, broadcast) shape (N, T, 4)
  G4  static        — one-hot crop attributes (time-invariant)       shape (N, T, D)

Wikipedia pageviews are NOT a feature channel.  They live in
src/data/wiki_corroboration.py as a panel validation tool (Gate A.4).

Leakage rules
-------------
  • All scalers/normalisation must be fit on the training slice only
    and applied to both train and test.
  • Fourier terms and season flags are calendar-derived — no fitting needed.
  • Static attributes are time-invariant — no fitting needed.
  • Weather: fit RobustScaler on training weather rows only.
  • Call build_feature_groups() inside each rolling-origin fold with
    panel_train and weather_train; then apply_scalers() to the test slice.

Ablation
--------
Pass include=["G0","G1","G2"] to concat_groups() to toggle off groups.
This is how the feature-group ablation in Phase 6 works.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

from src.features.lag_features      import make_lag_features, LAGS
from src.features.calendar_features import make_fourier_features, make_season_flags
from src.features.weather_features  import align_to_panel as align_weather
from src.features.static_attributes import get_attribute_matrix

# All group keys in order
ALL_GROUPS = ["G0", "G1", "G2", "G3", "G4"]


# ─── Per-crop RobustScaler for G0 (Trends values + lags) ─────────────────────

def fit_crop_scalers(panel_train: pd.DataFrame) -> dict:
    """Fit one RobustScaler per crop on the training slice.

    Returns a dict {crop: fitted_scaler} — store alongside the fold index
    and reuse for the test slice via transform_crop_scalers().
    """
    scalers = {}
    for crop in panel_train.columns:
        scaler = RobustScaler()
        vals = panel_train[[crop]].to_numpy(dtype=float)
        scaler.fit(vals[~np.isnan(vals).any(axis=1)])
        scalers[crop] = scaler
    return scalers


def transform_crop_scalers(panel: pd.DataFrame,
                            scalers: dict) -> pd.DataFrame:
    """Apply pre-fitted crop scalers to any slice (train or test)."""
    out = panel.copy()
    for crop in panel.columns:
        if crop in scalers:
            out[crop] = scalers[crop].transform(panel[[crop]])
    return out


# ─── Weather scaler ───────────────────────────────────────────────────────────

def fit_weather_scaler(weather_train: pd.DataFrame) -> RobustScaler:
    """Fit RobustScaler on training weather rows (drops NaN rows first)."""
    scaler = RobustScaler()
    arr = weather_train.to_numpy(dtype=float)
    scaler.fit(arr[~np.isnan(arr).any(axis=1)])
    return scaler


# ─── Core builder ─────────────────────────────────────────────────────────────

def build_feature_groups(panel:         pd.DataFrame,
                         weather:       pd.DataFrame,
                         crops:         list,
                         crop_scalers:  dict   = None,
                         weather_scaler: RobustScaler = None,
                         groups:        list   = None) -> dict:
    """Build feature group arrays for one fold slice (train OR test).

    Parameters
    ----------
    panel          : (T, N) Trends panel slice, columns ⊇ crops.
    weather        : (T, 4) NOAA weather slice, aligned to panel index.
    crops          : ordered list of N crop names.
    crop_scalers   : fitted scalers from fit_crop_scalers() on train slice.
                     Pass None to skip scaling (e.g. for quick EDA).
    weather_scaler : fitted scaler from fit_weather_scaler() on train slice.
    groups         : list of group keys to build; default = ALL_GROUPS.

    Returns
    -------
    dict: group_key -> np.ndarray of shape (N, T, C_g)
    """
    if groups is None:
        groups = ALL_GROUPS

    N = len(crops)
    T = len(panel)
    out = {}

    # ── G0: Trends value + causal lags ───────────────────────────────────────
    if "G0" in groups:
        p = panel[crops].copy()
        if crop_scalers:
            p = transform_crop_scalers(p, crop_scalers)

        lag_df = make_lag_features(p, lags=LAGS)   # (T, N*7)

        # For each crop: [value, lag1, lag2, lag4, lag8, lag13, lag26, lag52]
        g0 = np.stack([
            pd.concat([p[[c]],
                       lag_df[[f"{c}_lag{k}" for k in LAGS]]], axis=1).to_numpy()
            for c in crops
        ], axis=0)   # (N, T, 8)
        out["G0"] = g0.astype(np.float32)

    # ── G1: Fourier terms (shared across nodes) ───────────────────────────────
    if "G1" in groups:
        fourier = make_fourier_features(panel.index).to_numpy(dtype=np.float32)  # (T, 8)
        out["G1"] = np.broadcast_to(
            fourier[np.newaxis, :, :], (N, T, fourier.shape[1])
        ).copy()

    # ── G2: Season flags (per crop) ───────────────────────────────────────────
    if "G2" in groups:
        flags = make_season_flags(panel.index, crops).to_numpy(dtype=np.float32)  # (T, N)
        out["G2"] = flags.T[:, :, np.newaxis]   # (N, T, 1)

    # ── G3: Weather (shared across nodes) ────────────────────────────────────
    if "G3" in groups:
        wx_aligned = align_weather(weather, panel.index).to_numpy(dtype=np.float32)  # (T, 4)
        if weather_scaler:
            wx_aligned = weather_scaler.transform(wx_aligned).astype(np.float32)
        out["G3"] = np.broadcast_to(
            wx_aligned[np.newaxis, :, :], (N, T, wx_aligned.shape[1])
        ).copy()

    # ── G4: Static node attributes (time-invariant, broadcast) ───────────────
    if "G4" in groups:
        attr = get_attribute_matrix(crops).to_numpy(dtype=np.float32)  # (N, D)
        out["G4"] = np.broadcast_to(
            attr[:, np.newaxis, :], (N, T, attr.shape[1])
        ).copy()

    return out


def concat_groups(feature_groups: dict,
                  include: list = None) -> np.ndarray:
    """Concatenate selected groups along the channel axis.

    Parameters
    ----------
    feature_groups : dict from build_feature_groups()
    include        : which keys to concat; default = all present keys.

    Returns
    -------
    np.ndarray of shape (N, T, C_total)
    """
    keys = include if include is not None else list(feature_groups.keys())
    missing = [k for k in keys if k not in feature_groups]
    if missing:
        raise KeyError(f"Groups not built: {missing}")
    return np.concatenate([feature_groups[k] for k in keys], axis=2)


def channel_index(feature_groups: dict, include: list = None) -> list:
    """Return ordered list of (group, channel_idx) tuples for interpretability."""
    keys = include if include is not None else list(feature_groups.keys())
    index = []
    for k in keys:
        arr = feature_groups[k]
        for c in range(arr.shape[2]):
            index.append((k, c))
    return index


# ─── Convenience: build full tensor from paths (for notebooks) ───────────────

def build_from_files(crops: list = None,
                     groups: list = None) -> tuple:
    """Load panel + weather from config paths and build feature tensor.

    Returns (X, crops, panel_index, feature_groups) where
    X has shape (N, T, C_total).

    NOTE: this does NOT apply scalers — use for EDA only.
    For modelling, call build_feature_groups() inside the fold loop.
    """
    from src.utils.io import read_panel
    from src.features.weather_features import build_weather_panel

    panel = read_panel()
    if crops is None:
        crops = pd.read_csv(config.PANEL_DIR / "crop_list.csv")["crop"].tolist()
    panel = panel[crops]

    weather = build_weather_panel()

    fg = build_feature_groups(panel, weather, crops, groups=groups)
    X  = concat_groups(fg, include=groups)

    print(f"Feature tensor shape: {X.shape}  (nodes × time × channels)")
    for k, arr in fg.items():
        print(f"  {k}: {arr.shape}")

    return X, crops, panel.index, fg


if __name__ == "__main__":
    X, crops, idx, fg = build_from_files()
