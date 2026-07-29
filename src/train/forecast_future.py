'''Genuine out-of-sample forecasts: refit on the FULL panel and project forward
past the last observed week.

This is deliberately separate from experiment.py. Everything in experiment.py is
a backtest: every prediction is made at an origin inside the panel so it can
be scored against a known value. The forecasts produced here have no targets and
can never be scored; they are for the operational/illustrative figure, not for
the paper's evaluation tables.

Usage
    from src.train.forecast_future import forecast_future
    out = forecast_future(n_ahead=13, baselines=['SeasonalNaive', 'ETS'])
    out['forecast']   # tidy long DataFrame: date, crop, model, horizon, forecast

Leakage note
There is no leakage concern here (nothing is scored), but the scalers, graph and
model are still fit on the full panel only, never on anything later, because
nothing later exists.
'''

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.utils.io import read_panel
from src.train.cv import HORIZONS

DEFAULT_BASELINES = ['SeasonalNaive', 'Naive', 'ETS', 'ARIMA']


def future_index(index: pd.DatetimeIndex, n_ahead: int) -> pd.DatetimeIndex:
    '''The next n_ahead weekly timestamps after the end of index.'''
    freq = pd.infer_freq(index) or 'W-SUN'
    step = index[-1] - index[-2]
    return pd.DatetimeIndex([index[-1] + step * (k + 1) for k in range(n_ahead)],
                            name=index.name)


def forecast_baselines(panel_raw: np.ndarray, crops: list, index,
                       n_ahead: int, baselines: list,
                       verbose: bool = True) -> pd.DataFrame:
    '''Fit each baseline on the full panel and forecast weeks 1..n_ahead ahead.'''
    from src.models.baselines import make_baseline

    horizons = tuple(range(1, n_ahead + 1))
    fidx = future_index(index, n_ahead)
    rows = []

    for name in baselines:
        try:
            bl = make_baseline(name)
            bl.fit(panel_raw, crops, index=index)
            yhat = bl.forecast_from(panel_raw, horizons)        # (N, n_ahead)
            for i, crop in enumerate(crops):
                for j, h in enumerate(horizons):
                    rows.append(dict(date=fidx[j], crop=crop, model=bl.name,
                                     horizon=h, forecast=float(yhat[i, j])))
            if verbose:
                print(f'  [baseline] {name}: {len(crops)}x{n_ahead} forecasts')
        except Exception as e:
            warnings.warn(f'baseline {name} failed on future forecast: {e}')

    return pd.DataFrame(rows)


def forecast_torch(panel, weather, panel_raw, crops, index, n_ahead: int,
                   kinds: list, groups: list, window: int = 52,
                   epochs: int = 60, seed: int = 0, include_te: bool = False,
                   include_kg: bool = True,
                   quantiles: list | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    '''Train each ST-GNN on the full panel and forecast 1..n_ahead ahead.

    The models are DIRECT multi-horizon, so the forecast only needs the last
    window observed weeks as input; no future covariates are required.

    Note: not runtime-tested (torch unavailable in the dev sandbox). Shapes are
    reviewed against trainer.predict, which expects samples['X'] of (S,N,L,C).
    '''
    from src.models import build_model
    from src.train.trainer import train_model, predict, TrainConfig, pick_device
    from src.train.experiment import build_fold_tensor, _invert_targets, _build_fold_graph
    from src.train import dataset as ds

    T = panel_raw.shape[0]
    N = len(crops)
    horizons = tuple(range(1, n_ahead + 1))
    fidx = future_index(index, n_ahead)
    rows = []

    # Full-panel tensor + graph (train_end = T: everything is training data).
    X, y_model, crop_scalers = build_fold_tensor(panel, weather, crops, T, groups)
    graph = _build_fold_graph(panel, crops, T, include_te, verbose,
                              include_kg=include_kg)

    cfg = TrainConfig(epochs=epochs, seed=seed, device=pick_device(), verbose=False)

    # Training origins over the whole panel.
    tr_origins = ds.train_origins(T, window, horizons, stride=1)
    tr_o, va_o = ds.split_train_val(tr_origins, val_frac=0.2)
    train_samples = ds.make_samples(X, y_model, tr_o, horizons, window)
    val_samples = ds.make_samples(X, y_model, va_o, horizons, window)

    # The single inference window: the last `window` observed weeks.
    X_inf = X[:, T - window:T, :][None, ...].astype(np.float32)   # (1, N, L, C)

    for kind in kinds:
        try:
            model = build_model(kind, in_dim=X.shape[2], num_nodes=N,
                                horizons=list(horizons), graph=graph,
                                quantiles=quantiles)
            model, _ = train_model(model, train_samples, val_samples, cfg,
                                   quantiles=quantiles)
            out = predict(model, {'X': X_inf}, device=cfg.device)  # (1,N,H[,Q])
            if quantiles:                                          # point = median
                qi = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))
                out = out[..., qi]
            scaled = out[0].T                                      # (H, N) -> transpose
            raw = _invert_targets(out[0], crops, crop_scalers)      # (N, H) raw units
            for i, crop in enumerate(crops):
                for j, h in enumerate(horizons):
                    rows.append(dict(date=fidx[j], crop=crop, model=model.name,
                                     horizon=h, forecast=float(raw[i, j])))
            if verbose:
                print(f'  [gnn] {kind}: {N}x{n_ahead} forecasts')
        except Exception as e:
            warnings.warn(f'model {kind} failed on future forecast: {e}')

    return pd.DataFrame(rows)


def forecast_future(
    n_ahead:      int = 13,
    crops:        list | None = None,
    groups:       list | None = None,
    baselines:    list | None = None,
    torch_models: list | None = None,
    window:       int = 52,
    epochs:       int = 60,
    seed:         int = 0,
    include_te:   bool = False,
    include_kg:   bool = True,
    quantiles:    list | None = None,
    save:         bool = True,
    verbose:      bool = True,
) -> dict:
    '''Refit on the full panel and forecast n_ahead weeks into the future.

    Returns dict with 'forecast' (tidy long DataFrame) and 'wide' (date x crop,
    one block per model).
    '''
    t0 = time.time()
    from src.features.build_tensor import ALL_GROUPS
    from src.features.weather_features import build_weather_panel

    groups = groups or ALL_GROUPS
    baselines = DEFAULT_BASELINES if baselines is None else baselines

    panel = read_panel()
    if crops is None:
        crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    crops = [c for c in crops if c in panel.columns]
    panel = panel[crops]
    panel_raw = panel.to_numpy(dtype=float)
    index = panel.index

    if verbose:
        print(f'=== forecast_future | last observed = {index[-1].date()} | '
              f'n_ahead = {n_ahead} weeks ===')

    parts = []
    if baselines:
        parts.append(forecast_baselines(panel_raw, crops, index, n_ahead,
                                        baselines, verbose))

    if torch_models:
        from src.models import TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            weather = build_weather_panel()
            parts.append(forecast_torch(
                panel, weather, panel_raw, crops, index, n_ahead, torch_models,
                groups, window, epochs, seed, include_te, include_kg,
                quantiles, verbose))
        else:
            warnings.warn('PyTorch unavailable; skipping GNN future forecasts.')

    long = pd.concat([p for p in parts if len(p)], ignore_index=True) \
        if parts else pd.DataFrame()

    wide = pd.DataFrame()
    if len(long):
        wide = long.pivot_table(index='date', columns=['model', 'crop'],
                                values='forecast')

    if verbose:
        print(f'Done in {time.time() - t0:.1f}s. {len(long)} forecast rows.')

    if save and len(long):
        p_long = config.TABLES_DIR / 'future_forecast_long.csv'
        p_wide = config.TABLES_DIR / 'future_forecast_wide.csv'
        long.to_csv(p_long, index=False)
        wide.to_csv(p_wide)
        if verbose:
            print(f'Saved -> {p_long}\n       -> {p_wide}')

    return {'forecast': long, 'wide': wide}


if __name__ == '__main__':
    res = forecast_future(n_ahead=13, baselines=['SeasonalNaive', 'Naive'],
                          torch_models=None, verbose=True)
    if len(res['forecast']):
        print(res['forecast'].head(12).to_string(index=False))
