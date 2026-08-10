'''Rolling-origin experiment orchestrator.

Ties together, leakage-safely and per fold:
  panel (targets) + weather  ->  feature tensor (src/features/build_tensor.py)
  STL + multiplex graph      ->  src/graph/build_graph.py
  baselines + ST-GNNs        ->  src/models/*
  rolling-origin CV          ->  src/train/cv.py
  scale-free metrics         ->  src/train/metrics.py

Per-fold leakage discipline (design doc §5, §6, §12)
  * per-crop RobustScalers and the weather scaler are fit on the TRAIN slice
    only, then applied to the whole series so that test windows are scaled with
    train statistics;
  * the multiplex graph is rebuilt on the TRAIN slice only;
  * torch models are trained on origins whose window AND every horizon target
    lie inside the train slice; a chronological tail of those origins is held
    out for early stopping;
  * predictions are inverted to RAW space before metrics, and the MASE/RMSSE
    denominator is the seasonal-naive scale of the RAW train series.

Outputs
  A tidy long-format results DataFrame (one row per model x fold x crop x
  horizon) and an aggregated summary (mean +/- std across folds & crops).
  run_experiment writes both to outputs/tables/.
'''

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

from src.utils.io import read_panel
from src.features.build_tensor import (
    fit_crop_scalers, fit_weather_scaler, build_feature_groups, concat_groups,
    ALL_GROUPS,
)
from src.features.weather_features import build_weather_panel
from src.train.cv import (make_rolling_origin_folds, make_holdout_fold,
                          HORIZONS, describe_folds)
from src.train import dataset as ds
from src.train import metrics as M


def _scaled_target_matrix(panel: pd.DataFrame, crops: list, scalers: dict) -> np.ndarray:
    '''(T, N) matrix of the per-crop RobustScaler-transformed target values.'''
    cols = []
    for c in crops:
        v = panel[[c]].to_numpy(dtype=float)
        cols.append(scalers[c].transform(v)[:, 0] if c in scalers else v[:, 0])
    return np.stack(cols, axis=1)


def _invert_targets(scaled: np.ndarray, crops: list, scalers: dict) -> np.ndarray:
    '''Invert a (..., N) array of scaled values back to raw units, per crop.

    Convention: scaled has crops along axis 0 (N, ...). We invert per crop i
    along axis 0.
    '''
    out = np.array(scaled, dtype=float, copy=True)
    for i, c in enumerate(crops):
        sc = scalers.get(c)
        if sc is None:
            continue
        center = sc.center_[0]
        scale = sc.scale_[0]
        out[i] = scaled[i] * scale + center
    return out


def build_fold_tensor(panel, weather, crops, train_end, groups):
    '''Return (X, y_model, crop_scalers) for one fold.

    X       : (N, T, C) float32 feature tensor, NaNs -> 0.
    y_model : (N, T) scaled target matrix (model space).
    Scalers are fit on panel.iloc[:train_end] / weather rows in train only.
    '''
    panel_train = panel.iloc[:train_end][crops]
    crop_scalers = fit_crop_scalers(panel_train)

    wx_train = weather.reindex(panel.index).iloc[:train_end]
    weather_scaler = fit_weather_scaler(wx_train)

    fg = build_feature_groups(
        panel[crops], weather, crops,
        crop_scalers=crop_scalers, weather_scaler=weather_scaler, groups=groups,
    )
    X = concat_groups(fg, include=groups)                      # (N, T, C)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    y_model = _scaled_target_matrix(panel, crops, crop_scalers).T  # (N, T)
    return X, y_model, crop_scalers


def _records_from_predictions(pred_raw, panel_raw, crops, origins, horizons,
                              train_end, model_name, fold_id):
    '''Build long-format metric records from raw-space predictions.

    pred_raw  : dict h -> (N, n_origins_h) raw forecasts, aligned to
                origins truncated to those with a target < test_end.
    panel_raw : (T, N) raw target values (numpy).
    '''
    recs = []
    m = M.SEASONAL_PERIOD
    for hi, h in enumerate(horizons):
        tgt_pos = origins + h - 1
        valid = tgt_pos < panel_raw.shape[0]
        tp = tgt_pos[valid]
        if tp.size == 0:
            continue
        preds_h = pred_raw[h][:, valid]                        # (N, n_valid)
        for i, crop in enumerate(crops):
            y_true = panel_raw[tp, i]
            y_pred = preds_h[i]
            y_train = panel_raw[:train_end, i]
            row = M.evaluate_point(y_true, y_pred, y_train, m=m)
            row.update(dict(model=model_name, fold=fold_id, crop=crop,
                            horizon=h, n=int(tp.size)))
            recs.append(row)
    return recs


def eval_baseline_fold(model_ctor, panel_raw, crops, fold, index):
    from src.models.baselines import make_baseline
    train_end = fold.train_end
    horizons = fold.horizons

    bl = make_baseline(model_ctor)
    bl.fit(panel_raw[:train_end], crops, index=index[:train_end])

    # Forecast at every test origin.
    pred = {h: [] for h in horizons}
    for o in fold.origins:
        hist = panel_raw[:o]                                    # observed [0,o)
        yhat = bl.forecast_from(hist, horizons)                # (N, H)
        for hi, h in enumerate(horizons):
            pred[h].append(yhat[:, hi])
    pred = {h: np.stack(v, axis=1) for h, v in pred.items()}    # (N, n_origins)

    return _records_from_predictions(
        pred, panel_raw, crops, fold.origins, horizons, train_end,
        bl.name, fold.fold_id)


def eval_torch_fold(kind, panel, weather, panel_raw, crops, fold, groups,
                    graph, crop_scalers, X, y_model, cfg, quantiles,
                    window, model_kwargs):
    '''Train one ST-GNN on a fold and score it on the test origins.'''
    from src.models import build_model
    from src.train.trainer import train_model, predict, TrainConfig

    train_end = fold.train_end
    horizons = fold.horizons
    N = len(crops)

    # Training / validation samples (targets inside train slice).
    tr_origins = ds.train_origins(train_end, window, horizons, stride=1)
    tr_o, va_o = ds.split_train_val(tr_origins, val_frac=0.2)
    train_samples = ds.make_samples(X, y_model, tr_o, horizons, window)
    val_samples = ds.make_samples(X, y_model, va_o, horizons, window)

    in_dim = X.shape[2]
    model = build_model(kind, in_dim=in_dim, num_nodes=N, horizons=list(horizons),
                        graph=graph, quantiles=quantiles, **model_kwargs)

    model, _ = train_model(model, train_samples, val_samples, cfg, quantiles=quantiles)

    # Test samples at the fold's forecast origins
    test_samples = ds.make_samples(X, y_model, fold.origins, horizons, window)
    if test_samples['X'].shape[0] == 0:
        return []
    out = predict(model, test_samples, device=cfg.device)      # (S,N,H[,Q])

    kept_origins = test_samples['origins']
    if quantiles:                                              # point = median
        qi = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))
        point = out[..., qi]                                   # (S,N,H)
    else:
        point = out

    # Reshape to dict h -> (N, S_kept) in RAW space.
    pred = {}
    for hi, h in enumerate(horizons):
        scaled = point[:, :, hi].T                            # (N, S_kept)
        pred[h] = _invert_targets(scaled, crops, crop_scalers)

    recs = _records_from_predictions(
        pred, panel_raw, crops, kept_origins, horizons, train_end,
        model.name, fold.fold_id)

    # Probabilistic metrics for quantile models.
    if quantiles:
        recs = _augment_quantile_metrics(recs, out, panel_raw, crops,
                                         kept_origins, horizons, crop_scalers,
                                         quantiles, model.name, fold.fold_id)
    return recs


def _augment_quantile_metrics(recs, out, panel_raw, crops, origins, horizons,
                              crop_scalers, quantiles, model_name, fold_id):
    '''Add pinball loss + interval coverage to each quantile-model record.'''
    q = np.array(quantiles)
    lo_i, hi_i = int(np.argmin(q)), int(np.argmax(q))
    lookup = {(r['crop'], r['horizon']): r for r in recs}
    for hi, h in enumerate(horizons):
        tgt_pos = origins + h - 1
        valid = tgt_pos < panel_raw.shape[0]
        tp = tgt_pos[valid]
        if tp.size == 0:
            continue
        for i, crop in enumerate(crops):
            # Invert each quantile to raw space.
            qs_scaled = out[valid, i, hi, :]                  # (n_valid, Q)
            sc = crop_scalers.get(crop)
            if sc is not None:
                qs = qs_scaled * sc.scale_[0] + sc.center_[0]
            else:
                qs = qs_scaled
            y_true = panel_raw[tp, i]
            rec = lookup.get((crop, h))
            if rec is None:
                continue
            rec['pinball'] = M.pinball_loss(y_true, qs, quantiles)
            rec['coverage'] = M.interval_coverage(y_true, qs[:, lo_i], qs[:, hi_i])
    return recs


DEFAULT_BASELINES = ['SeasonalNaive', 'Naive', 'ETS', 'ARIMA']
DEFAULT_TORCH = ['nograph', 'identity', 'static', 'gat', 'relational', 'rgat', 'adaptive']


# Holdout guard: the final-test block must be scored EXACTLY ONCE, after all
# model/hyperparameter selection is finished on the CV folds (see cv.py's
# make_holdout_fold docstring, and design doc §12). Re-scoring it after
# peeking is model-selection leakage, not temporal leakage, so it can't be
# caught by index/window checks. This marker file makes the "once" rule
# something the code enforces instead of something the runner has to remember.


def _holdout_marker_path() -> Path:
    return config.OUTPUTS_DIR / '.holdout_consumed.json'


def _assert_holdout_available(force: bool) -> None:
    '''Raise if the final-test holdout was already scored and force is False.'''
    path = _holdout_marker_path()
    if not path.exists() or force:
        return
    try:
        prior = json.loads(path.read_text())
    except Exception:
        prior = {}
    events = prior.get('events') or []
    if not events:
        return
    last = events[-1]
    raise RuntimeError(
        'The final-test holdout has already been scored, on '
        f"{last.get('timestamp', 'an earlier run')} "
        f"(seed={last.get('seed')}, torch_models={last.get('torch_models')}, "
        f"baselines={last.get('baselines')}). Scoring it again risks "
        'model-selection leakage: it is meant to be touched exactly once, '
        'after every architecture/hyperparameter decision has already been '
        'made using the CV folds. If you have a deliberate, justified reason '
        'to re-run it (e.g. reproducing a result, fixing an unrelated bug; '
        'not re-selecting models based on what you saw), pass '
        'force_final_test=True (--force-final-test on the CLI). Every '
        f'invocation is logged to {path}.'
    )


def _record_holdout_consumed(meta: dict) -> None:
    '''Append an audit record marking the holdout as consumed.'''
    path = _holdout_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = {'events': []}
    if path.exists():
        try:
            prior = json.loads(path.read_text())
        except Exception:
            prior = {'events': []}
    record = {**meta, 'timestamp': datetime.now(timezone.utc).isoformat()}
    prior.setdefault('events', []).append(record)
    path.write_text(json.dumps(prior, indent=2))


def run_experiment(
    crops:        list | None = None,
    groups:       list | None = None,
    horizons:     tuple = HORIZONS,
    n_folds:      int = 5,
    window:       int = 52,
    baselines:    list | None = None,
    torch_models: list | None = None,
    quantiles:    list | None = None,
    include_te:   bool = False,
    include_kg:   bool = True,
    epochs:       int = 60,
    seed:         int = 0,
    holdout:      int = 0,
    final_test:   bool = False,
    force_final_test: bool = False,
    save:         bool = True,
    verbose:      bool = True,
) -> dict:
    '''Run the full rolling-origin experiment and return result tables.

    Returns dict with keys 'long' (per crop/horizon/fold) and 'summary'.
    Torch models are silently skipped (with a warning) if torch is unavailable.
    '''
    t0 = time.time()
    groups = groups or ALL_GROUPS
    baselines = DEFAULT_BASELINES if baselines is None else baselines
    torch_models = DEFAULT_TORCH if torch_models is None else torch_models

    panel = read_panel()
    if crops is None:
        crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    crops = [c for c in crops if c in panel.columns]
    panel = panel[crops]
    panel_raw = panel.to_numpy(dtype=float)                    # (T, N)
    index = panel.index

    weather = build_weather_panel()

    if final_test:
        if not holdout:
            raise ValueError('final_test=True requires holdout > 0.')
        _assert_holdout_available(force_final_test)
        folds = [make_holdout_fold(len(panel), holdout, horizons)]
        if verbose:
            print(f'*** FINAL HELD-OUT TEST ({holdout} weeks); this block should '
                  f'be scored ONCE, after all selection is done. ***')
    else:
        folds = make_rolling_origin_folds(len(panel), n_folds=n_folds,
                                          horizons=horizons, holdout=holdout)
    if verbose:
        print(describe_folds(folds, index))
        _warn_empty_eval_region(panel_raw, crops, folds, index)

    # Torch availability.
    from src.models import TORCH_AVAILABLE
    torch_kinds = []
    if torch_models:
        if TORCH_AVAILABLE:
            torch_kinds = torch_models
        else:
            warnings.warn('PyTorch unavailable; skipping GNN models; '
                          'running baselines only.')

    cfg = None
    if torch_kinds:
        from src.train.trainer import TrainConfig, pick_device
        cfg = TrainConfig(epochs=epochs, seed=seed, device=pick_device(),
                          verbose=False)

    all_recs = []
    for fold in folds:
        if verbose:
            print(f'\n=== Fold {fold.fold_id} | n_train={fold.n_train} '
                  f'| origins={len(fold.origins)} ===')

        # Baselines.
        for name in baselines:
            try:
                recs = eval_baseline_fold(name, panel_raw, crops, fold, index)
                all_recs.extend(recs)
                if verbose:
                    print(f'  [baseline] {name}: {len(recs)} records')
            except Exception as e:
                warnings.warn(f'baseline {name} failed on fold {fold.fold_id}: {e}')

        # Torch models (share the fold tensor + graph).
        if torch_kinds:
            X, y_model, crop_scalers = build_fold_tensor(
                panel, weather, crops, fold.train_end, groups)
            graph = _build_fold_graph(panel, crops, fold.train_end, include_te,
                                      verbose, include_kg=include_kg)
            for kind in torch_kinds:
                try:
                    recs = eval_torch_fold(
                        kind, panel, weather, panel_raw, crops, fold, groups,
                        graph, crop_scalers, X, y_model, cfg, quantiles,
                        window, model_kwargs={})
                    all_recs.extend(recs)
                    if verbose:
                        print(f'  [gnn] {kind}: {len(recs)} records')
                except Exception as e:
                    warnings.warn(f'model {kind} failed on fold {fold.fold_id}: {e}')

    long = pd.DataFrame(all_recs)
    summary = summarize(long)

    if verbose:
        print(f'\nDone in {time.time() - t0:.1f}s. '
              f"{len(long)} records, {long['model'].nunique() if len(long) else 0} models.")

    if final_test and len(long):
        _record_holdout_consumed(dict(
            seed=seed, torch_models=torch_kinds, baselines=baselines,
            groups=list(groups), holdout=holdout, epochs=epochs,
            forced=force_final_test, n_records=len(long),
        ))
        if verbose:
            print(f'Holdout marked consumed -> {_holdout_marker_path()}')

    if save and len(long):
        long_path = config.TABLES_DIR / 'experiment_results_long.csv'
        sum_path = config.TABLES_DIR / 'experiment_results_summary.csv'
        long.to_csv(long_path, index=False)
        summary.to_csv(sum_path, index=False)
        if verbose:
            print(f'Saved -> {long_path}\n       -> {sum_path}')

    return {'long': long, 'summary': summary, 'folds': folds}


def _warn_empty_eval_region(panel_raw, crops, folds, index=None):
    '''Warn about crops with no finite observations in the evaluation region.

    A crop whose series stops before the test block contributes no scoreable
    targets there; its metrics would be empty (or, worse, silently driven by a
    stale last-finite value carried forward by the naive baselines). Surfacing
    this is important: it is a data/stitching problem, not a modelling one.
    '''
    lo = min(f.test_start for f in folds)
    hi = max(f.test_end for f in folds)
    bad = []
    for i, c in enumerate(crops):
        seg = panel_raw[lo:hi, i]
        n_ok = int(np.isfinite(seg).sum())
        if n_ok == 0:
            bad.append((c, 0))
        elif n_ok < 0.5 * len(seg):
            bad.append((c, n_ok))
    if bad:
        span = ''
        if index is not None:
            span = f' ({index[lo].date()}..{index[min(hi, len(index)) - 1].date()})'
        msg = ', '.join(f'{c} [{n}/{hi - lo} finite]' for c, n in bad)
        warnings.warn(
            f'{len(bad)} crop(s) have little or no data in the evaluation '
            f'region{span}: {msg}. Their scores will be empty or unreliable; '
            f'fix the stitch or drop them from crop_list.csv.')
    return [c for c, _ in bad]


def _build_fold_graph(panel, crops, train_end, include_te, verbose,
                      include_kg=True):
    from src.graph.build_graph import build_multiplex_graph
    train = panel.iloc[:train_end][crops]
    return build_multiplex_graph(
        panel_train=train, crops=crops, include_te=include_te,
        include_kg=include_kg, verbose=False)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    '''Aggregate the long table to mean +/- std per model x horizon.'''
    if long is None or len(long) == 0:
        return pd.DataFrame()
    metric_cols = [c for c in ['MASE', 'RMSSE', 'MAE', 'RMSE', 'sMAPE',
                               'Pearson', 'Spearman', 'pinball', 'coverage']
                   if c in long.columns]
    agg = (long.groupby(['model', 'horizon'])[metric_cols]
                .agg(['mean', 'std']))
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    return agg.reset_index().sort_values(['horizon', 'MASE_mean'])


if __name__ == '__main__':
    # Fast smoke configuration (baselines only unless torch present).
    res = run_experiment(n_folds=2, epochs=5, verbose=True)
    if len(res['summary']):
        print('\n=== SUMMARY (head) ===')
        print(res['summary'].head(20).to_string(index=False))
