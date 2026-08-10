'''Classical / statistical forecasting baselines (design doc §8.1).

Ladder implemented here
  SeasonalNaive   the brutal bar: y_hat(o+h) = y(o+h-m), m = 52
  Naive           last observed value carried forward
  ETS             statsmodels ETSModel (additive, optional annual season)
  ARIMA           statsmodels SARIMAX (order/seasonal_order configurable)

Common interface (used by src/train/experiment.py)

    m = SeasonalNaive()
    m.fit(train_panel, crops, index=train_index)      # train slice only
    yhat = m.forecast_from(history, horizons)          # history = obs up to o
        -> np.ndarray (N, H)   raw-space point forecasts

history is the observed panel up to the forecast origin o (weeks [0, o),
shape (o, N)); the horizon-h target is week o+h-1. Nothing is fit on data
at/after the origin, so baselines are leakage-free by construction.

The statsmodels-backed baselines (ETS, ARIMA) estimate parameters once on
the training slice and then filter (not re-estimate) the extended history
at each origin via results.apply(..., refit=False). That is the standard,
cheap, leakage-safe rolling-origin approach. Any fit failure falls back to
seasonal-naive so the harness never crashes on a pathological series.
'''

from __future__ import annotations

import warnings

import numpy as np

SEASON = 52


class Baseline:
    name = 'baseline'
    is_multivariate = False

    def fit(self, train_panel: np.ndarray, crops: list, index=None):
        '''Fit on the training slice (T_train, N). Returns self.'''
        self.crops = list(crops)
        self.N = train_panel.shape[1]
        self.train_panel = np.asarray(train_panel, dtype=float)
        self.index = index
        return self

    def forecast_from(self, history: np.ndarray, horizons) -> np.ndarray:
        '''Return (N, H) point forecasts given observed history (o, N).'''
        raise NotImplementedError

    @staticmethod
    def _seasonal_naive_series(hist: np.ndarray, horizons, m: int = SEASON) -> np.ndarray:
        '''Seasonal-naive forecast for a single series hist (1-D, length o).'''
        o = len(hist)
        out = np.empty(len(horizons))
        for j, h in enumerate(horizons):
            src = o + h - 1 - m
            if 0 <= src < o and np.isfinite(hist[src]):
                out[j] = hist[src]
            elif o > 0:                       # fall back to last observed
                out[j] = _last_finite(hist)
            else:
                out[j] = np.nan
        return out


def _last_finite(x: np.ndarray, default: float = 0.0) -> float:
    x = x[np.isfinite(x)]
    return float(x[-1]) if x.size else default


class SeasonalNaive(Baseline):
    name = 'SeasonalNaive'

    def __init__(self, m: int = SEASON):
        self.m = m

    def forecast_from(self, history, horizons):
        hist = np.asarray(history, dtype=float)
        return np.stack([
            self._seasonal_naive_series(hist[:, i], horizons, self.m)
            for i in range(hist.shape[1])
        ], axis=0)


class Naive(Baseline):
    name = 'Naive'

    def forecast_from(self, history, horizons):
        hist = np.asarray(history, dtype=float)
        last = np.array([_last_finite(hist[:, i]) for i in range(hist.shape[1])])
        return np.repeat(last[:, None], len(horizons), axis=1)


class _StatsmodelsUnivariate(Baseline):
    '''Shared fit-once / filter-per-origin machinery for ETS and ARIMA.'''

    def __init__(self, seasonal: bool = True, m: int = SEASON, min_obs: int = 3 * SEASON):
        self.seasonal = seasonal
        self.m = m
        self.min_obs = min_obs

    def _make_result(self, series: np.ndarray):
        '''Estimate a model on series; return a fitted statsmodels result.'''
        raise NotImplementedError

    def fit(self, train_panel, crops, index=None):
        super().fit(train_panel, crops, index)
        self.results = []
        for i in range(self.N):
            s = self.train_panel[:, i]
            s = np.where(np.isfinite(s), s, np.nan)
            res = None
            if np.isfinite(s).sum() >= self.min_obs and np.nanstd(s) > 0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res = self._make_result(np.nan_to_num(s, nan=np.nanmean(s)))
                except Exception:
                    res = None
            self.results.append(res)
        return self

    def forecast_from(self, history, horizons):
        hist = np.asarray(history, dtype=float)
        hmax = max(horizons)
        idx = [h - 1 for h in horizons]
        out = np.empty((self.N, len(horizons)))
        for i in range(self.N):
            s = hist[:, i]
            res = self.results[i] if hasattr(self, 'results') else None
            fc = None
            if res is not None:
                try:
                    s_filled = np.nan_to_num(s, nan=_last_finite(s))
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        applied = res.apply(s_filled, refit=False)
                        fc = np.asarray(applied.forecast(hmax), dtype=float)
                except Exception:
                    fc = None
            if fc is None or not np.all(np.isfinite(fc[idx])):
                out[i, :] = self._seasonal_naive_series(s, horizons, self.m)
            else:
                out[i, :] = fc[idx]
        return out


class ETS(_StatsmodelsUnivariate):
    name = 'ETS'

    def _make_result(self, series):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        seasonal = 'add' if (self.seasonal and len(series) >= 2 * self.m) else None
        model = ETSModel(
            series, error='add', trend='add', damped_trend=True,
            seasonal=seasonal, seasonal_periods=self.m if seasonal else None,
        )
        return model.fit(disp=False)


class ARIMA(_StatsmodelsUnivariate):
    name = 'ARIMA'

    def __init__(self, order=(2, 0, 1), seasonal_order=(1, 0, 0, SEASON),
                 seasonal: bool = True, m: int = SEASON, min_obs: int = 3 * SEASON):
        super().__init__(seasonal=seasonal, m=m, min_obs=min_obs)
        self.order = order
        self.seasonal_order = seasonal_order if seasonal else (0, 0, 0, 0)

    def _make_result(self, series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            series, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.fit(disp=False)


BASELINES = {
    'SeasonalNaive': SeasonalNaive,
    'Naive':         Naive,
    'ETS':           ETS,
    'ARIMA':         ARIMA,
}


def make_baseline(name: str, **kwargs) -> Baseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'; choices: {list(BASELINES)}")
    return BASELINES[name](**kwargs)
