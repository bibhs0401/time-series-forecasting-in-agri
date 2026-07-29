'''Rolling-origin cross-validation for the weekly Trends panel.

Design doc §8.3: rolling-origin CV, multiple origins; horizons h in {1,4,8,13};
report mean +/- std across folds and horizons.

A fold is defined purely by integer positions into the (time-ordered) panel:

    Fold(train_end, ...)
      train slice  = panel.iloc[: train_end]
      test origins = the next test_size weeks
      for each origin o and horizon h, the target is week (o + h - 1).

Everything fit inside a fold (scalers, STL, graph, model) must use only
panel.iloc[:train_end]. This module produces the index bookkeeping; it does
not touch data values, so it cannot leak.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HORIZONS = (1, 4, 8, 13)   # weeks ahead, design doc §8.3


@dataclass
class Fold:
    '''One rolling-origin fold, expressed as integer positions into the panel.

    Attributes
    fold_id     : 0-based fold index.
    train_end   : exclusive end position of the training slice (== n_train).
    test_start  : first test-origin position (== train_end).
    test_end    : exclusive end of the region containing test targets.
    horizons    : tuple of forecast horizons evaluated in this fold.
    origins     : forecast-origin positions (last observed week is origin-1).
    '''
    fold_id:    int
    train_end:  int
    test_start: int
    test_end:   int
    horizons:   tuple = HORIZONS
    origins:    np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def n_train(self) -> int:
        return self.train_end

    def target_positions(self, h: int) -> np.ndarray:
        '''Panel positions of the targets for horizon h in this fold.

        Origin o observes weeks [0, o); the horizon-h target is week o+h-1.
        Only targets that fall strictly before test_end are returned.
        '''
        tgt = self.origins + h - 1
        return tgt[tgt < self.test_end]


def make_rolling_origin_folds(
    n_obs:       int,
    n_folds:     int = 5,
    horizons:    tuple = HORIZONS,
    test_size:   int | None = None,
    min_train:   int | None = None,
    expanding:   bool = True,
    holdout:     int = 0,
) -> list[Fold]:
    '''Generate rolling-origin folds over a series of length n_obs.

    Parameters
    n_obs     : total number of weekly observations in the panel.
    n_folds   : number of rolling origins blocks.
    horizons  : forecast horizons; the last fold must leave >= max(horizons)
                weeks of test region so every horizon has a target.
    test_size : number of consecutive forecast origins per fold. Default:
                sized so the folds tile the tail of the series after
                min_train with room for max(horizons).
    min_train : minimum training length before the first fold. Default:
                the larger of 2*52 (two seasonal cycles) and n_obs//2.
    expanding : True -> expanding window (train always starts at 0);
                False -> sliding window of fixed length min_train.
    holdout   : number of observations at the END of the series to reserve as a
                final untouched test block. CV folds are generated over
                n_obs - holdout only, so no fold's window, training slice or
                target ever enters the holdout region. Use with
                make_holdout_fold for the final evaluation.

    Returns
    list[Fold], chronologically ordered.
    '''
    hmax = max(horizons)

    if holdout < 0:
        raise ValueError(f'holdout must be >= 0, got {holdout}')
    if holdout:
        n_obs = n_obs - holdout      # CV never sees the reserved tail
    if min_train is None:
        min_train = max(2 * 52, n_obs // 2)
    if min_train + hmax >= n_obs:
        raise ValueError(
            f'min_train ({min_train}) + max horizon ({hmax}) >= n_obs ({n_obs}); '
            f'series too short for rolling-origin CV.'
        )

    # Total test region available after the initial training block.
    total_test = n_obs - min_train - hmax + 1     # #origins that have all horizons
    if total_test < n_folds:
        n_folds = max(1, total_test)
    if test_size is None:
        test_size = max(1, total_test // n_folds)

    folds: list[Fold] = []
    for k in range(n_folds):
        train_end = min_train + k * test_size
        # Origins that still leave hmax weeks of targets in-sample.
        last_origin = min(train_end + test_size, n_obs - hmax + 1)
        if last_origin <= train_end:
            break
        origins = np.arange(train_end, last_origin, dtype=int)
        test_end = int(origins[-1] + hmax)         # exclusive
        train_start = 0 if expanding else max(0, train_end - min_train)
        folds.append(Fold(
            fold_id=k,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            horizons=tuple(horizons),
            origins=origins,
        ))
        # Track sliding-window start via metadata on the object.
        folds[-1].train_start = train_start  # type: ignore[attr-defined]

    return folds


def make_holdout_fold(
    n_obs:    int,
    holdout:  int,
    horizons: tuple = HORIZONS,
    fold_id:  int = -1,
) -> Fold:
    '''The final held-out test block, as a single Fold.

    Train slice is everything before the holdout; forecast origins tile the
    holdout region. This block must be touched EXACTLY ONCE, after all
    model/hyperparameter selection is finished on the CV folds produced by
    make_rolling_origin_folds(..., holdout=holdout).

    Parameters
    n_obs    : total observations in the panel.
    holdout  : size of the reserved tail (e.g. 52 weeks = one seasonal cycle).
    horizons : forecast horizons.
    fold_id  : id for the returned Fold (default -1 marks it as not a CV fold).

    Returns
    Fold with train_end = n_obs - holdout and origins inside the holdout.
    '''
    hmax = max(horizons)
    if holdout <= 0:
        raise ValueError(f'holdout must be > 0, got {holdout}')
    if holdout < hmax:
        raise ValueError(
            f'holdout ({holdout}) < max horizon ({hmax}); no origin would have '
            f'a scoreable target for every horizon.')
    train_end = n_obs - holdout
    if train_end <= 0:
        raise ValueError(f'holdout ({holdout}) >= n_obs ({n_obs}).')

    # Origins that leave every horizon's target inside the panel.
    origins = np.arange(train_end, n_obs - hmax + 1, dtype=int)
    fold = Fold(
        fold_id=fold_id,
        train_end=train_end,
        test_start=train_end,
        test_end=n_obs,
        horizons=tuple(horizons),
        origins=origins,
    )
    fold.train_start = 0  # type: ignore[attr-defined]
    return fold


def describe_folds(folds: list[Fold], index=None) -> str:
    '''Human-readable summary of a fold list (optionally with dates).'''
    lines = [f'{len(folds)} rolling-origin folds:']
    for f in folds:
        span = ''
        if index is not None:
            span = (f'  train[..{index[f.train_end - 1].date()}] '
                    f'test origins {index[f.origins[0]].date()}'
                    f'..{index[f.origins[-1]].date()}')
        lines.append(
            f'  fold {f.fold_id}: n_train={f.n_train}, '
            f'n_origins={len(f.origins)}, horizons={f.horizons}{span}')
    return '\n'.join(lines)
