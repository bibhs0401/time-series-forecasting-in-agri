"""Tests for the final held-out test block and holdout-aware CV folds.

The critical property: with holdout=k, no CV fold's training slice, forecast
origin or target may touch the last k observations. If that ever breaks, every
"final test" number in the paper is contaminated.
"""

import numpy as np
import pytest

from src.train.cv import (make_rolling_origin_folds, make_holdout_fold,
                          HORIZONS)

N_OBS = 523          # the real panel length
HOLDOUT = 52         # one seasonal cycle


class TestHoldoutIsolation:

    def test_no_cv_target_enters_holdout(self):
        folds = make_rolling_origin_folds(N_OBS, n_folds=5, holdout=HOLDOUT)
        cutoff = N_OBS - HOLDOUT
        for f in folds:
            assert f.train_end <= cutoff
            assert f.origins.max() < cutoff
            for h in f.horizons:
                tgt = f.target_positions(h)
                if tgt.size:
                    assert tgt.max() < cutoff, (
                        f"fold {f.fold_id} horizon {h} target {tgt.max()} "
                        f"enters the holdout (cutoff {cutoff})")

    def test_holdout_zero_matches_original_behaviour(self):
        a = make_rolling_origin_folds(N_OBS, n_folds=3, holdout=0)
        b = make_rolling_origin_folds(N_OBS, n_folds=3)
        assert [f.train_end for f in a] == [f.train_end for f in b]

    def test_holdout_shrinks_cv_region(self):
        a = make_rolling_origin_folds(N_OBS, n_folds=3, holdout=0)
        b = make_rolling_origin_folds(N_OBS, n_folds=3, holdout=HOLDOUT)
        assert max(f.test_end for f in b) <= N_OBS - HOLDOUT
        assert max(f.test_end for f in b) < max(f.test_end for f in a)

    def test_negative_holdout_rejected(self):
        with pytest.raises(ValueError):
            make_rolling_origin_folds(N_OBS, holdout=-1)


class TestHoldoutFold:

    def test_train_end_and_origins(self):
        f = make_holdout_fold(N_OBS, HOLDOUT)
        assert f.train_end == N_OBS - HOLDOUT
        assert f.origins.min() == N_OBS - HOLDOUT
        # every origin must leave a target for the longest horizon
        assert f.origins.max() + max(f.horizons) - 1 < N_OBS

    def test_all_targets_inside_panel(self):
        f = make_holdout_fold(N_OBS, HOLDOUT)
        for h in f.horizons:
            tgt = f.target_positions(h)
            assert tgt.size > 0
            assert tgt.max() < N_OBS

    def test_holdout_fold_starts_where_cv_stops(self):
        folds = make_rolling_origin_folds(N_OBS, n_folds=5, holdout=HOLDOUT)
        hf = make_holdout_fold(N_OBS, HOLDOUT)
        assert hf.train_end >= max(f.test_end for f in folds)

    def test_holdout_shorter_than_max_horizon_rejected(self):
        with pytest.raises(ValueError):
            make_holdout_fold(N_OBS, holdout=max(HORIZONS) - 1)

    def test_zero_holdout_rejected(self):
        with pytest.raises(ValueError):
            make_holdout_fold(N_OBS, holdout=0)


class TestFutureIndex:

    def test_future_index_continues_weekly(self):
        import pandas as pd
        from src.train.forecast_future import future_index
        idx = pd.date_range("2024-01-07", periods=100, freq="W-SUN")
        f = future_index(idx, 13)
        assert len(f) == 13
        assert f[0] == idx[-1] + pd.Timedelta(weeks=1)
        assert (np.diff(f.values).astype("timedelta64[D]")
                == np.timedelta64(7, "D")).all()
