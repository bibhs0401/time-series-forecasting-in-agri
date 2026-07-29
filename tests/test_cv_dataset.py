"""tests/test_cv_dataset.py — tests for rolling-origin CV and windowing."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.cv import make_rolling_origin_folds, Fold
from src.train import dataset as ds


class TestFolds:
    def test_no_leakage_train_before_test(self):
        folds = make_rolling_origin_folds(523, n_folds=5, horizons=(1, 4, 8, 13))
        assert len(folds) >= 1
        for f in folds:
            assert f.test_start == f.train_end
            # every origin observes only weeks < origin
            assert f.origins.min() >= f.train_end
            # targets stay within the series
            assert f.target_positions(13).max() < 523

    def test_expanding_monotone_train_end(self):
        folds = make_rolling_origin_folds(523, n_folds=5)
        ends = [f.train_end for f in folds]
        assert ends == sorted(ends)
        assert len(set(ends)) == len(ends)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            make_rolling_origin_folds(60, n_folds=5, horizons=(13,), min_train=55)

    def test_target_positions_formula(self):
        f = Fold(0, train_end=100, test_start=100, test_end=113,
                 horizons=(1, 4), origins=np.array([100, 101]))
        assert list(f.target_positions(1)) == [100, 101]
        assert list(f.target_positions(4)) == [103, 104]


class TestDataset:
    def _tensor(self, N=3, T=200, C=4):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(N, T, C)).astype(np.float32)
        y = rng.normal(size=(N, T)).astype(np.float32)
        return X, y

    def test_sample_shapes(self):
        X, y = self._tensor()
        origins = np.array([60, 70, 80])
        s = ds.make_samples(X, y, origins, horizons=(1, 4, 8), window=52)
        S = s["X"].shape[0]
        assert s["X"].shape == (S, 3, 52, 4)
        assert s["y"].shape == (S, 3, 3)

    def test_window_target_alignment(self):
        # y[i, o+h-1] must equal the stored target
        X, y = self._tensor()
        origins = np.array([100])
        horizons = (1, 5)
        s = ds.make_samples(X, y, origins, horizons=horizons, window=10)
        for j, h in enumerate(horizons):
            assert np.allclose(s["y"][0, :, j], y[:, 100 + h - 1])
        # window is weeks [90,100)
        assert np.allclose(s["X"][0, :, :, :], np.transpose(X[:, 90:100, :], (0, 1, 2)))

    def test_train_origins_stay_in_sample(self):
        origins = ds.train_origins(train_end=200, window=52, horizons=(1, 13))
        assert origins.min() >= 52
        assert (origins + 13 - 1).max() < 200

    def test_split_train_val_chronological(self):
        origins = np.arange(52, 100)
        tr, va = ds.split_train_val(origins, val_frac=0.25)
        assert tr.max() < va.min()          # val is the most-recent block
