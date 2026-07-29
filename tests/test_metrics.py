"""tests/test_metrics.py — unit tests for src/train/metrics.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import metrics as M


class TestScales:
    def test_seasonal_naive_scale_constant_diff(self):
        # y_t = t  -> |y_t - y_{t-m}| = m everywhere
        y = np.arange(200, dtype=float)
        assert M.seasonal_naive_scale(y, m=52) == pytest.approx(52.0)

    def test_scale_positive_fallbacks(self):
        assert M.seasonal_naive_scale(np.array([5.0]), m=52) == 1.0  # last resort
        assert M.seasonal_naive_scale(np.array([1.0, 3.0, 5.0]), m=52) > 0


class TestPointMetrics:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0])
        assert M.mae(y, y) == 0.0
        assert M.rmse(y, y) == 0.0
        assert M.mase(y, y, y_train=np.arange(120.0)) == 0.0
        assert M.rmsse(y, y, y_train=np.arange(120.0)) == 0.0

    def test_mase_scaling(self):
        y_train = np.arange(120, dtype=float)          # scale = 52
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([100.0 + 52.0, 100.0 - 52.0])
        # abs error = 52 each -> MASE = 52/52 = 1
        assert M.mase(y_true, y_pred, y_train, m=52) == pytest.approx(1.0)

    def test_nan_pairwise_ignored(self):
        y = np.array([1.0, np.nan, 3.0])
        p = np.array([1.0, 5.0, 3.0])
        assert M.mae(y, p) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            M.mae(np.zeros(3), np.zeros(4))


class TestDirectional:
    def test_pearson_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert M.pearson(y, 2 * y + 1) == pytest.approx(1.0)

    def test_spearman_monotone(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert M.spearman(y, np.exp(y)) == pytest.approx(1.0)


class TestProbabilistic:
    def test_pinball_median_equals_half_mae(self):
        y = np.array([0.0, 10.0])
        pred_q = np.array([[5.0], [5.0]])              # single quantile 0.5
        # pinball at q=0.5 is 0.5 * |y - p|
        expected = 0.5 * np.mean(np.abs(y - pred_q[:, 0]))
        assert M.pinball_loss(y, pred_q, [0.5]) == pytest.approx(expected)

    def test_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 0.0, 10.0])
        hi = np.array([2.0, 2.0, 20.0])
        assert M.interval_coverage(y, lo, hi) == pytest.approx(2 / 3)


class TestDieboldMariano:
    def test_better_model_negative_stat(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        pred_a = y + rng.normal(scale=0.1, size=200)   # good
        pred_b = y + rng.normal(scale=1.0, size=200)   # bad
        dm, p = M.diebold_mariano(y, pred_a, pred_b, h=1)
        assert dm < 0            # A has lower loss
        assert p < 0.05

    def test_insufficient_data(self):
        dm, p = M.diebold_mariano([1.0], [1.0], [1.0])
        assert np.isnan(dm) and np.isnan(p)
