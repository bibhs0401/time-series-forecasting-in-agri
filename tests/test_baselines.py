"""tests/test_baselines.py — tests for src/models/baselines.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import (
    SeasonalNaive, Naive, make_baseline,
)


def _seasonal_panel(N=3, T=300, m=52):
    t = np.arange(T)
    base = np.stack([10 + 5 * np.sin(2 * np.pi * t / m + i) for i in range(N)], axis=1)
    return base.astype(float)


class TestSeasonalNaive:
    def test_recovers_prior_season(self):
        panel = _seasonal_panel()
        m = SeasonalNaive(m=52).fit(panel[:200], ["a", "b", "c"])
        o = 200
        yhat = m.forecast_from(panel[:o], horizons=(1, 4))
        assert yhat.shape == (3, 2)
        # horizon-1 target is week o; seasonal naive uses week o-52
        assert np.allclose(yhat[:, 0], panel[o - 52])

    def test_fallback_when_no_season(self):
        panel = _seasonal_panel(T=30)               # shorter than m
        m = SeasonalNaive(m=52).fit(panel, ["a", "b", "c"])
        yhat = m.forecast_from(panel, horizons=(1,))
        assert np.all(np.isfinite(yhat))            # falls back to last value


class TestSimpleNaives:
    def test_naive_is_last_value(self):
        panel = _seasonal_panel()
        m = Naive().fit(panel[:100], ["a", "b", "c"])
        yhat = m.forecast_from(panel[:100], horizons=(1, 4, 8))
        assert np.allclose(yhat[:, 0], panel[99])
        assert np.allclose(yhat[:, 0], yhat[:, 2])   # flat forecast


class TestRegistry:
    def test_registry(self):
        assert make_baseline("SeasonalNaive").name == "SeasonalNaive"
        with pytest.raises(KeyError):
            make_baseline("does-not-exist")
