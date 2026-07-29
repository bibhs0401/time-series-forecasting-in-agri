"""tests/test_adjacency_static.py
Unit tests for src/graph/adjacency_static.py (A_agro, A_sem, build_static_layers).

Run with:
    python -m pytest tests/test_adjacency_static.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.adjacency_static import (
    build_agro_adjacency,
    agro_edge_table,
    build_sem_adjacency,
    build_static_layers,
    _build_static_layers_cached,
)

# Subset of crops from src/data/static_attributes.csv with known
# family / season_class relationships, used to make assertions concrete.
#   Strawberry, Blueberry : family=Berry,       season=cool  -> family+season match
#   Onion,      Garlic    : family=Allium,      season=cool  -> family+season match
#   Tomato                : family=Nightshade,  season=warm  -> shares nothing with Onion
#   Potato                : family=Nightshade,  season=cool  -> shares season with Onion only
CROPS = ["Strawberry", "Blueberry", "Tomato", "Potato", "Onion", "Garlic"]


class TestBuildAgroAdjacency:
    def test_shape_and_dtype(self):
        A = build_agro_adjacency(CROPS)
        assert A.shape == (len(CROPS), len(CROPS))
        assert A.dtype == np.float32

    def test_symmetric_and_no_self_loops(self):
        A = build_agro_adjacency(CROPS)
        assert np.allclose(A, A.T)
        assert np.all(np.diag(A) == 0.0)

    def test_family_and_season_match_uses_max_weight(self):
        A = build_agro_adjacency(CROPS)
        i, j = CROPS.index("Strawberry"), CROPS.index("Blueberry")
        assert A[i, j] == pytest.approx(1.0)
        i, j = CROPS.index("Onion"), CROPS.index("Garlic")
        assert A[i, j] == pytest.approx(1.0)

    def test_season_only_match(self):
        # Potato (Nightshade, cool) vs Onion (Allium, cool): season only.
        A = build_agro_adjacency(CROPS)
        i, j = CROPS.index("Potato"), CROPS.index("Onion")
        assert A[i, j] == pytest.approx(0.5)

    def test_no_match_has_no_edge(self):
        # Tomato (Nightshade, warm) vs Onion (Allium, cool): no shared attribute.
        A = build_agro_adjacency(CROPS)
        i, j = CROPS.index("Tomato"), CROPS.index("Onion")
        assert A[i, j] == 0.0

    def test_custom_weights_are_respected(self):
        A = build_agro_adjacency(CROPS, family_weight=2.0, season_weight=0.1)
        i, j = CROPS.index("Strawberry"), CROPS.index("Blueberry")
        assert A[i, j] == pytest.approx(2.0)
        i, j = CROPS.index("Potato"), CROPS.index("Onion")
        assert A[i, j] == pytest.approx(0.1)


class TestAgroEdgeTable:
    def test_matches_adjacency_weights(self):
        A = build_agro_adjacency(CROPS)
        tbl = agro_edge_table(CROPS)
        assert len(tbl) == int((A > 0).sum() / 2)
        for _, row in tbl.iterrows():
            i, j = CROPS.index(row["crop_i"]), CROPS.index(row["crop_j"])
            assert A[i, j] == pytest.approx(row["weight"])

    def test_custom_weights_are_forwarded(self):
        tbl = agro_edge_table(CROPS, family_weight=3.0, season_weight=3.0)
        assert np.allclose(tbl["weight"].to_numpy(), 3.0)


class TestBuildSemAdjacency:
    @pytest.fixture(autouse=True)
    def _require_sentence_transformers(self):
        pytest.importorskip("sentence_transformers")

    def test_shape_and_dtype(self):
        A = build_sem_adjacency(CROPS)
        assert A.shape == (len(CROPS), len(CROPS))
        assert A.dtype == np.float32

    def test_symmetric_and_no_self_loops(self):
        A = build_sem_adjacency(CROPS)
        assert np.allclose(A, A.T, atol=1e-5)
        assert np.all(np.diag(A) == 0.0)

    def test_values_within_cosine_range(self):
        A = build_sem_adjacency(CROPS, threshold=-1.0)
        assert A.min() >= -1.0 - 1e-5
        assert A.max() <= 1.0 + 1e-5

    def test_higher_threshold_keeps_fewer_or_equal_edges(self):
        A_low = build_sem_adjacency(CROPS, threshold=-1.0)
        A_high = build_sem_adjacency(CROPS, threshold=0.99)
        assert (np.abs(A_high) > 0).sum() <= (np.abs(A_low) > 0).sum()

    def test_empty_crop_list_returns_empty_matrix(self):
        A = build_sem_adjacency([])
        assert A.shape == (0, 0)

    def test_missing_dependency_falls_back_to_zeros(self, monkeypatch):
        import src.graph.adjacency_static as mod

        def _raise_import_error(*args, **kwargs):
            raise ImportError("simulated missing package")

        monkeypatch.setattr(mod, "_get_sentence_model", _raise_import_error)
        with pytest.warns(UserWarning):
            A = mod.build_sem_adjacency(CROPS)
        assert A.shape == (len(CROPS), len(CROPS))
        assert np.all(A == 0.0)


class TestBuildStaticLayers:
    def setup_method(self):
        _build_static_layers_cached.cache_clear()

    def test_returns_both_layers_with_correct_shape(self):
        layers = build_static_layers(CROPS, verbose=False)
        assert set(layers.keys()) == {"A_agro", "A_sem"}
        for A in layers.values():
            assert A.shape == (len(CROPS), len(CROPS))

    def test_family_weight_is_forwarded(self):
        layers = build_static_layers(CROPS, family_weight=5.0, verbose=False)
        i, j = CROPS.index("Strawberry"), CROPS.index("Blueberry")
        assert layers["A_agro"][i, j] == pytest.approx(5.0)

    def test_cached_calls_are_value_equal_but_independent(self):
        layers_1 = build_static_layers(CROPS, verbose=False)
        layers_2 = build_static_layers(CROPS, verbose=False)
        assert np.array_equal(layers_1["A_agro"], layers_2["A_agro"])

        layers_1["A_agro"][0, 0] = 999.0
        assert layers_2["A_agro"][0, 0] != 999.0

    def test_use_cache_false_bypasses_cache(self):
        layers = build_static_layers(CROPS, use_cache=False, verbose=False)
        assert layers["A_agro"].shape == (len(CROPS), len(CROPS))
