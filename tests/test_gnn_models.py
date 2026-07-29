"""tests/test_gnn_models.py — ST-GNN forward/shape tests.

These require PyTorch; the whole module is skipped where torch is unavailable
(``pytest.importorskip``), so the suite stays green on machines without torch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.graph.build_graph import MultiplexGraph
from src.models.gnn import build_model, TORCH_MODELS
from src.models.layers import normalize_adjacency


N, T, C, L = 6, 60, 5, 12
HORIZONS = [1, 4, 8]


def _fake_graph(n=N):
    rng = np.random.default_rng(0)
    def sym():
        A = np.abs(rng.normal(size=(n, n))).astype(np.float32)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        return A
    layers = {"A_part": sym(), "A_lag": sym(), "A_agro": sym(), "A_sem": sym()}
    return MultiplexGraph(crops=[f"c{i}" for i in range(n)], layers=layers)


def _batch(b=4):
    return torch.randn(b, N, L, C)


@pytest.mark.parametrize(
    "kind",
    ["nograph", "identity", "static", "gat", "relational", "rgat", "adaptive"],
)
def test_point_forward_shape(kind):
    g = _fake_graph()
    model = build_model(kind, in_dim=C, num_nodes=N, horizons=HORIZONS,
                        graph=g, hidden=16)
    out = model(_batch(4))
    assert out.shape == (4, N, len(HORIZONS))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kind", ["nograph", "static", "gat", "relational", "rgat", "adaptive"])
def test_quantile_forward_shape_and_monotone(kind):
    g = _fake_graph()
    quantiles = [0.1, 0.5, 0.9]
    model = build_model(kind, in_dim=C, num_nodes=N, horizons=HORIZONS,
                        graph=g, hidden=16, quantiles=quantiles)
    out = model(_batch(3))
    assert out.shape == (3, N, len(HORIZONS), len(quantiles))
    # monotone non-decreasing across quantile axis
    diffs = out[..., 1:] - out[..., :-1]
    assert (diffs >= -1e-5).all()


def test_backward_updates_params():
    g = _fake_graph()
    model = build_model("relational", in_dim=C, num_nodes=N, horizons=HORIZONS,
                        graph=g, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = _batch(4)
    y = torch.randn(4, N, len(HORIZONS))
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    after = list(model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))


def test_adjacency_normalization_properties():
    A = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=float)
    A_hat = normalize_adjacency(A, directed=False)
    assert A_hat.shape == (3, 3)
    assert torch.allclose(A_hat, A_hat.t(), atol=1e-5)     # symmetric
    Ad = normalize_adjacency(A, directed=True)
    assert torch.allclose(Ad.sum(dim=1), torch.ones(3), atol=1e-5)  # row-stochastic


def test_capacity_control_has_no_graph_buffers():
    g = _fake_graph()
    ng = build_model("nograph", in_dim=C, num_nodes=N, horizons=HORIZONS, graph=g, hidden=16)
    buffers = dict(ng.named_buffers())
    assert not any("A_" in k for k in buffers)             # no adjacency baked in
