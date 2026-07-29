'''Spatio-temporal GNN forecasters (dense adjacency, pure PyTorch).

Model ladder
  NoGraphBackbone       capacity-matched control: the exact temporal backbone
                        with message passing OFF (the main control that
                        isolates cross-series pooling from topology).
  StaticGraphGNN        temporal encoder + GCN over a single fixed adjacency
                        (e.g. A_part or A_lag). The fixed-graph story.
  StaticGraphGATGNN     same fixed adjacency as StaticGraphGNN, but mixed via
                        masked multi-head attention (GAT) instead of GCN
                        degree normalisation. Operator-robustness check.
  RelationalMultiplexGNN  temporal encoder + RGCN over all edge layers stacked
                        (A_part, A_lag, A_te, A_agro, A_sem). Carries the
                        multi-relational claim.
  RelationalMultiplexGATGNN  same multiplex layers as RelationalMultiplexGNN,
                        but each relation is mixed via its own masked
                        attention instead of a fixed RGCN weight.
  AdaptiveGraphGNN      temporal encoder + learned adaptive adjacency
                        (MTGNN / Graph-WaveNet style). The learned-vs-fixed
                        comparison; can be seeded from a prior graph.
  IdentityGraphGNN      A = I sanity floor (StaticGraphGNN with identity adj).

All models share one temporal backbone (TemporalEncoder) and one output
head (point or quantile).

Interface
    forward(X) -> point:    (B, N, H)
                 quantile:  (B, N, H, Q)
where X is (B, N, L, C). Graphs are supplied as normalised dense tensors at
construction time (fixed within a fold).

Torch is imported here; the package __init__ guards this import so the rest
of the project (metrics, baselines, cv) works even where torch is absent.
'''

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.layers import (
    normalize_adjacency, DenseGraphConv, RelationalGraphConv,
    DenseGraphAttentionConv, RelationalGraphAttentionConv,
    AdaptiveAdjacency, AdaptiveGraphConv, TemporalEncoder,
    PointHead, QuantileHead,
)


class _STGNNBase(nn.Module):
    '''Temporal encode -> (optional) spatial mix -> forecast head.'''

    def __init__(self, in_dim: int, num_nodes: int, horizons: list,
                 hidden: int = 64, temporal_layers: int = 1,
                 dropout: float = 0.1, quantiles: list | None = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizons = list(horizons)
        self.H = len(horizons)
        self.quantiles = list(quantiles) if quantiles else None
        self.hidden = hidden

        self.encoder = TemporalEncoder(in_dim, hidden, temporal_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self._build_spatial(hidden)                 # sets self.spatial_out_dim

        head_in = self.spatial_out_dim
        if self.quantiles:
            self.head = QuantileHead(head_in, self.H, len(self.quantiles))
        else:
            self.head = PointHead(head_in, self.H)

    # Subclasses override these two.
    def _build_spatial(self, hidden: int):
        self.spatial_out_dim = hidden

    def _spatial(self, Z: torch.Tensor) -> torch.Tensor:
        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.encoder(X)                          # (B, N, hidden)
        Z = self.dropout(Z)
        Z = self._spatial(Z)                          # (B, N, spatial_out_dim)
        return self.head(Z)                           # (B, N, H) or (B,N,H,Q)


class NoGraphBackbone(_STGNNBase):
    '''Temporal backbone with graph mixing OFF (message passing disabled).

    Uses an extra node-wise MLP in place of the graph conv so the parameter
    count stays comparable to the graph models. That is the fair capacity
    control reviewers expect.
    '''
    name = 'NoGraphBackbone'

    def _build_spatial(self, hidden):
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        return self.mlp(Z)


class StaticGraphGNN(_STGNNBase):
    '''Temporal encoder + two GCN layers over one fixed adjacency.'''
    name = 'StaticGraphGNN'

    def __init__(self, in_dim, num_nodes, horizons, A_hat: torch.Tensor,
                 hidden=64, **kw):
        self._A_hat = A_hat
        super().__init__(in_dim, num_nodes, horizons, hidden=hidden, **kw)

    def _build_spatial(self, hidden):
        self.gc1 = DenseGraphConv(hidden, hidden, self._A_hat)
        self.gc2 = DenseGraphConv(hidden, hidden, self._A_hat)
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        Z = self.gc1(Z)
        Z = self.dropout(Z)
        return self.gc2(Z)


class IdentityGraphGNN(StaticGraphGNN):
    '''A = I sanity floor: message passing that mixes nothing across nodes.'''
    name = 'IdentityGraphGNN'

    def __init__(self, in_dim, num_nodes, horizons, hidden=64, **kw):
        A = torch.eye(num_nodes, dtype=torch.float32)
        super().__init__(in_dim, num_nodes, horizons, A_hat=A, hidden=hidden, **kw)


class StaticGraphGATGNN(_STGNNBase):
    '''Temporal encoder + two GAT-style attention layers over one fixed adjacency.

    Same graph support as StaticGraphGNN (e.g. A_part or A_lag), but edge
    weights are learned via masked multi-head attention instead of the GCN
    degree normalisation. Tests whether a fixed-graph result is an artifact
    of the specific message-passing operator.
    '''
    name = 'StaticGraphGATGNN'

    def __init__(self, in_dim, num_nodes, horizons, A_hat: torch.Tensor,
                 hidden=64, heads: int = 4, **kw):
        self._A_hat = A_hat
        self._heads = heads
        super().__init__(in_dim, num_nodes, horizons, hidden=hidden, **kw)

    def _build_spatial(self, hidden):
        self.gat1 = DenseGraphAttentionConv(hidden, hidden, self._A_hat, heads=self._heads)
        self.gat2 = DenseGraphAttentionConv(hidden, hidden, self._A_hat, heads=self._heads)
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        Z = self.gat1(Z)
        Z = self.dropout(Z)
        return self.gat2(Z)


class RelationalMultiplexGNN(_STGNNBase):
    '''Temporal encoder + two RGCN layers over the stacked multiplex layers.'''
    name = 'RelationalMultiplexGNN'

    def __init__(self, in_dim, num_nodes, horizons, A_list: list,
                 hidden=64, num_bases: int | None = None, **kw):
        self._A_list = A_list
        self._num_bases = num_bases
        super().__init__(in_dim, num_nodes, horizons, hidden=hidden, **kw)

    def _build_spatial(self, hidden):
        self.rgc1 = RelationalGraphConv(hidden, hidden, self._A_list, self._num_bases)
        self.rgc2 = RelationalGraphConv(hidden, hidden, self._A_list, self._num_bases)
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        Z = self.rgc1(Z)
        Z = self.dropout(Z)
        return self.rgc2(Z)


class RelationalMultiplexGATGNN(_STGNNBase):
    '''Temporal encoder + two relational-attention layers over the multiplex.

    Attention-based alternative to RelationalMultiplexGNN: each relation
    layer (A_part, A_lag, A_te, A_agro, A_sem) is mixed in via its own
    masked multi-head attention rather than a fixed RGCN weight matrix.
    '''
    name = 'RelationalMultiplexGATGNN'

    def __init__(self, in_dim, num_nodes, horizons, A_list: list,
                 hidden=64, heads: int = 2, **kw):
        self._A_list = A_list
        self._heads = heads
        super().__init__(in_dim, num_nodes, horizons, hidden=hidden, **kw)

    def _build_spatial(self, hidden):
        self.rgat1 = RelationalGraphAttentionConv(hidden, hidden, self._A_list, heads=self._heads)
        self.rgat2 = RelationalGraphAttentionConv(hidden, hidden, self._A_list, heads=self._heads)
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        Z = self.rgat1(Z)
        Z = self.dropout(Z)
        return self.rgat2(Z)


class AdaptiveGraphGNN(_STGNNBase):
    '''Temporal encoder + graph conv over a learned adaptive adjacency.'''
    name = 'AdaptiveGraphGNN'

    def __init__(self, in_dim, num_nodes, horizons, emb_dim: int = 16,
                 prior: torch.Tensor | None = None, hidden=64, **kw):
        self._emb_dim = emb_dim
        self._prior = prior
        super().__init__(in_dim, num_nodes, horizons, hidden=hidden, **kw)

    def _build_spatial(self, hidden):
        self.adj = AdaptiveAdjacency(self.num_nodes, self._emb_dim, self._prior)
        self.agc1 = AdaptiveGraphConv(hidden, hidden)
        self.agc2 = AdaptiveGraphConv(hidden, hidden)
        self.spatial_out_dim = hidden

    def _spatial(self, Z):
        A = self.adj()
        Z = self.agc1(Z, A)
        Z = self.dropout(Z)
        return self.agc2(Z, A)

    def learned_adjacency(self) -> np.ndarray:
        '''Return the current learned adjacency as a numpy array (for interp).'''
        with torch.no_grad():
            return self.adj().cpu().numpy()


def build_model(kind: str, *, in_dim: int, num_nodes: int, horizons: list,
                graph=None, layer_for_static: str = 'A_part',
                hidden: int = 64, quantiles: list | None = None,
                num_bases: int | None = None, emb_dim: int = 16,
                gat_heads: int = 4, rgat_heads: int = 2,
                directed_layers=('A_lag', 'A_te'), **kw):
    '''Construct a model by name, wiring in normalised adjacencies from a graph.

    Parameters
    kind : one of {'nograph','static','gat','identity','relational','rgat','adaptive'}.
    graph: a MultiplexGraph (needed for static/gat/relational/rgat/adaptive-with-prior).
    layer_for_static : which multiplex layer to use for StaticGraphGNN / StaticGraphGATGNN.
    directed_layers  : layer names normalised as directed (row-stochastic).
    gat_heads, rgat_heads : attention head counts for the GAT-style models.
    '''
    common = dict(in_dim=in_dim, num_nodes=num_nodes, horizons=horizons,
                  hidden=hidden, quantiles=quantiles, **kw)

    if kind == 'nograph':
        return NoGraphBackbone(**common)

    if kind == 'identity':
        return IdentityGraphGNN(**common)

    if kind == 'static':
        A = graph.layers[layer_for_static]
        A_hat = normalize_adjacency(A, directed=layer_for_static in directed_layers)
        return StaticGraphGNN(A_hat=A_hat, **common)

    if kind == 'gat':
        A = graph.layers[layer_for_static]
        A_hat = normalize_adjacency(A, directed=layer_for_static in directed_layers)
        return StaticGraphGATGNN(A_hat=A_hat, heads=gat_heads, **common)

    if kind == 'relational':
        A_list = [
            normalize_adjacency(A, directed=name in directed_layers)
            for name, A in graph.layers.items()
        ]
        return RelationalMultiplexGNN(A_list=A_list, num_bases=num_bases, **common)

    if kind == 'rgat':
        A_list = [
            normalize_adjacency(A, directed=name in directed_layers)
            for name, A in graph.layers.items()
        ]
        return RelationalMultiplexGATGNN(A_list=A_list, heads=rgat_heads, **common)

    if kind == 'adaptive':
        prior = None
        if graph is not None and layer_for_static in graph.layers:
            prior = normalize_adjacency(
                graph.layers[layer_for_static],
                directed=layer_for_static in directed_layers)
        return AdaptiveGraphGNN(emb_dim=emb_dim, prior=prior, **common)

    raise KeyError(f"unknown model kind '{kind}'")


TORCH_MODELS = ('nograph', 'identity', 'static', 'gat', 'relational', 'rgat', 'adaptive')
