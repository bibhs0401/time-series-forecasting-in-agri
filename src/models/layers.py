'''Dense-adjacency graph and temporal building blocks (pure PyTorch).

Why dense, not PyTorch-Geometric
The graph has only N ~= 43 nodes, so a full (N, N) adjacency costs < 2k
floats and message passing is a single dense matmul A_hat @ H. That removes
the heavy PyG / PyG-Temporal dependency, installs anywhere torch does, and
consumes the MultiplexGraph.layers dict (dense (N, N) arrays) directly.

Every layer here is a plain torch.nn.Module. Torch is imported at module
load; import this module only when torch is available (see src/models/gnn.py,
which guards the import for the rest of the package).

Adjacency normalisation
normalize_adjacency implements the standard GCN renormalisation
    A_hat = D^{-1/2} (A + I) D^{-1/2}
for symmetric layers, and row-stochastic D^{-1}(A+I) for directed layers.
Normalisation is done once per fold (adjacency is fixed within a fold) and
the result is registered as a buffer so it moves with the module across devices.
'''

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(A: np.ndarray, directed: bool = False,
                        add_self_loops: bool = True) -> torch.Tensor:
    '''Return a normalised dense adjacency as a float32 torch tensor (N, N).'''
    A = np.asarray(A, dtype=np.float64).copy()
    A = np.abs(A)                                  # weights used as connectivity
    N = A.shape[0]
    if add_self_loops:
        A = A + np.eye(N)
    if directed:
        d = A.sum(axis=1, keepdims=True)
        d[d == 0] = 1.0
        A_hat = A / d                               # row-stochastic
    else:
        A = np.maximum(A, A.T)                       # ensure symmetric
        d = A.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        A_hat = (A * d_inv_sqrt[None, :]) * d_inv_sqrt[:, None]
    return torch.tensor(A_hat, dtype=torch.float32)


class DenseGraphConv(nn.Module):
    '''One GCN layer: H' = act( A_hat @ H @ W + b ).

    Input  H : (B, N, F_in)
    Output   : (B, N, F_out)
    A_hat is a fixed (N, N) buffer set at construction (per fold).
    '''

    def __init__(self, in_dim: int, out_dim: int, A_hat: torch.Tensor,
                 activation=F.relu, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.register_buffer('A_hat', A_hat)
        self.activation = activation

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        H = self.lin(H)                              # (B, N, F_out)
        H = torch.einsum('nm,bmf->bnf', self.A_hat, H)
        return self.activation(H) if self.activation is not None else H


class RelationalGraphConv(nn.Module):
    '''RGCN-style layer over R fixed relation adjacencies (the multiplex).

    H' = act( sum_r  A_hat_r @ H @ W_r  +  H @ W_self + b )

    A_list : list of R normalised (N, N) buffers, one per edge layer
             (A_part, A_lag, A_te, A_agro, A_sem). Basis decomposition keeps
             the parameter count modest when R is large.
    '''

    def __init__(self, in_dim: int, out_dim: int, A_list: list,
                 num_bases: int | None = None, activation=F.relu):
        super().__init__()
        self.R = len(A_list)
        self.out_dim = out_dim
        for r, A in enumerate(A_list):
            self.register_buffer(f'A_{r}', A)
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.activation = activation

        if num_bases is None or num_bases >= self.R:
            # One weight matrix per relation.
            self.weight = nn.Parameter(torch.empty(self.R, in_dim, out_dim))
            self.comp = None
        else:
            # Basis decomposition: W_r = sum_b comp[r,b] * bases[b]
            self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
            self.comp = nn.Parameter(torch.empty(self.R, num_bases))
            self.weight = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.comp)

    def _relation_weights(self) -> torch.Tensor:
        if self.weight is not None:
            return self.weight
        return torch.einsum('rb,bio->rio', self.comp, self.bases)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        W = self._relation_weights()                 # (R, in, out)
        out = self.self_lin(H)
        for r in range(self.R):
            A = getattr(self, f'A_{r}')
            Hr = torch.einsum('bnf,fo->bno', H, W[r])
            out = out + torch.einsum('nm,bmo->bno', A, Hr)
        return self.activation(out) if self.activation is not None else out


class DenseGraphAttentionConv(nn.Module):
    '''One GAT-style layer: masked multi-head attention over a fixed adjacency.

    Input  H : (B, N, F_in)
    Output   : (B, N, F_out)   F_out = heads * (F_out // heads) when concat=True

    `mask` fixes *which* edges exist (nonzero entries of a normalised A_hat,
    e.g. from normalize_adjacency) but attention weights are learned rather
    than the GCN degree normalisation, so this is a drop-in alternative to
    DenseGraphConv over the same graph support.
    '''

    def __init__(self, in_dim: int, out_dim: int, mask: torch.Tensor,
                 heads: int = 4, concat: bool = True, dropout: float = 0.1,
                 negative_slope: float = 0.2, activation=F.elu):
        super().__init__()
        if concat and out_dim % heads != 0:
            raise ValueError(f'out_dim ({out_dim}) must be divisible by heads ({heads})')
        self.heads = heads
        self.concat = concat
        self.out_per_head = out_dim // heads if concat else out_dim
        self.lin = nn.Linear(in_dim, heads * self.out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, heads, self.out_per_head))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, self.out_per_head))
        self.bias = nn.Parameter(torch.zeros(heads * self.out_per_head if concat else self.out_per_head))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)
        self.activation = activation
        self.register_buffer('mask', (mask != 0))       # (N, N) bool, True = edge allowed
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, N, _ = H.shape
        Hp = self.lin(H).view(B, N, self.heads, self.out_per_head)   # (B, N, heads, F)
        alpha_i = (Hp * self.att_src).sum(-1)          # (B, N, heads)  "target" term
        alpha_j = (Hp * self.att_dst).sum(-1)          # (B, N, heads)  "source" term
        e = alpha_i.unsqueeze(2) + alpha_j.unsqueeze(1)   # (B, N_i, N_j, heads)
        e = self.leaky_relu(e)
        mask = self.mask.unsqueeze(0).unsqueeze(-1)     # (1, N, N, 1)
        e = e.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(e, dim=2)                      # normalise over neighbours j
        alpha = torch.nan_to_num(alpha, nan=0.0)          # rows with no edges: no-op
        alpha = self.attn_dropout(alpha)
        out = torch.einsum('bijh,bjhf->bihf', alpha, Hp)  # (B, N, heads, F)
        out = out.reshape(B, N, -1) if self.concat else out.mean(dim=2)
        out = out + self.bias
        return self.activation(out) if self.activation is not None else out


class RelationalGraphAttentionConv(nn.Module):
    '''GAT-style layer over R fixed relation adjacencies (the multiplex).

    H' = act( H @ W_self  +  sum_r  DenseGraphAttentionConv_r(H) )

    Attention-based alternative to RelationalGraphConv: each relation layer
    (A_part, A_lag, A_te, A_agro, A_sem, ...) gets its own masked multi-head
    attention instead of a fixed per-relation weight matrix.
    '''

    def __init__(self, in_dim: int, out_dim: int, mask_list: list,
                 heads: int = 2, dropout: float = 0.1, activation=F.elu):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.rel_convs = nn.ModuleList([
            DenseGraphAttentionConv(in_dim, out_dim, mask, heads=heads,
                                     concat=True, dropout=dropout, activation=None)
            for mask in mask_list
        ])
        self.activation = activation

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        out = self.self_lin(H)
        for conv in self.rel_convs:
            out = out + conv(H)
        return self.activation(out) if self.activation is not None else out


class AdaptiveAdjacency(nn.Module):
    '''Self-adaptive adjacency from learned node embeddings.

        A = softmax( relu( E1 @ E2^T ) )

    Discovers structure end-to-end (design doc §7, A_learned). Optionally
    seeded from a prior adjacency to warm-start toward known relations.
    '''

    def __init__(self, num_nodes: int, emb_dim: int = 16,
                 prior: torch.Tensor | None = None):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        if prior is not None:
            self.register_buffer('prior', prior)
        else:
            self.prior = None

    def forward(self) -> torch.Tensor:
        A = F.relu(self.E1 @ self.E2.t())
        A = F.softmax(A, dim=1)                       # row-stochastic
        if self.prior is not None:
            A = 0.5 * A + 0.5 * self.prior
        return A


class AdaptiveGraphConv(nn.Module):
    '''Graph conv over a dynamically produced adjacency (from AdaptiveAdjacency).'''

    def __init__(self, in_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = self.lin(H)
        H = torch.einsum('nm,bmf->bnf', A, H)
        return self.activation(H) if self.activation is not None else H


class TemporalEncoder(nn.Module):
    '''Encode each node's (L, C) window into a hidden vector with a GRU.

    Input  X : (B, N, L, C)
    Output   : (B, N, hidden)  last hidden state per node.
    The same GRU weights are shared across nodes (a global temporal model).
    '''

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, L, C = X.shape
        x = X.reshape(B * N, L, C)
        out, h = self.gru(x)
        last = out[:, -1, :]                          # (B*N, hidden)
        return last.reshape(B, N, self.hidden)


class PointHead(nn.Module):
    '''Map node embeddings to one point forecast per horizon. (B, N, H).'''

    def __init__(self, in_dim: int, horizons: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, horizons))
        else:
            self.net = nn.Linear(in_dim, horizons)

    def forward(self, Z):
        return self.net(Z)


class QuantileHead(nn.Module):
    '''Map node embeddings to quantile forecasts per horizon. (B, N, H, Q).

    Quantiles are made monotone in Q via a cumulative-softplus parameterisation
    so predicted intervals never cross.
    '''

    def __init__(self, in_dim: int, horizons: int, num_quantiles: int,
                 hidden: int = 0):
        super().__init__()
        self.H = horizons
        self.Q = num_quantiles
        out = horizons * num_quantiles
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        else:
            self.net = nn.Linear(in_dim, out)

    def forward(self, Z):
        B, N, _ = Z.shape
        raw = self.net(Z).reshape(B, N, self.H, self.Q)
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
