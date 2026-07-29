'''Shared helpers for building and converting adjacency matrices.
'''

import numpy as np


def symmetrize(A: np.ndarray, method: str = 'max') -> np.ndarray:
    '''Make A symmetric.

    method='max' : A_sym[i,j] = max(A[i,j], A[j,i])  (preserves signal)
    method='avg' : A_sym[i,j] = (A[i,j] + A[j,i]) / 2
    '''
    if method == 'max':
        return np.maximum(A, A.T)
    elif method == 'avg':
        return (A + A.T) / 2
    else:
        raise ValueError(f"Unknown method '{method}'; use 'max' or 'avg'.")


def threshold_topk(A: np.ndarray, k: int, directed: bool = False) -> np.ndarray:
    '''Keep the top-k neighbours per node; zero the rest.

    For directed graphs, keep the top-k outgoing edges per row.
    For undirected graphs, symmetrize after thresholding.
    '''
    N = A.shape[0]
    out = np.zeros_like(A)
    for i in range(N):
        row = A[i].copy()
        row[i] = -np.inf                          # exclude self
        top_idx = np.argsort(row)[::-1][:k]
        out[i, top_idx] = A[i, top_idx]
    if not directed:
        out = symmetrize(out, method='max')
    return out


def threshold_absolute(A: np.ndarray, tau: float) -> np.ndarray:
    '''Zero entries with |A[i,j]| < tau.'''
    out = A.copy()
    out[np.abs(out) < tau] = 0.0
    return out


def threshold_percentile(A: np.ndarray, pct: float = 80.0,
                          directed: bool = False) -> np.ndarray:
    '''Keep entries above the pct-th percentile of non-self absolute values.'''
    mask = ~np.eye(A.shape[0], dtype=bool)
    vals = np.abs(A[mask])
    tau = np.percentile(vals, pct)
    return threshold_absolute(A, tau)


def row_normalize(A: np.ndarray) -> np.ndarray:
    '''Row-stochastic normalisation: divide each row by its sum.'''
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0               # avoid divide-by-zero
    return A / row_sums


def sym_normalize(A: np.ndarray) -> np.ndarray:
    '''Symmetric D^{-1/2} A D^{-1/2} normalisation (GCN-style).'''
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def add_self_loops(A: np.ndarray, weight: float = 1.0) -> np.ndarray:
    '''Add self-loops by setting the diagonal to weight.'''
    out = A.copy()
    np.fill_diagonal(out, weight)
    return out


def to_pyg_edge_index(A: np.ndarray,
                       min_weight: float = 1e-6
                       ) -> tuple:
    '''Convert a dense adjacency matrix to PyG (edge_index, edge_weight).

    Returns
    edge_index  : np.ndarray of shape (2, E), dtype int64
    edge_weight : np.ndarray of shape (E,),   dtype float32
    '''
    rows, cols = np.where(np.abs(A) > min_weight)
    edge_index  = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = A[rows, cols].astype(np.float32)
    return edge_index, edge_weight


def stack_multiplex(layer_dict: dict,
                    min_weight: float = 1e-6
                    ) -> tuple:
    '''Stack multiple adjacency layers into one heterogeneous edge set.

    Parameters
    layer_dict : {layer_name: A (N,N)}, e.g. {'A_part': ..., 'A_lag': ...}

    Returns
    edge_index  : (2, E_total) int64
    edge_weight : (E_total,) float32
    edge_type   : (E_total,) int64 integer label per layer
    layer_names : list of str matching the edge_type integers
    '''
    all_ei, all_ew, all_et = [], [], []
    layer_names = list(layer_dict.keys())
    for t, name in enumerate(layer_names):
        A = layer_dict[name]
        ei, ew = to_pyg_edge_index(A, min_weight=min_weight)
        all_ei.append(ei)
        all_ew.append(ew)
        all_et.append(np.full(ei.shape[1], t, dtype=np.int64))
    edge_index  = np.concatenate(all_ei, axis=1)
    edge_weight = np.concatenate(all_ew)
    edge_type   = np.concatenate(all_et)
    return edge_index, edge_weight, edge_type, layer_names


def adjacency_stats(A: np.ndarray, name: str = 'A') -> dict:
    '''Print and return summary stats for an adjacency matrix.'''
    N = A.shape[0]
    mask = ~np.eye(N, dtype=bool)
    nnz  = (np.abs(A[mask]) > 0).sum()
    density = nnz / (N * (N - 1))
    stats = dict(
        name=name, N=N, nnz=int(nnz),
        density=round(float(density), 4),
        weight_mean=round(float(A[mask][A[mask] != 0].mean()) if nnz else 0, 4),
        weight_max=round(float(np.abs(A[mask]).max()), 4),
        is_symmetric=bool(np.allclose(A, A.T, atol=1e-6)),
    )
    print(f'  {name}: N={N}, nnz={nnz}, density={density:.3f}, '
          f"symmetric={stats['is_symmetric']}")
    return stats
