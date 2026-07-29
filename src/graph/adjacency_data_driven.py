'''Data-driven adjacency layers.
Learns relationships from historical trends, 
including correlation, lead–lag effects, and information transfer.

Input to all three layers: z-normalised STL remainders from the training
slice (from src/features/stl_decomp.py). Pass the output of
z_normalize_remainders(compute_stl_remainders(panel_train)).

Layers:
  A_part : Graphical-lasso partial-correlation precision matrix.
           Captures direct conditional dependence after removing shared
           trends. Symmetric, undirected.

  A_lag  : Directed cross-correlation argmax.
           A_lag[i, j] is the max normalised cross-correlation of crop i
           leading crop j at lags 1..max_lag weeks. Asymmetric (directed).

  A_te   : Transfer entropy from crop i -> crop j (PyIF).
           Nonlinear directed information; supplements A_lag.
           Asymmetric (directed).

Statistical edge validation (FDR) is deferred to Phase 6.
'''

import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV


def build_part_adjacency(remainders: pd.DataFrame,
                          alphas: list = None,
                          max_iter: int = 500) -> np.ndarray:
    '''Partial-correlation adjacency via Graphical Lasso.

    Parameters
    remainders : (T, N) DataFrame of z-normalised STL remainders (train only).
                 All-NaN columns are excluded; those crops get zeros in A.
    alphas     : regularisation grid for CV (default: log-spaced 0.01..1.0)
    max_iter   : max iterations for the lasso solver

    Returns
    A_part : (N, N) float32, symmetric, no self-loops.
             Entry [i, j] = |partial correlation| between crops i and j.
    '''
    if alphas is None:
        alphas = list(np.logspace(-2, 0, 10))

    N = len(remainders.columns)
    crops = remainders.columns.tolist()
    A = np.zeros((N, N), dtype=np.float32)

    # Drop crops with no usable data.
    valid = remainders.dropna(axis=1, how='all')
    dropped = [c for c in crops if c not in valid.columns]
    if dropped:
        warnings.warn(f'  A_part: dropped {len(dropped)} all-NaN crops: {dropped}')

    X = valid.dropna().to_numpy(dtype=float)   # (T', N_valid)
    if X.shape[0] < 2 * X.shape[1]:
        warnings.warn(
            f'  A_part: T={X.shape[0]} < 2*N={2*X.shape[1]}. '
            'Graphical Lasso may be unstable; enforcing stronger regularisation.'
        )
        alphas = [a * 2 for a in alphas]

    try:
        gl = GraphicalLassoCV(alphas=alphas, max_iter=max_iter, cv=5)
        gl.fit(X)
        prec = gl.precision_              # (N_valid, N_valid)
        # Convert precision to partial correlation.
        d    = np.sqrt(np.diag(prec))
        pcor = -prec / np.outer(d, d)
        np.fill_diagonal(pcor, 0.0)
        pcor = np.abs(pcor).astype(np.float32)

        # Map back into the full (N, N) layout.
        valid_idx = [crops.index(c) for c in valid.columns]
        for ii, vi in enumerate(valid_idx):
            for jj, vj in enumerate(valid_idx):
                A[vi, vj] = pcor[ii, jj]

    except Exception as e:
        warnings.warn(f'  A_part: GraphicalLassoCV failed ({e}). Returning zeros.')

    np.fill_diagonal(A, 0.0)
    return A


def build_lag_adjacency(remainders: pd.DataFrame,
                         max_lag: int = 13) -> np.ndarray:
    '''Directed lead-lag adjacency via normalised cross-correlation.

    A_lag[i, j] is the max normalised cross-correlation of series i leading
    series j at positive lags 1..max_lag. A positive weight means i leads j
    in raw correlation terms (before any statistical test).

    Parameters
    remainders : (T, N) DataFrame, z-normalised STL remainders (train only).
    max_lag    : maximum lead lag in weeks (default 13 = one quarter)

    Returns
    A_lag : (N, N) float32, asymmetric (directed), no self-loops.
    '''
    crops = remainders.columns.tolist()
    N = len(crops)
    A = np.zeros((N, N), dtype=np.float32)

    arr = remainders.to_numpy(dtype=float)  # (T, N)
    T = arr.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xi = arr[:, i]
            xj = arr[:, j]
            # Drop rows where either series is NaN.
            valid = ~(np.isnan(xi) | np.isnan(xj))
            xi_v = xi[valid]
            xj_v = xj[valid]
            if len(xi_v) < max_lag + 10:
                continue

            best_xcorr = 0.0
            for lag in range(1, max_lag + 1):
                # xi leads xj: correlate xi[t-lag] with xj[t]
                a = xi_v[:-lag]
                b = xj_v[lag:]
                if len(a) < 10:
                    continue
                std_a = np.std(a)
                std_b = np.std(b)
                if std_a < 1e-8 or std_b < 1e-8:
                    continue
                xcorr = np.mean((a - a.mean()) * (b - b.mean())) / (std_a * std_b)
                if abs(xcorr) > abs(best_xcorr):
                    best_xcorr = xcorr

            A[i, j] = max(0.0, float(best_xcorr))  # keep positive correlations only

    np.fill_diagonal(A, 0.0)
    return A


def build_te_adjacency(remainders: pd.DataFrame,
                        k: int = 1,
                        l: int = 1,
                        delay: int = 1,
                        normalize: bool = True) -> np.ndarray:
    '''Directed transfer-entropy adjacency via PyIF.

    A_te[i, j] = TE from crop i -> crop j. Captures nonlinear directed
    information beyond linear cross-correlation.

    Parameters
    remainders : (T, N) DataFrame, z-normalised STL remainders (train only).
    k          : history length of the target (j)
    l          : history length of the source (i)
    delay      : time delay (1 = one-step-ahead)
    normalize  : if True, divide each TE value by H(j) to get a [0,1] range

    Returns
    A_te : (N, N) float32, asymmetric (directed), no self-loops.

    Requires: pip install pyif
    '''
    try:
        from pyif import te
    except ImportError:
        warnings.warn(
            '  A_te: pyif not installed. Run: pip install pyif --break-system-packages\n'
            '  Returning zero matrix.'
        )
        N = len(remainders.columns)
        return np.zeros((N, N), dtype=np.float32)

    crops = remainders.columns.tolist()
    N = len(crops)
    A = np.zeros((N, N), dtype=np.float32)
    arr = remainders.dropna().to_numpy(dtype=float)  # (T', N)

    if arr.shape[0] < 50:
        warnings.warn('  A_te: too few clean rows after dropna. Returning zeros.')
        return A

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xi = arr[:, i]
            xj = arr[:, j]
            try:
                te_val = te(xi, xj, k=k, l=l, delay=delay)
                if normalize:
                    # Entropy of the target; use a simple binning estimate.
                    h_j = _entropy_hist(xj)
                    te_val = te_val / h_j if h_j > 1e-8 else 0.0
                A[i, j] = max(0.0, float(te_val))
            except Exception as e:
                warnings.warn(f'  A_te[{i},{j}]: TE failed ({e})')

    np.fill_diagonal(A, 0.0)
    return A


def _entropy_hist(x: np.ndarray, bins: int = 20) -> float:
    '''Histogram estimate of differential entropy (for TE normalisation).'''
    counts, _ = np.histogram(x, bins=bins, density=False)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def build_data_driven_layers(remainders: pd.DataFrame,
                              include_te: bool = True,
                              max_lag:    int   = 13,
                              verbose:    bool  = True) -> dict:
    '''Build A_part, A_lag, and optionally A_te from training remainders.

    Parameters
    remainders : z-normalised STL remainders, TRAINING SLICE ONLY.
    include_te : whether to compute transfer entropy (slow for large N)
    max_lag    : passed to build_lag_adjacency
    verbose    : print adjacency stats

    Returns
    dict with keys 'A_part', 'A_lag', and optionally 'A_te'
    '''
    from src.graph.graph_utils import adjacency_stats

    if verbose:
        print('Building data-driven adjacency layers ...')

    print('  Computing A_part (GraphicalLasso) ...')
    A_part = build_part_adjacency(remainders)

    print(f'  Computing A_lag (max_lag={max_lag}) ...')
    A_lag = build_lag_adjacency(remainders, max_lag=max_lag)

    layers = {'A_part': A_part, 'A_lag': A_lag}

    if include_te:
        print('  Computing A_te (transfer entropy via PyIF) ...')
        A_te = build_te_adjacency(remainders)
        layers['A_te'] = A_te

    if verbose:
        for name, A in layers.items():
            adjacency_stats(A, name)

    return layers


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import config
    from src.utils.io import read_panel
    from src.features.stl_decomp import compute_stl_remainders, z_normalize_remainders

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    panel = panel[crops]

    # Use the first 400 weeks as a mock train slice.
    panel_train = panel.iloc[:400]
    print(f'Train slice: {panel_train.shape}')

    R  = compute_stl_remainders(panel_train)
    Rz = z_normalize_remainders(R)

    layers = build_data_driven_layers(Rz, include_te=True)
    print('\nDone.')
