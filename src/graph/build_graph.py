'''Combines all relationship layers into one multiplex graph for the GNN.
'''

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.stl_decomp import compute_stl_remainders, z_normalize_remainders
from src.features.calendar_features import make_season_flags
from src.graph.adjacency_static      import build_static_layers
from src.graph.adjacency_data_driven import build_data_driven_layers
from src.graph.adjacency_kg          import build_kg_layers
from src.graph.graph_utils           import (
    threshold_percentile, stack_multiplex, adjacency_stats
)


@dataclass
class MultiplexGraph:
    '''Container for one fold's multiplex graph.

    Attributes
    crops        : ordered crop list (N crops)
    layers       : dict layer_name -> (N, N) float32 adjacency
    edge_index   : (2, E) int64; all layers stacked
    edge_weight  : (E,) float32
    edge_type    : (E,) int64; integer layer id
    layer_names  : list of str matching edge_type integers
    stats        : per-layer summary dicts
    '''
    crops:       list
    layers:      dict       = field(default_factory=dict)
    edge_index:  np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=np.int64))
    edge_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    edge_type:   np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    layer_names: list       = field(default_factory=list)
    stats:       dict       = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.crops)

    @property
    def E(self) -> int:
        return self.edge_index.shape[1]

    def summary(self):
        print(f'MultiplexGraph: N={self.N} nodes, E={self.E} total edges, '
              f'{len(self.layers)} layers: {self.layer_names}')
        for name, A in self.layers.items():
            adjacency_stats(A, name)


def build_multiplex_graph(
    panel_train:       pd.DataFrame,
    crops:             list,
    include_te:        bool  = True,
    include_kg:        bool  = True,
    kg_relations_csv:  Optional[str] = None,
    stl_period:        int   = 52,
    max_lag:           int   = 13,
    threshold_pct:     float = 80.0,
    min_edge_weight:   float = 1e-4,
    verbose:           bool  = True,
) -> MultiplexGraph:
    '''Build the multiplex graph for one fold.

    Steps
    1. Compute STL remainders on panel_train (train only).
    2. Z-normalise remainders.
    3. Build data-driven layers (A_part, A_lag, A_te) from remainders.
    4. Build static layers (A_agro, A_sem) from crop metadata.
    4b. Build knowledge-graph relational layers (A_taxon, A_subst, A_compl,
        A_pheno) from curated crop relations + the FL season calendar.
    5. Threshold each layer to keep only strong edges.
    6. Stack all layers into PyG (edge_index, edge_weight, edge_type).

    Parameters
    panel_train     : (T_train, N) Trends panel, training slice only.
    crops           : ordered list of N crop names.
    include_te      : whether to compute A_te (slow).
    include_kg      : whether to add the curated KG layers (A_taxon, A_subst,
                      A_compl) and the derived A_pheno season-overlap layer.
    kg_relations_csv: optional override path to crop_relations.csv
                      (defaults to src/data/crop_relations.csv).
    stl_period      : annual period for STL (default 52 weeks).
    max_lag         : max lead-lag for A_lag (weeks).
    threshold_pct   : percentile threshold applied per layer (drop weak edges).
    min_edge_weight : minimum weight to include an edge in PyG output.
    verbose         : print progress and stats.

    Returns
    MultiplexGraph
    '''
    if verbose:
        print(f'\n=== build_multiplex_graph | T_train={len(panel_train)}, N={len(crops)} ===')

    # 1+2. STL remainders
    if verbose:
        print('Step 1/4: STL decomposition + z-normalisation ...')
    panel_sub = panel_train[crops] if set(crops).issubset(panel_train.columns) else panel_train
    R  = compute_stl_remainders(panel_sub, period=stl_period)
    Rz = z_normalize_remainders(R)

    nan_crops = Rz.columns[Rz.isna().all()].tolist()
    if nan_crops:
        warnings.warn(f'  {len(nan_crops)} crops have all-NaN remainders: {nan_crops}')

    # 3. Data-driven layers
    if verbose:
        print('Step 2/4: Data-driven layers (A_part, A_lag, A_te) ...')
    dd_layers = build_data_driven_layers(
        Rz, include_te=include_te, max_lag=max_lag, verbose=verbose
    )

    # 4. Static layers
    if verbose:
        print('Step 3/4: Static layers (A_agro, A_sem) ...')
    st_layers = build_static_layers(crops, verbose=verbose)

    # 4b. Knowledge-graph relational layers
    # Curated (A_taxon/A_subst/A_compl) + derived A_pheno. All are static and
    # leakage-free: they depend only on crop identity and the FL season
    # calendar, never on panel_train values (A_pheno uses the date index only).
    kg_layers: dict = {}
    if include_kg:
        if verbose:
            print('Step 3b/4: KG layers (A_taxon, A_subst, A_compl, A_pheno) ...')
        try:
            season_flags = make_season_flags(panel_train.index, crops)
            # build_pheno_adjacency matches on crop names; strip the
            # '_in_season' suffix from make_season_flags.
            season_flags.columns = [c.replace('_in_season', '')
                                    for c in season_flags.columns]
            kg_layers = build_kg_layers(
                crops, season_flags=season_flags,
                csv_path=kg_relations_csv, verbose=verbose)
        except FileNotFoundError as e:
            warnings.warn(f'  KG layers skipped (relations CSV missing): {e}')
        except Exception as e:                       # pragma: no cover
            warnings.warn(f'  KG layers skipped (build error): {e}')

    # 5. Combine + threshold
    if verbose:
        print(f'Step 4/4: Thresholding at {threshold_pct}th percentile ...')

    all_layers = {**dd_layers, **st_layers, **kg_layers}
    thresholded = {}
    stats = {}
    for name, A in all_layers.items():
        directed = name in ('A_lag', 'A_te')
        A_thr = threshold_percentile(A, pct=threshold_pct, directed=directed)
        thresholded[name] = A_thr
        stats[name] = adjacency_stats(A_thr, name) if verbose else {}

    # 6. Stack into PyG format
    ei, ew, et, layer_names = stack_multiplex(thresholded, min_weight=min_edge_weight)

    if verbose:
        print(f'\n  Total edges across all layers: {ei.shape[1]}')
        print(f'  Layer order: {layer_names}')

    return MultiplexGraph(
        crops       = crops,
        layers      = thresholded,
        edge_index  = ei,
        edge_weight = ew,
        edge_type   = et,
        layer_names = layer_names,
        stats       = stats,
    )


def save_graph(graph: MultiplexGraph, path) -> None:
    '''Save a MultiplexGraph to an .npz file.'''
    path = Path(path)
    arrays = {f'layer_{k}': v for k, v in graph.layers.items()}
    arrays['edge_index']  = graph.edge_index
    arrays['edge_weight'] = graph.edge_weight
    arrays['edge_type']   = graph.edge_type
    np.savez(path, **arrays)
    # Save metadata separately.
    meta_path = path.with_suffix('.meta.csv')
    pd.DataFrame({
        'crop': graph.crops,
        'layer_names': [','.join(graph.layer_names)] + [''] * (len(graph.crops) - 1)
    }).to_csv(meta_path, index=False)
    print(f'  Graph saved -> {path}')


def load_graph(path) -> MultiplexGraph:
    '''Load a MultiplexGraph from .npz + .meta.csv.'''
    path = Path(path)
    data = np.load(path)
    meta = pd.read_csv(path.with_suffix('.meta.csv'))
    crops = meta['crop'].tolist()
    layer_names = meta['layer_names'].iloc[0].split(',')
    layers = {k.replace('layer_', ''): data[k] for k in data.files
              if k.startswith('layer_')}
    return MultiplexGraph(
        crops       = crops,
        layers      = layers,
        edge_index  = data['edge_index'],
        edge_weight = data['edge_weight'],
        edge_type   = data['edge_type'],
        layer_names = layer_names,
    )


if __name__ == '__main__':
    import config
    from src.utils.io import read_panel

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    panel = panel[crops]

    # Mock fold: first 400 weeks as train.
    train = panel.iloc[:400]

    graph = build_multiplex_graph(
        panel_train = train,
        crops       = crops,
        include_te  = True,
        verbose     = True,
    )

    graph.summary()
