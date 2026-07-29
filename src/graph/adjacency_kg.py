'''Knowledge-graph relational layers for the multiplex graph.

These are curated, static crop->crop relations that encode domain knowledge
the data-driven layers (A_part, A_lag, A_te) cannot see, and that A_agro only
captures coarsely (family + season_class mixed into one layer). Each layer is
a separate RGCN relation so the model can weight them independently.

Layers
  A_taxon : same botanical family (clean, single-relation version of the
            family component currently folded into A_agro).
  A_subst : search / culinary substitutes - crops a searcher may swap for
            one another (lemon<->lime, peach<->plum, spinach<->lettuce).
            This is the relation with the strongest demand-forecasting case.
  A_compl : recipe / co-search complements - crops used together
            (tomato+basil, onion+garlic, carrot+celery+onion = mirepoix).
  A_pheno : phenological overlap - Jaccard of per-crop in-season week masks,
            derived from the G2 UF/IFAS season flags (not hand-curated).

All matrices are (N, N) float32, symmetric, zero diagonal, matching the
convention in adjacency_static.py so they drop straight into the multiplex.

Edges for A_taxon / A_subst / A_compl come from
src/data/crop_relations.csv. Weights are priors, not ground truth; their
value is decided by the ablation, not asserted here.
'''

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import PROJECT_ROOT
except Exception:  # pragma: no cover - config import is environment dependent
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RELATIONS_CSV = PROJECT_ROOT / 'src' / 'data' / 'crop_relations.csv'

# Curated relations that live in the CSV (A_pheno is derived, not curated).
CURATED_RELATIONS = ('taxon', 'subst', 'compl')


def _norm(name: str) -> str:
    '''Canonicalise a crop name for matching (case/space insensitive).'''
    return str(name).strip().lower()


def load_relation_edges(csv_path: str | Path | None = None) -> pd.DataFrame:
    '''Load the curated crop-relation edge list.

    Comment lines (starting with '#') and blanks are ignored.

    Returns a DataFrame with columns: crop_i, crop_j, relation, weight, rationale.
    '''
    csv_path = Path(csv_path) if csv_path else DEFAULT_RELATIONS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f'KG relations CSV not found: {csv_path}')
    df = pd.read_csv(csv_path, comment='#', skip_blank_lines=True)
    df = df.dropna(subset=['crop_i', 'crop_j', 'relation'])
    df['relation'] = df['relation'].str.strip().str.lower()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
    return df.reset_index(drop=True)


def _adjacency_from_edges(edges: pd.DataFrame, crops: list) -> np.ndarray:
    '''Build a symmetric (N, N) adjacency from an edge subset for one relation.'''
    N = len(crops)
    idx = {_norm(c): i for i, c in enumerate(crops)}
    A = np.zeros((N, N), dtype=np.float32)
    unmatched = set()
    for _, r in edges.iterrows():
        a, b = _norm(r['crop_i']), _norm(r['crop_j'])
        if a not in idx:
            unmatched.add(r['crop_i'])
            continue
        if b not in idx:
            unmatched.add(r['crop_j'])
            continue
        i, j, w = idx[a], idx[b], float(r['weight'])
        A[i, j] = max(A[i, j], w)
        A[j, i] = max(A[j, i], w)  # keep the layer symmetric
    np.fill_diagonal(A, 0.0)
    if unmatched:
        warnings.warn(
            f'  adjacency_kg: {len(unmatched)} crop name(s) in CSV not in the '
            f'panel and were skipped: {sorted(unmatched)}'
        )
    return A


def build_curated_layer(relation: str, crops: list,
                        csv_path: str | Path | None = None,
                        edges: pd.DataFrame | None = None) -> np.ndarray:
    '''Build one curated relational adjacency (relation in CURATED_RELATIONS).'''
    if edges is None:
        edges = load_relation_edges(csv_path)
    sub = edges[edges['relation'] == relation.lower()]
    return _adjacency_from_edges(sub, crops)


def build_pheno_adjacency(season_flags: pd.DataFrame, crops: list,
                          min_jaccard: float = 0.0) -> np.ndarray:
    '''Phenological-overlap adjacency from per-crop in-season binary flags.

    Parameters
    season_flags : (T, N) 0/1 DataFrame, crop columns, 1 == in-season that week
                   (this is exactly the G2 calendar channel).
    crops        : ordered crop list defining node order.
    min_jaccard  : drop edges below this overlap (keeps the layer sparse).

    Edge weight[i, j] = |in_i AND in_j| / |in_i OR in_j|  (Jaccard of season masks)
    '''
    N = len(crops)
    A = np.zeros((N, N), dtype=np.float32)
    cols = {_norm(c): c for c in season_flags.columns}
    masks = {}
    for c in crops:
        col = cols.get(_norm(c))
        masks[c] = (season_flags[col].to_numpy() > 0) if col is not None else None

    for i, ci in enumerate(crops):
        mi = masks[ci]
        if mi is None:
            continue
        for j in range(i + 1, N):
            mj = masks[crops[j]]
            if mj is None:
                continue
            union = np.logical_or(mi, mj).sum()
            if union == 0:
                continue
            jacc = float(np.logical_and(mi, mj).sum()) / float(union)
            if jacc >= min_jaccard:
                A[i, j] = A[j, i] = jacc
    np.fill_diagonal(A, 0.0)
    return A


def build_kg_layers(crops: list,
                    season_flags: pd.DataFrame | None = None,
                    csv_path: str | Path | None = None,
                    include: tuple = ('A_taxon', 'A_subst', 'A_compl', 'A_pheno'),
                    verbose: bool = True) -> dict:
    '''Return a dict of KG relational adjacencies keyed like the other layers.

    Mirrors build_static_layers() in adjacency_static.py so the output merges
    directly into build_multiplex_graph's all_layers dict.

    A_pheno is only built if season_flags is provided.
    '''
    edges = load_relation_edges(csv_path)
    layers: dict = {}
    name_to_rel = {'A_taxon': 'taxon', 'A_subst': 'subst', 'A_compl': 'compl'}

    for name, rel in name_to_rel.items():
        if name in include:
            layers[name] = build_curated_layer(rel, crops, edges=edges)

    if 'A_pheno' in include:
        if season_flags is not None:
            layers['A_pheno'] = build_pheno_adjacency(season_flags, crops)
        elif verbose:
            warnings.warn('  adjacency_kg: A_pheno requested but no season_flags '
                          'given; skipping.')

    if verbose:
        for name, A in layers.items():
            nnz = int((A > 0).sum() // 2)
            print(f'  {name:8s}: {nnz} undirected edges, '
                  f'mean w={A[A > 0].mean() if nnz else 0:.3f}')
    return layers


def kg_edge_table(crops: list, csv_path: str | Path | None = None) -> pd.DataFrame:
    '''Human-readable table of curated KG edges that survive crop matching.

    Useful for a paper appendix or verification
    (columns: crop_i, crop_j, relation, weight, rationale).
    Only rows whose both crops are in the panel are kept.
    '''
    edges = load_relation_edges(csv_path)
    keep = {_norm(c) for c in crops}
    mask = edges['crop_i'].map(_norm).isin(keep) & edges['crop_j'].map(_norm).isin(keep)
    return edges[mask].reset_index(drop=True)


if __name__ == '__main__':
    # Smoke test on the real crop list.
    crop_csv = PROJECT_ROOT / 'outputs' / 'panel' / 'crop_list.csv'
    crops = pd.read_csv(crop_csv)['crop'].tolist()
    print(f'N crops = {len(crops)}')
    layers = build_kg_layers(crops, verbose=True)
    print('Layers built:', list(layers.keys()))
