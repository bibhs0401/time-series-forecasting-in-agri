'''Static adjacency layers.

connects crops that share botanical family, growing season, and semantic similarity.

  A_agro : agronomic prior graph. Edges between crops that share botanical
           family or season class, from static_attributes.py.

  A_sem  : semantic embedding graph. Cosine similarity over sentence
           embeddings of crop descriptions (sentence-transformers).
'''

import functools
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.static_attributes import (
    get_raw_attributes,
    family_edges,
    season_class_edges,
)
from src.graph.graph_utils import adjacency_stats


def build_agro_adjacency(crops: list,
                          family_weight: float = 1.0,
                          season_weight: float = 0.5) -> np.ndarray:
    '''Build the agronomic prior adjacency matrix.

    Edges:
      - Same family        : weight = family_weight (stronger prior)
      - Same season_class  : weight = season_weight (weaker prior)
      - Both               : weight = max of the two

    Self-loops are zeroed.

    Parameters
    crops          : ordered list of N crop names
    family_weight  : edge weight for same-family pairs
    season_weight  : edge weight for same-season-class pairs

    Returns
    A_agro : (N, N) float32, symmetric, no self-loops
    '''
    N = len(crops)
    idx = {c: i for i, c in enumerate(crops)}
    A = np.zeros((N, N), dtype=np.float32)

    for (a, b) in family_edges(crops):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = max(A[i, j], family_weight)
            A[j, i] = max(A[j, i], family_weight)

    for (a, b) in season_class_edges(crops):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = max(A[i, j], season_weight)
            A[j, i] = max(A[j, i], season_weight)

    np.fill_diagonal(A, 0.0)
    return A


def agro_edge_table(crops: list,
                     family_weight: float = 1.0,
                     season_weight: float = 0.5) -> pd.DataFrame:
    '''Return a human-readable table of all A_agro edges.

    Useful for a paper appendix or manual checks. Pass the same weights
    used in build_agro_adjacency() / build_static_layers() so the table
    matches the adjacency that was actually used.

    Columns: crop_i, crop_j, family_match, season_class_match, weight
    '''
    raw = get_raw_attributes(crops)
    A = build_agro_adjacency(crops, family_weight=family_weight, season_weight=season_weight)

    rows = []
    for i, ci in enumerate(crops):
        for j, cj in enumerate(crops):
            if j <= i:
                continue
            if A[i, j] > 0:
                fam_match = (raw.loc[ci, 'family'] ==
                             raw.loc[cj, 'family']
                             if ci in raw.index and cj in raw.index else False)
                seas_match = (raw.loc[ci, 'season_class'] ==
                              raw.loc[cj, 'season_class']
                              if ci in raw.index and cj in raw.index else False)
                rows.append(dict(
                    crop_i=ci, crop_j=cj,
                    family_match=fam_match,
                    season_class_match=seas_match,
                    weight=float(A[i, j]),
                ))
    return pd.DataFrame(rows)


_SEM_MODEL_CACHE: dict = {}
_DEFAULT_SEM_DESCRIPTOR = '{crop}, a Florida agricultural crop'


def _get_sentence_model(model_name: str):
    '''Lazily load and cache a sentence-transformers model by name.

    Loading the model is the expensive part of A_sem. Caching it
    module-wide means repeated calls (once per fold) only pay that cost
    once per process.
    '''
    if model_name not in _SEM_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _SEM_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _SEM_MODEL_CACHE[model_name]


def build_sem_adjacency(crops: list,
                         model_name: str = 'all-MiniLM-L6-v2',
                         threshold: float = 0.3,
                         descriptor_template: str = _DEFAULT_SEM_DESCRIPTOR) -> np.ndarray:
    '''Semantic adjacency from sentence-embedding cosine similarity.

    Steps:
      1. Load a sentence-transformers model (cached module-wide).
      2. Embed each crop with a short descriptor (descriptor_template).
      3. Compute pairwise cosine similarity (embeddings are L2-normalised,
         so this is just the dot product).
      4. Zero entries below threshold and self-loops.

    Falls back to a zero matrix (with a warning) if sentence-transformers
    is missing or embedding fails. Callers do not need to special-case this.

    Parameters
    crops               : ordered list of N crop names
    model_name          : sentence-transformers model id
    threshold           : cosine-similarity cutoff; entries below this are zeroed
    descriptor_template : format string with a single '{crop}' placeholder

    Returns
    A_sem : (N, N) float32, symmetric, no self-loops
    '''
    N = len(crops)
    A = np.zeros((N, N), dtype=np.float32)
    if N == 0:
        return A

    try:
        model = _get_sentence_model(model_name)
    except ImportError:
        warnings.warn(
            '  A_sem: sentence-transformers not installed. '
            'Run: pip install sentence-transformers\n'
            '  Returning zero matrix.'
        )
        return A
    except Exception as e:
        warnings.warn(f"  A_sem: failed to load model '{model_name}' ({e}). Returning zeros.")
        return A

    descriptors = [descriptor_template.format(crop=crop) for crop in crops]
    try:
        embeddings = model.encode(
            descriptors, normalize_embeddings=True, show_progress_bar=False
        )
        sim = np.asarray(embeddings, dtype=np.float32) @ np.asarray(embeddings, dtype=np.float32).T
    except Exception as e:
        warnings.warn(f'  A_sem: embedding/similarity computation failed ({e}). Returning zeros.')
        return A

    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    sim[sim < threshold] = 0.0
    return sim


@functools.lru_cache(maxsize=8)
def _build_static_layers_cached(crops_tuple: tuple,
                                 family_weight: float,
                                 season_weight: float,
                                 model_name: str,
                                 threshold: float,
                                 descriptor_template: str) -> tuple:
    '''Cached core of build_static_layers(); keyed on hashable args only.

    A_agro and A_sem are fold-invariant, so once computed for a given
    (crops, hyperparameters) pair the result is reused across folds.
    That mainly helps A_sem, which would otherwise reload a transformer
    model on every fold.
    '''
    crops = list(crops_tuple)
    A_agro = build_agro_adjacency(crops, family_weight=family_weight, season_weight=season_weight)
    A_sem = build_sem_adjacency(
        crops, model_name=model_name, threshold=threshold,
        descriptor_template=descriptor_template,
    )
    return A_agro, A_sem


def build_static_layers(crops: list,
                         family_weight: float = 1.0,
                         season_weight: float = 0.5,
                         model_name: str = 'all-MiniLM-L6-v2',
                         threshold: float = 0.3,
                         descriptor_template: str = _DEFAULT_SEM_DESCRIPTOR,
                         use_cache: bool = True,
                         verbose: bool = True) -> dict:
    '''Build and return all static adjacency layers.

    Hyperparameters of build_agro_adjacency() / build_sem_adjacency()
    are exposed here so callers (e.g. build_graph.py) can tune them without
    editing this module. Results are memoised by default (use_cache=True)
    since both layers only depend on crops and these hyperparameters.

    Parameters
    crops               : ordered list of N crop names
    family_weight       : forwarded to build_agro_adjacency
    season_weight       : forwarded to build_agro_adjacency
    model_name          : forwarded to build_sem_adjacency
    threshold           : forwarded to build_sem_adjacency
    descriptor_template : forwarded to build_sem_adjacency
    use_cache           : reuse a cached result for identical arguments
    verbose             : print per-layer stats

    Returns
    dict with keys 'A_agro', 'A_sem'; each (N, N) float32
    '''
    if use_cache:
        A_agro_c, A_sem_c = _build_static_layers_cached(
            tuple(crops), family_weight, season_weight,
            model_name, threshold, descriptor_template,
        )
        # Return copies: lru_cache reuses the same array objects, so callers
        # must not get a mutable reference into the cache.
        A_agro, A_sem = A_agro_c.copy(), A_sem_c.copy()
    else:
        A_agro = build_agro_adjacency(crops, family_weight=family_weight, season_weight=season_weight)
        A_sem = build_sem_adjacency(
            crops, model_name=model_name, threshold=threshold,
            descriptor_template=descriptor_template,
        )

    if verbose:
        print('Static adjacency layers:')
        adjacency_stats(A_agro, 'A_agro')
        adjacency_stats(A_sem,  'A_sem')

    return {'A_agro': A_agro, 'A_sem': A_sem}


if __name__ == '__main__':
    import time
    import config
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()

    layers = build_static_layers(crops)

    print('\nA_agro edge table (first 10):')
    tbl = agro_edge_table(crops)
    print(tbl.head(10).to_string(index=False))
    print(f'  Total A_agro edges: {len(tbl)}')

    print('\nA_sem: most similar crop pairs (top 10 by cosine similarity):')
    A_sem = layers['A_sem']
    pairs = [
        (crops[i], crops[j], float(A_sem[i, j]))
        for i in range(len(crops)) for j in range(i + 1, len(crops))
        if A_sem[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    for ci, cj, w in pairs[:10]:
        print(f'    {ci:15s} <-> {cj:15s}  {w:.3f}')
    print(f'  Total A_sem edges (threshold={0.3}): {len(pairs)}')

    print('\nCache check: rebuilding with identical args should be near-instant ...')
    t0 = time.perf_counter()
    build_static_layers(crops, verbose=False)
    print(f'  Second call took {time.perf_counter() - t0:.4f}s (cached)')
