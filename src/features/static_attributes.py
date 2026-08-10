'''Static per-crop node attributes, broadcast across time.

Used by attention / relational GNN layers (raw categoricals go to embeddings).

Attribute table: src/data/static_attributes.csv

Columns:
  family        : botanical / commodity family (string)
  season_class  : 'cool', 'warm', 'year_round' #crop’s main growing-temperature season
  #cool: crops that grow best in cool temperatures (e.g. lettuce, spinach)
  #warm: crops that grow best in warm temperatures (e.g. tomatoes, peppers)
  #year_round: crops that grow year-round (e.g. citrus, avocados)
  perishability : 'high', 'medium', 'low'
  # how quickly a crop typically loses quality after harvest:
  # high: crops that lose quality quickly (e.g. lettuce, spinach)
  # medium: crops that lose quality moderately (e.g. tomatoes, peppers)
  # low: crops that lose quality slowly (e.g. citrus, avocados)
'''

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ATTRS_PATH = Path(__file__).resolve().parents[1] / 'data' / 'static_attributes.csv'
_COLS = ['family', 'season_class', 'perishability']

_ATTRS_DF: pd.DataFrame | None = None


def _load_attrs() -> pd.DataFrame:
    '''Load and cache the static attribute table, indexed by crop.'''
    global _ATTRS_DF
    if _ATTRS_DF is None:
        df = pd.read_csv(_ATTRS_PATH)
        missing = [c for c in ['crop', *_COLS] if c not in df.columns]
        if missing:
            raise ValueError(f'{_ATTRS_PATH} missing columns: {missing}')
        _ATTRS_DF = df.set_index('crop')[_COLS]
    return _ATTRS_DF


def get_raw_attributes(crops: list) -> pd.DataFrame:
    '''Return the raw (un-encoded) attribute table, indexed by crop.

    Crops missing from the table get a NaN row.
    '''
    attrs = _load_attrs()
    rows = {}
    for crop in crops:
        if crop in attrs.index:
            rows[crop] = attrs.loc[crop].to_dict()
        else:
            print(f"  WARNING: no static attributes for '{crop}'; row will be NaN")
            rows[crop] = {c: None for c in _COLS}
    return pd.DataFrame(rows).T[_COLS]


def get_attribute_matrix(crops: list) -> pd.DataFrame:
    '''One-hot encode all categorical attributes.

    Returns a DataFrame indexed by crop with one-hot dummy columns.
    Shape is (N, D), where D depends on the cardinality of each column.
    '''
    raw = get_raw_attributes(crops)
    return pd.get_dummies(raw, dtype=float)


def _pairwise_edges(crops: list, col: str) -> list[tuple[str, str]]:
    '''Return (crop_i, crop_j) pairs that share the same value of col.'''
    raw = get_raw_attributes(crops)
    edges = []
    for _, members in raw.groupby(col):
        names = members.index.tolist()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                edges.append((a, b))
    return edges


def family_edges(crops: list) -> list[tuple[str, str]]:
    '''Return all (crop_i, crop_j) pairs that share the same family.

    Used to build the A_agro adjacency in Phase 4.
    '''
    return _pairwise_edges(crops, 'family')


def season_class_edges(crops: list) -> list[tuple[str, str]]:
    '''Return (crop_i, crop_j) pairs that share the same season_class.

    Secondary agro edge layer: crops that compete for the same growing window.
    '''
    return _pairwise_edges(crops, 'season_class')


if __name__ == '__main__':
    import config
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    raw = get_raw_attributes(crops)
    print(raw.to_string())
    print(f'\nOne-hot shape: {get_attribute_matrix(crops).shape}')
    print(f'Family edges: {len(family_edges(crops))}')
