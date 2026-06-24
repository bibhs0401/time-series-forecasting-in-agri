"""static_attributes.py
Static per-crop node attributes — broadcast across time.

Used by:
  • Attention / relational GNN layers  (raw categoricals → embeddings)
  • Agronomic edge layer A_agro        (shared season / commodity group)
  • Community-vs-season falsifiable test (Phase 6)

Source: LLM-drafted, human-verified against UF/IFAS and USDA before use.
Keep a diff between this draft and the verified final as an appendix artifact.

Columns
-------
  family        : botanical / commodity family (string)
  season_class  : 'cool', 'warm', 'year_round'
  perishability : 'high', 'medium', 'low'
  price_tier    : 'premium', 'mid', 'commodity'
  commodity_grp : broad market group — used to build A_agro edges
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# (family, season_class, perishability, price_tier, commodity_grp)
_ATTRS: dict[str, tuple] = {
    "Strawberry":   ("Berry",          "cool",       "high",   "premium",   "berry"),
    "Blueberry":    ("Berry",          "cool",       "high",   "premium",   "berry"),
    "Blackberry":   ("Berry",          "cool",       "high",   "premium",   "berry"),
    "Watermelon":   ("Cucurbit",       "warm",       "medium", "mid",       "melon"),
    "Cucumber":     ("Cucurbit",       "warm",       "high",   "mid",       "cucurbit"),
    "Zucchini":     ("Cucurbit",       "warm",       "high",   "mid",       "cucurbit"),
    "Pumpkin":      ("Cucurbit",       "warm",       "low",    "mid",       "cucurbit"),
    "Chayote":      ("Cucurbit",       "warm",       "medium", "mid",       "cucurbit"),
    "Tomato":       ("Nightshade",     "warm",       "high",   "mid",       "nightshade"),
    "Eggplant":     ("Nightshade",     "warm",       "high",   "mid",       "nightshade"),
    "Bell Pepper":  ("Nightshade",     "warm",       "high",   "mid",       "nightshade"),
    "Potato":       ("Nightshade",     "cool",       "low",    "commodity", "starch"),
    "Sweet Potato": ("Convolvulaceae", "warm",       "low",    "mid",       "starch"),
    "Cabbage":      ("Brassica",       "cool",       "medium", "commodity", "leafy"),
    "Cauliflower":  ("Brassica",       "cool",       "medium", "mid",       "leafy"),
    "Spinach":      ("Amaranth",       "cool",       "high",   "mid",       "leafy"),
    "Lettuce":      ("Asteraceae",     "cool",       "high",   "mid",       "leafy"),
    "Celery":       ("Apiaceae",       "cool",       "high",   "mid",       "leafy"),
    "Coriander":    ("Apiaceae",       "cool",       "high",   "mid",       "herb"),
    "Parsley":      ("Apiaceae",       "cool",       "high",   "mid",       "herb"),
    "Rosemary":     ("Lamiaceae",      "year_round", "low",    "mid",       "herb"),
    "Basil":        ("Lamiaceae",      "warm",       "high",   "mid",       "herb"),
    "Onion":        ("Allium",         "cool",       "low",    "commodity", "allium"),
    "Garlic":       ("Allium",         "cool",       "low",    "mid",       "allium"),
    "Avocado":      ("Lauraceae",      "year_round", "high",   "premium",   "tropical_fruit"),
    "Guava":        ("Myrtaceae",      "warm",       "high",   "premium",   "tropical_fruit"),
    "Papaya":       ("Caricaceae",     "year_round", "high",   "premium",   "tropical_fruit"),
    "Pineapple":    ("Bromeliaceae",   "year_round", "medium", "mid",       "tropical_fruit"),
    "Banana":       ("Musaceae",       "year_round", "high",   "commodity", "tropical_fruit"),
    "Carambola":    ("Oxalidaceae",    "warm",       "high",   "premium",   "tropical_fruit"),
    "Coconut":      ("Arecaceae",      "year_round", "low",    "mid",       "tropical_fruit"),
    "Lemon":        ("Citrus",         "year_round", "medium", "mid",       "citrus"),
    "Lime":         ("Citrus",         "year_round", "medium", "mid",       "citrus"),
    "Grapefruit":   ("Citrus",         "cool",       "medium", "mid",       "citrus"),
    "Citrus Fruit": ("Citrus",         "cool",       "medium", "mid",       "citrus"),
    "Grape":        ("Vitaceae",       "warm",       "high",   "premium",   "vine_fruit"),
    "Peach":        ("Rosaceae",       "warm",       "high",   "premium",   "stone_fruit"),
    "Plum":         ("Rosaceae",       "warm",       "high",   "premium",   "stone_fruit"),
    "Peanut":       ("Legume",         "warm",       "low",    "commodity", "nut"),
    "Pecan":        ("Juglandaceae",   "warm",       "low",    "premium",   "nut"),
    "Chestnut":     ("Fagaceae",       "cool",       "low",    "premium",   "nut"),
    "Okra":         ("Malvaceae",      "warm",       "high",   "mid",       "vegetable"),
    "Sugarcane":    ("Poaceae",        "warm",       "low",    "commodity", "field_crop"),
}

_COLS = ["family", "season_class", "perishability", "price_tier", "commodity_grp"]


def get_raw_attributes(crops: list) -> pd.DataFrame:
    """Return raw (un-encoded) attribute table, indexed by crop name.

    Missing crops get NaN rows — check the printout and fill gaps.
    """
    rows = {}
    for crop in crops:
        if crop in _ATTRS:
            rows[crop] = dict(zip(_COLS, _ATTRS[crop]))
        else:
            print(f"  WARNING: no static attributes for '{crop}' — row will be NaN")
            rows[crop] = {c: None for c in _COLS}
    return pd.DataFrame(rows).T[_COLS]


def get_attribute_matrix(crops: list) -> pd.DataFrame:
    """One-hot encode all categorical attributes.

    Returns DataFrame indexed by crop, columns = one-hot dummies.
    Shape: (N, D) where D depends on cardinality of each column.
    Use .to_numpy(dtype=float) to get the raw array for model input.
    """
    raw = get_raw_attributes(crops)
    return pd.get_dummies(raw, dtype=float)


def commodity_group_edges(crops: list) -> list[tuple[str, str]]:
    """Return all (crop_i, crop_j) pairs that share the same commodity_grp.

    Used to build the A_agro adjacency in Phase 4.
    """
    raw = get_raw_attributes(crops)
    edges = []
    for grp, members in raw.groupby("commodity_grp"):
        names = members.index.tolist()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                edges.append((a, b))
    return edges


def season_class_edges(crops: list) -> list[tuple[str, str]]:
    """Return (crop_i, crop_j) pairs that share the same season_class.

    Secondary agro edge layer — crops that compete for the same growing window.
    """
    raw = get_raw_attributes(crops)
    edges = []
    for cls, members in raw.groupby("season_class"):
        names = members.index.tolist()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                edges.append((a, b))
    return edges


if __name__ == "__main__":
    import config
    crops = pd.read_csv(config.PANEL_DIR / "crop_list.csv")["crop"].tolist()
    raw = get_raw_attributes(crops)
    print(raw.to_string())
    print(f"\nOne-hot shape: {get_attribute_matrix(crops).shape}")
    print(f"Commodity edges: {len(commodity_group_edges(crops))}")
