"""calendar_features.py
Fourier seasonality terms + per-crop FL in/out-of-season binary flags.

FL crop calendar sources:
  [N] USDA NASS FL Statistical Bulletin 2024, Table E-7
  [F] FDACS / Fresh from Florida Seasonal Availability Calendar 2023
  [U] UF/IFAS Miami-Dade -- Tropical Fruit Seasons in Florida
  [P] Pickyourown.org Florida Harvest Calendar

Peak-season windows are (start_iso_week, end_iso_week), 1-52.
start > end means the window wraps across the new year.
"""

import numpy as np
import pandas as pd

PERIOD      = 52
N_HARMONICS = 4

FL_SEASON = {
    # Berries
    "Strawberry":    (44, 17),  # Nov->Apr [N][F]
    "Blueberry":     (14, 26),  # Apr->Jun [N][F]
    "Blackberry":    (14, 21),  # Apr->May [P]
    # Cucurbits
    "Watermelon":    (14, 35),  # Apr->Aug [N][F]
    "Cucumber":      (40, 21),  # Oct->May [N] two-season
    "Zucchini":      (40, 26),  # Oct->Jun [N]
    "Pumpkin":       (40,  4),  # Oct->Jan [P]
    "Chayote":       (31, 52),  # Aug->Dec [U]
    # Nightshades
    "Tomato":        (40, 26),  # Oct->Jun [N][F]
    "Eggplant":      (40, 26),  # Oct->Jun [N]
    "Bell Pepper":   (40, 26),  # Oct->Jun [N][F]
    # Root / starchy
    "Potato":        ( 5, 26),  # Feb->Jun [N]
    "Sweet Potato":  (36, 52),  # Sep->Dec [N][P]
    # Brassicas / leafy
    "Cabbage":       (48, 13),  # Dec->Mar [N][F]
    "Cauliflower":   (44, 17),  # Nov->Apr [F]
    "Spinach":       (40, 13),  # Oct->Mar [F][P]
    "Lettuce":       (48, 13),  # Dec->Mar [N][F]
    "Celery":        (48, 13),  # Dec->Mar [N][F]
    # Herbs
    "Coriander":     (40, 17),  # Oct->Apr [F]
    "Parsley":       (40, 17),  # Oct->Apr [N]
    "Rosemary":      ( 1, 52),  # year-round [U]
    "Basil":         (14, 43),  # Apr->Oct [P]
    # Alliums
    "Onion":         ( 5, 21),  # Feb->May [P][N]
    "Garlic":        (10, 21),  # Mar->May [P]
    # Tropical fruits
    "Avocado":       (22,  8),  # Jun->Feb [U]
    "Guava":         (31, 43),  # Aug->Oct [U]
    "Papaya":        ( 1, 52),  # year-round [U]
    "Pineapple":     (22, 39),  # Jun->Sep [U]
    "Banana":        ( 1, 52),  # year-round [U]
    "Carambola":     (22,  8),  # Jun->Feb [U]
    "Coconut":       ( 1, 52),  # year-round [U]
    # Citrus
    "Lemon":         (40, 21),  # Oct->May [U]
    "Lime":          (22, 43),  # Jun->Oct [U]
    "Grapefruit":    (40, 17),  # Oct->Apr [U][F]
    "Citrus Fruit":  (40, 21),  # Oct->May [U]
    # Stone / vine fruits
    "Grape":         (31, 43),  # Aug->Oct [P]
    "Peach":         (10, 21),  # Mar->May [P][F]
    "Plum":          (14, 21),  # Apr->May [P]
    # Nuts / field crops
    "Peanut":        (31, 43),  # Aug->Oct [N]
    "Pecan":         (40, 52),  # Oct->Dec [P]
    "Chestnut":      (40, 52),  # Oct->Dec [P]
    "Okra":          (18, 43),  # May->Oct [P]
    "Sugarcane":     (40, 13),  # Oct->Mar [N]
}


def make_fourier_features(index, period=PERIOD, n_harmonics=N_HARMONICS):
    """sin/cos Fourier terms k=1..n_harmonics.  Shape: (T, 2*n_harmonics)."""
    t = np.arange(len(index))
    cols = {}
    for k in range(1, n_harmonics + 1):
        cols[f"sin_k{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_k{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def _week_of_year(index):
    """ISO week number clipped to 1-52."""
    return np.clip(index.isocalendar().week.to_numpy(dtype=int), 1, 52)


def _in_season_mask(week, start, end):
    """Binary mask; handles wrap-around (start > end)."""
    if start <= end:
        return ((week >= start) & (week <= end)).astype(float)
    return ((week >= start) | (week <= end)).astype(float)


def make_season_flags(index, crops, calendar=None):
    """Return one binary column per crop: 1 = within peak FL season.

    Column names: {crop}_in_season.
    Crops absent from the calendar default to 1.0 (always in season).
    """
    if calendar is None:
        calendar = FL_SEASON
    week = _week_of_year(index)
    cols = {}
    for crop in crops:
        if crop in calendar:
            s, e = calendar[crop]
            cols[f"{crop}_in_season"] = _in_season_mask(week, s, e)
        else:
            cols[f"{crop}_in_season"] = np.ones(len(index), dtype=float)
    return pd.DataFrame(cols, index=index)
