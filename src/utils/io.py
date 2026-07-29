"""Shared I/O helpers for loading raw Trends CSVs and reading/writing panels."""

import re
from pathlib import Path

import pandas as pd

import config

# Google Trends multi-timeline exports start with two metadata rows:
#   Category: All categories
#   <blank>
# then the real header: Week,Crop1: (Florida),...
_GEO_SUFFIX = re.compile(r"\s*:\s*\([^)]*\)\s*$")


def _clean_crop_name(name: str) -> str:
    """Strip geo suffix and normalise to lowercase crop label."""
    return _GEO_SUFFIX.sub("", str(name)).strip().lower()


def load_trends_csv(path: Path) -> pd.DataFrame:
    """Load one raw Google Trends CSV.

    The header row is (Week, Crop1, Crop2, ...). Returns a DataFrame indexed
    by date with lowercased column names and numeric values. Google Trends
    encodes near-zero values as the string "<1", which is mapped to
    ``config.LT1_VALUE``.
    """
    df = pd.read_csv(path, skiprows=2)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.columns = [_clean_crop_name(c) for c in df.columns]
    df = df.replace(r"^\s*<\s*1\s*$", config.LT1_VALUE, regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_group_windows(group: str) -> list:
    """Return all half-year windows for `group`, sorted chronologically."""
    folder = config.DATA_DIR / group
    files = sorted(folder.glob(f"{group}_google_trends_p*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    windows = []
    for f in files:
        df = load_trends_csv(f)
        windows.append(df)
        print(f"    {f.name}  "
              f"{df.index.min().date()} → {df.index.max().date()}")
    return windows


def read_panel(csv_path: Path = config.PANEL_CSV) -> pd.DataFrame:
    """Read the stitched panel CSV into a date-indexed DataFrame."""
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def write_panel(panel: pd.DataFrame, csv_path: Path = config.PANEL_CSV) -> None:
    """Write the stitched panel to CSV, creating parent dirs as needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(csv_path)
