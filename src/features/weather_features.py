"""weather_features.py
NOAA Climate Data Online — Florida statewide weekly weather features.

One-time bulk download; results cached to outputs/panel/noaa_weather_fl.csv.
Delete the cache file to force a re-download.

Variables downloaded
--------------------
  TMAX  — daily max temperature (°F)
  TMIN  — daily min temperature (°F)
  PRCP  — daily precipitation (NOAA "standard" units = inches, converted to mm)

Derived
-------
  GDD   — growing degree days: max(0, (TMAX + TMIN)/2 - 50°F) per day,
           summed to weekly totals.

Aggregation
-----------
Daily station readings are averaged across all FL GHCND stations
(locationid="FIPS:12") then resampled to weekly (W-SUN):
  TMAX, TMIN → weekly mean
  PRCP       → weekly sum
  GDD        → weekly sum

PRISM alternative
-----------------
If the NOAA station API is too slow or patchy, download Florida statewide
monthly/daily gridded rasters from https://prism.oregonstate.edu/recent/
and replace the fetch logic below.  The cache CSV format and align_to_panel()
are unchanged — only build_weather_panel() needs updating.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CACHE_PATH = config.PANEL_DIR / "noaa_weather_fl.csv"
NOAA_BASE  = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
PAGE_LIMIT = 1000   # max records per NOAA API call


def _fetch_datatype(datatype: str,
                    start: str,
                    end: str,
                    token: str,
                    locationid: str = "FIPS:12") -> pd.DataFrame:
    """Fetch all daily records for one datatype in [start, end] using pagination.

    Parameters
    ----------
    datatype   : NOAA CDO datatype id, e.g. 'TMAX'
    start, end : 'YYYY-MM-DD'
    token      : NOAA API token
    locationid : 'FIPS:12' = Florida statewide

    Returns
    -------
    DataFrame with columns ['date', 'value'], date as DatetimeIndex.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    headers = {"token": token}
    all_rows = []
    offset = 1

    while True:
        params = dict(
            datasetid="GHCND",
            locationid=locationid,
            datatypeid=datatype,
            startdate=start,
            enddate=end,
            units="standard",
            limit=PAGE_LIMIT,
            offset=offset,
        )
        resp = requests.get(NOAA_BASE, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results", [])
        if not results:
            break
        all_rows.extend(results)
        total = body["metadata"]["resultset"]["count"]
        offset += PAGE_LIMIT
        if offset > total:
            break
        time.sleep(0.2)   # stay within NOAA rate limits

    if not all_rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    # average across stations on the same day
    daily = df.groupby("date")["value"].mean().rename(datatype)
    return daily.to_frame()


def build_weather_panel(start: str = "2016-01-01",
                        end:   str = "2025-12-31",
                        token: str = "PMcVuObDvgpZKlSHGJZrTSUMSzYMmlZc",
                        cache: Path = CACHE_PATH) -> pd.DataFrame:
    """Download FL weather, derive GDD, resample to weekly, cache result.

    Parameters
    ----------
    start, end : date range matching the Trends panel
    token      : NOAA API token (falls back to NOAA_TOKEN env var)
    cache      : path to cache CSV; skips download if it exists

    Returns
    -------
    DataFrame with columns [TMAX, TMIN, PRCP, GDD], weekly W-SUN index.
    """
    if cache.exists():
        print(f"  Loading weather from cache: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    token = token or os.environ.get("NOAA_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "NOAA API token required.\n"
            "  Register at https://www.ncdc.noaa.gov/cdo-web/token\n"
            "  Then: export NOAA_TOKEN=your_token"
        )

    print("  Fetching NOAA TMAX …")
    tmax = _fetch_datatype("TMAX", start, end, token)
    print("  Fetching NOAA TMIN …")
    tmin = _fetch_datatype("TMIN", start, end, token)
    print("  Fetching NOAA PRCP …")
    prcp = _fetch_datatype("PRCP", start, end, token)

    daily = tmax.join(tmin, how="outer").join(prcp, how="outer")
    daily["PRCP"] = daily["PRCP"] * 25.4   # inches (NOAA "standard" units) → mm

    # Derive daily GDD (base 50°F, standard for most FL crops)
    daily["GDD"] = ((daily["TMAX"] + daily["TMIN"]) / 2 - 50).clip(lower=0)

    # Resample to weekly (Sunday-ending to match Trends panel)
    weekly = daily.resample("W-SUN").agg(
        {"TMAX": "mean", "TMIN": "mean", "PRCP": "sum", "GDD": "sum"}
    )
    weekly.index.name = "Date"

    cache.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(cache)
    print(f"  Saved weather panel → {cache}")
    return weekly


def align_to_panel(weather: pd.DataFrame,
                   panel_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex weekly weather to the exact panel dates.

    Forward-fills at most 2 consecutive missing weeks (e.g. at year-start
    boundary); anything beyond is left as NaN and flagged downstream.
    """
    aligned = weather.reindex(panel_index)
    n_missing = aligned.isna().any(axis=1).sum()
    if n_missing:
        print(f"  Weather: {n_missing} panel weeks missing → forward-filling (max 2)")
    return aligned.ffill(limit=2)


if __name__ == "__main__":
    wx = build_weather_panel()
    print(wx.tail())
    print(f"\nShape: {wx.shape}")
