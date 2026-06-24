"""weather_features.py
NOAA Climate Data Online — Florida statewide weekly weather features.

One-time bulk download; results cached to outputs/panel/noaa_weather_fl.csv.
Delete the cache file to force a re-download.

Variables downloaded (daily, station-averaged across FL)
-------------------------------------------------------
  TMAX  — daily max temperature (°F)
  TMIN  — daily min temperature (°F)
  PRCP  — daily precipitation (NOAA "standard" units = inches, converted to mm)

Weekly panel columns (resampled W-SUN)
--------------------------------------
LEVEL group (smooth, seasonal — partly redundant with Fourier/season flags):
  TMAX, TMIN  — weekly mean (°F)
  TAVG        — weekly mean of (TMAX+TMIN)/2 (°F)
  PRCP        — weekly total precipitation (mm)
  PRCP_max    — wettest single day in the week (mm); storm-intensity proxy
  GDD         — growing degree days, base 50°F, weekly sum
  GDD_accum   — GDD accumulated since Jan 1 (resets each year); phenology proxy

SHOCK group (event counts — the news-driven, search-relevant signal):
  freeze_days — # days TMIN <= 32°F  (FL freeze events → citrus/strawberry spikes)
  heat_days   — # days TMAX >= 90°F  (heat stress)
  rain_days   — # days PRCP >= 1 mm  (wet-spell frequency)
  chill_days  — # days 32°F <= TAVG <= 45°F  (chill accumulation: peach/blueberry)

ANOMALY group (computed separately, leakage-safe — see add_weather_anomalies):
  {var}_anom  — z-score vs week-of-year climatology FIT ON TRAIN FOLD ONLY.
                This is the non-redundant signal: departure from the normal
                seasonal cycle already captured by the calendar features.

Aggregation
-----------
Daily station readings are averaged across all FL GHCND stations
(locationid="FIPS:12") then resampled to weekly (W-SUN). Means for levels,
sums for totals/counts, max for PRCP_max.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CACHE_PATH       = config.PANEL_DIR / "noaa_weather_fl.csv"
RAW_CACHE_PATH   = config.PANEL_DIR / "noaa_weather_fl_daily_raw.csv"
FETCH_CACHE_DIR  = config.PANEL_DIR / "_wx_fetch_cache"   # per-(datatype, window) resume cache
NOAA_BASE  = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
PAGE_LIMIT = 1000   # max records per NOAA API call
DEFAULT_TOKEN = "PMcVuObDvgpZKlSHGJZrTSUMSzYMmlZc"
MAX_RETRIES = 6     # retry transient 5xx / 429 responses
RATE_DELAY  = 0.35  # seconds between successful page requests (NOAA: 5 req/s max)

# Daily-event thresholds (°F, mm) — all in NOAA "standard" units
GDD_BASE_F   = 50.0   # growing degree day base temperature
FREEZE_F     = 32.0   # TMIN at/below this = freeze day
HEAT_F       = 90.0   # TMAX at/above this = heat-stress day
CHILL_LO_F   = 32.0   # chill window lower bound (TAVG)
CHILL_HI_F   = 45.0   # chill window upper bound (TAVG)
RAIN_MIN_MM  = 1.0    # PRCP at/above this = measurable rain day

# Column groups (for the §9 feature-group ablation)
WX_LEVEL_COLS = ["TMAX", "TMIN", "TAVG", "PRCP", "PRCP_max", "GDD", "GDD_accum"]
WX_SHOCK_COLS = ["freeze_days", "heat_days", "rain_days", "chill_days"]
WX_ANOM_BASE  = ["TMAX", "TMIN", "TAVG", "PRCP", "GDD"]   # vars to z-score


def _get_with_retry(requests, params, headers):
    """GET one NOAA page, retrying transient failures with exponential backoff.

    Two kinds of transient failure are retried:
      * HTTP 429 / 5xx — NOAA throttling or temporary server overload.
      * Network errors (ConnectionResetError, timeouts, chunked-encoding) — the
        server forcibly closed the connection mid-transfer (Windows WinError
        10054). These raise a requests.exceptions.RequestException before any
        response object exists, so they must be caught separately.

    A 4xx other than 429 (e.g. bad token / bad params) is raised immediately.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        wait = min(2 ** attempt, 30)   # 2,4,8,16,30,30 …
        try:
            resp = requests.get(NOAA_BASE, headers=headers, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"      network error ({type(exc).__name__}) — "
                  f"retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            print(f"      {resp.status_code} from NOAA — retry {attempt}/{MAX_RETRIES} "
                  f"in {wait}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp
    return resp


def _yearly_windows(start: str, end: str):
    """Yield (win_start, win_end) 'YYYY-MM-DD' pairs ≤ 1 calendar year each.

    NOAA CDO v2 rejects any single request whose date span exceeds one year,
    so daily GHCND pulls must be chunked.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        win_end = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        yield s.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d")
        s = win_end + pd.Timedelta(days=1)


def _fetch_window(datatype: str,
                  start: str,
                  end: str,
                  token: str,
                  locationid: str) -> list:
    """Fetch all raw records for one datatype in a single ≤1-year window."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    headers = {"token": token}
    rows = []
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
        resp = _get_with_retry(requests, params, headers)
        body = resp.json()
        results = body.get("results", [])
        if not results:
            break
        rows.extend(results)
        total = body["metadata"]["resultset"]["count"]
        offset += PAGE_LIMIT
        if offset > total:
            break
        time.sleep(RATE_DELAY)   # stay within NOAA rate limits

    return rows


def _fetch_datatype(datatype: str,
                    start: str,
                    end: str,
                    token: str,
                    locationid: str = "FIPS:12") -> pd.DataFrame:
    """Fetch all daily records for one datatype in [start, end].

    The range is split into ≤1-year windows (NOAA CDO v2 limit) and paginated
    within each window.

    Parameters
    ----------
    datatype   : NOAA CDO datatype id, e.g. 'TMAX'
    start, end : 'YYYY-MM-DD'
    token      : NOAA API token
    locationid : 'FIPS:12' = Florida statewide

    Each ≤1-year window is cached to FETCH_CACHE_DIR as soon as it completes, so
    a crash (network reset, throttling) only loses the in-progress window: rerun
    and the function skips every window already on disk and continues where it
    stopped.

    Returns
    -------
    DataFrame with a single column named `datatype`, indexed by date.
    """
    FETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for win_start, win_end in _yearly_windows(start, end):
        win_cache = FETCH_CACHE_DIR / f"{datatype}_{win_start}_{win_end}.csv"
        if win_cache.exists():
            print(f"    {datatype}: {win_start} → {win_end} (cached)")
            frames.append(pd.read_csv(win_cache, index_col=0, parse_dates=True))
            continue

        print(f"    {datatype}: {win_start} → {win_end}")
        rows = _fetch_window(datatype, win_start, win_end, token, locationid)
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            # average across stations on the same day
            daily = df.groupby("date")["value"].mean().rename(datatype).to_frame()
        else:
            daily = pd.DataFrame(columns=[datatype])
            daily.index.name = "date"
        daily.to_csv(win_cache)   # checkpoint this window before moving on
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=[datatype])

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = "date"
    return out


def _resolve_token(token: str) -> str:
    """Return a usable NOAA token or raise with registration instructions."""
    token = token or os.environ.get("NOAA_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "NOAA API token required.\n"
            "  Register at https://www.ncdc.noaa.gov/cdo-web/token\n"
            "  Then: export NOAA_TOKEN=your_token"
        )
    return token


def build_raw_daily(start: str = "2016-01-01",
                    end:   str = "2025-12-31",
                    token: str = DEFAULT_TOKEN,
                    cache: Path = RAW_CACHE_PATH) -> pd.DataFrame:
    """Download raw daily FL weather — no derived columns, no resampling.

    This is the unprocessed observation panel: station-averaged daily TMAX,
    TMIN (°F) and PRCP (mm). No GDD, no weekly aggregation.

    Parameters
    ----------
    start, end : date range
    token      : NOAA API token (falls back to NOAA_TOKEN env var)
    cache      : path to raw cache CSV; skips download if it exists

    Returns
    -------
    DataFrame with columns [TMAX, TMIN, PRCP], daily DatetimeIndex.
    """
    if cache.exists():
        print(f"  Loading raw daily weather from cache: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    token = _resolve_token(token)

    print("  Fetching NOAA TMAX …")
    tmax = _fetch_datatype("TMAX", start, end, token)
    print("  Fetching NOAA TMIN …")
    tmin = _fetch_datatype("TMIN", start, end, token)
    print("  Fetching NOAA PRCP …")
    prcp = _fetch_datatype("PRCP", start, end, token)

    daily = tmax.join(tmin, how="outer").join(prcp, how="outer")
    daily["PRCP"] = daily["PRCP"] * 25.4   # inches (NOAA "standard" units) → mm
    daily.index.name = "Date"

    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache)
    print(f"  Saved raw daily weather → {cache}")
    return daily


def build_weather_panel(start: str = "2016-01-01",
                        end:   str = "2025-12-31",
                        token: str = DEFAULT_TOKEN,
                        cache: Path = CACHE_PATH) -> pd.DataFrame:
    """Download FL weather, derive GDD, resample to weekly, cache result.

    Parameters
    ----------
    start, end : date range matching the Trends panel
    token      : NOAA API token (falls back to NOAA_TOKEN env var)
    cache      : path to cache CSV; skips download if it exists

    Returns
    -------
    DataFrame with the LEVEL + SHOCK columns documented in the module header,
    on a weekly W-SUN index. Anomaly (z-score) columns are NOT included here —
    add them per training fold via add_weather_anomalies() to avoid leakage.
    """
    if cache.exists():
        print(f"  Loading weather from cache: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    daily = build_raw_daily(start, end, token).copy()

    # ── Daily derived quantities ────────────────────────────────────────────
    daily["TAVG"] = (daily["TMAX"] + daily["TMIN"]) / 2
    daily["GDD"]  = (daily["TAVG"] - GDD_BASE_F).clip(lower=0)

    # Daily event flags (NaN-safe: NaN comparisons → False, i.e. not counted)
    daily["freeze_days"] = (daily["TMIN"] <= FREEZE_F).astype(float)
    daily["heat_days"]   = (daily["TMAX"] >= HEAT_F).astype(float)
    daily["rain_days"]   = (daily["PRCP"] >= RAIN_MIN_MM).astype(float)
    daily["chill_days"]  = daily["TAVG"].between(CHILL_LO_F, CHILL_HI_F).astype(float)

    # ── Resample to weekly (Sunday-ending to match Trends panel) ────────────
    weekly = daily.resample("W-SUN").agg({
        "TMAX":        "mean",
        "TMIN":        "mean",
        "TAVG":        "mean",
        "PRCP":        "sum",
        "GDD":         "sum",
        "freeze_days": "sum",
        "heat_days":   "sum",
        "rain_days":   "sum",
        "chill_days":  "sum",
    })
    weekly["PRCP_max"] = daily["PRCP"].resample("W-SUN").max()

    # GDD accumulated within each calendar year (phenology proxy); resets Jan 1
    weekly["GDD_accum"] = weekly.groupby(weekly.index.year)["GDD"].cumsum()

    weekly = weekly[WX_LEVEL_COLS + WX_SHOCK_COLS]
    weekly.index.name = "Date"

    cache.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(cache)
    print(f"  Saved weather panel → {cache}")
    return weekly


def add_weather_anomalies(weather: pd.DataFrame,
                          train_index: pd.DatetimeIndex,
                          cols: list = None) -> pd.DataFrame:
    """Append week-of-year z-score anomaly columns, fit on the TRAIN fold only.

    The seasonal *level* of weather is already encoded by the calendar Fourier
    terms / season flags, so it is largely redundant as a forecaster input. The
    informative, non-redundant signal is the departure from the normal seasonal
    cycle — captured here as a standardized anomaly.

    LEAKAGE RULE (design doc §6/§12): the climatology (per-week-of-year mean and
    std) is computed using ONLY rows whose dates fall in `train_index`, then
    applied to every row. Call this once per rolling-origin fold with that fold's
    training dates.

    Parameters
    ----------
    weather     : weekly weather panel from build_weather_panel()
    train_index : DatetimeIndex of the current fold's training weeks
    cols        : variables to standardize (default WX_ANOM_BASE)

    Returns
    -------
    Copy of `weather` with extra columns ``{col}_anom``.
    """
    if cols is None:
        cols = [c for c in WX_ANOM_BASE if c in weather.columns]

    woy = weather.index.isocalendar().week.astype(int).clip(upper=52)
    train = weather.loc[weather.index.intersection(train_index)]
    train_woy = train.index.isocalendar().week.astype(int).clip(upper=52)

    out = weather.copy()
    for c in cols:
        grp = train[c].groupby(train_woy.values)
        clim_mean = grp.mean()
        clim_std = grp.std(ddof=0).replace(0, np.nan)
        mean_aligned = woy.map(clim_mean)
        std_aligned = woy.map(clim_std)
        out[f"{c}_anom"] = (weather[c] - mean_aligned) / std_aligned
    return out


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
    print("=== RAW DAILY (no derived calculations) ===")
    raw = build_raw_daily()
    pd.set_option("display.max_columns", None, "display.width", 120)
    print(raw.head(10))
    print("  …")
    print(raw.tail(10))
    print(f"\nRaw shape: {raw.shape}  columns: {list(raw.columns)}")

    print("\n=== DERIVED WEEKLY PANEL ===")
    wx = build_weather_panel()
    print(wx.tail())
    print(f"\nShape: {wx.shape}  columns: {list(wx.columns)}")

    # Demo: leakage-safe anomalies using the first 70% of weeks as a 'train' fold
    n_train = int(len(wx) * 0.70)
    wx_anom = add_weather_anomalies(wx, train_index=wx.index[:n_train])
    anom_cols = [c for c in wx_anom.columns if c.endswith("_anom")]
    print("\n=== ANOMALY DEMO (train = first 70% of weeks) ===")
    print(wx_anom[anom_cols].tail())
