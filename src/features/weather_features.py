'''Build weekly weather features for Florida from the NOAA API.

Downloads daily NOAA weather for Florida, averages readings across stations,
and rolls them up into a weekly feature table.

Daily variables:
  TMAX  - daily maximum temperature (F)
  TMIN  - daily minimum temperature (F)
  PRCP  - daily precipitation (NOAA standard units are inches; converted to mm)

Weekly output columns:
  TMAX, TMIN  - weekly average temperature (F)
  TAVG        - weekly average of (TMAX + TMIN) / 2
  PRCP        - total weekly precipitation (mm)
  PRCP_max    - wettest day of the week (mm)
  GDD         - weekly growing degree days, base 50F
  GDD_accum   - running yearly GDD total, reset each January
  freeze_days - days with TMIN <= 32F
  heat_days   - days with TMAX >= 90F
  rain_days   - days with PRCP >= 1 mm
  chill_days  - days with 32F <= TAVG <= 45F

Anomaly columns are added separately with add_weather_anomalies().
Fit those on the training period only so they do not leak future data.
'''

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

CACHE_PATH       = config.PANEL_DIR / 'noaa_weather_fl.csv'
RAW_CACHE_PATH   = config.PANEL_DIR / 'noaa_weather_fl_daily_raw.csv'
FETCH_CACHE_DIR  = config.PANEL_DIR / '_wx_fetch_cache'   # one file per datatype/year window
NOAA_BASE  = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
PAGE_LIMIT = 1000   # max records per NOAA API call
DEFAULT_TOKEN = 'PMcVuObDvgpZKlSHGJZrTSUMSzYMmlZc'
MAX_RETRIES = 6     # retry temporary server or rate-limit errors
RATE_DELAY  = 0.35  # seconds between successful page requests

# Thresholds used to turn daily weather into event counts
GDD_BASE_F   = 50.0   # growing degree day base temperature
FREEZE_F     = 32.0   # TMIN at/below this = freeze day
HEAT_F       = 90.0   # TMAX at/above this = heat-stress day
CHILL_LO_F   = 32.0   # chill window lower bound (TAVG)
CHILL_HI_F   = 45.0   # chill window upper bound (TAVG)
RAIN_MIN_MM  = 1.0    # PRCP at/above this = measurable rain day

# Feature groups used later in the pipeline
WX_LEVEL_COLS = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'PRCP_max', 'GDD', 'GDD_accum']
WX_SHOCK_COLS = ['freeze_days', 'heat_days', 'rain_days', 'chill_days']
WX_ANOM_BASE  = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'GDD']   # vars to z-score


def _get_with_retry(requests, params, headers):
    '''Fetch one NOAA API page; retry if the failure looks temporary.'''
    for attempt in range(1, MAX_RETRIES + 1):
        wait = min(2 ** attempt, 30)   # simple exponential backoff
        try:
            resp = requests.get(NOAA_BASE, headers=headers, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f'      network error ({type(exc).__name__}): '
                  f'retry {attempt}/{MAX_RETRIES} in {wait}s')
            time.sleep(wait)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            print(f'      {resp.status_code} from NOAA: retry {attempt}/{MAX_RETRIES} '
                  f'in {wait}s')
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp
    return resp


def _yearly_windows(start: str, end: str):
    '''Split a date range into chunks no longer than one calendar year.'''
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        win_end = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        yield s.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')
        s = win_end + pd.Timedelta(days=1)


def _fetch_window(datatype: str,
                  start: str,
                  end: str,
                  token: str,
                  locationid: str) -> list:
    '''Fetch all raw NOAA rows for one datatype in one time window.'''
    try:
        import requests
    except ImportError:
        raise ImportError('pip install requests')

    headers = {'token': token}
    rows = []
    offset = 1

    while True:
        params = dict(
            datasetid='GHCND',
            locationid=locationid,
            datatypeid=datatype,
            startdate=start,
            enddate=end,
            units='standard',
            limit=PAGE_LIMIT,
            offset=offset,
        )
        resp = _get_with_retry(requests, params, headers)
        body = resp.json()
        results = body.get('results', [])
        if not results:
            break
        rows.extend(results)
        total = body['metadata']['resultset']['count']
        offset += PAGE_LIMIT
        if offset > total:
            break
        time.sleep(RATE_DELAY)   # avoid hitting NOAA too fast

    return rows


def _fetch_datatype(datatype: str,
                    start: str,
                    end: str,
                    token: str,
                    locationid: str = 'FIPS:12') -> pd.DataFrame:
    '''Fetch one daily NOAA variable for the given date range.

    NOAA caps requests at one year, so we download in yearly chunks and
    paginate within each chunk.

    Finished chunks are saved under FETCH_CACHE_DIR. If a run stops halfway,
    the next run reuses cached chunks and continues from the missing one.
    '''
    FETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for win_start, win_end in _yearly_windows(start, end):
        win_cache = FETCH_CACHE_DIR / f'{datatype}_{win_start}_{win_end}.csv'
        if win_cache.exists():
            print(f'    {datatype}: {win_start} -> {win_end} (cached)')
            frames.append(pd.read_csv(win_cache, index_col=0, parse_dates=True))
            continue

        print(f'    {datatype}: {win_start} -> {win_end}')
        rows = _fetch_window(datatype, win_start, win_end, token, locationid)
        if rows:
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            # Average all Florida station readings for each day.
            daily = df.groupby('date')['value'].mean().rename(datatype).to_frame()
        else:
            daily = pd.DataFrame(columns=[datatype])
            daily.index.name = 'date'
        daily.to_csv(win_cache)   # save progress before the next chunk
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=[datatype])

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep='first')].sort_index()
    out.index.name = 'date'
    return out


def _resolve_token(token: str) -> str:
    '''Return a NOAA token, or raise if none is available.'''
    token = token or os.environ.get('NOAA_TOKEN', '')
    if not token:
        raise EnvironmentError(
            'NOAA API token required.\n'
            '  Register at https://www.ncdc.noaa.gov/cdo-web/token\n'
            '  Then: export NOAA_TOKEN=your_token'
        )
    return token


def build_raw_daily(start: str = '2016-01-01',
                    end:   str = '2025-12-31',
                    token: str = DEFAULT_TOKEN,
                    cache: Path = RAW_CACHE_PATH) -> pd.DataFrame:
    '''Build the raw daily Florida weather table.

    Returns station-averaged daily values for TMAX, TMIN, and PRCP.
    Does not add derived features or resample to weeks.
    '''
    if cache.exists():
        print(f'  Loading raw daily weather from cache: {cache}')
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    token = _resolve_token(token)

    print('  Fetching NOAA TMAX ...')
    tmax = _fetch_datatype('TMAX', start, end, token)
    print('  Fetching NOAA TMIN ...')
    tmin = _fetch_datatype('TMIN', start, end, token)
    print('  Fetching NOAA PRCP ...')
    prcp = _fetch_datatype('PRCP', start, end, token)

    daily = tmax.join(tmin, how='outer').join(prcp, how='outer')
    daily['PRCP'] = daily['PRCP'] * 25.4   # inches (NOAA standard units) to mm
    daily.index.name = 'Date'

    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache)
    print(f'  Saved raw daily weather -> {cache}')
    return daily


def build_weather_panel(start: str = '2016-01-01',
                        end:   str = '2025-12-31',
                        token: str = DEFAULT_TOKEN,
                        cache: Path = CACHE_PATH) -> pd.DataFrame:
    '''Build the weekly Florida weather feature panel.

    Starts from the daily table, adds a few daily derived features, rolls
    everything up to weekly values, and saves the result. Anomaly columns
    are not added here; call add_weather_anomalies() later so they can be
    fit only on the training period.
    '''
    if cache.exists():
        print(f'  Loading weather from cache: {cache}')
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    daily = build_raw_daily(start, end, token).copy()

    # Daily values derived from the raw weather columns.
    daily['TAVG'] = (daily['TMAX'] + daily['TMIN']) / 2
    daily['GDD']  = (daily['TAVG'] - GDD_BASE_F).clip(lower=0)

    # Count simple daily events. Missing values are not counted.
    daily['freeze_days'] = (daily['TMIN'] <= FREEZE_F).astype(float)
    daily['heat_days']   = (daily['TMAX'] >= HEAT_F).astype(float)
    daily['rain_days']   = (daily['PRCP'] >= RAIN_MIN_MM).astype(float)
    daily['chill_days']  = daily['TAVG'].between(CHILL_LO_F, CHILL_HI_F).astype(float)

    # Roll daily weather up to Sunday-ending weeks.
    weekly = daily.resample('W-SUN').agg({
        'TMAX':        'mean',
        'TMIN':        'mean',
        'TAVG':        'mean',
        'PRCP':        'sum',
        'GDD':         'sum',
        'freeze_days': 'sum',
        'heat_days':   'sum',
        'rain_days':   'sum',
        'chill_days':  'sum',
    })
    weekly['PRCP_max'] = daily['PRCP'].resample('W-SUN').max()

    # Running yearly GDD total. Restart each January.
    weekly['GDD_accum'] = weekly.groupby(weekly.index.year)['GDD'].cumsum()

    weekly = weekly[WX_LEVEL_COLS + WX_SHOCK_COLS]
    weekly.index.name = 'Date'

    cache.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(cache)
    print(f'  Saved weather panel -> {cache}')
    return weekly


def add_weather_anomalies(weather: pd.DataFrame,
                          train_index: pd.DatetimeIndex,
                          cols: list = None) -> pd.DataFrame:
    '''Add week-of-year anomaly columns using only the training period.

    For each selected weather column, measure how unusual a week is relative
    to the typical value for that week of year.

    Fit the baseline only on train_index so future information stays out
    of the features.
    '''
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
        out[f'{c}_anom'] = (weather[c] - mean_aligned) / std_aligned
    return out


def align_to_panel(weather: pd.DataFrame,
                   panel_index: pd.DatetimeIndex) -> pd.DataFrame:
    '''Match weekly weather to the target panel dates.

    Small gaps are forward-filled for up to two weeks. Larger gaps stay
    missing so they can be handled downstream.
    '''
    aligned = weather.reindex(panel_index)
    n_missing = aligned.isna().any(axis=1).sum()
    if n_missing:
        print(f'  Weather: {n_missing} panel weeks missing; forward-filling (max 2)')
    return aligned.ffill(limit=2)


if __name__ == '__main__':
    print('=== RAW DAILY (no derived calculations) ===')
    raw = build_raw_daily()
    pd.set_option('display.max_columns', None, 'display.width', 120)
    print(raw.head(10))
    print('  ...')
    print(raw.tail(10))
    print(f'\nRaw shape: {raw.shape}  columns: {list(raw.columns)}')

    print('\n=== DERIVED WEEKLY PANEL ===')
    wx = build_weather_panel()
    print(wx.tail())
    print(f'\nShape: {wx.shape}  columns: {list(wx.columns)}')

    # Example: fit anomalies on the first 70% of weeks only.
    n_train = int(len(wx) * 0.70)
    wx_anom = add_weather_anomalies(wx, train_index=wx.index[:n_train])
    anom_cols = [c for c in wx_anom.columns if c.endswith('_anom')]
    print('\n=== ANOMALY DEMO (train = first 70% of weeks) ===')
    print(wx_anom[anom_cols].tail())
