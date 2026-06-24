"""wiki_corroboration.py — Gate A.4: independent corroboration of the panel.

The strongest evidence that the stitched panel is *real interest* rather than
a Google-Trends sampling artifact is agreement with a second, independently
measured signal. English-Wikipedia daily pageviews (free Wikimedia REST API,
not subject to Trends renormalization) are the cleanest such signal.

For each crop we fetch weekly pageviews, align to the panel's weekly index,
and report Pearson + Spearman correlation. High correlation ⇒ the stitched
series tracks an independent interest signal.

Run:
    python -m src.data.wiki_corroboration
    # or
    python src/data/wiki_corroboration.py

Uses only the Python standard library for the HTTP call (no `requests`).
Requires internet access; crops that fail to fetch are reported, not fatal.
"""

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
from src.utils.io import read_panel

# Wikimedia requires a descriptive User-Agent; requests without one are blocked.
USER_AGENT = "Agri-Trends-Research/1.0 (academic GNN forecasting study)"
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")

# Panel column (title-cased crop) → English-Wikipedia article title.
# Anything not listed defaults to the crop name with spaces → underscores.
ARTICLE_OVERRIDES = {
    "Maize": "Maize",
    "Bell Pepper": "Bell_pepper",
    "Sweet Potato": "Sweet_potato",
    "Citrus Fruit": "Citrus",
    "Citrus × Tangerina": "Tangerine",
    "Carambola": "Carambola",
    "Chayote": "Chayote",
    "Lime": "Lime_(fruit)",
    "Plum": "Plum",
    "Mint": "Mentha",
    "Guava": "Guava",
    "Papaya": "Papaya",
}


def article_for(crop: str) -> str:
    return ARTICLE_OVERRIDES.get(crop, crop.replace(" ", "_"))


def fetch_weekly_pageviews(article: str,
                           start: pd.Timestamp,
                           end: pd.Timestamp,
                           freq: str = "W-SUN",
                           retries: int = 4) -> pd.Series:
    """Fetch daily pageviews for one article and resample to weekly sums.

    Retries with exponential backoff on HTTP 429 (rate limit), which the
    Wikimedia API returns under bursty access.
    """
    url = API.format(
        article=quote(article, safe=""),
        start=start.strftime("%Y%m%d") + "00",
        end=end.strftime("%Y%m%d") + "00",
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))  # 2s, 4s, 6s backoff
                continue
            raise
    items = payload.get("items", [])
    if not items:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
    views = pd.Series([it["views"] for it in items], index=dates, dtype=float)
    return views.resample(freq).sum()


def corroborate(panel: pd.DataFrame = None) -> pd.DataFrame:
    panel = read_panel() if panel is None else panel
    start, end = panel.index.min(), panel.index.max()
    print("\n" + "═" * 62)
    print("  A4 — Wikipedia pageview corroboration")
    print("═" * 62)
    print(f"  Panel: {panel.shape[1]} crops, {start.date()} → {end.date()}")
    print(f"  Source: Wikimedia REST pageviews API (en.wikipedia, user traffic)\n")

    rows = []
    for crop in panel.columns:
        article = article_for(crop)
        try:
            wk = fetch_weekly_pageviews(article, start, end)
            status = "ok" if not wk.empty else "empty"
        except HTTPError as e:
            wk, status = pd.Series(dtype=float), f"http_{e.code}"
        except (URLError, TimeoutError) as e:
            wk, status = pd.Series(dtype=float), "fetch_failed"
        except Exception as e:  # noqa: BLE001 — never let one crop abort the run
            wk, status = pd.Series(dtype=float), f"error:{type(e).__name__}"

        pearson = spearman = seas_pearson = seas_spearman = np.nan
        n = 0
        if not wk.empty:
            joined = pd.concat(
                [panel[crop].rename("trends"), wk.rename("wiki")],
                axis=1, join="inner",
            ).dropna()
            joined = joined[joined["trends"] > 0]
            n = len(joined)
            if n >= 12:
                pearson = float(joined["trends"].corr(joined["wiki"]))
                spearman = float(
                    joined["trends"].corr(joined["wiki"], method="spearman"))
                # Seasonal-profile correlation (primary metric): compare the
                # week-of-year climatology of both signals. This isolates the
                # shared annual cycle from differing long-term trends and the
                # FL-vs-global geography mismatch.
                woy = joined.index.isocalendar().week.to_numpy()
                prof = joined.groupby(woy).mean()
                if len(prof) >= 12:
                    seas_pearson = float(prof["trends"].corr(prof["wiki"]))
                    seas_spearman = float(
                        prof["trends"].corr(prof["wiki"], method="spearman"))

        rows.append({
            "crop": crop, "article": article, "n_weeks": n,
            "seasonal_pearson": seas_pearson, "seasonal_spearman": seas_spearman,
            "raw_pearson": pearson, "raw_spearman": spearman, "status": status,
        })
        flag = ""
        if not np.isnan(seas_spearman):
            flag = "  ← WEAK" if seas_spearman < 0.5 else ""
        sp = f"{seas_spearman:5.2f}" if not np.isnan(seas_spearman) else "  n/a"
        pe = f"{seas_pearson:5.2f}" if not np.isnan(seas_pearson) else "  n/a"
        print(f"    {crop:<22} seasonal: r {pe}  ρ {sp}  "
              f"[{status}]{flag}")
        time.sleep(0.5)  # be polite to the API (avoid 429 rate limiting)

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "a4_wiki_corroboration.csv", index=False)

    valid = out.dropna(subset=["seasonal_spearman"])
    print("\n  ── summary (primary = seasonal-profile correlation) ──")
    print(f"  Crops corroborated:       {len(valid)} / {len(out)}")
    if len(valid):
        print(f"  Median seasonal ρ:        {valid['seasonal_spearman'].median():.2f}")
        print(f"  Median seasonal r:        {valid['seasonal_pearson'].median():.2f}")
        print(f"  (raw-level median ρ:      {valid['raw_spearman'].median():.2f}  "
              f"— trend/geo-contaminated, secondary)")
        strong = valid[valid["seasonal_spearman"] >= 0.5]
        weak = valid[valid["seasonal_spearman"] < 0.5]
        print(f"  Strong seasonal match:    {len(strong)} crops (ρ ≥ 0.5)")
        if len(weak):
            print(f"  Weak (ρ<0.5), inspect:    {list(weak['crop'])}")
    failed = out[~out["status"].isin(["ok"])]
    if len(failed):
        print(f"  Not fetched/empty:        {list(failed['crop'])}")
    print(f"  → wrote a4_wiki_corroboration.csv")
    return out


if __name__ == "__main__":
    corroborate()
