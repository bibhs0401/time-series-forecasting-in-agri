"""validate_stitch.py — Gate A calibration validation for the stitched panel.

These checks prove the *stitching itself* is correct (the data layer), as
opposed to ``validate_panel.py`` which only checks the final panel's shape.
Run this before locking the target and building any models.

Implements:
  A1  Held-out overlap reconstruction error (the decisive seam check)
  A2  Cross-group anchor leave-one-out (how fragile is the g1 calibration?)
  A3  Estimator sensitivity (ratio-of-means vs median-of-ratios)
  A5  Noise-floor proxy from overlap disagreement
  (A4 — Wikipedia corroboration — lives in wiki_corroboration.py)

Run:
    python -m src.data.validate_stitch
    # or
    python src/data/validate_stitch.py

All tables are written to ``outputs/tables/`` and (if matplotlib is present)
a seam-error figure to ``outputs/figures/``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
from src.data.stitch_panel import (
    build_panel,
    cross_scale_groups,
    per_anchor_factors,
    stitch_all_groups,
)


def _hr(title: str) -> None:
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


# ═══════════════════════════════════════════════════════════════════
#  A0 — Seam integrity (empty windows / un-scaled joins)
# ═══════════════════════════════════════════════════════════════════

def a0_integrity(anomalies: list) -> pd.DataFrame:
    """Report empty windows and joins that were appended WITHOUT scaling.

    An empty window (header-only CSV) or a low-overlap join silently breaks
    the within-group calibration: every window after it sits on an unscaled
    baseline. These must be fixed (re-download the window) before the panel
    is trusted, so this is a hard gate, not just a metric.
    """
    _hr("A0 — Seam integrity (empty / un-scaled windows)")
    df = pd.DataFrame(anomalies)
    if df.empty:
        print("  ✓ No empty windows or un-scaled joins. All seams calibrated.")
        return df
    df.to_csv(config.TABLES_DIR / "a0_seam_integrity.csv", index=False)
    print(f"  ✗ {len(df)} integrity problem(s) found — FIX BEFORE MODELING:")
    for _, r in df.iterrows():
        if r["type"] == "empty_window":
            print(f"      {r['group']} {r['seam']}: EMPTY window file "
                  f"(re-download this pull)")
        else:
            print(f"      {r['group']} {r['seam']}: appended UNSCALED "
                  f"(only {r['n_overlap']} overlap wk) — calibration break")
    print(f"  → wrote a0_seam_integrity.csv")
    return df


# ═══════════════════════════════════════════════════════════════════
#  A1 — Held-out overlap reconstruction error
# ═══════════════════════════════════════════════════════════════════

def a1_overlap_error(seam_records: list) -> pd.DataFrame:
    """Summarise per-seam overlap reconstruction error.

    ``seam_records`` is the per-crop residual list captured during Phase 1.
    A correct stitch has small, roughly unbiased residuals on the weeks the
    two adjacent windows share. Large median error or a consistent sign in
    ``mean_signed_rel`` means a single multiplicative scale is the wrong
    model for that seam.
    """
    _hr("A1 — Held-out overlap reconstruction error (Phase 1 seams)")
    detail = pd.DataFrame(seam_records)
    if detail.empty:
        print("  ⚠ no seam records captured — cannot evaluate.")
        return detail

    detail.to_csv(config.TABLES_DIR / "a1_seam_residuals.csv", index=False)

    by_seam = (
        detail.groupby(["group", "seam"])
        .agg(n_crops=("crop", "size"),
             n_overlap=("n_overlap", "max"),
             factor=("factor", "first"),
             median_abs_rel=("median_abs_rel", "median"),
             p90_abs_rel=("p90_abs_rel", "median"),
             mean_signed_rel=("mean_signed_rel", "mean"))
        .reset_index()
    )
    by_seam.to_csv(config.TABLES_DIR / "a1_seam_summary.csv", index=False)

    overall_med = detail["median_abs_rel"].median()
    overall_p90 = detail["median_abs_rel"].quantile(0.90)
    bias = detail["mean_signed_rel"].mean()
    flagged = by_seam[by_seam["median_abs_rel"] > config.SEAM_REL_GATE]

    print(f"  Seams evaluated:        {len(by_seam)}  "
          f"across {by_seam['group'].nunique()} groups")
    print(f"  Median overlap error:   {overall_med:6.2%}  "
          f"(per-crop-seam median)")
    print(f"  90th-pct seam error:    {overall_p90:6.2%}")
    print(f"  Mean signed bias:       {bias:+.2%}   (≈0 ⇒ no systematic drift)")
    print(f"  Seams over {config.SEAM_REL_GATE:.0%} gate:    "
          f"{len(flagged)}")
    if len(flagged):
        for _, r in flagged.sort_values("median_abs_rel", ascending=False).iterrows():
            print(f"      {r['group']:>4} {r['seam']}  "
                  f"median {r['median_abs_rel']:.1%}  "
                  f"({int(r['n_overlap'])} wks)")
    print(f"  → wrote a1_seam_residuals.csv, a1_seam_summary.csv")
    _maybe_plot_a1(by_seam)
    return by_seam


def _maybe_plot_a1(by_seam: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for g, sub in by_seam.groupby("group"):
        order = sub["seam"].str.extract(r"(\d+)").astype(int)[0]
        sub = sub.assign(_o=order.values).sort_values("_o")
        ax.plot(sub["_o"], sub["median_abs_rel"] * 100, marker="o",
                ms=3, lw=0.8, alpha=0.7, label=g)
    ax.axhline(config.SEAM_REL_GATE * 100, color="red", ls="--", lw=1,
               label=f"{config.SEAM_REL_GATE:.0%} gate")
    ax.set_xlabel("seam (window index)")
    ax.set_ylabel("median overlap error (%)")
    ax.set_title("A1 — Phase-1 seam reconstruction error")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "a1_seam_error.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════
#  A2 — Cross-group anchor leave-one-out
# ═══════════════════════════════════════════════════════════════════

def a2_anchor_loo(stitched: dict) -> pd.DataFrame:
    """How much does each group's cross-scale factor depend on the anchor?

    For every non-reference group we compute a ratio-of-means factor from
    *every* crop it shares with the reference group, not just the configured
    anchor. If the factors disagree, the g1 calibration is fragile and the
    configured single-anchor scaling should be reconsidered.
    """
    _hr("A2 — Cross-group anchor leave-one-out (vs reference group)")
    ref = stitched[config.REF_GROUP]
    rows, summary = [], []

    for g in config.GROUPS:
        if g == config.REF_GROUP:
            continue
        candidates = [c for c in stitched[g].columns if c in ref.columns]
        factors = per_anchor_factors(ref, stitched[g], candidates)
        configured = config.GROUP_ANCHORS.get(g, [])
        used = [factors[a][0] for a in configured if a in factors]
        used_factor = float(np.mean(used)) if used else np.nan

        for anchor, (f, n) in factors.items():
            rows.append({
                "group": g, "anchor": anchor, "factor": f, "n_points": n,
                "is_configured": anchor in configured,
            })

        fvals = [f for f, _ in factors.values()]
        ratio = (max(fvals) / min(fvals)) if fvals else np.nan
        n_anchors = len(fvals)
        summary.append({
            "group": g,
            "n_candidate_anchors": n_anchors,
            "anchors": ",".join(factors.keys()),
            "factor_used": used_factor,
            "factor_min": min(fvals) if fvals else np.nan,
            "factor_max": max(fvals) if fvals else np.nan,
            "max_min_ratio": ratio,
            "loo_possible": n_anchors >= 2,
        })
        tag = "" if n_anchors >= 2 else "  ← SINGLE ANCHOR (no LOO, fragile)"
        rng = f"{min(fvals):.3f}–{max(fvals):.3f}" if fvals else "n/a"
        print(f"  {g:>4}: {n_anchors} anchor(s) [{', '.join(factors.keys())}]  "
              f"factor {rng}  (ratio {ratio:.3f}){tag}")

    detail = pd.DataFrame(rows)
    summ = pd.DataFrame(summary)
    detail.to_csv(config.TABLES_DIR / "a2_anchor_factors.csv", index=False)
    summ.to_csv(config.TABLES_DIR / "a2_anchor_summary.csv", index=False)

    multi = summ[summ["loo_possible"]]
    n_single = int((~summ["loo_possible"]).sum())
    print(f"\n  Groups with >1 usable anchor: {len(multi)} / {len(summ)}")
    if len(multi):
        worst = multi.sort_values("max_min_ratio", ascending=False).iloc[0]
        print(f"  Largest anchor disagreement:  {worst['group']} "
              f"(max/min ratio {worst['max_min_ratio']:.3f})")
    print(f"  Groups anchored on cucumber ALONE: {n_single}  "
          f"(only direct g1 bridge is tomato in g3)")
    print(f"  → wrote a2_anchor_factors.csv, a2_anchor_summary.csv")
    return summ


# ═══════════════════════════════════════════════════════════════════
#  A3 — Estimator sensitivity
# ═══════════════════════════════════════════════════════════════════

def a3_estimator_sensitivity() -> pd.DataFrame:
    """Does the panel depend on the scale-factor estimator?

    Rebuilds the whole panel with median-of-ratios and compares it cell-by
    -cell to the ratio-of-means panel. Small deviation ⇒ the choice is
    immaterial and the result is robust (state this in the paper).
    """
    _hr("A3 — Estimator sensitivity (ratio-of-means vs median-of-ratios)")
    base = build_panel(estimator="ratio_of_means",
                       renormalize=False, verbose=False)
    alt = build_panel(estimator="median_of_ratios",
                      renormalize=False, verbose=False)

    cols = base.columns.intersection(alt.columns)
    idx = base.index.intersection(alt.index)
    b = base.loc[idx, cols].to_numpy(dtype=float)
    a = alt.loc[idx, cols].to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (b > 0)
    rel = np.abs(a[m] - b[m]) / b[m]

    bf = base.loc[idx, cols].to_numpy(dtype=float).ravel()
    af = alt.loc[idx, cols].to_numpy(dtype=float).ravel()
    mm = np.isfinite(af) & np.isfinite(bf)
    corr = float(np.corrcoef(bf[mm], af[mm])[0, 1])

    per_crop = []
    for c in cols:
        cb = base.loc[idx, c].to_numpy(dtype=float)
        ca = alt.loc[idx, c].to_numpy(dtype=float)
        mc = np.isfinite(ca) & np.isfinite(cb) & (cb > 0)
        if mc.sum() == 0:
            continue
        per_crop.append({
            "crop": c,
            "median_abs_rel_dev": float(np.median(np.abs(ca[mc] - cb[mc]) / cb[mc])),
            "max_abs_rel_dev": float(np.max(np.abs(ca[mc] - cb[mc]) / cb[mc])),
        })
    per_crop_df = pd.DataFrame(per_crop).sort_values(
        "median_abs_rel_dev", ascending=False)
    per_crop_df.to_csv(config.TABLES_DIR / "a3_estimator_sensitivity.csv",
                       index=False)

    print(f"  Median cell deviation:  {np.median(rel):6.2%}")
    print(f"  90th-pct deviation:     {np.quantile(rel, 0.90):6.2%}")
    print(f"  Max cell deviation:     {np.max(rel):6.2%}")
    print(f"  Panel correlation:      {corr:.5f}")
    if len(per_crop_df):
        w = per_crop_df.iloc[0]
        print(f"  Most-affected crop:     {w['crop']} "
              f"(median dev {w['median_abs_rel_dev']:.1%})")
    print(f"  → wrote a3_estimator_sensitivity.csv")
    return per_crop_df


# ═══════════════════════════════════════════════════════════════════
#  A5 — Noise-floor proxy from overlap disagreement
# ═══════════════════════════════════════════════════════════════════

def a5_noise_floor(seam_records: list) -> pd.DataFrame:
    """Per-crop lower-bound noise floor from adjacent-window disagreement.

    With one pull per window there are no repeat-pull replicates, so the
    disagreement between adjacent windows on their shared weeks is used as a
    lower bound on Trends sampling jitter. No forecast may honestly claim
    error below this floor.
    """
    _hr("A5 — Noise-floor proxy (adjacent-window overlap disagreement)")
    detail = pd.DataFrame(seam_records)
    if detail.empty:
        print("  ⚠ no seam records — cannot estimate floor.")
        return detail

    floor = (
        detail.groupby("crop")
        .agg(n_seams=("seam", "nunique"),
             noise_floor_med_abs_rel=("median_abs_rel", "median"),
             noise_floor_p90_abs_rel=("median_abs_rel", lambda s: s.quantile(0.90)))
        .reset_index()
        .sort_values("noise_floor_med_abs_rel", ascending=False)
    )
    floor.to_csv(config.TABLES_DIR / "a5_noise_floor.csv", index=False)

    print(f"  Per-crop floor (median abs overlap disagreement):")
    print(f"    panel-wide median:  {floor['noise_floor_med_abs_rel'].median():.2%}")
    print(f"    noisiest crops:")
    for _, r in floor.head(6).iterrows():
        print(f"      {r['crop']:<22} {r['noise_floor_med_abs_rel']:6.2%}")
    print(f"\n  NOTE: this is a *lower bound* (one pull/window, no replicates).")
    print(f"        For a true floor, collect 3× repeat pulls on a few cells.")
    print(f"  → wrote a5_noise_floor.csv")
    return floor


# ═══════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 62)
    print("#  GATE A — Stitching / calibration validation")
    print("#" * 62)

    # Phase 1 once, capturing seam residuals + anomalies (drives A0/A1/A5).
    _hr("Running Phase 1 (capturing seam residuals)")
    seam_records: list = []
    anomalies: list = []
    stitched = stitch_all_groups(
        seam_records=seam_records, anomalies=anomalies, verbose=False)
    print(f"  Stitched {len(stitched)} groups, "
          f"{len(seam_records)} per-crop seam residuals captured.")

    a0_integrity(anomalies)
    a1_overlap_error(seam_records)
    a2_anchor_loo(stitched)
    a3_estimator_sensitivity()
    a5_noise_floor(seam_records)

    _hr("Gate A complete")
    print(f"  Tables → {config.TABLES_DIR}")
    print(f"  Figures → {config.FIGURES_DIR}")
    print(f"  (A4 Wikipedia corroboration: run "
          f"`python -m src.data.wiki_corroboration`)")


if __name__ == "__main__":
    main()
