#!/usr/bin/env python3
"""
stitch_panel.py
===============
Produce one common-scale weekly panel (2016-2025) for all Florida crops
from the manually-downloaded Google Trends CSVs in data/g1 … g12.

STITCHING LOGIC (three phases)
───────────────────────────────
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — Within-group stitching                                       ║
║                                                                          ║
║  Each group has 19 half-year windows that overlap by ~26-28 weeks.       ║
║  Google Trends rescales each pull to 0-100 independently, so p02 is on   ║
║  a different absolute scale than p01 even though they share 6 months.    ║
║                                                                          ║
║  Fix: ratio-of-means over the shared weeks.                              ║
║                                                                          ║
║  p01:  Jan 2016 ──────────────────── Dec 2016                            ║
║  p02:              Jul 2016 ──────────────────── Jun 2017                ║
║                    ↑────────────────↑                                    ║
║                     overlap (~26 wk)                                     ║
║                                                                          ║
║  For each join p_(i) → p_(i+1):                                          ║
║    1. Find the overlap date range.                                        ║
║    2. Compute scale factor = mean(running_panel[overlap]) /              ║
║                              mean(p_(i+1)[overlap])                      ║
║       pooled across ALL crops in the window for robustness.              ║
║    3. Multiply all of p_(i+1) by that factor.                            ║
║    4. Append only the new (non-overlapping) tail of p_(i+1).             ║
║       The earlier window always wins on shared dates.                    ║
║                                                                          ║
║  Result after Phase 1: one ~520-week series per crop per group.          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PHASE 2 — Cross-group rescaling                                         ║
║                                                                          ║
║  The groups were pulled in separate Trends queries, so they live on      ║
║  different absolute scales even after within-group stitching.            ║
║                                                                          ║
║  Cucumber is the primary anchor (present in every group). For each        ║
║  non-reference group, compute one scale factor per anchor crop, then      ║
║  average those factors and multiply every column in the group by it.      ║
║  All groups are then on the reference (g1) absolute scale.               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PHASE 3 — Deduplication and final assembly                              ║
║                                                                          ║
║  Bridge crops appear in more than one group (cucumber, tomato, lime,     ║
║  cabbage, peach). After cross-scaling they should agree numerically, so  ║
║  we keep each crop from the first group it appears in and drop later      ║
║  copies. The panel is renormalized to [0, 100], then written to CSV.     ║
╚══════════════════════════════════════════════════════════════════════════╝

Run:
    python -m src.data.stitch_panel
    # or
    python src/data/stitch_panel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Box-drawing characters in the console output need UTF-8 (Windows defaults to
# cp1252). Reconfigure stdout where the runtime supports it.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
from src.utils.io import load_group_windows, write_panel

# Scale-factor estimators supported by ``_scale_factor`` (used by the
# estimator-sensitivity check in validate_stitch.py / Gate A.3).
ESTIMATORS = ("ratio_of_means", "median_of_ratios")


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: within-group stitching
# ═══════════════════════════════════════════════════════════════════

def _scale_factor(panel_overlap: np.ndarray,
                  incoming_overlap: np.ndarray,
                  estimator: str = "ratio_of_means") -> float:
    """Multiplicative scale factor aligning ``incoming`` onto ``panel``.

    Pooled across all crops × weeks of the overlap so one noisy crop can't
    skew the rescaling. Two estimators are supported so the calibration can
    be sensitivity-tested (Gate A):

    - ``ratio_of_means``  : sum(panel) / sum(incoming)  (the default; gives
      high-volume weeks more leverage, matches the published behaviour).
    - ``median_of_ratios``: median(panel / incoming)    (robust to outliers;
      every cell counts equally regardless of volume).
    """
    mask = (
        np.isfinite(panel_overlap) & np.isfinite(incoming_overlap)
        & (panel_overlap > 0) & (incoming_overlap > 0)
    )
    if mask.sum() < 8:
        raise ValueError(
            f"Only {mask.sum()} valid overlap points — cannot estimate scale."
        )
    p = panel_overlap[mask]
    q = incoming_overlap[mask]
    if estimator == "ratio_of_means":
        return float(p.sum() / q.sum())
    if estimator == "median_of_ratios":
        return float(np.median(p / q))
    raise ValueError(f"Unknown estimator '{estimator}'. Use one of {ESTIMATORS}.")


# Backwards-compatible alias for the original pooled ratio-of-means helper.
def _pooled_scale_factor(panel_overlap: np.ndarray,
                         incoming_overlap: np.ndarray) -> float:
    return _scale_factor(panel_overlap, incoming_overlap, "ratio_of_means")


def _seam_residuals(panel_overlap: pd.DataFrame,
                    scaled_incoming_overlap: pd.DataFrame,
                    shared_cols: list,
                    group: str,
                    seam: str,
                    factor: float,
                    n_overlap: int) -> list:
    """Per-crop reconstruction error on a single within-group seam.

    After scaling, the two adjacent windows should agree on their shared
    weeks. For each crop we report the relative disagreement
    ``(scaled_incoming - panel) / panel`` on the overlap. This is the
    held-out validation set for Phase-1 calibration (Gate A.1) and the
    lower-bound noise-floor proxy (Gate A.5).
    """
    records = []
    for crop in shared_cols:
        p = panel_overlap[crop].to_numpy(dtype=float)
        q = scaled_incoming_overlap[crop].to_numpy(dtype=float)
        m = np.isfinite(p) & np.isfinite(q) & (p > 0)
        if m.sum() == 0:
            continue
        rel = (q[m] - p[m]) / p[m]
        abs_rel = np.abs(rel)
        records.append({
            "group": group,
            "seam": seam,
            "crop": crop,
            "n_overlap": int(m.sum()),
            "factor": factor,
            "median_abs_rel": float(np.median(abs_rel)),
            "p90_abs_rel": float(np.quantile(abs_rel, 0.90)),
            "mean_signed_rel": float(np.mean(rel)),
        })
    return records


def stitch_windows(windows: list,
                   estimator: str = "ratio_of_means",
                   group: str = "",
                   seam_records: list = None,
                   anomalies: list = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Chain p01 → p02 → … → p19 into one long series per crop.

    At each join:
      1. Identify shared dates (the ~26-week overlap).
      2. Compute a pooled scale factor across ALL crops.
      3. Scale the entire incoming window by that factor.
      4. Append only the new tail (earlier window wins on shared dates).

    If ``seam_records`` is a list, per-crop overlap-reconstruction errors
    are appended to it (used by Gate A validation). If ``anomalies`` is a
    list, empty windows and un-scaled (low-overlap) joins are recorded there
    — these are silent calibration breaks that Gate A must surface.
    """
    result = windows[0].copy()

    for idx, incoming in enumerate(windows[1:], start=2):
        seam = f"p{idx:02d}"
        overlap_idx = result.index.intersection(incoming.index)
        shared_cols = [c for c in incoming.columns if c in result.columns]

        if incoming.shape[0] == 0:
            if anomalies is not None:
                anomalies.append({"group": group, "seam": seam,
                                  "type": "empty_window", "n_overlap": 0})
            if verbose:
                print(f"      {seam}: ⚠ empty window — skipped")
            continue

        if len(overlap_idx) < 4 or not shared_cols:
            # Edge case: no usable overlap — appended WITHOUT scaling.
            new_dates = incoming.index.difference(result.index)
            result = pd.concat([result, incoming.loc[new_dates]]).sort_index()
            if anomalies is not None:
                anomalies.append({"group": group, "seam": seam,
                                  "type": "appended_raw_no_scale",
                                  "n_overlap": int(len(overlap_idx))})
            if verbose:
                print(f"      {seam}: ⚠ no overlap — appended raw (UNSCALED)")
            continue

        panel_overlap = result.loc[overlap_idx, shared_cols]
        incoming_overlap = incoming.loc[overlap_idx, shared_cols]

        factor = _scale_factor(
            panel_overlap.to_numpy(dtype=float),
            incoming_overlap.to_numpy(dtype=float),
            estimator,
        )

        scaled = incoming * factor

        if seam_records is not None:
            seam_records.extend(
                _seam_residuals(
                    panel_overlap,
                    scaled.loc[overlap_idx, shared_cols],
                    shared_cols, group, seam, factor, len(overlap_idx),
                )
            )

        new_dates = scaled.index.difference(result.index)
        result = pd.concat([result, scaled.loc[new_dates]]).sort_index()

        if verbose:
            print(f"      {seam}: scale ×{factor:.4f}  "
                  f"({len(overlap_idx)} overlap wks)")

    return result.sort_index()


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: cross-group rescaling
# ═══════════════════════════════════════════════════════════════════

def per_anchor_factors(ref_panel: pd.DataFrame,
                       other_panel: pd.DataFrame,
                       anchors: list) -> dict:
    """Ratio-of-means factor for each usable anchor crop.

    Returns ``{anchor: (factor, n_points)}`` for every anchor present in
    both panels with at least 4 valid shared weeks. Exposed so Gate A.2
    can test how much the cross-group level depends on the anchor choice.
    """
    shared_dates = ref_panel.index.intersection(other_panel.index)
    out = {}
    for anchor in anchors:
        if anchor not in ref_panel.columns or anchor not in other_panel.columns:
            continue
        r = ref_panel.loc[shared_dates, anchor]
        o = other_panel.loc[shared_dates, anchor]
        mask = r.notna() & o.notna() & (r > 0) & (o > 0)
        if mask.sum() < 4:
            continue
        out[anchor] = (float(r[mask].mean() / o[mask].mean()), int(mask.sum()))
    return out


def anchor_scale_factor(ref_panel: pd.DataFrame,
                        other_panel: pd.DataFrame,
                        anchors: list,
                        verbose: bool = True) -> float:
    """Mean of per-anchor ratio-of-means factors.

    Averaging one factor per anchor (rather than pooling all anchor values
    into a single ratio) prevents a high-volume crop (e.g. tomato) from
    swamping a low-volume one.
    """
    factors = per_anchor_factors(ref_panel, other_panel, anchors)
    if not factors:
        raise ValueError("No valid anchor crops — cannot cross-scale.")
    if verbose:
        for anchor, (f, n) in factors.items():
            print(f"    anchor '{anchor}':  ×{f:.4f}  ({n} pts)")
    mean_f = float(np.mean([f for f, _ in factors.values()]))
    if verbose:
        print(f"    → averaged factor:  ×{mean_f:.4f}")
    return mean_f


# ═══════════════════════════════════════════════════════════════════
#  Reusable pipeline (phase 1 → 2 → 3)
# ═══════════════════════════════════════════════════════════════════

def stitch_all_groups(estimator: str = "ratio_of_means",
                      seam_records: list = None,
                      anomalies: list = None,
                      verbose: bool = True) -> dict:
    """Phase 1 for every group → ``{group: weekly DataFrame}``."""
    stitched = {}
    for g in config.GROUPS:
        if verbose:
            print(f"\n  [{g}]")
        windows = load_group_windows(g)
        stitched[g] = stitch_windows(
            windows, estimator=estimator, group=g,
            seam_records=seam_records, anomalies=anomalies, verbose=verbose,
        )
        if verbose:
            panel = stitched[g]
            print(f"    result: {panel.shape[0]} weeks, "
                  f"{panel.index.min().date()} → {panel.index.max().date()}")
    return stitched


def cross_scale_groups(stitched: dict,
                       anchors_map: dict = None,
                       verbose: bool = True) -> dict:
    """Phase 2: rescale every non-reference group onto the reference scale.

    Returns a new ``{group: DataFrame}`` dict with each group multiplied by
    its averaged anchor factor. Does not mutate the input.
    """
    anchors_map = anchors_map or config.GROUP_ANCHORS
    ref = stitched[config.REF_GROUP]
    out = {config.REF_GROUP: stitched[config.REF_GROUP].copy()}
    for g, anchors in anchors_map.items():
        if verbose:
            print(f"\n  [{g} → {config.REF_GROUP}]  anchors: {anchors}")
        factor = anchor_scale_factor(ref, stitched[g], anchors, verbose=verbose)
        out[g] = stitched[g] * factor
    return out


def assemble_panel(stitched_scaled: dict,
                   renormalize: bool = True,
                   title_case: bool = True,
                   verbose: bool = True) -> pd.DataFrame:
    """Phase 3: dedupe bridge crops, assemble, and (optionally) renormalize."""
    panel = stitched_scaled[config.REF_GROUP].copy()

    for g in [g for g in config.GROUPS if g != config.REF_GROUP]:
        for col in stitched_scaled[g].columns:
            if col in panel.columns:
                if verbose and col in config.DUPLICATE_CROPS:
                    print(f"    drop '{col}' from {g}  (kept from earlier group)")
                continue
            panel[col] = stitched_scaled[g][col]

    if renormalize:
        mx = np.nanmax(panel.to_numpy(dtype=float))
        if mx > 0:
            panel = panel * (100.0 / mx)

    if title_case:
        panel.columns = [c.title() for c in panel.columns]
    panel.index.name = "Date"
    return panel


def build_panel(estimator: str = "ratio_of_means",
                renormalize: bool = True,
                title_case: bool = True,
                seam_records: list = None,
                anomalies: list = None,
                verbose: bool = True) -> pd.DataFrame:
    """Run the full Phase 1→2→3 pipeline and return the assembled panel."""
    stitched = stitch_all_groups(estimator, seam_records, anomalies, verbose)
    scaled = cross_scale_groups(stitched, verbose=verbose)
    return assemble_panel(scaled, renormalize, title_case, verbose)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 62)
    print("  PHASE 1 — Within-group stitching")
    print("═" * 62)
    stitched = stitch_all_groups(verbose=True)

    print("\n" + "═" * 62)
    print("  PHASE 2 — Cross-group rescaling  (all → g1 scale)")
    print("═" * 62)
    scaled = cross_scale_groups(stitched, verbose=True)

    print("\n" + "═" * 62)
    print("  PHASE 3 — Deduplication and final assembly")
    print("═" * 62)
    panel = assemble_panel(scaled, verbose=True)

    print("\n  Zero / NaN fraction per crop:")
    zfrac = (panel.fillna(0) <= 0).mean().sort_values()
    for crop, z in zfrac.items():
        flag = "  ← SPARSE" if z > config.ZERO_GATE else ""
        print(f"    {crop:<14}  {z:5.1%}{flag}")

    write_panel(panel)
    print(f"\n  ✓ Wrote {config.PANEL_CSV}")
    print(f"    {panel.shape[0]} weeks × {panel.shape[1]} crops")
    print(f"    Crops: {list(panel.columns)}")


if __name__ == "__main__":
    main()
