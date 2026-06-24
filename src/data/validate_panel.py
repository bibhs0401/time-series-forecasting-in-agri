"""validate_panel.py — sanity checks on the stitched panel.

Run:
    python -m src.data.validate_panel
    # or
    python src/data/validate_panel.py

Tables are written to ``outputs/tables/``; passing crops are written to
``outputs/panel/crop_list.csv`` for downstream feature scripts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import config
from src.utils.io import read_panel


def validate(csv_path=config.PANEL_CSV, out_dir=config.TABLES_DIR):
    df = read_panel(csv_path)
    print(f"\n{'='*55}")
    print(f"  Panel shape:  {df.shape[0]} weeks × {df.shape[1]} crops")
    print(f"  Date range:   {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Annual cycles: ~{df.shape[0]/52:.1f}")

    # 1. Check for gaps in the weekly index
    expected = pd.date_range(df.index.min(), df.index.max(), freq="W-SUN")
    missing_dates = expected.difference(df.index)
    print(f"\n  Missing date rows: {len(missing_dates)}")
    if len(missing_dates):
        print(f"    {list(missing_dates[:5])}")

    missing_path = out_dir / "panel_missing_dates.csv"
    pd.DataFrame({"date": missing_dates}).to_csv(missing_path, index=False)

    # 2. Zero/NaN fraction per crop
    print(f"\n  Zero/NaN fraction (gate = {config.ZERO_GATE:.0%}):")
    zfrac = (df.fillna(0) <= 0).mean().sort_values(ascending=False)
    sparse = []
    crop_rows = []
    for crop, z in zfrac.items():
        is_sparse = z > config.ZERO_GATE
        flag = "  ← SPARSE" if is_sparse else ""
        print(f"    {crop:<25} {z:5.1%}{flag}")
        if is_sparse:
            sparse.append(crop)
        crop_rows.append({
            "crop": crop,
            "zero_frac": z,
            "sparse": is_sparse,
            "passes_gate": not is_sparse,
        })

    crop_quality = pd.DataFrame(crop_rows)
    crop_quality_path = out_dir / "panel_crop_quality.csv"
    crop_quality.to_csv(crop_quality_path, index=False)

    passing = crop_quality.loc[crop_quality["passes_gate"], "crop"]
    crop_list_path = config.PANEL_DIR / "crop_list.csv"
    passing.to_frame().to_csv(crop_list_path, index=False)

    # 3. Value range sanity (should be 0–100 after renorm)
    vmin, vmax = np.nanmin(df.values), np.nanmax(df.values)
    print(f"\n  Value range: [{vmin:.2f}, {vmax:.2f}]  (expect ~0–100)")

    # 4. Duplicate column names
    dupes = df.columns[df.columns.duplicated()].tolist()
    print(f"  Duplicate columns: {dupes if dupes else 'none'}")

    # 5. Summary for paper
    print(f"\n  Crops to DROP (sparse gate): {sparse}")
    print(f"  Crops that PASS: {df.shape[1] - len(sparse)}")

    summary = pd.DataFrame([{
        "n_weeks": df.shape[0],
        "n_crops": df.shape[1],
        "date_min": df.index.min().date().isoformat(),
        "date_max": df.index.max().date().isoformat(),
        "annual_cycles": round(df.shape[0] / 52, 1),
        "n_missing_dates": len(missing_dates),
        "zero_gate": config.ZERO_GATE,
        "value_min": vmin,
        "value_max": vmax,
        "duplicate_columns": ", ".join(dupes) if dupes else "",
        "n_sparse": len(sparse),
        "n_pass": df.shape[1] - len(sparse),
        "sparse_crops": ", ".join(sparse),
    }])
    summary_path = out_dir / "panel_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n  → wrote {crop_quality_path.name}")
    print(f"  → wrote {summary_path.name}")
    print(f"  → wrote {missing_path.name}")
    print(f"  → wrote {crop_list_path}")
    print(f"    Tables → {out_dir}")
    print(f"{'='*55}\n")
    return df, sparse


if __name__ == "__main__":
    validate()
