#!/usr/bin/env python3
"""run_graph.py
Command-line entry point for building and inspecting the multiplex crop graph
(src/graph) outside of a full experiment run.

This is a thin CLI around src.graph.build_graph.build_multiplex_graph(): it
loads the stitched panel, takes a single train slice (a "mock fold"), builds
every adjacency layer (data-driven + static + knowledge-graph), prints a
summary, and optionally saves the graph to disk.

Examples
--------
    # Quick smoke test on all crops, first 400 weeks as train (A_te skipped
    # by default since it needs `pip install pyif` and is slow):
    python run_graph.py

    # Restrict to a handful of crops for a fast sanity check:
    python run_graph.py --crops Tomato Onion Garlic Basil Corn

    # Include transfer entropy (slow) and save the result to disk:
    python run_graph.py --include-te --save outputs/panel/graph_smoke.npz

    # Drop the knowledge-graph layers (A_taxon/A_subst/A_compl/A_pheno):
    python run_graph.py --no-kg

    # Use a different train window (e.g. everything up to week 300):
    python run_graph.py --train-end 300

Reload a saved graph later with:
    from src.graph.build_graph import load_graph
    graph = load_graph("outputs/panel/graph_smoke.npz")
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from src.utils.io import read_panel
from src.graph.build_graph import build_multiplex_graph, save_graph


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", type=str, nargs="+", default=None,
                   help="crop names to include (default: all crops in "
                        "outputs/panel/crop_list.csv that are in the panel)")
    p.add_argument("--n-crops", type=int, default=None,
                   help="use only the first N crops (after --crops filtering); "
                        "handy for a fast sanity check")
    p.add_argument("--train-end", type=int, default=400,
                   help="number of weeks (from the start of the panel) to use "
                        "as the train slice for this mock fold (default: 400)")
    p.add_argument("--include-te", action="store_true",
                   help="build the transfer-entropy layer A_te (slow; "
                        "requires `pip install pyif`, otherwise returns zeros)")
    p.add_argument("--no-kg", action="store_true",
                   help="drop the knowledge-graph layers (A_taxon, A_subst, "
                        "A_compl, A_pheno)")
    p.add_argument("--stl-period", type=int, default=52,
                   help="annual period (in weeks) for STL decomposition")
    p.add_argument("--max-lag", type=int, default=13,
                   help="max lead-lag (in weeks) considered for A_lag")
    p.add_argument("--threshold-pct", type=float, default=80.0,
                   help="percentile threshold applied per layer to drop weak edges")
    p.add_argument("--save", type=str, default=None,
                   help="path to save the built graph to (.npz + .meta.csv)")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    panel = read_panel()

    crops = args.crops or pd.read_csv(config.PANEL_DIR / "crop_list.csv")["crop"].tolist()
    crops = [c for c in crops if c in panel.columns]
    missing = set(args.crops or []) - set(crops)
    if missing:
        print(f"WARNING: crops not found in panel and skipped: {sorted(missing)}")
    if args.n_crops:
        crops = crops[:args.n_crops]

    panel = panel[crops]
    train = panel.iloc[:args.train_end]
    print(f"Panel: {panel.shape}, train slice: {train.shape}, N crops: {len(crops)}")

    graph = build_multiplex_graph(
        panel_train=train,
        crops=crops,
        include_te=args.include_te,
        include_kg=not args.no_kg,
        stl_period=args.stl_period,
        max_lag=args.max_lag,
        threshold_pct=args.threshold_pct,
        verbose=verbose,
    )

    print()
    graph.summary()

    if args.save:
        save_graph(graph, args.save)


if __name__ == "__main__":
    main()
