#!/usr/bin/env python3
"""run_experiment.py
Command-line entry point for the Florida-Agri rolling-origin forecasting study.

Examples
--------
    # Baselines only (fast, no torch needed):
    python run_experiment.py --no-torch --folds 3

    # Full ladder (baselines + ST-GNNs), point forecasts:
    python run_experiment.py --folds 5 --epochs 80

    # Probabilistic (quantile) ST-GNNs:
    python run_experiment.py --quantiles 0.1 0.5 0.9

    # Ablate feature groups (drop weather G3 and static G4):
    python run_experiment.py --groups G0 G1 G2

Results are written to outputs/tables/experiment_results_long.csv and
outputs/tables/experiment_results_summary.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train.cv import HORIZONS
from src.train.experiment import (
    run_experiment, DEFAULT_BASELINES, DEFAULT_TORCH,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, default=5, help="number of rolling origins")
    p.add_argument("--window", type=int, default=52, help="input window length (weeks)")
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS),
                   help="forecast horizons in weeks")
    p.add_argument("--groups", type=str, nargs="+", default=None,
                   help="feature groups to include (default all): G0 G1 G2 G3 G4")
    p.add_argument("--baselines", type=str, nargs="+", default=None,
                   help=f"baselines to run (default: {DEFAULT_BASELINES})")
    p.add_argument("--torch-models", type=str, nargs="+", default=None,
                   help=f"GNN models to run (default: {DEFAULT_TORCH})")
    p.add_argument("--no-torch", action="store_true",
                   help="skip all GNN models (baselines only)")
    p.add_argument("--quantiles", type=float, nargs="+", default=None,
                   help="quantile levels for probabilistic forecasts, e.g. 0.1 0.5 0.9")
    p.add_argument("--include-te", action="store_true",
                   help="build the transfer-entropy graph layer (slow)")
    p.add_argument("--no-kg", action="store_true",
                   help="drop the knowledge-graph layers (A_taxon, A_subst, "
                        "A_compl, A_pheno); use for the KG ablation")
    p.add_argument("--epochs", type=int, default=60, help="max epochs per GNN")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=52,
                   help="weeks reserved at the END of the panel as a final "
                        "untouched test block (0 disables). Default 52 = one "
                        "seasonal cycle.")
    p.add_argument("--final-test", action="store_true",
                   help="score the held-out block instead of the CV folds. "
                        "Run this ONCE, after all model selection is done. "
                        "Refuses to re-run if the holdout was already scored "
                        "(see --force-final-test).")
    p.add_argument("--force-final-test", action="store_true",
                   help="bypass the holdout-already-scored guard for --final-test. "
                        "Only use this for a deliberate, justified re-run (e.g. "
                        "reproducing a result or fixing an unrelated bug) -- NOT "
                        "for re-selecting models after seeing the holdout score. "
                        "The re-run is still logged.")
    p.add_argument("--forecast-future", action="store_true",
                   help="refit on the FULL panel and forecast beyond the last "
                        "observed week (no scoring possible).")
    p.add_argument("--n-ahead", type=int, default=13,
                   help="weeks ahead for --forecast-future")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch_models = [] if args.no_torch else args.torch_models

    if args.forecast_future:
        from src.train.forecast_future import forecast_future
        out = forecast_future(
            n_ahead=args.n_ahead,
            groups=args.groups,
            baselines=args.baselines,
            torch_models=torch_models or None,
            window=args.window,
            epochs=args.epochs,
            seed=args.seed,
            include_te=args.include_te,
            include_kg=not args.no_kg,
            quantiles=args.quantiles,
            save=True,
            verbose=not args.quiet,
        )
        fc = out["forecast"]
        if len(fc):
            print("\n================ FUTURE FORECAST (head) ================")
            print(fc.head(20).to_string(index=False))
        else:
            print("No future forecasts produced.")
        return

    res = run_experiment(
        groups=args.groups,
        horizons=tuple(args.horizons),
        n_folds=args.folds,
        window=args.window,
        baselines=args.baselines,
        torch_models=torch_models,
        quantiles=args.quantiles,
        include_te=args.include_te,
        include_kg=not args.no_kg,
        epochs=args.epochs,
        seed=args.seed,
        holdout=args.holdout,
        final_test=args.final_test,
        force_final_test=args.force_final_test,
        save=True,
        verbose=not args.quiet,
    )
    summary = res["summary"]
    if len(summary):
        print("\n================ SUMMARY (mean across folds & crops) ================")
        print(summary.to_string(index=False))
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
