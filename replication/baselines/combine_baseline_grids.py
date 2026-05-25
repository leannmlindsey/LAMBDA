#!/usr/bin/env python3
"""
combine_baseline_grids.py

Build the "combined" baseline Filtered genome-wide metrics the same way the
gLM combined best was assembled: for each (baseline, window), take whichever
of the v2 grid or the v3-extended-minsize grid achieved the higher
macro-averaged MCC.

  v2 grid : min_size 8k-20k, merge_gap 1k-8k, smooth 3/5/7,
            thresh 0.5-3.0
  v3 grid : min_size 20k-40k, merge_gap 1k, smooth 3/5/7,
            thresh 0.5-2.0

Inputs:
  grid_search_baselines_v2grid/grid_search_best.csv
  grid_search_baselines_v3grid/grid_search_best.csv

Output:
  grid_search_baselines_combined/grid_search_best.csv   (max-MCC row per baseline x window)
  + prints a paper-ready Filtered table (FPR / Recall / MCC).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2", required=True, help="v2 grid_search_best.csv for baselines")
    p.add_argument("--v3", required=True, help="v3 grid_search_best.csv for baselines")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    v2 = pd.read_csv(args.v2)
    v3 = pd.read_csv(args.v3)
    both = pd.concat([v2.assign(grid="v2"), v3.assign(grid="v3")], ignore_index=True)

    # For each (model, window) keep the row with the highest macro_mcc.
    idx = both.groupby(["model", "window_size"])["macro_mcc"].idxmax()
    best = both.loc[idx].sort_values(["model", "window_size"]).reset_index(drop=True)

    best["filt_fpr_pct"] = 100 * best["total_fp"] / (best["total_fp"] + best["total_tn"])
    best["filt_recall_pct"] = 100 * best["macro_recall"]
    best["filt_mcc"] = best["macro_mcc"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "grid_search_best.csv"
    best.drop(columns=["filt_fpr_pct", "filt_recall_pct", "filt_mcc"]).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    show = best[["model", "window_size", "grid", "norm_method", "smooth_window",
                 "threshold", "min_region_size", "merge_gap",
                 "filt_fpr_pct", "filt_recall_pct", "filt_mcc"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nCombined baseline Filtered (max-MCC of v2 vs v3 per baseline x window):")
        print(show.to_string(index=False,
              formatters={"filt_fpr_pct": "{:.2f}".format,
                          "filt_recall_pct": "{:.2f}".format,
                          "filt_mcc": "{:.3f}".format}))


if __name__ == "__main__":
    main()
