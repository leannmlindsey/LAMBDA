#!/usr/bin/env python3
"""
extract_baseline_raw_table.py

Consolidate the per-(baseline, window) summary CSVs produced by the paper's
analyze_genome_wide_results.py into a single table containing the rows that
match the paper Table 3 "Raw" columns:

    type == "raw"
    method == "average (mean per-genome)"

Output: outputs/reproduction/baseline_summaries/baseline_raw_table3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASELINES = ["baseline_gc", "baseline_kmer4", "baseline_kmer6"]
WINDOWS = ["2k", "4k", "8k"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summaries_dir", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    summaries_dir = Path(args.summaries_dir)
    rows = []
    for b in BASELINES:
        for w in WINDOWS:
            path = summaries_dir / f"{b}_{w}_summary.csv"
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            df = pd.read_csv(path)
            raw_avg = df[(df["type"] == "raw") & (df["method"] == "average (mean per-genome)")]
            if raw_avg.empty:
                print(f"[skip] {path}: no raw/average row")
                continue
            r = raw_avg.iloc[0]
            rows.append({
                "baseline": b,
                "window": w,
                "n_genomes": int(r["num_files"]),
                "raw_recall_pct": round(100 * r["recall"], 2),
                "raw_fpr_pct": round(100 * r["fpr"], 2),
                "raw_mcc": round(r["mcc"], 3),
                "raw_precision": round(r["precision"], 3),
                "raw_f1": round(r["f1"], 3),
                "raw_specificity": round(r["specificity"], 3),
                "raw_fnr_pct": round(100 * r["fnr"], 2),
            })

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv} ({len(out_df)} rows)")
    print("\nBaseline 'Raw' values for paper Table 3 (macro-averaged across genomes):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out_df[["baseline", "window", "raw_fpr_pct", "raw_recall_pct", "raw_mcc"]].to_string(index=False))


if __name__ == "__main__":
    main()
