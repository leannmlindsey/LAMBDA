#!/usr/bin/env python3
"""
stage_baselines_for_grid_search.py

Stage baseline genome-wide predictions in the directory layout that the
paper's grid_search_clustering.py expects:

    <data_dir>/per_segment_<ws>/<model_name>/<assembly>*predictions*.csv

Source layout (transferred from Biowulf):
    outputs/results/<baseline>/genome_wide/<window>/<assembly>_*_predictions.csv

Output layout (what grid_search_clustering.py reads):
    outputs/reproduction/grid_search_baselines_data/per_segment_<ws>/<baseline>/<assembly>_*_predictions.csv

grid_search_clustering.py reads each CSV with `load_segments`, which needs
columns: start, end, prob_1 (or activation_value), pred_label, label.
Our baseline CSVs already have all of these, so we just need to link/copy
them into the right place.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BASELINES = ["baseline_gc", "baseline_kmer4", "baseline_kmer6"]
WINDOWS = ["2k", "4k", "8k"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--staged_root", required=True,
                   help="Will create <staged_root>/per_segment_{2k,4k,8k}/<baseline>/...")
    args = p.parse_args()

    src_root = Path(args.results_root)
    dst_root = Path(args.staged_root)

    total = 0
    for w in WINDOWS:
        for b in BASELINES:
            src_dir = src_root / b / "genome_wide" / w
            dst_dir = dst_root / f"per_segment_{w}" / b
            if not src_dir.is_dir():
                print(f"[skip] missing {src_dir}")
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            csvs = sorted(src_dir.glob("*_predictions.csv"))
            for csv in csvs:
                dst_csv = dst_dir / csv.name
                if not dst_csv.exists():
                    shutil.copy2(csv, dst_csv)
            total += len(csvs)
            print(f"[stage] {w} / {b}: {len(csvs)} CSVs -> {dst_dir}")

    print(f"\nDone. Staged {total} CSVs under {dst_root}")


if __name__ == "__main__":
    main()
