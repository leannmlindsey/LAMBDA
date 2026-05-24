#!/usr/bin/env python3
"""
stage_baselines_for_paper_scripts.py

Stage the baseline genome-wide predictions in the same directory layout that
the paper's analyze_genome_wide_results.py expects, with JSONs rewritten to
the gLM key schema (true_positives / true_negatives / false_positives /
false_negatives).

Input layout (what we transferred from Biowulf):
    outputs/results/<baseline>/genome_wide/<window>/
        GCA_*_genomic_segments_predictions.csv
        GCA_*_genomic_segments_predictions_metrics.json   (keys: tp/tn/fp/fn)

Output layout (what analyze_genome_wide_results.py wants):
    outputs/reproduction/staged_baselines/<baseline>_<window>/
        GCA_*_genomic_segments_predictions.csv             (symlinked)
        GCA_*_genomic_segments_predictions_metrics.json    (rewritten keys)

We also write a *_predictions_metrics.json key set that matches the gLM
JSONs exactly so analyze_genome_wide_results.py's CSV-discovery + clustering
path can find the predictions CSV next door.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


BASELINES = ["baseline_gc", "baseline_kmer4", "baseline_kmer6"]
WINDOWS = ["2k", "4k", "8k"]


def rewrite_json(src_json: Path, dst_json: Path) -> None:
    """Translate baseline JSON keys to gLM JSON keys.

    Baseline JSON:  {"tp": ..., "tn": ..., "fp": ..., "fn": ..., "mcc": ..., ...}
    gLM JSON:       {"true_positives": ..., "true_negatives": ..., "false_positives": ..., "false_negatives": ..., "mcc": ..., ...}
    """
    with src_json.open() as f:
        data = json.load(f)
    if "true_positives" in data:  # already correct schema, copy as-is
        translated = data
    else:
        translated = dict(data)
        translated["true_positives"] = data.get("tp", 0)
        translated["true_negatives"] = data.get("tn", 0)
        translated["false_positives"] = data.get("fp", 0)
        translated["false_negatives"] = data.get("fn", 0)
        # Map "n" to "num_samples" for completeness (gLM JSON uses num_samples).
        if "num_samples" not in translated and "n" in data:
            translated["num_samples"] = data["n"]
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w") as f:
        json.dump(translated, f, indent=2)


def stage_one(src_dir: Path, dst_dir: Path) -> int:
    """Stage one (baseline, window) directory. Returns number of genomes."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(src_dir.glob("*_predictions.csv"))
    n = 0
    for csv in csvs:
        json_src = csv.with_name(csv.stem + "_metrics.json")
        if not json_src.exists():
            print(f"  [skip] {csv.name}: no matching _metrics.json")
            continue
        # Copy CSV (cheap; relative path issues if we symlink, so just copy)
        csv_dst = dst_dir / csv.name
        if not csv_dst.exists():
            shutil.copy2(csv, csv_dst)
        # Rewrite JSON to gLM schema
        json_dst = dst_dir / json_src.name
        rewrite_json(json_src, json_dst)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True,
                   help="Directory containing baseline_gc/, baseline_kmer4/, baseline_kmer6/")
    p.add_argument("--staged_root", required=True,
                   help="Output root: one <baseline>_<window>/ dir per combo")
    args = p.parse_args()

    results_root = Path(args.results_root)
    staged_root = Path(args.staged_root)

    for b in BASELINES:
        for w in WINDOWS:
            src = results_root / b / "genome_wide" / w
            dst = staged_root / f"{b}_{w}"
            if not src.is_dir():
                print(f"[skip] {src} missing")
                continue
            n = stage_one(src, dst)
            print(f"[stage] {b} / {w}: {n} genomes -> {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
