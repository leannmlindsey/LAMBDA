#!/usr/bin/env python3
"""
compute_pooled_genome_wide.py

Pool baseline genome-wide predictions into a single nucleotide/segment-level
confusion matrix per (baseline, window), the same way PhiSpy/VirSorter2 etc.
report whole-corpus genome-wide metrics in lambda_results_022826.csv and
all_models_summary.csv.

For each baseline x window:
  - Walk every per-genome prediction CSV under
        <results_root>/<baseline>/genome_wide/<window>/*_predictions.csv
  - Sum tp/tn/fp/fn across genomes -> compute pooled accuracy / precision /
    recall / specificity / F1 / MCC / AUC.
  - Also keep per-genome MCC values for the distribution plot.

Outputs (next to the input results root):
  - <out_dir>/baseline_pooled_genome_wide.csv   (one row per baseline x window)
  - <out_dir>/baseline_per_genome_mcc.csv       (one row per baseline x window x genome)

These slot directly into the existing all_models_summary.csv comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASELINES = ["baseline_gc", "baseline_kmer4", "baseline_kmer6"]
WINDOWS = ["2k", "4k", "8k"]


def per_genome_metrics(df: pd.DataFrame) -> dict:
    """Confusion matrix + MCC/etc. for a single per-genome CSV."""
    y = df["label"].astype(int).to_numpy()
    p = df["pred_label"].astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    out = {"n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    out["mcc"] = mcc_from_cm(tp, tn, fp, fn)
    out["precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
    out["recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
    out["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    out["f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    out["accuracy"] = (tp + tn) / len(y) if len(y) else float("nan")
    try:
        if "prob_1" in df.columns and y.min() != y.max():
            out["auc"] = float(roc_auc_score(y, df["prob_1"].to_numpy()))
        else:
            out["auc"] = float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


def mcc_from_cm(tp: int, tn: int, fp: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else 0.0


def pool_directory(csvs: list[Path]) -> dict:
    """Sum TP/TN/FP/FN across all per-genome CSVs in `csvs`."""
    TP = TN = FP = FN = N = 0
    all_labels = []
    all_probs = []
    n_genomes = 0
    for csv in csvs:
        df = pd.read_csv(csv)
        if not {"label", "pred_label"}.issubset(df.columns):
            continue
        n_genomes += 1
        y = df["label"].astype(int).to_numpy()
        p = df["pred_label"].astype(int).to_numpy()
        TP += int(((y == 1) & (p == 1)).sum())
        TN += int(((y == 0) & (p == 0)).sum())
        FP += int(((y == 0) & (p == 1)).sum())
        FN += int(((y == 1) & (p == 0)).sum())
        N += len(y)
        if "prob_1" in df.columns:
            all_labels.append(y)
            all_probs.append(df["prob_1"].to_numpy())
    pooled = {
        "n_genomes": n_genomes,
        "samples": N,
        "tp": TP, "tn": TN, "fp": FP, "fn": FN,
        "accuracy": (TP + TN) / N if N else float("nan"),
        "precision": TP / (TP + FP) if (TP + FP) else float("nan"),
        "recall": TP / (TP + FN) if (TP + FN) else float("nan"),
        "specificity": TN / (TN + FP) if (TN + FP) else float("nan"),
        "fpr": FP / (TN + FP) if (TN + FP) else float("nan"),
        "fnr": FN / (TP + FN) if (TP + FN) else float("nan"),
        "f1": (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) else float("nan"),
        "mcc": mcc_from_cm(TP, TN, FP, FN),
    }
    if all_labels:
        y_all = np.concatenate(all_labels)
        p_all = np.concatenate(all_probs)
        try:
            pooled["auc"] = float(roc_auc_score(y_all, p_all)) if y_all.min() != y_all.max() else float("nan")
        except ValueError:
            pooled["auc"] = float("nan")
    else:
        pooled["auc"] = float("nan")
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True,
                        help="Directory containing baseline_gc/, baseline_kmer4/, baseline_kmer6/")
    parser.add_argument("--out_dir", required=True,
                        help="Where to write the two summary CSVs")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    per_genome_rows = []

    for baseline in BASELINES:
        for window in WINDOWS:
            gw_dir = results_root / baseline / "genome_wide" / window
            if not gw_dir.is_dir():
                print(f"[skip] {gw_dir} not found")
                continue
            csvs = sorted(gw_dir.glob("*_predictions.csv"))
            if not csvs:
                print(f"[skip] {gw_dir}: no *_predictions.csv files")
                continue

            print(f"[pool] {baseline} / {window}: {len(csvs)} per-genome CSVs")
            pooled = pool_directory(csvs)
            pooled_rows.append({"baseline": baseline, "window": window, **pooled})

            for csv in csvs:
                df = pd.read_csv(csv)
                if not {"label", "pred_label"}.issubset(df.columns):
                    continue
                m = per_genome_metrics(df)
                per_genome_rows.append({
                    "baseline": baseline,
                    "window": window,
                    "assembly": csv.stem.replace("_genomic_segments_predictions", "")
                                          .replace("_segments_predictions", ""),
                    "file": csv.name,
                    **m,
                })

    pooled_df = pd.DataFrame(pooled_rows)
    pg_df = pd.DataFrame(per_genome_rows)

    pooled_path = out_dir / "baseline_pooled_genome_wide.csv"
    pg_path = out_dir / "baseline_per_genome_mcc.csv"
    pooled_df.to_csv(pooled_path, index=False, float_format="%.6f")
    pg_df.to_csv(pg_path, index=False, float_format="%.6f")
    print(f"\nwrote {pooled_path} ({len(pooled_df)} rows)")
    print(f"wrote {pg_path}    ({len(pg_df)} rows)")

    # Pretty-print pooled table for quick reading
    show = pooled_df[["baseline", "window", "n_genomes", "samples",
                      "tp", "tn", "fp", "fn",
                      "mcc", "precision", "recall", "specificity", "f1", "auc"]]
    print("\nPooled genome-wide metrics:")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
