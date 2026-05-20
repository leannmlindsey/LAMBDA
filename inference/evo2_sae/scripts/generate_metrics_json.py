#!/usr/bin/env python3
"""
Generate metrics JSON files from existing prediction CSVs.

Processes CSV files that have 'label' and 'pred_label' columns and generates
a companion _metrics.json file for each, matching the format produced by
evo2_nn_inference.py and evo2_lp_inference.py.

Works with SAE results (*_sae_results.csv), NN predictions (*_nn_predictions.csv),
and LP predictions (*_lp_predictions.csv).

Usage:
    # Single file
    python scripts/generate_metrics_json.py --input results.csv

    # Multiple files
    python scripts/generate_metrics_json.py --input file1.csv file2.csv file3.csv

    # All result CSVs in a directory
    python scripts/generate_metrics_json.py --input_dir /path/to/results/

    # Also produce an aggregate summary JSON
    python scripts/generate_metrics_json.py --input_dir /path/to/results/ --aggregate
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def calculate_metrics(labels, predictions, prob_col=None):
    """Calculate comprehensive classification metrics."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)) if len(set(labels)) > 1 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "num_samples": int(len(labels)),
    }

    # AUC if probability column available
    if prob_col is not None:
        try:
            metrics["auc"] = float(roc_auc_score(labels, prob_col))
        except ValueError:
            metrics["auc"] = 0.0

    return metrics


def print_metrics(metrics, filename):
    """Print metrics in a formatted table."""
    print(f"\n  File: {filename}")
    print(f"  Samples: {metrics['num_samples']} "
          f"(TP={metrics['true_positives']} TN={metrics['true_negatives']} "
          f"FP={metrics['false_positives']} FN={metrics['false_negatives']})")
    print(f"  {'Metric':<14} {'Value':>8}")
    print(f"  {'-' * 24}")
    print(f"  {'Accuracy':<14} {metrics['accuracy']:>8.4f}")
    print(f"  {'Precision':<14} {metrics['precision']:>8.4f}")
    print(f"  {'Recall':<14} {metrics['recall']:>8.4f}")
    print(f"  {'F1':<14} {metrics['f1']:>8.4f}")
    print(f"  {'MCC':<14} {metrics['mcc']:>8.4f}")
    if 'auc' in metrics:
        print(f"  {'AUC':<14} {metrics['auc']:>8.4f}")
    print(f"  {'Sensitivity':<14} {metrics['sensitivity']:>8.4f}")
    print(f"  {'Specificity':<14} {metrics['specificity']:>8.4f}")


def process_csv(csv_path, label_col="label", pred_col="pred_label"):
    """Process a single CSV and save metrics JSON."""
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        print(f"  SKIP: '{label_col}' column not found in {csv_path}")
        return None
    if pred_col not in df.columns:
        print(f"  SKIP: '{pred_col}' column not found in {csv_path}")
        return None

    df = df.dropna(subset=[label_col, pred_col])
    if len(df) == 0:
        print(f"  SKIP: No valid rows in {csv_path}")
        return None

    labels = df[label_col].values
    preds = df[pred_col].values

    # Check for probability columns
    prob_col = None
    if 'prob_1' in df.columns:
        prob_col = df['prob_1'].values
    elif 'mean_activation' in df.columns:
        prob_col = df['mean_activation'].values.astype(float)

    metrics = calculate_metrics(labels, preds, prob_col)
    metrics["input_csv"] = os.path.abspath(csv_path)

    # Save JSON
    metrics_path = csv_path.replace(".csv", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate metrics JSON from existing prediction CSVs"
    )
    parser.add_argument("--input", nargs="+", default=[],
                        help="One or more result CSV files")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing result CSVs")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Column name for ground truth (default: label)")
    parser.add_argument("--pred_col", type=str, default="pred_label",
                        help="Column name for predictions (default: pred_label)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Save aggregate metrics across all files")
    args = parser.parse_args()

    # Collect input files
    csv_files = list(args.input)
    if args.input_dir:
        for pattern in ["*_sae_results.csv", "*_nn_predictions.csv", "*_lp_predictions.csv", "*_results.csv"]:
            csv_files.extend(sorted(glob.glob(os.path.join(args.input_dir, pattern))))
        # Deduplicate
        seen = set()
        deduped = []
        for f in csv_files:
            f_abs = os.path.abspath(f)
            if f_abs not in seen:
                seen.add(f_abs)
                deduped.append(f)
        csv_files = deduped

    if not csv_files:
        print("ERROR: No input files. Use --input or --input_dir.")
        sys.exit(1)

    print("=" * 60)
    print("Generate Metrics JSON from Prediction CSVs")
    print("=" * 60)
    print(f"  Files to process: {len(csv_files)}")

    all_results = {}
    processed = 0
    skipped = 0

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"\n  WARNING: File not found: {csv_path}")
            skipped += 1
            continue

        metrics = process_csv(csv_path, args.label_col, args.pred_col)
        if metrics is None:
            skipped += 1
            continue

        filename = os.path.basename(csv_path)
        all_results[filename] = metrics
        print_metrics(metrics, filename)
        metrics_path = csv_path.replace(".csv", "_metrics.json")
        print(f"  Saved: {metrics_path}")
        processed += 1

    # Aggregate
    if args.aggregate and len(all_results) > 1 and args.input_dir:
        all_labels = []
        all_preds = []
        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if args.label_col in df.columns and args.pred_col in df.columns:
                all_labels.extend(df[args.label_col].dropna().values.tolist())
                all_preds.extend(df[args.pred_col].dropna().values.tolist())

        if all_labels:
            agg_metrics = calculate_metrics(all_labels, all_preds)
            print(f"\n{'=' * 60}")
            print(f"AGGREGATE ({len(all_results)} files, {agg_metrics['num_samples']} samples)")
            print_metrics(agg_metrics, "AGGREGATE")

            agg_path = os.path.join(args.input_dir, "aggregate_metrics.json")
            with open(agg_path, "w") as f:
                json.dump(agg_metrics, f, indent=2)
            print(f"  Saved: {agg_path}")

    print(f"\n{'=' * 60}")
    print(f"Done: {processed} processed, {skipped} skipped")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
