#!/usr/bin/env python3
"""
Calculate classification metrics from SAE inference result CSVs.

Reads one or more CSV files containing 'label' (ground truth) and 'pred_label'
(predicted) columns and computes: accuracy, precision, recall, F1, MCC,
false positive rate (FPR), and false negative rate (FNR).

Can be run standalone on existing results or called from other scripts.

Usage:
    # Single file
    python src/calculate_metrics.py --input results.csv

    # Multiple files
    python src/calculate_metrics.py --input results1.csv results2.csv results3.csv

    # All CSVs in a directory
    python src/calculate_metrics.py --input_dir ./output_directory

    # Save metrics to JSON
    python src/calculate_metrics.py --input results.csv --output_json metrics.json
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
    confusion_matrix,
)


def calculate_metrics(labels, predictions):
    """Calculate classification metrics from ground truth and predictions.

    Handles edge cases: empty arrays, single-class data (all 0s or all 1s).

    Args:
        labels: array-like of ground truth labels (0/1)
        predictions: array-like of predicted labels (0/1)

    Returns:
        Dict of metric name -> float value, or None if no samples
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    if len(labels) == 0:
        return None

    # Confusion matrix with explicit labels so it always returns 2x2
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Accuracy is always safe with non-empty arrays
    acc = float((tp + tn) / len(labels))

    # Precision, recall, F1 — handle single-class gracefully
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # MCC — returns 0 when undefined (single class)
    if len(set(labels)) < 2 or len(set(predictions)) < 2:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(labels, predictions))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(len(labels)),
    }

    return metrics


def print_metrics(metrics, filename=None):
    """Print metrics in a formatted table."""
    if filename:
        print(f"\n  File: {filename}")
        print(f"  Samples: {metrics['total']} (TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']})")
    # Warn if single-class ground truth
    if metrics['tp'] + metrics['fn'] == 0:
        print(f"  NOTE: No positive samples in ground truth (all label=0)")
    elif metrics['tn'] + metrics['fp'] == 0:
        print(f"  NOTE: No negative samples in ground truth (all label=1)")
    print(f"  {'Metric':<12} {'Value':>8}")
    print(f"  {'-'*22}")
    print(f"  {'Accuracy':<12} {metrics['accuracy']:>8.4f}")
    print(f"  {'Precision':<12} {metrics['precision']:>8.4f}")
    print(f"  {'Recall':<12} {metrics['recall']:>8.4f}")
    print(f"  {'F1':<12} {metrics['f1']:>8.4f}")
    print(f"  {'MCC':<12} {metrics['mcc']:>8.4f}")
    print(f"  {'FPR':<12} {metrics['fpr']:>8.4f}")
    print(f"  {'FNR':<12} {metrics['fnr']:>8.4f}")


def calculate_metrics_from_csv(csv_path, label_col="label", pred_col="pred_label"):
    """Load a CSV and calculate metrics.

    Args:
        csv_path: Path to CSV file
        label_col: Column name for ground truth labels
        pred_col: Column name for predicted labels

    Returns:
        Dict of metrics, or None if columns not found
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        print(f"  WARNING: '{label_col}' column not found in {csv_path}, skipping")
        return None
    if pred_col not in df.columns:
        print(f"  WARNING: '{pred_col}' column not found in {csv_path}, skipping")
        return None

    # Drop rows with missing labels or predictions
    df = df.dropna(subset=[label_col, pred_col])

    if len(df) == 0:
        print(f"  WARNING: No valid rows in {csv_path}, skipping")
        return None

    labels = df[label_col].values
    predictions = df[pred_col].values

    return calculate_metrics(labels, predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate classification metrics from SAE inference results"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        help="One or more result CSV files",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing result CSVs (processes all *_sae_results.csv and *_results.csv files)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for ground truth labels (default: label)",
    )
    parser.add_argument(
        "--pred_col",
        type=str,
        default="pred_label",
        help="Column name for predicted labels (default: pred_label)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Save all metrics to a JSON file",
    )
    args = parser.parse_args()

    # Collect input files
    csv_files = list(args.input)
    if args.input_dir:
        for pattern in ["*_sae_results.csv", "*_results.csv"]:
            csv_files.extend(sorted(glob.glob(os.path.join(args.input_dir, pattern))))
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for f in csv_files:
            f_abs = os.path.abspath(f)
            if f_abs not in seen:
                seen.add(f_abs)
                deduped.append(f)
        csv_files = deduped

    if not csv_files:
        print("ERROR: No input files specified. Use --input or --input_dir.")
        sys.exit(1)

    print("=" * 60)
    print("Classification Metrics")
    print("=" * 60)

    all_results = {}
    all_labels = []
    all_preds = []

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"\n  WARNING: File not found, skipping: {csv_path}")
            continue

        metrics = calculate_metrics_from_csv(csv_path, args.label_col, args.pred_col)
        if metrics is None:
            continue

        filename = os.path.basename(csv_path)
        all_results[filename] = metrics
        print_metrics(metrics, filename=filename)

        # Accumulate for aggregate metrics
        df = pd.read_csv(csv_path)
        all_labels.extend(df[args.label_col].values.tolist())
        all_preds.extend(df[args.pred_col].values.tolist())

    # Aggregate metrics across all files
    if len(all_results) > 1:
        aggregate = calculate_metrics(all_labels, all_preds)
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ({len(all_results)} files, {aggregate['total']} total samples)")
        print(f"{'=' * 60}")
        print_metrics(aggregate)
        all_results["AGGREGATE"] = aggregate

    # Save to JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved metrics to: {args.output_json}")

    print()


if __name__ == "__main__":
    main()
