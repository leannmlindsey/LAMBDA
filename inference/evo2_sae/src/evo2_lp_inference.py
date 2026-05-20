#!/usr/bin/env python3
"""
Inference Script for Evo2 + Linear Probe (Logistic Regression) Classifier

This script performs inference on a CSV file using pre-extracted Evo2 embeddings
and a trained linear probe classifier (from evo2_embedding_analysis.py).

Workflow:
    1. Load embeddings (pre-extracted or extract from Evo2)
    2. Load the trained linear probe and scaler
    3. Standardize embeddings with the saved scaler
    4. Classify with the linear probe and output predictions

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for metrics)

Output CSV format:
    - All original columns preserved
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label

Usage:
    # With pre-extracted embeddings (recommended):
    python src/evo2_lp_inference.py \
        --input_csv /path/to/test.csv \
        --classifier_path /path/to/linear_probe.pkl \
        --scaler_path /path/to/linear_probe_scaler.pkl \
        --embeddings_path /path/to/embeddings_pretrained.npz \
        --output_csv /path/to/predictions.csv \
        --save_metrics

    # With live Evo2 extraction:
    python src/evo2_lp_inference.py \
        --input_csv /path/to/test.csv \
        --classifier_path /path/to/linear_probe.pkl \
        --scaler_path /path/to/linear_probe_scaler.pkl \
        --output_csv /path/to/predictions.csv \
        --model evo2_7b \
        --layer blocks.28.mlp.l3
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with Evo2 embeddings + trained linear probe classifier"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--classifier_path", type=str, default=None,
        help="Path to trained linear probe (linear_probe.pkl)",
    )
    parser.add_argument(
        "--scaler_path", type=str, default=None,
        help="Path to saved StandardScaler (linear_probe_scaler.pkl)",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Directory containing classifier .pkl and scaler .pkl files. "
             "Auto-discovers linear_probe.pkl and linear_probe_scaler.pkl.",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Path to output CSV (default: input with _lp_predictions suffix)",
    )
    parser.add_argument(
        "--embeddings_path", type=str, default=None,
        help="Path to pre-extracted embeddings (.npz). If provided, skips Evo2 model loading.",
    )
    parser.add_argument(
        "--model", type=str, default="evo2_7b",
        choices=["evo2_7b", "evo2_40b"],
        help="Evo2 model to use (ignored if --embeddings_path is provided)",
    )
    parser.add_argument(
        "--layer", type=str, default="blocks.28.mlp.l3",
        help="Layer for embedding extraction (must match training)",
    )
    parser.add_argument(
        "--pooling", type=str, default="mean",
        choices=["mean", "first", "last", "max"],
        help="Pooling strategy (must match training)",
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Maximum sequence length (truncate longer sequences)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold for prob_1",
    )
    parser.add_argument(
        "--save_metrics", action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    return parser.parse_args()


def extract_embeddings(
    model,
    sequences: List[str],
    layer_name: str,
    batch_size: int,
    max_length,
    pooling: str,
) -> np.ndarray:
    """Extract embeddings from Evo2 model."""
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        for seq in batch_seqs:
            if max_length is not None and len(seq) > max_length:
                seq = seq[:max_length]

            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')

            with torch.no_grad():
                outputs, embeddings = model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[layer_name]
                )
                layer_emb = embeddings[layer_name]

                if pooling == "mean":
                    pooled = layer_emb.mean(dim=1)
                elif pooling == "first":
                    pooled = layer_emb[:, 0, :]
                elif pooling == "last":
                    pooled = layer_emb[:, -1, :]
                elif pooling == "max":
                    pooled = layer_emb.max(dim=1)[0]

                all_embeddings.append(pooled.cpu().float().numpy())

    return np.vstack(all_embeddings)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    args = parse_arguments()

    # Auto-discover classifier and scaler from --checkpoint_path directory
    if args.checkpoint_path:
        cp = args.checkpoint_path
        if args.classifier_path is None:
            candidate = os.path.join(cp, "linear_probe.pkl")
            if os.path.exists(candidate):
                args.classifier_path = candidate
            else:
                raise FileNotFoundError(
                    f"Could not find linear_probe.pkl in {cp}"
                )
        if args.scaler_path is None:
            candidate = os.path.join(cp, "linear_probe_scaler.pkl")
            if os.path.exists(candidate):
                args.scaler_path = candidate
            else:
                raise FileNotFoundError(
                    f"Could not find linear_probe_scaler.pkl in {cp}"
                )

    # Validate required paths
    if args.classifier_path is None:
        raise ValueError("--classifier_path is required (or use --checkpoint_path directory)")
    if args.scaler_path is None:
        raise ValueError("--scaler_path is required (or use --checkpoint_path directory)")

    print("\n" + "=" * 60)
    print("Evo2 + Linear Probe Inference")
    print("=" * 60)

    start_time = time.time()

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Step 1: Get embeddings (extract or load)
    if args.embeddings_path:
        print(f"\n1. Loading pre-extracted embeddings from: {args.embeddings_path}")
        loaded = np.load(args.embeddings_path)
        for key in ["embeddings", "test_embeddings", "train_embeddings"]:
            if key in loaded:
                embeddings = loaded[key]
                print(f"  Loaded key '{key}', shape: {embeddings.shape}")
                break
        else:
            keys = list(loaded.keys())
            embeddings = loaded[keys[0]]
            print(f"  Loaded key '{keys[0]}', shape: {embeddings.shape}")

        if len(embeddings) != len(df):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) does not match CSV rows ({len(df)}). "
                f"Make sure the embeddings file matches the input CSV."
            )
    else:
        print(f"\n1. Loading Evo2 model: {args.model}")
        from evo2 import Evo2
        evo2_model = Evo2(args.model)
        print(f"  Model loaded")

        print(f"\n2. Extracting embeddings (layer={args.layer}, pooling={args.pooling})...")
        sequences = df["sequence"].tolist()
        embeddings = extract_embeddings(
            evo2_model, sequences,
            args.layer, args.batch_size,
            args.max_length, args.pooling,
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        del evo2_model
        torch.cuda.empty_cache()

    # Step 2: Load scaler and standardize
    print(f"\nLoading scaler from: {args.scaler_path}")
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    embeddings_scaled = scaler.transform(embeddings)
    print(f"  Embeddings standardized")

    # Step 3: Load linear probe classifier
    print(f"\nLoading linear probe from: {args.classifier_path}")
    with open(args.classifier_path, "rb") as f:
        clf = pickle.load(f)
    print(f"  Linear probe loaded")

    # Step 4: Run classification
    print(f"\nRunning classification...")
    preds = clf.predict(embeddings_scaled)
    probs = clf.predict_proba(embeddings_scaled)

    if args.threshold != 0.5:
        print(f"  Applying custom threshold: {args.threshold}")
        preds = (probs[:, 1] >= args.threshold).astype(int)

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_lp_predictions{ext}"

    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and print metrics if labels present
    if has_labels:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds, probs[:, 1])

        print("\n" + "=" * 60)
        print(f"METRICS (threshold = {args.threshold:.2f})")
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

        if args.save_metrics:
            metrics["classifier_path"] = args.classifier_path
            metrics["input_csv"] = args.input_csv
            metrics["threshold"] = args.threshold
            metrics["num_samples"] = len(df)

            metrics_path = args.output_csv.replace(".csv", "_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
