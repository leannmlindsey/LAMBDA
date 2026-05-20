#!/usr/bin/env python3
"""
Inference Script for DNABERT-2

This script performs inference on a CSV file using a fine-tuned DNABERT-2 model.
It outputs predictions with probability scores for threshold analysis.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or thresholded)

Usage:
    python inference_dnabert2.py \
        --input_csv /path/to/test.csv \
        --model_path /path/to/finetuned/model \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with fine-tuned DNABERT-2 model"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for prob_1 (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: uses model_path, or falls back to base model)",
    )
    return parser.parse_args()


def run_inference(
    model,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple:
    """
    Run inference on sequences.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        sequences: List of DNA sequences
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        Tuple of (probabilities array shape (n, 2), predictions array)
    """
    model.eval()
    all_probs = []
    all_preds = []

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Running inference"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)

    probs_array = np.vstack(all_probs)
    preds_array = np.array(all_preds)

    return probs_array, preds_array


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

    # AUC
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("DNABERT-2 Inference")
    print("=" * 60)

    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")

    # Determine tokenizer path (with fallback to base model)
    tokenizer_path = args.tokenizer_path or args.model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except (OSError, EnvironmentError) as e:
        # Fallback to base model tokenizer if custom tokenizer not found
        print(f"  Tokenizer not found at {tokenizer_path}, falling back to base model...")
        base_model = "zhihan1996/DNABERT-2-117M"
        print(f"  Loading tokenizer from: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run inference
    sequences = df["sequence"].tolist()
    probs, preds = run_inference(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, device,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"\nApplying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)

        # Add metadata
        metrics["model_path"] = args.model_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["num_samples"] = len(df)

        # Save metrics
        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Print metrics
        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
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

    elif has_labels:
        # Just print basic accuracy even if not saving
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
