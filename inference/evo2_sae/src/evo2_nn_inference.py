#!/usr/bin/env python3
"""
Inference Script for Evo2 + 3-Layer NN Classifier

This script performs inference on a CSV file using the pretrained Evo2 model
for embedding extraction and a trained 3-layer NN classifier (from
evo2_embedding_analysis.py).

Workflow:
    1. Load the Evo2 backbone and extract embeddings
    2. Load the trained 3-layer NN classifier and scaler
    3. Standardize embeddings with the saved scaler
    4. Classify with the NN and output predictions

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for metrics)

Output CSV format:
    - All original columns preserved
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label

Usage:
    python src/evo2_nn_inference.py \
        --input_csv /path/to/test.csv \
        --classifier_path /path/to/three_layer_nn.pt \
        --scaler_path /path/to/three_layer_nn_scaler.pkl \
        --output_csv /path/to/predictions.csv \
        --model evo2_7b \
        --layer blocks.28.mlp.l3

    # With pre-extracted embeddings (skip Evo2 model loading):
    python src/evo2_nn_inference.py \
        --input_csv /path/to/test.csv \
        --classifier_path /path/to/three_layer_nn.pt \
        --scaler_path /path/to/three_layer_nn_scaler.pkl \
        --embeddings_path /path/to/embeddings.npz \
        --output_csv /path/to/predictions.csv
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
import torch.nn as nn
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


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with Evo2 embeddings + trained 3-layer NN classifier"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--classifier_path", type=str, default=None,
        help="Path to trained 3-layer NN checkpoint (three_layer_nn.pt)",
    )
    parser.add_argument(
        "--scaler_path", type=str, default=None,
        help="Path to saved StandardScaler (three_layer_nn_scaler.pkl)",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Directory containing classifier .pt and scaler .pkl files. "
             "Auto-discovers three_layer_nn.pt and three_layer_nn_scaler.pkl.",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Path to output CSV (default: input with _nn_predictions suffix)",
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
        "--batch_size", type=int, default=1,
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
    parser.add_argument(
        "--save_embeddings", type=str, default=None,
        help="Path to save extracted embeddings as .npz (for reuse by other classifiers)",
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
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
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
            candidate = os.path.join(cp, "three_layer_nn.pt")
            if os.path.exists(candidate):
                args.classifier_path = candidate
            else:
                raise FileNotFoundError(
                    f"Could not find three_layer_nn.pt in {cp}"
                )
        if args.scaler_path is None:
            candidate = os.path.join(cp, "three_layer_nn_scaler.pkl")
            if os.path.exists(candidate):
                args.scaler_path = candidate
            else:
                raise FileNotFoundError(
                    f"Could not find three_layer_nn_scaler.pkl in {cp}"
                )

    # Validate required paths
    if args.classifier_path is None:
        raise ValueError("--classifier_path is required (or use --checkpoint_path directory)")
    if args.scaler_path is None:
        raise ValueError("--scaler_path is required (or use --checkpoint_path directory)")

    print("\n" + "=" * 60)
    print("Evo2 + 3-Layer NN Inference")
    print("=" * 60)

    start_time = time.time()

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

    # Step 1: Get embeddings (extract or load)
    if args.embeddings_path:
        print(f"\n1. Loading pre-extracted embeddings from: {args.embeddings_path}")
        loaded = np.load(args.embeddings_path)
        # Try common key names
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

    # Save embeddings for reuse if requested
    if args.save_embeddings:
        np.savez(args.save_embeddings, embeddings=embeddings)
        print(f"  Saved embeddings to: {args.save_embeddings}")

    # Step 2: Load scaler and standardize
    print(f"\nLoading scaler from: {args.scaler_path}")
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    embeddings_scaled = scaler.transform(embeddings)
    print(f"  Embeddings standardized")

    # Step 3: Load 3-layer NN classifier
    print(f"\nLoading classifier from: {args.classifier_path}")
    checkpoint = torch.load(args.classifier_path, map_location=device, weights_only=True)
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    if input_dim != embeddings.shape[1]:
        raise ValueError(
            f"Classifier input_dim ({input_dim}) does not match embedding dim "
            f"({embeddings.shape[1]}). Make sure --layer and --pooling match training."
        )

    classifier = ThreeLayerNN(input_dim, hidden_dim).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    print(f"  Classifier loaded (input_dim={input_dim}, hidden_dim={hidden_dim})")

    # Step 4: Run classification
    print(f"\nRunning classification...")
    embeddings_tensor = torch.FloatTensor(embeddings_scaled).to(device)

    with torch.no_grad():
        logits = classifier(embeddings_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

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
        args.output_csv = f"{base}_nn_predictions{ext}"

    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and print metrics if labels present
    if has_labels:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds, probs)

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
            metrics["layer"] = args.layer
            metrics["pooling"] = args.pooling
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
