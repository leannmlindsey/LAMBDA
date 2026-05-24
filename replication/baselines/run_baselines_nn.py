#!/usr/bin/env python3
"""
run_baselines_nn.py

Train the SAME 3-Layer Neural Network the gLM "3-Layer NN" column in paper
Table 2 uses, but on the simple sequence-feature baselines (GC, k=4, k=6
k-mer composition) instead of gLM hidden-state embeddings. This produces the
analogous Table 2 entries for the baseline rows.

Architecture (verbatim from upstream gLM repos --- DNABERT2, NTv2, Caduceus
all share this class definition, line-for-line):

    class ThreeLayerNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
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

Training protocol (verbatim):
    - StandardScaler.fit_transform(train), transform(val), transform(test)
    - Adam, lr=1e-3
    - CrossEntropyLoss
    - Full-batch (no minibatches)
    - max 100 epochs, early stopping patience=10 on val loss
    - seed=42

Source: CLAUDE_DNABERT2/DNABERT2_generic_sequence_classification/finetune/embedding_analysis_dnabert2.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, normalize


# ─── Architecture (verbatim from gLM upstream repos) ─────────────────────────

class ThreeLayerNN(nn.Module):
    """Identical to the gLM 3-Layer NN class in DNABERT2/NTv2/Caduceus repos."""

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


# ─── Featurization (same code path as run_baselines.py) ─────────────────────

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s)


def featurize_gc(sequences: Iterable[str]) -> np.ndarray:
    return np.array([[gc_content(s)] for s in sequences], dtype=np.float32)


def _kmer_vocabulary(k: int) -> List[str]:
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def featurize_kmers(
    sequences: Iterable[str],
    k: int,
    vectorizer: CountVectorizer | None = None,
) -> Tuple[np.ndarray, CountVectorizer]:
    if vectorizer is None:
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
            vocabulary=_kmer_vocabulary(k),
        )
        X = vectorizer.fit_transform(s.upper() for s in sequences)
    else:
        X = vectorizer.transform(s.upper() for s in sequences)
    X = normalize(X.astype(np.float32), norm="l1", axis=1)
    return X.toarray(), vectorizer


# ─── Metrics (same as the gLM embedding_analysis_*.py) ──────────────────────

def calculate_metrics(labels: np.ndarray, predictions: np.ndarray,
                      probabilities: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }
    if probabilities is not None:
        try:
            metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        except ValueError:
            metrics["auc"] = 0.0
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["tp"] = int(tp); metrics["tn"] = int(tn)
    metrics["fp"] = int(fp); metrics["fn"] = int(fn)
    return metrics


# ─── Training (verbatim from gLM train_three_layer_nn) ──────────────────────

def train_three_layer_nn(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    hidden_dim: int = 256, epochs: int = 100, lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)

    X_train = torch.FloatTensor(train_scaled).to(device)
    y_train = torch.LongTensor(train_y).to(device)
    X_val = torch.FloatTensor(val_scaled).to(device)
    y_val = torch.LongTensor(val_y).to(device)
    X_test = torch.FloatTensor(test_scaled).to(device)

    input_dim = train_scaled.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0
    epochs_run = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        epochs_run = epoch + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_probs = torch.softmax(test_out, dim=-1).cpu().numpy()
        test_preds = torch.argmax(test_out, dim=-1).cpu().numpy()

    metrics = calculate_metrics(test_y, test_preds, test_probs)
    metrics["epochs_run"] = epochs_run
    metrics["best_val_loss"] = best_val_loss
    metrics["hidden_dim"] = hidden_dim
    metrics["input_dim"] = input_dim
    return metrics


# ─── Per-baseline pipeline ──────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def featurize_split(seqs: List[str], baseline: str,
                    vec: CountVectorizer | None = None) -> Tuple[np.ndarray, CountVectorizer | None]:
    if baseline == "gc":
        return featurize_gc(seqs), None
    elif baseline == "kmer4":
        return featurize_kmers(seqs, k=4, vectorizer=vec)
    elif baseline == "kmer6":
        return featurize_kmers(seqs, k=6, vectorizer=vec)
    raise ValueError(f"unknown baseline {baseline}")


# Same candidate layout resolver used by run_baselines.py, so this script
# auto-detects whichever binary-split layout exists at DATASET_ROOT (the
# Zenodo-style binary_segments_<w>/, the Biowulf merged_datasets_filtered/<w>/,
# or a bare <w>/ directory).
BINARY_SUBDIR_CANDIDATES = {
    "2k": ["binary_segments_2k", "merged_datasets_filtered/2k", "2k"],
    "4k": ["binary_segments_4k", "merged_datasets_filtered/4k", "4k"],
    "8k": ["binary_segments_8k", "merged_datasets_filtered/8k", "8k"],
}


def resolve_binary_dir(dataset_root: Path, window: str) -> Path | None:
    for cand in BINARY_SUBDIR_CANDIDATES.get(window, []):
        p = dataset_root / cand
        if p.is_dir() and (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    return None


def load_val_split(binary_dir: Path, train_df: pd.DataFrame, seed: int):
    """Return a validation DataFrame.

    Prefers an on-disk dev.csv or val.csv. If neither exists, carve a
    stratified 10% slice off the training set (deterministic via seed) so the
    NN's early-stopping criterion still has a held-out set that is NOT the test
    set -- matching the gLM protocol, which never touches test for model
    selection.
    """
    for name in ("dev.csv", "val.csv", "valid.csv"):
        p = binary_dir / name
        if p.exists():
            return pd.read_csv(p), f"on-disk {name}"
    # Fall back to a held-out slice of train.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    n_val = max(1, int(0.10 * len(train_df)))
    val_idx = idx[:n_val]
    return train_df.iloc[val_idx].reset_index(drop=True), "10% carved from train"


def run_one(baseline: str, window: str, dataset_root: Path,
            seed: int, device: torch.device) -> dict | None:
    binary_dir = resolve_binary_dir(dataset_root, window)
    if binary_dir is None:
        tried = BINARY_SUBDIR_CANDIDATES.get(window, [])
        print(f"[skip] {baseline}/{window}: no binary dir with train.csv+test.csv under "
              f"{dataset_root} (tried: {tried})")
        return None
    train_csv = binary_dir / "train.csv"
    test_csv = binary_dir / "test.csv"

    print(f"\n=== baseline={baseline}, window={window} ===")
    print(f"  binary dir: {binary_dir}")
    set_seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    dev_df, dev_src = load_val_split(binary_dir, train_df, seed)
    print(f"  val split: {dev_src}")
    print(f"  sizes: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    train_seqs = train_df["sequence"].astype(str).tolist()
    dev_seqs = dev_df["sequence"].astype(str).tolist()
    test_seqs = test_df["sequence"].astype(str).tolist()

    train_X, vec = featurize_split(train_seqs, baseline)
    dev_X, _ = featurize_split(dev_seqs, baseline, vec=vec)
    test_X, _ = featurize_split(test_seqs, baseline, vec=vec)

    train_y = train_df["label"].astype(int).to_numpy()
    dev_y = dev_df["label"].astype(int).to_numpy()
    test_y = test_df["label"].astype(int).to_numpy()

    print(f"  feature dim: {train_X.shape[1]}")
    metrics = train_three_layer_nn(
        train_X, train_y, dev_X, dev_y, test_X, test_y,
        hidden_dim=256, epochs=100, lr=1e-3, device=device,
    )
    metrics["baseline"] = baseline
    metrics["window"] = window
    print(f"  -> test MCC = {metrics['mcc']:.4f}  (epochs run = {metrics['epochs_run']})")
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description="3-Layer NN baselines for LAMBDA Table 2.")
    p.add_argument("--dataset_root", required=True,
                   help="Path containing binary_segments_{2k,4k,8k}/")
    p.add_argument("--output_root", required=True,
                   help="Where to write metrics JSON and consolidated CSV")
    p.add_argument("--windows", nargs="+", default=["2k", "4k", "8k"],
                   choices=["2k", "4k", "8k"])
    p.add_argument("--baselines", nargs="+",
                   default=["gc", "kmer4", "kmer6"],
                   choices=["gc", "kmer4", "kmer6"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if unset)")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = []
    for baseline in args.baselines:
        for window in args.windows:
            m = run_one(baseline, window, dataset_root, args.seed, device)
            if m is None:
                continue
            rows.append(m)
            # Per-(baseline, window) JSON for audit
            j = out_root / f"baseline_{baseline}_{window}_nn_metrics.json"
            with j.open("w") as f:
                json.dump(m, f, indent=2)

    df = pd.DataFrame(rows)
    summary_csv = out_root / "baseline_nn_summary.csv"
    df.to_csv(summary_csv, index=False, float_format="%.6f")
    print(f"\nwrote {summary_csv} ({len(df)} rows)")

    if len(df):
        with pd.option_context("display.width", 200, "display.max_columns", 12):
            print("\n3-Layer NN baseline results (test MCC):")
            print(df[["baseline", "window", "input_dim", "epochs_run",
                      "accuracy", "precision", "recall", "f1", "mcc"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
