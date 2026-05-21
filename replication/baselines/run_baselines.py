#!/usr/bin/env python3
"""
run_baselines.py

Compute simple sequence-feature baselines for the LAMBDA prophage-detection
benchmark, matching the linear-probe protocol used to evaluate genomic
language model embeddings (StandardScaler → LogisticRegression).

Three baselines, all using sklearn LogisticRegression(max_iter=1000):

    baseline_gc      GC content (1 feature) — equivalent to learning an
                     optimal GC threshold.
    baseline_kmer4   Normalized tetranucleotide frequencies (256 features).
    baseline_kmer6   Normalized hexanucleotide frequencies (4 096 features).

All three are trained on `binary_segments_<window>/train.csv` (the only
labeled training data in LAMBDA) and evaluated on every test split at the
same window size:

    binary_segments_<window>/test.csv
    error_and_bias_<window>/test.csv
    genome_wide_segments_<...>/*.csv          (one CSV per assembly)

Outputs are written in the same layout 02_submit_inference_jobs.sh uses, so
the aggregator (03_build_website_data.py) auto-discovers them without code
changes:

    RESULTS_ROOT/
        baseline_gc/{binary,error_bias,genome_wide}/<window>/<file>_predictions.csv
        baseline_kmer4/...
        baseline_kmer6/...
        figures/gc_distribution.pdf            (with --plot)

Usage:
    python run_baselines.py \\
        --dataset_root /path/to/LAMBDA_dataset \\
        --output_root /path/to/LAMBDA_results \\
        --windows 2k 4k 8k \\
        --plot
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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


# ─── Standard LAMBDA palette ────────────────────────────────────────────────
# Phage = pink-red (from the ProkBERT family color slot, here used as a class
# color); Bacteria = teal (DNABERT2 color slot). Both come from the LAMBDA
# project palette so the figure matches the rest of the manuscript figures.
PHAGE_COLOR = "#f35f73"
BACTERIA_COLOR = "#02afbd"
THRESHOLD_COLOR = "#333333"

CATEGORIES = ["binary", "error_bias", "genome_wide"]

DATASET_SUBDIR = {
    ("binary", "2k"): "binary_segments_2k",
    ("binary", "4k"): "binary_segments_4k",
    ("binary", "8k"): "binary_segments_8k",
    ("error_bias", "2k"): "error_and_bias_2k",
    ("error_bias", "4k"): "error_and_bias_4k",
    ("error_bias", "8k"): "error_and_bias_8k",
    ("genome_wide", "2k"): "genome_wide_segments_2k_1k",
    ("genome_wide", "4k"): "genome_wide_segments_4k_2k",
    ("genome_wide", "8k"): "genome_wide_segments_8k_4k",
}


# ─── Featurization ──────────────────────────────────────────────────────────

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def featurize_gc(sequences: Iterable[str]) -> np.ndarray:
    """GC content alone — shape (n, 1)."""
    return np.array([[gc_content(s)] for s in sequences], dtype=np.float32)


def _kmer_vocabulary(k: int) -> List[str]:
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def featurize_kmers(
    sequences: Iterable[str],
    k: int,
    vectorizer: CountVectorizer | None = None,
) -> Tuple[np.ndarray, CountVectorizer]:
    """Normalized k-mer frequencies — shape (n, 4**k).

    Vocabulary is fixed to the full A/C/G/T k-mer set; sequences with N's
    contribute zero to those positions. L1-normalized per row to give
    frequencies rather than counts.
    """
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


# ─── Classifier ──────────────────────────────────────────────────────────────

def fit_classifier(
    train_X: np.ndarray, train_y: np.ndarray, seed: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    """Same protocol as the gLM linear probes."""
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    clf = LogisticRegression(
        max_iter=1000, random_state=seed, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(train_Xs, train_y)
    return clf, scaler


def predict(
    clf: LogisticRegression, scaler: StandardScaler, X: np.ndarray, threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)
    preds = (probs[:, 1] >= threshold).astype(int)
    return probs, preds


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
    except ValueError:
        metrics["auc"] = None
    return metrics


# ─── GC plot helpers ────────────────────────────────────────────────────────

def optimal_gc_threshold(labels: np.ndarray, gc_values: np.ndarray) -> Tuple[float, float]:
    """Sweep thresholds and find the one that maximizes MCC.

    Tries both polarities (phage = GC>t vs. phage = GC<t) because either can
    win depending on the corpus.
    """
    thresholds = np.linspace(0.10, 0.90, 161)  # 0.5% steps
    best_t, best_mcc = 0.5, -2.0
    for t in thresholds:
        for polarity in (1, -1):
            if polarity == 1:
                preds = (gc_values >= t).astype(int)
            else:
                preds = (gc_values < t).astype(int)
            mcc = matthews_corrcoef(labels, preds)
            if mcc > best_mcc:
                best_mcc, best_t = mcc, t
    return float(best_t), float(best_mcc)


def make_gc_plot(plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]],
                 output_path: Path) -> None:
    """3-panel KDE figure — one panel per window size."""
    windows = ["2k", "4k", "8k"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    x_grid = np.linspace(0, 1, 400)

    for ax, win in zip(axes, windows):
        if win not in plot_data:
            ax.set_visible(False)
            continue
        gc_phage, gc_bact, threshold, mcc = plot_data[win]
        kde_phage = gaussian_kde(gc_phage)(x_grid)
        kde_bact = gaussian_kde(gc_bact)(x_grid)
        ax.fill_between(x_grid, kde_bact, color=BACTERIA_COLOR, alpha=0.55, label="Bacteria")
        ax.fill_between(x_grid, kde_phage, color=PHAGE_COLOR, alpha=0.55, label="Phage")
        ax.plot(x_grid, kde_bact, color=BACTERIA_COLOR, linewidth=1.2)
        ax.plot(x_grid, kde_phage, color=PHAGE_COLOR, linewidth=1.2)
        ax.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("GC content")
        ax.set_title(f"{win}  —  MCC = {mcc:.3f},  threshold = {threshold:.2f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Density")
            ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path} (+ .png)")


# ─── Output ─────────────────────────────────────────────────────────────────

def write_predictions_and_metrics(
    in_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
) -> None:
    out = in_df.copy()
    out["prob_0"] = probs[:, 0]
    out["prob_1"] = probs[:, 1]
    out["pred_label"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if "label" in out.columns and out["label"].notna().all():
        try:
            labels = out["label"].astype(int).to_numpy()
            metrics = compute_metrics(labels, preds, probs)
            metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2)
        except (ValueError, TypeError):
            pass


# ─── Per-window pipeline ────────────────────────────────────────────────────

def gather_eval_csvs(dataset_root: Path, window: str) -> List[Tuple[Path, str]]:
    """Return list of (csv_path, category) for every eval slice at this window."""
    out: List[Tuple[Path, str]] = []
    for category in CATEGORIES:
        subdir = DATASET_SUBDIR.get((category, window))
        if not subdir:
            continue
        cat_dir = dataset_root / subdir
        if not cat_dir.exists():
            continue
        if category == "genome_wide":
            for p in sorted(cat_dir.glob("*.csv")):
                out.append((p, category))
        else:
            test_csv = cat_dir / "test.csv"
            if test_csv.exists():
                out.append((test_csv, category))
    return out


def process_window(
    dataset_root: Path,
    output_root: Path,
    window: str,
    plot_data: dict | None,
) -> None:
    train_csv = dataset_root / f"binary_segments_{window}" / "train.csv"
    if not train_csv.exists():
        print(f"[skip] {train_csv} not found")
        return

    print(f"\n=== window {window} ===")
    print(f"loading train: {train_csv}")
    train_df = pd.read_csv(train_csv)
    if "sequence" not in train_df.columns or "label" not in train_df.columns:
        print(f"[skip] {train_csv} missing required columns")
        return
    train_seqs = train_df["sequence"].astype(str).tolist()
    train_y = train_df["label"].astype(int).to_numpy()
    print(f"  {len(train_seqs)} training sequences")

    print("fitting GC baseline...")
    train_X_gc = featurize_gc(train_seqs)
    clf_gc, scaler_gc = fit_classifier(train_X_gc, train_y)

    print("fitting k=4 baseline...")
    train_X_k4, vec_k4 = featurize_kmers(train_seqs, k=4)
    clf_k4, scaler_k4 = fit_classifier(train_X_k4, train_y)

    print("fitting k=6 baseline...")
    train_X_k6, vec_k6 = featurize_kmers(train_seqs, k=6)
    clf_k6, scaler_k6 = fit_classifier(train_X_k6, train_y)

    eval_csvs = gather_eval_csvs(dataset_root, window)
    print(f"evaluating on {len(eval_csvs)} CSV(s) across categories...")

    for csv_path, category in eval_csvs:
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            continue
        seqs = df["sequence"].astype(str).tolist()
        if not seqs:
            continue

        base_name = csv_path.stem
        # GC
        Xg = featurize_gc(seqs)
        p, pred = predict(clf_gc, scaler_gc, Xg)
        out = output_root / "baseline_gc" / category / window / f"{base_name}_predictions.csv"
        write_predictions_and_metrics(df, p, pred, out)
        # k = 4
        Xk4, _ = featurize_kmers(seqs, k=4, vectorizer=vec_k4)
        p, pred = predict(clf_k4, scaler_k4, Xk4)
        out = output_root / "baseline_kmer4" / category / window / f"{base_name}_predictions.csv"
        write_predictions_and_metrics(df, p, pred, out)
        # k = 6
        Xk6, _ = featurize_kmers(seqs, k=6, vectorizer=vec_k6)
        p, pred = predict(clf_k6, scaler_k6, Xk6)
        out = output_root / "baseline_kmer6" / category / window / f"{base_name}_predictions.csv"
        write_predictions_and_metrics(df, p, pred, out)

    # ─── GC plot data: use the binary_segments test.csv ───────────────────
    if plot_data is not None:
        binary_test = dataset_root / f"binary_segments_{window}" / "test.csv"
        if binary_test.exists():
            df = pd.read_csv(binary_test)
            seqs = df["sequence"].astype(str).tolist()
            y = df["label"].astype(int).to_numpy()
            gc_vals = np.array([gc_content(s) for s in seqs])
            t, mcc = optimal_gc_threshold(y, gc_vals)
            plot_data[window] = (gc_vals[y == 1], gc_vals[y == 0], t, mcc)
            print(f"  GC plot: phage={int((y==1).sum())}, bacteria={int((y==0).sum())}, "
                  f"threshold={t:.3f}, MCC={mcc:.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple sequence-feature baselines (GC, k=4 k-mer, k=6 k-mer) for LAMBDA.",
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Path containing binary_segments_*, error_and_bias_*, genome_wide_segments_*")
    parser.add_argument("--output_root", required=True,
                        help="Where to write per-baseline prediction CSVs and metrics JSON")
    parser.add_argument("--windows", nargs="+", default=["2k", "4k", "8k"],
                        choices=["2k", "4k", "8k"])
    parser.add_argument("--plot", action="store_true",
                        help="Generate the 3-panel GC-distribution figure into output_root/figures/")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    if not dataset_root.exists():
        sys.exit(f"ERROR: dataset_root {dataset_root} does not exist")
    output_root.mkdir(parents=True, exist_ok=True)

    plot_data: dict = {} if args.plot else None

    for window in args.windows:
        process_window(dataset_root, output_root, window, plot_data)

    if plot_data:
        make_gc_plot(plot_data, output_root / "figures" / "gc_distribution.pdf")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
