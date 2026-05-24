#!/usr/bin/env python3
"""
Grid search over clustering hyperparameters for prophage detection.

Pipeline steps:
  1. Collapse overlapping windows via majority voting (2k window, 1k stride → 1k segments)
  2. Per-genome baseline normalization (none, zscore, median_subtract, percentile)
  3. Bidirectional EWM smoothing
  4. Threshold → cluster → filter by min region size

Searches over:
  - Normalization method (none, zscore, median_subtract)
  - Score threshold (applied after normalization + smoothing)
  - Min region size (minimum cluster size in bp to keep)
  - Smoothing window (EWM span for bidirectional smoothing)

For each combination, computes macro-averaged MCC across all genomes (NaN → 0).

Outputs:
  - grid_search_results.csv: full results table
  - grid_search_best.csv: best hyperparams per model/window_size
  - grid_search_heatmap_{size}.png: heatmap figures

Usage:
  python grid_search_clustering.py --data-dir ../DATA --gt ../DATA/lambda_ground_truth.csv --output-dir grid_search_results
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Ground truth loading ─────────────────────────────────────────────────────

def load_ground_truth(gt_path):
    """Load ground truth prophage regions, keyed by assembly."""
    gt = defaultdict(list)
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row["Assembly"]
            gt[assembly].append({
                "start": int(row["start"]),
                "end": int(row["end"]),
            })
    return gt


# ── Per-segment CSV loading ──────────────────────────────────────────────────

def extract_assembly_from_filename(filename):
    """Extract assembly ID (GCA_*, GCF_*, NC_*) from a prediction CSV filename."""
    basename = os.path.basename(filename)
    m = re.match(r'(GC[AF]_\d+\.\d+)', basename)
    if m:
        return m.group(1)
    m = re.match(r'(NC_\d+)', basename)
    if m:
        return m.group(1)
    return None


def load_segments(csv_path):
    """Load per-segment predictions from a CSV file.

    Returns DataFrame with columns: start, end, prob_1, pred_label, label
    Handles both standard and ProkBERT column formats.
    """
    df = pd.read_csv(csv_path)

    result = pd.DataFrame()
    result["start"] = df["start"]
    result["end"] = df["end"]

    # Score column
    if "prob_1" in df.columns:
        result["prob_1"] = df["prob_1"].astype(float)
    elif "activation_value" in df.columns:
        result["prob_1"] = df["activation_value"].astype(float)
    else:
        result["prob_1"] = 0.5

    # Prediction column (for majority voting)
    if "pred_label" in df.columns:
        result["pred_label"] = df["pred_label"].astype(int)
    else:
        result["pred_label"] = (result["prob_1"] >= 0.5).astype(int)

    # Label column (ground truth)
    if "label" in df.columns:
        result["label"] = df["label"].astype(int)
    elif "in_prophage" in df.columns:
        result["label"] = df["in_prophage"].astype(int)
    else:
        result["label"] = 0

    return result.sort_values("start").reset_index(drop=True)


# ── Step 1: Collapse overlapping windows ─────────────────────────────────────

def collapse_overlapping_windows(df, window_size=2000, step_size=1000):
    """Collapse overlapping window predictions using majority voting.

    With 2k windows and 1k stride, each 1k region is covered by ~2 windows.
    Uses majority voting for predictions and averages scores.

    Returns DataFrame with non-overlapping segments: start, end, prob_1, pred_label, label
    """
    if len(df) == 0:
        return df

    min_pos = int(df["start"].min())
    max_pos = int(df["end"].max())

    segments = []
    starts = np.array(df["start"].values)
    ends = np.array(df["end"].values)
    scores = np.array(df["prob_1"].values)
    preds = np.array(df["pred_label"].values)
    labels = np.array(df["label"].values)

    for pos in range(min_pos, max_pos, step_size):
        seg_start = pos
        seg_end = min(pos + step_size, max_pos)

        # Find all windows that cover this segment
        mask = (starts <= seg_start) & (ends >= seg_end)
        n_overlap = mask.sum()

        if n_overlap == 0:
            continue

        # Majority vote for prediction
        votes_for_phage = preds[mask].sum()
        majority_pred = 1 if votes_for_phage > n_overlap / 2 else 0

        # Average scores
        avg_score = scores[mask].mean()

        # Label (should be same for all overlapping windows)
        true_label = labels[mask][0]

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "prob_1": avg_score,
            "pred_label": majority_pred,
            "label": true_label,
        })

    return pd.DataFrame(segments)


# ── Step 2: Per-genome normalization ─────────────────────────────────────────

def normalize_scores(scores, method="none"):
    """Normalize prob_1 scores per genome to account for baseline differences.

    Methods:
        none: No normalization, use raw scores
        zscore: (score - mean) / std — standard z-score
        robust_zscore: (score - median) / MAD — robust to outliers
        median_subtract: score - median — shift baseline to 0
    """
    scores = np.array(scores, dtype=float)

    if method == "none":
        return scores

    elif method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std

    elif method == "robust_zscore":
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        if mad == 0:
            return scores - median
        return (scores - median) / mad

    elif method == "median_subtract":
        median = np.median(scores)
        return scores - median

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ── Step 3: Bidirectional EWM smoothing ──────────────────────────────────────

def bidirectional_smooth(scores, span=5):
    """Apply bidirectional exponentially-weighted-mean smoothing.

    Forward pass + backward pass, averaged.
    """
    if span <= 1 or len(scores) == 0:
        return scores

    s = pd.Series(scores)
    forward = s.ewm(span=span, adjust=False).mean()
    backward = s[::-1].ewm(span=span, adjust=False).mean()[::-1]
    return ((forward + backward) / 2).values


# ── Step 4: Threshold → cluster → filter ─────────────────────────────────────

def cluster_and_filter(df, scores_to_threshold, threshold, merge_gap, min_region_size):
    """Apply threshold → cluster → filter pipeline.

    Parameters:
        df: DataFrame with start, end, prob_1, label
        scores_to_threshold: array of (possibly normalized+smoothed) scores
        threshold: score threshold for calling phage
        merge_gap: max gap (bp) between segments to merge
        min_region_size: min cluster size (bp) to keep

    Returns:
        filtered_pred: array of 0/1 predictions (same length as df)
        phage_regions: list of dicts with start, end, avg_score, size
    """
    n = len(df)
    pred = (scores_to_threshold >= threshold).astype(int)

    positive_indices = np.where(pred == 1)[0]

    if len(positive_indices) == 0:
        return np.zeros(n, dtype=int), []

    # Cluster nearby positive segments
    clusters = []
    current_cluster = [positive_indices[0]]
    starts = df["start"].values
    ends = df["end"].values

    for i in range(1, len(positive_indices)):
        prev_idx = positive_indices[i - 1]
        curr_idx = positive_indices[i]

        gap = starts[curr_idx] - ends[prev_idx]

        if gap <= merge_gap:
            current_cluster.append(curr_idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr_idx]

    clusters.append(current_cluster)

    # Filter by min region size
    filtered_pred = np.zeros(n, dtype=int)
    phage_regions = []
    raw_scores = df["prob_1"].values

    for cluster in clusters:
        cluster_start = starts[cluster[0]]
        cluster_end = ends[cluster[-1]]
        cluster_size = cluster_end - cluster_start

        if cluster_size >= min_region_size:
            filtered_pred[cluster] = 1
            avg_score = raw_scores[cluster].mean()
            phage_regions.append({
                "start": int(cluster_start),
                "end": int(cluster_end),
                "size": int(cluster_size),
                "avg_score": float(avg_score),
            })

    return filtered_pred, phage_regions


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    denom_val = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_val <= 0:
        return 0.0
    denom = math.sqrt(denom_val)
    if denom == 0:
        return 0.0
    return num / denom


def compute_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    mcc = compute_mcc(tp, tn, fp, fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
          else float("nan"))

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mcc": mcc, "precision": precision, "recall": recall, "f1": f1,
    }


# ── Main grid search ────────────────────────────────────────────────────────

def run_grid_search(data_dir, gt_path, output_dir,
                    norm_methods, thresholds, min_sizes, merge_gaps,
                    smooth_windows, window_sizes):
    """Run grid search across all models, window sizes, and hyperparameters."""

    os.makedirs(output_dir, exist_ok=True)
    gt = load_ground_truth(gt_path)
    gt_assemblies = set(gt.keys())

    print(f"Ground truth: {len(gt_assemblies)} assemblies")

    all_results = []

    for ws in window_sizes:
        seg_dir = os.path.join(data_dir, f"per_segment_{ws}")
        if not os.path.isdir(seg_dir):
            print(f"  Skipping {ws}: {seg_dir} not found")
            continue

        model_dirs = sorted([
            d for d in os.listdir(seg_dir)
            if os.path.isdir(os.path.join(seg_dir, d))
        ])

        for model_name in model_dirs:
            model_dir = os.path.join(seg_dir, model_name)

            csv_files = glob.glob(os.path.join(model_dir, "*predictions*.csv"))
            csv_files += glob.glob(os.path.join(model_dir, "*_results.csv"))

            # Load and collapse all genome CSVs for this model
            genome_data = {}  # assembly → collapsed DataFrame
            for csv_path in csv_files:
                assembly = extract_assembly_from_filename(csv_path)
                if assembly and assembly in gt_assemblies:
                    try:
                        raw_df = load_segments(csv_path)
                        collapsed = collapse_overlapping_windows(raw_df)
                        if len(collapsed) > 0:
                            genome_data[assembly] = collapsed
                    except Exception as e:
                        print(f"    Error loading {csv_path}: {e}")

            if not genome_data:
                print(f"  {model_name} {ws}: no matching genome CSVs found")
                continue

            print(f"  {model_name} {ws}: {len(genome_data)} genomes loaded & collapsed")

            # Pre-compute normalized + smoothed scores for each (norm, smooth) combo
            # to avoid redundant computation in the inner loop
            precomputed = {}
            for norm_method in norm_methods:
                for smooth_win in smooth_windows:
                    key = (norm_method, smooth_win)
                    precomputed[key] = {}
                    for assembly, cdf in genome_data.items():
                        scores = cdf["prob_1"].values.copy()
                        scores = normalize_scores(scores, method=norm_method)
                        scores = bidirectional_smooth(scores, span=smooth_win)
                        precomputed[key][assembly] = scores

            # Grid search over all hyperparameter combinations
            total_combos = (len(norm_methods) * len(smooth_windows) *
                            len(thresholds) * len(min_sizes) * len(merge_gaps))
            combo_count = 0

            for norm_method in norm_methods:
                for smooth_win in smooth_windows:
                    precomp_key = (norm_method, smooth_win)

                    for threshold in thresholds:
                        for min_size in min_sizes:
                            for merge_gap in merge_gaps:
                                combo_count += 1

                                per_genome_mcc = []
                                per_genome_precision = []
                                per_genome_recall = []
                                per_genome_f1 = []
                                total_tp = total_tn = total_fp = total_fn = 0

                                for assembly, cdf in genome_data.items():
                                    scores = precomputed[precomp_key][assembly]

                                    filtered_pred, regions = cluster_and_filter(
                                        cdf, scores, threshold, merge_gap, min_size
                                    )

                                    m = compute_metrics(cdf["label"].values, filtered_pred)

                                    total_tp += m["tp"]
                                    total_tn += m["tn"]
                                    total_fp += m["fp"]
                                    total_fn += m["fn"]

                                    per_genome_mcc.append(m["mcc"] if not math.isnan(m["mcc"]) else 0.0)
                                    per_genome_precision.append(m["precision"] if not math.isnan(m["precision"]) else 0.0)
                                    per_genome_recall.append(m["recall"] if not math.isnan(m["recall"]) else 0.0)
                                    per_genome_f1.append(m["f1"] if not math.isnan(m["f1"]) else 0.0)

                                n_genomes = len(genome_data)
                                macro_mcc = sum(per_genome_mcc) / n_genomes
                                macro_precision = sum(per_genome_precision) / n_genomes
                                macro_recall = sum(per_genome_recall) / n_genomes
                                macro_f1 = sum(per_genome_f1) / n_genomes
                                micro_mcc = compute_mcc(total_tp, total_tn, total_fp, total_fn)

                                all_results.append({
                                    "model": model_name,
                                    "window_size": ws,
                                    "norm_method": norm_method,
                                    "smooth_window": smooth_win,
                                    "threshold": threshold,
                                    "min_region_size": min_size,
                                    "merge_gap": merge_gap,
                                    "n_genomes": n_genomes,
                                    "macro_mcc": macro_mcc,
                                    "macro_precision": macro_precision,
                                    "macro_recall": macro_recall,
                                    "macro_f1": macro_f1,
                                    "micro_mcc": micro_mcc,
                                    "total_tp": total_tp,
                                    "total_tn": total_tn,
                                    "total_fp": total_fp,
                                    "total_fn": total_fn,
                                })

            print(f"    {combo_count} hyperparameter combinations evaluated")

    # Write full results
    df = pd.DataFrame(all_results)
    full_path = os.path.join(output_dir, "grid_search_results.csv")
    df.to_csv(full_path, index=False)
    print(f"\nWrote {full_path} ({len(df)} rows)")

    # Find best per model/window_size (by macro MCC)
    if len(df) > 0:
        best_idx = df.groupby(["model", "window_size"])["macro_mcc"].idxmax()
        best_df = df.loc[best_idx].sort_values(["window_size", "macro_mcc"], ascending=[True, False])
        best_path = os.path.join(output_dir, "grid_search_best.csv")
        best_df.to_csv(best_path, index=False)
        print(f"Wrote {best_path}")

        print("\n=== Best hyperparameters per model/window_size ===")
        print(f"{'Model':<22} {'WS':<4} {'Norm':<16} {'Smooth':>6} {'Thresh':>6} "
              f"{'MinSize':>8} {'MCC':>7} {'Prec':>7} {'Recall':>7}")
        print("-" * 100)
        for _, row in best_df.iterrows():
            print(f"{row['model']:<22} {row['window_size']:<4} {row['norm_method']:<16} "
                  f"{row['smooth_window']:>6} {row['threshold']:>6.2f} "
                  f"{row['min_region_size']:>8} "
                  f"{row['macro_mcc']:>7.3f} {row['macro_precision']:>7.3f} {row['macro_recall']:>7.3f}")

    return df


def generate_heatmaps(output_dir):
    """Generate heatmap figures from grid search results.

    For each window size, shows the best MCC achievable at each (threshold, min_size)
    across all normalization/smoothing combos.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping heatmap generation")
        return

    results_path = os.path.join(output_dir, "grid_search_results.csv")
    df = pd.read_csv(results_path)

    window_sizes = sorted(df["window_size"].unique())

    for ws in window_sizes:
        ws_df = df[df["window_size"] == ws]
        models = sorted(ws_df["model"].unique())

        n_models = len(models)
        if n_models == 0:
            continue

        # For the heatmap, show best MCC across all norm/smooth combos
        # at each (threshold, min_size) point
        ncols = min(3, n_models)
        nrows = math.ceil(n_models / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(f"Hyperparameter Grid Search — {ws} Input Length (best across norm/smooth)",
                     fontsize=14, fontweight="bold")

        if n_models == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        thresholds = sorted(ws_df["threshold"].unique())
        min_sizes = sorted(ws_df["min_region_size"].unique())

        vmin = 0
        vmax = max(ws_df["macro_mcc"].max(), 0.01)

        for idx, model in enumerate(models):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax = axes[row_idx, col_idx]

            model_df = ws_df[ws_df["model"] == model]

            # For each (threshold, min_size), take the best MCC across norm/smooth/merge_gap
            matrix = np.full((len(thresholds), len(min_sizes)), np.nan)
            for ti, t in enumerate(thresholds):
                for si, s in enumerate(min_sizes):
                    subset = model_df[
                        (model_df["threshold"] == t) &
                        (model_df["min_region_size"] == s)
                    ]
                    if len(subset) > 0:
                        matrix[ti, si] = subset["macro_mcc"].max()

            im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto",
                          vmin=vmin, vmax=vmax, origin="upper")

            ax.set_xticks(range(len(min_sizes)))
            ax.set_xticklabels([f"{s // 1000}kb" for s in min_sizes], fontsize=8)
            ax.set_yticks(range(len(thresholds)))
            ax.set_yticklabels([f"{t:.2f}" for t in thresholds], fontsize=8)
            ax.set_xlabel("Min Region Size", fontsize=9)
            ax.set_ylabel("Score Threshold", fontsize=9)

            # Show best MCC in title
            best_mcc = model_df["macro_mcc"].max()
            best_row = model_df.loc[model_df["macro_mcc"].idxmax()]
            ax.set_title(f"{model} ({ws}) — best={best_mcc:.3f} [{best_row['norm_method']}]",
                        fontsize=9)

            # Annotate cells
            for ti_idx in range(len(thresholds)):
                for si_idx in range(len(min_sizes)):
                    val = matrix[ti_idx, si_idx]
                    if not np.isnan(val):
                        color = "white" if val > (vmax * 0.6) else "black"
                        ax.text(si_idx, ti_idx, f"{val:.3f}",
                               ha="center", va="center", fontsize=7, color=color)

        for idx in range(n_models, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"grid_search_heatmap_{ws}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")

    # Also generate a normalization comparison chart
    # For each model, show best MCC by normalization method
    for ws in window_sizes:
        ws_df = df[df["window_size"] == ws]
        models = sorted(ws_df["model"].unique())
        norm_methods = sorted(ws_df["norm_method"].unique())

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5))

        x = np.arange(len(models))
        width = 0.8 / len(norm_methods)

        for i, nm in enumerate(norm_methods):
            mccs = []
            for model in models:
                subset = ws_df[(ws_df["model"] == model) & (ws_df["norm_method"] == nm)]
                mccs.append(subset["macro_mcc"].max() if len(subset) > 0 else 0)
            ax.bar(x + i * width, mccs, width, label=nm)

        ax.set_ylabel("Best Macro MCC")
        ax.set_title(f"Normalization Method Comparison — {ws}")
        ax.set_xticks(x + width * (len(norm_methods) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"norm_comparison_{ws}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")

    # Generate merge_gap comparison if multiple values were searched
    for ws in window_sizes:
        ws_df = df[df["window_size"] == ws]
        merge_gaps = sorted(ws_df["merge_gap"].unique())
        if len(merge_gaps) <= 1:
            continue

        models = sorted(ws_df["model"].unique())
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5))

        x = np.arange(len(models))
        width = 0.8 / len(merge_gaps)

        for i, mg in enumerate(merge_gaps):
            mccs = []
            for model in models:
                subset = ws_df[(ws_df["model"] == model) & (ws_df["merge_gap"] == mg)]
                mccs.append(subset["macro_mcc"].max() if len(subset) > 0 else 0)
            ax.bar(x + i * width, mccs, width, label=f"{mg // 1000}kb")

        ax.set_ylabel("Best Macro MCC")
        ax.set_title(f"Merge Gap Comparison — {ws}")
        ax.set_xticks(x + width * (len(merge_gaps) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.legend(title="Merge Gap")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"merge_gap_comparison_{ws}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")

    # Generate smooth_window comparison if multiple values were searched
    for ws in window_sizes:
        ws_df = df[df["window_size"] == ws]
        smooth_wins = sorted(ws_df["smooth_window"].unique())
        if len(smooth_wins) <= 1:
            continue

        models = sorted(ws_df["model"].unique())
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5))

        x = np.arange(len(models))
        width = 0.8 / len(smooth_wins)

        for i, sw in enumerate(smooth_wins):
            mccs = []
            for model in models:
                subset = ws_df[(ws_df["model"] == model) & (ws_df["smooth_window"] == sw)]
                mccs.append(subset["macro_mcc"].max() if len(subset) > 0 else 0)
            ax.bar(x + i * width, mccs, width, label=f"span={sw}")

        ax.set_ylabel("Best Macro MCC")
        ax.set_title(f"Smoothing Window Comparison — {ws}")
        ax.set_xticks(x + width * (len(smooth_wins) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.legend(title="EWM Span")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"smooth_window_comparison_{ws}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search clustering hyperparameters")
    parser.add_argument("--data-dir", required=True,
                        help="Base DATA directory containing per_segment_{2k,4k,8k}")
    parser.add_argument("--gt", required=True,
                        help="Path to ground truth CSV")
    parser.add_argument("--output-dir", default="grid_search_results",
                        help="Output directory for results")
    parser.add_argument("--norm-methods", nargs="+",
                        default=["none", "zscore", "robust_zscore", "median_subtract"],
                        help="Normalization methods to search")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                        help="Score thresholds to search")
    parser.add_argument("--min-sizes", type=int, nargs="+",
                        default=[3000, 5000, 8000, 10000, 12000, 15000],
                        help="Min region sizes (bp) to search")
    parser.add_argument("--merge-gaps", type=int, nargs="+",
                        default=[3000],
                        help="Merge gaps (bp) to search")
    parser.add_argument("--smooth-windows", type=int, nargs="+",
                        default=[5],
                        help="EWM smoothing spans to search")
    parser.add_argument("--window-sizes", nargs="+",
                        default=["2k", "4k", "8k"],
                        help="Window sizes to process")
    parser.add_argument("--no-heatmaps", action="store_true",
                        help="Skip heatmap generation")

    args = parser.parse_args()

    df = run_grid_search(
        data_dir=args.data_dir,
        gt_path=args.gt,
        output_dir=args.output_dir,
        norm_methods=args.norm_methods,
        thresholds=args.thresholds,
        min_sizes=args.min_sizes,
        merge_gaps=args.merge_gaps,
        smooth_windows=args.smooth_windows,
        window_sizes=args.window_sizes,
    )

    if not args.no_heatmaps and len(df) > 0:
        generate_heatmaps(args.output_dir)


if __name__ == "__main__":
    main()
