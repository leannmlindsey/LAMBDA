#!/usr/bin/env python3
"""
Parse BLAST results from analyze_fragment_leakage.sh and produce summary
statistics + plots for the fragment-level leakage analysis.

For each (window x class x split) cell, compute:
  - distribution of best-hit pident and alignment coverage (qcovs)
  - fraction of queries with no hit, with hit >=50%, >=70%, >=80%, >=90%, >=95%
  - same metrics for the within-train background (with self-matches excluded)

Outputs:
  - summary_stats.csv        (machine-readable)
  - summary_stats.tsv        (human-readable)
  - cumulative_pident.png    (one panel per window, lines for each split+class)
  - threshold_counts.png     (bar chart of fragments above each pident threshold)
  - high_similarity_examples.tsv (top high-similarity matches with metadata)

Usage:
  python analyze_fragment_leakage.py \
      --input-root /path/to/fragment_leakage_analysis \
      --output-dir /path/to/fragment_leakage_analysis/summary
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "qlen", "slen",
              "qcovs", "evalue", "bitscore"]
WINDOWS = ["2k", "4k", "8k"]
CLASSES = ["phage", "bacteria"]
SPLITS = ["dev", "test"]
THRESHOLDS = [50, 70, 80, 90, 95]


def parse_blast(tsv_path):
    """Load a BLAST outfmt-6 result file. Returns DataFrame or None."""
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        return None
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=BLAST_COLS)
    return df


def best_hit_per_query(df, exclude_self=False):
    """Reduce to one row per query (best by bitscore). Optionally drop self-hits."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=BLAST_COLS)
    if exclude_self:
        df = df[df["qseqid"] != df["sseqid"]].copy()
    df = df.sort_values("bitscore", ascending=False)
    return df.drop_duplicates(subset="qseqid", keep="first").copy()


def threshold_counts(best_df, n_queries):
    """For each pident threshold, return (count, fraction) of queries above it."""
    out = {}
    if n_queries == 0:
        for t in THRESHOLDS:
            out[t] = (0, 0.0)
        out["any_hit"] = (0, 0.0)
        return out

    out["any_hit"] = (len(best_df), len(best_df) / n_queries)
    for t in THRESHOLDS:
        # Combined pident >= t AND coverage >= 50%
        n = ((best_df["pident"] >= t) & (best_df["qcovs"] >= 50)).sum()
        out[t] = (n, n / n_queries)
    return out


def count_queries_in_fasta(fasta_path):
    """Count number of records in a FASTA file."""
    if not os.path.exists(fasta_path):
        return 0
    with open(fasta_path) as f:
        return sum(1 for line in f if line.startswith(">"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True,
                        help="Root dir written by analyze_fragment_leakage.sh "
                             "(contains 2k/, 4k/, 8k/ subdirs)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write summary CSV + plots")
    parser.add_argument("--top-n-examples", type=int, default=50,
                        help="How many high-similarity matches to dump for manual inspection")
    args = parser.parse_args()

    in_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []                    # for summary CSV
    all_best = {}                # (window, class, split) -> best-hit DataFrame
    n_queries_lookup = {}        # (window, class, split) -> total queries

    for window in WINDOWS:
        win_dir = in_root / window
        if not win_dir.exists():
            continue
        for cls in CLASSES:
            cls_dir = win_dir / cls
            if not cls_dir.exists():
                continue

            # dev/test BLAST results
            for split in SPLITS:
                fasta = win_dir / f"{split}_{cls}.fasta"
                blast_tsv = cls_dir / f"blast_{split}_vs_train.tsv"
                n_q = count_queries_in_fasta(fasta)
                n_queries_lookup[(window, cls, split)] = n_q

                df = parse_blast(blast_tsv)
                best = best_hit_per_query(df, exclude_self=False)
                all_best[(window, cls, split)] = best

                tc = threshold_counts(best, n_q)
                row = {
                    "window": window, "class": cls, "split": split,
                    "n_queries": n_q,
                    "n_with_any_hit": tc["any_hit"][0],
                    "frac_with_any_hit": round(tc["any_hit"][1], 4),
                    "median_best_pident": round(best["pident"].median(), 2) if len(best) else None,
                    "median_best_qcovs": round(best["qcovs"].median(), 2) if len(best) else None,
                }
                for t in THRESHOLDS:
                    row[f"n_pident_ge_{t}_cov_ge_50"] = tc[t][0]
                    row[f"frac_pident_ge_{t}_cov_ge_50"] = round(tc[t][1], 4)
                rows.append(row)

            # within-train background (exclude self-matches)
            bg_fasta = cls_dir / "bg_sample.fasta"
            bg_tsv = cls_dir / "blast_within_train.tsv"
            n_q = count_queries_in_fasta(bg_fasta)
            n_queries_lookup[(window, cls, "bg_train")] = n_q

            df = parse_blast(bg_tsv)
            best = best_hit_per_query(df, exclude_self=True)
            all_best[(window, cls, "bg_train")] = best

            tc = threshold_counts(best, n_q)
            row = {
                "window": window, "class": cls, "split": "bg_train",
                "n_queries": n_q,
                "n_with_any_hit": tc["any_hit"][0],
                "frac_with_any_hit": round(tc["any_hit"][1], 4),
                "median_best_pident": round(best["pident"].median(), 2) if len(best) else None,
                "median_best_qcovs": round(best["qcovs"].median(), 2) if len(best) else None,
            }
            for t in THRESHOLDS:
                row[f"n_pident_ge_{t}_cov_ge_50"] = tc[t][0]
                row[f"frac_pident_ge_{t}_cov_ge_50"] = round(tc[t][1], 4)
            rows.append(row)

    # ─── Write summary tables ───────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    csv_path = out_dir / "summary_stats.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    tsv_path = out_dir / "summary_stats.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False)
    print(f"Wrote {tsv_path}")

    # Pretty-print the headline columns to stdout
    print("\n=== Summary (frac of queries with best-hit pident >= threshold AND coverage >= 50%) ===")
    cols = ["window", "class", "split", "n_queries", "frac_with_any_hit",
            "frac_pident_ge_70_cov_ge_50", "frac_pident_ge_90_cov_ge_50",
            "frac_pident_ge_95_cov_ge_50"]
    print(summary[cols].to_string(index=False))

    # ─── Plot 1: cumulative distribution of best-hit pident per cell ────────
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(15, 4.5), sharey=True)
    if len(WINDOWS) == 1:
        axes = [axes]

    for ax, window in zip(axes, WINDOWS):
        for cls in CLASSES:
            for split in SPLITS + ["bg_train"]:
                key = (window, cls, split)
                if key not in all_best:
                    continue
                best = all_best[key]
                if len(best) == 0:
                    continue
                pidents = np.sort(best["pident"].values)[::-1]   # descending
                # cumulative fraction of queries with hit >= each pident
                n_q = n_queries_lookup[key]
                if n_q == 0:
                    continue
                cum_frac = np.arange(1, len(pidents) + 1) / n_q
                label = f"{cls} {split}"
                ls = "--" if split == "bg_train" else "-"
                ax.plot(pidents, cum_frac, label=label, linestyle=ls, alpha=0.85)

        ax.set_xlabel("Best-hit % identity")
        ax.set_xlim(50, 100)
        ax.set_title(f"{window} fragments")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()       # show high-identity matches at left

    axes[0].set_ylabel("Cumulative fraction of queries\nwith best-hit pident >= x")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    plt.tight_layout()
    plot_path = out_dir / "cumulative_pident.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {plot_path}")

    # ─── Plot 2: bar chart of threshold counts ──────────────────────────────
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(15, 4.5), sharey=True)
    if len(WINDOWS) == 1:
        axes = [axes]
    x_labels = [f">={t}%" for t in THRESHOLDS]

    for ax, window in zip(axes, WINDOWS):
        groups = []
        labels = []
        for cls in CLASSES:
            for split in SPLITS + ["bg_train"]:
                key = (window, cls, split)
                if key not in n_queries_lookup or n_queries_lookup[key] == 0:
                    continue
                row = summary[(summary["window"] == window) &
                              (summary["class"] == cls) &
                              (summary["split"] == split)].iloc[0]
                fracs = [row[f"frac_pident_ge_{t}_cov_ge_50"] for t in THRESHOLDS]
                groups.append(fracs)
                labels.append(f"{cls} {split}")

        if not groups:
            continue
        n_groups = len(groups)
        width = 0.8 / n_groups
        x = np.arange(len(THRESHOLDS))
        for i, (g, lbl) in enumerate(zip(groups, labels)):
            ax.bar(x + i * width - 0.4 + width / 2, g, width=width, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Best-hit pident threshold (and qcovs >= 50%)")
        ax.set_title(f"{window} fragments")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Fraction of queries above threshold")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    plt.tight_layout()
    plot_path = out_dir / "threshold_counts.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # ─── High-similarity examples (top N per cell for manual inspection) ────
    rows_examples = []
    for (window, cls, split), best in all_best.items():
        if split == "bg_train" or len(best) == 0:
            continue
        # Top by combined pident * qcovs / 10000 (ranges 0..1)
        best = best.copy()
        best["combined"] = best["pident"] * best["qcovs"] / 10000.0
        best = best.sort_values("combined", ascending=False).head(args.top_n_examples)
        for _, r in best.iterrows():
            rows_examples.append({
                "window": window, "class": cls, "split": split,
                "query": r["qseqid"], "train_hit": r["sseqid"],
                "pident": round(r["pident"], 2), "qcovs": round(r["qcovs"], 2),
                "length": int(r["length"]), "qlen": int(r["qlen"]),
                "slen": int(r["slen"]), "evalue": r["evalue"],
            })

    ex_path = out_dir / "high_similarity_examples.tsv"
    pd.DataFrame(rows_examples).to_csv(ex_path, sep="\t", index=False)
    print(f"Wrote {ex_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
