#!/usr/bin/env python3
"""
compare_baselines_vs_glms.py

Build the genome-wide comparison table + per-genome MCC distribution plot that
puts the simple sequence-feature baselines next to the gLMs and the traditional
prophage detection tools.

Inputs:
  - baseline_pooled_genome_wide.csv  (from compute_pooled_genome_wide.py)
  - baseline_per_genome_mcc.csv      (from compute_pooled_genome_wide.py)
  - grid_search_best.csv             (gLM pooled metrics, from CLAUDE_LAMBDA_VISUALIZE_RESULTS)
  - all_models_summary.csv           (traditional tools + pLM, from CLAUDE_LAMBDA_VISUALIZE_RESULTS)
  - per-genome gLM metrics: DATA/per_segment_<window>/<MODEL>/<MODEL>_<window>_individual.csv
                                   (for the per-genome MCC distribution plot)

Outputs:
  outputs/results/aggregated/genome_wide_full_comparison.csv
  outputs/results/figures/per_genome_mcc_distribution.{pdf,png}

The comparison table is the cleaned, paper-ready version: one row per model
x window (or per tool, for traditional tools that are window-independent), with
pooled MCC / precision / recall / specificity / F1 / AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHAGE_COLOR = "#f35f73"
BACTERIA_COLOR = "#02afbd"
BASELINE_COLORS = {
    "baseline_gc": "#8E8E8E",
    "baseline_kmer4": "#C66E59",
    "baseline_kmer6": "#9C2B17",
}
GLM_COLOR = "#3a4dfd"


def load_baseline_pooled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"baseline": "model"})
    df["category"] = "baseline"
    return df


def load_glm_grid(path: Path) -> pd.DataFrame:
    """grid_search_best.csv (one row per gLM x window).

    The 'micro_mcc' column is the pooled (segment-level) MCC across all
    genomes — that's what we want for the comparison.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={
        "window_size": "window",
        "n_genomes": "n_genomes",
        "macro_mcc": "macro_mcc",
        "macro_precision": "precision",
        "macro_recall": "recall",
        "macro_f1": "f1",
        "micro_mcc": "mcc",
        "total_tp": "tp",
        "total_tn": "tn",
        "total_fp": "fp",
        "total_fn": "fn",
    })
    df["specificity"] = df["tn"] / (df["tn"] + df["fp"])
    df["category"] = "gLM"
    return df[["model", "window", "n_genomes", "tp", "tn", "fp", "fn",
               "mcc", "macro_mcc", "precision", "recall", "specificity",
               "f1", "category"]]


def load_traditional(path: Path) -> pd.DataFrame:
    """all_models_summary.csv — traditional tools + PIDE at 2k only."""
    df = pd.read_csv(path)
    # Keep only non-gLM rows (gLM rows come from grid_search_best.csv at all windows)
    df = df[df["category"].isin(["Traditional", "pLM"])].copy()
    df["window"] = "2k"  # The traditional comparison was done at 2k stride
    df = df.rename(columns={
        "TP": "tp", "TN": "tn", "FP": "fp", "FN": "fn",
        "MCC": "mcc", "F1": "f1", "num_genomes": "n_genomes",
    })
    return df[["model", "window", "n_genomes", "tp", "tn", "fp", "fn",
               "mcc", "precision", "recall", "specificity", "f1", "category"]]


def build_combined(baseline_path: Path, grid_path: Path, all_models_path: Path) -> pd.DataFrame:
    base = load_baseline_pooled(baseline_path)
    glm = load_glm_grid(grid_path)
    trad = load_traditional(all_models_path)

    keep = ["model", "category", "window", "n_genomes", "tp", "tn", "fp", "fn",
            "mcc", "precision", "recall", "specificity", "f1"]
    base = base[[c for c in keep if c in base.columns] + (["auc"] if "auc" in base.columns else [])]
    glm = glm[[c for c in keep if c in glm.columns] + (["macro_mcc"] if "macro_mcc" in glm.columns else [])]
    trad = trad[[c for c in keep if c in trad.columns]]

    combined = pd.concat([base, glm, trad], ignore_index=True, sort=False)
    # Sort: baselines first (by window), then gLMs (by window then MCC desc), then traditional
    cat_order = {"baseline": 0, "gLM": 1, "Traditional": 2, "pLM": 3}
    combined["_cat_order"] = combined["category"].map(cat_order)
    combined["_window_order"] = combined["window"].map({"2k": 0, "4k": 1, "8k": 2, "50k": 3}).fillna(9)
    combined = combined.sort_values(["_cat_order", "_window_order", "mcc"],
                                     ascending=[True, True, False])
    combined = combined.drop(columns=["_cat_order", "_window_order"])
    return combined


def mcc_from_df(df: pd.DataFrame) -> float:
    """Per-genome MCC from a single per-genome predictions CSV."""
    if not {"label", "pred_label"}.issubset(df.columns):
        return float("nan")
    y = df["label"].astype(int).to_numpy()
    p = df["pred_label"].astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return num / den if den else 0.0


def glm_per_genome_mccs(model_dir: Path, prefer_variant: str | None = None) -> tuple[str, np.ndarray]:
    """Compute per-genome MCC for one gLM directory by reading *_predictions.csv.

    Some models have multiple variants (e.g., EVO2 has _lp_predictions.csv,
    _nn_predictions.csv, _sae_results.csv). `prefer_variant` selects which
    suffix to use; falls back to plain *_predictions.csv.
    """
    candidates: list[Path] = []
    if prefer_variant:
        # EVO2_SAE per-genome files are *_sae_results.csv, not *_sae_predictions.csv
        candidates = sorted(model_dir.glob(f"*_{prefer_variant}_predictions.csv"))
        if not candidates:
            candidates = sorted(model_dir.glob(f"*_{prefer_variant}_results.csv"))
    if not candidates:
        # match files like GC*_*_predictions.csv or NC_*_predictions.csv, but
        # exclude rollup files (DNABERT2_2k_individual.csv, *_summary.csv, etc.)
        candidates = [p for p in sorted(model_dir.glob("*_predictions.csv"))
                      if "_individual" not in p.name
                      and "_summary" not in p.name
                      and "phage_predictions" not in p.name]
        # strip out variant files if any (we want a single, canonical set)
        canonical = [p for p in candidates
                     if not any(v in p.name for v in ["_lp_", "_nn_", "_sae_"])]
        if canonical:
            candidates = canonical
    mccs: list[float] = []
    for p in candidates:
        try:
            df = pd.read_csv(p, usecols=["label", "pred_label"])
        except (ValueError, KeyError):
            continue
        mccs.append(mcc_from_df(df))
    return model_dir.name, np.array(mccs, dtype=float)


def make_distribution_plot(
    baseline_per_genome: Path,
    glm_individual_dir: Path,
    out_path: Path,
    window: str = "2k",
) -> None:
    """Per-genome MCC distribution: baselines (boxes) vs each gLM (boxes).

    For each gLM, per-genome MCC is computed directly from the per-genome
    *_predictions.csv files in CLAUDE_LAMBDA_VISUALIZE_RESULTS/DATA/per_segment_<window>/<MODEL>/.
    """
    base_df = pd.read_csv(baseline_per_genome)
    base_df = base_df[base_df["window"] == window]

    # EVO2 has lp / nn / sae variants. Pick "nn" (the strongest) for the
    # main panel; the other variants can be added to a supplementary plot.
    variant_choice = {"EVO2": "nn", "EVO2_LP": "lp", "EVO2_SAE": "sae"}

    glm_box_data: dict[str, np.ndarray] = {}
    if glm_individual_dir.is_dir():
        for model_dir in sorted(glm_individual_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            name, vals = glm_per_genome_mccs(
                model_dir,
                prefer_variant=variant_choice.get(model_dir.name),
            )
            if len(vals):
                glm_box_data[name] = vals
                print(f"  [glm] {name}: {len(vals)} genomes, median MCC = {np.median(vals):.3f}")

    # Build ordered box list: GC, k=4, k=6, then each gLM by median MCC
    fig, ax = plt.subplots(figsize=(11, 5.5))
    box_labels = []
    box_values = []
    box_colors = []

    for baseline in ["baseline_gc", "baseline_kmer4", "baseline_kmer6"]:
        vals = base_df[base_df["baseline"] == baseline]["mcc"].dropna().to_numpy()
        if len(vals):
            box_labels.append({"baseline_gc": "GC", "baseline_kmer4": "k=4", "baseline_kmer6": "k=6"}[baseline])
            box_values.append(vals)
            box_colors.append(BASELINE_COLORS[baseline])

    glm_sorted = sorted(glm_box_data.items(), key=lambda kv: np.median(kv[1]))
    for name, vals in glm_sorted:
        box_labels.append(name)
        box_values.append(vals)
        box_colors.append(GLM_COLOR)

    if not box_values:
        print(f"[warn] no per-genome data for {window}, skipping plot")
        plt.close(fig)
        return

    bp = ax.boxplot(box_values, labels=box_labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor(c)

    ax.set_ylabel("Per-genome MCC")
    ax.set_title(f"Genome-wide prophage detection — per-genome MCC distribution ({window})")
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0.0, color="#888", linewidth=0.8, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  (+ .png)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_pooled", required=True)
    p.add_argument("--baseline_per_genome", required=True)
    p.add_argument("--glm_grid", required=True)
    p.add_argument("--all_models", required=True)
    p.add_argument("--glm_per_genome_dir", required=True,
                   help="DATA/per_segment_2k (or _4k, _8k) — directory of per-model dirs")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined(
        Path(args.baseline_pooled),
        Path(args.glm_grid),
        Path(args.all_models),
    )
    out_csv = out_dir / "genome_wide_full_comparison.csv"
    combined.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"wrote {out_csv}  ({len(combined)} rows)")

    print("\nGenome-wide comparison (pooled MCC, sorted within category):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(combined[["model", "category", "window", "mcc", "precision",
                        "recall", "specificity", "f1"]].to_string(index=False))

    # Per-genome MCC distribution plot at 2k (matches the gLM individual files)
    make_distribution_plot(
        Path(args.baseline_per_genome),
        Path(args.glm_per_genome_dir),
        figs_dir / "per_genome_mcc_distribution_2k.pdf",
        window="2k",
    )


if __name__ == "__main__":
    main()
