"""
Analyze predictions by annotation category.

This script groups predictions by gene annotation and calculates metrics
per category to understand which types of genes are being misclassified.

Usage:
    python -m src.analyze_predictions_by_category \
        --predictions_csv /path/to/predictions.csv \
        --output_dir /path/to/output

The predictions CSV should have columns: label, annotation, pred_label, prob_0, prob_1
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze predictions by annotation category"
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum count to include category in output (default: 5)",
    )
    parser.add_argument(
        "--use_broad_categories",
        action="store_true",
        help="Group annotations into broader categories",
    )
    return parser.parse_args()


def get_broad_category(annotation):
    """Map detailed annotations to broader categories."""
    if pd.isna(annotation) or annotation == "":
        return "unknown"

    annotation_lower = annotation.lower()

    # Hypothetical/unknown
    if "hypothetical" in annotation_lower or "uncharacterized" in annotation_lower:
        return "hypothetical protein"

    # Ribosomal proteins
    if "ribosom" in annotation_lower:
        return "ribosomal protein"

    # tRNA related
    if "trna" in annotation_lower or "t-rna" in annotation_lower:
        return "tRNA related"

    # DNA replication/repair
    if any(x in annotation_lower for x in ["dna polymerase", "helicase", "primase", "gyrase", "topoisomerase", "recombinase", "integrase", "transposase"]):
        return "DNA replication/recombination"

    # Transcription
    if any(x in annotation_lower for x in ["rna polymerase", "sigma factor", "transcription"]):
        return "transcription"

    # Translation
    if any(x in annotation_lower for x in ["translation", "elongation factor", "initiation factor"]):
        return "translation"

    # Membrane/transport
    if any(x in annotation_lower for x in ["transporter", "permease", "channel", "pump", "membrane", "porin"]):
        return "membrane/transport"

    # Cell wall/envelope
    if any(x in annotation_lower for x in ["cell wall", "peptidoglycan", "lipopolysaccharide", "lps"]):
        return "cell wall/envelope"

    # Metabolism - energy
    if any(x in annotation_lower for x in ["atp synthase", "cytochrome", "nadh", "electron transport"]):
        return "energy metabolism"

    # Metabolism - amino acids
    if any(x in annotation_lower for x in ["aminotransferase", "synthase", "synthetase", "dehydrogenase", "kinase", "phosphatase", "reductase", "oxidase"]):
        return "metabolic enzyme"

    # Regulatory
    if any(x in annotation_lower for x in ["regulator", "repressor", "activator", "response regulator", "sensor"]):
        return "regulatory"

    # Stress response
    if any(x in annotation_lower for x in ["chaperone", "heat shock", "cold shock", "stress"]):
        return "stress response"

    # Phage-related keywords (interesting for this analysis!)
    if any(x in annotation_lower for x in ["phage", "capsid", "tail", "portal", "terminase", "holin", "lysin", "lysozyme"]):
        return "phage-related"

    # Mobile elements
    if any(x in annotation_lower for x in ["transpos", "insertion", "mobile element", "is element"]):
        return "mobile genetic element"

    # Toxin/antitoxin
    if any(x in annotation_lower for x in ["toxin", "antitoxin"]):
        return "toxin-antitoxin"

    # Secretion
    if any(x in annotation_lower for x in ["secretion", "type ii", "type iii", "type iv", "type vi"]):
        return "secretion system"

    return "other"


def analyze_by_category(df, category_col="annotation", use_broad=False):
    """Analyze predictions grouped by category."""

    if use_broad:
        df = df.copy()
        df["category"] = df[category_col].apply(get_broad_category)
        group_col = "category"
    else:
        group_col = category_col

    results = []

    for category, group in df.groupby(group_col):
        total = len(group)

        # For bacteria (label=0): FP means predicted as phage (pred_label=1)
        # For phage (label=1): FN means predicted as bacteria (pred_label=0)

        # Count by actual label
        bacteria = group[group["label"] == 0]
        phage = group[group["label"] == 1]

        n_bacteria = len(bacteria)
        n_phage = len(phage)

        # False positives (bacteria predicted as phage)
        fp = len(bacteria[bacteria["pred_label"] == 1]) if n_bacteria > 0 else 0
        # True negatives (bacteria correctly predicted)
        tn = len(bacteria[bacteria["pred_label"] == 0]) if n_bacteria > 0 else 0

        # False negatives (phage predicted as bacteria)
        fn = len(phage[phage["pred_label"] == 0]) if n_phage > 0 else 0
        # True positives (phage correctly predicted)
        tp = len(phage[phage["pred_label"] == 1]) if n_phage > 0 else 0

        # Rates
        fp_rate = fp / n_bacteria if n_bacteria > 0 else 0
        fn_rate = fn / n_phage if n_phage > 0 else 0

        # Overall accuracy for this category
        correct = tp + tn
        accuracy = correct / total if total > 0 else 0

        # Average probabilities
        avg_prob_phage = group["prob_1"].mean()

        results.append({
            "category": category,
            "total": total,
            "n_bacteria": n_bacteria,
            "n_phage": n_phage,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "FP_rate": fp_rate,
            "FN_rate": fn_rate,
            "accuracy": accuracy,
            "avg_prob_phage": avg_prob_phage,
        })

    return pd.DataFrame(results)


def main():
    args = parse_arguments()

    print("=" * 70)
    print("Prediction Analysis by Category")
    print("=" * 70)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    print(f"  Total samples: {len(df)}")

    # Check required columns
    required_cols = ["label", "pred_label", "prob_1"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Find annotation column
    annotation_col = None
    for col in ["annotation", "product", "description", "function"]:
        if col in df.columns:
            annotation_col = col
            break

    if annotation_col is None:
        print("  WARNING: No annotation column found. Using index as category.")
        df["annotation"] = "unknown"
        annotation_col = "annotation"
    else:
        print(f"  Using annotation column: {annotation_col}")

    # Overall stats
    n_bacteria = len(df[df["label"] == 0])
    n_phage = len(df[df["label"] == 1])
    print(f"  Bacteria (label=0): {n_bacteria}")
    print(f"  Phage (label=1): {n_phage}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze by detailed annotation
    print("\n" + "=" * 70)
    print("Analysis by Detailed Annotation")
    print("=" * 70)

    detailed_results = analyze_by_category(df, annotation_col, use_broad=False)
    detailed_results = detailed_results[detailed_results["total"] >= args.min_count]
    detailed_results = detailed_results.sort_values("FP", ascending=False)

    # Save detailed results
    detailed_path = os.path.join(args.output_dir, "analysis_by_annotation.csv")
    detailed_results.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed analysis to: {detailed_path}")

    # Show top FP categories
    print("\nTop 20 annotations with most False Positives (bacteria predicted as phage):")
    print("-" * 70)
    top_fp = detailed_results.head(20)
    for _, row in top_fp.iterrows():
        print(f"  {row['category'][:50]:<50} FP={row['FP']:>4}  FP_rate={row['FP_rate']:.1%}  (n={row['total']})")

    # Analyze by broad category
    if args.use_broad_categories or True:  # Always do broad analysis
        print("\n" + "=" * 70)
        print("Analysis by Broad Category")
        print("=" * 70)

        broad_results = analyze_by_category(df, annotation_col, use_broad=True)
        broad_results = broad_results.sort_values("FP", ascending=False)

        # Save broad results
        broad_path = os.path.join(args.output_dir, "analysis_by_broad_category.csv")
        broad_results.to_csv(broad_path, index=False)
        print(f"\nSaved broad category analysis to: {broad_path}")

        # Print broad category summary
        print("\nBroad Category Summary:")
        print("-" * 70)
        print(f"{'Category':<30} {'Total':>7} {'Bact':>6} {'Phage':>6} {'FP':>5} {'FN':>5} {'FP%':>7} {'Acc':>7}")
        print("-" * 70)
        for _, row in broad_results.iterrows():
            print(f"{row['category']:<30} {row['total']:>7} {row['n_bacteria']:>6} {row['n_phage']:>6} "
                  f"{row['FP']:>5} {row['FN']:>5} {row['FP_rate']:>6.1%} {row['accuracy']:>6.1%}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    total_fp = df[(df["label"] == 0) & (df["pred_label"] == 1)].shape[0]
    total_fn = df[(df["label"] == 1) & (df["pred_label"] == 0)].shape[0]

    print(f"\nTotal False Positives (bacteria → phage): {total_fp}")
    print(f"Total False Negatives (phage → bacteria): {total_fn}")

    if n_bacteria > 0:
        print(f"\nOverall FP rate (bacteria only): {total_fp/n_bacteria:.1%}")
    if n_phage > 0:
        print(f"Overall FN rate (phage only): {total_fn/n_phage:.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
