"""
Calculate prediction metrics grouped by category column.

Usage:
    python -m src.metrics_by_category \
        --input_csv /path/to/predictions_with_cog.csv \
        --category_column category \
        --output_csv /path/to/metrics_by_category.csv
"""

import argparse
import os

import pandas as pd
import numpy as np


COG_DESCRIPTIONS = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure",
    "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "Z": "Cytoskeleton",
    "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, vesicular transport",
    "O": "Posttranslational modification, protein turnover, chaperones",
    "C": "Energy production and conversion",
    "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport, catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
    "X": "Mobilome: prophages, transposons",
    "hypothetical_protein": "Hypothetical protein (no COG)",
    "OTHER": "Other (no COG, not hypothetical)",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate metrics by category"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to predictions CSV with category column",
    )
    parser.add_argument(
        "--category_column",
        type=str,
        default="category",
        help="Name of category column (default: category)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV (default: input with _metrics suffix)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="FP",
        choices=["FP", "FN", "FP_rate", "accuracy", "total"],
        help="Column to sort by (default: FP)",
    )
    return parser.parse_args()


def calculate_metrics(df, category_column):
    """Calculate metrics for each category."""

    results = []

    for category, group in df.groupby(category_column):
        total = len(group)

        # Count by label
        bacteria = group[group["label"] == 0]
        phage = group[group["label"] == 1]

        n_bacteria = len(bacteria)
        n_phage = len(phage)

        # Confusion matrix
        tp = len(phage[phage["pred_label"] == 1]) if n_phage > 0 else 0
        tn = len(bacteria[bacteria["pred_label"] == 0]) if n_bacteria > 0 else 0
        fp = len(bacteria[bacteria["pred_label"] == 1]) if n_bacteria > 0 else 0
        fn = len(phage[phage["pred_label"] == 0]) if n_phage > 0 else 0

        # Rates
        fp_rate = fp / n_bacteria if n_bacteria > 0 else 0
        fn_rate = fn / n_phage if n_phage > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        # Average probabilities
        avg_prob_phage = group["prob_1"].mean()

        results.append({
            "category": category,
            "description": COG_DESCRIPTIONS.get(category, ""),
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

    print("=" * 80)
    print("Metrics by Category")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  Rows: {len(df)}")

    # Check required columns
    required = ["label", "pred_label", "prob_1", args.category_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Calculate metrics
    results = calculate_metrics(df, args.category_column)
    results = results.sort_values(args.sort_by, ascending=False)

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_metrics_by_category{ext}"

    # Save
    results.to_csv(args.output_csv, index=False)
    print(f"\nSaved to: {args.output_csv}")

    # Print table
    print("\n" + "-" * 100)
    print(f"{'Category':<25} {'Description':<40} {'Total':>6} {'FP':>5} {'FN':>5} {'FP%':>7} {'Acc':>7}")
    print("-" * 100)

    for _, row in results.iterrows():
        cat = str(row['category'])[:23]
        desc = str(row['description'])[:38]
        print(f"{cat:<25} {desc:<40} {row['total']:>6} {row['FP']:>5} {row['FN']:>5} "
              f"{row['FP_rate']:>6.1%} {row['accuracy']:>6.1%}")

    # Summary
    print("\n" + "-" * 100)

    total_fp = results['FP'].sum()
    total_fn = results['FN'].sum()

    print(f"\nTotal FP: {total_fp}")
    print(f"Total FN: {total_fn}")

    # Top contributors to FP
    if total_fp > 0:
        print("\nTop categories contributing to FP:")
        top = results[results['FP'] > 0].head(5)
        cumulative = 0
        for _, row in top.iterrows():
            cumulative += row['FP']
            pct = 100 * row['FP'] / total_fp
            cum_pct = 100 * cumulative / total_fp
            print(f"  {row['category']}: {row['FP']} ({pct:.1f}%, cumulative: {cum_pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
