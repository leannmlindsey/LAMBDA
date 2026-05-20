"""
Analyze predictions by COG (Clusters of Orthologous Groups) category.

This script extracts COG categories from Bakta JSON output and calculates
metrics per COG category to understand which functional categories are
being misclassified.

COG Categories:
    J: Translation, ribosomal structure and biogenesis
    A: RNA processing and modification
    K: Transcription
    L: Replication, recombination and repair
    B: Chromatin structure and dynamics
    D: Cell cycle control, cell division, chromosome partitioning
    V: Defense mechanisms
    T: Signal transduction mechanisms
    M: Cell wall/membrane/envelope biogenesis
    N: Cell motility
    U: Intracellular trafficking, secretion, and vesicular transport
    O: Posttranslational modification, protein turnover, chaperones
    C: Energy production and conversion
    G: Carbohydrate transport and metabolism
    E: Amino acid transport and metabolism
    F: Nucleotide transport and metabolism
    H: Coenzyme transport and metabolism
    I: Lipid transport and metabolism
    P: Inorganic ion transport and metabolism
    Q: Secondary metabolites biosynthesis, transport and catabolism
    R: General function prediction only
    S: Function unknown
    X: Mobilome: prophages, transposons

Usage:
    python -m src.analyze_predictions_by_cog \
        --predictions_csv /path/to/predictions.csv \
        --bakta_json /path/to/bakta.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from collections import defaultdict

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
    "-": "No COG assigned",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze predictions by COG category"
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--bakta_json",
        type=str,
        required=True,
        help="Path to Bakta JSON annotation file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as predictions)",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=None,
        help="Column name containing locus tag in predictions CSV (auto-detected if not specified)",
    )
    return parser.parse_args()


def extract_cog_from_bakta(bakta_json_path):
    """Extract locus_tag -> COG category mapping from Bakta JSON."""

    print(f"Parsing Bakta JSON: {bakta_json_path}")

    with open(bakta_json_path, 'r') as f:
        data = json.load(f)

    locus_to_cog = {}
    locus_to_product = {}
    locus_to_cog_id = {}

    features = data.get('features', [])
    print(f"  Found {len(features)} features")

    cds_count = 0
    cog_count = 0

    for feature in features:
        if feature.get('type') != 'cds':
            continue

        cds_count += 1
        locus_tag = feature.get('locus')
        product = feature.get('product', '')

        if not locus_tag:
            continue

        locus_to_product[locus_tag] = product

        # Extract COG info
        cog_category = feature.get('cog_category', '-')
        cog_id = feature.get('cog_id', '')

        if cog_category and cog_category != '-':
            cog_count += 1

        locus_to_cog[locus_tag] = cog_category if cog_category else '-'
        locus_to_cog_id[locus_tag] = cog_id

    print(f"  CDS features: {cds_count}")
    print(f"  With COG annotation: {cog_count} ({100*cog_count/cds_count:.1f}%)")

    return locus_to_cog, locus_to_product, locus_to_cog_id


def find_locus_tag_in_id(id_value, locus_tags):
    """Try to find a locus tag within an ID string."""
    # Check if any known locus tag is contained in the ID
    for locus in locus_tags:
        if locus in str(id_value):
            return locus
    return None


def calculate_metrics_by_cog(df):
    """Calculate metrics grouped by COG category."""

    results = []

    # Handle multi-letter COG categories by splitting into individual letters
    # and counting each gene once per category it belongs to
    cog_to_indices = defaultdict(list)

    for idx, row in df.iterrows():
        cog = row.get('cog_category', '-')
        if pd.isna(cog) or cog == '':
            cog = '-'

        # Split multi-letter COG categories (e.g., "JLK" -> ["J", "L", "K"])
        for letter in cog:
            cog_to_indices[letter].append(idx)

    for cog_letter, indices in sorted(cog_to_indices.items()):
        group = df.loc[indices]
        total = len(group)

        # Count by actual label
        bacteria = group[group["label"] == 0]
        phage = group[group["label"] == 1]

        n_bacteria = len(bacteria)
        n_phage = len(phage)

        # Confusion matrix
        fp = len(bacteria[bacteria["pred_label"] == 1]) if n_bacteria > 0 else 0
        tn = len(bacteria[bacteria["pred_label"] == 0]) if n_bacteria > 0 else 0
        fn = len(phage[phage["pred_label"] == 0]) if n_phage > 0 else 0
        tp = len(phage[phage["pred_label"] == 1]) if n_phage > 0 else 0

        # Rates
        fp_rate = fp / n_bacteria if n_bacteria > 0 else 0
        fn_rate = fn / n_phage if n_phage > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        # Average probability
        avg_prob_phage = group["prob_1"].mean()

        results.append({
            "cog_category": cog_letter,
            "cog_description": COG_DESCRIPTIONS.get(cog_letter, "Unknown"),
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
    print("Prediction Analysis by COG Category")
    print("=" * 80)

    # Load predictions
    print(f"\nLoading predictions: {args.predictions_csv}")
    pred_df = pd.read_csv(args.predictions_csv)
    print(f"  Total predictions: {len(pred_df)}")

    # Extract COG from Bakta JSON
    print()
    locus_to_cog, locus_to_product, locus_to_cog_id = extract_cog_from_bakta(args.bakta_json)

    # Find the ID column in predictions
    id_column = args.id_column
    if id_column is None:
        # Try to auto-detect
        for col in pred_df.columns:
            if col.lower() in ['id', 'locus', 'locus_tag', 'name', 'gene_id']:
                id_column = col
                break
        if id_column is None:
            # Use first column
            id_column = pred_df.columns[0]

    print(f"\nUsing ID column: {id_column}")

    # Match predictions to COG categories
    locus_tags = set(locus_to_cog.keys())

    matched = 0
    cog_categories = []
    cog_ids = []
    products = []

    for idx, row in pred_df.iterrows():
        id_value = row[id_column]

        # Try to find locus tag in the ID
        locus = find_locus_tag_in_id(id_value, locus_tags)

        if locus:
            matched += 1
            cog_categories.append(locus_to_cog.get(locus, '-'))
            cog_ids.append(locus_to_cog_id.get(locus, ''))
            products.append(locus_to_product.get(locus, ''))
        else:
            cog_categories.append('-')
            cog_ids.append('')
            products.append('')

    pred_df['cog_category'] = cog_categories
    pred_df['cog_id'] = cog_ids
    pred_df['product'] = products

    print(f"  Matched to COG: {matched} / {len(pred_df)} ({100*matched/len(pred_df):.1f}%)")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate metrics by COG
    print("\n" + "=" * 80)
    print("Metrics by COG Category")
    print("=" * 80)

    cog_results = calculate_metrics_by_cog(pred_df)
    cog_results = cog_results.sort_values("FP", ascending=False)

    # Save results
    output_path = os.path.join(args.output_dir, "analysis_by_cog.csv")
    cog_results.to_csv(output_path, index=False)
    print(f"\nSaved COG analysis to: {output_path}")

    # Save predictions with COG annotations
    pred_with_cog_path = os.path.join(args.output_dir, "predictions_with_cog.csv")
    pred_df.to_csv(pred_with_cog_path, index=False)
    print(f"Saved predictions with COG to: {pred_with_cog_path}")

    # Print summary table
    print("\n" + "-" * 100)
    print(f"{'COG':<4} {'Description':<55} {'Total':>6} {'FP':>5} {'FN':>5} {'FP%':>7} {'Acc':>7}")
    print("-" * 100)

    for _, row in cog_results.iterrows():
        desc = row['cog_description'][:53]
        print(f"{row['cog_category']:<4} {desc:<55} {row['total']:>6} {row['FP']:>5} {row['FN']:>5} "
              f"{row['FP_rate']:>6.1%} {row['accuracy']:>6.1%}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_fp = pred_df[(pred_df["label"] == 0) & (pred_df["pred_label"] == 1)].shape[0]
    total_fn = pred_df[(pred_df["label"] == 1) & (pred_df["pred_label"] == 0)].shape[0]
    n_bacteria = len(pred_df[pred_df["label"] == 0])
    n_phage = len(pred_df[pred_df["label"] == 1])

    print(f"\nTotal samples: {len(pred_df)}")
    print(f"  Bacteria (label=0): {n_bacteria}")
    print(f"  Phage (label=1): {n_phage}")
    print(f"\nTotal False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")

    if n_bacteria > 0:
        print(f"\nOverall FP rate: {total_fp/n_bacteria:.1%}")

    # Top COG categories contributing to FP
    print("\n" + "-" * 80)
    print("Top COG categories contributing to False Positives:")
    print("-" * 80)

    top_fp = cog_results[cog_results['FP'] > 0].head(10)
    cumulative_fp = 0
    for _, row in top_fp.iterrows():
        cumulative_fp += row['FP']
        pct_of_total = row['FP'] / total_fp * 100 if total_fp > 0 else 0
        cumulative_pct = cumulative_fp / total_fp * 100 if total_fp > 0 else 0
        print(f"  {row['cog_category']}: {row['cog_description'][:45]:<45} "
              f"FP={row['FP']:>4} ({pct_of_total:>5.1f}%, cumulative: {cumulative_pct:>5.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
