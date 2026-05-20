#!/usr/bin/env python3
"""
Select a taxonomically balanced subset of bacteria for Bacteria-Only benchmark.

This ensures representation across different phyla/classes rather than
being biased toward well-represented taxa.
"""

import argparse
import random
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select taxonomically balanced bacteria for Bacteria-Only benchmark"
    )
    parser.add_argument(
        "--accessions", "-a",
        required=True,
        help="Path to test_accessions.txt"
    )
    parser.add_argument(
        "--metadata", "-m",
        required=True,
        help="Path to GTDB bac120_metadata.tsv"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n-genomes",
        type=int,
        default=100,
        help="Number of genomes to select (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def extract_taxonomy_level(taxonomy_string, level):
    """Extract a specific taxonomy level from GTDB taxonomy string."""
    pattern = f'{level}__([^;]*)'
    match = re.search(pattern, taxonomy_string)
    return match.group(1) if match else "Unknown"


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Select Taxonomically Balanced Bacteria")
    print(f"=" * 50)

    # Load test accessions
    print(f"\nLoading test accessions from {args.accessions}...")
    with open(args.accessions, 'r') as f:
        test_accessions = set(line.strip() for line in f if line.strip())
    print(f"  Test accessions: {len(test_accessions)}")

    # Load GTDB metadata
    print(f"\nLoading GTDB metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata, sep='\t', low_memory=False)
    print(f"  Total genomes in metadata: {len(df)}")

    # Filter to test accessions only
    df_test = df[df['accession'].isin(test_accessions)].copy()
    print(f"  Matched test accessions: {len(df_test)}")

    # Extract taxonomy
    df_test['phylum'] = df_test['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'p'))
    df_test['class'] = df_test['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'c'))
    df_test['order'] = df_test['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'o'))
    df_test['family'] = df_test['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'f'))
    df_test['genus'] = df_test['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'g'))

    # Show taxonomy distribution
    print(f"\nTaxonomy distribution in test set:")
    print(f"  Phyla: {df_test['phylum'].nunique()}")
    print(f"  Classes: {df_test['class'].nunique()}")
    print(f"  Orders: {df_test['order'].nunique()}")
    print(f"  Families: {df_test['family'].nunique()}")
    print(f"  Genera: {df_test['genus'].nunique()}")

    # Strategy: Select proportionally from each phylum, then fill remaining randomly
    n_target = args.n_genomes
    phylum_counts = df_test['phylum'].value_counts()
    n_phyla = len(phylum_counts)

    print(f"\nSelection strategy:")
    print(f"  Target genomes: {n_target}")
    print(f"  Phyla available: {n_phyla}")

    # Calculate proportional selection per phylum
    total_genomes = len(df_test)
    selected = []
    selected_accessions = set()

    # First pass: proportional selection from each phylum (minimum 1 per phylum if possible)
    phylum_allocations = {}
    remaining = n_target

    for phylum, count in phylum_counts.items():
        # Proportional allocation, but at least 1 if phylum has genomes
        proportion = count / total_genomes
        allocation = max(1, int(n_target * proportion))
        allocation = min(allocation, count)  # Can't select more than available
        phylum_allocations[phylum] = allocation
        remaining -= allocation

    # Adjust if we over-allocated
    while sum(phylum_allocations.values()) > n_target:
        # Reduce from largest allocations
        max_phylum = max(phylum_allocations, key=phylum_allocations.get)
        if phylum_allocations[max_phylum] > 1:
            phylum_allocations[max_phylum] -= 1

    # Adjust if we under-allocated (add to largest phyla)
    while sum(phylum_allocations.values()) < n_target:
        for phylum in phylum_counts.index:
            if sum(phylum_allocations.values()) >= n_target:
                break
            if phylum_allocations[phylum] < phylum_counts[phylum]:
                phylum_allocations[phylum] += 1

    print(f"\n  Allocation per phylum:")
    for phylum, alloc in sorted(phylum_allocations.items(), key=lambda x: -x[1])[:15]:
        available = phylum_counts[phylum]
        print(f"    {phylum}: {alloc} (of {available} available)")
    if len(phylum_allocations) > 15:
        print(f"    ... and {len(phylum_allocations) - 15} more phyla")

    # Select genomes from each phylum
    for phylum, n_select in phylum_allocations.items():
        phylum_genomes = df_test[df_test['phylum'] == phylum]

        # Try to get diversity within phylum by selecting from different genera
        genera = phylum_genomes['genus'].unique()

        if len(genera) >= n_select:
            # Select one per genus until we have enough
            selected_genera = random.sample(list(genera), n_select)
            for genus in selected_genera:
                genus_genomes = phylum_genomes[phylum_genomes['genus'] == genus]
                chosen = genus_genomes.sample(n=1, random_state=args.seed).iloc[0]
                selected.append(chosen)
                selected_accessions.add(chosen['accession'])
        else:
            # Not enough genera, select randomly from phylum
            chosen_rows = phylum_genomes.sample(n=n_select, random_state=args.seed)
            for _, row in chosen_rows.iterrows():
                selected.append(row)
                selected_accessions.add(row['accession'])

    selected_df = pd.DataFrame(selected)
    print(f"\n  Selected: {len(selected_df)} genomes")
    print(f"  Phyla represented: {selected_df['phylum'].nunique()}")
    print(f"  Genera represented: {selected_df['genus'].nunique()}")

    # Save outputs
    # Accession list
    acc_file = output_dir / "bacteria_only_100_accessions.txt"
    with open(acc_file, 'w') as f:
        for acc in selected_df['accession']:
            f.write(f"{acc}\n")
    print(f"\n  Saved: {acc_file}")

    # Metadata
    meta_file = output_dir / "bacteria_only_100_metadata.tsv"
    selected_df[['accession', 'phylum', 'class', 'order', 'family', 'genus',
                 'gtdb_taxonomy']].to_csv(meta_file, sep='\t', index=False)
    print(f"  Saved: {meta_file}")

    # Summary
    summary_file = output_dir / "bacteria_only_100_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Taxonomically Balanced Bacteria Selection\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Source: {args.accessions}\n")
        f.write(f"Target: {args.n_genomes} genomes\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Selected: {len(selected_df)} genomes\n")
        f.write(f"Phyla represented: {selected_df['phylum'].nunique()}\n")
        f.write(f"Genera represented: {selected_df['genus'].nunique()}\n")
        f.write(f"\n")
        f.write(f"Phylum distribution:\n")
        for phylum, count in selected_df['phylum'].value_counts().items():
            f.write(f"  {phylum}: {count}\n")
    print(f"  Saved: {summary_file}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
