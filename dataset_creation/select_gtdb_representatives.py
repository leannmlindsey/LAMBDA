#!/usr/bin/env python3
"""
Select GTDB bacterial genomes for LAMBDA benchmark.

Strategy:
1. Filter for high-quality representatives
2. Randomly select N unique genera
3. Pick 1 genome per selected genus
4. Split into train/dev/test by genus (cluster-aware)
"""

import argparse
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select GTDB bacterial genomes with taxonomic subsampling"
    )
    parser.add_argument(
        "--metadata", "-m",
        required=True,
        help="Path to bac120_metadata.tsv"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n-genera",
        type=int,
        default=None,
        help="Number of genera to select (default: all available)"
    )
    parser.add_argument(
        "--min-completeness",
        type=float,
        default=95.0,
        help="Minimum CheckM2 completeness (default: 95.0)"
    )
    parser.add_argument(
        "--max-contamination",
        type=float,
        default=5.0,
        help="Maximum CheckM2 contamination (default: 5.0)"
    )
    parser.add_argument(
        "--split-ratio",
        default="80:10:10",
        help="Train:dev:test split ratio (default: 80:10:10)"
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
    # Format: d__Bacteria;p__Phylum;c__Class;o__Order;f__Family;g__Genus;s__Species
    pattern = f'{level}__([^;]*)'
    match = re.search(pattern, taxonomy_string)
    return match.group(1) if match else "Unknown"


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GTDB metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata, sep='\t', low_memory=False)
    print(f"  Total genomes: {len(df):,}")

    # Filter for representatives (accession == gtdb_genome_representative)
    df_reps = df[df['accession'] == df['gtdb_genome_representative']].copy()
    print(f"  Representatives: {len(df_reps):,}")

    # Filter for quality
    df_hq = df_reps[
        (df_reps['checkm2_completeness'] >= args.min_completeness) &
        (df_reps['checkm2_contamination'] <= args.max_contamination)
    ].copy()
    print(f"  High-quality reps (≥{args.min_completeness}% complete, ≤{args.max_contamination}% contam): {len(df_hq):,}")

    # Extract genus from taxonomy
    df_hq['genus'] = df_hq['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'g'))
    df_hq['family'] = df_hq['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'f'))
    df_hq['order'] = df_hq['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'o'))
    df_hq['phylum'] = df_hq['gtdb_taxonomy'].apply(lambda x: extract_taxonomy_level(x, 'p'))

    # Get unique genera
    unique_genera = df_hq['genus'].unique().tolist()
    print(f"  Unique genera: {len(unique_genera):,}")

    # Select genera (all or subsample)
    if args.n_genera is None or args.n_genera >= len(unique_genera):
        selected_genera = unique_genera
        print(f"\n  Using all {len(selected_genera)} genera")
    else:
        selected_genera = random.sample(unique_genera, args.n_genera)
        print(f"\n  Randomly selected {len(selected_genera)} genera")

    # Get one genome per selected genus
    selected_genomes = []
    for genus in selected_genera:
        genus_genomes = df_hq[df_hq['genus'] == genus]
        # Pick one randomly
        selected = genus_genomes.sample(n=1, random_state=args.seed).iloc[0]
        selected_genomes.append(selected)

    selected_df = pd.DataFrame(selected_genomes)
    print(f"  Selected {len(selected_df)} genomes")

    # Split by genus (cluster-aware - genera are our clusters)
    parts = [int(x) for x in args.split_ratio.split(':')]
    total = sum(parts)
    train_frac = parts[0] / total
    dev_frac = parts[1] / total

    # Shuffle genera for random split
    genera_list = selected_df['genus'].tolist()
    random.shuffle(genera_list)

    n = len(genera_list)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)

    train_genera = set(genera_list[:train_end])
    dev_genera = set(genera_list[train_end:dev_end])
    test_genera = set(genera_list[dev_end:])

    train_df = selected_df[selected_df['genus'].isin(train_genera)]
    dev_df = selected_df[selected_df['genus'].isin(dev_genera)]
    test_df = selected_df[selected_df['genus'].isin(test_genera)]

    print(f"\nSplit ({args.split_ratio}):")
    print(f"  Train: {len(train_df)} genomes ({len(train_genera)} genera)")
    print(f"  Dev: {len(dev_df)} genomes ({len(dev_genera)} genera)")
    print(f"  Test: {len(test_df)} genomes ({len(test_genera)} genera)")

    # Verify no overlap
    assert len(train_genera & dev_genera) == 0, "Train-Dev genus overlap!"
    assert len(train_genera & test_genera) == 0, "Train-Test genus overlap!"
    assert len(dev_genera & test_genera) == 0, "Dev-Test genus overlap!"
    print("\n  No genus overlap - no data leakage!")

    # Save outputs
    for split_name, split_df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        # Save accession list
        acc_file = output_dir / f"{split_name}_accessions.txt"
        split_df['accession'].to_csv(acc_file, index=False, header=False)
        print(f"  Saved {acc_file}")

        # Save with metadata
        meta_file = output_dir / f"{split_name}_metadata.tsv"
        split_df[['accession', 'genus', 'family', 'order', 'phylum',
                  'gtdb_taxonomy', 'genome_size', 'checkm2_completeness',
                  'checkm2_contamination']].to_csv(meta_file, sep='\t', index=False)

    # Save all selected
    all_file = output_dir / "all_accessions.txt"
    selected_df['accession'].to_csv(all_file, index=False, header=False)

    # Save full metadata
    full_meta_file = output_dir / "all_metadata.tsv"
    selected_df.to_csv(full_meta_file, sep='\t', index=False)

    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GTDB Bacterial Dataset Selection\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source: {args.metadata}\n")
        f.write(f"Quality filters: completeness >= {args.min_completeness}%, contamination <= {args.max_contamination}%\n")
        f.write(f"Target genera: {args.n_genera}\n")
        f.write(f"Selected genera: {len(selected_genera)}\n")
        f.write(f"Selected genomes: {len(selected_df)}\n")
        f.write(f"Split ratio: {args.split_ratio}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Train: {len(train_df)} genomes\n")
        f.write(f"Dev: {len(dev_df)} genomes\n")
        f.write(f"Test: {len(test_df)} genomes\n")
        f.write(f"\n")
        f.write(f"Genus overlap: None (verified)\n")

    # Print taxonomy distribution
    print(f"\nPhylum distribution (top 10):")
    phylum_counts = selected_df['phylum'].value_counts().head(10)
    for phylum, count in phylum_counts.items():
        print(f"  {phylum}: {count}")

    print(f"\n  Summary saved to {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
