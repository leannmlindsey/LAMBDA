#!/usr/bin/env python3
"""
Extract cluster representatives from vclust and split into train/dev/test.

The first genome in each cluster is the representative.
Splitting is done by cluster to prevent data leakage.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract cluster representatives and split into train/dev/test"
    )
    parser.add_argument(
        "--clusters", "-c",
        required=True,
        help="Path to vclust_info.tsv"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
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


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading clusters from {args.clusters}...")

    # Load vclust file and extract representatives
    cluster_to_genomes = defaultdict(list)

    with open(args.clusters, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                genome = parts[0]
                cluster = int(parts[1])
                cluster_to_genomes[cluster].append(genome)

    # First genome in each cluster is the representative
    representatives = []
    cluster_ids = []

    for cluster in sorted(cluster_to_genomes.keys()):
        genomes = cluster_to_genomes[cluster]
        representatives.append(genomes[0])
        cluster_ids.append(cluster)

    print(f"  Total clusters: {len(cluster_to_genomes)}")
    print(f"  Representatives: {len(representatives)}")

    # Parse split ratio
    parts = [int(x) for x in args.split_ratio.split(':')]
    total = sum(parts)
    train_frac = parts[0] / total
    dev_frac = parts[1] / total

    # Shuffle clusters for random split
    combined = list(zip(cluster_ids, representatives))
    random.shuffle(combined)
    cluster_ids, representatives = zip(*combined)
    cluster_ids = list(cluster_ids)
    representatives = list(representatives)

    # Split by cluster count
    n = len(representatives)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)

    train_genomes = representatives[:train_end]
    train_clusters = cluster_ids[:train_end]

    dev_genomes = representatives[train_end:dev_end]
    dev_clusters = cluster_ids[train_end:dev_end]

    test_genomes = representatives[dev_end:]
    test_clusters = cluster_ids[dev_end:]

    print(f"\nSplit ({args.split_ratio}):")
    print(f"  Train: {len(train_genomes)} genomes ({len(train_clusters)} clusters)")
    print(f"  Dev: {len(dev_genomes)} genomes ({len(dev_clusters)} clusters)")
    print(f"  Test: {len(test_genomes)} genomes ({len(test_clusters)} clusters)")

    # Verify no overlap
    train_set = set(train_clusters)
    dev_set = set(dev_clusters)
    test_set = set(test_clusters)

    assert len(train_set & dev_set) == 0, "Train-Dev cluster overlap!"
    assert len(train_set & test_set) == 0, "Train-Test cluster overlap!"
    assert len(dev_set & test_set) == 0, "Dev-Test cluster overlap!"
    print("\n  No cluster overlap - no data leakage!")

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, genomes, clusters in [
        ("train", train_genomes, train_clusters),
        ("dev", dev_genomes, dev_clusters),
        ("test", test_genomes, test_clusters)
    ]:
        # Save accession list
        acc_file = output_dir / f"{split_name}_accessions.txt"
        with open(acc_file, 'w') as f:
            for g in genomes:
                f.write(f"{g}\n")

        # Save with cluster info
        cluster_file = output_dir / f"{split_name}_with_clusters.tsv"
        with open(cluster_file, 'w') as f:
            f.write("accession\tcluster\n")
            for g, c in zip(genomes, clusters):
                f.write(f"{g}\t{c}\n")

        print(f"  Saved {acc_file}")

    # Save all representatives
    all_file = output_dir / "all_representatives.txt"
    with open(all_file, 'w') as f:
        for g in train_genomes + dev_genomes + test_genomes:
            f.write(f"{g}\n")

    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Cluster Representatives Dataset\n")
        f.write(f"{'='*40}\n")
        f.write(f"Source: {args.clusters}\n")
        f.write(f"Total representatives: {len(representatives)}\n")
        f.write(f"Split ratio: {args.split_ratio}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Train: {len(train_genomes)}\n")
        f.write(f"Dev: {len(dev_genomes)}\n")
        f.write(f"Test: {len(test_genomes)}\n")
        f.write(f"\n")
        f.write(f"Cluster overlap: None (verified)\n")

    print(f"  Saved {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
