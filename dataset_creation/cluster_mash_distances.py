#!/usr/bin/env python3
"""
Cluster genomes based on MASH distances using single-linkage clustering.

Input: MASH all-vs-all distance file (tabular format from mash dist -t)
Output: Cluster membership file mapping each genome to a cluster ID
"""

import argparse
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster genomes based on MASH distances"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to MASH distance file (tabular format from mash dist -t)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output cluster membership file"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="MASH distance threshold for clustering (default: 0.05 ≈ 95%% ANI)"
    )
    parser.add_argument(
        "--stats", "-s",
        default=None,
        help="Path to output cluster statistics file (optional)"
    )
    return parser.parse_args()


class UnionFind:
    """Union-Find data structure for efficient clustering."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self):
        """Return dict mapping cluster_id -> list of members."""
        clusters = defaultdict(list)
        for x in self.parent:
            clusters[self.find(x)].append(x)
        return clusters


def main():
    args = parse_args()

    print(f"Clustering MASH distances")
    print(f"  Input: {args.input}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {args.output}")
    print()

    uf = UnionFind()
    genomes = set()
    edges_below_threshold = 0
    total_edges = 0

    print("Reading MASH distances and clustering...")

    # Read the tabular MASH output
    # Format: query\treference\tdistance\tp-value\tshared-hashes
    with open(args.input, 'r') as f:
        # Skip header line if present
        first_line = f.readline().strip()
        if not first_line.startswith('#'):
            # First line is data, process it
            parts = first_line.split('\t')
            if len(parts) >= 3:
                try:
                    genome1 = parts[0].split('/')[-1].replace('.fa', '').replace('.fasta', '').replace('.fna', '')
                    genome2 = parts[1].split('/')[-1].replace('.fa', '').replace('.fasta', '').replace('.fna', '')
                    distance = float(parts[2])

                    genomes.add(genome1)
                    genomes.add(genome2)
                    uf.find(genome1)  # Initialize in union-find
                    uf.find(genome2)

                    if genome1 != genome2 and distance <= args.threshold:
                        uf.union(genome1, genome2)
                        edges_below_threshold += 1
                    total_edges += 1
                except (ValueError, IndexError):
                    pass

        # Process rest of file
        for line_num, line in enumerate(f, start=2):
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} lines...")

            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            try:
                # Extract genome names (remove path and extension)
                genome1 = parts[0].split('/')[-1].replace('.fa', '').replace('.fasta', '').replace('.fna', '')
                genome2 = parts[1].split('/')[-1].replace('.fa', '').replace('.fasta', '').replace('.fna', '')
                distance = float(parts[2])

                genomes.add(genome1)
                genomes.add(genome2)
                uf.find(genome1)  # Initialize in union-find
                uf.find(genome2)

                if genome1 != genome2 and distance <= args.threshold:
                    uf.union(genome1, genome2)
                    edges_below_threshold += 1
                total_edges += 1

            except (ValueError, IndexError) as e:
                continue

    print(f"\nClustering complete:")
    print(f"  Total genomes: {len(genomes):,}")
    print(f"  Total pairwise comparisons: {total_edges:,}")
    print(f"  Pairs below threshold: {edges_below_threshold:,}")

    # Get clusters
    clusters = uf.get_clusters()

    # Assign numeric cluster IDs (sorted by size, largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    cluster_id_map = {rep: i for i, (rep, members) in enumerate(sorted_clusters)}

    print(f"  Number of clusters: {len(clusters):,}")

    # Calculate cluster size distribution
    sizes = [len(members) for members in clusters.values()]
    singletons = sum(1 for s in sizes if s == 1)

    print(f"  Singleton clusters: {singletons:,}")
    print(f"  Largest cluster: {max(sizes):,} genomes")
    print(f"  Mean cluster size: {sum(sizes)/len(sizes):.2f}")

    # Write cluster membership file
    print(f"\nWriting cluster membership to {args.output}...")
    with open(args.output, 'w') as f:
        f.write("genome\tcluster_id\tcluster_size\n")
        for rep, members in sorted_clusters:
            cluster_id = cluster_id_map[rep]
            cluster_size = len(members)
            for genome in sorted(members):
                f.write(f"{genome}\t{cluster_id}\t{cluster_size}\n")

    # Write statistics if requested
    if args.stats:
        print(f"Writing cluster statistics to {args.stats}...")
        with open(args.stats, 'w') as f:
            f.write("cluster_id\tsize\trepresentative\n")
            for rep, members in sorted_clusters:
                cluster_id = cluster_id_map[rep]
                f.write(f"{cluster_id}\t{len(members)}\t{rep}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
