#!/usr/bin/env python3
"""
Analyze factors that correlate with SAE prophage detection performance.

Compares high vs low performing genomes by:
1. GC content
2. Taxonomy (organism name)
3. Publication source
4. Genome size
5. Number of prophage regions
6. Average prophage size
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_ground_truth(csv_path):
    """Load ground truth with all metadata."""
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['Assembly']
            if assembly not in gt:
                gt[assembly] = {
                    'regions': [],
                    'organism': row.get('Organism Name', ''),
                    'ncbi_id': row.get('NCBI Id', ''),
                    'publication': row.get('Publication', row.get('Source', '')),
                }
            gt[assembly]['regions'].append({
                'start': int(row['start']),
                'end': int(row['end']),
            })
    return gt


def load_fasta_stats(fasta_path):
    """Calculate GC content and length from FASTA."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_name:
            sequences[current_name] = ''.join(current_seq)

    if not sequences:
        return None, None

    # Use first/main sequence
    seq = list(sequences.values())[0]
    length = len(seq)

    # Calculate GC content
    gc_count = seq.count('G') + seq.count('C')
    gc_content = gc_count / length if length > 0 else 0

    return length, gc_content


def get_assembly_id(fasta_filename):
    """Extract assembly ID from FASTA filename."""
    name = Path(fasta_filename).stem
    if name.startswith('NC_'):
        return name
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] in ['GCF', 'GCA']:
        return f"{parts[0]}_{parts[1]}"
    return name


def load_clustering_results(results_file):
    """Load clustering results JSON."""
    with open(results_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze performance factors")
    parser.add_argument("--clustering_results", type=str, required=True,
                        help="Path to clustering_results.json")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Ground truth CSV file")
    parser.add_argument("--fasta_dir", type=str, required=True,
                        help="Directory containing FASTA files")
    parser.add_argument("--output_dir", type=str, default="./performance_analysis",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analyze Performance Factors")
    print("=" * 60)

    # Load data
    print("\nLoading clustering results...")
    results = load_clustering_results(args.clustering_results)
    print(f"  Loaded {len(results)} genomes")

    print("\nLoading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  Loaded {len(gt)} assemblies with metadata")

    # Print CSV columns to see what's available
    with open(args.ground_truth, 'r') as f:
        reader = csv.DictReader(f)
        print(f"  CSV columns: {reader.fieldnames}")

    # Find FASTA files
    fasta_dir = Path(args.fasta_dir)
    fasta_files = {get_assembly_id(f): f for f in fasta_dir.glob("*.fna")}
    fasta_files.update({get_assembly_id(f): f for f in fasta_dir.glob("*.fasta")})
    print(f"\nFound {len(fasta_files)} FASTA files")

    # Bin results by F1 score
    high_acc = [r for r in results if r['simple_metrics']['f1'] >= 0.7]
    med_acc = [r for r in results if 0.3 <= r['simple_metrics']['f1'] < 0.7]
    low_acc = [r for r in results if r['simple_metrics']['f1'] < 0.3]

    print(f"\nAccuracy bins:")
    print(f"  High (F1 >= 0.7):      {len(high_acc)} genomes")
    print(f"  Medium (0.3 <= F1 < 0.7): {len(med_acc)} genomes")
    print(f"  Low (F1 < 0.3):        {len(low_acc)} genomes")

    # Collect stats for each genome
    genome_stats = []

    for r in results:
        assembly = r['assembly']
        m = r['simple_metrics']

        stats = {
            'assembly': assembly,
            'f1': m['f1'],
            'precision': m['precision'],
            'recall': m['recall'],
            'mcc': m['mcc'],
            'bin': 'high' if m['f1'] >= 0.7 else ('medium' if m['f1'] >= 0.3 else 'low'),
        }

        # Get ground truth metadata
        gt_info = gt.get(assembly, {})
        stats['organism'] = gt_info.get('organism', 'Unknown')
        stats['publication'] = gt_info.get('publication', 'Unknown')
        stats['num_gt_regions'] = len(gt_info.get('regions', []))

        if gt_info.get('regions'):
            sizes = [reg['end'] - reg['start'] for reg in gt_info['regions']]
            stats['avg_prophage_size'] = np.mean(sizes)
            stats['total_prophage_bp'] = sum(sizes)
        else:
            stats['avg_prophage_size'] = 0
            stats['total_prophage_bp'] = 0

        # Get FASTA stats
        if assembly in fasta_files:
            length, gc = load_fasta_stats(fasta_files[assembly])
            stats['genome_length'] = length
            stats['gc_content'] = gc
        else:
            stats['genome_length'] = None
            stats['gc_content'] = None

        genome_stats.append(stats)

    # Analysis by bin
    print("\n" + "=" * 60)
    print("ANALYSIS BY ACCURACY BIN")
    print("=" * 60)

    for bin_name, bin_data in [('high', high_acc), ('medium', med_acc), ('low', low_acc)]:
        bin_stats = [s for s in genome_stats if s['bin'] == bin_name]
        if not bin_stats:
            continue

        print(f"\n--- {bin_name.upper()} ACCURACY ({len(bin_stats)} genomes) ---")

        # GC content
        gc_values = [s['gc_content'] for s in bin_stats if s['gc_content'] is not None]
        if gc_values:
            print(f"\nGC Content:")
            print(f"  Mean: {np.mean(gc_values):.1%}")
            print(f"  Std:  {np.std(gc_values):.1%}")
            print(f"  Range: {np.min(gc_values):.1%} - {np.max(gc_values):.1%}")

        # Genome size
        sizes = [s['genome_length'] for s in bin_stats if s['genome_length'] is not None]
        if sizes:
            print(f"\nGenome Size:")
            print(f"  Mean: {np.mean(sizes)/1e6:.2f} Mb")
            print(f"  Range: {np.min(sizes)/1e6:.2f} - {np.max(sizes)/1e6:.2f} Mb")

        # Number of prophage regions
        num_regions = [s['num_gt_regions'] for s in bin_stats]
        print(f"\nNumber of GT Prophage Regions:")
        print(f"  Mean: {np.mean(num_regions):.1f}")
        print(f"  Range: {np.min(num_regions)} - {np.max(num_regions)}")

        # Prophage sizes
        prophage_sizes = [s['avg_prophage_size'] for s in bin_stats if s['avg_prophage_size'] > 0]
        if prophage_sizes:
            print(f"\nAverage Prophage Size:")
            print(f"  Mean: {np.mean(prophage_sizes)/1000:.1f} kb")
            print(f"  Range: {np.min(prophage_sizes)/1000:.1f} - {np.max(prophage_sizes)/1000:.1f} kb")

        # Publications
        pubs = [s['publication'] for s in bin_stats if s['publication'] and s['publication'] != 'Unknown']
        if pubs:
            pub_counts = Counter(pubs)
            print(f"\nPublications:")
            for pub, count in pub_counts.most_common(5):
                print(f"  {count}: {pub[:60]}...")

        # Organisms (extract genus)
        organisms = [s['organism'] for s in bin_stats if s['organism'] and s['organism'] != 'Unknown']
        if organisms:
            # Extract genus (first word)
            genera = [org.split()[0] if org else 'Unknown' for org in organisms]
            genus_counts = Counter(genera)
            print(f"\nGenera:")
            for genus, count in genus_counts.most_common(10):
                print(f"  {count}: {genus}")

    # Create comparison plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # GC content by bin
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. GC content distribution
    ax = axes[0, 0]
    for bin_name, color in [('high', 'green'), ('medium', 'orange'), ('low', 'red')]:
        gc_vals = [s['gc_content'] for s in genome_stats if s['bin'] == bin_name and s['gc_content'] is not None]
        if gc_vals:
            ax.hist(gc_vals, bins=20, alpha=0.5, label=f'{bin_name} (n={len(gc_vals)})', color=color)
    ax.set_xlabel('GC Content')
    ax.set_ylabel('Count')
    ax.set_title('GC Content by Accuracy Bin')
    ax.legend()

    # 2. Genome size distribution
    ax = axes[0, 1]
    for bin_name, color in [('high', 'green'), ('medium', 'orange'), ('low', 'red')]:
        sizes = [s['genome_length']/1e6 for s in genome_stats if s['bin'] == bin_name and s['genome_length'] is not None]
        if sizes:
            ax.hist(sizes, bins=20, alpha=0.5, label=f'{bin_name} (n={len(sizes)})', color=color)
    ax.set_xlabel('Genome Size (Mb)')
    ax.set_ylabel('Count')
    ax.set_title('Genome Size by Accuracy Bin')
    ax.legend()

    # 3. Number of prophage regions
    ax = axes[1, 0]
    for bin_name, color in [('high', 'green'), ('medium', 'orange'), ('low', 'red')]:
        num_reg = [s['num_gt_regions'] for s in genome_stats if s['bin'] == bin_name]
        if num_reg:
            ax.hist(num_reg, bins=range(0, max(num_reg)+2), alpha=0.5, label=f'{bin_name} (n={len(num_reg)})', color=color)
    ax.set_xlabel('Number of GT Prophage Regions')
    ax.set_ylabel('Count')
    ax.set_title('Number of Prophage Regions by Accuracy Bin')
    ax.legend()

    # 4. Average prophage size
    ax = axes[1, 1]
    for bin_name, color in [('high', 'green'), ('medium', 'orange'), ('low', 'red')]:
        sizes = [s['avg_prophage_size']/1000 for s in genome_stats if s['bin'] == bin_name and s['avg_prophage_size'] > 0]
        if sizes:
            ax.hist(sizes, bins=20, alpha=0.5, label=f'{bin_name} (n={len(sizes)})', color=color)
    ax.set_xlabel('Average Prophage Size (kb)')
    ax.set_ylabel('Count')
    ax.set_title('Prophage Size by Accuracy Bin')
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / "performance_factors.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # F1 vs GC scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    gc_vals = [s['gc_content'] for s in genome_stats if s['gc_content'] is not None]
    f1_vals = [s['f1'] for s in genome_stats if s['gc_content'] is not None]
    ax.scatter(gc_vals, f1_vals, alpha=0.6)
    ax.set_xlabel('GC Content')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs GC Content')
    ax.grid(True, alpha=0.3)

    # Add correlation
    if gc_vals and f1_vals:
        corr = np.corrcoef(gc_vals, f1_vals)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=12)

    plot_path = output_dir / "f1_vs_gc.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Save detailed stats to CSV
    csv_path = output_dir / "genome_stats.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'assembly', 'bin', 'f1', 'precision', 'recall', 'mcc',
            'gc_content', 'genome_length', 'num_gt_regions', 'avg_prophage_size',
            'total_prophage_bp', 'organism', 'publication'
        ])
        writer.writeheader()
        writer.writerows(genome_stats)
    print(f"Saved stats: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
