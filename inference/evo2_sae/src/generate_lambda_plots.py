#!/usr/bin/env python3
"""
Generate PNG visualizations from saved LAMBDA batch results.

This script reads the .npy activation files and ground truth CSV to create
visualization plots for each genome showing SAE feature activations
overlaid with ground truth prophage regions.

Usage:
    python generate_lambda_plots.py \
        --results_dir ./lambda_results_7b \
        --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
        --output_dir ./lambda_plots
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate PNG visualizations from LAMBDA batch results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing *_activations.npy files and all_results.json",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Ground truth CSV file (Lambda_Genome_Wide_Evaluation_Test_Set.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Activation threshold for highlighting (default: 0.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output images (default: 150)",
    )
    return parser.parse_args()


def load_ground_truth(csv_path):
    """Load all ground truth prophage regions."""
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['Assembly']
            if assembly not in gt:
                gt[assembly] = []
            gt[assembly].append({
                'ncbi_id': row['NCBI Id'],
                'start': int(row['start']),
                'end': int(row['end']),
                'organism': row['Organism Name'],
            })
    return gt


def get_assembly_from_filename(filename):
    """Extract assembly ID from activation filename."""
    # filename like: GCF_000006665.1_activations.npy
    name = Path(filename).stem  # GCF_000006665.1_activations
    return name.replace('_activations', '')


def downsample_for_plotting(activations, max_points=50000):
    """Downsample activations for plotting if sequence is too long."""
    seq_len = len(activations)
    if seq_len <= max_points:
        return np.arange(seq_len), activations

    # Use max pooling to preserve peaks
    bin_size = seq_len // max_points
    n_bins = seq_len // bin_size
    truncated = activations[:n_bins * bin_size]
    reshaped = truncated.reshape(n_bins, bin_size)

    # Take max within each bin to preserve peaks
    downsampled = reshaped.max(axis=1)
    x_coords = np.arange(n_bins) * bin_size + bin_size // 2

    return x_coords, downsampled


def generate_plot(
    activations: np.ndarray,
    gt_regions: list,
    assembly_id: str,
    output_path: str,
    threshold: float = 0.5,
    dpi: int = 150,
):
    """Generate a visualization plot for a single genome."""

    seq_len = len(activations)

    # Calculate stats
    max_act = activations.max()
    mean_act = activations.mean()
    firing_count = np.sum(activations > threshold)

    # Calculate how much firing is in GT
    in_gt = 0
    for r in gt_regions:
        start, end = r['start'], r['end']
        if start < seq_len and end <= seq_len:
            in_gt += np.sum(activations[start:end] > threshold)

    precision = in_gt / firing_count if firing_count > 0 else 0

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 6), height_ratios=[3, 1], sharex=True)

    # Top: Feature activation
    ax1 = axes[0]

    # Downsample for plotting if sequence is too long
    x_coords, plot_activations = downsample_for_plotting(activations)

    # Plot activation line
    ax1.fill_between(x_coords, 0, plot_activations, alpha=0.3, color='blue')
    ax1.plot(x_coords, plot_activations, lw=0.5, alpha=0.9, color='blue')

    # Add threshold line
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')

    # Add ground truth shading
    for i, r in enumerate(gt_regions):
        if r['start'] < seq_len:
            label = 'Ground Truth' if i == 0 else None
            ax1.axvspan(r['start'], min(r['end'], seq_len),
                       alpha=0.2, color='red', label=label)

    ax1.set_ylabel('Feature f/19746\nActivation')
    ax1.set_title(f'{assembly_id} - Evo2 SAE Prophage Feature\n'
                  f'Max: {max_act:.2f} | Firing: {firing_count:,} positions | '
                  f'Precision: {precision:.1%} in GT | {len(gt_regions)} GT regions')
    ax1.set_xlim(0, seq_len)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: Ground truth blocks
    ax2 = axes[1]
    ax2.set_xlim(0, seq_len)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Ground\nTruth')
    ax2.set_xlabel('Genomic Position (bp)')
    ax2.set_yticks([])

    for r in gt_regions:
        if r['start'] < seq_len:
            rect_start = r['start']
            rect_end = min(r['end'], seq_len)
            ax2.axvspan(rect_start, rect_end, alpha=0.7, color='red')
            # Label with size
            mid = (rect_start + rect_end) / 2
            size_kb = (rect_end - rect_start) / 1000
            ax2.text(mid, 0.5, f'{size_kb:.1f}kb',
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    if not gt_regions:
        ax2.text(0.5, 0.5, 'No ground truth regions', transform=ax2.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # Format x-axis with Mb labels
    def format_mb(x, p):
        return f'{x/1e6:.1f} Mb'
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_mb))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return {
        'assembly': assembly_id,
        'max_activation': float(max_act),
        'firing_count': int(firing_count),
        'precision': float(precision),
        'gt_regions': len(gt_regions),
    }


def main():
    args = parse_arguments()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generate LAMBDA Visualization Plots")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output dir: {output_dir}")
    print(f"Threshold: {args.threshold}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  Found {len(gt)} assemblies with ground truth")

    # Find all activation files
    npy_files = sorted(results_dir.glob("*_activations.npy"))
    print(f"\nFound {len(npy_files)} activation files")

    if len(npy_files) == 0:
        print("No activation files found! Make sure the batch job has processed some genomes.")
        return

    # Generate plots
    all_stats = []

    for npy_file in tqdm(npy_files, desc="Generating plots"):
        assembly_id = get_assembly_from_filename(npy_file)

        # Load activations
        activations = np.load(npy_file)

        # Get ground truth regions for this assembly
        gt_regions = gt.get(assembly_id, [])

        # Also try matching by NCBI ID
        if not gt_regions:
            for gt_assembly, regions in gt.items():
                if regions and regions[0].get('ncbi_id') == assembly_id:
                    gt_regions = regions
                    break

        # Generate plot
        output_path = output_dir / f"{assembly_id}_plot.png"
        stats = generate_plot(
            activations,
            gt_regions,
            assembly_id,
            str(output_path),
            threshold=args.threshold,
            dpi=args.dpi,
        )
        all_stats.append(stats)

    # Save summary
    summary_path = output_dir / "plot_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nGenerated {len(all_stats)} plots")
    print(f"Saved to: {output_dir}")
    print(f"Summary: {summary_path}")

    # Print quick stats
    if all_stats:
        precisions = [s['precision'] for s in all_stats if s['gt_regions'] > 0]
        if precisions:
            print(f"\nPrecision stats (genomes with GT):")
            print(f"  Mean: {np.mean(precisions):.1%}")
            print(f"  Min:  {np.min(precisions):.1%}")
            print(f"  Max:  {np.max(precisions):.1%}")


if __name__ == "__main__":
    main()
