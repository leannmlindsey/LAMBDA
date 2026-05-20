#!/usr/bin/env python3
"""
Convert SAE activation signals to genomic regions using density-based clustering.

This script:
1. Loads activation data from .npy files
2. Identifies positions above a threshold
3. Clusters positions using HDBSCAN and/or OPTICS
4. Filters clusters to keep only those > min_size (default 1kb)
5. Merges clusters within merge_distance (default 3kb) of each other
6. Outputs predicted regions in BED format and compares to ground truth

Usage:
    python cluster_activations.py \
        --results_dir ./lambda_results_7b \
        --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
        --output_dir ./cluster_results
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Clustering imports
from sklearn.cluster import HDBSCAN, OPTICS


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_activations(activations, method='none', window_size=10000):
    """
    Normalize activations to account for varying baseline levels across genomes.

    Args:
        activations: Raw activation values (1D numpy array)
        method: Normalization method:
            - 'none': No normalization (original behavior)
            - 'zscore': Standard z-score (x - mean) / std
            - 'robust_zscore': Robust z-score using median/MAD
            - 'percentile': (x - P25) / IQR - robust to outliers
            - 'local_baseline': x - rolling_median (handles varying baseline within genome)
            - 'minmax': Scale to [0, 1] range
            - 'quantile': Rank-based normalization to uniform distribution
        window_size: Window size for local_baseline method (default 10kb)

    Returns:
        Normalized activations (same shape as input)
    """
    if method == 'none':
        return activations

    acts = activations.astype(np.float64)

    if method == 'zscore':
        # Standard z-score normalization
        mean = np.mean(acts)
        std = np.std(acts)
        if std > 0:
            return (acts - mean) / std
        return acts - mean

    elif method == 'robust_zscore':
        # Robust z-score using median and MAD (median absolute deviation)
        # More robust to outliers (like prophage peaks)
        median = np.median(acts)
        mad = np.median(np.abs(acts - median))
        # Scale MAD to be consistent with std for normal distribution
        mad_scaled = mad * 1.4826
        if mad_scaled > 0:
            return (acts - median) / mad_scaled
        return acts - median

    elif method == 'percentile':
        # Percentile-based normalization
        # Baseline = 25th percentile, scale by IQR
        p25 = np.percentile(acts, 25)
        p75 = np.percentile(acts, 75)
        iqr = p75 - p25
        if iqr > 0:
            return (acts - p25) / iqr
        return acts - p25

    elif method == 'local_baseline':
        # Local baseline subtraction using rolling median
        # Handles cases where baseline varies across the genome
        import pandas as pd
        df = pd.Series(acts)
        # Use rolling median as local baseline estimate
        baseline = df.rolling(window=window_size, center=True, min_periods=1).median()
        # Also estimate local spread using rolling MAD
        local_mad = df.rolling(window=window_size, center=True, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))) * 1.4826, raw=True
        )
        # Normalize: (x - local_baseline) / local_spread
        local_mad = local_mad.fillna(1.0).replace(0, 1.0)
        normalized = (acts - baseline.values) / local_mad.values
        return normalized

    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = np.min(acts)
        max_val = np.max(acts)
        if max_val > min_val:
            return (acts - min_val) / (max_val - min_val)
        return np.zeros_like(acts)

    elif method == 'quantile':
        # Quantile normalization - transform to uniform distribution
        from scipy import stats
        ranks = stats.rankdata(acts)
        return ranks / len(ranks)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_adaptive_threshold(activations, method='mad', k=3.0):
    """
    Compute an adaptive threshold based on the genome's activation distribution.

    Args:
        activations: Raw or normalized activation values
        method: Method for threshold computation:
            - 'mad': median + k * MAD (robust)
            - 'std': mean + k * std (standard)
            - 'percentile': k-th percentile (e.g., k=95 for 95th percentile)
        k: Multiplier for mad/std, or percentile value

    Returns:
        Computed threshold value
    """
    acts = activations.astype(np.float64)

    if method == 'mad':
        median = np.median(acts)
        mad = np.median(np.abs(acts - median)) * 1.4826
        return median + k * mad

    elif method == 'std':
        mean = np.mean(acts)
        std = np.std(acts)
        return mean + k * std

    elif method == 'percentile':
        return np.percentile(acts, k)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def load_ground_truth(csv_path):
    """Load ground truth prophage regions."""
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
    name = Path(filename).stem
    return name.replace('_activations', '')


def cluster_mws(activations, window=85, threshold=0.4, min_region_size=1000, merge_distance=3000):
    """
    Moving Window Sum (MWS) algorithm.

    1. Apply centered rolling window SUM (looks forward and backward)
    2. Threshold to get binary predictions
    3. Find contiguous regions
    4. Filter by min size and merge nearby regions

    Args:
        activations: Raw activation values per nucleotide
        window: Rolling window size (centered, so looks window//2 in each direction)
        threshold: Threshold for rolling sum (higher values since not normalized)
        min_region_size: Minimum region size in bp
        merge_distance: Merge regions within this distance

    Returns:
        List of (start, end) tuples
    """
    import pandas as pd

    # Step 1: Centered rolling window SUM (not normalized)
    # center=True means it looks window//2 positions forward and backward
    df = pd.DataFrame({'act': activations})
    df['smoothed'] = df['act'].rolling(window=window, center=True, min_periods=1).sum()
    df['smoothed'] = df['smoothed'].fillna(0)

    # Step 2: Threshold to binary
    binary = (df['smoothed'] > threshold).values

    # Step 3: Find contiguous regions
    regions = []
    in_region = False
    start = 0

    for i, is_positive in enumerate(binary):
        if is_positive and not in_region:
            start = i
            in_region = True
        elif not is_positive and in_region:
            regions.append((start, i))
            in_region = False

    if in_region:
        regions.append((start, len(binary)))

    # Step 4: Merge nearby regions
    if regions:
        merged = [list(regions[0])]
        for start, end in regions[1:]:
            if start - merged[-1][1] <= merge_distance:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        regions = [(s, e) for s, e in merged]

    # Step 5: Filter by minimum size
    regions = [(s, e) for s, e in regions if e - s >= min_region_size]

    return regions


def cluster_positions_simple(positions, max_gap=100):
    """
    Simple O(n) clustering for 1D genomic positions.

    Groups positions that are within max_gap of each other into regions.
    Much faster than HDBSCAN/OPTICS for genomic data.

    Args:
        positions: 1D array of genomic positions (must be sorted)
        max_gap: Maximum gap between positions to consider them part of same cluster

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) == 0:
        return []

    positions = np.sort(positions)
    regions = []

    # Start first region
    region_start = positions[0]
    region_end = positions[0]

    for pos in positions[1:]:
        if pos - region_end <= max_gap:
            # Extend current region
            region_end = pos
        else:
            # Save current region and start new one
            regions.append((int(region_start), int(region_end)))
            region_start = pos
            region_end = pos

    # Don't forget the last region
    regions.append((int(region_start), int(region_end)))

    return regions


def cluster_positions_hdbscan(positions, min_cluster_size=100, min_samples=10, cluster_selection_epsilon=0.0):
    """
    Cluster genomic positions using HDBSCAN.

    Args:
        positions: 1D array of genomic positions
        min_cluster_size: Minimum number of positions to form a cluster
        min_samples: HDBSCAN min_samples parameter
        cluster_selection_epsilon: Distance threshold for cluster merging (0 = disabled)

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) < min_cluster_size:
        return []

    # Reshape for sklearn (needs 2D input)
    X = positions.reshape(-1, 1).astype(np.float64)

    try:
        # Use simpler HDBSCAN config without epsilon
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'  # Excess of Mass
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        print(f"    HDBSCAN failed: {e}")
        return []

    # Extract regions from clusters
    regions = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for label in unique_labels:
        cluster_positions = positions[labels == label]
        start = int(cluster_positions.min())
        end = int(cluster_positions.max())
        regions.append((start, end))

    return sorted(regions, key=lambda x: x[0])


def cluster_positions_optics(positions, min_samples=50, xi=0.05, min_cluster_size=100):
    """
    Cluster genomic positions using OPTICS.

    Args:
        positions: 1D array of genomic positions
        min_samples: OPTICS min_samples parameter
        xi: Steepness threshold for cluster extraction
        min_cluster_size: Minimum cluster size

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) < min_cluster_size:
        return []

    X = positions.reshape(-1, 1).astype(np.float64)

    try:
        clusterer = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            n_jobs=-1  # Use all cores
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        print(f"    OPTICS failed: {e}")
        return []

    regions = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for label in unique_labels:
        cluster_positions = positions[labels == label]
        start = int(cluster_positions.min())
        end = int(cluster_positions.max())
        regions.append((start, end))

    return sorted(regions, key=lambda x: x[0])


def merge_nearby_regions(regions, merge_distance=3000):
    """
    Merge regions that are within merge_distance of each other.

    Args:
        regions: List of (start, end) tuples, sorted by start
        merge_distance: Maximum gap between regions to merge (default 3kb)

    Returns:
        List of merged (start, end) tuples
    """
    if not regions:
        return []

    merged = [list(regions[0])]

    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]

        # Check if this region is within merge_distance of the previous
        if start - prev_end <= merge_distance:
            # Merge: extend the previous region
            merged[-1][1] = max(prev_end, end)
        else:
            # Start new region
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def filter_by_size(regions, min_size=1000):
    """Filter regions to keep only those >= min_size."""
    return [(s, e) for s, e in regions if e - s >= min_size]


def calculate_metrics(predicted_regions, gt_regions, seq_len):
    """
    Calculate precision, recall, F1, MCC, and Jaccard at NUCLEOTIDE level.

    All metrics use the same TP/FP/FN/TN counts:
    - TP = nucleotides correctly predicted as prophage
    - FP = nucleotides incorrectly predicted as prophage
    - FN = prophage nucleotides missed
    - TN = non-prophage nucleotides correctly not predicted
    """
    # Create binary masks for predicted and ground truth
    pred_mask = np.zeros(seq_len, dtype=bool)
    gt_mask = np.zeros(seq_len, dtype=bool)

    for start, end in predicted_regions:
        pred_mask[start:min(end, seq_len)] = True

    for gt in gt_regions:
        start, end = gt['start'], gt['end']
        gt_mask[start:min(end, seq_len)] = True

    # Calculate TP, FP, FN, TN at nucleotide level
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)

    # Convert to float64 to avoid overflow
    tp = np.float64(tp)
    fp = np.float64(fp)
    fn = np.float64(fn)
    tn = np.float64(tn)

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Jaccard (IoU) = TP / (TP + FP + FN)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    numerator = (tp * tn) - (fp * fn)
    # Split sqrt to avoid overflow
    denom_part1 = np.sqrt((tp + fp) * (tp + fn))
    denom_part2 = np.sqrt((tn + fp) * (tn + fn))
    denominator = denom_part1 * denom_part2
    mcc = numerator / denominator if denominator > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'jaccard': float(jaccard),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'gt_regions': len(gt_regions),
        'pred_regions': len(predicted_regions),
    }


def process_genome(activations, assembly_id, gt_regions, args, threshold_override=None):
    """Process a single genome and return clustering results.

    Args:
        activations: Activation values (possibly normalized)
        assembly_id: Assembly identifier
        gt_regions: Ground truth regions
        args: Command line arguments
        threshold_override: If provided, use this threshold instead of args.threshold
    """

    seq_len = len(activations)
    threshold = threshold_override if threshold_override is not None else args.threshold

    # Find positions above threshold (for simple clustering)
    positions = np.where(activations > threshold)[0]

    results = {
        'assembly': assembly_id,
        'seq_len': seq_len,
        'positions_above_threshold': len(positions),
        'gt_regions': len(gt_regions),
    }

    # MWS clustering (Moving Window Sum from Phoenix) - works on raw activations
    if args.use_mws:
        mws_regions = cluster_mws(
            activations,
            window=args.mws_window,
            threshold=args.mws_threshold,
            min_region_size=args.min_region_size,
            merge_distance=args.merge_distance
        )
        results['mws_regions'] = mws_regions
        results['mws_metrics'] = calculate_metrics(mws_regions, gt_regions, seq_len)
    else:
        results['mws_regions'] = []
        results['mws_metrics'] = {}

    if len(positions) == 0:
        results['simple_regions'] = []
        results['hdbscan_regions'] = []
        results['optics_regions'] = []
        results['simple_metrics'] = calculate_metrics([], gt_regions, seq_len)
        results['hdbscan_metrics'] = calculate_metrics([], gt_regions, seq_len)
        results['optics_metrics'] = calculate_metrics([], gt_regions, seq_len)
        return results

    # Simple clustering (fast, O(n))
    simple_regions = cluster_positions_simple(positions, max_gap=args.max_gap)
    simple_regions = merge_nearby_regions(simple_regions, args.merge_distance)
    simple_regions = filter_by_size(simple_regions, args.min_region_size)
    results['simple_regions'] = simple_regions
    results['simple_metrics'] = calculate_metrics(simple_regions, gt_regions, seq_len)

    # HDBSCAN/OPTICS only if requested (slow)
    if args.use_hdbscan:
        hdbscan_regions = cluster_positions_hdbscan(
            positions,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
        hdbscan_regions = merge_nearby_regions(hdbscan_regions, args.merge_distance)
        hdbscan_regions = filter_by_size(hdbscan_regions, args.min_region_size)
        results['hdbscan_regions'] = hdbscan_regions
        results['hdbscan_metrics'] = calculate_metrics(hdbscan_regions, gt_regions, seq_len)
    else:
        results['hdbscan_regions'] = []
        results['hdbscan_metrics'] = {}

    if args.use_optics:
        optics_regions = cluster_positions_optics(
            positions,
            min_samples=args.min_samples,
            xi=args.xi,
            min_cluster_size=args.min_cluster_size
        )
        optics_regions = merge_nearby_regions(optics_regions, args.merge_distance)
        optics_regions = filter_by_size(optics_regions, args.min_region_size)
        results['optics_regions'] = optics_regions
        results['optics_metrics'] = calculate_metrics(optics_regions, gt_regions, seq_len)
    else:
        results['optics_regions'] = []
        results['optics_metrics'] = {}

    return results


def save_bed(regions, output_path, assembly_id):
    """Save regions in BED format."""
    with open(output_path, 'w') as f:
        for i, (start, end) in enumerate(regions):
            name = f"{assembly_id}_prophage_{i+1}"
            f.write(f"{assembly_id}\t{start}\t{end}\t{name}\t0\t+\n")


def plot_comparison(activations, simple_regions, second_regions, gt_regions,
                    assembly_id, output_path, threshold, second_method_name="HDBSCAN",
                    metrics=None, taxonomy=None, gc_content=None):
    """Generate comparison plot showing clustering results vs ground truth.

    Args:
        metrics: dict with 'precision', 'recall', 'mcc', 'f1' keys
        taxonomy: organism name string
        gc_content: GC content as float (0-1)
    """

    seq_len = len(activations)

    # Downsample for plotting
    max_points = 50000
    if seq_len > max_points:
        bin_size = seq_len // max_points
        n_bins = seq_len // bin_size
        truncated = activations[:n_bins * bin_size]
        reshaped = truncated.reshape(n_bins, bin_size)
        plot_acts = reshaped.max(axis=1)
        x_coords = np.arange(n_bins) * bin_size + bin_size // 2
    else:
        plot_acts = activations
        x_coords = np.arange(seq_len)

    fig, axes = plt.subplots(4, 1, figsize=(20, 10), height_ratios=[3, 1, 1, 1], sharex=True)

    # Build title with metrics and taxonomy
    title_parts = [assembly_id]
    if taxonomy and taxonomy not in ('Unknown', 'N/A', ''):
        # Extract genus + species (first two words)
        tax_short = ' '.join(taxonomy.split()[:2])
        if len(tax_short) > 35:
            tax_short = tax_short[:32] + "..."
        title_parts.append(tax_short)

    title_line1 = " - ".join(title_parts)

    # Metrics line
    metrics_parts = []
    if metrics:
        if metrics.get('precision') is not None:
            metrics_parts.append(f"P:{metrics['precision']:.3f}")
        if metrics.get('recall') is not None:
            metrics_parts.append(f"R:{metrics['recall']:.3f}")
        if metrics.get('mcc') is not None:
            metrics_parts.append(f"MCC:{metrics['mcc']:.3f}")
    if gc_content is not None:
        metrics_parts.append(f"GC:{gc_content:.1%}")

    title_line2 = "  ".join(metrics_parts) if metrics_parts else ""

    # Combine title
    full_title = title_line1
    if title_line2:
        full_title += f"\n{title_line2}"

    # Top: Activation signal
    ax1 = axes[0]
    ax1.fill_between(x_coords, 0, plot_acts, alpha=0.3, color='blue')
    ax1.plot(x_coords, plot_acts, lw=0.5, alpha=0.9, color='blue')
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    ax1.set_ylabel('SAE Activation')
    ax1.set_title(full_title, fontsize=12, fontweight='bold')
    ax1.set_xlim(0, seq_len)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Simple clustering regions (fast method)
    ax2 = axes[1]
    ax2.set_xlim(0, seq_len)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Predicted')
    ax2.set_yticks([])
    for start, end in simple_regions:
        ax2.axvspan(start, end, alpha=0.7, color='green')
        mid = (start + end) / 2
        size_kb = (end - start) / 1000
        ax2.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not simple_regions:
        ax2.text(0.5, 0.5, 'No regions', transform=ax2.transAxes, ha='center', va='center', color='gray')

    # Second method regions (MWS or HDBSCAN)
    ax3 = axes[2]
    ax3.set_xlim(0, seq_len)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel(second_method_name)
    ax3.set_yticks([])
    for start, end in second_regions:
        ax3.axvspan(start, end, alpha=0.7, color='purple')
        mid = (start + end) / 2
        size_kb = (end - start) / 1000
        ax3.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not second_regions:
        ax3.text(0.5, 0.5, f'Not run (use --use_mws or --use_hdbscan)', transform=ax3.transAxes, ha='center', va='center', color='gray')

    # Ground truth
    ax4 = axes[3]
    ax4.set_xlim(0, seq_len)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Ground\nTruth')
    ax4.set_xlabel('Genomic Position (bp)')
    ax4.set_yticks([])
    for gt in gt_regions:
        start, end = gt['start'], gt['end']
        if start < seq_len:
            ax4.axvspan(start, min(end, seq_len), alpha=0.7, color='red')
            mid = (start + min(end, seq_len)) / 2
            size_kb = (min(end, seq_len) - start) / 1000
            ax4.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not gt_regions:
        ax4.text(0.5, 0.5, 'No ground truth', transform=ax4.transAxes, ha='center', va='center', color='gray')

    # Format x-axis
    def format_mb(x, p):
        return f'{x/1e6:.1f} Mb'
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_mb))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cluster SAE activations into prophage regions")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with *_activations.npy files")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth CSV file")
    parser.add_argument("--output_dir", type=str, default="./cluster_results", help="Output directory")

    # Threshold for considering a position "active"
    parser.add_argument("--threshold", type=float, default=0.3, help="Activation threshold (default: 0.3)")

    # Normalization options
    parser.add_argument("--normalize", type=str, default="none",
                        choices=['none', 'zscore', 'robust_zscore', 'percentile', 'local_baseline', 'minmax', 'quantile'],
                        help="Normalization method: none, zscore, robust_zscore, percentile, local_baseline, minmax, quantile (default: none)")
    parser.add_argument("--norm_window", type=int, default=10000,
                        help="Window size for local_baseline normalization (default: 10kb)")
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use adaptive threshold per genome instead of fixed threshold")
    parser.add_argument("--adaptive_method", type=str, default="mad",
                        choices=['mad', 'std', 'percentile'],
                        help="Method for adaptive threshold: mad, std, percentile (default: mad)")
    parser.add_argument("--adaptive_k", type=float, default=3.0,
                        help="Multiplier for adaptive threshold (default: 3.0 = median + 3*MAD)")

    # Clustering parameters
    parser.add_argument("--max_gap", type=int, default=100, help="Max gap between positions in simple clustering (default: 100bp)")
    parser.add_argument("--use_mws", action="store_true", help="Use Moving Window Sum algorithm (from Phoenix dashboard)")
    parser.add_argument("--mws_window", type=int, default=85, help="MWS rolling window size (default: 85 from Phoenix)")
    parser.add_argument("--mws_threshold", type=float, default=0.4, help="MWS threshold (default: 0.4 from Phoenix)")
    parser.add_argument("--use_hdbscan", action="store_true", help="Also run HDBSCAN clustering (slow)")
    parser.add_argument("--use_optics", action="store_true", help="Also run OPTICS clustering (slow)")
    parser.add_argument("--min_cluster_size", type=int, default=100, help="Minimum positions for HDBSCAN/OPTICS")
    parser.add_argument("--min_samples", type=int, default=20, help="HDBSCAN/OPTICS min_samples")
    parser.add_argument("--xi", type=float, default=0.05, help="OPTICS xi parameter")

    # Region filtering
    parser.add_argument("--min_region_size", type=int, default=1000, help="Minimum region size in bp (default: 1kb)")
    parser.add_argument("--merge_distance", type=int, default=3000, help="Merge regions within this distance (default: 3kb)")

    # Options
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--fix_artifacts", action="store_true", help="Fix window boundary artifacts before clustering")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    bed_dir = output_dir / "bed"
    bed_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Cluster SAE Activations into Prophage Regions")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"Threshold: {args.threshold}")
    if args.normalize != 'none':
        print(f"Normalization: {args.normalize}" + (f" (window={args.norm_window})" if args.normalize == 'local_baseline' else ""))
    if args.adaptive_threshold:
        print(f"Adaptive threshold: {args.adaptive_method} (k={args.adaptive_k})")
    print(f"Min region size: {args.min_region_size} bp")
    print(f"Merge distance: {args.merge_distance} bp")
    print(f"Clustering params: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  Found {len(gt)} assemblies with ground truth")

    # Find activation files
    results_dir = Path(args.results_dir)
    npy_files = sorted(results_dir.glob("*_activations.npy"))
    print(f"\nFound {len(npy_files)} activation files")

    # Process each genome
    all_results = []

    for npy_file in tqdm(npy_files, desc="Processing genomes"):
        assembly_id = get_assembly_from_filename(npy_file)

        # Load activations
        activations = np.load(npy_file)

        # Optionally fix window artifacts
        if args.fix_artifacts:
            window_size, overlap, startup_trim = 50000, 1000, 10
            stride = window_size - overlap
            win_idx = 1
            while True:
                win_start = win_idx * stride
                if win_start >= len(activations):
                    break
                trim_end = min(win_start + startup_trim, len(activations))
                activations[win_start:trim_end] = 0.0
                win_idx += 1

        # Store raw activations for plotting (before normalization)
        raw_activations = activations.copy()

        # Apply normalization if requested
        if args.normalize != 'none':
            activations = normalize_activations(
                activations,
                method=args.normalize,
                window_size=args.norm_window
            )

        # Compute adaptive threshold if requested
        if args.adaptive_threshold:
            effective_threshold = compute_adaptive_threshold(
                activations,
                method=args.adaptive_method,
                k=args.adaptive_k
            )
        else:
            effective_threshold = args.threshold

        # Get ground truth
        gt_regions = gt.get(assembly_id, [])
        if not gt_regions:
            for gt_assembly, regions in gt.items():
                if regions and regions[0].get('ncbi_id') == assembly_id:
                    gt_regions = regions
                    break

        # Process
        results = process_genome(activations, assembly_id, gt_regions, args,
                                  threshold_override=effective_threshold)
        results['effective_threshold'] = effective_threshold  # Store for reference
        all_results.append(results)

        # Save BED files
        if results['simple_regions']:
            save_bed(results['simple_regions'], bed_dir / f"{assembly_id}_simple.bed", assembly_id)
        if results.get('mws_regions'):
            save_bed(results['mws_regions'], bed_dir / f"{assembly_id}_mws.bed", assembly_id)
        if results.get('hdbscan_regions'):
            save_bed(results['hdbscan_regions'], bed_dir / f"{assembly_id}_hdbscan.bed", assembly_id)
        if results.get('optics_regions'):
            save_bed(results['optics_regions'], bed_dir / f"{assembly_id}_optics.bed", assembly_id)

        # Generate plot
        if not args.no_plots:
            # Use MWS regions if available, otherwise HDBSCAN
            second_method_regions = results.get('mws_regions', []) if args.use_mws else results.get('hdbscan_regions', [])
            second_method_name = "MWS" if args.use_mws else "HDBSCAN"

            # Get taxonomy from ground truth (use first region's organism)
            taxonomy = None
            if gt_regions and gt_regions[0].get('organism'):
                taxonomy = gt_regions[0]['organism']

            # Get metrics for the primary method (simple clustering)
            metrics = results.get('simple_metrics', {})

            # Use normalized activations for plot if normalization applied
            # This shows the signal as the algorithm sees it
            plot_activations = activations if args.normalize != 'none' else raw_activations

            plot_comparison(
                plot_activations,
                results['simple_regions'],
                second_method_regions,
                gt_regions,
                assembly_id,
                plots_dir / f"{assembly_id}_clusters.png",
                effective_threshold,
                second_method_name,
                metrics=metrics,
                taxonomy=taxonomy,
                gc_content=None  # GC content requires FASTA files
            )

    # Save results
    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    genomes_with_gt = [r for r in all_results if r['gt_regions'] > 0]
    print(f"Genomes with ground truth: {len(genomes_with_gt)} / {len(all_results)}")

    if genomes_with_gt:
        print("\nSimple Clustering Performance (threshold={}, max_gap={}):".format(args.threshold, args.max_gap))
        simple_precision = np.mean([r['simple_metrics']['precision'] for r in genomes_with_gt])
        simple_recall = np.mean([r['simple_metrics']['recall'] for r in genomes_with_gt])
        simple_f1 = np.mean([r['simple_metrics']['f1'] for r in genomes_with_gt])
        simple_mcc = np.mean([r['simple_metrics']['mcc'] for r in genomes_with_gt])
        simple_jaccard = np.mean([r['simple_metrics']['jaccard'] for r in genomes_with_gt])
        print(f"  Precision: {simple_precision:.1%}")
        print(f"  Recall:    {simple_recall:.1%}")
        print(f"  F1:        {simple_f1:.1%}")
        print(f"  MCC:       {simple_mcc:.3f}")
        print(f"  Jaccard:   {simple_jaccard:.3f}")

        if args.use_mws:
            print("\nMWS Clustering Performance (window={}, threshold={}):".format(args.mws_window, args.mws_threshold))
            mws_precision = np.mean([r['mws_metrics']['precision'] for r in genomes_with_gt])
            mws_recall = np.mean([r['mws_metrics']['recall'] for r in genomes_with_gt])
            mws_f1 = np.mean([r['mws_metrics']['f1'] for r in genomes_with_gt])
            mws_mcc = np.mean([r['mws_metrics']['mcc'] for r in genomes_with_gt])
            mws_jaccard = np.mean([r['mws_metrics']['jaccard'] for r in genomes_with_gt])
            print(f"  Precision: {mws_precision:.1%}")
            print(f"  Recall:    {mws_recall:.1%}")
            print(f"  F1:        {mws_f1:.1%}")
            print(f"  MCC:       {mws_mcc:.3f}")
            print(f"  Jaccard:   {mws_jaccard:.3f}")

        if args.use_hdbscan:
            print("\nHDBSCAN Performance:")
            hdb_precision = np.mean([r['hdbscan_metrics'].get('precision', 0) for r in genomes_with_gt])
            hdb_recall = np.mean([r['hdbscan_metrics'].get('recall', 0) for r in genomes_with_gt])
            hdb_f1 = np.mean([r['hdbscan_metrics'].get('f1', 0) for r in genomes_with_gt])
            hdb_mcc = np.mean([r['hdbscan_metrics'].get('mcc', 0) for r in genomes_with_gt])
            print(f"  Precision: {hdb_precision:.1%}")
            print(f"  Recall:    {hdb_recall:.1%}")
            print(f"  F1:        {hdb_f1:.1%}")
            print(f"  MCC:       {hdb_mcc:.3f}")

        if args.use_optics:
            print("\nOPTICS Performance:")
            opt_precision = np.mean([r['optics_metrics'].get('precision', 0) for r in genomes_with_gt])
            opt_recall = np.mean([r['optics_metrics'].get('recall', 0) for r in genomes_with_gt])
            opt_f1 = np.mean([r['optics_metrics'].get('f1', 0) for r in genomes_with_gt])
            opt_mcc = np.mean([r['optics_metrics'].get('mcc', 0) for r in genomes_with_gt])
            print(f"  Precision: {opt_precision:.1%}")
            print(f"  Recall:    {opt_recall:.1%}")
            print(f"  F1:        {opt_f1:.1%}")
            print(f"  MCC:       {opt_mcc:.3f}")

        # Count total regions
        total_simple = sum(len(r['simple_regions']) for r in all_results)
        total_gt = sum(r['gt_regions'] for r in genomes_with_gt)
        print(f"\nTotal regions predicted (Simple): {total_simple}")
        print(f"Total ground truth regions:       {total_gt}")

        # Bin genomes by accuracy
        print("\n" + "=" * 60)
        print("GENOME ACCURACY BINS (by F1 score)")
        print("=" * 60)

        high_acc = [r for r in genomes_with_gt if r['simple_metrics']['f1'] >= 0.7]
        med_acc = [r for r in genomes_with_gt if 0.3 <= r['simple_metrics']['f1'] < 0.7]
        low_acc = [r for r in genomes_with_gt if r['simple_metrics']['f1'] < 0.3]

        print(f"\nHigh accuracy (F1 >= 0.7): {len(high_acc)} genomes")
        for r in sorted(high_acc, key=lambda x: -x['simple_metrics']['f1'])[:10]:
            m = r['simple_metrics']
            print(f"  {r['assembly']}: P={m['precision']:.2f}, R={m['recall']:.2f}, F1={m['f1']:.2f}, MCC={m['mcc']:.3f}, Jaccard={m['jaccard']:.3f}")
            print(f"      TP={m['tp']:,}, FP={m['fp']:,}, FN={m['fn']:,}, Regions: {m['pred_regions']} pred / {m['gt_regions']} GT")

        print(f"\nMedium accuracy (0.3 <= F1 < 0.7): {len(med_acc)} genomes")
        for r in sorted(med_acc, key=lambda x: -x['simple_metrics']['f1'])[:5]:
            m = r['simple_metrics']
            print(f"  {r['assembly']}: F1={m['f1']:.2f}, P={m['precision']:.2f}, R={m['recall']:.2f}, MCC={m['mcc']:.3f}")

        print(f"\nLow accuracy (F1 < 0.3): {len(low_acc)} genomes")
        for r in sorted(low_acc, key=lambda x: -x['simple_metrics']['f1'])[:5]:
            m = r['simple_metrics']
            print(f"  {r['assembly']}: F1={m['f1']:.2f}, P={m['precision']:.2f}, R={m['recall']:.2f}, MCC={m['mcc']:.3f}")

        # Save binned results to separate files
        bins_dir = output_dir / "accuracy_bins"
        bins_dir.mkdir(exist_ok=True)

        for bin_name, bin_data in [("high", high_acc), ("medium", med_acc), ("low", low_acc)]:
            bin_file = bins_dir / f"{bin_name}_accuracy_genomes.txt"
            with open(bin_file, 'w') as f:
                f.write(f"# {bin_name.upper()} accuracy genomes (F1-based)\n")
                f.write("# All metrics at NUCLEOTIDE level\n")
                f.write("# assembly\tprecision\trecall\tF1\tMCC\tJaccard\tTP\tFP\tFN\tgt_regions\tpred_regions\n")
                for r in sorted(bin_data, key=lambda x: -x['simple_metrics']['f1']):
                    m = r['simple_metrics']
                    f.write(f"{r['assembly']}\t{m['precision']:.3f}\t{m['recall']:.3f}\t{m['f1']:.3f}\t{m['mcc']:.3f}\t{m['jaccard']:.3f}\t{m['tp']}\t{m['fp']}\t{m['fn']}\t{m['gt_regions']}\t{m['pred_regions']}\n")
            print(f"\nSaved {bin_name} accuracy list to: {bin_file}")

    print(f"\nResults saved to: {output_dir}")
    print(f"BED files: {bed_dir}")
    if not args.no_plots:
        print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
