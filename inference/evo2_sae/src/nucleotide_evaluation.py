#!/usr/bin/env python3
"""
Nucleotide-level prophage detection from segment SAE activations.

Stitches overlapping segment activations back into genome-wide activation tracks,
calls prophage regions from those tracks, and evaluates at nucleotide resolution
against ground truth boundaries from the input CSV.

Pipeline:
  A. Load segment metadata (seq_id, start, end) and per-segment .npy activations
  B. Stitch into genome-wide activation tracks using MAX pooling in overlap regions
  C. Normalize (optional) → threshold → cluster → merge → filter → predicted regions
  D. Extract ground truth prophage regions from segment metadata
  E. Evaluate at nucleotide level (precision, recall, F1, MCC, Jaccard)

Prerequisites:
  - Run sae_inference.py with --save_activations first
  - Input CSV must have columns: seq_id, start, end, prophage_start, prophage_end

Usage:
    python src/nucleotide_evaluation.py \
        --input_csv results.csv \
        --activations_dir results_activations/ \
        --output_dir nucleotide_eval_results/
"""

import argparse
import csv
import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import reusable functions from cluster_activations.py
from cluster_activations import (
    normalize_activations,
    compute_adaptive_threshold,
    cluster_positions_simple,
    merge_nearby_regions,
    filter_by_size,
    calculate_metrics,
)


# =============================================================================
# PHASE A: Load & Stitch Activations
# =============================================================================

def load_segment_metadata(csv_path):
    """Load segment metadata from the SAE inference output CSV.

    Returns:
        list of dicts with keys: segment_id, seq_id, start, end, prophage_start, prophage_end
    """
    segments = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        required = ['segment_id', 'seq_id', 'start', 'end']
        missing = [c for c in required if c not in columns]
        if missing:
            print(f"ERROR: Input CSV missing required columns: {missing}")
            print(f"  Available columns: {columns}")
            print(f"  Make sure to run sae_inference.py with an input CSV that has seq_id, start, end columns.")
            sys.exit(1)

        for row in reader:
            seg = {
                'segment_id': row['segment_id'],
                'seq_id': row['seq_id'],
                'start': int(row['start']),
                'end': int(row['end']),
            }
            # Optional ground truth columns
            if 'prophage_start' in columns and row.get('prophage_start', '').strip():
                seg['prophage_start'] = int(row['prophage_start'])
            else:
                seg['prophage_start'] = None
            if 'prophage_end' in columns and row.get('prophage_end', '').strip():
                seg['prophage_end'] = int(row['prophage_end'])
            else:
                seg['prophage_end'] = None

            segments.append(seg)

    return segments


def stitch_activations(segments, activations_dir, feature_idx):
    """Stitch segment activations into genome-wide tracks using MAX pooling.

    Args:
        segments: list of segment dicts (grouped by seq_id)
        activations_dir: Path to directory with per-segment .npy files
        feature_idx: SAE feature index to extract

    Returns:
        dict mapping seq_id -> genome-wide activation array (1D numpy)
    """
    activations_dir = Path(activations_dir)

    # Group segments by seq_id
    by_genome = defaultdict(list)
    for seg in segments:
        by_genome[seg['seq_id']].append(seg)

    genome_activations = {}
    load_errors = []

    for seq_id, segs in sorted(by_genome.items()):
        # Determine genome length from max segment end
        genome_length = max(s['end'] for s in segs)
        genome_act = np.zeros(genome_length, dtype=np.float64)

        for seg in segs:
            npy_path = activations_dir / f"{seg['segment_id']}.npy"
            if not npy_path.exists():
                load_errors.append(str(npy_path))
                continue

            raw = np.load(npy_path)

            # Handle multi-feature arrays: shape (seq_tokens, n_features)
            if raw.ndim == 2:
                if feature_idx >= raw.shape[1]:
                    print(f"  WARNING: feature_idx {feature_idx} >= array width {raw.shape[1]} for {seg['segment_id']}")
                    continue
                seg_act = raw[:, feature_idx].astype(np.float64)
            elif raw.ndim == 1:
                # Already a single feature track
                seg_act = raw.astype(np.float64)
            else:
                print(f"  WARNING: unexpected array shape {raw.shape} for {seg['segment_id']}")
                continue

            start = seg['start']
            end = seg['end']
            # The activation array length may differ from end-start (tokenization)
            act_len = len(seg_act)
            seg_len = end - start

            if act_len < seg_len:
                # Pad with zeros if activation is shorter
                padded = np.zeros(seg_len, dtype=np.float64)
                padded[:act_len] = seg_act
                seg_act = padded
            elif act_len > seg_len:
                # Truncate if activation is longer
                seg_act = seg_act[:seg_len]

            # MAX pooling: preserve sparse prophage signal in overlap regions
            genome_act[start:end] = np.maximum(genome_act[start:end], seg_act)

        genome_activations[seq_id] = genome_act

    if load_errors:
        print(f"  WARNING: Could not load {len(load_errors)} activation files")
        if len(load_errors) <= 5:
            for p in load_errors:
                print(f"    {p}")

    return genome_activations


# =============================================================================
# PHASE C: Extract Ground Truth Regions
# =============================================================================

def extract_ground_truth(segments):
    """Extract unique ground truth prophage regions per genome from segment metadata.

    Returns:
        dict mapping seq_id -> list of {'start': int, 'end': int}
    """
    by_genome = defaultdict(set)

    for seg in segments:
        if seg['prophage_start'] is not None and seg['prophage_end'] is not None:
            by_genome[seg['seq_id']].add((seg['prophage_start'], seg['prophage_end']))

    gt_regions = {}
    for seq_id, region_set in by_genome.items():
        gt_regions[seq_id] = [{'start': s, 'end': e} for s, e in sorted(region_set)]

    return gt_regions


# =============================================================================
# PHASE B & D: Call Regions and Evaluate
# =============================================================================

def call_regions(activations, threshold, max_gap, merge_distance, min_region_size):
    """Apply threshold → cluster → merge → filter pipeline to get predicted regions.

    Args:
        activations: 1D genome-wide activation array
        threshold: activation threshold for calling positions
        max_gap: max gap for clustering positions
        merge_distance: max distance for merging regions
        min_region_size: minimum region size in bp

    Returns:
        list of (start, end) tuples
    """
    positions = np.where(activations > threshold)[0]
    if len(positions) == 0:
        return []

    regions = cluster_positions_simple(positions, max_gap=max_gap)
    regions = merge_nearby_regions(regions, merge_distance=merge_distance)
    regions = filter_by_size(regions, min_size=min_region_size)
    return regions


# =============================================================================
# PHASE E: Output
# =============================================================================

def save_bed(predicted_regions, output_path):
    """Save predicted regions as BED file across all genomes.

    Args:
        predicted_regions: dict mapping seq_id -> list of (start, end)
        output_path: output BED file path
    """
    region_num = 0
    with open(output_path, 'w') as f:
        for seq_id in sorted(predicted_regions.keys()):
            for start, end in predicted_regions[seq_id]:
                region_num += 1
                name = f"prophage_{region_num}"
                score = 0
                f.write(f"{seq_id}\t{start}\t{end}\t{name}\t{score}\t+\n")


def plot_activation_track(genome_act, predicted_regions, gt_regions, seq_id,
                          output_path, threshold):
    """Plot activation track with predicted and ground truth regions overlaid.

    Args:
        genome_act: 1D activation array for this genome
        predicted_regions: list of (start, end) tuples
        gt_regions: list of {'start': int, 'end': int} dicts
        seq_id: genome identifier
        output_path: output PNG path
        threshold: threshold line value
    """
    seq_len = len(genome_act)

    # Downsample for plotting
    max_points = 50000
    if seq_len > max_points:
        bin_size = seq_len // max_points
        n_bins = seq_len // bin_size
        truncated = genome_act[:n_bins * bin_size]
        reshaped = truncated.reshape(n_bins, bin_size)
        plot_acts = reshaped.max(axis=1)
        x_coords = np.arange(n_bins) * bin_size + bin_size // 2
    else:
        plot_acts = genome_act
        x_coords = np.arange(seq_len)

    fig, axes = plt.subplots(3, 1, figsize=(20, 8), height_ratios=[3, 1, 1], sharex=True)

    # Activation track
    ax1 = axes[0]
    ax1.fill_between(x_coords, 0, plot_acts, alpha=0.3, color='blue')
    ax1.plot(x_coords, plot_acts, lw=0.5, alpha=0.9, color='blue')
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold:.2f})')
    ax1.set_ylabel('SAE Activation')
    ax1.set_title(f'{seq_id}  (length: {seq_len:,} bp)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, seq_len)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Predicted regions
    ax2 = axes[1]
    ax2.set_xlim(0, seq_len)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Predicted')
    ax2.set_yticks([])
    for start, end in predicted_regions:
        ax2.axvspan(start, end, alpha=0.7, color='green')
        mid = (start + end) / 2
        size_kb = (end - start) / 1000
        ax2.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center',
                 fontsize=7, color='white', fontweight='bold')
    if not predicted_regions:
        ax2.text(0.5, 0.5, 'No regions predicted', transform=ax2.transAxes,
                 ha='center', va='center', color='gray')

    # Ground truth
    ax3 = axes[2]
    ax3.set_xlim(0, seq_len)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Ground\nTruth')
    ax3.set_xlabel('Genomic Position (bp)')
    ax3.set_yticks([])
    for gt in gt_regions:
        start, end = gt['start'], gt['end']
        if start < seq_len:
            ax3.axvspan(start, min(end, seq_len), alpha=0.7, color='red')
            mid = (start + min(end, seq_len)) / 2
            size_kb = (min(end, seq_len) - start) / 1000
            ax3.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center',
                     fontsize=7, color='white', fontweight='bold')
    if not gt_regions:
        ax3.text(0.5, 0.5, 'No ground truth', transform=ax3.transAxes,
                 ha='center', va='center', color='gray')

    # Format x-axis
    def format_mb(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f} Mb'
        elif x >= 1e3:
            return f'{x/1e3:.0f} kb'
        return f'{x:.0f}'
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_mb))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nucleotide-level prophage detection from segment SAE activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument("--input_csv", required=True,
                        help="Output CSV from sae_inference.py (must have seq_id, start, end columns)")
    parser.add_argument("--activations_dir", required=True,
                        help="Directory with per-segment .npy activation files")
    parser.add_argument("--output_dir", default="./nucleotide_eval_results",
                        help="Output directory for results")
    parser.add_argument("--output_prefix", default="nucleotide_eval",
                        help="Prefix for output files")

    # Feature selection
    parser.add_argument("--feature_idx", type=int, default=19746,
                        help="SAE feature index to extract from multi-feature .npy files")

    # Thresholding
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Activation threshold for calling positions")
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use per-genome adaptive threshold instead of fixed")
    parser.add_argument("--adaptive_method", type=str, default="mad",
                        choices=['mad', 'std', 'percentile'],
                        help="Adaptive threshold method")
    parser.add_argument("--adaptive_k", type=float, default=3.0,
                        help="Adaptive threshold sensitivity (multiplier for mad/std, or percentile value)")

    # Normalization
    parser.add_argument("--normalization", type=str, default="none",
                        choices=['none', 'zscore', 'robust_zscore', 'percentile',
                                 'local_baseline', 'minmax', 'quantile'],
                        help="Normalization method for genome-wide activations")
    parser.add_argument("--norm_window", type=int, default=10000,
                        help="Window size for local_baseline normalization")

    # Region calling
    parser.add_argument("--max_gap", type=int, default=100,
                        help="Max gap between above-threshold positions for clustering (bp)")
    parser.add_argument("--merge_distance", type=int, default=3000,
                        help="Max distance for merging nearby regions (bp)")
    parser.add_argument("--min_region_size", type=int, default=1000,
                        help="Minimum prophage region size (bp)")

    # Output options
    parser.add_argument("--plot", action="store_true",
                        help="Generate per-genome activation track plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Nucleotide-Level Prophage Evaluation")
    print("=" * 60)
    print(f"Input CSV:       {args.input_csv}")
    print(f"Activations dir: {args.activations_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Feature index:   {args.feature_idx}")
    print(f"Threshold:       {args.threshold}" + (" (adaptive)" if args.adaptive_threshold else " (fixed)"))
    if args.adaptive_threshold:
        print(f"  Method: {args.adaptive_method}, k={args.adaptive_k}")
    if args.normalization != 'none':
        print(f"Normalization:   {args.normalization}" +
              (f" (window={args.norm_window})" if args.normalization == 'local_baseline' else ""))
    print(f"Max gap:         {args.max_gap} bp")
    print(f"Merge distance:  {args.merge_distance} bp")
    print(f"Min region size: {args.min_region_size} bp")
    print()

    # ---- Phase A: Load segment metadata ----
    print("Phase A: Loading segment metadata...")
    segments = load_segment_metadata(args.input_csv)
    seq_ids = sorted(set(s['seq_id'] for s in segments))
    print(f"  {len(segments)} segments across {len(seq_ids)} genomes")

    # ---- Phase A: Stitch activations ----
    print("\nPhase A: Stitching segment activations into genome-wide tracks...")
    genome_activations = stitch_activations(segments, args.activations_dir, args.feature_idx)
    for seq_id, act in sorted(genome_activations.items()):
        print(f"  {seq_id}: {len(act):,} bp, max={act.max():.4f}, mean={act.mean():.6f}")

    # ---- Phase C: Extract ground truth ----
    print("\nPhase C: Extracting ground truth prophage regions...")
    gt_regions = extract_ground_truth(segments)
    genomes_with_gt = {sid for sid, regions in gt_regions.items() if regions}
    print(f"  {len(genomes_with_gt)} genomes with ground truth regions")
    for seq_id in sorted(genomes_with_gt):
        regions = gt_regions[seq_id]
        total_bp = sum(r['end'] - r['start'] for r in regions)
        print(f"  {seq_id}: {len(regions)} regions, {total_bp:,} bp total")

    # ---- Phase B: Normalize, threshold, call regions ----
    print("\nPhase B: Calling prophage regions...")
    all_predicted = {}
    per_genome_thresholds = {}

    for seq_id in sorted(genome_activations.keys()):
        act = genome_activations[seq_id]

        # Normalize
        if args.normalization != 'none':
            act = normalize_activations(act, method=args.normalization, window_size=args.norm_window)
            genome_activations[seq_id] = act  # store normalized for plotting

        # Determine threshold
        if args.adaptive_threshold:
            threshold = compute_adaptive_threshold(act, method=args.adaptive_method, k=args.adaptive_k)
        else:
            threshold = args.threshold

        per_genome_thresholds[seq_id] = threshold

        # Call regions
        predicted = call_regions(act, threshold, args.max_gap, args.merge_distance, args.min_region_size)
        all_predicted[seq_id] = predicted

        n_regions = len(predicted)
        total_bp = sum(e - s for s, e in predicted)
        print(f"  {seq_id}: threshold={threshold:.4f}, {n_regions} regions, {total_bp:,} bp predicted")

    # ---- Phase D: Nucleotide-level evaluation ----
    print("\nPhase D: Nucleotide-level evaluation...")
    per_genome_metrics = []

    # Aggregate TP/FP/FN/TN across all genomes
    agg_tp, agg_fp, agg_fn, agg_tn = 0, 0, 0, 0

    for seq_id in sorted(genome_activations.keys()):
        seq_len = len(genome_activations[seq_id])
        predicted = all_predicted.get(seq_id, [])
        gt = gt_regions.get(seq_id, [])

        if not gt:
            # No ground truth — record but skip evaluation
            per_genome_metrics.append({
                'seq_id': seq_id,
                'genome_length': seq_len,
                'num_predicted_regions': len(predicted),
                'num_gt_regions': 0,
                'has_ground_truth': False,
                'threshold': per_genome_thresholds.get(seq_id, args.threshold),
            })
            continue

        metrics = calculate_metrics(predicted, gt, seq_len)

        per_genome_metrics.append({
            'seq_id': seq_id,
            'genome_length': seq_len,
            'num_predicted_regions': len(predicted),
            'num_gt_regions': len(gt),
            'has_ground_truth': True,
            'threshold': per_genome_thresholds.get(seq_id, args.threshold),
            **metrics,
        })

        agg_tp += metrics['tp']
        agg_fp += metrics['fp']
        agg_fn += metrics['fn']
        agg_tn += metrics['tn']

        print(f"  {seq_id}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} MCC={metrics['mcc']:.3f} Jaccard={metrics['jaccard']:.3f}")

    # Aggregate metrics (micro-averaged over nucleotides)
    agg_tp, agg_fp, agg_fn, agg_tn = float(agg_tp), float(agg_fp), float(agg_fn), float(agg_tn)
    agg_precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    agg_recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0
    agg_jaccard = agg_tp / (agg_tp + agg_fp + agg_fn) if (agg_tp + agg_fp + agg_fn) > 0 else 0.0
    denom_p1 = np.sqrt((agg_tp + agg_fp) * (agg_tp + agg_fn))
    denom_p2 = np.sqrt((agg_tn + agg_fp) * (agg_tn + agg_fn))
    agg_mcc_denom = denom_p1 * denom_p2
    agg_mcc = (agg_tp * agg_tn - agg_fp * agg_fn) / agg_mcc_denom if agg_mcc_denom > 0 else 0.0

    # Also compute macro-averaged metrics (mean over genomes with GT)
    gt_metrics = [m for m in per_genome_metrics if m.get('has_ground_truth')]
    if gt_metrics:
        macro_precision = np.mean([m['precision'] for m in gt_metrics])
        macro_recall = np.mean([m['recall'] for m in gt_metrics])
        macro_f1 = np.mean([m['f1'] for m in gt_metrics])
        macro_mcc = np.mean([m['mcc'] for m in gt_metrics])
        macro_jaccard = np.mean([m['jaccard'] for m in gt_metrics])
    else:
        macro_precision = macro_recall = macro_f1 = macro_mcc = macro_jaccard = 0.0

    aggregate = {
        'micro': {
            'precision': agg_precision,
            'recall': agg_recall,
            'f1': agg_f1,
            'mcc': agg_mcc,
            'jaccard': agg_jaccard,
            'tp': int(agg_tp),
            'fp': int(agg_fp),
            'fn': int(agg_fn),
            'tn': int(agg_tn),
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'mcc': macro_mcc,
            'jaccard': macro_jaccard,
        },
        'num_genomes': len(genome_activations),
        'num_genomes_with_gt': len(gt_metrics),
        'total_predicted_regions': sum(len(v) for v in all_predicted.values()),
        'total_gt_regions': sum(len(v) for v in gt_regions.values()),
        'parameters': {
            'feature_idx': args.feature_idx,
            'threshold': args.threshold,
            'adaptive_threshold': args.adaptive_threshold,
            'normalization': args.normalization,
            'max_gap': args.max_gap,
            'merge_distance': args.merge_distance,
            'min_region_size': args.min_region_size,
        },
    }

    # ---- Phase E: Save outputs ----
    print("\nPhase E: Saving results...")

    prefix = args.output_prefix

    # Per-genome metrics CSV
    metrics_csv_path = output_dir / f"{prefix}_per_genome.csv"
    metric_fields = ['seq_id', 'genome_length', 'num_predicted_regions', 'num_gt_regions',
                     'has_ground_truth', 'threshold', 'precision', 'recall', 'f1', 'mcc',
                     'jaccard', 'tp', 'fp', 'fn', 'tn', 'gt_regions', 'pred_regions']
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields, extrasaction='ignore')
        writer.writeheader()
        for m in per_genome_metrics:
            writer.writerow(m)
    print(f"  Per-genome metrics: {metrics_csv_path}")

    # Aggregate metrics JSON
    agg_json_path = output_dir / f"{prefix}_aggregate.json"
    with open(agg_json_path, 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"  Aggregate metrics:  {agg_json_path}")

    # BED file of predicted regions
    bed_path = output_dir / f"{prefix}_predicted.bed"
    save_bed(all_predicted, bed_path)
    print(f"  Predicted BED:      {bed_path}")

    # Optional: plots
    if args.plot:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        print(f"  Generating plots in {plots_dir}/...")
        for seq_id in sorted(genome_activations.keys()):
            threshold = per_genome_thresholds.get(seq_id, args.threshold)
            plot_activation_track(
                genome_activations[seq_id],
                all_predicted.get(seq_id, []),
                gt_regions.get(seq_id, []),
                seq_id,
                plots_dir / f"{seq_id}_activation_track.png",
                threshold,
            )
        print(f"  {len(genome_activations)} plots saved")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Genomes processed:      {len(genome_activations)}")
    print(f"Genomes with GT:        {len(gt_metrics)}")
    print(f"Total predicted regions: {aggregate['total_predicted_regions']}")
    print(f"Total GT regions:        {aggregate['total_gt_regions']}")

    if gt_metrics:
        print(f"\nMicro-averaged metrics (nucleotide-level, across all genomes):")
        m = aggregate['micro']
        print(f"  Precision: {m['precision']:.1%}")
        print(f"  Recall:    {m['recall']:.1%}")
        print(f"  F1:        {m['f1']:.1%}")
        print(f"  MCC:       {m['mcc']:.3f}")
        print(f"  Jaccard:   {m['jaccard']:.3f}")
        print(f"  TP: {m['tp']:,}  FP: {m['fp']:,}  FN: {m['fn']:,}  TN: {m['tn']:,}")

        print(f"\nMacro-averaged metrics (mean over {len(gt_metrics)} genomes):")
        m = aggregate['macro']
        print(f"  Precision: {m['precision']:.1%}")
        print(f"  Recall:    {m['recall']:.1%}")
        print(f"  F1:        {m['f1']:.1%}")
        print(f"  MCC:       {m['mcc']:.3f}")
        print(f"  Jaccard:   {m['jaccard']:.3f}")
    else:
        print("\nNo genomes with ground truth — metrics not computed")

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
