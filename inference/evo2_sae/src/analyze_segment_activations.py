#!/usr/bin/env python3
"""
Analyze SAE activation distributions across segment datasets to determine
optimal thresholds for prophage detection.

Loads result CSVs from sae_inference.py and produces:
  1. Activation distribution plots (max, mean, fraction) per dataset
  2. ROC and Precision-Recall curves (for datasets with labels)
  3. Threshold sweep analysis showing metrics vs threshold
  4. Cross-dataset comparison of activation distributions
  5. Recommended thresholds based on the analysis

Expected datasets:
  - bacteria_only: All negative (no prophage) — for false positive analysis
  - phage_only: All positive — for true positive / sensitivity analysis
  - test_set: Mixed labels — control to verify inference works
  - gc_control: Shuffled nucleotides (same GC content) — should be negative

Usage:
    # Single file
    python src/analyze_segment_activations.py \
        --input results/test_results.csv \
        --output_dir ./segment_analysis

    # Multiple files (auto-detected dataset names from filenames)
    python src/analyze_segment_activations.py \
        --input results/bacteria_results.csv results/phage_results.csv \
               results/test_results.csv results/gc_control_results.csv \
        --output_dir ./segment_analysis

    # Entire directory
    python src/analyze_segment_activations.py \
        --input_dir results/ \
        --output_dir ./segment_analysis
"""

import argparse
import csv
import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(csv_path):
    """Load a sae_inference.py result CSV.

    Returns:
        dict with keys: path, name, rows (list of dicts),
        max_activations, mean_activations, fractions, labels (numpy arrays)
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"  WARNING: {csv_path} is empty")
        return None

    # Extract numeric arrays
    max_acts = np.array([float(r['max_activation']) for r in rows])
    mean_acts = np.array([float(r['mean_activation']) for r in rows])
    fractions = np.array([float(r['fraction_firing']) for r in rows])

    # Labels (may not exist for all datasets)
    labels = None
    if 'label' in columns:
        try:
            labels = np.array([int(r['label']) for r in rows])
        except (ValueError, KeyError):
            labels = None

    pred_labels = None
    if 'pred_label' in columns:
        try:
            pred_labels = np.array([int(r['pred_label']) for r in rows])
        except (ValueError, KeyError):
            pred_labels = None

    # Extract seq_id for per-genome analysis (may not exist)
    seq_ids = None
    if 'seq_id' in columns:
        seq_ids = [r['seq_id'] for r in rows]
    elif 'source' in columns:
        # Fall back to 'source' as a grouping key
        seq_ids = [r['source'] for r in rows]

    return {
        'path': str(csv_path),
        'name': Path(csv_path).stem,
        'columns': columns,
        'n_segments': len(rows),
        'max_activations': max_acts,
        'mean_activations': mean_acts,
        'fractions': fractions,
        'labels': labels,
        'pred_labels': pred_labels,
        'seq_ids': seq_ids,
    }


def guess_dataset_type(filename):
    """Guess dataset type from filename."""
    name = filename.lower()
    if 'bacter' in name:
        return 'bacteria_only'
    elif 'phage' in name and 'prophage' not in name:
        return 'phage_only'
    elif 'gc' in name and ('control' in name or 'shuffle' in name or 'content' in name):
        return 'gc_control'
    elif 'test' in name:
        return 'test_set'
    elif 'train' in name:
        return 'train_set'
    elif 'dev' in name or 'val' in name:
        return 'dev_set'
    return 'unknown'


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_metrics_at_threshold(labels, scores, threshold):
    """Compute classification metrics at a given threshold."""
    preds = (scores > threshold).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # MCC
    denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'fpr': fpr,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def threshold_sweep(labels, scores, metric_name, n_points=200):
    """Sweep thresholds and compute metrics at each point.

    Returns:
        list of metric dicts, sorted by threshold
    """
    # Use score distribution to set threshold range
    min_t = 0.0
    max_t = max(np.percentile(scores, 99.5), np.max(scores) * 0.5)
    if max_t <= min_t:
        max_t = 1.0

    thresholds = np.linspace(min_t, max_t, n_points)
    results = []
    for t in thresholds:
        m = compute_metrics_at_threshold(labels, scores, t)
        m['metric_name'] = metric_name
        results.append(m)
    return results


def compute_roc(labels, scores):
    """Compute ROC curve points (FPR, TPR) at various thresholds."""
    thresholds = np.sort(np.unique(scores))
    # Sample if too many unique values
    if len(thresholds) > 500:
        idx = np.linspace(0, len(thresholds) - 1, 500, dtype=int)
        thresholds = thresholds[idx]

    fprs, tprs = [], []
    for t in thresholds:
        preds = (scores > t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)

    # Add endpoints
    fprs = [1.0] + fprs + [0.0]
    tprs = [1.0] + tprs + [0.0]

    return np.array(fprs), np.array(tprs), thresholds


def compute_pr_curve(labels, scores):
    """Compute Precision-Recall curve points."""
    thresholds = np.sort(np.unique(scores))
    if len(thresholds) > 500:
        idx = np.linspace(0, len(thresholds) - 1, 500, dtype=int)
        thresholds = thresholds[idx]

    precisions, recalls = [], []
    for t in thresholds:
        preds = (scores > t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    return np.array(precisions), np.array(recalls), thresholds


# =============================================================================
# PER-GENOME ANALYSIS
# =============================================================================

def compute_per_genome_stats(data):
    """Group segments by genome (seq_id) and compute per-genome activation statistics.

    Returns:
        list of dicts, one per genome, with statistics and classification info
    """
    seq_ids = data.get('seq_ids')
    if seq_ids is None:
        return None

    max_acts = data['max_activations']
    mean_acts = data['mean_activations']
    fractions = data['fractions']
    labels = data.get('labels')
    pred_labels = data.get('pred_labels')

    # Group indices by seq_id
    genome_indices = defaultdict(list)
    for i, sid in enumerate(seq_ids):
        genome_indices[sid].append(i)

    genome_stats = []
    for sid in sorted(genome_indices.keys()):
        idx = np.array(genome_indices[sid])
        g_max = max_acts[idx]
        g_mean = mean_acts[idx]
        g_frac = fractions[idx]

        stats = {
            'seq_id': sid,
            'n_segments': len(idx),
            # Baseline and noise characterization
            'max_act_median': float(np.median(g_max)),
            'max_act_mean': float(np.mean(g_max)),
            'max_act_std': float(np.std(g_max)),
            'max_act_max': float(np.max(g_max)),
            'max_act_p90': float(np.percentile(g_max, 90)),
            'max_act_p95': float(np.percentile(g_max, 95)),
            'max_act_iqr': float(np.percentile(g_max, 75) - np.percentile(g_max, 25)),
            'mean_act_median': float(np.median(g_mean)),
            'mean_act_mean': float(np.mean(g_mean)),
            'mean_act_std': float(np.std(g_mean)),
            'mean_act_max': float(np.max(g_mean)),
            'frac_median': float(np.median(g_frac)),
            'frac_mean': float(np.mean(g_frac)),
            'frac_max': float(np.max(g_frac)),
            # Signal-to-noise: ratio of max to median (higher = cleaner signal)
            'snr_max': float(np.max(g_max) / np.median(g_max)) if np.median(g_max) > 0 else float('inf'),
            # Fraction of segments with any signal
            'pct_segments_above_0': float(np.mean(g_max > 0) * 100),
            'pct_segments_above_0.5': float(np.mean(g_max > 0.5) * 100),
            'pct_segments_above_1.0': float(np.mean(g_max > 1.0) * 100),
        }

        # Per-genome adaptive threshold (MAD-based)
        mad = np.median(np.abs(g_max - np.median(g_max))) * 1.4826
        stats['mad'] = float(mad)
        stats['adaptive_threshold_3mad'] = float(np.median(g_max) + 3 * mad)
        stats['adaptive_threshold_5mad'] = float(np.median(g_max) + 5 * mad)

        # Label info if available
        if labels is not None:
            g_labels = labels[idx]
            n_pos = int(np.sum(g_labels == 1))
            n_neg = int(np.sum(g_labels == 0))
            stats['n_positive'] = n_pos
            stats['n_negative'] = n_neg
            stats['pct_positive'] = float(n_pos / len(idx) * 100) if len(idx) > 0 else 0

            # Per-genome metrics at current predictions
            if pred_labels is not None:
                g_preds = pred_labels[idx]
                tp = int(np.sum((g_preds == 1) & (g_labels == 1)))
                fp = int(np.sum((g_preds == 1) & (g_labels == 0)))
                fn = int(np.sum((g_preds == 0) & (g_labels == 1)))
                tn = int(np.sum((g_preds == 0) & (g_labels == 0)))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                stats['current_tp'] = tp
                stats['current_fp'] = fp
                stats['current_fn'] = fn
                stats['current_tn'] = tn
                stats['current_precision'] = float(prec)
                stats['current_recall'] = float(rec)
                stats['current_f1'] = float(f1)

            # Separation between positive and negative segments within this genome
            if n_pos > 0 and n_neg > 0:
                pos_max_median = float(np.median(g_max[g_labels == 1]))
                neg_max_median = float(np.median(g_max[g_labels == 0]))
                stats['pos_max_median'] = pos_max_median
                stats['neg_max_median'] = neg_max_median
                stats['class_separation'] = pos_max_median - neg_max_median

                # Per-genome optimal threshold (sweep on max_activation)
                best_f1 = 0
                best_t = 0
                for t in np.linspace(0, np.max(g_max), 50):
                    preds = (g_max > t).astype(int)
                    tp_ = np.sum((preds == 1) & (g_labels == 1))
                    fp_ = np.sum((preds == 1) & (g_labels == 0))
                    fn_ = np.sum((preds == 0) & (g_labels == 1))
                    p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
                    r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
                    f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
                    if f1_ > best_f1:
                        best_f1 = f1_
                        best_t = float(t)
                stats['optimal_threshold_max'] = best_t
                stats['optimal_f1_max'] = float(best_f1)

        genome_stats.append(stats)

    return genome_stats


def plot_per_genome_analysis(datasets, output_dir):
    """Generate per-genome analysis plots."""
    genome_dir = output_dir / 'per_genome'
    genome_dir.mkdir(exist_ok=True)

    for name, data in datasets.items():
        genome_stats = compute_per_genome_stats(data)
        if genome_stats is None:
            continue

        data['genome_stats'] = genome_stats
        n_genomes = len(genome_stats)

        if n_genomes < 2:
            continue

        print(f"  {name}: {n_genomes} genomes")

        # --- Plot 1: Per-genome activation profiles (sorted by median) ---
        sorted_stats = sorted(genome_stats, key=lambda g: g['max_act_median'])
        sids = [g['seq_id'][:20] for g in sorted_stats]  # truncate long names
        medians = [g['max_act_median'] for g in sorted_stats]
        p90s = [g['max_act_p90'] for g in sorted_stats]
        maxes = [g['max_act_max'] for g in sorted_stats]
        stds = [g['max_act_std'] for g in sorted_stats]

        fig, ax = plt.subplots(figsize=(max(14, n_genomes * 0.3), 6))
        x = np.arange(n_genomes)
        ax.bar(x, medians, color='steelblue', alpha=0.7, label='Median')
        ax.scatter(x, p90s, color='orange', s=20, zorder=3, label='P90')
        ax.scatter(x, maxes, color='red', s=15, zorder=3, marker='^', label='Max')
        ax.errorbar(x, medians, yerr=stds, fmt='none', color='gray', alpha=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sids, rotation=90, fontsize=6)
        ax.set_ylabel('Max Activation')
        ax.set_title(f'{name}: Per-Genome Activation Profiles (sorted by median)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(genome_dir / f'genome_profiles_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- Plot 2: Signal-to-noise ratio per genome ---
        snrs = [min(g['snr_max'], 100) for g in sorted_stats]  # cap at 100 for plotting
        fig, ax = plt.subplots(figsize=(max(14, n_genomes * 0.3), 5))
        colors_snr = ['green' if s > 10 else 'orange' if s > 3 else 'red' for s in snrs]
        ax.bar(x, snrs, color=colors_snr, alpha=0.7)
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Clean (SNR>10)')
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='Moderate (SNR>3)')
        ax.set_xticks(x)
        ax.set_xticklabels(sids, rotation=90, fontsize=6)
        ax.set_ylabel('Signal-to-Noise Ratio (max/median)')
        ax.set_title(f'{name}: Per-Genome Signal-to-Noise Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(genome_dir / f'genome_snr_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- Plot 3: Adaptive vs fixed threshold per genome ---
        adaptive_3mad = [g['adaptive_threshold_3mad'] for g in sorted_stats]
        adaptive_5mad = [g['adaptive_threshold_5mad'] for g in sorted_stats]

        fig, ax = plt.subplots(figsize=(max(14, n_genomes * 0.3), 5))
        ax.plot(x, adaptive_3mad, 'o-', color='purple', markersize=4, label='Adaptive (3*MAD)')
        ax.plot(x, adaptive_5mad, 's-', color='darkred', markersize=4, label='Adaptive (5*MAD)')
        ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Fixed (0.5)')
        ax.axhline(y=0.1, color='cyan', linestyle='--', alpha=0.5, label='Fixed (0.1)')
        ax.fill_between(x, medians, maxes, alpha=0.1, color='gray', label='Median–Max range')
        ax.set_xticks(x)
        ax.set_xticklabels(sids, rotation=90, fontsize=6)
        ax.set_ylabel('Threshold Value')
        ax.set_title(f'{name}: Adaptive vs Fixed Thresholds Per Genome')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(genome_dir / f'genome_thresholds_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # --- Plot 4: Per-genome optimal threshold (if labels available) ---
        has_optimal = any('optimal_threshold_max' in g for g in sorted_stats)
        if has_optimal:
            opt_thresholds = [g.get('optimal_threshold_max', 0) for g in sorted_stats]
            opt_f1s = [g.get('optimal_f1_max', 0) for g in sorted_stats]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n_genomes * 0.3), 8),
                                            sharex=True)

            ax1.bar(x, opt_thresholds, color='teal', alpha=0.7)
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Current fixed (0.5)')
            ax1.axhline(y=np.median(opt_thresholds), color='green', linestyle='--', alpha=0.5,
                        label=f'Median optimal ({np.median(opt_thresholds):.3f})')
            ax1.set_ylabel('Optimal Threshold (max_activation)')
            ax1.set_title(f'{name}: Per-Genome Optimal Thresholds')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')

            # Color F1 bars by quality
            colors_f1 = ['green' if f > 0.7 else 'orange' if f > 0.3 else 'red' for f in opt_f1s]
            ax2.bar(x, opt_f1s, color=colors_f1, alpha=0.7)
            ax2.set_ylabel('Best Achievable F1')
            ax2.set_title(f'{name}: Best F1 per Genome (with optimal per-genome threshold)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(sids, rotation=90, fontsize=6)
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(genome_dir / f'genome_optimal_{name}.png', dpi=150, bbox_inches='tight')
            plt.close()

        # --- Plot 5: Positive vs negative activation distributions per genome ---
        has_separation = any('class_separation' in g for g in sorted_stats)
        if has_separation:
            fig, ax = plt.subplots(figsize=(max(14, n_genomes * 0.3), 5))
            pos_meds = [g.get('pos_max_median', 0) for g in sorted_stats]
            neg_meds = [g.get('neg_max_median', 0) for g in sorted_stats]
            seps = [g.get('class_separation', 0) for g in sorted_stats]

            width = 0.35
            ax.bar(x - width/2, neg_meds, width, color='blue', alpha=0.6, label='Non-prophage median')
            ax.bar(x + width/2, pos_meds, width, color='red', alpha=0.6, label='Prophage median')
            ax.set_xticks(x)
            ax.set_xticklabels(sids, rotation=90, fontsize=6)
            ax.set_ylabel('Max Activation (median)')
            ax.set_title(f'{name}: Per-Genome Class Separation (prophage vs non-prophage)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(genome_dir / f'genome_separation_{name}.png', dpi=150, bbox_inches='tight')
            plt.close()

        # --- Plot 6: Fixed threshold F1 vs adaptive threshold F1 per genome ---
        if has_optimal:
            # Compare: current F1 (fixed) vs best achievable F1 (per-genome optimal)
            has_current = any('current_f1' in g for g in sorted_stats)
            if has_current:
                current_f1s = [g.get('current_f1', 0) for g in sorted_stats]

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(current_f1s, opt_f1s, c='teal', alpha=0.6, s=40)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (no improvement)')
                ax.set_xlabel('F1 with Fixed Threshold')
                ax.set_ylabel('F1 with Per-Genome Optimal Threshold')
                ax.set_title(f'{name}: Headroom from Adaptive Thresholds')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)

                # Annotate worst performers
                improvements = np.array(opt_f1s) - np.array(current_f1s)
                worst_idx = np.argsort(improvements)[-5:]  # top 5 most improved
                for i in worst_idx:
                    if improvements[i] > 0.05:
                        ax.annotate(sids[i], (current_f1s[i], opt_f1s[i]),
                                    fontsize=7, alpha=0.7)

                avg_improvement = np.mean(improvements)
                ax.text(0.05, 0.95, f'Mean improvement: {avg_improvement:+.3f}',
                        transform=ax.transAxes, fontsize=10, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

                plt.tight_layout()
                plt.savefig(genome_dir / f'genome_headroom_{name}.png', dpi=150, bbox_inches='tight')
                plt.close()

    # Save per-genome stats CSV
    all_genome_stats = []
    for name, data in datasets.items():
        if 'genome_stats' not in data:
            continue
        for g in data['genome_stats']:
            g['dataset'] = name
            all_genome_stats.append(g)

    if all_genome_stats:
        csv_path = genome_dir / 'per_genome_stats.csv'
        fieldnames = list(all_genome_stats[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for g in all_genome_stats:
                writer.writerow(g)
        print(f"  Per-genome stats saved to: {csv_path}")


def print_per_genome_summary(datasets):
    """Print per-genome summary highlighting problematic genomes."""
    for name, data in datasets.items():
        genome_stats = data.get('genome_stats')
        if genome_stats is None:
            continue

        n_genomes = len(genome_stats)
        if n_genomes == 0:
            continue

        print(f"\n  {'─' * 55}")
        print(f"  Per-Genome Analysis: {name} ({n_genomes} genomes)")

        # Categorize genomes by signal quality
        medians = [g['max_act_median'] for g in genome_stats]
        stds = [g['max_act_std'] for g in genome_stats]

        print(f"\n  Baseline activation (max_act median across genomes):")
        print(f"    Range: {min(medians):.4f} – {max(medians):.4f}")
        print(f"    Mean:  {np.mean(medians):.4f} ± {np.std(medians):.4f}")
        print(f"    This {'varies a lot' if np.std(medians) > np.mean(medians) * 0.5 else 'is relatively stable'} across genomes")

        # Noisy vs clean
        snrs = [g['snr_max'] for g in genome_stats]
        clean = [g for g in genome_stats if g['snr_max'] > 10]
        moderate = [g for g in genome_stats if 3 <= g['snr_max'] <= 10]
        noisy = [g for g in genome_stats if g['snr_max'] < 3]
        print(f"\n  Signal quality (SNR = max/median):")
        print(f"    Clean (SNR > 10):    {len(clean)} genomes")
        print(f"    Moderate (3-10):     {len(moderate)} genomes")
        print(f"    Noisy (SNR < 3):     {len(noisy)} genomes")

        if noisy:
            print(f"    Noisiest genomes:")
            for g in sorted(noisy, key=lambda g: g['snr_max'])[:5]:
                print(f"      {g['seq_id']}: SNR={g['snr_max']:.1f}, "
                      f"median={g['max_act_median']:.4f}, max={g['max_act_max']:.4f}")

        # Adaptive threshold variation
        adaptive_ts = [g['adaptive_threshold_3mad'] for g in genome_stats]
        print(f"\n  Adaptive threshold (median + 3*MAD) across genomes:")
        print(f"    Range: {min(adaptive_ts):.4f} – {max(adaptive_ts):.4f}")
        print(f"    Mean:  {np.mean(adaptive_ts):.4f}")
        print(f"    → A fixed threshold of 0.5 is {'too high' if np.mean(adaptive_ts) < 0.5 else 'too low' if np.mean(adaptive_ts) > 1.0 else 'in the right ballpark'} for most genomes")

        # Per-genome optimal thresholds
        has_optimal = any('optimal_threshold_max' in g for g in genome_stats)
        if has_optimal:
            opt_ts = [g['optimal_threshold_max'] for g in genome_stats if 'optimal_threshold_max' in g]
            opt_f1s = [g['optimal_f1_max'] for g in genome_stats if 'optimal_f1_max' in g]
            print(f"\n  Per-genome optimal thresholds (on max_activation):")
            print(f"    Threshold range: {min(opt_ts):.4f} – {max(opt_ts):.4f}")
            print(f"    Threshold mean:  {np.mean(opt_ts):.4f}")
            print(f"    Best achievable F1 range: {min(opt_f1s):.3f} – {max(opt_f1s):.3f}")
            print(f"    Best achievable F1 mean:  {np.mean(opt_f1s):.3f}")

            # Compare to current fixed threshold performance
            has_current = any('current_f1' in g for g in genome_stats)
            if has_current:
                current_f1s = [g['current_f1'] for g in genome_stats if 'current_f1' in g]
                improvement = np.mean(opt_f1s) - np.mean(current_f1s)
                print(f"\n    Current fixed threshold F1 mean: {np.mean(current_f1s):.3f}")
                print(f"    Improvement with adaptive:       {improvement:+.3f}")
                if improvement > 0.05:
                    print(f"    → Significant headroom! Adaptive thresholds would help.")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_distributions(datasets, output_dir):
    """Plot activation distributions for each metric across all datasets."""
    metrics = [
        ('max_activations', 'Max Activation'),
        ('mean_activations', 'Mean Activation'),
        ('fractions', 'Fraction Firing'),
    ]

    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 3 * len(datasets)),
                                 sharex=False)
        if len(datasets) == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, datasets.items()):
            values = data[metric_key]
            labels = data.get('labels')

            if labels is not None:
                pos_vals = values[labels == 1]
                neg_vals = values[labels == 0]

                bins = np.linspace(0, max(np.percentile(values, 99.5), 0.01), 80)
                if len(neg_vals) > 0:
                    ax.hist(neg_vals, bins=bins, alpha=0.6, color='blue',
                            label=f'Non-prophage (n={len(neg_vals)})', density=True)
                if len(pos_vals) > 0:
                    ax.hist(pos_vals, bins=bins, alpha=0.6, color='red',
                            label=f'Prophage (n={len(pos_vals)})', density=True)
                ax.legend(fontsize=9)
            else:
                bins = np.linspace(0, max(np.percentile(values, 99.5), 0.01), 80)
                ax.hist(values, bins=bins, alpha=0.6, color='gray',
                        label=f'All (n={len(values)})', density=True)
                ax.legend(fontsize=9)

            ax.set_ylabel('Density')
            ax.set_title(f'{name}: {metric_label}', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add summary stats as text
            stats_text = f'median={np.median(values):.4f}  mean={np.mean(values):.4f}  max={np.max(values):.4f}'
            ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', alpha=0.8))

        axes[-1].set_xlabel(metric_label)
        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{metric_key}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_cross_dataset_comparison(datasets, output_dir):
    """Plot all datasets on the same axes for direct comparison."""
    metrics = [
        ('max_activations', 'Max Activation'),
        ('mean_activations', 'Mean Activation'),
        ('fractions', 'Fraction Firing'),
    ]

    colors = {'bacteria_only': 'blue', 'phage_only': 'red', 'test_set': 'green',
              'gc_control': 'orange', 'train_set': 'purple', 'dev_set': 'cyan',
              'unknown': 'gray'}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for name, data in datasets.items():
            values = data[metric_key]
            dtype = data.get('dataset_type', 'unknown')
            color = colors.get(dtype, 'gray')

            # Use CDF for cleaner comparison
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, label=f'{name} (n={len(values)})',
                    color=color, linewidth=1.5)

        ax.set_xlabel(metric_label)
        ax.set_ylabel('CDF')
        ax.set_title(f'{metric_label} — CDF Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_cdf.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also do boxplot comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        box_data = []
        box_labels = []
        for name, data in datasets.items():
            box_data.append(data[metric_key])
            box_labels.append(name)

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
        for patch, (name, data) in zip(bp['boxes'], datasets.items()):
            dtype = data.get('dataset_type', 'unknown')
            patch.set_facecolor(colors.get(dtype, 'gray'))
            patch.set_alpha(0.6)

        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} — Box Plot')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_labeled_separation(datasets, output_dir):
    """For datasets with labels, plot 2D scatter of max_activation vs mean_activation."""
    for name, data in datasets.items():
        labels = data.get('labels')
        if labels is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        combos = [
            ('max_activations', 'mean_activations', 'Max Activation', 'Mean Activation'),
            ('max_activations', 'fractions', 'Max Activation', 'Fraction Firing'),
            ('mean_activations', 'fractions', 'Mean Activation', 'Fraction Firing'),
        ]

        for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, combos):
            pos = labels == 1
            neg = labels == 0
            ax.scatter(data[xkey][neg], data[ykey][neg], c='blue', alpha=0.3,
                       s=10, label=f'Non-prophage ({neg.sum()})')
            ax.scatter(data[xkey][pos], data[ykey][pos], c='red', alpha=0.3,
                       s=10, label=f'Prophage ({pos.sum()})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{name}: Feature Space Separation', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'scatter_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_threshold_sweep(datasets, output_dir):
    """For datasets with labels, plot metrics vs threshold for each activation metric."""
    metric_keys = [
        ('max_activations', 'Max Activation'),
        ('mean_activations', 'Mean Activation'),
        ('fractions', 'Fraction Firing'),
    ]

    for name, data in datasets.items():
        labels = data.get('labels')
        if labels is None:
            continue
        if len(np.unique(labels)) < 2:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        best_thresholds = {}

        for ax, (metric_key, metric_label) in zip(axes, metric_keys):
            scores = data[metric_key]
            sweep = threshold_sweep(labels, scores, metric_label)

            thresholds = [s['threshold'] for s in sweep]
            f1s = [s['f1'] for s in sweep]
            mccs = [s['mcc'] for s in sweep]
            precisions = [s['precision'] for s in sweep]
            recalls = [s['recall'] for s in sweep]
            fprs = [s['fpr'] for s in sweep]

            ax.plot(thresholds, f1s, label='F1', linewidth=2, color='green')
            ax.plot(thresholds, mccs, label='MCC', linewidth=2, color='purple')
            ax.plot(thresholds, precisions, label='Precision', linewidth=1.5,
                    color='blue', linestyle='--')
            ax.plot(thresholds, recalls, label='Recall', linewidth=1.5,
                    color='red', linestyle='--')
            ax.plot(thresholds, fprs, label='FPR', linewidth=1.5,
                    color='orange', linestyle=':')

            # Mark best F1
            best_idx = np.argmax(f1s)
            best_t = thresholds[best_idx]
            best_f1 = f1s[best_idx]
            ax.axvline(x=best_t, color='green', linestyle=':', alpha=0.5)
            ax.text(best_t, best_f1 + 0.02, f'best F1={best_f1:.3f}\nt={best_t:.4f}',
                    fontsize=8, ha='center')

            # Mark best MCC
            best_mcc_idx = np.argmax(mccs)
            best_mcc_t = thresholds[best_mcc_idx]
            best_mcc_val = mccs[best_mcc_idx]
            ax.axvline(x=best_mcc_t, color='purple', linestyle=':', alpha=0.5)

            best_thresholds[metric_key] = {
                'best_f1_threshold': best_t,
                'best_f1': best_f1,
                'best_mcc_threshold': best_mcc_t,
                'best_mcc': best_mcc_val,
                'metrics_at_best_f1': sweep[best_idx],
                'metrics_at_best_mcc': sweep[best_mcc_idx],
            }

            ax.set_xlabel(f'{metric_label} Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'{metric_label}')
            ax.legend(fontsize=8, loc='center right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle(f'{name}: Threshold Sweep (single-metric thresholds)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'threshold_sweep_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save best thresholds
        data['best_thresholds'] = best_thresholds


def plot_roc_pr(datasets, output_dir):
    """Plot ROC and PR curves for datasets with labels."""
    metric_keys = [
        ('max_activations', 'Max Activation'),
        ('mean_activations', 'Mean Activation'),
        ('fractions', 'Fraction Firing'),
    ]

    for name, data in datasets.items():
        labels = data.get('labels')
        if labels is None or len(np.unique(labels)) < 2:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        for col, (metric_key, metric_label) in enumerate(metric_keys):
            scores = data[metric_key]

            # ROC
            fprs, tprs, _ = compute_roc(labels, scores)
            # np.trapezoid in numpy>=2.0, np.trapz in older versions
            _trapz = getattr(np, 'trapezoid', None) or np.trapz
            auc = _trapz(tprs[::-1], fprs[::-1])
            axes[0, col].plot(fprs, tprs, linewidth=2, label=f'AUC={auc:.3f}')
            axes[0, col].plot([0, 1], [0, 1], 'k--', alpha=0.3)
            axes[0, col].set_xlabel('False Positive Rate')
            axes[0, col].set_ylabel('True Positive Rate')
            axes[0, col].set_title(f'ROC — {metric_label}')
            axes[0, col].legend()
            axes[0, col].grid(True, alpha=0.3)

            # PR
            precs, recs, _ = compute_pr_curve(labels, scores)
            axes[1, col].plot(recs, precs, linewidth=2, color='green')
            axes[1, col].set_xlabel('Recall')
            axes[1, col].set_ylabel('Precision')
            axes[1, col].set_title(f'PR — {metric_label}')
            axes[1, col].grid(True, alpha=0.3)

        fig.suptitle(f'{name}: ROC and Precision-Recall Curves', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'roc_pr_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_combined_score_sweep(datasets, output_dir):
    """Test combined scoring: max, mean, fraction with different weights.

    The current sae_inference.py uses OR logic: pred=1 if ANY metric exceeds its threshold.
    This tests whether that's optimal vs. a weighted combination.
    """
    for name, data in datasets.items():
        labels = data.get('labels')
        if labels is None or len(np.unique(labels)) < 2:
            continue

        max_acts = data['max_activations']
        mean_acts = data['mean_activations']
        fracs = data['fractions']

        # Normalize each to [0, 1] for fair combination
        def safe_normalize(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

        max_norm = safe_normalize(max_acts)
        mean_norm = safe_normalize(mean_acts)
        frac_norm = safe_normalize(fracs)

        # Test different combinations
        combos = {
            'max_only': max_norm,
            'mean_only': mean_norm,
            'fraction_only': frac_norm,
            'equal_weight': (max_norm + mean_norm + frac_norm) / 3,
            'max_heavy': 0.6 * max_norm + 0.2 * mean_norm + 0.2 * frac_norm,
            'mean_heavy': 0.2 * max_norm + 0.6 * mean_norm + 0.2 * frac_norm,
            'max_mean': 0.5 * max_norm + 0.5 * mean_norm,
        }

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        best_overall = {'name': '', 'f1': 0, 'mcc': 0, 'threshold': 0}

        for combo_name, scores in combos.items():
            sweep = threshold_sweep(labels, scores, combo_name, n_points=100)
            thresholds = [s['threshold'] for s in sweep]
            f1s = [s['f1'] for s in sweep]

            best_idx = np.argmax(f1s)
            best_f1 = f1s[best_idx]
            best_t = thresholds[best_idx]

            ax.plot(thresholds, f1s, label=f'{combo_name} (best F1={best_f1:.3f} @ {best_t:.3f})',
                    linewidth=1.5)

            if best_f1 > best_overall['f1']:
                best_overall = {'name': combo_name, 'f1': best_f1,
                                'threshold': best_t, 'metrics': sweep[best_idx]}

        ax.set_xlabel('Threshold (on normalized score)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{name}: Combined Score Strategies — F1 vs Threshold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'combined_scores_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        data['best_combined'] = best_overall


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary(datasets, output_dir):
    """Generate a text summary and JSON report."""
    report = {}

    for name, data in datasets.items():
        entry = {
            'file': data['path'],
            'dataset_type': data.get('dataset_type', 'unknown'),
            'n_segments': data['n_segments'],
            'statistics': {},
        }

        for metric_key, metric_label in [('max_activations', 'max_activation'),
                                          ('mean_activations', 'mean_activation'),
                                          ('fractions', 'fraction_firing')]:
            vals = data[metric_key]
            entry['statistics'][metric_label] = {
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75)),
                'p90': float(np.percentile(vals, 90)),
                'p95': float(np.percentile(vals, 95)),
                'p99': float(np.percentile(vals, 99)),
                'pct_zero': float(np.mean(vals == 0) * 100),
                'pct_above_0.1': float(np.mean(vals > 0.1) * 100),
                'pct_above_0.5': float(np.mean(vals > 0.5) * 100),
                'pct_above_1.0': float(np.mean(vals > 1.0) * 100),
            }

        labels = data.get('labels')
        if labels is not None:
            n_pos = int(np.sum(labels == 1))
            n_neg = int(np.sum(labels == 0))
            entry['label_distribution'] = {
                'positive': n_pos,
                'negative': n_neg,
                'pct_positive': float(n_pos / len(labels) * 100) if len(labels) > 0 else 0,
            }

        pred_labels = data.get('pred_labels')
        if pred_labels is not None and labels is not None:
            tp = int(np.sum((pred_labels == 1) & (labels == 1)))
            fp = int(np.sum((pred_labels == 1) & (labels == 0)))
            fn = int(np.sum((pred_labels == 0) & (labels == 1)))
            tn = int(np.sum((pred_labels == 0) & (labels == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
            entry['current_predictions'] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc,
            }

        if 'best_thresholds' in data:
            entry['optimal_single_thresholds'] = {}
            for metric_key, info in data['best_thresholds'].items():
                entry['optimal_single_thresholds'][metric_key] = {
                    'best_f1_threshold': info['best_f1_threshold'],
                    'best_f1': info['best_f1'],
                    'best_mcc_threshold': info['best_mcc_threshold'],
                    'best_mcc': info['best_mcc'],
                }

        if 'best_combined' in data:
            entry['optimal_combined_score'] = data['best_combined']
            # Remove numpy types
            if 'metrics' in entry['optimal_combined_score']:
                del entry['optimal_combined_score']['metrics']

        # Per-genome summary stats
        if 'genome_stats' in data:
            n_g = len(data['genome_stats'])
            medians = [g['max_act_median'] for g in data['genome_stats']]
            adaptive_ts = [g['adaptive_threshold_3mad'] for g in data['genome_stats']]
            entry['per_genome_summary'] = {
                'n_genomes': n_g,
                'baseline_range': [float(min(medians)), float(max(medians))],
                'baseline_mean': float(np.mean(medians)),
                'baseline_std': float(np.std(medians)),
                'adaptive_threshold_3mad_range': [float(min(adaptive_ts)), float(max(adaptive_ts))],
                'adaptive_threshold_3mad_mean': float(np.mean(adaptive_ts)),
            }
            # Include per-genome optimal F1 if available
            opt_f1s = [g.get('optimal_f1_max') for g in data['genome_stats'] if 'optimal_f1_max' in g]
            if opt_f1s:
                entry['per_genome_summary']['optimal_f1_mean'] = float(np.mean(opt_f1s))
                entry['per_genome_summary']['optimal_f1_range'] = [float(min(opt_f1s)), float(max(opt_f1s))]

        report[name] = entry

    # Save JSON
    json_path = output_dir / 'analysis_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print text summary
    print("\n" + "=" * 70)
    print("SEGMENT ACTIVATION ANALYSIS SUMMARY")
    print("=" * 70)

    for name, entry in report.items():
        print(f"\n{'─' * 60}")
        print(f"Dataset: {name}  ({entry['dataset_type']})")
        print(f"  Segments: {entry['n_segments']}")

        if 'label_distribution' in entry:
            ld = entry['label_distribution']
            print(f"  Labels: {ld['positive']} positive, {ld['negative']} negative "
                  f"({ld['pct_positive']:.1f}% positive)")

        print(f"\n  Activation Statistics:")
        for metric_label, stats in entry['statistics'].items():
            print(f"    {metric_label}:")
            print(f"      median={stats['median']:.4f}  mean={stats['mean']:.4f}  "
                  f"std={stats['std']:.4f}  max={stats['max']:.4f}")
            print(f"      P90={stats['p90']:.4f}  P95={stats['p95']:.4f}  "
                  f"P99={stats['p99']:.4f}")
            print(f"      %zero={stats['pct_zero']:.1f}%  %>0.5={stats['pct_above_0.5']:.1f}%  "
                  f"  %>1.0={stats['pct_above_1.0']:.1f}%")

        if 'current_predictions' in entry:
            cp = entry['current_predictions']
            print(f"\n  Current predictions (from sae_inference.py thresholds):")
            print(f"    P={cp['precision']:.3f}  R={cp['recall']:.3f}  "
                  f"F1={cp['f1']:.3f}  MCC={cp['mcc']:.3f}")
            print(f"    TP={cp['tp']}  FP={cp['fp']}  FN={cp['fn']}  TN={cp['tn']}")

        if 'optimal_single_thresholds' in entry:
            print(f"\n  Optimal single-metric thresholds:")
            for metric_key, info in entry['optimal_single_thresholds'].items():
                clean_name = metric_key.replace('_activations', '').replace('_', ' ')
                print(f"    {clean_name}: F1={info['best_f1']:.3f} @ t={info['best_f1_threshold']:.4f}  |  "
                      f"MCC={info['best_mcc']:.3f} @ t={info['best_mcc_threshold']:.4f}")

        if 'optimal_combined_score' in entry:
            bc = entry['optimal_combined_score']
            print(f"\n  Best combined score strategy:")
            print(f"    {bc['name']}: F1={bc['f1']:.3f} @ normalized_t={bc['threshold']:.3f}")

    print(f"\n{'=' * 70}")
    print(f"Full report saved to: {json_path}")
    print(f"Plots saved to: {output_dir}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE activation distributions across segment datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", nargs='+',
                             help="One or more result CSV files from sae_inference.py")
    input_group.add_argument("--input_dir",
                             help="Directory containing result CSV files")

    parser.add_argument("--output_dir", default="./segment_analysis",
                        help="Output directory for plots and reports")
    parser.add_argument("--labels", nargs='+',
                        help="Optional dataset labels (same order as --input files)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if args.input_dir:
        input_files = sorted(Path(args.input_dir).glob("*.csv"))
        if not input_files:
            print(f"ERROR: No CSV files found in {args.input_dir}")
            sys.exit(1)
    else:
        input_files = [Path(f) for f in args.input]

    # Load all datasets
    print("=" * 60)
    print("Segment Activation Analysis")
    print("=" * 60)
    print(f"Loading {len(input_files)} dataset(s)...\n")

    datasets = {}
    for i, csv_path in enumerate(input_files):
        if not csv_path.exists():
            print(f"  WARNING: File not found: {csv_path}")
            continue

        # Determine name
        if args.labels and i < len(args.labels):
            name = args.labels[i]
        else:
            name = csv_path.stem

        print(f"  Loading {name} ({csv_path})...")
        data = load_results(csv_path)
        if data is None:
            continue

        data['dataset_type'] = guess_dataset_type(str(csv_path))
        datasets[name] = data
        print(f"    {data['n_segments']} segments, type={data['dataset_type']}")

        if data['labels'] is not None:
            n_pos = np.sum(data['labels'] == 1)
            n_neg = np.sum(data['labels'] == 0)
            print(f"    Labels: {n_pos} positive, {n_neg} negative")

    if not datasets:
        print("ERROR: No datasets loaded")
        sys.exit(1)

    # Run analyses
    print(f"\nGenerating plots in {output_dir}/...")

    print("  Distribution plots...")
    plot_distributions(datasets, output_dir)

    if len(datasets) > 1:
        print("  Cross-dataset comparison...")
        plot_cross_dataset_comparison(datasets, output_dir)

    print("  Feature space scatter plots...")
    plot_labeled_separation(datasets, output_dir)

    print("  Threshold sweep analysis...")
    plot_threshold_sweep(datasets, output_dir)

    print("  ROC and PR curves...")
    plot_roc_pr(datasets, output_dir)

    print("  Combined score analysis...")
    plot_combined_score_sweep(datasets, output_dir)

    # Per-genome analysis (if seq_id available)
    has_seq_ids = any(d.get('seq_ids') is not None for d in datasets.values())
    if has_seq_ids:
        print("  Per-genome analysis...")
        plot_per_genome_analysis(datasets, output_dir)

    # Generate summary
    report = generate_summary(datasets, output_dir)

    # Per-genome summary (printed after main report)
    if has_seq_ids:
        print_per_genome_summary(datasets)

    print("\nDone!")


if __name__ == "__main__":
    main()
