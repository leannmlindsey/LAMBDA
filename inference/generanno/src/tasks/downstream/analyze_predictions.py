"""
Analyze prediction probabilities from inference output.

This script analyzes the distribution of predicted probabilities to understand
model confidence and calibration, especially useful for control experiments.

Usage:
    python -m src.tasks.downstream.analyze_predictions \
        --input_csv /path/to/predictions.csv \
        --output_dir ./analysis_results \
        --title "Shuffled Control"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze prediction probability distributions"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to predictions CSV (from inference.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: same as input)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Predictions",
        help="Title for plots (e.g., 'Shuffled Phage Control')",
    )
    parser.add_argument(
        "--prob_column",
        type=str,
        default="prob_1",
        help="Column name for probability to analyze (default: prob_1)",
    )
    return parser.parse_args()


def analyze_probabilities(probs, title="Predictions"):
    """Compute statistics about probability distribution."""
    stats = {
        "count": len(probs),
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "median": float(np.median(probs)),
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "q25": float(np.percentile(probs, 25)),
        "q75": float(np.percentile(probs, 75)),
    }

    # Count samples in different confidence ranges
    stats["prob_0_to_0.1"] = int(np.sum(probs < 0.1))
    stats["prob_0.1_to_0.3"] = int(np.sum((probs >= 0.1) & (probs < 0.3)))
    stats["prob_0.3_to_0.5"] = int(np.sum((probs >= 0.3) & (probs < 0.5)))
    stats["prob_0.5_to_0.7"] = int(np.sum((probs >= 0.5) & (probs < 0.7)))
    stats["prob_0.7_to_0.9"] = int(np.sum((probs >= 0.7) & (probs < 0.9)))
    stats["prob_0.9_to_1.0"] = int(np.sum(probs >= 0.9))

    # Percentages
    stats["pct_confident_class0"] = float(np.mean(probs < 0.3) * 100)
    stats["pct_uncertain"] = float(np.mean((probs >= 0.3) & (probs < 0.7)) * 100)
    stats["pct_confident_class1"] = float(np.mean(probs >= 0.7) * 100)

    return stats


def create_histogram(probs, output_path, title="Predictions"):
    """Create histogram of probability distribution."""
    plt.figure(figsize=(10, 6))

    # Histogram
    n, bins, patches = plt.hist(probs, bins=50, edgecolor='black', alpha=0.7)

    # Color bars by region
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.3:
            patch.set_facecolor('#2ecc71')  # Green - confident class 0
        elif bin_center > 0.7:
            patch.set_facecolor('#e74c3c')  # Red - confident class 1
        else:
            patch.set_facecolor('#f39c12')  # Orange - uncertain

    # Reference lines
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    plt.axvline(x=0.3, color='gray', linestyle=':', linewidth=1, label='Uncertainty region')
    plt.axvline(x=0.7, color='gray', linestyle=':', linewidth=1)

    # Labels
    plt.xlabel('Predicted Probability (Class 1)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Probability Distribution - {title}', fontsize=14)
    plt.legend(loc='upper right')

    # Add statistics text box
    mean_prob = np.mean(probs)
    std_prob = np.std(probs)
    textstr = f'Mean: {mean_prob:.3f}\nStd: {std_prob:.3f}\nn={len(probs)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved histogram to: {output_path}")


def create_box_violin_plot(probs, output_path, title="Predictions", label_column=None, labels=None):
    """Create box and violin plot of probabilities, optionally by label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if labels is not None and len(np.unique(labels)) > 1:
        # Separate by label
        unique_labels = sorted(np.unique(labels))
        data_by_label = [probs[labels == lbl] for lbl in unique_labels]
        label_names = [f'Class {lbl}' for lbl in unique_labels]

        # Violin plot
        vp = axes[0].violinplot(data_by_label, positions=range(len(unique_labels)), showmeans=True)
        axes[0].set_xticks(range(len(unique_labels)))
        axes[0].set_xticklabels(label_names)
        axes[0].set_ylabel('Predicted Probability (Class 1)')
        axes[0].set_title(f'Violin Plot by True Label - {title}')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        # Box plot
        axes[1].boxplot(data_by_label, labels=label_names)
        axes[1].set_ylabel('Predicted Probability (Class 1)')
        axes[1].set_title(f'Box Plot by True Label - {title}')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    else:
        # Single distribution
        axes[0].violinplot([probs], positions=[0], showmeans=True)
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(['All'])
        axes[0].set_ylabel('Predicted Probability (Class 1)')
        axes[0].set_title(f'Violin Plot - {title}')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        axes[1].boxplot([probs])
        axes[1].set_ylabel('Predicted Probability (Class 1)')
        axes[1].set_title(f'Box Plot - {title}')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved box/violin plot to: {output_path}")


def print_interpretation(stats):
    """Print interpretation of the results."""
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    mean = stats["mean"]
    std = stats["std"]
    pct_uncertain = stats["pct_uncertain"]

    if std < 0.15 and 0.4 < mean < 0.6:
        print("Model is UNCERTAIN about these samples.")
        print("  - Probabilities clustered around 0.5")
        print("  - This is EXPECTED for out-of-distribution data (e.g., shuffled sequences)")
        print("  - Shows model learned real patterns, not just composition")
    elif std > 0.3:
        print("Model shows HIGH VARIANCE in predictions.")
        print("  - Bimodal distribution (some confident class 0, some confident class 1)")
        print("  - May indicate model is guessing randomly but confidently")
    elif mean < 0.3:
        print("Model is CONFIDENT these are Class 0.")
        print(f"  - Mean probability: {mean:.3f}")
    elif mean > 0.7:
        print("Model is CONFIDENT these are Class 1.")
        print(f"  - Mean probability: {mean:.3f}")
    else:
        print("Model shows MODERATE confidence.")
        print(f"  - Mean: {mean:.3f}, Std: {std:.3f}")

    print(f"\nConfidence breakdown:")
    print(f"  - Confident Class 0 (prob < 0.3): {stats['pct_confident_class0']:.1f}%")
    print(f"  - Uncertain (0.3 - 0.7): {stats['pct_uncertain']:.1f}%")
    print(f"  - Confident Class 1 (prob > 0.7): {stats['pct_confident_class1']:.1f}%")
    print("=" * 60)


def main():
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Prediction Probability Analysis")
    print("=" * 60)

    # Load predictions
    print(f"\nLoading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.prob_column not in df.columns:
        raise ValueError(f"Column '{args.prob_column}' not found. Available: {list(df.columns)}")

    probs = df[args.prob_column].values
    labels = df['label'].values if 'label' in df.columns else None

    print(f"  Samples: {len(probs)}")
    print(f"  Analyzing column: {args.prob_column}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute statistics
    stats = analyze_probabilities(probs, args.title)

    print("\n" + "=" * 60)
    print("PROBABILITY STATISTICS")
    print("=" * 60)
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Std:    {stats['std']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Min:    {stats['min']:.4f}")
    print(f"  Max:    {stats['max']:.4f}")
    print(f"  Q25:    {stats['q25']:.4f}")
    print(f"  Q75:    {stats['q75']:.4f}")

    print("\n  Probability ranges:")
    print(f"    [0.0 - 0.1): {stats['prob_0_to_0.1']:5d} ({stats['prob_0_to_0.1']/len(probs)*100:.1f}%)")
    print(f"    [0.1 - 0.3): {stats['prob_0.1_to_0.3']:5d} ({stats['prob_0.1_to_0.3']/len(probs)*100:.1f}%)")
    print(f"    [0.3 - 0.5): {stats['prob_0.3_to_0.5']:5d} ({stats['prob_0.3_to_0.5']/len(probs)*100:.1f}%)")
    print(f"    [0.5 - 0.7): {stats['prob_0.5_to_0.7']:5d} ({stats['prob_0.5_to_0.7']/len(probs)*100:.1f}%)")
    print(f"    [0.7 - 0.9): {stats['prob_0.7_to_0.9']:5d} ({stats['prob_0.7_to_0.9']/len(probs)*100:.1f}%)")
    print(f"    [0.9 - 1.0]: {stats['prob_0.9_to_1.0']:5d} ({stats['prob_0.9_to_1.0']/len(probs)*100:.1f}%)")

    # Print interpretation
    print_interpretation(stats)

    # Create plots
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]

    hist_path = os.path.join(args.output_dir, f"{base_name}_histogram.png")
    create_histogram(probs, hist_path, args.title)

    box_path = os.path.join(args.output_dir, f"{base_name}_boxplot.png")
    create_box_violin_plot(probs, box_path, args.title, labels=labels)

    # Save statistics to JSON
    stats["title"] = args.title
    stats["input_csv"] = args.input_csv
    stats_path = os.path.join(args.output_dir, f"{base_name}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
