#!/usr/bin/env python3
"""
Calculate nucleotide-level metrics for prophage prediction evaluation.

Compares predicted prophage locations against ground truth and calculates:
TP, TN, FP, FN, Accuracy, Precision, Recall, F1, MCC

Metrics are calculated per genome and then averaged across the dataset.
"""

import argparse
import csv
import os
import math
import re
from collections import defaultdict
from pathlib import Path


def normalize_genome_id(genome_id):
    """
    Normalize genome ID by removing version number suffix (.1, .2, etc.)
    Example: NC_023149.1 -> NC_023149
    """
    return re.sub(r'\.\d+$', '', genome_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate nucleotide-level prophage prediction metrics"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        help="Path to ground truth CSV file with true prophage locations"
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--fasta-dir", "-f",
        required=True,
        help="Path to directory containing FASTA files for genome lengths"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output CSV file (optional, prints to stdout if not specified)"
    )
    return parser.parse_args()


def get_genome_length_from_fasta(fasta_path):
    """Read a FASTA file and return total sequence length."""
    length = 0
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                length += len(line.strip())
    return length


def get_genome_id_from_fasta(fasta_path):
    """Extract genome ID from FASTA header."""
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract ID (first word after >)
                header = line[1:].strip()
                genome_id = header.split()[0]
                return genome_id
    return None


def load_genome_lengths(fasta_dir):
    """Load genome lengths from all FASTA files in directory."""
    genome_lengths = {}
    fasta_extensions = ['.fasta', '.fa', '.fna', '.fsa']

    fasta_path = Path(fasta_dir)
    for ext in fasta_extensions:
        for fasta_file in fasta_path.glob(f'*{ext}'):
            genome_id = get_genome_id_from_fasta(fasta_file)
            if genome_id:
                normalized_id = normalize_genome_id(genome_id)
                length = get_genome_length_from_fasta(fasta_file)
                genome_lengths[normalized_id] = length
                print(f"Loaded genome {genome_id} -> {normalized_id}: {length:,} bp")

    return genome_lengths


def load_ground_truth(csv_path):
    """Load ground truth prophage regions from CSV."""
    regions = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            genome_id = normalize_genome_id(row['NCBI Id'].strip())
            start = int(row['start'])
            end = int(row['end'])
            regions[genome_id].append((start, end))

    return regions


def load_predictions(csv_path):
    """Load predicted prophage regions from CSV."""
    regions = defaultdict(list)
    errors = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        line_num = 1
        for row in reader:
            line_num += 1
            if len(row) < 3:
                error_msg = f"Line {line_num}: Fewer than 3 columns"
                errors.append((line_num, row, error_msg))
                continue
            # Read by position: Contig=0, Start=1, End=2
            genome_id = normalize_genome_id(row[0].strip())
            try:
                start = int(row[1])
                end = int(row[2])
                regions[genome_id].append((start, end))
            except ValueError as e:
                error_msg = f"Line {line_num}: {e}"
                errors.append((line_num, row, error_msg))

    return regions, errors


def regions_to_positions(regions):
    """Convert list of (start, end) regions to set of positions."""
    positions = set()
    for start, end in regions:
        # Include both start and end positions (1-based coordinates)
        positions.update(range(start, end + 1))
    return positions


def calculate_metrics(tp, tn, fp, fn):
    """Calculate all metrics from confusion matrix values."""
    total = tp + tn + fp + fn

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Matthews Correlation Coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    }


def calculate_genome_metrics(true_regions, pred_regions, genome_length):
    """Calculate nucleotide-level metrics for a single genome."""
    true_positions = regions_to_positions(true_regions)
    pred_positions = regions_to_positions(pred_regions)

    all_positions = set(range(1, genome_length + 1))

    # Calculate confusion matrix at nucleotide level
    tp = len(true_positions & pred_positions)  # In both
    fp = len(pred_positions - true_positions)  # Predicted but not true
    fn = len(true_positions - pred_positions)  # True but not predicted
    tn = len(all_positions - true_positions - pred_positions)  # Neither

    return calculate_metrics(tp, tn, fp, fn)


def main():
    args = parse_args()

    print("Loading genome lengths from FASTA files...")
    genome_lengths = load_genome_lengths(args.fasta_dir)
    print(f"Loaded {len(genome_lengths)} genomes\n")

    print("Loading ground truth...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(ground_truth)} genomes\n")

    print("Loading predictions...")
    predictions, pred_errors = load_predictions(args.predictions)
    print(f"Loaded predictions for {len(predictions)} genomes")
    if pred_errors:
        print(f"Warning: {len(pred_errors)} rows had parsing errors (details at end)\n")
    else:
        print()

    # Get all unique genome IDs from both datasets
    all_genomes = set(ground_truth.keys()) | set(predictions.keys())

    # Filter to genomes we have lengths for
    valid_genomes = all_genomes & set(genome_lengths.keys())

    if len(valid_genomes) < len(all_genomes):
        missing = all_genomes - valid_genomes
        print(f"Warning: {len(missing)} genomes missing from FASTA directory:")
        for g in sorted(missing):
            print(f"  - {g}")
        print()

    # Calculate metrics for each genome
    results = []
    print("Calculating metrics per genome...")
    print("-" * 80)

    for genome_id in sorted(valid_genomes):
        true_regions = ground_truth.get(genome_id, [])
        pred_regions = predictions.get(genome_id, [])
        genome_length = genome_lengths[genome_id]

        metrics = calculate_genome_metrics(true_regions, pred_regions, genome_length)
        metrics['Genome'] = genome_id
        metrics['Genome_Length'] = genome_length
        metrics['True_Regions'] = len(true_regions)
        metrics['Pred_Regions'] = len(pred_regions)

        results.append(metrics)

        print(f"{genome_id}:")
        print(f"  Genome length: {genome_length:,} bp")
        print(f"  True regions: {len(true_regions)}, Predicted regions: {len(pred_regions)}")
        print(f"  TP: {metrics['TP']:,}, TN: {metrics['TN']:,}, FP: {metrics['FP']:,}, FN: {metrics['FN']:,}")
        print(f"  Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}, MCC: {metrics['MCC']:.4f}")
        print()

    # Calculate average metrics across all genomes
    if results:
        print("-" * 80)
        print("AVERAGE METRICS ACROSS ALL GENOMES:")
        print("-" * 80)

        avg_metrics = {}
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']

        for metric in metric_names:
            values = [r[metric] for r in results]
            avg_metrics[metric] = sum(values) / len(values)
            print(f"  {metric}: {avg_metrics[metric]:.4f}")

        # Also calculate total TP, TN, FP, FN
        total_tp = sum(r['TP'] for r in results)
        total_tn = sum(r['TN'] for r in results)
        total_fp = sum(r['FP'] for r in results)
        total_fn = sum(r['FN'] for r in results)

        print()
        print("AGGREGATE METRICS (summed across all genomes):")
        print("-" * 80)
        print(f"  Total TP: {total_tp:,}")
        print(f"  Total TN: {total_tn:,}")
        print(f"  Total FP: {total_fp:,}")
        print(f"  Total FN: {total_fn:,}")

        # Calculate metrics from aggregated confusion matrix
        agg_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
        print(f"  Aggregate Precision: {agg_metrics['Precision']:.4f}")
        print(f"  Aggregate Recall: {agg_metrics['Recall']:.4f}")
        print(f"  Aggregate F1: {agg_metrics['F1']:.4f}")
        print(f"  Aggregate Accuracy: {agg_metrics['Accuracy']:.4f}")
        print(f"  Aggregate MCC: {agg_metrics['MCC']:.4f}")

    # Write results to CSV if output specified
    if args.output:
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['Genome', 'Genome_Length', 'True_Regions', 'Pred_Regions',
                          'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

            # Write average row
            avg_row = {
                'Genome': 'AVERAGE',
                'Genome_Length': '',
                'True_Regions': '',
                'Pred_Regions': '',
                'TP': '',
                'TN': '',
                'FP': '',
                'FN': '',
                'Accuracy': avg_metrics['Accuracy'],
                'Precision': avg_metrics['Precision'],
                'Recall': avg_metrics['Recall'],
                'F1': avg_metrics['F1'],
                'MCC': avg_metrics['MCC']
            }
            writer.writerow(avg_row)

            # Write aggregate row
            agg_row = {
                'Genome': 'AGGREGATE',
                'Genome_Length': sum(genome_lengths.get(g, 0) for g in valid_genomes),
                'True_Regions': sum(r['True_Regions'] for r in results),
                'Pred_Regions': sum(r['Pred_Regions'] for r in results),
                'TP': total_tp,
                'TN': total_tn,
                'FP': total_fp,
                'FN': total_fn,
                'Accuracy': agg_metrics['Accuracy'],
                'Precision': agg_metrics['Precision'],
                'Recall': agg_metrics['Recall'],
                'F1': agg_metrics['F1'],
                'MCC': agg_metrics['MCC']
            }
            writer.writerow(agg_row)

        print(f"\nResults written to: {args.output}")

    # Print and log any parsing errors
    if pred_errors:
        print("\n" + "=" * 80)
        print(f"PARSING ERRORS ({len(pred_errors)} rows)")
        print("=" * 80)
        for line_num, row, error_msg in pred_errors:
            print(f"  {error_msg}")
            print(f"    Data: {row[:5]}...")
            print()

        # Write errors to log file
        error_log_path = args.output.replace('.csv', '_errors.log') if args.output else 'prediction_errors.log'
        with open(error_log_path, 'w') as f:
            f.write(f"Parsing Errors from: {args.predictions}\n")
            f.write(f"Total errors: {len(pred_errors)}\n")
            f.write("=" * 80 + "\n\n")
            for line_num, row, error_msg in pred_errors:
                f.write(f"{error_msg}\n")
                f.write(f"Full row: {row}\n")
                f.write("-" * 40 + "\n")
        print(f"Error log written to: {error_log_path}")


if __name__ == "__main__":
    main()
