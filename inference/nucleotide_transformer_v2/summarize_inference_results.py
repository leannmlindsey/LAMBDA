#!/usr/bin/env python3
"""
Summarize Inference Results from NT-v2 Directory Inference

This script extracts metrics from inference output and creates summary CSVs
in the same format as DNABERT2's analyze_genome_wide_results.py for
downstream analysis compatibility.

Can read from:
1. summary.json (produced by inference_nt_dir.py)
2. Directory of individual *_metrics.json files

Outputs:
1. {model_name}_individual.csv - One row per genome with metrics
2. {model_name}_summary.csv - Aggregate and averaged metrics

Usage:
    python summarize_inference_results.py -d /path/to/output_dir -m NTv2
    python summarize_inference_results.py -d /path/to/output_dir -m NTv2 --filter
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


def collapse_overlapping_windows(predictions_df, window_size=2000, step_size=1000):
    """
    Collapse overlapping window predictions using majority voting

    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction, score (optional), label
    window_size : int
        Size of each window (default: 2000)
    step_size : int
        Step size between windows (default: 1000)

    Returns:
    --------
    collapsed_df : DataFrame
        Non-overlapping segments with majority vote predictions
    """

    # Determine the genome range
    min_pos = predictions_df['start'].min()
    max_pos = predictions_df['end'].max()

    # Create non-overlapping segments of step_size
    segments = []

    for pos in range(min_pos, max_pos, step_size):
        seg_start = pos
        seg_end = min(pos + step_size, max_pos)

        # Find all windows that cover this segment
        overlapping = predictions_df[
            (predictions_df['start'] <= seg_start) &
            (predictions_df['end'] >= seg_end)
        ]

        if len(overlapping) == 0:
            continue

        # Majority vote for prediction
        votes = overlapping['prediction'].sum()
        total_votes = len(overlapping)
        majority_pred = 1 if votes > total_votes / 2 else 0

        # Average the scores if available
        avg_score = overlapping['score'].mean() if 'score' in overlapping.columns else 0.5

        # Get the true label (should be same for all overlapping windows)
        true_label = overlapping['label'].iloc[0] if 'label' in overlapping.columns else 0

        segments.append({
            'start': seg_start,
            'end': seg_end,
            'prediction': majority_pred,
            'score': avg_score,
            'label': true_label,
            'num_votes': total_votes,
            'votes_for_phage': votes
        })

    return pd.DataFrame(segments)


def apply_phage_clustering_filter(predictions_df, merge_gap=3000, min_cluster_size=1000, window_size=5):
    """
    Apply clustering filter to reduce false positives and merge nearby phage predictions

    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction (0 or 1), score (optional)
    merge_gap : int
        Maximum gap (nt) between segments to merge into same cluster (default: 3000)
    min_cluster_size : int
        Minimum total size (nt) for a phage cluster to be kept (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)

    Returns:
    --------
    filtered_df : DataFrame
        DataFrame with filtered predictions
    """

    # Sort by start position
    df = predictions_df.sort_values('start').copy()

    # Apply bidirectional smoothing to predictions if we have scores
    if 'score' in df.columns:
        # Forward pass (left to right)
        forward_smooth = df['score'].ewm(span=window_size, adjust=False).mean()

        # Backward pass (right to left) - reverse, smooth, reverse back
        backward_smooth = df['score'][::-1].ewm(span=window_size, adjust=False).mean()[::-1]

        # Average both directions for bidirectional smoothing
        df['smoothed_score'] = (forward_smooth + backward_smooth) / 2

        # Update predictions based on smoothed scores (threshold 0.5)
        df['prediction'] = (df['smoothed_score'] >= 0.5).astype(int)

    # Filter to only phage predictions
    phage_df = df[df['prediction'] == 1].copy()

    if len(phage_df) == 0:
        return df  # No phage predicted, return original

    # Cluster nearby phage segments
    clusters = []
    current_cluster = [phage_df.iloc[0]]

    for idx in range(1, len(phage_df)):
        prev_segment = current_cluster[-1]
        curr_segment = phage_df.iloc[idx]

        # Check if gap between segments is less than merge_gap
        gap = curr_segment['start'] - prev_segment['end']

        if gap <= merge_gap:
            current_cluster.append(curr_segment)
        else:
            # Save current cluster and start new one
            clusters.append(current_cluster)
            current_cluster = [curr_segment]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    # Filter clusters by minimum size and mark segments
    valid_indices = set()

    for cluster in clusters:
        cluster_start = cluster[0]['start']
        cluster_end = cluster[-1]['end']
        cluster_size = cluster_end - cluster_start

        if cluster_size >= min_cluster_size:
            # Keep all segments in this cluster
            for segment in cluster:
                valid_indices.add(segment.name)

    # Update predictions: set to 0 if not in valid cluster
    df['filtered_prediction'] = df['prediction'].copy()
    df.loc[~df.index.isin(valid_indices), 'filtered_prediction'] = 0

    return df


def calculate_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient"""
    numerator = (tp * tn) - (fp * fn)

    # Use float to avoid overflow
    denominator_val = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)

    if denominator_val <= 0:
        return 0.0

    denominator = math.sqrt(denominator_val)

    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(tp, tn, fp, fn):
    """Calculate all metrics from confusion matrix"""
    total = tp + tn + fp + fn

    metrics = {}
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['mcc'] = calculate_mcc(tp, tn, fp, fn)

    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0

    return metrics


def load_from_summary_json(json_path):
    """Load results from summary.json file"""
    with open(json_path, 'r') as f:
        summary = json.load(f)

    results = []
    for result in summary.get('results', []):
        if 'metrics' not in result:
            continue

        m = result['metrics']
        results.append({
            'filename': result['file'] + '_metrics.json',  # Match DNABERT2 format
            'basename': result['file'],
            'samples': result['samples'],
            'true_positives': m.get('true_positives', 0),
            'true_negatives': m.get('true_negatives', 0),
            'false_positives': m.get('false_positives', 0),
            'false_negatives': m.get('false_negatives', 0),
            'accuracy': m.get('accuracy', 0),
            'precision': m.get('precision', 0),
            'recall': m.get('recall', 0),
            'specificity': m.get('specificity', 0),
            'mcc': m.get('mcc', 0),
            'f1': m.get('f1', 0),
        })

    return results, summary.get('output_dir', os.path.dirname(json_path))


def load_from_json_files(directory_path):
    """Load results from individual *_metrics.json files"""
    json_files = glob.glob(os.path.join(directory_path, '*_metrics.json'))

    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            filename = os.path.basename(json_file)
            basename = filename.replace('_metrics.json', '')

            tp = data.get('true_positives', 0)
            tn = data.get('true_negatives', 0)
            fp = data.get('false_positives', 0)
            fn = data.get('false_negatives', 0)

            results.append({
                'filename': filename,
                'basename': basename,
                'samples': data.get('num_samples', data.get('samples', tp + tn + fp + fn)),
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'accuracy': data.get('accuracy', 0),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'specificity': data.get('specificity', 0),
                'mcc': data.get('mcc', calculate_mcc(tp, tn, fp, fn)),
                'f1': data.get('f1', 0),
            })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return results, directory_path


def summarize_genome_predictions(directory_path, model_name, output_dir='.',
                                output_individual=None,
                                output_summary=None,
                                apply_filter=False,
                                merge_gap=3000,
                                min_cluster_size=1000,
                                window_size=5,
                                verbose=False):
    """
    Read prediction results and summarize metrics

    Parameters:
    -----------
    directory_path : str
        Path to directory containing results or summary.json
    model_name : str
        Name/identifier for this model
    output_dir : str
        Directory to save output CSV files
    apply_filter : bool
        Apply phage clustering filter (default: False)
    merge_gap : int
        Maximum gap (nt) between segments to merge (default: 3000)
    min_cluster_size : int
        Minimum cluster size (nt) to keep (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)
    verbose : bool
        Print detailed information (default: False)
    """

    # Set default output filenames
    if output_individual is None:
        suffix = "_filtered_individual.csv" if apply_filter else "_individual.csv"
        output_individual = f"{model_name}{suffix}"
    if output_summary is None:
        suffix = "_filtered_summary.csv" if apply_filter else "_summary.csv"
        output_summary = f"{model_name}{suffix}"

    # Add output directory to paths
    output_individual = os.path.join(output_dir, output_individual)
    output_summary = os.path.join(output_dir, output_summary)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load results - try summary.json first, then individual JSON files
    summary_json_path = os.path.join(directory_path, 'summary.json')
    if os.path.isfile(directory_path) and directory_path.endswith('.json'):
        summary_json_path = directory_path
        directory_path = os.path.dirname(directory_path)

    if os.path.exists(summary_json_path):
        print(f"Loading from summary.json: {summary_json_path}")
        results, data_dir = load_from_summary_json(summary_json_path)
    else:
        print(f"Loading from individual JSON files in: {directory_path}")
        results, data_dir = load_from_json_files(directory_path)

    if not results:
        print(f"No results found in {directory_path}")
        return None, None

    print(f"Found {len(results)} files with metrics")

    # Process each result (optionally with filtering)
    processed_results = []

    # Accumulators for aggregate calculation
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_samples = 0

    for result in results:
        tp = result['true_positives']
        tn = result['true_negatives']
        fp = result['false_positives']
        fn = result['false_negatives']

        # If filtering is requested, look for corresponding CSV file
        if apply_filter:
            csv_file = os.path.join(data_dir, f"{result['basename']}_predictions.csv")

            if os.path.exists(csv_file):
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Filename: {result['filename']}")

                pred_df = pd.read_csv(csv_file)

                if verbose:
                    print(f"Total predictions: {len(pred_df)}")

                if 'pred_label' in pred_df.columns and 'start' in pred_df.columns:
                    # Calculate metrics before filtering
                    original_mcc = calculate_mcc(tp, tn, fp, fn)
                    original_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    phage_before = pred_df['pred_label'].sum()
                    if verbose:
                        print(f"Phage predictions before filtering: {phage_before}")

                    # Rename columns to match expected format
                    pred_df = pred_df.rename(columns={'pred_label': 'prediction'})
                    if 'prob_1' in pred_df.columns:
                        pred_df = pred_df.rename(columns={'prob_1': 'score'})

                    # Collapse overlapping windows
                    collapsed_df = collapse_overlapping_windows(pred_df, window_size=2000, step_size=1000)

                    # Apply clustering filter
                    filtered_df = apply_phage_clustering_filter(
                        collapsed_df,
                        merge_gap=merge_gap,
                        min_cluster_size=min_cluster_size,
                        window_size=window_size
                    )

                    phage_after = filtered_df['filtered_prediction'].sum() if 'filtered_prediction' in filtered_df.columns else 0
                    if verbose:
                        print(f"Phage predictions after filtering: {phage_after}")

                    # Recalculate confusion matrix
                    if 'label' in filtered_df.columns and 'filtered_prediction' in filtered_df.columns:
                        tp = int(((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 1)).sum())
                        tn = int(((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 0)).sum())
                        fp = int(((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 1)).sum())
                        fn = int(((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 0)).sum())

                        new_mcc = calculate_mcc(tp, tn, fp, fn)
                        new_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                        if verbose:
                            print(f"MCC before: {original_mcc:.4f}")
                            print(f"MCC after: {new_mcc:.4f}")
                            print(f"Recall before: {original_recall:.4f}")
                            print(f"Recall after: {new_recall:.4f}")
                elif verbose:
                    print(f"WARNING: CSV missing required columns (pred_label, start)")
            elif verbose:
                print(f"WARNING: CSV file not found: {csv_file}")

        # Calculate metrics for this file
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        mcc = calculate_mcc(tp, tn, fp, fn)
        metrics = calculate_metrics(tp, tn, fp, fn)

        processed_results.append({
            'filename': result['filename'],
            'samples': result['samples'],
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'fnr': fnr,
            'fpr': fpr,
            'mcc': mcc,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'specificity': metrics['specificity'],
            'f1': metrics['f1'],
        })

        # Accumulate totals
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_samples += result['samples']

    # Create individual results DataFrame
    df_individual = pd.DataFrame(processed_results)

    # Calculate aggregate metrics (from summed confusion matrix)
    aggregate_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)

    # Create summary DataFrame (matching DNABERT2 format)
    summary_data = [
        {
            'model': model_name,
            'method': 'aggregate',
            'num_files': len(processed_results),
            'total_samples': total_samples,
            'total_tp': total_tp,
            'total_tn': total_tn,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'fnr': aggregate_metrics['fnr'],
            'fpr': aggregate_metrics['fpr'],
            'mcc': aggregate_metrics['mcc'],
            'accuracy': aggregate_metrics['accuracy'],
            'precision': aggregate_metrics['precision'],
            'recall': aggregate_metrics['recall'],
            'specificity': aggregate_metrics['specificity'],
            'f1': aggregate_metrics['f1']
        },
        {
            'model': model_name,
            'method': 'average',
            'num_files': len(processed_results),
            'total_samples': total_samples,
            'total_tp': '',
            'total_tn': '',
            'total_fp': '',
            'total_fn': '',
            'fnr': df_individual['fnr'].mean(),
            'fpr': df_individual['fpr'].mean(),
            'mcc': df_individual['mcc'].mean(),
            'accuracy': df_individual['accuracy'].mean(),
            'precision': df_individual['precision'].mean(),
            'recall': df_individual['recall'].mean(),
            'specificity': df_individual['specificity'].mean(),
            'f1': df_individual['f1'].mean()
        }
    ]

    df_summary = pd.DataFrame(summary_data)

    # Save CSVs
    df_individual.to_csv(output_individual, index=False)
    df_summary.to_csv(output_summary, index=False)

    print(f"\nIndividual results saved to {output_individual}")
    print(f"Summary metrics saved to {output_summary}")
    print(f"\nProcessed {len(processed_results)} files")
    print(f"Total samples: {total_samples}")
    print(f"\nAggregate Metrics (from summed confusion matrix):")
    print(f"  MCC: {aggregate_metrics['mcc']:.4f}")
    print(f"  FNR: {aggregate_metrics['fnr']:.4f} ({aggregate_metrics['fnr']*100:.2f}%)")
    print(f"  FPR: {aggregate_metrics['fpr']:.4f} ({aggregate_metrics['fpr']*100:.2f}%)")
    print(f"  Recall: {aggregate_metrics['recall']:.4f} ({aggregate_metrics['recall']*100:.2f}%)")
    print(f"  F1: {aggregate_metrics['f1']:.4f}")
    print(f"\nAveraged Metrics (mean across files):")
    print(f"  MCC: {df_individual['mcc'].mean():.4f}")
    print(f"  FNR: {df_individual['fnr'].mean():.4f} ({df_individual['fnr'].mean()*100:.2f}%)")
    print(f"  FPR: {df_individual['fpr'].mean():.4f} ({df_individual['fpr'].mean()*100:.2f}%)")
    print(f"  Recall: {df_individual['recall'].mean():.4f} ({df_individual['recall'].mean()*100:.2f}%)")
    print(f"  F1: {df_individual['f1'].mean():.4f}")

    return df_individual, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Summarize genome-wide prediction metrics (DNABERT2-compatible format)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from summary.json
  python summarize_inference_results.py -d /path/to/output_dir -m NTv2 -r results

  # With clustering filter
  python summarize_inference_results.py -d /path/to/output_dir -m NTv2 --filter

  # Compare multiple models
  python summarize_inference_results.py -d /path/to/ntv2_results -m NTv2 -r comparison
  python summarize_inference_results.py -d /path/to/dnabert2_results -m DNABERT2 -r comparison
        """
    )

    parser.add_argument('-d', '--directory', required=True,
                        help='Directory containing summary.json or *_metrics.json files')
    parser.add_argument('-m', '--model-name', required=True,
                        help='Model identifier/name for this run')
    parser.add_argument('-r', '--output-dir', default='.',
                        help='Directory to save output CSV files (default: current directory)')
    parser.add_argument('-i', '--output-individual', default=None,
                        help='Output CSV filename for individual genome results')
    parser.add_argument('-s', '--output-summary', default=None,
                        help='Output CSV filename for summary metrics')
    parser.add_argument('--filter', action='store_true',
                        help='Apply phage clustering filter to reduce false positives')
    parser.add_argument('--merge-gap', type=int, default=3000,
                        help='Maximum gap (nt) between segments to merge (default: 3000)')
    parser.add_argument('--min-cluster-size', type=int, default=1000,
                        help='Minimum cluster size (nt) to keep (default: 1000)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for bidirectional smoothing (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed processing information')

    args = parser.parse_args()

    df_individual, df_summary = summarize_genome_predictions(
        directory_path=args.directory,
        model_name=args.model_name,
        output_dir=args.output_dir,
        output_individual=args.output_individual,
        output_summary=args.output_summary,
        apply_filter=args.filter,
        merge_gap=args.merge_gap,
        min_cluster_size=args.min_cluster_size,
        window_size=args.window_size,
        verbose=args.verbose
    )

    if df_individual is not None and df_summary is not None:
        print("\nFirst 5 individual files:")
        print(df_individual.head())
        print("\nSummary metrics:")
        print(df_summary)
