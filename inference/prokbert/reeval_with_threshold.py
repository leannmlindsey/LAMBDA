#!/usr/bin/env python3
"""
Re-evaluate predictions using optimal threshold.
Applies the optimal threshold found from threshold analysis to generate new predictions and metrics.

Usage:
    python reeval_with_threshold.py --predictions_file <path> --threshold <value> [options]
    
Examples:
    # Using optimal threshold from analysis
    python reeval_with_threshold.py --predictions_file inference_results_epoch3/predictions_dataset_1024_512_prokbert-mini.csv --threshold 0.99
    
    # Compare multiple thresholds
    python reeval_with_threshold.py --predictions_file predictions.csv --threshold 0.5,0.7,0.9,0.99
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
from pathlib import Path


def evaluate_at_threshold(df, threshold, verbose=True):
    """
    Evaluate predictions at a specific threshold.
    
    Args:
        df: DataFrame with label and prob_1 columns
        threshold: Classification threshold
        verbose: Print detailed results
    
    Returns:
        dict: Dictionary of metrics and predictions
    """
    y_true = df['label'].values
    y_prob = df['prob_1'].values
    
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_positive_predictions': np.sum(y_pred),
        'total_actual_positives': np.sum(y_true),
        'total_samples': len(y_true)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*60}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                 Neg      Pos")
        print(f"Actual  Neg  {tn:8,} {fp:8,}")
        print(f"        Pos  {fn:8,} {tp:8,}")
        
        print(f"\nMETRICS:")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"Precision:          {metrics['precision']:.4f}")
        print(f"Recall (TPR):       {metrics['recall']:.4f}")
        print(f"Specificity (TNR):  {metrics['specificity']:.4f}")
        print(f"F1-Score:           {metrics['f1']:.4f}")
        print(f"MCC:                {metrics['mcc']:.4f}")
        print(f"AUC-ROC:            {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:             {metrics['auc_pr']:.4f}")
        
        print(f"\nPREDICTION SUMMARY:")
        print(f"Total samples:         {metrics['total_samples']:,}")
        print(f"Actual positives:      {metrics['total_actual_positives']:,} ({metrics['total_actual_positives']/metrics['total_samples']*100:.2f}%)")
        print(f"Predicted positives:   {metrics['total_positive_predictions']:,} ({metrics['total_positive_predictions']/metrics['total_samples']*100:.2f}%)")
        print(f"True positives found:  {tp:,} / {metrics['total_actual_positives']:,} ({metrics['recall']*100:.1f}%)")
        print(f"False positive rate:   {fp:,} / {tn+fp:,} ({fp/(tn+fp)*100:.2f}%)")
        
        # Performance assessment
        print(f"\nPERFORMANCE ASSESSMENT:")
        if metrics['mcc'] > 0.3:
            print("✓ Good performance (MCC > 0.3)")
        elif metrics['mcc'] > 0.1:
            print("⚠ Moderate performance (0.1 < MCC < 0.3)")
        else:
            print("✗ Poor performance (MCC < 0.1)")
        
        if metrics['precision'] < 0.5 and metrics['recall'] > 0.5:
            print("⚠ High recall but low precision - many false positives")
        elif metrics['precision'] > 0.5 and metrics['recall'] < 0.5:
            print("⚠ High precision but low recall - missing many positives")
        elif metrics['precision'] > 0.7 and metrics['recall'] > 0.7:
            print("✓ Good balance of precision and recall")
    
    return metrics, y_pred


def save_new_predictions(df, predictions, threshold, output_path):
    """
    Save new predictions with the applied threshold.
    
    Args:
        df: Original dataframe
        predictions: New predictions
        threshold: Applied threshold
        output_path: Path to save the new predictions
    """
    # Create new dataframe with updated predictions
    new_df = df.copy()
    new_df['predicted_label_original'] = new_df['pred_label'] if 'pred_label' in new_df.columns else (new_df['prob_1'] >= 0.5).astype(int)
    new_df['pred_label'] = predictions
    new_df['threshold_used'] = threshold
    new_df['correct'] = (predictions == df['label']).astype(int)
    
    # Reorder columns for clarity
    cols = ['segment_id'] if 'segment_id' in new_df.columns else []
    cols.extend(['label', 'pred_label', 'predicted_label_original',
                 'prob_class_0', 'prob_1', 'threshold_used', 'correct'])
    
    # Add any additional columns
    for col in new_df.columns:
        if col not in cols:
            cols.append(col)
    
    new_df = new_df[cols]
    
    # Save to file
    new_df.to_csv(output_path, index=False)
    print(f"\nNew predictions saved to: {output_path}")
    
    return new_df


def create_comparison_report(metrics_list, output_path):
    """
    Create a comparison report for multiple thresholds.
    
    Args:
        metrics_list: List of metrics dictionaries
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Create comparison table
        f.write("METRICS COMPARISON TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Threshold':<10} {'Acc':<8} {'Bal.Acc':<8} {'Prec':<8} {'Recall':<8} "
                f"{'F1':<8} {'MCC':<8} {'Spec':<8}\n")
        f.write("-"*80 + "\n")
        
        for m in metrics_list:
            f.write(f"{m['threshold']:<10.3f} {m['accuracy']:<8.4f} {m['balanced_accuracy']:<8.4f} "
                    f"{m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f} "
                    f"{m['mcc']:<8.4f} {m['specificity']:<8.4f}\n")
        
        f.write("\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*80 + "\n")
        
        for m in metrics_list:
            f.write(f"\nThreshold {m['threshold']:.3f}:\n")
            f.write(f"  TP: {m['tp']:8,}    FP: {m['fp']:8,}\n")
            f.write(f"  FN: {m['fn']:8,}    TN: {m['tn']:8,}\n")
            f.write(f"  Predicted Positive: {m['total_positive_predictions']:,} / {m['total_samples']:,}\n")
        
        # Find best threshold for each metric
        f.write("\n" + "="*80 + "\n")
        f.write("BEST THRESHOLDS BY METRIC\n")
        f.write("-"*80 + "\n")
        
        metrics_to_optimize = ['f1', 'mcc', 'balanced_accuracy', 'precision', 'recall']
        for metric in metrics_to_optimize:
            best_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i][metric])
            best = metrics_list[best_idx]
            f.write(f"Best {metric.replace('_', ' ').title()}: {best[metric]:.4f} at threshold {best['threshold']:.3f}\n")
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        
        # Find threshold with best MCC
        best_mcc_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['mcc'])
        best_mcc = metrics_list[best_mcc_idx]
        
        f.write(f"\n1. RECOMMENDED THRESHOLD: {best_mcc['threshold']:.3f}\n")
        f.write(f"   Reason: Highest MCC ({best_mcc['mcc']:.3f}) - best for imbalanced data\n")
        f.write(f"   Performance: Precision={best_mcc['precision']:.3f}, Recall={best_mcc['recall']:.3f}, F1={best_mcc['f1']:.3f}\n")
        
        # Check if default threshold was included
        default = next((m for m in metrics_list if abs(m['threshold'] - 0.5) < 0.01), None)
        if default and best_mcc['threshold'] != 0.5:
            improvement_mcc = best_mcc['mcc'] - default['mcc']
            improvement_f1 = best_mcc['f1'] - default['f1']
            f.write(f"\n2. IMPROVEMENT OVER DEFAULT (0.5):\n")
            sign_mcc = "+" if improvement_mcc > 0 else ""
            sign_f1 = "+" if improvement_f1 > 0 else ""
            f.write(f"   MCC:  {default['mcc']:.3f} → {best_mcc['mcc']:.3f} ({sign_mcc}{improvement_mcc:.3f})\n")
            f.write(f"   F1:   {default['f1']:.3f} → {best_mcc['f1']:.3f} ({sign_f1}{improvement_f1:.3f})\n")
            f.write(f"   FP reduction: {default['fp']:,} → {best_mcc['fp']:,} ({(default['fp']-best_mcc['fp'])/default['fp']*100:.1f}% reduction)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate predictions with optimal threshold')
    
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--threshold', type=str, required=True,
                        help='Threshold value(s) to apply (comma-separated for multiple)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: same as predictions)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save new predictions with applied threshold')
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.threshold.split(',')]
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions_file}")
    df = pd.read_csv(args.predictions_file)
    
    # Check required columns
    required_cols = ['label', 'prob_1']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating {len(df):,} predictions at threshold(s): {thresholds}")
    
    # Evaluate at each threshold
    metrics_list = []
    for threshold in thresholds:
        metrics, predictions = evaluate_at_threshold(df, threshold, verbose=True)
        metrics_list.append(metrics)
        
        # Save predictions if requested
        if args.save_predictions:
            base_name = Path(args.predictions_file).stem
            output_path = os.path.join(args.output_dir, f"{base_name}_threshold_{threshold:.3f}.csv")
            save_new_predictions(df, predictions, threshold, output_path)
    
    # Create comparison report if multiple thresholds
    if len(thresholds) > 1:
        report_path = os.path.join(args.output_dir, 'threshold_comparison_report.txt')
        create_comparison_report(metrics_list, report_path)
        
        # Save metrics comparison as CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv_path = os.path.join(args.output_dir, 'threshold_comparison_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nMetrics comparison saved to: {metrics_csv_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()