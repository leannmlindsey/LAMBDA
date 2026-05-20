#!/usr/bin/env python3
"""
Threshold analysis script for finding optimal classification threshold.
Generates ROC curves, Precision-Recall curves, and metrics at different thresholds.

Usage:
    python analyze_threshold.py --predictions_file <path_to_predictions_csv> [options]
    
Example:
    python analyze_threshold.py --predictions_file inference_results/predictions_dataset_1024_512_prokbert-mini.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, balanced_accuracy_score
)
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """
    Calculate metrics at a specific threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
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
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'positive_predictions': np.sum(y_pred),
        'positive_rate': np.mean(y_pred)
    }
    
    return metrics


def find_optimal_thresholds(y_true, y_prob):
    """
    Find optimal thresholds for different metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
    
    Returns:
        dict: Dictionary of optimal thresholds
    """
    thresholds = np.linspace(0, 1, 101)
    metrics_list = []
    
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold)
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    optimal = {
        'f1': metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold'],
        'mcc': metrics_df.loc[metrics_df['mcc'].idxmax(), 'threshold'],
        'balanced_acc': metrics_df.loc[metrics_df['balanced_accuracy'].idxmax(), 'threshold'],
        'precision_90': metrics_df[metrics_df['precision'] >= 0.9]['threshold'].min() if any(metrics_df['precision'] >= 0.9) else None,
        'recall_90': metrics_df[metrics_df['recall'] >= 0.9]['threshold'].min() if any(metrics_df['recall'] >= 0.9) else None,
    }
    
    # Find threshold for Youden's J statistic (maximize TPR - FPR)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal['youden'] = roc_thresholds[optimal_idx]
    
    return optimal, metrics_df


def create_threshold_analysis_plots(y_true, y_prob, output_dir):
    """
    Create comprehensive threshold analysis plots.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_dir: Directory to save plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. ROC Curve
    ax1 = plt.subplot(3, 3, 1)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Mark optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
            label=f'Optimal (thresh={optimal_threshold:.3f})')
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(3, 3, 2)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    ax2.plot(recall, precision, linewidth=2, label=f'PR (AP = {avg_precision:.3f})')
    
    # Mark F1-optimal point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    ax2.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=10,
            label=f'Best F1 (thresh={pr_thresholds[optimal_idx]:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics vs Threshold
    ax3 = plt.subplot(3, 3, 3)
    thresholds = np.linspace(0, 1, 101)
    metrics = {'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'balanced_acc': []}
    
    for threshold in thresholds:
        m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
        metrics['precision'].append(m['precision'])
        metrics['recall'].append(m['recall'])
        metrics['f1'].append(m['f1'])
        metrics['mcc'].append(m['mcc'])
        metrics['balanced_acc'].append(m['balanced_accuracy'])
    
    ax3.plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
    ax3.plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
    ax3.plot(thresholds, metrics['f1'], label='F1-Score', linewidth=2)
    ax3.plot(thresholds, metrics['balanced_acc'], label='Balanced Acc', linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Metrics vs Threshold')
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. MCC vs Threshold
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(thresholds, metrics['mcc'], linewidth=2, color='purple')
    optimal_mcc_idx = np.argmax(metrics['mcc'])
    ax4.plot(thresholds[optimal_mcc_idx], metrics['mcc'][optimal_mcc_idx], 'ro', markersize=10,
            label=f'Max MCC = {metrics["mcc"][optimal_mcc_idx]:.3f} at {thresholds[optimal_mcc_idx]:.3f}')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Matthews Correlation Coefficient')
    ax4.set_title('MCC vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    
    # 5. Predicted Positive Rate vs Threshold
    ax5 = plt.subplot(3, 3, 5)
    positive_rates = []
    for threshold in thresholds:
        pred_positive = np.mean(y_prob >= threshold)
        positive_rates.append(pred_positive)
    
    actual_positive_rate = np.mean(y_true)
    ax5.plot(thresholds, positive_rates, linewidth=2, label='Predicted Positive Rate')
    ax5.axhline(y=actual_positive_rate, color='red', linestyle='--', 
               label=f'Actual Positive Rate ({actual_positive_rate:.3f})')
    ax5.set_xlabel('Threshold')
    ax5.set_ylabel('Positive Prediction Rate')
    ax5.set_title('Positive Prediction Rate vs Threshold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # 6. Distribution of Probabilities
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Class 0', color='blue', density=True)
    ax6.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Class 1', color='red', density=True)
    ax6.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax6.set_xlabel('Predicted Probability')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution of Predicted Probabilities')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Confusion Matrix at Different Thresholds
    ax7 = plt.subplot(3, 3, 7)
    test_thresholds = [0.3, 0.5, 0.7, 0.9]
    cm_data = []
    for thresh in test_thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_data.append([thresh, tp, tn, fp, fn])
    
    cm_df = pd.DataFrame(cm_data, columns=['Threshold', 'TP', 'TN', 'FP', 'FN'])
    cm_df.set_index('Threshold').plot(kind='bar', ax=ax7, width=0.8)
    ax7.set_title('Confusion Matrix Components')
    ax7.set_xlabel('Threshold')
    ax7.set_ylabel('Count')
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)
    
    # 8. Trade-off Plot
    ax8 = plt.subplot(3, 3, 8)
    specificity = []
    sensitivity = []
    for threshold in thresholds:
        m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
        specificity.append(m['specificity'])
        sensitivity.append(m['recall'])
    
    ax8.plot(thresholds, sensitivity, label='Sensitivity (Recall)', linewidth=2)
    ax8.plot(thresholds, specificity, label='Specificity', linewidth=2)
    ax8.plot(thresholds, np.array(sensitivity) + np.array(specificity) - 1, 
            label="Youden's J", linewidth=2, linestyle='--')
    ax8.set_xlabel('Threshold')
    ax8.set_ylabel('Score')
    ax8.set_title('Sensitivity-Specificity Trade-off')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([0, 1])
    ax8.set_ylim([-0.1, 1.1])
    
    # 9. Summary Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    optimal, _ = find_optimal_thresholds(y_true, y_prob)
    
    summary_data = []
    for name, thresh in optimal.items():
        if thresh is not None:
            m = calculate_metrics_at_threshold(y_true, y_prob, thresh)
            summary_data.append([
                name.replace('_', ' ').title(),
                f"{thresh:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['mcc']:.3f}"
            ])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Optimization', 'Threshold', 'Precision', 'Recall', 'F1', 'MCC'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax9.set_title('Optimal Thresholds Summary', fontsize=11, pad=20)
    
    plt.suptitle('Threshold Analysis Report', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Threshold analysis plot saved to: {plot_path}")
    
    plt.close()


def save_analysis_results(y_true, y_prob, optimal_thresholds, metrics_df, output_dir):
    """
    Save analysis results to files.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        optimal_thresholds: Dictionary of optimal thresholds
        metrics_df: DataFrame with metrics at all thresholds
        output_dir: Directory to save results
    """
    # Save metrics at all thresholds
    metrics_csv_path = os.path.join(output_dir, 'metrics_all_thresholds.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics at all thresholds saved to: {metrics_csv_path}")
    
    # Save optimal thresholds with detailed metrics
    optimal_metrics = []
    for name, threshold in optimal_thresholds.items():
        if threshold is not None:
            m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
            m['optimization_method'] = name
            optimal_metrics.append(m)
    
    optimal_df = pd.DataFrame(optimal_metrics)
    optimal_csv_path = os.path.join(output_dir, 'optimal_thresholds.csv')
    optimal_df.to_csv(optimal_csv_path, index=False)
    print(f"Optimal thresholds saved to: {optimal_csv_path}")
    
    # Save text summary
    summary_path = os.path.join(output_dir, 'threshold_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Dataset statistics
        f.write("DATASET STATISTICS\n")
        f.write("-"*40 + "\n")
        n_samples = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_samples - n_positive
        f.write(f"Total samples: {n_samples:,}\n")
        f.write(f"Positive samples: {n_positive:,} ({n_positive/n_samples*100:.2f}%)\n")
        f.write(f"Negative samples: {n_negative:,} ({n_negative/n_samples*100:.2f}%)\n")
        f.write(f"Class imbalance ratio: {n_negative/n_positive:.1f}:1\n\n")
        
        # Default threshold (0.5) performance
        f.write("DEFAULT THRESHOLD (0.5) PERFORMANCE\n")
        f.write("-"*40 + "\n")
        default_metrics = calculate_metrics_at_threshold(y_true, y_prob, 0.5)
        f.write(f"Accuracy:           {default_metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:  {default_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Precision:          {default_metrics['precision']:.4f}\n")
        f.write(f"Recall:             {default_metrics['recall']:.4f}\n")
        f.write(f"Specificity:        {default_metrics['specificity']:.4f}\n")
        f.write(f"F1-Score:           {default_metrics['f1']:.4f}\n")
        f.write(f"MCC:                {default_metrics['mcc']:.4f}\n")
        f.write(f"Positive predictions: {default_metrics['positive_predictions']:,} "
                f"({default_metrics['positive_rate']*100:.2f}% of all predictions)\n\n")
        
        # Optimal thresholds
        f.write("OPTIMAL THRESHOLDS\n")
        f.write("-"*40 + "\n")
        for name, threshold in optimal_thresholds.items():
            if threshold is not None:
                m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
                f.write(f"\n{name.replace('_', ' ').upper()}:\n")
                f.write(f"  Threshold:         {threshold:.4f}\n")
                f.write(f"  Precision:         {m['precision']:.4f}\n")
                f.write(f"  Recall:            {m['recall']:.4f}\n")
                f.write(f"  F1-Score:          {m['f1']:.4f}\n")
                f.write(f"  MCC:               {m['mcc']:.4f}\n")
                f.write(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}\n")
                f.write(f"  Positive predictions: {m['positive_predictions']:,}\n")
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        best_mcc_thresh = optimal_thresholds['mcc']
        best_f1_thresh = optimal_thresholds['f1']
        
        f.write(f"\n1. For best overall balance (MCC), use threshold: {best_mcc_thresh:.4f}\n")
        mcc_m = calculate_metrics_at_threshold(y_true, y_prob, best_mcc_thresh)
        f.write(f"   This gives: Precision={mcc_m['precision']:.3f}, "
                f"Recall={mcc_m['recall']:.3f}, MCC={mcc_m['mcc']:.3f}\n")
        
        f.write(f"\n2. For best F1-Score, use threshold: {best_f1_thresh:.4f}\n")
        f1_m = calculate_metrics_at_threshold(y_true, y_prob, best_f1_thresh)
        f.write(f"   This gives: Precision={f1_m['precision']:.3f}, "
                f"Recall={f1_m['recall']:.3f}, F1={f1_m['f1']:.3f}\n")
        
        if optimal_thresholds['precision_90']:
            f.write(f"\n3. For 90% precision, use threshold: {optimal_thresholds['precision_90']:.4f}\n")
            p90_m = calculate_metrics_at_threshold(y_true, y_prob, optimal_thresholds['precision_90'])
            f.write(f"   This gives: Precision={p90_m['precision']:.3f}, "
                    f"Recall={p90_m['recall']:.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze classification thresholds and find optimal values')
    
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to predictions CSV file with probabilities')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: same as predictions file)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions_file}")
    df = pd.read_csv(args.predictions_file)
    
    # Check required columns
    required_cols = ['label', 'prob_1']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found in file")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    y_true = df['label'].values
    y_prob = df['prob_1'].values
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions_file)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    print(f"Analyzing {len(y_true):,} predictions...")
    
    # Find optimal thresholds
    print("\nFinding optimal thresholds...")
    optimal_thresholds, metrics_df = find_optimal_thresholds(y_true, y_prob)
    
    # Create visualizations
    if not args.no_plots:
        print("\nGenerating threshold analysis plots...")
        create_threshold_analysis_plots(y_true, y_prob, args.output_dir)
    
    # Save results
    print("\nSaving analysis results...")
    save_analysis_results(y_true, y_prob, optimal_thresholds, metrics_df, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print("-"*40)
    for name, threshold in optimal_thresholds.items():
        if threshold is not None:
            m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
            print(f"{name.replace('_', ' ').title():15s}: threshold={threshold:.3f}, "
                  f"F1={m['f1']:.3f}, MCC={m['mcc']:.3f}")


if __name__ == "__main__":
    main()