#!/usr/bin/env python3
"""
Analysis script for comparing prediction outputs from multiple software programs.
Generates metrics tables and taxonomy-based visualizations.

Usage:
    python analyze_predictions.py --input_dir <directory_with_csv_files> [options]
    
Examples:
    # Basic comparison
    python analyze_predictions.py --input_dir results/ --output_dir analysis/
    
    # With taxonomy analysis at genus level
    python analyze_predictions.py --input_dir results/ --taxonomy_file taxonomy.tsv --tax_level genus
    
    # Custom taxonomy level
    python analyze_predictions.py --input_dir results/ --taxonomy_file taxonomy.tsv --tax_level family
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_prediction_files(input_dir):
    """
    Load all CSV prediction files from the input directory.
    
    Args:
        input_dir: Directory containing prediction CSV files
    
    Returns:
        dict: Dictionary mapping software names to DataFrames
    """
    prediction_files = {}
    
    # Find all CSV files
    csv_files = list(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    print(f"Found {len(csv_files)} prediction files:")
    
    for csv_file in csv_files:
        # Extract software name from filename
        # Assumes format: predictions_<dataset>_<software>.csv or similar
        software_name = csv_file.stem.replace("predictions_", "").replace("_output", "")
        
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['label', 'pred_label']
        if not all(col in df.columns for col in required_cols):
            print(f"  ⚠️  Skipping {csv_file.name}: missing required columns")
            continue
        
        prediction_files[software_name] = df
        print(f"  ✓ {software_name}: {len(df)} predictions")
    
    return prediction_files


def calculate_metrics(df, software_name):
    """
    Calculate comprehensive metrics for a single prediction file.
    
    Args:
        df: DataFrame with predictions
        software_name: Name of the software
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {'Software': software_name}
    
    y_true = df['label'].values
    y_pred = df['pred_label'].values
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # Calculate specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['TP'] = tp
        metrics['TN'] = tn
        metrics['FP'] = fp
        metrics['FN'] = fn
    
    # AUC if probabilities are available
    if 'prob_1' in df.columns:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, df['prob_1'].values)
            metrics['AUC-PR'] = average_precision_score(y_true, df['prob_1'].values)
        except:
            metrics['AUC-ROC'] = np.nan
            metrics['AUC-PR'] = np.nan
    else:
        metrics['AUC-ROC'] = np.nan
        metrics['AUC-PR'] = np.nan
    
    # Total predictions
    metrics['Total'] = len(df)
    
    return metrics


def create_metrics_comparison_table(prediction_files, output_dir):
    """
    Create a comparison table of metrics across all software.
    
    Args:
        prediction_files: Dictionary of software names to DataFrames
        output_dir: Directory to save results
    
    Returns:
        pd.DataFrame: Metrics comparison table
    """
    metrics_list = []
    
    for software_name, df in prediction_files.items():
        metrics = calculate_metrics(df, software_name)
        metrics_list.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by F1-Score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'metrics_comparison.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"\nMetrics table saved to: {output_path}")
    
    # Also create a formatted version for display
    display_cols = ['Software', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'MCC', 'AUC-ROC']
    display_df = metrics_df[display_cols].round(4)
    
    # Save formatted version
    formatted_path = os.path.join(output_dir, 'metrics_comparison_formatted.txt')
    with open(formatted_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE METRICS COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(display_df.to_string(index=False))
        f.write("\n\n")
        f.write("Confusion Matrix Details:\n")
        f.write("-"*40 + "\n")
        cm_cols = ['Software', 'TP', 'TN', 'FP', 'FN']
        f.write(metrics_df[cm_cols].to_string(index=False))
    
    print(f"Formatted table saved to: {formatted_path}")
    
    return metrics_df


def load_taxonomy_mapping(taxonomy_file, seq_id_column='seq_id'):
    """
    Load taxonomy mapping file.
    
    Expected format (TSV):
    seq_id  kingdom  phylum  class  order  family  genus  species
    
    Args:
        taxonomy_file: Path to taxonomy TSV file
        seq_id_column: Column name for sequence IDs
    
    Returns:
        pd.DataFrame: Taxonomy mapping
    """
    if not os.path.exists(taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
    
    # Detect delimiter
    with open(taxonomy_file, 'r') as f:
        first_line = f.readline()
        if '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
    
    taxonomy_df = pd.read_csv(taxonomy_file, sep=delimiter)
    
    # Check if seq_id column exists
    if seq_id_column not in taxonomy_df.columns:
        # Try alternative names
        alt_names = ['sequence_id', 'id', 'ID', 'seq_ID', 'segment_id']
        for alt in alt_names:
            if alt in taxonomy_df.columns:
                taxonomy_df[seq_id_column] = taxonomy_df[alt]
                break
        else:
            raise ValueError(f"Column '{seq_id_column}' not found in taxonomy file. Available columns: {taxonomy_df.columns.tolist()}")
    
    # Convert seq_id to string for merging
    taxonomy_df[seq_id_column] = taxonomy_df[seq_id_column].astype(str)
    
    return taxonomy_df


def analyze_by_taxonomy(prediction_files, taxonomy_df, tax_level='genus', output_dir='analysis'):
    """
    Analyze predictions by taxonomic level.
    
    Args:
        prediction_files: Dictionary of software names to DataFrames
        taxonomy_df: DataFrame with taxonomy mapping
        tax_level: Taxonomic level to analyze
        output_dir: Directory to save results
    
    Returns:
        pd.DataFrame: Taxonomy-based metrics
    """
    if tax_level not in taxonomy_df.columns:
        available_levels = [col for col in taxonomy_df.columns if col not in ['seq_id', 'segment_id']]
        raise ValueError(f"Taxonomic level '{tax_level}' not found. Available levels: {available_levels}")
    
    tax_metrics_all = []
    
    for software_name, pred_df in prediction_files.items():
        # Ensure segment_id is string for merging
        if 'segment_id' in pred_df.columns:
            pred_df['segment_id'] = pred_df['segment_id'].astype(str)
        elif 'seq_id' in pred_df.columns:
            pred_df['segment_id'] = pred_df['seq_id'].astype(str)
        
        # Merge with taxonomy
        if 'seq_id' in taxonomy_df.columns:
            taxonomy_df['segment_id'] = taxonomy_df['seq_id'].astype(str)
        
        merged_df = pd.merge(
            pred_df, 
            taxonomy_df[['segment_id', tax_level] if 'segment_id' in taxonomy_df.columns else ['seq_id', tax_level]], 
            left_on='segment_id', 
            right_on='segment_id' if 'segment_id' in taxonomy_df.columns else 'seq_id',
            how='left'
        )
        
        # Group by taxonomic level and calculate metrics
        for taxon in merged_df[tax_level].unique():
            if pd.isna(taxon):
                continue
                
            taxon_df = merged_df[merged_df[tax_level] == taxon]
            
            if len(taxon_df) < 5:  # Skip taxa with too few samples
                continue
            
            y_true = taxon_df['label'].values
            y_pred = taxon_df['pred_label'].values
            
            tax_metrics = {
                'Software': software_name,
                tax_level.capitalize(): taxon,
                'N_samples': len(taxon_df),
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            tax_metrics_all.append(tax_metrics)
    
    tax_metrics_df = pd.DataFrame(tax_metrics_all)
    
    # Save taxonomy metrics
    tax_output_path = os.path.join(output_dir, f'metrics_by_{tax_level}.csv')
    tax_metrics_df.to_csv(tax_output_path, index=False)
    print(f"\nTaxonomy metrics saved to: {tax_output_path}")
    
    return tax_metrics_df


def create_visualizations(metrics_df, tax_metrics_df, tax_level, output_dir):
    """
    Create visualization plots for metrics comparison.
    
    Args:
        metrics_df: Overall metrics DataFrame
        tax_metrics_df: Taxonomy-based metrics DataFrame
        tax_level: Taxonomic level used
        output_dir: Directory to save plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Overall metrics comparison bar plot
    ax1 = plt.subplot(2, 3, 1)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    plot_df = metrics_df[['Software'] + metrics_to_plot].set_index('Software')
    plot_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Overall Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Software')
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. F1-Score ranking
    ax2 = plt.subplot(2, 3, 2)
    sorted_df = metrics_df.sort_values('F1-Score', ascending=True)
    colors = plt.cm.RdYlGn(sorted_df['F1-Score'].values)
    ax2.barh(sorted_df['Software'], sorted_df['F1-Score'], color=colors)
    ax2.set_title('F1-Score Ranking')
    ax2.set_xlabel('F1-Score')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision vs Recall scatter plot
    ax3 = plt.subplot(2, 3, 3)
    for idx, row in metrics_df.iterrows():
        ax3.scatter(row['Recall'], row['Precision'], s=100, alpha=0.7, label=row['Software'])
    ax3.set_title('Precision vs Recall Trade-off')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim([0, 1.05])
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add diagonal line
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 4. MCC comparison
    ax4 = plt.subplot(2, 3, 4)
    mcc_sorted = metrics_df.sort_values('MCC', ascending=True)
    colors = plt.cm.RdYlGn((mcc_sorted['MCC'].values + 1) / 2)  # Normalize MCC from [-1,1] to [0,1]
    ax4.barh(mcc_sorted['Software'], mcc_sorted['MCC'], color=colors)
    ax4.set_title('Matthews Correlation Coefficient (MCC)')
    ax4.set_xlabel('MCC')
    ax4.set_xlim([-1, 1])
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # If taxonomy metrics are available
    if tax_metrics_df is not None and not tax_metrics_df.empty:
        # 5. F1-Score by taxonomy (top taxa)
        ax5 = plt.subplot(2, 3, 5)
        
        # Get top taxa by sample size
        top_taxa = tax_metrics_df.groupby(tax_level.capitalize())['N_samples'].sum().nlargest(10).index
        tax_subset = tax_metrics_df[tax_metrics_df[tax_level.capitalize()].isin(top_taxa)]
        
        # Pivot for heatmap
        pivot_df = tax_subset.pivot_table(
            index=tax_level.capitalize(),
            columns='Software',
            values='F1-Score',
            aggfunc='mean'
        )
        
        if not pivot_df.empty:
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                       vmin=0, vmax=1, ax=ax5, cbar_kws={'label': 'F1-Score'})
            ax5.set_title(f'F1-Score by {tax_level.capitalize()} (Top 10)')
            ax5.set_xlabel('Software')
            ax5.set_ylabel(tax_level.capitalize())
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax5.yaxis.get_majorticklabels(), rotation=0)
        
        # 6. Average performance by taxonomy
        ax6 = plt.subplot(2, 3, 6)
        avg_by_software = tax_subset.groupby('Software')['F1-Score'].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(avg_by_software.values)
        ax6.barh(avg_by_software.index, avg_by_software.values, color=colors)
        ax6.set_title(f'Average F1-Score across {tax_level.capitalize()}')
        ax6.set_xlabel('Average F1-Score')
        ax6.set_xlim([0, 1])
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Performance Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    # Create additional taxonomy-specific plots if available
    if tax_metrics_df is not None and not tax_metrics_df.empty:
        create_taxonomy_detailed_plots(tax_metrics_df, tax_level, output_dir)


def create_taxonomy_detailed_plots(tax_metrics_df, tax_level, output_dir):
    """
    Create detailed taxonomy-specific plots.
    
    Args:
        tax_metrics_df: Taxonomy metrics DataFrame
        tax_level: Taxonomic level
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot of F1-scores by software
    ax1 = axes[0, 0]
    tax_metrics_df.boxplot(column='F1-Score', by='Software', ax=ax1)
    ax1.set_title(f'F1-Score Distribution by Software\n(across all {tax_level})')
    ax1.set_xlabel('Software')
    ax1.set_ylabel('F1-Score')
    ax1.set_ylim([0, 1.05])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Performance vs sample size
    ax2 = axes[0, 1]
    for software in tax_metrics_df['Software'].unique():
        software_df = tax_metrics_df[tax_metrics_df['Software'] == software]
        ax2.scatter(software_df['N_samples'], software_df['F1-Score'], 
                   alpha=0.6, label=software, s=50)
    ax2.set_title(f'F1-Score vs Sample Size by {tax_level.capitalize()}')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('F1-Score')
    ax2.set_xscale('log')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Metric correlation heatmap for best software
    ax3 = axes[1, 0]
    best_software = tax_metrics_df.groupby('Software')['F1-Score'].mean().idxmax()
    best_df = tax_metrics_df[tax_metrics_df['Software'] == best_software]
    
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    corr_matrix = best_df[metrics_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, ax=ax3, vmin=-1, vmax=1)
    ax3.set_title(f'Metric Correlations\n({best_software})')
    
    # 4. Top performing taxa
    ax4 = axes[1, 1]
    avg_performance = tax_metrics_df.groupby(tax_level.capitalize())['F1-Score'].mean()
    top_performers = avg_performance.nlargest(15).sort_values()
    colors = plt.cm.RdYlGn(top_performers.values)
    ax4.barh(range(len(top_performers)), top_performers.values, color=colors)
    ax4.set_yticks(range(len(top_performers)))
    ax4.set_yticklabels(top_performers.index, fontsize=8)
    ax4.set_title(f'Top 15 {tax_level.capitalize()} by Average F1-Score')
    ax4.set_xlabel('Average F1-Score')
    ax4.set_xlim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Detailed {tax_level.capitalize()}-Level Analysis', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    detail_plot_path = os.path.join(output_dir, f'{tax_level}_detailed_analysis.png')
    plt.savefig(detail_plot_path, dpi=150, bbox_inches='tight')
    print(f"Detailed {tax_level} analysis saved to: {detail_plot_path}")


def generate_summary_report(metrics_df, tax_metrics_df, tax_level, output_dir):
    """
    Generate a text summary report of the analysis.
    
    Args:
        metrics_df: Overall metrics DataFrame
        tax_metrics_df: Taxonomy metrics DataFrame (can be None)
        tax_level: Taxonomic level used
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PREDICTION ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall performance summary
        f.write("1. OVERALL PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n\n")
        
        # Best performing software
        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
        f.write(f"Best F1-Score: {best_f1['Software']} ({best_f1['F1-Score']:.4f})\n")
        
        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        f.write(f"Best Accuracy: {best_acc['Software']} ({best_acc['Accuracy']:.4f})\n")
        
        best_prec = metrics_df.loc[metrics_df['Precision'].idxmax()]
        f.write(f"Best Precision: {best_prec['Software']} ({best_prec['Precision']:.4f})\n")
        
        best_recall = metrics_df.loc[metrics_df['Recall'].idxmax()]
        f.write(f"Best Recall: {best_recall['Software']} ({best_recall['Recall']:.4f})\n")
        
        best_mcc = metrics_df.loc[metrics_df['MCC'].idxmax()]
        f.write(f"Best MCC: {best_mcc['Software']} ({best_mcc['MCC']:.4f})\n\n")
        
        # Performance ranges
        f.write("Performance Ranges:\n")
        for metric in ['F1-Score', 'Accuracy', 'Precision', 'Recall']:
            min_val = metrics_df[metric].min()
            max_val = metrics_df[metric].max()
            range_val = max_val - min_val
            f.write(f"  {metric}: {min_val:.4f} - {max_val:.4f} (range: {range_val:.4f})\n")
        
        f.write("\n")
        
        # Detailed metrics table
        f.write("2. DETAILED METRICS TABLE\n")
        f.write("-"*40 + "\n\n")
        display_cols = ['Software', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'MCC']
        f.write(metrics_df[display_cols].round(4).to_string(index=False))
        f.write("\n\n")
        
        # Taxonomy analysis if available
        if tax_metrics_df is not None and not tax_metrics_df.empty:
            f.write(f"3. TAXONOMY ANALYSIS ({tax_level.upper()})\n")
            f.write("-"*40 + "\n\n")
            
            # Average performance by software
            f.write("Average Performance by Software:\n")
            avg_by_software = tax_metrics_df.groupby('Software')[['F1-Score', 'Accuracy']].mean()
            f.write(avg_by_software.round(4).to_string())
            f.write("\n\n")
            
            # Most variable taxa
            f.write("Most Variable Taxa (by F1-Score std dev across software):\n")
            std_by_taxa = tax_metrics_df.groupby(tax_level.capitalize())['F1-Score'].std().nlargest(10)
            for taxon, std in std_by_taxa.items():
                f.write(f"  {taxon}: {std:.4f}\n")
            f.write("\n")
            
            # Taxa with most samples
            f.write("Taxa with Most Samples:\n")
            samples_by_taxa = tax_metrics_df.groupby(tax_level.capitalize())['N_samples'].sum().nlargest(10)
            for taxon, n in samples_by_taxa.items():
                f.write(f"  {taxon}: {n} samples\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nAnalysis report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare prediction outputs from multiple software')
    
    # Input arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing prediction CSV files')
    
    # Taxonomy arguments
    parser.add_argument('--taxonomy_file', type=str, default=None,
                        help='Path to taxonomy mapping file (TSV/CSV format)')
    parser.add_argument('--tax_level', type=str, default='genus',
                        choices=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'],
                        help='Taxonomic level for analysis (default: genus)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results (default: analysis_results)')
    
    # Display options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating visualization plots')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PREDICTION ANALYSIS TOOL")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    if args.taxonomy_file:
        print(f"Taxonomy file: {args.taxonomy_file}")
        print(f"Taxonomic level: {args.tax_level}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction files
    print("\n1. Loading prediction files...")
    prediction_files = load_prediction_files(args.input_dir)
    
    if not prediction_files:
        print("Error: No valid prediction files found")
        sys.exit(1)
    
    # Calculate overall metrics
    print("\n2. Calculating metrics...")
    metrics_df = create_metrics_comparison_table(prediction_files, args.output_dir)
    
    # Taxonomy analysis if file provided
    tax_metrics_df = None
    if args.taxonomy_file:
        print(f"\n3. Performing taxonomy analysis at {args.tax_level} level...")
        try:
            taxonomy_df = load_taxonomy_mapping(args.taxonomy_file)
            print(f"   Loaded taxonomy for {len(taxonomy_df)} sequences")
            
            tax_metrics_df = analyze_by_taxonomy(
                prediction_files, 
                taxonomy_df, 
                args.tax_level, 
                args.output_dir
            )
            print(f"   Analyzed {len(tax_metrics_df)} {args.tax_level}-software combinations")
        except Exception as e:
            print(f"   Warning: Taxonomy analysis failed: {e}")
            print("   Continuing with overall metrics only...")
    
    # Create visualizations
    if not args.no_plots:
        print("\n4. Creating visualizations...")
        create_visualizations(metrics_df, tax_metrics_df, args.tax_level, args.output_dir)
    
    # Generate summary report
    print("\n5. Generating summary report...")
    generate_summary_report(metrics_df, tax_metrics_df, args.tax_level, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)
    
    # Print summary to console
    print("\nQUICK SUMMARY:")
    print("-"*40)
    best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    print(f"Best F1-Score: {best_f1['Software']} ({best_f1['F1-Score']:.4f})")
    print(f"Total software compared: {len(prediction_files)}")
    if tax_metrics_df is not None:
        print(f"Taxa analyzed: {tax_metrics_df[args.tax_level.capitalize()].nunique()}")


if __name__ == "__main__":
    main()