#!/usr/bin/env python3
"""
Analyze phage segment prediction results.
Calculate false negative rates per functional category.
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_predictions(prediction_file, output_file=None):
    """
    Analyze prediction results and calculate false negative rates per category.
    
    Args:
        prediction_file: Path to CSV with predictions
        output_file: Optional path to save results (default: print to console)
    
    Expected columns:
        seq_id, start, end, sequence, label, phrog_category, function, 
        phrog_product, phrog_db_category, prob_0, prob_1, pred_label
    """
    # Dictionary to store counts per category
    category_stats = defaultdict(lambda: {
        'total': 0,
        'false_negatives': 0,
        'true_positives': 0
    })
    
    total_phage = 0
    total_fn = 0
    total_tp = 0
    
    try:
        with open(prediction_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Verify required columns exist
            required_cols = ['label', 'pred_label', 'phrog_db_category']
            for col in required_cols:
                if col not in reader.fieldnames:
                    print(f"Error: Required column '{col}' not found in input file")
                    print(f"Available columns: {reader.fieldnames}")
                    return
            
            for row in reader:
                true_label = int(row['label'])
                pred_label = int(row['pred_label'])
                category = row['phrog_db_category'].strip()
                
                # Only analyze true phage segments (label=1)
                if true_label != 1:
                    continue
                
                # Use "unknown" for empty categories
                if not category:
                    category = "no_category"
                
                # Update counts
                category_stats[category]['total'] += 1
                total_phage += 1
                
                if pred_label == 0:
                    # False negative (true phage predicted as non-phage)
                    category_stats[category]['false_negatives'] += 1
                    total_fn += 1
                else:
                    # True positive (true phage predicted as phage)
                    category_stats[category]['true_positives'] += 1
                    total_tp += 1
    
    except FileNotFoundError:
        print(f"Error: File not found: {prediction_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if total_phage == 0:
        print("No phage segments found in the input file")
        return
    
    # Calculate rates and prepare results
    results = []
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        fn_rate = stats['false_negatives'] / stats['total'] if stats['total'] > 0 else 0
        sensitivity = stats['true_positives'] / stats['total'] if stats['total'] > 0 else 0
        
        results.append({
            'category': category,
            'total_segments': stats['total'],
            'true_positives': stats['true_positives'],
            'false_negatives': stats['false_negatives'],
            'sensitivity': sensitivity,
            'false_negative_rate': fn_rate
        })
    
    # Calculate overall statistics
    overall_fn_rate = total_fn / total_phage if total_phage > 0 else 0
    overall_sensitivity = total_tp / total_phage if total_phage > 0 else 0
    
    # Output results
    if output_file:
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['category', 'total_segments', 'true_positives', 
                         'false_negatives', 'sensitivity', 'false_negative_rate']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("PHAGE SEGMENT PREDICTION ANALYSIS")
    print("="*80)
    print(f"\nTotal phage segments analyzed: {total_phage}")
    print(f"Overall sensitivity: {overall_sensitivity:.4f} ({total_tp}/{total_phage})")
    print(f"Overall false negative rate: {overall_fn_rate:.4f} ({total_fn}/{total_phage})")
    
    print("\n" + "-"*80)
    print(f"{'Category':<45} {'Total':>8} {'TP':>8} {'FN':>8} {'Sens':>8} {'FNR':>8}")
    print("-"*80)
    
    for result in results:
        print(f"{result['category']:<45} "
              f"{result['total_segments']:>8} "
              f"{result['true_positives']:>8} "
              f"{result['false_negatives']:>8} "
              f"{result['sensitivity']:>8.4f} "
              f"{result['false_negative_rate']:>8.4f}")
    
    print("-"*80)
    print(f"{'OVERALL':<45} "
          f"{total_phage:>8} "
          f"{total_tp:>8} "
          f"{total_fn:>8} "
          f"{overall_sensitivity:>8.4f} "
          f"{overall_fn_rate:>8.4f}")
    print("="*80)
    
    # Additional insights
    print("\nCATEGORIES WITH HIGHEST FALSE NEGATIVE RATES:")
    sorted_by_fnr = sorted(results, key=lambda x: x['false_negative_rate'], reverse=True)
    for i, result in enumerate(sorted_by_fnr[:10], 1):
        if result['total_segments'] >= 10:  # Only show categories with sufficient samples
            print(f"{i}. {result['category']}: {result['false_negative_rate']:.4f} "
                  f"({result['false_negatives']}/{result['total_segments']})")
    
    print("\nCATEGORIES WITH HIGHEST SENSITIVITY:")
    sorted_by_sens = sorted(results, key=lambda x: x['sensitivity'], reverse=True)
    for i, result in enumerate(sorted_by_sens[:10], 1):
        if result['total_segments'] >= 10:  # Only show categories with sufficient samples
            print(f"{i}. {result['category']}: {result['sensitivity']:.4f} "
                  f"({result['true_positives']}/{result['total_segments']})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze phage segment prediction results. '
                    'Calculate false negative rates per functional category.'
    )
    parser.add_argument(
        'prediction_file',
        help='CSV file with predictions (must include: label, pred_label, phrog_db_category)'
    )
    parser.add_argument(
        '--output_file', '-o',
        help='Output CSV file for results (optional, default: print to console only)'
    )
    
    args = parser.parse_args()
    
    analyze_predictions(args.prediction_file, args.output_file)


if __name__ == '__main__':
    main()
