#!/usr/bin/env python3
"""
Create categorized PDF reports of prophage detection plots.

Organizes plots by performance category (high, medium, low) and creates
PDFs with 6 images per page, annotated with taxonomy and GC content.

Can work in two modes:
1. With genome_stats.csv from analyze_performance_factors.py (has F1, GC, taxonomy)
2. Standalone with plot_summary.json (uses precision, no GC/taxonomy)

Usage:
  # Mode 1: With full stats (run on remote machine with all data)
  python create_categorized_pdfs.py \
      --plots_dir ./lambda_plots \
      --genome_stats ./performance_analysis/genome_stats.csv

  # Mode 2: Standalone with plot_summary.json
  python create_categorized_pdfs.py \
      --plots_dir ./lambda_plots \
      --summary_json ./lambda_plots/plot_summary.json
"""

import argparse
import csv
import json
import math
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np


def load_plot_summary(summary_path):
    """Load plot summary JSON with performance metrics.

    Auto-detects format: plot_summary.json (flat) vs clustering_results.json (nested).
    """
    with open(summary_path, 'r') as f:
        data = json.load(f)

    # Check if this is clustering_results.json format (has simple_metrics nested)
    if data and isinstance(data, list) and len(data) > 0:
        first_entry = data[0]
        if isinstance(first_entry, dict) and 'simple_metrics' in first_entry:
            # Convert clustering_results.json format to flat format
            print("  (Detected clustering_results.json format, extracting metrics)")
            flat_data = []
            for entry in data:
                metrics = entry.get('simple_metrics', {})
                flat_entry = {
                    'assembly': entry['assembly'],
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1'),
                    'mcc': metrics.get('mcc'),
                    'gt_regions': entry.get('gt_regions', metrics.get('gt_regions')),
                }
                flat_data.append(flat_entry)

            # Debug: show first entry
            if flat_data:
                first = flat_data[0]
                print(f"  Sample metrics - R:{first.get('recall'):.3f} P:{first.get('precision'):.3f} MCC:{first.get('mcc'):.3f}")

            return flat_data

    return data


def load_ground_truth_metadata(csv_path):
    """Load taxonomy and other metadata from ground truth CSV."""
    metadata = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['Assembly']
            if assembly not in metadata:
                metadata[assembly] = {
                    'organism': row.get('Organism Name', row.get('organism', '')),
                    'ncbi_id': row.get('NCBI Id', row.get('ncbi_id', '')),
                }
    return metadata


def load_genome_stats(stats_path):
    """Load genome stats CSV from analyze_performance_factors.py output.

    This CSV contains: assembly, bin, f1, precision, recall, mcc,
    gc_content, genome_length, num_gt_regions, avg_prophage_size,
    total_prophage_bp, organism, publication
    """
    stats = {}
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['assembly']

            # Parse numeric fields safely
            gc_val = row.get('gc_content', '')
            f1_val = row.get('f1', '')
            precision_val = row.get('precision', '')
            recall_val = row.get('recall', '')
            mcc_val = row.get('mcc', '')

            stats[assembly] = {
                'gc_content': float(gc_val) if gc_val and gc_val != 'None' else None,
                'organism': row.get('organism', ''),
                'f1': float(f1_val) if f1_val and f1_val != 'None' else None,
                'precision': float(precision_val) if precision_val and precision_val != 'None' else None,
                'recall': float(recall_val) if recall_val and recall_val != 'None' else None,
                'mcc': float(mcc_val) if mcc_val and mcc_val != 'None' else None,
                'bin': row.get('bin', ''),
                'publication': row.get('publication', ''),
            }
    return stats


def load_clustering_results(results_path):
    """Load clustering_results.json from cluster_activations.py output.

    This provides precision, recall, F1, MCC metrics per genome.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    stats = {}
    for entry in data:
        assembly = entry['assembly']
        metrics = entry.get('simple_metrics', {})
        stats[assembly] = {
            'f1': metrics.get('f1'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'mcc': metrics.get('mcc'),
        }
    return stats


def calculate_gc_from_fasta(fasta_path):
    """Calculate GC content from FASTA file."""
    seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq.append(line.strip().upper())
    sequence = ''.join(seq)
    if not sequence:
        return None
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def categorize_genomes_from_stats(genome_stats, high_thresh=0.7, low_thresh=0.3):
    """Categorize genomes using genome_stats. Priority: recall > MCC > F1 > precision."""
    categories = {'high': [], 'medium': [], 'low': []}

    # Determine which metric to use (priority: recall > mcc > f1 > precision)
    has_recall = any(gs.get('recall') is not None for gs in genome_stats.values())
    has_mcc = any(gs.get('mcc') is not None for gs in genome_stats.values())
    has_f1 = any(gs.get('f1') is not None for gs in genome_stats.values())

    if has_recall:
        metric_key = 'recall'
    elif has_mcc:
        metric_key = 'mcc'
    elif has_f1:
        metric_key = 'f1'
    else:
        metric_key = 'precision'

    for assembly, stats in genome_stats.items():
        metric_value = stats.get(metric_key)

        if metric_value is None:
            continue

        # Clamp to valid range (MCC can be -1 to 1, but we treat negative as low)
        metric_value = min(max(metric_value, -1.0), 1.0)

        item = {
            'assembly': assembly,
            'f1': stats.get('f1'),
            'mcc': stats.get('mcc'),
            'precision': stats.get('precision'),
            'recall': stats.get('recall'),
            'gc_content': stats.get('gc_content'),
            'organism': stats.get('organism'),
        }

        if metric_value >= high_thresh:
            categories['high'].append((assembly, metric_value, item))
        elif metric_value >= low_thresh:
            categories['medium'].append((assembly, metric_value, item))
        else:
            categories['low'].append((assembly, metric_value, item))

    # Sort each category by metric value (descending)
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1], reverse=True)

    return categories, metric_key


def categorize_genomes_from_summary(summary_data, genome_stats=None,
                                    high_thresh=0.7, low_thresh=0.3):
    """Categorize genomes using summary data. Priority: recall > MCC > F1 > precision."""
    categories = {'high': [], 'medium': [], 'low': []}

    # Determine which metric to use (priority: recall > mcc > f1 > precision)
    # First check summary_data itself (may have metrics from clustering_results.json)
    has_recall = any(item.get('recall') is not None for item in summary_data)
    has_mcc = any(item.get('mcc') is not None for item in summary_data)
    has_f1 = any(item.get('f1') is not None for item in summary_data)

    # Also check genome_stats if provided
    if genome_stats:
        has_recall = has_recall or any(gs.get('recall') is not None for gs in genome_stats.values())
        has_mcc = has_mcc or any(gs.get('mcc') is not None for gs in genome_stats.values())
        has_f1 = has_f1 or any(gs.get('f1') is not None for gs in genome_stats.values())

    if has_recall:
        metric_key = 'recall'
    elif has_mcc:
        metric_key = 'mcc'
    elif has_f1:
        metric_key = 'f1'
    else:
        metric_key = 'precision'

    for item in summary_data:
        assembly = item['assembly']

        # Get metric value - check item first, then genome_stats
        value = item.get(metric_key)
        if value is None and genome_stats and assembly in genome_stats:
            value = genome_stats[assembly].get(metric_key)
        if value is None:
            value = item.get('precision', 0)

        # Handle edge cases (precision > 1 means calculation error, treat as high)
        if value is not None and value > 1.0:
            value = 1.0

        if value is None:
            continue

        item['metric_used'] = metric_key
        item['metric_value'] = value

        if value >= high_thresh:
            categories['high'].append((assembly, value, item))
        elif value >= low_thresh:
            categories['medium'].append((assembly, value, item))
        else:
            categories['low'].append((assembly, value, item))

    # Sort each category by metric value (descending)
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1], reverse=True)

    return categories, metric_key


def create_annotated_plot(plot_path, assembly, metric_value, taxonomy, gc_content,
                          metric_name='Precision'):
    """Create an annotated version of the plot with taxonomy and GC content."""
    # Load the original plot
    img = Image.open(plot_path)

    # Create a new figure with space for annotations
    fig_width = 10
    fig_height = 8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Display the image
    ax.imshow(img)
    ax.axis('off')

    # Add annotations as title/subtitle
    title_text = f"{assembly}"
    subtitle_parts = []
    subtitle_parts.append(f"{metric_name}: {metric_value:.3f}")

    if taxonomy and taxonomy != 'Unknown' and taxonomy != 'N/A':
        # Truncate long taxonomy names
        if len(taxonomy) > 50:
            taxonomy = taxonomy[:47] + "..."
        subtitle_parts.append(f"Taxonomy: {taxonomy}")

    if gc_content is not None:
        subtitle_parts.append(f"GC: {gc_content:.1%}")

    subtitle_text = " | ".join(subtitle_parts)

    # Add title at top
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(subtitle_text, fontsize=10, pad=5)

    plt.tight_layout()

    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def create_pdf_report(categories, category_name, plots_dir, output_path,
                      genome_stats=None, plots_per_page=3, use_f1=True):
    """Create a PDF with plots arranged 3 per page (single column)."""
    genomes = categories.get(category_name, [])

    if not genomes:
        print(f"  No genomes in {category_name} category, skipping PDF")
        return

    print(f"  Creating {category_name} PDF with {len(genomes)} genomes...")

    # Calculate grid layout (1 column x 3 rows = 3 per page)
    cols = 1
    rows = 3

    with PdfPages(output_path) as pdf:
        n_pages = math.ceil(len(genomes) / plots_per_page)

        for page_idx in range(n_pages):
            start_idx = page_idx * plots_per_page
            end_idx = min(start_idx + plots_per_page, len(genomes))
            page_genomes = genomes[start_idx:end_idx]

            # Create figure for this page (single column, taller aspect ratio)
            fig, axes = plt.subplots(rows, cols, figsize=(12, 18))
            metric_label = "F1" if use_f1 else "Precision"
            fig.suptitle(f'{category_name.upper()} Performance Genomes (Page {page_idx + 1}/{n_pages})',
                        fontsize=16, fontweight='bold', y=0.995)

            # Ensure axes is always a list (even with single column)
            if rows == 1:
                axes_flat = [axes]
            else:
                axes_flat = axes.flatten() if hasattr(axes, 'flatten') else list(axes)

            for i, (assembly, metric_value, item) in enumerate(page_genomes):
                ax = axes_flat[i]

                # Find plot file - try multiple naming patterns
                plot_path = plots_dir / f"{assembly}_plot.png"
                if not plot_path.exists():
                    plot_path = plots_dir / f"{assembly}_clusters.png"
                if not plot_path.exists():
                    # Try alternative naming with glob
                    possible_plots = list(plots_dir.glob(f"*{assembly}*.png"))
                    if possible_plots:
                        plot_path = possible_plots[0]
                    else:
                        ax.text(0.5, 0.5, f"Plot not found:\n{assembly}",
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(assembly, fontsize=10)
                        ax.axis('off')
                        continue

                # Get metadata from item or genome_stats
                taxonomy = item.get('organism', 'N/A')
                gc_content = item.get('gc_content')
                precision = item.get('precision')
                recall = item.get('recall')
                mcc = item.get('mcc')
                f1 = item.get('f1')

                # Override with genome_stats if available
                if genome_stats and assembly in genome_stats:
                    gs = genome_stats[assembly]
                    if gs.get('gc_content') is not None:
                        gc_content = gs['gc_content']
                    if gs.get('organism') and gs['organism'] != 'Unknown':
                        taxonomy = gs['organism']
                    if gs.get('precision') is not None:
                        precision = gs['precision']
                    if gs.get('recall') is not None:
                        recall = gs['recall']
                    if gs.get('mcc') is not None:
                        mcc = gs['mcc']
                    if gs.get('f1') is not None:
                        f1 = gs['f1']

                # Load and display plot
                try:
                    img = Image.open(plot_path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading plot:\n{e}",
                           ha='center', va='center', transform=ax.transAxes)

                ax.axis('off')
                # No title needed - metrics are already on the PNG from cluster_activations.py

            # Hide unused subplots
            for i in range(len(page_genomes), plots_per_page):
                axes_flat[i].axis('off')
                axes_flat[i].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create categorized PDF reports of prophage detection plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: With genome_stats.csv (preferred - has F1, taxonomy, GC)
  python create_categorized_pdfs.py \\
      --plots_dir ./lambda_plots \\
      --genome_stats ./performance_analysis/genome_stats.csv

  # Mode 2: Standalone with plot_summary.json (uses precision only)
  python create_categorized_pdfs.py \\
      --plots_dir ./lambda_plots \\
      --summary_json ./lambda_plots/plot_summary.json
"""
    )
    parser.add_argument("--plots_dir", type=str, required=True,
                        help="Directory containing plot PNG files")
    parser.add_argument("--summary_json", type=str, default=None,
                        help="Path to plot_summary.json (fallback if no genome_stats)")
    parser.add_argument("--output_dir", type=str, default="./categorized_pdfs",
                        help="Output directory for PDFs")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Ground truth CSV for taxonomy (optional)")
    parser.add_argument("--genome_stats", type=str, default=None,
                        help="genome_stats.csv from analyze_performance_factors.py (preferred)")
    parser.add_argument("--clustering_results", type=str, default=None,
                        help="clustering_results.json from cluster_activations.py (for P/R/MCC)")
    parser.add_argument("--fasta_dir", type=str, default=None,
                        help="Directory with FASTA files for GC calculation (optional)")
    parser.add_argument("--high_thresh", type=float, default=0.7,
                        help="Threshold for high performance (default: 0.7)")
    parser.add_argument("--low_thresh", type=float, default=0.3,
                        help="Threshold for low performance (default: 0.3)")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Create Categorized PDF Reports")
    print("=" * 60)

    # Determine which mode to use
    genome_stats = None
    summary_data = None
    use_f1 = False

    # Prefer genome_stats.csv if available (has F1, taxonomy, GC content)
    if args.genome_stats and Path(args.genome_stats).exists():
        print("\nLoading genome stats (from analyze_performance_factors.py)...")
        genome_stats = load_genome_stats(args.genome_stats)
        print(f"  Loaded stats for {len(genome_stats)} genomes")

        # Check which metrics are available (priority: recall > mcc > f1)
        has_recall = any(gs.get('recall') is not None for gs in genome_stats.values())
        has_mcc = any(gs.get('mcc') is not None for gs in genome_stats.values())
        has_f1 = any(gs.get('f1') is not None for gs in genome_stats.values())
        if has_recall:
            use_f1 = True  # This flag means "use proper metrics" not just precision
            print("  Will use RECALL for categorization (priority metric)")
        elif has_mcc:
            use_f1 = True
            print("  Will use MCC for categorization (no recall available)")
        elif has_f1:
            use_f1 = True
            print("  Will use F1 for categorization (no recall/MCC available)")
        else:
            print("  Warning: No recall/MCC/F1 scores found in genome_stats")

    # Load clustering_results.json for metrics (P/R/F1/MCC)
    if args.clustering_results and Path(args.clustering_results).exists():
        print("\nLoading clustering results (for P/R/MCC metrics)...")
        clustering_metrics = load_clustering_results(args.clustering_results)
        print(f"  Loaded metrics for {len(clustering_metrics)} genomes")

        # Merge into genome_stats
        if genome_stats is None:
            genome_stats = {}

        for assembly, metrics in clustering_metrics.items():
            if assembly not in genome_stats:
                genome_stats[assembly] = {}
            # Only update if not already set
            for key in ['f1', 'precision', 'recall', 'mcc']:
                if metrics.get(key) is not None and genome_stats[assembly].get(key) is None:
                    genome_stats[assembly][key] = metrics[key]

        # Re-check for metrics
        has_recall = any(gs.get('recall') is not None for gs in genome_stats.values())
        has_mcc = any(gs.get('mcc') is not None for gs in genome_stats.values())
        has_f1 = any(gs.get('f1') is not None for gs in genome_stats.values())
        if has_recall or has_mcc or has_f1:
            use_f1 = True

    # Load plot_summary.json as fallback or for assembly list
    summary_json_path = args.summary_json
    if not summary_json_path:
        # Try default location
        default_summary = plots_dir / "plot_summary.json"
        if default_summary.exists():
            summary_json_path = str(default_summary)

    if summary_json_path and Path(summary_json_path).exists():
        print(f"\nLoading plot summary from {summary_json_path}...")
        summary_data = load_plot_summary(summary_json_path)
        print(f"  Loaded {len(summary_data)} genome entries")

        # Check if summary_data has proper metrics (from clustering_results.json conversion)
        if summary_data:
            has_recall_in_summary = any(item.get('recall') is not None for item in summary_data)
            has_mcc_in_summary = any(item.get('mcc') is not None for item in summary_data)
            has_f1_in_summary = any(item.get('f1') is not None for item in summary_data)
            if has_recall_in_summary or has_mcc_in_summary or has_f1_in_summary:
                use_f1 = True
                if has_recall_in_summary:
                    print("  Found RECALL metrics - will use for categorization")
                elif has_mcc_in_summary:
                    print("  Found MCC metrics - will use for categorization")
                else:
                    print("  Found F1 metrics - will use for categorization")

    # Validate we have some data
    if not genome_stats and not summary_data:
        print("\nError: Need either --genome_stats or --summary_json")
        return

    # Load optional ground truth metadata
    metadata = None
    if args.ground_truth and Path(args.ground_truth).exists():
        print("\nLoading ground truth metadata...")
        metadata = load_ground_truth_metadata(args.ground_truth)
        print(f"  Loaded metadata for {len(metadata)} assemblies")

    # Calculate GC content from FASTA if available
    if args.fasta_dir and Path(args.fasta_dir).exists():
        fasta_dir = Path(args.fasta_dir)
        print("\nCalculating GC content from FASTA files...")
        if genome_stats is None:
            genome_stats = {}

        assemblies = list(genome_stats.keys()) if genome_stats else [item['assembly'] for item in summary_data]
        for assembly in assemblies:
            if assembly in genome_stats and genome_stats[assembly].get('gc_content') is not None:
                continue

            # Find FASTA file
            fasta_files = list(fasta_dir.glob(f"*{assembly}*.fna")) + \
                         list(fasta_dir.glob(f"*{assembly}*.fasta"))
            if fasta_files:
                gc = calculate_gc_from_fasta(fasta_files[0])
                if assembly not in genome_stats:
                    genome_stats[assembly] = {}
                genome_stats[assembly]['gc_content'] = gc

    # Categorize genomes
    print("\nCategorizing genomes by performance...")

    if use_f1 and genome_stats:
        # Use genome_stats (priority: recall > mcc > f1 > precision)
        categories, metric_label = categorize_genomes_from_stats(
            genome_stats,
            high_thresh=args.high_thresh,
            low_thresh=args.low_thresh
        )
        metric_label = metric_label.upper()
    else:
        # Use plot_summary.json with genome_stats overlay if available
        categories, metric_key = categorize_genomes_from_summary(
            summary_data,
            genome_stats=genome_stats,
            high_thresh=args.high_thresh,
            low_thresh=args.low_thresh
        )
        metric_label = metric_key.upper()
        if metric_key == 'precision':
            print(f"  Note: Using {metric_label} for categorization (no recall/MCC/F1 available)")

    print(f"  High ({metric_label} >= {args.high_thresh}): {len(categories['high'])} genomes")
    print(f"  Medium ({args.low_thresh} <= {metric_label} < {args.high_thresh}): {len(categories['medium'])} genomes")
    print(f"  Low ({metric_label} < {args.low_thresh}): {len(categories['low'])} genomes")

    # Create PDFs
    print("\nCreating PDF reports...")

    for category in ['high', 'medium', 'low']:
        output_path = output_dir / f"prophage_detection_{category}_performance.pdf"
        create_pdf_report(
            categories,
            category,
            plots_dir,
            output_path,
            genome_stats=genome_stats,
            use_f1=use_f1
        )

    # Create summary text file
    summary_path = output_dir / "category_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PROPHAGE DETECTION PERFORMANCE CATEGORIES\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Metric used: {metric_label}\n")
        f.write(f"High threshold: >= {args.high_thresh}\n")
        f.write(f"Low threshold: < {args.low_thresh}\n\n")

        for category in ['high', 'medium', 'low']:
            f.write(f"\n{category.upper()} PERFORMANCE ({len(categories[category])} genomes)\n")
            f.write("-" * 40 + "\n")
            for assembly, metric_val, item in categories[category]:
                taxonomy = item.get('organism', '')
                gc_val = item.get('gc_content')

                # Override with genome_stats if available
                if genome_stats and assembly in genome_stats:
                    if genome_stats[assembly].get('organism'):
                        taxonomy = genome_stats[assembly]['organism']
                    if genome_stats[assembly].get('gc_content') is not None:
                        gc_val = genome_stats[assembly]['gc_content']

                taxonomy_short = taxonomy[:35] if taxonomy else ''
                gc_str = f"GC:{gc_val:.1%}" if gc_val is not None else ''
                f.write(f"  {assembly}: {metric_label}={metric_val:.3f}  {taxonomy_short}  {gc_str}\n")

    print(f"\nSaved summary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
