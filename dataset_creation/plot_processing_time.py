#!/usr/bin/env python3
"""
Create a bar graph showing mean per genome processing time by tool.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create bar graph of processing times by model"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to CSV file with Model and Seconds columns"
    )
    parser.add_argument(
        "--output", "-o",
        default="processing_time.png",
        help="Path to output image file (default: processing_time.png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image (default: 150)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the CSV file
    df = pd.read_csv(args.input)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get tab10 colors
    colors = plt.cm.tab10.colors

    # Create bar positions
    x_positions = range(len(df))

    # Create bars with tab10 colors
    bars = ax.bar(x_positions, df['Seconds'], color=[colors[i % len(colors)] for i in x_positions])

    # Set title and labels
    ax.set_title("Mean Per Genome Processing Time by Tool", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean per Genome Inference Duration (seconds)", fontsize=12)

    # Set x-axis labels (diagonal)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")

    # Optionally display
    # plt.show()


if __name__ == "__main__":
    main()
