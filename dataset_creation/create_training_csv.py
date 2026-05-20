#!/usr/bin/env python3
"""
Create training-ready CSV files from merged FASTA and labels.

Combines the merged FASTA sequences with labels to create CSV files
suitable for training genomic language models.

Output columns:
- segment_id: The sequence identifier
- sequence: The actual nucleotide sequence
- label: 0 for bacteria, 1 for phage
- source: 'gtdb' or 'inphared'

Usage:
    python create_training_csv.py \
        --fasta train_merged.fasta \
        --labels train_labels.tsv \
        --output train.csv
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training CSV from merged FASTA and labels"
    )
    parser.add_argument(
        "--fasta", "-f",
        required=True,
        help="Merged FASTA file with sequences"
    )
    parser.add_argument(
        "--labels", "-l",
        required=True,
        help="Labels TSV file (segment_id, label, source)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file"
    )
    parser.add_argument(
        "--phage-label",
        type=int,
        default=1,
        help="Numeric label for phage (default: 1)"
    )
    parser.add_argument(
        "--bacteria-label",
        type=int,
        default=0,
        help="Numeric label for bacteria (default: 0)"
    )
    return parser.parse_args()


def read_fasta(fasta_file):
    """Read FASTA file and return dict of {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                # Extract ID (first word after >)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def read_labels(labels_file):
    """Read labels TSV and return dict of {segment_id: (label, source)}."""
    labels = {}

    with open(labels_file, 'r') as f:
        # Skip header
        header = f.readline()

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                segment_id = parts[0]
                label = parts[1]
                source = parts[2]
                labels[segment_id] = (label, source)

    return labels


def main():
    args = parse_args()

    print(f"Creating training CSV...")
    print(f"  FASTA: {args.fasta}")
    print(f"  Labels: {args.labels}")
    print(f"  Output: {args.output}")
    print()

    # Read sequences
    print("Reading sequences...")
    sequences = read_fasta(args.fasta)
    print(f"  Loaded {len(sequences):,} sequences")

    # Read labels
    print("Reading labels...")
    labels = read_labels(args.labels)
    print(f"  Loaded {len(labels):,} labels")

    # Map text labels to numeric
    label_map = {
        'phage': args.phage_label,
        'bacteria': args.bacteria_label
    }

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    print(f"Writing CSV to {args.output}...")

    phage_count = 0
    bacteria_count = 0
    missing_seq = 0
    missing_label = 0

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id', 'sequence', 'label', 'source'])

        # Use labels as the source of truth for order
        for segment_id, (text_label, source) in labels.items():
            if segment_id not in sequences:
                missing_seq += 1
                continue

            sequence = sequences[segment_id]
            numeric_label = label_map.get(text_label)

            if numeric_label is None:
                print(f"  Warning: Unknown label '{text_label}' for {segment_id}")
                continue

            writer.writerow([segment_id, sequence, numeric_label, source])

            if text_label == 'phage':
                phage_count += 1
            else:
                bacteria_count += 1

    # Check for sequences without labels
    for seq_id in sequences:
        if seq_id not in labels:
            missing_label += 1

    print()
    print("Summary:")
    print(f"  Phage sequences:    {phage_count:,} (label={args.phage_label})")
    print(f"  Bacteria sequences: {bacteria_count:,} (label={args.bacteria_label})")
    print(f"  Total written:      {phage_count + bacteria_count:,}")

    if missing_seq > 0:
        print(f"  WARNING: {missing_seq} labels had no matching sequence")
    if missing_label > 0:
        print(f"  WARNING: {missing_label} sequences had no matching label")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
