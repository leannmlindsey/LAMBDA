#!/usr/bin/env python3
"""
Merge INPHARED (phage) and GTDB (bacteria) segment datasets and shuffle.

Creates a combined dataset with labels for binary classification.
"""

import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge and shuffle phage/bacteria segment datasets"
    )
    parser.add_argument(
        "--phage-fasta", "-p",
        required=True,
        help="FASTA file with phage segments (INPHARED)"
    )
    parser.add_argument(
        "--bacteria-fasta", "-b",
        required=True,
        help="FASTA file with bacteria segments (GTDB)"
    )
    parser.add_argument(
        "--output-fasta", "-o",
        required=True,
        help="Output merged and shuffled FASTA file"
    )
    parser.add_argument(
        "--output-labels", "-l",
        required=True,
        help="Output TSV file with labels (segment_id, label, source)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--phage-label",
        default="phage",
        help="Label for phage sequences (default: phage)"
    )
    parser.add_argument(
        "--bacteria-label",
        default="bacteria",
        help="Label for bacteria sequences (default: bacteria)"
    )
    return parser.parse_args()


def read_fasta(fasta_file):
    """Read FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))

    return sequences


def write_fasta(sequences, output_file):
    """Write sequences to FASTA file with 80-char line wrapping."""
    with open(output_file, 'w') as f:
        for header, seq in sequences:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    output_fasta = Path(args.output_fasta)
    output_labels = Path(args.output_labels)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    print(f"Merge and Shuffle Dataset")
    print(f"=" * 50)
    print(f"Phage FASTA: {args.phage_fasta}")
    print(f"Bacteria FASTA: {args.bacteria_fasta}")
    print(f"Seed: {args.seed}")
    print()

    # Read phage sequences
    print(f"Reading phage sequences...")
    phage_seqs = read_fasta(args.phage_fasta)
    print(f"  Loaded {len(phage_seqs):,} phage segments")

    # Read bacteria sequences
    print(f"Reading bacteria sequences...")
    bacteria_seqs = read_fasta(args.bacteria_fasta)
    print(f"  Loaded {len(bacteria_seqs):,} bacteria segments")

    # Create combined list with labels
    # Each entry: (header, sequence, label, source)
    combined = []
    for header, seq in phage_seqs:
        combined.append((header, seq, args.phage_label, "inphared"))
    for header, seq in bacteria_seqs:
        combined.append((header, seq, args.bacteria_label, "gtdb"))

    print(f"\nTotal combined: {len(combined):,} segments")

    # Shuffle
    print(f"Shuffling with seed {args.seed}...")
    random.shuffle(combined)

    # Write output FASTA
    print(f"\nWriting merged FASTA to {output_fasta}...")
    sequences_only = [(header, seq) for header, seq, _, _ in combined]
    write_fasta(sequences_only, output_fasta)

    # Write labels file
    print(f"Writing labels to {output_labels}...")
    with open(output_labels, 'w') as f:
        f.write("segment_id\tlabel\tsource\n")
        for header, _, label, source in combined:
            # Extract just the ID part (before any space)
            segment_id = header.split()[0]
            f.write(f"{segment_id}\t{label}\t{source}\n")

    # Print summary
    print(f"\nSummary:")
    print(f"  Phage segments: {len(phage_seqs):,} ({args.phage_label})")
    print(f"  Bacteria segments: {len(bacteria_seqs):,} ({args.bacteria_label})")
    print(f"  Total segments: {len(combined):,}")
    print(f"  Balance: {len(phage_seqs)/len(combined)*100:.1f}% phage, {len(bacteria_seqs)/len(combined)*100:.1f}% bacteria")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
