#!/usr/bin/env python3
"""
Shuffle nucleotides within each segment to create GC-content control sequences.

This preserves:
- Sequence length
- GC content (same nucleotides, just reordered)
- Number of sequences

This destroys:
- Actual sequence patterns
- Any k-mer structure
- Biological signal

Input: FASTA file with segments
Output: FASTA file with shuffled segments
"""

import argparse
import random
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shuffle nucleotides within each segment for GC-content control"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input FASTA file with segments"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output FASTA file with shuffled segments"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--label-suffix",
        default="_shuffled",
        help="Suffix to add to sequence IDs (default: _shuffled)"
    )
    return parser.parse_args()


def shuffle_sequence(seq, rng):
    """Shuffle nucleotides in a sequence using provided RNG."""
    seq_list = list(str(seq))
    rng.shuffle(seq_list)
    return ''.join(seq_list)


def main():
    args = parse_args()

    # Use seeded RNG for reproducibility
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Shuffling segments for GC-content control")
    print(f"=" * 50)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Seed: {args.seed}")
    print()

    # Process sequences
    shuffled_records = []
    total_length = 0

    for record in SeqIO.parse(input_path, "fasta"):
        # Shuffle the sequence
        shuffled_seq = shuffle_sequence(record.seq, rng)

        # Create new record with shuffled suffix
        new_id = f"{record.id}{args.label_suffix}"
        shuffled_record = SeqRecord(
            Seq(shuffled_seq),
            id=new_id,
            description=f"shuffled from {record.id}"
        )
        shuffled_records.append(shuffled_record)
        total_length += len(shuffled_seq)

    # Write output
    SeqIO.write(shuffled_records, output_path, "fasta")

    print(f"Processed {len(shuffled_records)} sequences")
    print(f"Total nucleotides: {total_length:,}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
