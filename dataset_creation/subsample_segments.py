#!/usr/bin/env python3
"""
Subsample non-overlapping segments from phage genomes.

Sampling rules:
- Genomes < 5kb: excluded
- Genomes 5kb - 10kb: 1 sample
- Genomes >= 10kb: 1 sample per 10kb (floor division)

Segments are placed randomly but guaranteed non-overlapping.
"""

import argparse
import random
from pathlib import Path
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subsample non-overlapping segments from phage genomes"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir", "-i",
        help="Directory containing FASTA files (one genome per file)"
    )
    input_group.add_argument(
        "--input-fasta", "-f",
        help="Multi-FASTA file containing all genomes"
    )
    parser.add_argument(
        "--accession-list", "-a",
        required=True,
        help="File with list of accessions to process"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output FASTA file with all segments"
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=2000,
        help="Length of each segment in bp (default: 2000)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5000,
        help="Minimum genome length to include (default: 5000)"
    )
    parser.add_argument(
        "--sample-per-bp",
        type=int,
        default=10000,
        help="Take 1 sample per this many bp (default: 10000)"
    )
    parser.add_argument(
        "--total-segments",
        type=int,
        default=None,
        help="Target total segments (overrides --sample-per-bp, distributes evenly across genomes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--suffix",
        default=".fasta",
        help="Suffix for FASTA files when using --input-dir (default: .fasta)"
    )
    parser.add_argument(
        "--output-metadata",
        default=None,
        help="Output TSV with segment metadata (optional)"
    )
    return parser.parse_args()


def get_random_nonoverlapping_positions(genome_length, n_samples, segment_length, max_attempts=1000):
    """
    Get n_samples random non-overlapping start positions.

    Returns list of start positions, or fewer if couldn't fit all.
    """
    if genome_length < segment_length:
        return []

    valid_range = genome_length - segment_length
    positions = []

    for _ in range(n_samples):
        # Try to find a valid position
        for attempt in range(max_attempts):
            pos = random.randint(0, valid_range)

            # Check for overlap with existing positions
            overlaps = False
            for existing_pos in positions:
                # Two segments overlap if their ranges intersect
                # Segment 1: [pos, pos + segment_length)
                # Segment 2: [existing_pos, existing_pos + segment_length)
                if not (pos + segment_length <= existing_pos or pos >= existing_pos + segment_length):
                    overlaps = True
                    break

            if not overlaps:
                positions.append(pos)
                break
        else:
            # Couldn't find non-overlapping position after max_attempts
            # This might happen if genome is too small for requested samples
            break

    return sorted(positions)


def calculate_n_samples(genome_length, min_length, sample_per_bp):
    """Calculate number of samples to take from a genome."""
    if genome_length < min_length:
        return 0
    elif genome_length < sample_per_bp:
        return 1
    else:
        return genome_length // sample_per_bp


def load_genomes_from_multifasta(fasta_file, accession_set):
    """Load genomes from a multi-FASTA file, filtering by accession set."""
    genomes = {}
    print(f"  Loading genomes from multi-FASTA: {fasta_file}")

    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract accession from record ID (handle various formats)
        # Could be: "AB002632", "AB002632.1", "AB002632 description..."
        accession = record.id.split('.')[0].split()[0]

        if accession in accession_set:
            genomes[accession] = record

    print(f"  Loaded {len(genomes)} genomes matching accession list")
    return genomes


def main():
    args = parse_args()
    random.seed(args.seed)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load accession list
    with open(args.accession_list, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    accession_set = set(accessions)

    print(f"Subsampling segments from {len(accessions)} genomes")
    print(f"  Segment length: {args.segment_length} bp")
    print(f"  Min genome length: {args.min_length} bp")
    if args.total_segments:
        print(f"  Target total segments: {args.total_segments}")
    else:
        print(f"  Sample per: {args.sample_per_bp} bp")
    print(f"  Seed: {args.seed}")
    print()

    # Load genomes based on input type
    if args.input_fasta:
        # Multi-FASTA mode
        genomes = load_genomes_from_multifasta(args.input_fasta, accession_set)
        use_multifasta = True
    else:
        # Directory mode
        input_dir = Path(args.input_dir)
        genomes = None
        use_multifasta = False

    all_segments = []
    metadata = []

    stats = {
        'processed': 0,
        'excluded_short': 0,
        'excluded_not_found': 0,
        'total_segments': 0,
        'genomes_with_segments': 0,
    }

    # If total-segments mode, first count valid genomes
    if args.total_segments:
        print("  Counting valid genomes for total-segments mode...")
        valid_genomes = []
        for accession in accessions:
            if use_multifasta:
                if accession in genomes:
                    genome_length = len(genomes[accession].seq)
                    if genome_length >= args.min_length:
                        valid_genomes.append((accession, genome_length))
            else:
                # For directory mode, we'd need to check each file
                # For now, assume all are valid and filter during processing
                valid_genomes.append((accession, None))

        n_valid = len(valid_genomes)
        if n_valid == 0:
            print("  Error: No valid genomes found!")
            return

        # Calculate base samples per genome
        base_samples = args.total_segments // n_valid
        extra_samples = args.total_segments % n_valid

        # Assign samples to each genome (some get +1 to reach exact total)
        random.shuffle(valid_genomes)
        genome_sample_counts = {}
        for i, (acc, _) in enumerate(valid_genomes):
            genome_sample_counts[acc] = base_samples + (1 if i < extra_samples else 0)

        print(f"  Valid genomes: {n_valid}")
        print(f"  Samples per genome: {base_samples} (+ {extra_samples} genomes get 1 extra)")
        print()

    for i, accession in enumerate(accessions):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(accessions)} ({stats['total_segments']} segments)")

        try:
            if use_multifasta:
                # Get from pre-loaded dict
                if accession not in genomes:
                    stats['excluded_not_found'] += 1
                    continue
                record = genomes[accession]
            else:
                # Find FASTA file - try multiple naming patterns
                fasta_file = None
                for pattern in [
                    f"{accession}{args.suffix}",
                    f"{accession}.fa",
                    f"{accession}.fna",
                    f"{accession}_genome.fasta",
                    f"{accession}_shuffled.fasta",
                ]:
                    candidate = input_dir / pattern
                    if candidate.exists():
                        fasta_file = candidate
                        break

                if fasta_file is None:
                    stats['excluded_not_found'] += 1
                    continue

                # Read genome
                record = SeqIO.read(fasta_file, "fasta")

            genome_length = len(record.seq)

            # Calculate number of samples
            if args.total_segments:
                # Use pre-computed counts for balanced mode
                n_samples = genome_sample_counts.get(accession, 0)
                if genome_length < args.min_length:
                    n_samples = 0
            else:
                n_samples = calculate_n_samples(genome_length, args.min_length, args.sample_per_bp)

            if n_samples == 0:
                stats['excluded_short'] += 1
                continue

            # Get random non-overlapping positions
            positions = get_random_nonoverlapping_positions(
                genome_length, n_samples, args.segment_length
            )

            if not positions:
                stats['excluded_short'] += 1
                continue

            stats['processed'] += 1
            stats['genomes_with_segments'] += 1

            # Extract segments
            for seg_idx, start_pos in enumerate(positions):
                end_pos = start_pos + args.segment_length
                segment_seq = str(record.seq[start_pos:end_pos])

                # Create segment ID
                segment_id = f"{accession}_seg{seg_idx+1}_{start_pos}_{end_pos}"

                all_segments.append((segment_id, segment_seq))
                stats['total_segments'] += 1

                metadata.append({
                    'segment_id': segment_id,
                    'accession': accession,
                    'segment_num': seg_idx + 1,
                    'start': start_pos,
                    'end': end_pos,
                    'genome_length': genome_length,
                    'n_samples_from_genome': len(positions),
                })

        except Exception as e:
            print(f"  Error processing {accession}: {e}")
            continue

    # Write output FASTA
    print(f"\nWriting {len(all_segments)} segments to {output_file}")
    with open(output_file, 'w') as f:
        for segment_id, segment_seq in all_segments:
            f.write(f">{segment_id}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(segment_seq), 80):
                f.write(f"{segment_seq[i:i+80]}\n")

    # Write metadata if requested
    if args.output_metadata:
        metadata_file = Path(args.output_metadata)
        print(f"Writing metadata to {metadata_file}")
        with open(metadata_file, 'w') as f:
            headers = ['segment_id', 'accession', 'segment_num', 'start', 'end',
                       'genome_length', 'n_samples_from_genome']
            f.write('\t'.join(headers) + '\n')
            for m in metadata:
                f.write('\t'.join(str(m[h]) for h in headers) + '\n')

    # Print summary
    print(f"\nSummary:")
    print(f"  Genomes processed: {stats['processed']}")
    print(f"  Genomes excluded (too short): {stats['excluded_short']}")
    print(f"  Genomes excluded (not found): {stats['excluded_not_found']}")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Avg segments per genome: {stats['total_segments'] / max(1, stats['genomes_with_segments']):.2f}")

    # Write summary file
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Segment Subsampling Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Input accessions: {len(accessions)}\n")
        f.write(f"Segment length: {args.segment_length}\n")
        f.write(f"Min genome length: {args.min_length}\n")
        f.write(f"Sample per bp: {args.sample_per_bp}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Genomes processed: {stats['processed']}\n")
        f.write(f"Genomes excluded (short): {stats['excluded_short']}\n")
        f.write(f"Genomes excluded (not found): {stats['excluded_not_found']}\n")
        f.write(f"Total segments: {stats['total_segments']}\n")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
