#!/usr/bin/env python3
"""
Subsample non-overlapping segments from GTDB bacterial genomes.

Handles GTDB's nested directory structure and gzipped files:
  /database/GCA/XXX/XXX/XXX/GCA_XXXXXXXXX.1_genomic.fna.gz

Uses threading to process multiple genomes in parallel while managing
disk space by unzipping files temporarily.
"""

import argparse
import gzip
import random
import tempfile
import os
from pathlib import Path
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subsample segments from GTDB genomes (handles gzipped nested files)"
    )
    parser.add_argument(
        "--gtdb-dir", "-g",
        required=True,
        help="Base directory for GTDB genomes (containing database/GCA/... structure)"
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
        help="Target total segments (overrides --sample-per-bp)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-metadata",
        default=None,
        help="Output TSV with segment metadata (optional)"
    )
    return parser.parse_args()


def normalize_accession(accession):
    """
    Normalize GTDB accession to standard format (GCA_XXXXXXXXX.Y or GCF_XXXXXXXXX.Y).

    GTDB metadata often has prefixes like:
    - GB_GCA_000005845.2 -> GCA_000005845.2
    - RS_GCF_000005845.2 -> GCF_000005845.2
    """
    # Strip common GTDB prefixes
    if accession.startswith('GB_'):
        accession = accession[3:]
    elif accession.startswith('RS_'):
        accession = accession[3:]
    return accession


def accession_to_path(gtdb_dir, accession):
    """
    Convert GTDB accession to file path.

    Example: GCA_964231105.1 -> /gtdb_dir/database/GCA/964/231/105/GCA_964231105.1_genomic.fna.gz
    Also handles: GB_GCA_964231105.1 -> same path
    """
    # Normalize accession (strip GB_/RS_ prefix if present)
    accession = normalize_accession(accession)

    # Parse accession: GCA_XXXXXXXXX.Y or GCF_XXXXXXXXX.Y
    parts = accession.split('_')
    if len(parts) != 2:
        return None, None

    prefix = parts[0]  # GCA or GCF
    number_version = parts[1]  # XXXXXXXXX.Y
    number = number_version.split('.')[0]  # XXXXXXXXX

    # Split number into 3-digit chunks: 964231105 -> 964, 231, 105
    # Pad to 9 digits if needed
    number = number.zfill(9)
    chunk1 = number[0:3]
    chunk2 = number[3:6]
    chunk3 = number[6:9]

    # Build path
    fna_path = Path(gtdb_dir) / "database" / prefix / chunk1 / chunk2 / chunk3 / f"{accession}_genomic.fna.gz"

    return fna_path, accession


def get_random_nonoverlapping_positions(genome_length, n_samples, segment_length, rng, max_attempts=1000):
    """Get n_samples random non-overlapping start positions."""
    if genome_length < segment_length:
        return []

    valid_range = genome_length - segment_length
    positions = []

    for _ in range(n_samples):
        for attempt in range(max_attempts):
            pos = rng.randint(0, valid_range)
            overlaps = False
            for existing_pos in positions:
                if not (pos + segment_length <= existing_pos or pos >= existing_pos + segment_length):
                    overlaps = True
                    break
            if not overlaps:
                positions.append(pos)
                break
        else:
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


def process_genome(args_tuple):
    """
    Process a single genome: decompress, extract segments, return results.

    args_tuple: (accession, gtdb_dir, segment_length, min_length, n_samples_override, seed)

    Returns: (accession, segments_list, metadata_list, status)
    """
    accession, gtdb_dir, segment_length, min_length, sample_per_bp, n_samples_override, seed = args_tuple

    # Create a random generator with a seed derived from the global seed + accession
    # This ensures reproducibility while allowing parallel processing
    rng = random.Random(seed + hash(accession))

    segments = []
    metadata = []

    # Find the file
    fna_path, normalized_acc = accession_to_path(gtdb_dir, accession)

    if fna_path is None:
        return (accession, [], [], 'invalid_accession')

    if not fna_path.exists():
        # Try alternative: might be GCF instead of GCA or vice versa
        if normalized_acc.startswith('GCA_'):
            alt_accession = normalized_acc.replace('GCA_', 'GCF_')
        else:
            alt_accession = normalized_acc.replace('GCF_', 'GCA_')
        fna_path, normalized_acc = accession_to_path(gtdb_dir, alt_accession)
        if fna_path is None or not fna_path.exists():
            return (accession, [], [], 'not_found')

    try:
        # Read directly from gzipped file (no need to decompress to disk)
        with gzip.open(fna_path, 'rt') as f:
            # GTDB files may have multiple contigs; concatenate them
            sequences = list(SeqIO.parse(f, "fasta"))

            if not sequences:
                return (accession, [], [], 'empty_file')

            # Use the first/longest sequence or concatenate all
            # For bacterial genomes, usually want the main chromosome
            # Sort by length and use the longest
            sequences.sort(key=lambda x: len(x.seq), reverse=True)
            record = sequences[0]
            genome_length = len(record.seq)

        # Calculate number of samples
        if n_samples_override is not None:
            n_samples = n_samples_override
            if genome_length < min_length:
                n_samples = 0
        else:
            n_samples = calculate_n_samples(genome_length, min_length, sample_per_bp)

        if n_samples == 0:
            return (accession, [], [], 'too_short')

        # Get random non-overlapping positions
        positions = get_random_nonoverlapping_positions(
            genome_length, n_samples, segment_length, rng
        )

        if not positions:
            return (accession, [], [], 'no_valid_positions')

        # Extract segments
        for seg_idx, start_pos in enumerate(positions):
            end_pos = start_pos + segment_length
            segment_seq = str(record.seq[start_pos:end_pos])
            segment_id = f"{accession}_seg{seg_idx+1}_{start_pos}_{end_pos}"

            segments.append((segment_id, segment_seq))
            metadata.append({
                'segment_id': segment_id,
                'accession': accession,
                'segment_num': seg_idx + 1,
                'start': start_pos,
                'end': end_pos,
                'genome_length': genome_length,
                'n_samples_from_genome': len(positions),
            })

        return (accession, segments, metadata, 'success')

    except Exception as e:
        return (accession, [], [], f'error: {str(e)}')


def main():
    args = parse_args()
    random.seed(args.seed)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load accession list
    with open(args.accession_list, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    print(f"GTDB Segment Subsampling")
    print(f"=" * 50)
    print(f"  Accessions to process: {len(accessions)}")
    print(f"  GTDB directory: {args.gtdb_dir}")
    print(f"  Segment length: {args.segment_length} bp")
    print(f"  Min genome length: {args.min_length} bp")
    if args.total_segments:
        print(f"  Target total segments: {args.total_segments}")
    else:
        print(f"  Sample per: {args.sample_per_bp} bp")
    print(f"  Threads: {args.threads}")
    print(f"  Seed: {args.seed}")
    print()

    # Show sample path resolution for debugging
    print("Sample path resolution (first 3 accessions):")
    for acc in accessions[:3]:
        fna_path, norm_acc = accession_to_path(args.gtdb_dir, acc)
        exists = fna_path.exists() if fna_path else False
        print(f"  {acc} -> {norm_acc}")
        print(f"    Path: {fna_path}")
        print(f"    Exists: {exists}")
    print()

    # Prepare arguments for each genome
    n_samples_override = None
    if args.total_segments:
        # For total-segments mode, distribute evenly
        base_samples = args.total_segments // len(accessions)
        extra_samples = args.total_segments % len(accessions)

        # Shuffle to randomize which genomes get extra samples
        shuffled_accessions = accessions.copy()
        random.shuffle(shuffled_accessions)

        genome_sample_counts = {}
        for i, acc in enumerate(shuffled_accessions):
            genome_sample_counts[acc] = base_samples + (1 if i < extra_samples else 0)

        print(f"  Total-segments mode: {base_samples} samples per genome (+{extra_samples} extra)")

        task_args = [
            (acc, args.gtdb_dir, args.segment_length, args.min_length,
             args.sample_per_bp, genome_sample_counts.get(acc), args.seed)
            for acc in accessions
        ]
    else:
        task_args = [
            (acc, args.gtdb_dir, args.segment_length, args.min_length,
             args.sample_per_bp, None, args.seed)
            for acc in accessions
        ]

    # Process genomes in parallel
    all_segments = []
    all_metadata = []
    stats = {
        'success': 0,
        'not_found': 0,
        'too_short': 0,
        'invalid_accession': 0,
        'error': 0,
        'empty_file': 0,
        'no_valid_positions': 0,
    }

    print(f"Processing {len(accessions)} genomes with {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_genome, task): task[0] for task in task_args}

        completed = 0
        for future in as_completed(futures):
            accession = futures[future]
            completed += 1

            if completed % 100 == 0:
                print(f"  Progress: {completed}/{len(accessions)} ({len(all_segments)} segments)")

            try:
                acc, segments, metadata, status = future.result()

                if status == 'success':
                    stats['success'] += 1
                    all_segments.extend(segments)
                    all_metadata.extend(metadata)
                elif status == 'not_found':
                    stats['not_found'] += 1
                elif status == 'too_short':
                    stats['too_short'] += 1
                elif status == 'invalid_accession':
                    stats['invalid_accession'] += 1
                elif status == 'empty_file':
                    stats['empty_file'] += 1
                elif status == 'no_valid_positions':
                    stats['no_valid_positions'] += 1
                else:
                    stats['error'] += 1
                    if 'error' in status:
                        print(f"  Warning: {acc} - {status}")

            except Exception as e:
                stats['error'] += 1
                print(f"  Error processing {accession}: {e}")

    # Write output FASTA
    print(f"\nWriting {len(all_segments)} segments to {output_file}")
    with open(output_file, 'w') as f:
        for segment_id, segment_seq in all_segments:
            f.write(f">{segment_id}\n")
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
            for m in all_metadata:
                f.write('\t'.join(str(m[h]) for h in headers) + '\n')

    # Print summary
    print(f"\nSummary:")
    print(f"  Genomes processed successfully: {stats['success']}")
    print(f"  Genomes not found: {stats['not_found']}")
    print(f"  Genomes too short: {stats['too_short']}")
    print(f"  Invalid accessions: {stats['invalid_accession']}")
    print(f"  Empty files: {stats['empty_file']}")
    print(f"  No valid positions: {stats['no_valid_positions']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Total segments: {len(all_segments)}")
    if stats['success'] > 0:
        print(f"  Avg segments per genome: {len(all_segments) / stats['success']:.2f}")

    # Write summary file
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GTDB Segment Subsampling Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Input accessions: {len(accessions)}\n")
        f.write(f"Segment length: {args.segment_length}\n")
        f.write(f"Min genome length: {args.min_length}\n")
        f.write(f"Sample per bp: {args.sample_per_bp}\n")
        f.write(f"Threads: {args.threads}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"\n")
        f.write(f"Genomes processed: {stats['success']}\n")
        f.write(f"Genomes not found: {stats['not_found']}\n")
        f.write(f"Genomes too short: {stats['too_short']}\n")
        f.write(f"Errors: {stats['error']}\n")
        f.write(f"Total segments: {len(all_segments)}\n")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
