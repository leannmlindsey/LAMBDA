#!/usr/bin/env python3
"""
Create a mapping from contig/sequence IDs to genome accessions.

Reads through all GTDB genome files and extracts the sequence IDs
from FASTA headers, mapping them to their parent genome accession.

Output: TSV file with columns: contig_id, genome_accession
"""

import argparse
import gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create contig ID to genome accession mapping"
    )
    parser.add_argument(
        "--gtdb-dir", "-g",
        required=True,
        help="GTDB base directory (containing database/GCA/... structure)"
    )
    parser.add_argument(
        "--accession-list", "-a",
        required=True,
        help="File with list of accessions to process"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output TSV file (contig_id, genome_accession)"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=16,
        help="Number of parallel threads (default: 16)"
    )
    return parser.parse_args()


def normalize_accession(accession):
    """Strip GB_/RS_ prefix from GTDB accession."""
    if accession.startswith('GB_'):
        return accession[3:]
    elif accession.startswith('RS_'):
        return accession[3:]
    return accession


def accession_to_path(gtdb_dir, accession):
    """Convert accession to file path."""
    accession = normalize_accession(accession)

    parts = accession.split('_')
    if len(parts) != 2:
        return None, None

    prefix = parts[0]  # GCA or GCF
    number_version = parts[1]
    number = number_version.split('.')[0]

    # Pad to 9 digits and split into chunks
    number = number.zfill(9)
    chunk1 = number[0:3]
    chunk2 = number[3:6]
    chunk3 = number[6:9]

    fna_path = Path(gtdb_dir) / "database" / prefix / chunk1 / chunk2 / chunk3 / f"{accession}_genomic.fna.gz"
    return fna_path, accession


def extract_contig_ids(args_tuple):
    """Extract contig IDs from a genome file."""
    accession, gtdb_dir = args_tuple

    fna_path, normalized_acc = accession_to_path(gtdb_dir, accession)

    if fna_path is None or not fna_path.exists():
        # Try alternate prefix
        if normalized_acc and normalized_acc.startswith('GCA_'):
            alt_acc = normalized_acc.replace('GCA_', 'GCF_')
        elif normalized_acc and normalized_acc.startswith('GCF_'):
            alt_acc = normalized_acc.replace('GCF_', 'GCA_')
        else:
            return (accession, [], 'invalid')

        fna_path, normalized_acc = accession_to_path(gtdb_dir, alt_acc)
        if fna_path is None or not fna_path.exists():
            return (accession, [], 'not_found')

    contig_ids = []
    try:
        with gzip.open(fna_path, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    # Extract sequence ID (first word after >)
                    header = line[1:].strip()
                    seq_id = header.split()[0]
                    contig_ids.append(seq_id)
        return (normalized_acc, contig_ids, 'success')
    except Exception as e:
        return (accession, [], f'error: {e}')


def main():
    args = parse_args()

    # Load accession list
    with open(args.accession_list, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    print(f"Creating contig ID to genome accession mapping")
    print(f"=" * 50)
    print(f"Accessions to process: {len(accessions)}")
    print(f"GTDB directory: {args.gtdb_dir}")
    print(f"Output file: {args.output}")
    print(f"Threads: {args.threads}")
    print()

    # Process genomes in parallel
    task_args = [(acc, args.gtdb_dir) for acc in accessions]

    all_mappings = []
    stats = {'success': 0, 'not_found': 0, 'error': 0}

    print(f"Processing {len(accessions)} genomes...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(extract_contig_ids, task): task[0] for task in task_args}

        completed = 0
        for future in as_completed(futures):
            completed += 1

            if completed % 1000 == 0:
                print(f"  Progress: {completed}/{len(accessions)}")

            try:
                accession, contig_ids, status = future.result()

                if status == 'success':
                    stats['success'] += 1
                    for contig_id in contig_ids:
                        all_mappings.append((contig_id, accession))
                elif status == 'not_found':
                    stats['not_found'] += 1
                else:
                    stats['error'] += 1

            except Exception as e:
                stats['error'] += 1
                print(f"  Error: {e}")

    # Write output
    print(f"\nWriting {len(all_mappings)} mappings to {args.output}")

    with open(args.output, 'w') as f:
        f.write("contig_id\tgenome_accession\n")
        for contig_id, accession in all_mappings:
            f.write(f"{contig_id}\t{accession}\n")

    # Summary
    print(f"\nSummary:")
    print(f"  Genomes processed: {stats['success']}")
    print(f"  Genomes not found: {stats['not_found']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Total contig mappings: {len(all_mappings)}")
    print(f"  Average contigs per genome: {len(all_mappings) / max(stats['success'], 1):.1f}")


if __name__ == "__main__":
    main()
