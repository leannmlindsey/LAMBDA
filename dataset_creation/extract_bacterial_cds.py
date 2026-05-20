#!/usr/bin/env python3
"""
Extract non-phage CDS from Bakta-annotated bacterial genomes.

This creates a dataset of definitively bacterial coding sequences
by excluding any CDS with phage-related annotations.

Input: Bakta annotation directories (GFF3 + FNA files)
Output: CSV with CDS sequences and metadata

Usage:
    python extract_bacterial_cds.py \
        --annotations-dir /path/to/bacteria_only_annotations \
        --accessions /path/to/accessions.txt \
        --output bacterial_cds.csv \
        --min-length 500 \
        --max-length 2000
"""

import argparse
import csv
import re
from pathlib import Path
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract non-phage CDS from Bakta annotations"
    )
    parser.add_argument(
        "--annotations-dir", "-a",
        required=True,
        help="Directory containing Bakta annotation subdirectories"
    )
    parser.add_argument(
        "--accessions", "-l",
        required=True,
        help="File with list of accessions to process"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum CDS length in bp (default: 200)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum CDS length in bp (default: 8000)"
    )
    parser.add_argument(
        "--max-per-genome",
        type=int,
        default=50,
        help="Maximum CDS to extract per genome (default: 50)"
    )
    parser.add_argument(
        "--exclude-hypothetical",
        action="store_true",
        help="Exclude hypothetical proteins"
    )
    parser.add_argument(
        "--blast-results",
        help="BLAST results file to check for phage hits (optional)"
    )
    parser.add_argument(
        "--min-identity",
        type=float,
        default=90.0,
        help="Minimum identity for BLAST hit filtering (default: 90)"
    )
    parser.add_argument(
        "--min-blast-length",
        type=int,
        default=200,
        help="Minimum alignment length for BLAST hit filtering (default: 200)"
    )
    return parser.parse_args()


# Keywords that indicate phage-related annotations
PHAGE_KEYWORDS = [
    'phage', 'prophage', 'viral', 'virus', 'bacteriophage',
    'capsid', 'tail', 'baseplate', 'portal', 'terminase',
    'integrase', 'transposase', 'recombinase',
    'holin', 'lysin', 'lysozyme', 'endolysin',
    'repressor', 'antirepressor',
    'scaffold', 'head', 'collar', 'sheath',
    'tape measure', 'tapemeasure',
]


def is_phage_related(product, note=""):
    """Check if a CDS annotation suggests phage origin."""
    text = (product + " " + note).lower()

    for keyword in PHAGE_KEYWORDS:
        if keyword in text:
            return True

    return False


def load_blast_hits_by_contig(blast_file, min_identity=90.0, min_length=200):
    """Load significant BLAST hits organized by contig.

    Returns dict: {contig_id: [(start, end), ...]}

    BLAST output format 6 columns:
    0: qseqid, 1: sseqid, 2: pident, 3: length, 4: mismatch, 5: gapopen,
    6: qstart, 7: qend, 8: sstart, 9: send, 10: evalue, 11: bitscore
    """
    hits_by_contig = {}

    with open(blast_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue

            try:
                contig = parts[1]  # subject (bacterial contig)
                identity = float(parts[2])
                length = int(parts[3])
                sstart = int(parts[8])
                send = int(parts[9])
            except (ValueError, IndexError):
                continue

            # Only keep significant hits
            if identity >= min_identity and length >= min_length:
                # Normalize coordinates (start < end)
                start = min(sstart, send)
                end = max(sstart, send)

                if contig not in hits_by_contig:
                    hits_by_contig[contig] = []
                hits_by_contig[contig].append((start, end))

    return hits_by_contig


def overlaps_blast_hit(contig, cds_start, cds_end, hits_by_contig, min_overlap=50):
    """Check if a CDS overlaps with any significant BLAST hit.

    Returns True if there's an overlap of at least min_overlap bp.
    """
    if contig not in hits_by_contig:
        return False

    for hit_start, hit_end in hits_by_contig[contig]:
        # Calculate overlap
        overlap_start = max(cds_start, hit_start)
        overlap_end = min(cds_end, hit_end)
        overlap = overlap_end - overlap_start

        if overlap >= min_overlap:
            return True

    return False


def parse_gff3(gff_file):
    """Parse GFF3 file and extract CDS features."""
    cds_features = []

    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            seqid, source, feature_type, start, end, score, strand, phase, attributes = parts

            if feature_type != 'CDS':
                continue

            # Parse attributes
            attr_dict = {}
            for attr in attributes.split(';'):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    attr_dict[key] = value

            cds_features.append({
                'seqid': seqid,
                'start': int(start),
                'end': int(end),
                'strand': strand,
                'id': attr_dict.get('ID', ''),
                'product': attr_dict.get('product', attr_dict.get('Product', '')),
                'name': attr_dict.get('Name', ''),
                'note': attr_dict.get('note', attr_dict.get('Note', '')),
                'locus_tag': attr_dict.get('locus_tag', ''),
            })

    return cds_features


def extract_sequence(genome_seqs, seqid, start, end, strand):
    """Extract sequence from genome."""
    if seqid not in genome_seqs:
        return None

    seq = genome_seqs[seqid][start-1:end]  # GFF is 1-based

    if strand == '-':
        seq = seq.reverse_complement()

    return str(seq.seq)


def main():
    args = parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Extract Non-Phage Bacterial CDS")
    print("=" * 60)
    print()
    print(f"Annotations directory: {annotations_dir}")
    print(f"Length range: {args.min_length}-{args.max_length} bp")
    print(f"Max per genome: {args.max_per_genome}")
    print()

    # Load BLAST hits if provided
    hits_by_contig = {}
    if args.blast_results:
        print(f"Loading BLAST hits from {args.blast_results}...")
        print(f"  Filtering: identity ≥{args.min_identity}% AND length ≥{args.min_blast_length}bp")
        hits_by_contig = load_blast_hits_by_contig(
            args.blast_results,
            min_identity=args.min_identity,
            min_length=args.min_blast_length
        )
        total_hits = sum(len(hits) for hits in hits_by_contig.values())
        print(f"  Loaded {total_hits:,} significant hits across {len(hits_by_contig):,} contigs")
        print()

    # Load accessions
    with open(args.accessions, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(accessions)} genomes...")
    print()

    # Collect all CDS
    all_cds = []
    stats = {
        'total_cds': 0,
        'phage_related': 0,
        'hypothetical': 0,
        'too_short': 0,
        'too_long': 0,
        'blast_overlap': 0,
        'extracted': 0,
    }

    for accession in accessions:
        # Find annotation files
        acc_dir = annotations_dir / accession

        if not acc_dir.exists():
            print(f"  Warning: No annotations for {accession}")
            continue

        # Find GFF3 and FNA files
        gff_files = list(acc_dir.glob("*.gff3"))
        fna_files = list(acc_dir.glob("*.fna"))

        if not gff_files:
            print(f"  Warning: No GFF3 for {accession}")
            continue
        if not fna_files:
            print(f"  Warning: No FNA for {accession}")
            continue

        gff_file = gff_files[0]
        fna_file = fna_files[0]

        # Load genome sequences
        genome_seqs = SeqIO.to_dict(SeqIO.parse(fna_file, "fasta"))

        # Parse GFF3
        cds_features = parse_gff3(gff_file)
        stats['total_cds'] += len(cds_features)

        # Filter and extract CDS
        genome_cds = []

        for cds in cds_features:
            length = cds['end'] - cds['start'] + 1

            # Check length
            if length < args.min_length:
                stats['too_short'] += 1
                continue
            if length > args.max_length:
                stats['too_long'] += 1
                continue

            # Check if phage-related by annotation
            if is_phage_related(cds['product'], cds['note']):
                stats['phage_related'] += 1
                continue

            # Check if overlaps with BLAST hit (phage similarity)
            if hits_by_contig and overlaps_blast_hit(cds['seqid'], cds['start'], cds['end'], hits_by_contig):
                stats['blast_overlap'] += 1
                continue

            # Optionally exclude hypothetical
            if args.exclude_hypothetical:
                if 'hypothetical' in cds['product'].lower():
                    stats['hypothetical'] += 1
                    continue

            # Extract sequence
            seq = extract_sequence(genome_seqs, cds['seqid'], cds['start'], cds['end'], cds['strand'])
            if seq is None:
                continue

            cds_id = f"{accession}_{cds['locus_tag']}_{cds['start']}_{cds['end']}"

            genome_cds.append({
                'cds_id': cds_id,
                'accession': accession,
                'contig': cds['seqid'],
                'start': cds['start'],
                'end': cds['end'],
                'strand': cds['strand'],
                'length': length,
                'product': cds['product'],
                'sequence': seq,
            })

        # Limit per genome
        if len(genome_cds) > args.max_per_genome:
            # Sample evenly across the genome
            import random
            random.seed(42)
            genome_cds = random.sample(genome_cds, args.max_per_genome)

        all_cds.extend(genome_cds)
        stats['extracted'] += len(genome_cds)

        print(f"  {accession}: {len(genome_cds)} CDS extracted")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total CDS found:      {stats['total_cds']:,}")
    print(f"Phage-related:        {stats['phage_related']:,} (excluded - annotation keywords)")
    print(f"BLAST overlap:        {stats['blast_overlap']:,} (excluded - ≥90% id, ≥200bp hit)")
    print(f"Too short (<{args.min_length}bp):   {stats['too_short']:,} (excluded)")
    print(f"Too long (>{args.max_length}bp):    {stats['too_long']:,} (excluded)")
    if args.exclude_hypothetical:
        print(f"Hypothetical:         {stats['hypothetical']:,} (excluded)")
    print(f"Extracted:            {stats['extracted']:,}")
    print()

    # Write output CSV
    print(f"Writing to {args.output}...")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id', 'sequence', 'label', 'source', 'product', 'length'])

        for cds in all_cds:
            writer.writerow([
                cds['cds_id'],
                cds['sequence'],
                0,  # bacteria label
                'gtdb_cds',
                cds['product'],
                cds['length'],
            ])

    print(f"Done! Wrote {len(all_cds):,} CDS to {args.output}")


if __name__ == "__main__":
    main()
