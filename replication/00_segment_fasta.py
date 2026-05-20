#!/usr/bin/env python3
"""
00_segment_fasta.py

Slide a window across each contig in one or more bacterial-genome FASTA files
and write segment CSVs in the LAMBDA segment schema, so that the genome-wide
inference + aggregation pipeline (steps 02 + 03) can run on user data.

This script is the entry point for **use case 3** in the replication README:
"use the pretrained gLMs to predict prophage locations in an independent
dataset provided by the user." Skip this script if you already have segment
CSVs in the LAMBDA format (e.g., from the Zenodo download).

Output layout (matching the convention 02_submit_inference_jobs.sh expects):

    DATASET_ROOT/
    └── genome_wide_segments_2k_1k/      ← one subdir per window size
        ├── GCF_000007845.1.csv          ← one CSV per input FASTA
        ├── GCF_000008805.1.csv
        └── ...

Each output CSV has the columns:

    segment_id,seq_id,start,end,sequence,label
    GCF_000007845.1_seg_00001,NC_007795.1,0,2000,ACGT...,0
    ...

* `start` / `end` are 0-based half-open coordinates relative to the contig.
* `label` is always 0; if you have ground-truth prophage annotations you can
  overwrite this column afterwards (the inference scripts ignore it).

Usage:
    # Single FASTA, all three LAMBDA window sizes
    python 00_segment_fasta.py \\
        --fasta /path/to/genome.fna \\
        --output-root /path/to/DATASET_ROOT \\
        --windows 2k 4k 8k

    # Whole directory of FASTAs (one per assembly)
    python 00_segment_fasta.py \\
        --fasta-dir /path/to/fasta_dir \\
        --output-root /path/to/DATASET_ROOT \\
        --windows 2k 4k 8k

The stride is set to half the window size by default (matching the LAMBDA
genome-wide split: 2k/1k, 4k/2k, 8k/4k). Override with --stride if needed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

WINDOW_BP = {"2k": 2000, "4k": 4000, "8k": 8000}
DEFAULT_STRIDE = {"2k": 1000, "4k": 2000, "8k": 4000}
SUBDIR_FOR = {
    "2k": "genome_wide_segments_2k_1k",
    "4k": "genome_wide_segments_4k_2k",
    "8k": "genome_wide_segments_8k_4k",
}


def parse_fasta(path: Path):
    """Yield (seq_id, sequence) tuples from a FASTA file. Uppercases the sequence."""
    seq_id = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks).upper()
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if seq_id is not None:
        yield seq_id, "".join(chunks).upper()


def assembly_from_path(fasta_path: Path) -> str:
    """Derive an assembly accession from the FASTA filename, stripping common suffixes."""
    name = fasta_path.stem
    # Strip a trailing _genomic / _genome if present
    for suffix in ("_genomic", "_genome"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def segment_fasta(fasta_path: Path, output_root: Path, windows: list[str],
                  strides: dict[str, int], min_segment_length: int) -> None:
    """Slide windows across every contig in `fasta_path` and write one CSV per window."""
    assembly = assembly_from_path(fasta_path)
    print(f"[{assembly}] reading {fasta_path}")

    contigs = list(parse_fasta(fasta_path))
    if not contigs:
        print(f"  [warn] no sequences in {fasta_path}, skipping")
        return

    for window in windows:
        win_bp = WINDOW_BP[window]
        stride = strides[window]
        subdir = output_root / SUBDIR_FOR[window]
        subdir.mkdir(parents=True, exist_ok=True)
        out_csv = subdir / f"{assembly}.csv"

        n_rows = 0
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["segment_id", "seq_id", "start", "end", "sequence", "label"])

            seg_idx = 0
            for seq_id, seq in contigs:
                pos = 0
                while pos < len(seq):
                    end = min(pos + win_bp, len(seq))
                    if end - pos < min_segment_length:
                        break
                    seg_idx += 1
                    seg_id = f"{assembly}_seg_{seg_idx:05d}"
                    writer.writerow([seg_id, seq_id, pos, end, seq[pos:end], 0])
                    n_rows += 1
                    if end == len(seq):
                        break
                    pos += stride

        print(f"  wrote {out_csv} ({n_rows} segments, window={window}, stride={stride})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Slide windows across bacterial FASTA files to produce LAMBDA-format segment CSVs."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="A single FASTA file")
    src.add_argument("--fasta-dir", help="Directory containing one FASTA per assembly")
    parser.add_argument("--output-root", required=True,
                        help="Where to write the per-window subdirectories (set DATASET_ROOT to this in paths.env)")
    parser.add_argument("--windows", nargs="+", default=["2k", "4k", "8k"],
                        choices=list(WINDOW_BP.keys()),
                        help="Window sizes to generate (default: 2k 4k 8k)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override the stride (default: half the window size)")
    parser.add_argument("--min-segment-length", type=int, default=500,
                        help="Drop trailing segments smaller than this many bp (default 500)")
    parser.add_argument("--fasta-glob", default="*.fna",
                        help="Glob pattern for --fasta-dir (default: *.fna; tries *.fasta and *.fa as fallback)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.stride is not None:
        strides = {w: args.stride for w in args.windows}
    else:
        strides = {w: DEFAULT_STRIDE[w] for w in args.windows}

    if args.fasta:
        fastas = [Path(args.fasta)]
    else:
        fasta_dir = Path(args.fasta_dir)
        fastas = sorted(fasta_dir.glob(args.fasta_glob))
        if not fastas:
            for fallback in ("*.fasta", "*.fa", "*.fna.gz"):
                fastas = sorted(fasta_dir.glob(fallback))
                if fastas:
                    break
        if not fastas:
            sys.exit(f"ERROR: no FASTAs found in {fasta_dir} (tried *.fna, *.fasta, *.fa)")

    for fasta_path in fastas:
        if not fasta_path.exists():
            print(f"[skip] {fasta_path} does not exist", file=sys.stderr)
            continue
        segment_fasta(fasta_path, output_root, args.windows, strides,
                      args.min_segment_length)

    print(f"\nDone. {len(fastas)} FASTA(s) segmented into {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
