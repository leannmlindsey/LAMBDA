#!/usr/bin/env python3
"""
Annotate the high_similarity_examples.tsv produced by analyze_fragment_leakage.py
with the original segment_id and (optionally) the source genome accession and
coordinates for each query and train hit.

Steps:
  1. The leakage analysis BLAST used synthetic fragment IDs of the form
     "p_<row_index>" and "b_<row_index>" where row_index is the position of
     the segment in the merged train/dev/test CSV (one such CSV per window).
     This script reverses that mapping to recover the original segment_id.
  2. If --metadata-root is supplied, segment_ids are further mapped to
     (accession, start, end) using the per-split metadata TSVs produced by
     subsample_segments.py and subsample_gtdb_segments.py
     (gtdb_dataset_filtered_v3/<split>_segments.tsv and the phage equivalent).
  3. Output is a TSV with the same rows as high_similarity_examples.tsv plus
     new columns: query_segment_id, query_accession, query_start, query_end,
                  hit_segment_id,   hit_accession,   hit_start,   hit_end

Usage:
  python annotate_leakage_examples.py \
      --examples /path/to/high_similarity_examples.tsv \
      --csv-root /gpfs/.../lambda_final/merged_datasets_filtered \
      [--phage-metadata-root /path/to/inphared_metadata] \
      [--bacteria-metadata-root /path/to/gtdb_metadata] \
      --output /path/to/high_similarity_examples_annotated.tsv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True,
                   help="high_similarity_examples.tsv from analyze_fragment_leakage.py")
    p.add_argument("--csv-root", required=True,
                   help="Directory containing <window>/{train,dev,test}.csv merged splits")
    p.add_argument("--phage-metadata-root", default=None,
                   help="Directory containing phage <split>_segments.tsv files "
                        "(if omitted, only segment_id is added for phage queries)")
    p.add_argument("--bacteria-metadata-root", default=None,
                   help="Directory containing bacterial <split>_segments.tsv files")
    p.add_argument("--output", required=True,
                   help="Output annotated TSV")
    return p.parse_args()


def load_row_index_to_segment_id(csv_path, target_label):
    """Read a merged CSV and return a dict mapping row_index -> segment_id
    for rows whose label matches target_label (1 = phage, 0 = bacteria).

    The row_index here matches the enumerate position used in
    analyze_fragment_leakage.sh's csv_to_class_fasta step.
    """
    mapping = {}
    csv.field_size_limit(sys.maxsize)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq = row.get("sequence", "").strip().upper()
            if not seq:
                continue
            label = int(row["label"])
            if label != target_label:
                continue
            seg_id = row.get("segment_id", "")
            mapping[i] = seg_id
    return mapping


def load_segment_metadata(metadata_dir, splits=("train", "dev", "test")):
    """Load all per-split segment metadata TSVs and return a dict
    segment_id -> {accession, start, end}.
    """
    out = {}
    if metadata_dir is None:
        return out
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.is_dir():
        print(f"  WARNING: metadata dir not found: {metadata_dir}", file=sys.stderr)
        return out
    for split in splits:
        meta_path = metadata_dir / f"{split}_segments.tsv"
        if not meta_path.exists():
            continue
        df = pd.read_csv(meta_path, sep="\t")
        for _, row in df.iterrows():
            seg_id = row["segment_id"]
            out[seg_id] = {
                "accession": row.get("accession", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
            }
    return out


def resolve_id(synthetic_id, window, csv_root, row_to_segid_cache):
    """Map an ID like 'p_123' or 'b_456' (window-specific) to its segment_id."""
    if not synthetic_id or "_" not in synthetic_id:
        return None
    prefix, idx_str = synthetic_id.split("_", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    target_label = 1 if prefix == "p" else 0

    # ID space is window-specific (the FASTAs were rebuilt per window). But
    # we don't know which split the synthetic ID came from at this point —
    # only that it was the train split for the database side, and dev/test
    # for the query side. The high_similarity_examples.tsv tells us the
    # split for each row, so we look up the right CSV.
    return None  # caller fills this in using row_to_segid_cache[split]


def main():
    args = parse_args()
    csv_root = Path(args.csv_root)

    examples = pd.read_csv(args.examples, sep="\t")
    if len(examples) == 0:
        print("WARNING: examples file is empty; nothing to annotate.", file=sys.stderr)
        examples.to_csv(args.output, sep="\t", index=False)
        return

    # Build caches: (window, split, label) -> row_index -> segment_id
    # The query side is always dev/test (per the analysis); the hit side is
    # always train. So we need both query and train caches per window.
    row_cache = {}
    for window in examples["window"].unique():
        win_csv_dir = csv_root / str(window)
        for split in ("train", "dev", "test"):
            csv_path = win_csv_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"  WARNING: missing CSV {csv_path}", file=sys.stderr)
                continue
            for label, lbl_name in [(1, "phage"), (0, "bacteria")]:
                key = (str(window), split, lbl_name)
                if key in row_cache:
                    continue
                row_cache[key] = load_row_index_to_segment_id(csv_path, label)

    # Build segment_id -> metadata caches
    phage_meta = load_segment_metadata(args.phage_metadata_root)
    bacteria_meta = load_segment_metadata(args.bacteria_metadata_root)

    # Per-row annotation
    def annotate_id(synthetic_id, window, split, cls):
        seg_id = ""
        accession, start, end = "", "", ""
        if not synthetic_id or "_" not in synthetic_id:
            return seg_id, accession, start, end
        prefix, idx_str = synthetic_id.split("_", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            return seg_id, accession, start, end

        key = (str(window), split, cls)
        if key in row_cache:
            seg_id = row_cache[key].get(idx, "")

        if seg_id:
            meta = phage_meta if cls == "phage" else bacteria_meta
            entry = meta.get(seg_id, {})
            accession = entry.get("accession", "")
            start = entry.get("start", "")
            end = entry.get("end", "")
        return seg_id, accession, start, end

    new_rows = []
    for _, r in examples.iterrows():
        cls = r["class"]
        window = r["window"]
        q_split = r["split"]
        q_seg, q_acc, q_start, q_end = annotate_id(r["query"], window, q_split, cls)
        # The hit always comes from the train split
        h_seg, h_acc, h_start, h_end = annotate_id(r["train_hit"], window, "train", cls)
        new_rows.append({
            **r.to_dict(),
            "query_segment_id": q_seg,
            "query_accession":  q_acc,
            "query_start":      q_start,
            "query_end":        q_end,
            "hit_segment_id":   h_seg,
            "hit_accession":    h_acc,
            "hit_start":        h_start,
            "hit_end":          h_end,
        })

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(args.output, sep="\t", index=False)

    # Report coverage
    n = len(out_df)
    n_q_seg = (out_df["query_segment_id"] != "").sum()
    n_q_acc = (out_df["query_accession"] != "").sum()
    n_h_seg = (out_df["hit_segment_id"] != "").sum()
    n_h_acc = (out_df["hit_accession"] != "").sum()
    print(f"\nAnnotated {n} examples")
    print(f"  query segment_id resolved: {n_q_seg}/{n}")
    print(f"  query accession resolved:  {n_q_acc}/{n}")
    print(f"  hit segment_id resolved:   {n_h_seg}/{n}")
    print(f"  hit accession resolved:    {n_h_acc}/{n}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
