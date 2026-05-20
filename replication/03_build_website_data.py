#!/usr/bin/env python3
"""
03_build_website_data.py

Aggregate per-model inference outputs from RESULTS_ROOT into two products:

  (1) `metrics_summary.csv` — one row per (model, category, window) with all
      classification metrics computed across the corresponding LAMBDA split.
      Covers the binary_segments and error_and_bias categories.

  (2) `website_data/{assembly}_{window}.json` + `website_data/index.json` —
      per-genome JSON files for the genome-wide category, in the schema
      consumed by the LAMBDA visualization dashboard
      (https://github.com/leannmlindsey/lambda-benchmark).

Discovery is automatic — the script walks RESULTS_ROOT looking for the layout
produced by 02_submit_inference_jobs.sh:

    RESULTS_ROOT/
        <model>/
            <category>/                 # binary | error_bias | genome_wide
                <window>/               # 2k | 4k | 8k
                    *_predictions.csv

Each prediction CSV is expected to have these columns:
    segment_id (or seq_id)   — identifier of the segment
    sequence                 — DNA sequence (optional, dropped from the JSON)
    label                    — ground truth (0/1), optional
    prob_0, prob_1           — class probabilities
    pred_label               — predicted label (0/1)
    start, end               — segment coordinates (genome_wide only)

The taxonomy file can be either of:
    (a) Simple CSV (recommended for your-own-data):
            assembly,phylum,class,order,family,genus,species
    (b) GTDB-Tk output (`gtdbtk.bac120.summary.tsv`) — the paper's format

The script auto-detects which based on the file extension and column headers.

Usage:
    python 03_build_website_data.py \\
        --predictions /path/to/RESULTS_ROOT \\
        --ground-truth /path/to/lambda_ground_truth.csv \\
        --taxonomy /path/to/taxonomy.csv \\
        --output /path/to/RESULTS_ROOT/website_data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


# ─── Display names ──────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "dnabert2": "DNABERT2",
    "nucleotide_transformer_v2": "NT-v2",
    "caduceus": "Caduceus",
    "prokbert": "ProkBERT",
    "megadna": "megaDNA",
    "generanno": "GENERanno",
    "evo2_sae": "Evo2+SAE",
}


# ─── Utility ─────────────────────────────────────────────────────────────────

def extract_assembly(name: str) -> str:
    """Pull a GCA/GCF/NC assembly accession out of a longer filename or ID."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] in ("GCA", "GCF", "NC"):
        return f"{parts[0]}_{parts[1]}"
    return name


def round_float(val, decimals=3):
    if val is None or val == "":
        return None
    try:
        f = float(val)
        if f != f:
            return None
        return round(f, decimals)
    except (ValueError, TypeError):
        return None


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_ground_truth(path: Path):
    """Load ground-truth prophage regions.

    Returns:
        gt_by_assembly: dict[assembly] -> list of {start, end}
        organisms: dict[assembly] -> organism name
    """
    gt_by_assembly = defaultdict(list)
    organisms: dict[str, str] = {}

    if not path.exists():
        print(f"  [warn] ground truth file not found: {path}", file=sys.stderr)
        return {}, {}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            asm = row.get("Assembly") or row.get("assembly")
            if not asm:
                continue
            try:
                start = int(row.get("start") or row.get("Start"))
                end = int(row.get("end") or row.get("End"))
            except (TypeError, ValueError):
                continue
            gt_by_assembly[asm].append({"start": start, "end": end})
            organism = row.get("Organism Name") or row.get("organism")
            if organism:
                organisms[asm] = organism

    for asm in gt_by_assembly:
        gt_by_assembly[asm].sort(key=lambda r: r["start"])

    return dict(gt_by_assembly), organisms


def load_taxonomy(path: Path):
    """Load taxonomy. Auto-detect simple CSV vs GTDB-Tk format."""
    if not path or not path.exists():
        print(f"  [warn] taxonomy file not found: {path}", file=sys.stderr)
        return {}

    is_tsv = path.suffix.lower() in (".tsv", ".tab")
    delim = "\t" if is_tsv else ","

    taxonomy: dict[str, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = {c.lower() for c in (reader.fieldnames or [])}
        gtdbtk_format = "user_genome" in cols and "classification" in cols

        for row in reader:
            row_lc = {k.lower(): v for k, v in row.items()}
            if gtdbtk_format:
                asm = extract_assembly(row_lc.get("user_genome", ""))
                classification = row_lc.get("classification", "")
                tax = {}
                for part in classification.split(";"):
                    part = part.strip()
                    if part.startswith("p__"):
                        tax["phylum"] = part[3:]
                    elif part.startswith("c__"):
                        tax["class"] = part[3:]
                    elif part.startswith("o__"):
                        tax["order"] = part[3:]
                    elif part.startswith("f__"):
                        tax["family"] = part[3:]
                    elif part.startswith("g__"):
                        tax["genus"] = part[3:]
                    elif part.startswith("s__"):
                        tax["species"] = part[3:]
                if asm:
                    taxonomy[asm] = tax
            else:
                asm = row_lc.get("assembly") or row_lc.get("assembly_id") or ""
                if not asm:
                    continue
                taxonomy[asm] = {
                    rank: row_lc.get(rank, "") for rank in
                    ("phylum", "class", "order", "family", "genus", "species")
                }

    return taxonomy


def load_checkv(path: Optional[Path]):
    """Optional: load CheckV quality summary. Returns dict[assembly] -> [records]."""
    if not path:
        return {}
    if not path.exists():
        print(f"  [warn] CheckV file not found: {path}", file=sys.stderr)
        return {}

    quality = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            contig_id = row.get("contig_id", "")
            asm = extract_assembly(contig_id)
            quality[asm].append({
                "contig_id": contig_id,
                "checkv_quality": row.get("checkv_quality", ""),
                "completeness": round_float(row.get("completeness")),
                "viral_genes": int(row.get("viral_genes", 0) or 0),
            })
    return dict(quality)


def load_phrog(path: Optional[Path]):
    """Optional: load PHROG annotations. Returns dict[assembly] -> [features]."""
    if not path:
        return {}
    if not path.exists():
        print(f"  [warn] PHROG file not found: {path}", file=sys.stderr)
        return {}

    by_asm = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            asm = extract_assembly(row.get("seq_id", ""))
            try:
                start = int(row.get("start", 0))
                end = int(row.get("end", 0))
            except (TypeError, ValueError):
                continue
            by_asm[asm].append({
                "seq_id": row.get("seq_id", ""),
                "start": start,
                "end": end,
                "category": (row.get("phrog_db_category") or "unknown function").strip(),
                "product": row.get("phrog_product", ""),
            })
    return dict(by_asm)


# ─── Discovery ───────────────────────────────────────────────────────────────

def discover_predictions(results_root: Path):
    """Walk RESULTS_ROOT and return a list of (model, category, window, csv_path)."""
    found = []
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        for cat_dir in sorted(model_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for win_dir in sorted(cat_dir.iterdir()):
                if not win_dir.is_dir():
                    continue
                for csv_path in sorted(win_dir.glob("*_predictions.csv")):
                    found.append((model_dir.name, cat_dir.name, win_dir.name, csv_path))
    return found


# ─── Metrics ────────────────────────────────────────────────────────────────

METRICS_FIELDS = [
    "model", "category", "window", "input_file", "n",
    "accuracy", "precision", "recall", "f1", "mcc",
    "sensitivity", "specificity", "auc", "tp", "tn", "fp", "fn",
]


def compute_metrics(csv_path: Path) -> Optional[dict]:
    """Compute classification metrics from a single predictions CSV."""
    labels, preds, probs = [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label" not in row or row["label"] in ("", None):
                return None
            try:
                labels.append(int(float(row["label"])))
                preds.append(int(float(row["pred_label"])))
                probs.append(float(row.get("prob_1", "nan")))
            except (TypeError, ValueError):
                continue

    if not labels:
        return None

    metrics = {
        "n": len(labels),
        "accuracy": round_float(accuracy_score(labels, preds)),
        "precision": round_float(precision_score(labels, preds, zero_division=0)),
        "recall": round_float(recall_score(labels, preds, zero_division=0)),
        "f1": round_float(f1_score(labels, preds, zero_division=0)),
        "mcc": round_float(matthews_corrcoef(labels, preds)),
    }

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics.update({
        "sensitivity": round_float(tp / (tp + fn) if (tp + fn) else 0),
        "specificity": round_float(tn / (tn + fp) if (tn + fp) else 0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    })

    try:
        if any(not (p != p) for p in probs):
            metrics["auc"] = round_float(roc_auc_score(labels, probs))
        else:
            metrics["auc"] = None
    except ValueError:
        metrics["auc"] = None

    return metrics


def write_metrics_summary(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in METRICS_FIELDS})


# ─── Genome-wide JSON assembly ──────────────────────────────────────────────

def load_genome_wide_csv(csv_path: Path):
    """Load a genome-wide prediction CSV into a list of per-segment dicts.

    Returns list of {start, end, prob_1, pred_label, label}.
    """
    segs = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row.get("start", 0))
                end = int(row.get("end", 0))
            except (TypeError, ValueError):
                continue
            segs.append({
                "start": start,
                "end": end,
                "prob_1": round_float(row.get("prob_1")),
                "pred_label": int(float(row["pred_label"])) if row.get("pred_label") else None,
                "label": int(float(row["label"])) if row.get("label") not in (None, "") else None,
            })
    return segs


def cluster_segments(segments: list[dict], threshold: float = 0.5,
                     merge_distance: int = 5000, min_region_size: int = 1000) -> list[dict]:
    """Convert per-segment predictions into discrete predicted prophage regions.

    Simple algorithm: mark segments with prob_1 > threshold as positive, merge
    adjacent/nearby positive segments (gap <= merge_distance), drop regions
    smaller than min_region_size.
    """
    positives = [s for s in segments if s.get("prob_1") is not None and s["prob_1"] > threshold]
    positives.sort(key=lambda s: s["start"])

    if not positives:
        return []

    regions = []
    cur_start, cur_end = positives[0]["start"], positives[0]["end"]
    for s in positives[1:]:
        if s["start"] - cur_end <= merge_distance:
            cur_end = max(cur_end, s["end"])
        else:
            if cur_end - cur_start >= min_region_size:
                regions.append({"start": cur_start, "end": cur_end})
            cur_start, cur_end = s["start"], s["end"]
    if cur_end - cur_start >= min_region_size:
        regions.append({"start": cur_start, "end": cur_end})

    return regions


def build_genome_jsons(predictions: list, ground_truth: dict, organisms: dict,
                       taxonomy: dict, checkv: dict, phrog: dict,
                       output_dir: Path) -> list[dict]:
    """Build per-genome JSON files for the genome_wide category.

    Returns the list of (assembly, window) pairs that were written, for the
    index.json manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by (assembly, window) — each genome × window gets one JSON.
    by_asm_win = defaultdict(lambda: {"per_segment": {}, "clustered": {}, "metrics": {}})

    for model, category, window, csv_path in predictions:
        if category != "genome_wide":
            continue
        assembly = extract_assembly(csv_path.stem.replace("_predictions", ""))
        segs = load_genome_wide_csv(csv_path)
        if not segs:
            continue
        label = f"{MODEL_DISPLAY.get(model, model)} {window}"
        bucket = by_asm_win[(assembly, window)]
        bucket["per_segment"][label] = segs
        bucket["clustered"][label] = cluster_segments(segs)
        m = compute_metrics(csv_path)
        if m:
            bucket["metrics"][label] = {
                "mcc": m["mcc"], "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "auc": m["auc"],
            }

    manifest = []
    for (assembly, window), bucket in sorted(by_asm_win.items()):
        if not bucket["per_segment"]:
            continue
        genome_length = max(
            (seg["end"] for segs in bucket["per_segment"].values() for seg in segs),
            default=0,
        )
        data = {
            "assembly": assembly,
            "organism": organisms.get(assembly, ""),
            "genome_length": genome_length,
            "window_size": window,
            "taxonomy": taxonomy.get(assembly, {}),
            "ground_truth": ground_truth.get(assembly, []),
            "metrics": bucket["metrics"],
            "clustered_predictions": bucket["clustered"],
            "per_segment": bucket["per_segment"],
            "phrog_annotations": phrog.get(assembly, []),
            "checkv_quality": checkv.get(assembly, []),
            "model_categories": {label: "genomic_lm" for label in bucket["per_segment"]},
        }
        json_path = output_dir / f"{assembly}_{window}.json"
        with json_path.open("w") as f:
            json.dump(data, f, separators=(",", ":"))
        manifest.append({
            "assembly": assembly,
            "window_size": window,
            "organism": data["organism"],
            "file": json_path.name,
        })

    with (output_dir / "index.json").open("w") as f:
        json.dump({"genomes": manifest}, f, indent=2)

    return manifest


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LAMBDA inference outputs into the website JSON format."
    )
    parser.add_argument("--predictions", required=True,
                        help="RESULTS_ROOT — directory containing {model}/{category}/{window}/ structure")
    parser.add_argument("--ground-truth", default=None,
                        help="CSV with ground-truth prophage regions (Assembly,start,end,Organism Name)")
    parser.add_argument("--taxonomy", default=None,
                        help="Per-assembly taxonomy CSV (simple format or GTDB-Tk)")
    parser.add_argument("--checkv", default=None,
                        help="Optional CheckV quality_summary.tsv")
    parser.add_argument("--phrog", default=None,
                        help="Optional PHROG annotation CSV")
    parser.add_argument("--output", required=True,
                        help="Output directory (will contain metrics_summary.csv and website_data/)")
    parser.add_argument("--cluster-threshold", type=float, default=0.5,
                        help="prob_1 threshold for marking a segment positive (default 0.5)")
    parser.add_argument("--merge-distance", type=int, default=5000,
                        help="Merge adjacent positive segments within this distance in bp (default 5000)")
    parser.add_argument("--min-region-size", type=int, default=1000,
                        help="Drop predicted regions smaller than this in bp (default 1000)")
    args = parser.parse_args()

    results_root = Path(args.predictions)
    output_root = Path(args.output)
    if not results_root.exists():
        sys.exit(f"ERROR: predictions directory not found: {results_root}")

    print(f"Walking {results_root}...")
    predictions = discover_predictions(results_root)
    print(f"Found {len(predictions)} prediction CSV files across "
          f"{len({p[0] for p in predictions})} models")

    # ── Metrics summary across all categories ────────────────────────────────
    print("\nComputing metrics summary...")
    records = []
    for model, category, window, csv_path in predictions:
        m = compute_metrics(csv_path)
        if m is None:
            continue
        records.append({
            "model": model, "category": category, "window": window,
            "input_file": csv_path.name, **m,
        })
    metrics_path = output_root / "metrics_summary.csv"
    write_metrics_summary(records, metrics_path)
    print(f"  wrote {metrics_path} ({len(records)} rows)")

    # ── Genome-wide per-genome JSONs ─────────────────────────────────────────
    if any(p[1] == "genome_wide" for p in predictions):
        print("\nBuilding per-genome JSON files for genome_wide category...")
        ground_truth, organisms = ({}, {})
        if args.ground_truth:
            ground_truth, organisms = load_ground_truth(Path(args.ground_truth))
        taxonomy = load_taxonomy(Path(args.taxonomy)) if args.taxonomy else {}
        checkv = load_checkv(Path(args.checkv)) if args.checkv else {}
        phrog = load_phrog(Path(args.phrog)) if args.phrog else {}

        website_dir = output_root / "website_data"
        manifest = build_genome_jsons(predictions, ground_truth, organisms,
                                      taxonomy, checkv, phrog, website_dir)
        print(f"  wrote {len(manifest)} per-genome JSON files to {website_dir}")
    else:
        print("\nNo genome_wide predictions found — skipping website JSON build.")

    print("\nDone.")


if __name__ == "__main__":
    main()
