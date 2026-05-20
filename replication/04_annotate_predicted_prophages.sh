#!/bin/bash
#
# 04_annotate_predicted_prophages.sh  (OPTIONAL)
#
# Extracts the predicted prophage regions from the website JSON files
# produced by 03_build_website_data.py, runs Pharokka annotation on them
# (https://github.com/gbouras13/pharokka), and produces a CSV of PHROG
# category annotations suitable for re-feeding into 03_build_website_data.py
# via its --phrog argument.
#
# This step is **NOT** required to reproduce the LAMBDA paper results —
# PHROG annotations for the LAMBDA test genomes are already included in
# the Zenodo deposit. Use this script if you are running the LAMBDA
# pipeline on your own data and want the dashboard to display PHROG
# functional categories alongside the predicted prophages.
#
# Pipeline:
#   1. For each {assembly}_{window}.json in WEBSITE_DATA_DIR, extract the
#      union of predicted prophage regions across all models.
#   2. Pull the corresponding sequence ranges out of the per-assembly FASTA.
#   3. Run Pharokka on each extracted FASTA.
#   4. Parse Pharokka's PHROG annotations into a single annotations.csv with
#      columns: seq_id,start,end,phrog_db_category,phrog_product.
#   5. Re-run 03_build_website_data.py with --phrog pointing at the new CSV
#      to fold the annotations into the JSONs.
#
# Prerequisites:
#   - Pharokka installed (`mamba install -c bioconda pharokka`) and its
#     database downloaded (`pharokka_db_download.py -o <dbdir>`).
#   - One FASTA per assembly in FASTA_DIR (named `<assembly>.fna` or
#     `<assembly>.fasta`).
#
# Usage:
#   bash 04_annotate_predicted_prophages.sh \\
#       --website-data /path/to/RESULTS_ROOT/website_data \\
#       --fasta-dir /path/to/per_assembly_fasta \\
#       --pharokka-db /path/to/pharokka_db \\
#       --output-dir /path/to/RESULTS_ROOT/pharokka

set -euo pipefail

WEBSITE_DATA_DIR=""
FASTA_DIR=""
PHAROKKA_DB=""
OUTPUT_DIR=""
THREADS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --website-data)  WEBSITE_DATA_DIR="$2"; shift 2 ;;
        --fasta-dir)     FASTA_DIR="$2";        shift 2 ;;
        --pharokka-db)   PHAROKKA_DB="$2";      shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";       shift 2 ;;
        --threads)       THREADS="$2";          shift 2 ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | head -n -1
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

: "${WEBSITE_DATA_DIR:?--website-data is required}"
: "${FASTA_DIR:?--fasta-dir is required}"
: "${PHAROKKA_DB:?--pharokka-db is required}"
: "${OUTPUT_DIR:?--output-dir is required}"

command -v pharokka.py >/dev/null 2>&1 || {
    echo "ERROR: pharokka.py not on PATH. Install with: mamba install -c bioconda pharokka"
    exit 1
}
command -v samtools >/dev/null 2>&1 || {
    echo "ERROR: samtools not on PATH. Install with: mamba install -c bioconda samtools"
    exit 1
}

mkdir -p "${OUTPUT_DIR}/regions" "${OUTPUT_DIR}/pharokka_out"

# ── Step 1: extract union of predicted regions from each JSON ───────────────
python3 - "${WEBSITE_DATA_DIR}" "${OUTPUT_DIR}/regions" <<'PY'
import json
import sys
from pathlib import Path

website_data_dir = Path(sys.argv[1])
regions_dir = Path(sys.argv[2])
regions_dir.mkdir(parents=True, exist_ok=True)

bed_rows = []
for json_path in sorted(website_data_dir.glob("*.json")):
    if json_path.name == "index.json":
        continue
    data = json.loads(json_path.read_text())
    assembly = data["assembly"]
    # Union all model predictions for this assembly
    regions = []
    for label, preds in data.get("clustered_predictions", {}).items():
        for r in preds:
            regions.append((int(r["start"]), int(r["end"])))
    if not regions:
        continue
    # Merge overlapping regions across models
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    for i, (s, e) in enumerate(merged):
        bed_rows.append((assembly, s, e, f"{assembly}_predicted_{i+1}"))

bed_path = regions_dir / "predicted_regions.bed"
with bed_path.open("w") as f:
    for asm, s, e, name in bed_rows:
        f.write(f"{asm}\t{s}\t{e}\t{name}\n")

print(f"  wrote {bed_path} ({len(bed_rows)} regions)")
PY

# ── Step 2: extract sequence for each region ────────────────────────────────
echo "[step 2] extracting region sequences from FASTAs..."
REGIONS_BED="${OUTPUT_DIR}/regions/predicted_regions.bed"
REGIONS_FASTA="${OUTPUT_DIR}/regions/predicted_regions.fasta"
> "${REGIONS_FASTA}"

while IFS=$'\t' read -r asm start end name; do
    # Find the FASTA for this assembly (supports .fna and .fasta)
    fasta=""
    for ext in fna fasta fa; do
        candidate="${FASTA_DIR}/${asm}.${ext}"
        if [ -f "${candidate}" ]; then
            fasta="${candidate}"
            break
        fi
    done
    if [ -z "${fasta}" ]; then
        echo "  [warn] no FASTA found for ${asm}, skipping"
        continue
    fi
    # samtools faidx (creates .fai on first use)
    [ -f "${fasta}.fai" ] || samtools faidx "${fasta}"
    # samtools regions are 1-based inclusive; BED is 0-based half-open
    region_1based="$((start + 1))-${end}"
    # Use the first sequence name from the FASTA index
    seq_id=$(head -1 "${fasta}.fai" | cut -f 1)
    {
        echo ">${name}"
        samtools faidx "${fasta}" "${seq_id}:${region_1based}" | tail -n +2
    } >> "${REGIONS_FASTA}"
done < "${REGIONS_BED}"

echo "  wrote ${REGIONS_FASTA}"

# ── Step 3: run Pharokka ───────────────────────────────────────────────────
echo "[step 3] running Pharokka..."
pharokka.py \
    -i "${REGIONS_FASTA}" \
    -o "${OUTPUT_DIR}/pharokka_out" \
    -d "${PHAROKKA_DB}" \
    -t "${THREADS}" \
    -m \
    -f

# ── Step 4: parse Pharokka output → annotations.csv ────────────────────────
echo "[step 4] parsing Pharokka annotations..."
python3 - "${OUTPUT_DIR}/pharokka_out" "${OUTPUT_DIR}/annotations.csv" <<'PY'
import csv
import sys
from pathlib import Path

pharokka_out = Path(sys.argv[1])
out_csv = Path(sys.argv[2])

# Pharokka writes per-CDS annotations to pharokka_cds_functions.tsv
funcs_tsv = pharokka_out / "pharokka_cds_functions.tsv"
gff = pharokka_out / "pharokka.gff"

if not funcs_tsv.exists() or not gff.exists():
    sys.exit(f"ERROR: expected Pharokka outputs not found in {pharokka_out}")

# Build a map: gene_id -> (start, end, seq_id) from the GFF
gene_coords = {}
with gff.open() as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip().split("\t")
        if len(parts) < 9 or parts[2] != "CDS":
            continue
        seq_id, start, end, attrs = parts[0], int(parts[3]), int(parts[4]), parts[8]
        for kv in attrs.split(";"):
            if kv.startswith("ID="):
                gene_coords[kv[3:]] = (seq_id, start, end)
                break

with funcs_tsv.open() as f_in, out_csv.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in, delimiter="\t")
    writer = csv.DictWriter(
        f_out,
        fieldnames=["seq_id", "start", "end", "phrog_db_category", "phrog_product"],
    )
    writer.writeheader()
    for row in reader:
        gene_id = row.get("Gene") or row.get("gene_id")
        if gene_id not in gene_coords:
            continue
        seq_id, start, end = gene_coords[gene_id]
        writer.writerow({
            "seq_id": seq_id,
            "start": start,
            "end": end,
            "phrog_db_category": row.get("category") or row.get("Category", ""),
            "phrog_product": row.get("annot") or row.get("Annotation") or row.get("Product", ""),
        })

print(f"  wrote {out_csv}")
PY

echo
echo "============================================================"
echo "Done. Re-run 03_build_website_data.py with:"
echo "  --phrog ${OUTPUT_DIR}/annotations.csv"
echo "to fold the new annotations into the website JSON files."
echo "============================================================"
