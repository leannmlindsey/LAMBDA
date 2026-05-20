#!/bin/bash
# Extract non-phage CDS from Bakta-annotated bacterial genomes
#
# This creates a dataset of definitively bacterial coding sequences
# for testing which bacterial genes get misclassified as phage.
#
# Usage: bash extract_bacterial_cds.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Input
ANNOTATIONS_DIR="${SCRIPT_DIR}/bacteria_only_annotations"
ACCESSIONS="${SCRIPT_DIR}/bacteria_only_dataset/bacteria_only_100_accessions.txt"

# BLAST results for cross-checking (exclude CDS overlapping phage hits)
BLAST_RESULTS="/work/hdd/bfzj/llindsey1/prophage_validation_v3/validation_blast_raw.tsv"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/bacteria_cds_benchmark"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Extract Bacterial CDS"
echo "=========================================="
echo ""

# Check if annotations exist
if [ ! -d "${ANNOTATIONS_DIR}" ]; then
    echo "Error: Annotations directory not found: ${ANNOTATIONS_DIR}"
    echo "Run run_bakta_bacteria.slurm first to annotate the genomes."
    exit 1
fi

# Extract CDS at different length ranges to match segment sizes
echo "=== Extracting CDS (2k range: 500-2000bp) ==="
python ${SCRIPT_DIR}/extract_bacterial_cds.py \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --accessions "${ACCESSIONS}" \
    --output "${OUTPUT_DIR}/bacterial_cds_2k.csv" \
    --min-length 500 \
    --max-length 2000 \
    --max-per-genome 50 \
    --blast-results "${BLAST_RESULTS}" \
    --min-identity 90 \
    --min-blast-length 200

echo ""
echo "=== Extracting CDS (4k range: 1000-4000bp) ==="
python ${SCRIPT_DIR}/extract_bacterial_cds.py \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --accessions "${ACCESSIONS}" \
    --output "${OUTPUT_DIR}/bacterial_cds_4k.csv" \
    --min-length 1000 \
    --max-length 4000 \
    --max-per-genome 30 \
    --blast-results "${BLAST_RESULTS}" \
    --min-identity 90 \
    --min-blast-length 200

echo ""
echo "=== Extracting CDS (8k range: 2000-8000bp) ==="
python ${SCRIPT_DIR}/extract_bacterial_cds.py \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --accessions "${ACCESSIONS}" \
    --output "${OUTPUT_DIR}/bacterial_cds_8k.csv" \
    --min-length 2000 \
    --max-length 8000 \
    --max-per-genome 20 \
    --blast-results "${BLAST_RESULTS}" \
    --min-identity 90 \
    --min-blast-length 200

echo ""
echo "=========================================="
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.csv

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
