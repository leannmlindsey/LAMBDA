#!/bin/bash
# Select prophage-free bacteria for the bacteria-only benchmark
#
# These are bacteria with ZERO BLAST hits against INPHARED phages,
# making them truly prophage-free.
#
# Usage: bash select_prophage_free_bacteria.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Input files
BLAST_RESULTS="/work/hdd/bfzj/llindsey1/prophage_validation_v3/validation_blast_raw.tsv"
CONTIG_MAP="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv"
TEST_ACCESSIONS="${SCRIPT_DIR}/gtdb_dataset_filtered_v3/test_accessions.txt"
GTDB_METADATA="/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata/bac120_metadata.tsv"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/bacteria_only_prophage_free"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Select Prophage-Free Bacteria"
echo "=========================================="
echo ""

python ${SCRIPT_DIR}/find_prophage_free_bacteria.py \
    --blast-results "${BLAST_RESULTS}" \
    --contig-map "${CONTIG_MAP}" \
    --accessions "${TEST_ACCESSIONS}" \
    --metadata "${GTDB_METADATA}" \
    --output "${OUTPUT_DIR}/prophage_free_100_accessions.txt" \
    --output-metadata "${OUTPUT_DIR}/prophage_free_100_metadata.tsv" \
    --n-select 100 \
    --balance-taxonomy

echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
