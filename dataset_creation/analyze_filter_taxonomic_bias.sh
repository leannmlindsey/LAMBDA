#!/bin/bash
#SBATCH --job-name=lambda_filter_bias
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --output=lambda_filter_bias_%j.out
#SBATCH --error=lambda_filter_bias_%j.err
#
# Run the taxonomic-bias check for the prophage contamination filter
# (analyze_filter_taxonomic_bias.py).
#
# Reads the contaminated_accessions.txt and the per-split accession lists
# produced by filter_and_subsample_gtdb_v3.slurm, looks up GTDB taxonomy
# from bac120_metadata.tsv, and reports whether the exclusion was
# taxonomically biased at the phylum level.
#
# Usage:
#   sbatch dataset_creation/analyze_filter_taxonomic_bias.sh

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment (tries modern `conda activate` then falls back
# to legacy `source activate`)
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true

# ─── Configuration ──────────────────────────────────────────────────────────
INPUTS_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA_DATASET_CONSTRUCTION/outputs"

CONTAMINATED="${INPUTS_DIR}/contaminated_accessions.txt"
TRAIN_ACC="${INPUTS_DIR}/train_accessions.txt"
DEV_ACC="${INPUTS_DIR}/dev_accessions.txt"
TEST_ACC="${INPUTS_DIR}/test_accessions.txt"
GTDB_METADATA="${INPUTS_DIR}/bac120_metadata.tsv"

SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/dataset_creation"
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/LAMBDA/results/filter_taxonomic_bias"

# ─── Validation ─────────────────────────────────────────────────────────────
for f in "${CONTAMINATED}" "${TRAIN_ACC}" "${DEV_ACC}" "${TEST_ACC}" "${GTDB_METADATA}" \
         "${SCRIPT_DIR}/analyze_filter_taxonomic_bias.py"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file missing: ${f}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "LAMBDA filter taxonomic bias analysis"
echo "============================================================"
echo "Contaminated list: ${CONTAMINATED}"
echo "Retained lists:    ${TRAIN_ACC}, ${DEV_ACC}, ${TEST_ACC}"
echo "GTDB metadata:     ${GTDB_METADATA}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "============================================================"

python3 "${SCRIPT_DIR}/analyze_filter_taxonomic_bias.py" \
    --contaminated "${CONTAMINATED}" \
    --retained-files "${TRAIN_ACC}" "${DEV_ACC}" "${TEST_ACC}" \
    --metadata "${GTDB_METADATA}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done. Inspect:"
echo "  ${OUTPUT_DIR}/filter_taxonomic_bias.csv"
echo "  ${OUTPUT_DIR}/filter_taxonomic_bias.png"
echo "  ${OUTPUT_DIR}/filter_taxonomic_bias_summary.txt"
