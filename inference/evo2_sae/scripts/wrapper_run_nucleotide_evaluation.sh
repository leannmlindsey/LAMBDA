#!/bin/bash

# Wrapper script for running nucleotide-level prophage evaluation
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_nucleotide_evaluation.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input/Output ===
# Output CSV from sae_inference.py (must have seq_id, start, end columns)
export INPUT_CSV="/path/to/sae_inference_results.csv"

# Directory containing per-segment .npy activation files (from --save_activations)
export ACTIVATIONS_DIR="/path/to/activations_dir"

# Output directory for evaluation results
export OUTPUT_DIR="./nucleotide_eval_results"

# === OPTIONAL: Feature Selection ===
export FEATURE_IDX="19746"

# === OPTIONAL: Thresholding ===
export THRESHOLD="0.5"
# Set to "true" to use per-genome adaptive threshold
export ADAPTIVE_THRESHOLD="false"
export ADAPTIVE_METHOD="mad"        # mad, std, percentile
export ADAPTIVE_K="3.0"

# === OPTIONAL: Normalization ===
# Options: none, zscore, robust_zscore, percentile, local_baseline, minmax, quantile
export NORMALIZATION="none"
export NORM_WINDOW="10000"

# === OPTIONAL: Region Calling ===
export MAX_GAP="100"
export MERGE_DISTANCE="3000"
export MIN_REGION_SIZE="1000"

# === OPTIONAL: Output ===
export OUTPUT_PREFIX="nucleotide_eval"
# Set to "true" to generate per-genome activation track plots
export PLOT="false"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_CSV}" == "/path/to/sae_inference_results.csv" ]; then
    echo "ERROR: Please set INPUT_CSV to your actual results CSV"
    exit 1
fi

if [ "${ACTIVATIONS_DIR}" == "/path/to/activations_dir" ]; then
    echo "ERROR: Please set ACTIVATIONS_DIR to your actual activations directory"
    exit 1
fi

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV does not exist: ${INPUT_CSV}"
    exit 1
fi

if [ ! -d "${ACTIVATIONS_DIR}" ]; then
    echo "ERROR: ACTIVATIONS_DIR does not exist: ${ACTIVATIONS_DIR}"
    exit 1
fi

echo "=========================================="
echo "Submitting Nucleotide Evaluation Job"
echo "=========================================="
echo "Input CSV:       ${INPUT_CSV}"
echo "Activations dir: ${ACTIVATIONS_DIR}"
echo "Output dir:      ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  Feature index:   ${FEATURE_IDX}"
echo "  Threshold:       ${THRESHOLD}"
echo "  Adaptive:        ${ADAPTIVE_THRESHOLD}"
echo "  Normalization:   ${NORMALIZATION}"
echo "  Max gap:         ${MAX_GAP} bp"
echo "  Merge distance:  ${MERGE_DISTANCE} bp"
echo "  Min region size: ${MIN_REGION_SIZE} bp"
echo "  Plot:            ${PLOT}"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Submit job
echo "Submitting job..."
sbatch --export=ALL \
    --job-name="nuc_eval" \
    "${SCRIPT_DIR}/run_nucleotide_evaluation.sh"

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
