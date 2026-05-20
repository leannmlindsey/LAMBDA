#!/bin/bash

# Wrapper script for running NT-v2 inference on all CSV files in a directory
#
# This script loads the model ONCE and processes all files sequentially,
# which is much faster than running separate jobs for each file.
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_inference_dir.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Directory ===
# Directory containing CSV files with 'sequence' column
INPUT_DIR="/home/lindseylm/lindseylm/lambda_final/phoenix/segments/2k_1k"

# === REQUIRED: Output Directory ===
# All predictions will be saved here as {basename}_predictions.csv
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/results/inference/lambda_dir_test"

# === REQUIRED: Model Configuration ===
# Path to fine-tuned model directory
MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/output/filtered/2k/nt_lambda_filtered_2k_8_3e-5_20260120_063339/checkpoint-40995"

# === OPTIONAL: Inference Parameters ===
BATCH_SIZE="16"
MAX_LENGTH="2048"
THRESHOLD="0.5"

# === OPTIONAL: File Pattern ===
# Glob pattern for input files (default: *.csv)
PATTERN="*.csv"

# === OPTIONAL: Precision ===
# Set to "fp16" for faster inference, "bf16" for A100, or "fp32" for default
PRECISION="fp16"

# === OPTIONAL: Save Metrics ===
# Set to "true" to save metrics JSON for each file (requires labels in CSV)
SAVE_METRICS="true"

#####################################################################
# END CONFIGURATION
#####################################################################

# Script directory and project root
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/slurm_scripts"
PROJECT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification"

# Validate configuration
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build precision flag
PRECISION_FLAG=""
if [ "${PRECISION}" == "fp16" ]; then
    PRECISION_FLAG="--fp16"
elif [ "${PRECISION}" == "bf16" ]; then
    PRECISION_FLAG="--bf16"
fi

# Build metrics flag
METRICS_FLAG=""
if [ "${SAVE_METRICS}" == "true" ]; then
    METRICS_FLAG="--save_metrics"
fi

echo "=========================================="
echo "NT-v2 Directory Inference"
echo "=========================================="
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold: ${THRESHOLD}"
echo "  Precision: ${PRECISION}"
echo "  Pattern: ${PATTERN}"
echo "  Save metrics: ${SAVE_METRICS}"
echo "=========================================="
echo ""

# Count files to process
NUM_FILES=$(ls -1 ${INPUT_DIR}/${PATTERN} 2>/dev/null | wc -l)
echo "Files to process: ${NUM_FILES}"
echo ""

# Run inference
cd "${PROJECT_DIR}"

python inference_nt_dir.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --threshold ${THRESHOLD} \
    --pattern "${PATTERN}" \
    ${PRECISION_FLAG} \
    ${METRICS_FLAG}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Inference completed successfully"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "Summary file: ${OUTPUT_DIR}/summary.json"
else
    echo "Inference failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
