#!/bin/bash

# Wrapper script for running NT-v2 inference on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_inference.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,INPUT_CSV=/path/to/test.csv,MODEL_PATH=/path/to/model run_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input CSV ===
# Path to CSV file with 'sequence' column (and optionally 'label')
export INPUT_CSV="/path/to/your/test.csv"

# === REQUIRED: Model Configuration ===
# Path to fine-tuned model directory
export MODEL_PATH="/path/to/finetuned/model"

# === OPTIONAL: Output CSV ===
# Leave empty to use default: input_csv with _predictions suffix
export OUTPUT_CSV=""

# === OPTIONAL: Inference Parameters ===
export BATCH_SIZE="16"
export MAX_LENGTH="2048"           # NT-v2 supports up to 2048 tokens
export THRESHOLD="0.5"             # Classification threshold for prob_1

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_CSV}" == "/path/to/your/test.csv" ]; then
    echo "ERROR: Please set INPUT_CSV to your actual input file"
    exit 1
fi

if [ "${MODEL_PATH}" == "/path/to/finetuned/model" ]; then
    echo "ERROR: Please set MODEL_PATH to your fine-tuned model directory"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: Model directory does not contain config.json: ${MODEL_PATH}"
    exit 1
fi

# Get input name for job naming
INPUT_NAME=$(basename "${INPUT_CSV}" .csv)

echo "=========================================="
echo "Submitting NT-v2 Inference Job"
echo "=========================================="
echo "Input: ${INPUT_CSV}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_CSV:-<auto>}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold: ${THRESHOLD}"
echo "=========================================="

# Get script directory
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/slurm_scripts"

# Submit job
echo "Submitting job..."
sbatch --export=ALL \
    --job-name="nt_inf_${INPUT_NAME}" \
    "${SCRIPT_DIR}/run_inference.sh"

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
