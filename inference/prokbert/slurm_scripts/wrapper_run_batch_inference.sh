#!/bin/bash

# Wrapper script for running batch inference with ProkBERT on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_batch_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Example contents of input_files.txt:
#   /path/to/dataset1.csv
#   /path/to/dataset2.csv
#   /path/to/dataset3.csv
# Each CSV file must have a 'sequence' column (and optionally 'label')
INPUT_LIST="/path/to/input_files.txt"

# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/path/to/output_directory"

# === REQUIRED: Model Configuration ===
# Path to fine-tuned ProkBERT model checkpoint directory
# This should contain pytorch_model.bin and config.json
# Example: /data/user/finetuning_outputs/lambda_finetuned
MODEL_PATH="/path/to/finetuned/checkpoint"

# === OPTIONAL: Inference Parameters ===
# Batch size for inference (default: 32)
BATCH_SIZE="32"

# Maximum sequence length (default: 1024 for prokbert-mini)
MAX_LENGTH="1024"

# Base model the checkpoint was fine-tuned from
# Options: neuralbioinfo/prokbert-mini, neuralbioinfo/prokbert-mini-long, neuralbioinfo/prokbert-mini-c
BASE_MODEL="neuralbioinfo/prokbert-mini"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ]; then
    echo "ERROR: Please set INPUT_LIST to your input files list"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ]; then
    echo "ERROR: Please set OUTPUT_DIR to your output directory"
    exit 1
fi

if [ "${MODEL_PATH}" == "/path/to/finetuned/checkpoint" ]; then
    echo "ERROR: Please set MODEL_PATH to your fine-tuned model checkpoint directory"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Check if MODEL_PATH is a valid directory with model files
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/pytorch_model.bin" ] && [ ! -f "${MODEL_PATH}/model.safetensors" ]; then
    echo "ERROR: Model checkpoint not found in: ${MODEL_PATH}"
    echo "Expected pytorch_model.bin or model.safetensors"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: config.json not found in: ${MODEL_PATH}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting ProkBERT Batch Inference Jobs"
echo "=========================================="
echo "Input list: ${INPUT_LIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Checkpoint: ${MODEL_PATH}"
echo "  Base model: ${BASE_MODEL}"
echo ""
echo "Inference Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_inference.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --base_model "${BASE_MODEL}"
