#!/bin/bash

# Wrapper script for running batch inference with DNABERT-2 on Biowulf
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
#INPUT_LIST="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/scripts/input_files.txt"
INPUT_LIST="genome_files_4k.txt"
#INPUT_LIST="input_files_4k.txt"
#INPUT_LIST="genome_files_2k.txt"
#INPUT_LIST="input_files_2k.txt"
# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/results/inference/4k"
#OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/results/inference/2k"

# === REQUIRED: Model Configuration ===
# HuggingFace model name or path to fine-tuned model directory
MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/scripts/output/dnabert2_lambda_filtered_4k_lr3e-5_seed10/checkpoint-20400"
#MODEL_PATH="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/scripts/output/lambda_filtered/dnabert2_lambda_filtered_2k_lr3e-5_seed7/checkpoint-20400"
# === OPTIONAL: Inference Parameters ===
BATCH_SIZE="16"
MAX_LENGTH="512"
THRESHOLD="0.5"

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

if [ "${MODEL_PATH}" == "/path/to/finetuned/model" ]; then
    echo "ERROR: Please set MODEL_PATH to your fine-tuned model directory"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Check if MODEL_PATH is a directory (local model) or HuggingFace model name
if [[ "${MODEL_PATH}" != *"/"* ]] || [ -d "${MODEL_PATH}" ]; then
    if [ -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
        echo "ERROR: Model directory does not contain config.json: ${MODEL_PATH}"
        exit 1
    fi
fi

# Get script directory
#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/DNABERT2/DNABERT_2/finetune/scripts"
echo "=========================================="
echo "Submitting DNABERT-2 Batch Inference Jobs"
echo "=========================================="
echo "Input list: ${INPUT_LIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Model: ${MODEL_PATH}"
echo ""
echo "Inference Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Threshold: ${THRESHOLD}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_inference.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --threshold "${THRESHOLD}"
