#!/bin/bash

# Wrapper script for running batch inference with a HuggingFace-hosted ProkBERT model
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_batch_inference_hf.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Each CSV file must have a 'sequence' column (and optionally 'label')
INPUT_LIST="/data/lindseylm/GLM_EVALUATIONS/MODELS/PROKBERT/prokbert/slurm_scripts/inference_filepaths_2k.txt"
#INPUT_LIST="/data/lindseylm/GLM_EVALUATIONS/MODELS/PROKBERT/prokbert/slurm_scripts/inference_filepaths_4k.txt"
#INPUT_LIST="/data/lindseylm/GLM_EVALUATIONS/MODELS/PROKBERT/prokbert/slurm_scripts/inference_filepaths_8k.txt"

len="2k"
#len="4k"
#len="8k"

# === REQUIRED: HuggingFace Model ===
# The model will be downloaded automatically from HuggingFace Hub
MODEL_NAME="neuralbioinfo/prokbert-mini-c-phage"

# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/PROKBERT/prokbert/results/inference/error_and_bias/hf_$(basename ${MODEL_NAME})/${len}"

# === OPTIONAL: Inference Parameters ===
# Batch size for inference (default: 32)
BATCH_SIZE="32"

# Maximum sequence length (default: 1024)
# NOTE: prokbert-mini-c has max_position_embeddings=1024, so sequences
# longer than ~1022 bp will be truncated at the token level.
# For 4k and 8k inputs, the sequences WILL be truncated to fit the model.
MAX_LENGTH="1024"

#####################################################################
# END CONFIGURATION
#####################################################################

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Validate configuration
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

echo "=========================================="
echo "Submitting ProkBERT HuggingFace Batch Inference Jobs"
echo "=========================================="
echo "Input list:  ${INPUT_LIST}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  HF Model:  ${MODEL_NAME}"
echo ""
echo "Inference Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_inference_hf.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}"
