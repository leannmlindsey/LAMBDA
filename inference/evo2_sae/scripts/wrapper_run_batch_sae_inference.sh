#!/bin/bash

# Wrapper script for running batch SAE inference with Evo2
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_batch_sae_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Example contents of input_files.txt:
#   /path/to/dataset1.csv
#   /path/to/dataset2.csv
#   /path/to/dataset3.csv
INPUT_LIST="/path/to/input_files.txt"

# === REQUIRED: Output Directory ===
# All predictions and SLURM logs will be saved here
OUTPUT_DIR="/path/to/output_directory"

# === OPTIONAL: Model Configuration ===
MODEL="evo2_7b"                     # evo2_7b or evo2_40b

# === OPTIONAL: SAE Feature Configuration ===
FEATURE_IDX="19746"                 # SAE feature index (19746 = prophage)

# === OPTIONAL: Threshold Parameters ===
MAX_THRESHOLD="0.5"                 # Max activation threshold for pred_label
MEAN_THRESHOLD="0.1"               # Mean activation threshold for pred_label
FRACTION_THRESHOLD="0.3"           # Fraction firing threshold for pred_label

# === OPTIONAL: Other Parameters ===
BATCH_SIZE="1"
SAVE_ACTIVATIONS="true"            # Save per-segment .npy activation arrays

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

# Verify files exist
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Submitting Evo2 SAE Batch Inference Jobs"
echo "=========================================="
echo "Input list: ${INPUT_LIST}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Model: ${MODEL}"
echo "  Feature index: ${FEATURE_IDX}"
echo ""
echo "Thresholds:"
echo "  Max threshold: ${MAX_THRESHOLD}"
echo "  Mean threshold: ${MEAN_THRESHOLD}"
echo "  Fraction threshold: ${FRACTION_THRESHOLD}"
echo ""
echo "Other:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Save activations: ${SAVE_ACTIVATIONS}"
echo "=========================================="

# Call the batch submission script
"${SCRIPT_DIR}/submit_batch_sae_inference.sh" \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --feature_idx "${FEATURE_IDX}" \
    --max_threshold "${MAX_THRESHOLD}" \
    --mean_threshold "${MEAN_THRESHOLD}" \
    --fraction_threshold "${FRACTION_THRESHOLD}" \
    --batch_size "${BATCH_SIZE}" \
    --save_activations "${SAVE_ACTIVATIONS}"
