#!/bin/bash

# Interactive script for running Evo2 SAE batch inference WITHOUT sbatch
# Usage: bash run_batch_sae_inference_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_batch_sae_inference.sh (or specify another)
# and runs inference directly on the current node (sequentially for each input file).

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_batch_sae_inference.sh}"

# Also check in scripts/ directory
if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_run_batch_sae_inference.sh"
fi

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_batch_sae_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Extract variable assignments from wrapper (lines with = that aren't comments)
source <(grep -E '^[A-Z_]+=' "${WRAPPER_SCRIPT}" | grep -v '^#')

echo ""
echo "Evo2 SAE Batch Inference (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU detected (nvidia-smi not found)"
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ] || [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: INPUT_LIST is not set properly in wrapper"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set properly in wrapper"
    exit 1
fi

# Verify input list exists
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set defaults
MODEL=${MODEL:-evo2_7b}
FEATURE_IDX=${FEATURE_IDX:-19746}
MAX_THRESHOLD=${MAX_THRESHOLD:-0.5}
MEAN_THRESHOLD=${MEAN_THRESHOLD:-0.1}
FRACTION_THRESHOLD=${FRACTION_THRESHOLD:-0.3}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_ACTIVATIONS=${SAVE_ACTIVATIONS:-true}

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input list:          ${INPUT_LIST}"
echo "  Output dir:          ${OUTPUT_DIR}"
echo "  Model:               ${MODEL}"
echo "  Feature index:       ${FEATURE_IDX}"
echo "  Max threshold:       ${MAX_THRESHOLD}"
echo "  Mean threshold:      ${MEAN_THRESHOLD}"
echo "  Fraction threshold:  ${FRACTION_THRESHOLD}"
echo "  Batch size:          ${BATCH_SIZE}"
echo "  Save activations:    ${SAVE_ACTIVATIONS}"
echo "============================================================"
echo ""

# Build save_activations flag
SAVE_ACT_FLAG=""
if [ "${SAVE_ACTIVATIONS}" == "true" ]; then
    SAVE_ACT_FLAG="--save_activations"
fi

# Count input files (non-empty, non-comment lines)
NUM_FILES=$(grep -c -v -E '^[[:space:]]*$|^[[:space:]]*#' "${INPUT_LIST}" || echo 0)
echo "Found ${NUM_FILES} input files to process"
echo ""

# Process each input file sequentially
COUNT=0
while IFS= read -r INPUT_CSV || [ -n "${INPUT_CSV}" ]; do
    # Skip empty lines and comments
    if [[ -z "${INPUT_CSV}" ]] || [[ "${INPUT_CSV}" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    INPUT_CSV=$(echo "${INPUT_CSV}" | xargs)

    # Validate input file exists
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "WARNING: Input file not found, skipping: ${INPUT_CSV}"
        continue
    fi

    COUNT=$((COUNT + 1))
    INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)
    OUTPUT_CSV="${OUTPUT_DIR}/${INPUT_BASENAME}_sae_results.csv"

    echo "============================================================"
    echo "Processing file ${COUNT}/${NUM_FILES}: ${INPUT_BASENAME}"
    echo "  Input:  ${INPUT_CSV}"
    echo "  Output: ${OUTPUT_CSV}"
    echo "============================================================"

    python src/sae_inference.py \
        --input_csv "${INPUT_CSV}" \
        --output "${OUTPUT_CSV}" \
        --model "${MODEL}" \
        --feature_idx ${FEATURE_IDX} \
        --max_threshold ${MAX_THRESHOLD} \
        --mean_threshold ${MEAN_THRESHOLD} \
        --fraction_threshold ${FRACTION_THRESHOLD} \
        --batch_size ${BATCH_SIZE} \
        ${SAVE_ACT_FLAG}

    # Calculate and display metrics for this file
    if [ -f "${OUTPUT_CSV}" ]; then
        python src/calculate_metrics.py --input "${OUTPUT_CSV}"
    fi

    echo ""

done < "${INPUT_LIST}"

# Aggregate metrics across all result files
echo "============================================================"
echo "Aggregate Metrics Across All Files"
echo "============================================================"
python src/calculate_metrics.py \
    --input_dir "${OUTPUT_DIR}" \
    --output_json "${OUTPUT_DIR}/metrics.json"

echo "============================================================"
echo "Batch Inference Complete"
echo "============================================================"
echo "Processed ${COUNT} files"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Job completed at: $(date)"
echo "============================================================"
