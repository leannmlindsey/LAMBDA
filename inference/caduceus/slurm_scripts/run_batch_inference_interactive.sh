#!/bin/bash

# Interactive script for running Caduceus batch inference WITHOUT sbatch
# Usage: bash run_batch_inference_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_batch_inference.sh (or specify another)
# and runs inference directly on the current node (sequentially for each input file).

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_batch_inference.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_batch_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Extract variable assignments from wrapper (lines with = that aren't comments)
source <(grep -E '^[A-Z_]+=' "${WRAPPER_SCRIPT}" | grep -v '^#')

echo ""
echo "Caduceus Batch Inference (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

# Activate conda environment
source activate caduceus_env

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "Python environment:"
which python
python --version
echo ""

# ============================================================
# IMPORTANT: Update this path to your repo location on the cluster
# ============================================================
SCRIPT_DIR="/gpfs/gsfs12/users/lindseylm/GLM_EVALUATIONS/MODELS/CADUCEUS_GENERIC/Caduceus_generic_sequence_classification"

cd "${SCRIPT_DIR}" || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ] || [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: INPUT_LIST is not set properly in wrapper"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set properly in wrapper"
    exit 1
fi

if [ "${CONFIG_PATH}" == "/path/to/model_config.json" ] || [ -z "${CONFIG_PATH}" ]; then
    echo "ERROR: CONFIG_PATH is not set properly in wrapper"
    exit 1
fi

if [ "${CHECKPOINT_PATH}" == "/path/to/checkpoint.ckpt" ] || [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set properly in wrapper"
    exit 1
fi

# Verify files exist
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-1024}
D_OUTPUT=${D_OUTPUT:-2}
THRESHOLD=${THRESHOLD:-0.5}
CONJOIN_TEST=${CONJOIN_TEST:-true}

# Build conjoin flag
CONJOIN_FLAG=""
if [ "${CONJOIN_TEST}" == "true" ]; then
    CONJOIN_FLAG="--conjoin_test"
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input list: ${INPUT_LIST}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Config: ${CONFIG_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  D output: ${D_OUTPUT}"
echo "  Threshold: ${THRESHOLD}"
echo "  Conjoin test: ${CONJOIN_TEST}"
echo "============================================================"
echo ""

# Run batch inference with single model load
python -m src.batch_inference \
    --input_list="${INPUT_LIST}" \
    --output_dir="${OUTPUT_DIR}" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --config_path="${CONFIG_PATH}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --d_output=${D_OUTPUT} \
    --threshold=${THRESHOLD} \
    ${CONJOIN_FLAG} \
    --save_metrics

echo ""
echo "Job completed at: $(date)"
echo "============================================================"
