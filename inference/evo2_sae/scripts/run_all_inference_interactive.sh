#!/bin/bash

# Interactive script for running all inference methods (SAE, NN, LP)
# Loads Evo2 ONCE and processes all input files.
#
# Usage: bash scripts/run_all_inference_interactive.sh [wrapper_script.sh]

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_all_inference.sh}"

# Also check in scripts/ directory
if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_run_all_inference.sh"
fi

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_all_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Extract variable assignments from wrapper (lines with = that aren't comments)
source <(grep -E '^[A-Z_]+=' "${WRAPPER_SCRIPT}" | grep -v '^#')

echo ""
echo "Evo2 Batch Inference (Interactive Mode)"
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
nvidia-smi 2>/dev/null || echo "No GPU detected"
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

# Validate
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ] || [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: INPUT_LIST is not set properly in wrapper"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set properly in wrapper"
    exit 1
fi

if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Set defaults
MODEL=${MODEL:-evo2_7b}
LAYER=${LAYER:-blocks.28.mlp.l3}
POOLING=${POOLING:-mean}
BATCH_SIZE=${BATCH_SIZE:-16}
FEATURE_IDX=${FEATURE_IDX:-19746}
SAE_MAX_THRESHOLD=${SAE_MAX_THRESHOLD:-0.5}
SAE_MEAN_THRESHOLD=${SAE_MEAN_THRESHOLD:-0.1}
SAE_FRACTION_THRESHOLD=${SAE_FRACTION_THRESHOLD:-0.3}
SAVE_ACTIVATIONS=${SAVE_ACTIVATIONS:-true}
RUN_SAE=${RUN_SAE:-true}
RUN_NN=${RUN_NN:-true}
RUN_LP=${RUN_LP:-true}

# Build flags
METHOD_FLAGS=""
if [ "${RUN_SAE}" == "true" ]; then METHOD_FLAGS="${METHOD_FLAGS} --run_sae"; fi
if [ "${RUN_NN}" == "true" ]; then METHOD_FLAGS="${METHOD_FLAGS} --run_nn"; fi
if [ "${RUN_LP}" == "true" ]; then METHOD_FLAGS="${METHOD_FLAGS} --run_lp"; fi

SAVE_ACT_FLAG=""
if [ "${SAVE_ACTIVATIONS}" == "true" ]; then SAVE_ACT_FLAG="--save_activations"; fi

MODEL_DIR_FLAG=""
if [ -n "${MODEL_DIR}" ] && [ "${MODEL_DIR}" != "/path/to/results/embedding_analysis/2k" ]; then
    MODEL_DIR_FLAG="--model_dir ${MODEL_DIR}"
fi

echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input list:    ${INPUT_LIST}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Model dir:     ${MODEL_DIR}"
echo "  Model:         ${MODEL}"
echo "  Layer:         ${LAYER}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Methods:       SAE=${RUN_SAE}  NN=${RUN_NN}  LP=${RUN_LP}"
echo "============================================================"
echo ""

python src/batch_inference.py \
    --input_list "${INPUT_LIST}" \
    --output_dir "${OUTPUT_DIR}" \
    ${MODEL_DIR_FLAG} \
    --model "${MODEL}" \
    --layer "${LAYER}" \
    --pooling "${POOLING}" \
    --batch_size ${BATCH_SIZE} \
    --feature_idx ${FEATURE_IDX} \
    --sae_max_threshold ${SAE_MAX_THRESHOLD} \
    --sae_mean_threshold ${SAE_MEAN_THRESHOLD} \
    --sae_fraction_threshold ${SAE_FRACTION_THRESHOLD} \
    ${SAVE_ACT_FLAG} \
    ${METHOD_FLAGS}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
