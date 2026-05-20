#!/bin/bash

# Interactive script for running nucleotide-level prophage evaluation WITHOUT sbatch
# Usage: bash run_nucleotide_evaluation_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_nucleotide_evaluation.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_nucleotide_evaluation.sh}"

# Also check in scripts/ directory
if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_run_nucleotide_evaluation.sh"
fi

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_nucleotide_evaluation_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but just get the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

echo ""
echo "Nucleotide-Level Prophage Evaluation (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

echo "Python environment:"
which python
python --version
echo ""

# Set defaults
INPUT_CSV=${INPUT_CSV:-}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-./nucleotide_eval_results}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-nucleotide_eval}
FEATURE_IDX=${FEATURE_IDX:-19746}
THRESHOLD=${THRESHOLD:-0.5}
ADAPTIVE_THRESHOLD=${ADAPTIVE_THRESHOLD:-false}
ADAPTIVE_METHOD=${ADAPTIVE_METHOD:-mad}
ADAPTIVE_K=${ADAPTIVE_K:-3.0}
NORMALIZATION=${NORMALIZATION:-none}
NORM_WINDOW=${NORM_WINDOW:-10000}
MAX_GAP=${MAX_GAP:-100}
MERGE_DISTANCE=${MERGE_DISTANCE:-3000}
MIN_REGION_SIZE=${MIN_REGION_SIZE:-1000}
PLOT=${PLOT:-false}

# Validate required parameters
if [ -z "${INPUT_CSV}" ] || [ "${INPUT_CSV}" == "/path/to/sae_inference_results.csv" ]; then
    echo "ERROR: INPUT_CSV is not set or still has placeholder value"
    echo "Please edit the wrapper script: ${WRAPPER_SCRIPT}"
    exit 1
fi

if [ -z "${ACTIVATIONS_DIR}" ] || [ "${ACTIVATIONS_DIR}" == "/path/to/activations_dir" ]; then
    echo "ERROR: ACTIVATIONS_DIR is not set or still has placeholder value"
    echo "Please edit the wrapper script: ${WRAPPER_SCRIPT}"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input CSV:       ${INPUT_CSV}"
echo "  Activations dir: ${ACTIVATIONS_DIR}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Output prefix:   ${OUTPUT_PREFIX}"
echo "  Feature index:   ${FEATURE_IDX}"
echo "  Threshold:       ${THRESHOLD}"
echo "  Adaptive:        ${ADAPTIVE_THRESHOLD}"
echo "  Normalization:   ${NORMALIZATION}"
echo "  Max gap:         ${MAX_GAP}"
echo "  Merge distance:  ${MERGE_DISTANCE}"
echo "  Min region size: ${MIN_REGION_SIZE}"
echo "  Plot:            ${PLOT}"
echo "============================================================"
echo ""

# Build optional flags
ADAPTIVE_FLAG=""
if [ "${ADAPTIVE_THRESHOLD}" == "true" ]; then
    ADAPTIVE_FLAG="--adaptive_threshold --adaptive_method=${ADAPTIVE_METHOD} --adaptive_k=${ADAPTIVE_K}"
fi

NORM_FLAG=""
if [ "${NORMALIZATION}" != "none" ]; then
    NORM_FLAG="--normalization=${NORMALIZATION} --norm_window=${NORM_WINDOW}"
else
    NORM_FLAG="--normalization=none"
fi

PLOT_FLAG=""
if [ "${PLOT}" == "true" ]; then
    PLOT_FLAG="--plot"
fi

# Run nucleotide evaluation
python src/nucleotide_evaluation.py \
    --input_csv="${INPUT_CSV}" \
    --activations_dir="${ACTIVATIONS_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --output_prefix="${OUTPUT_PREFIX}" \
    --feature_idx=${FEATURE_IDX} \
    --threshold=${THRESHOLD} \
    --max_gap=${MAX_GAP} \
    --merge_distance=${MERGE_DISTANCE} \
    --min_region_size=${MIN_REGION_SIZE} \
    ${NORM_FLAG} \
    ${ADAPTIVE_FLAG} \
    ${PLOT_FLAG}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
