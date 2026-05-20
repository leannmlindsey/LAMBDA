#!/bin/bash
#SBATCH --job-name=nuc_eval
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=nuc_eval_%j.out
#SBATCH --error=nuc_eval_%j.err

# Batch script for nucleotide-level prophage evaluation (CPU only — no GPU needed)
# Usage: sbatch run_nucleotide_evaluation.sh
#
# Required environment variables:
#   INPUT_CSV: Output CSV from sae_inference.py
#   ACTIVATIONS_DIR: Directory with per-segment .npy files

echo "============================================================"
echo "Nucleotide-Level Prophage Evaluation"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

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
if [ -z "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV is not set"
    exit 1
fi

if [ -z "${ACTIVATIONS_DIR}" ]; then
    echo "ERROR: ACTIVATIONS_DIR is not set"
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
