#!/bin/bash
#SBATCH --job-name=lambda_aggregate
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=lambda_aggregate_%j.out
#SBATCH --error=lambda_aggregate_%j.err
#
# SLURM wrapper for step 03 — aggregate per-model predictions into
# metrics_summary.csv and per-genome website JSON files.
#
# CPU-only. Use the SLURM wrapper only if RESULTS_ROOT contains a large
# number of genome-wide JSONs; otherwise it's fine to run on a login node:
#   python 03_build_website_data.py --predictions ${RESULTS_ROOT} ...
#
# Usage:
#   sbatch run_aggregate.slurm

set -euo pipefail

# ============================================================
# CONFIGURATION — edit if your paths differ
# ============================================================

# Full path to this script's parent directory (the replication/ dir of the
# LAMBDA repo on Biowulf). SLURM copies the script to a temp dir before
# running, so BASH_SOURCE-based discovery does not work — must be hardcoded.
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/replication"

# Conda env (needs python>=3.10, pandas, scikit-learn).
# The gena_lm env created by GENA_LM_generic_sequence_classification/setup_lambda.sh
# has everything needed.
AGGREGATE_ENV="gena_lm"

# ============================================================

# paths.env lives next to this script — sourced for DATASET_ROOT / RESULTS_ROOT
# / TAXONOMY_CSV / etc.
if [ ! -f "${SCRIPT_DIR}/paths.env" ]; then
    echo "ERROR: ${SCRIPT_DIR}/paths.env not found"
    exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

: "${RESULTS_ROOT:?RESULTS_ROOT not set in paths.env}"

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment (tries modern `conda activate` then falls back
# to legacy `source activate`)
conda activate "${AGGREGATE_ENV}" 2>/dev/null || source activate "${AGGREGATE_ENV}" 2>/dev/null || true

OUTPUT_DIR="${RESULTS_ROOT}/aggregated"
mkdir -p "${OUTPUT_DIR}"

OPTS=(--predictions "${RESULTS_ROOT}" --output "${OUTPUT_DIR}")

# Optional metadata — only added if the variables are non-empty in paths.env.
if [ -n "${DATASET_ROOT:-}" ] && [ -f "${DATASET_ROOT}/ground_truth.csv" ]; then
    OPTS+=(--ground-truth "${DATASET_ROOT}/ground_truth.csv")
fi
if [ -n "${TAXONOMY_CSV:-}" ]; then
    OPTS+=(--taxonomy "${TAXONOMY_CSV}")
fi
if [ -n "${CHECKV_TSV:-}" ]; then
    OPTS+=(--checkv "${CHECKV_TSV}")
fi
if [ -n "${PHROG_CSV:-}" ]; then
    OPTS+=(--phrog "${PHROG_CSV}")
fi

echo "============================================================"
echo "Aggregating LAMBDA results"
echo "  SCRIPT_DIR   = ${SCRIPT_DIR}"
echo "  RESULTS_ROOT = ${RESULTS_ROOT}"
echo "  OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "  ENV          = ${AGGREGATE_ENV}"
echo "  Job ID       = ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/03_build_website_data.py" "${OPTS[@]}"

echo "============================================================"
echo "Done. Outputs:"
echo "  ${OUTPUT_DIR}/metrics_summary.csv"
echo "  ${OUTPUT_DIR}/website_data/{assembly}_{window}.json + index.json"
echo "============================================================"
