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
# This is CPU-only and reasonably fast. Use the SLURM wrapper only if
# RESULTS_ROOT contains a large number of genome-wide JSONs; otherwise it's
# fine to run on a login node:
#   python 03_build_website_data.py --predictions ${RESULTS_ROOT} ...
#
# Usage:
#   sbatch run_aggregate.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${SCRIPT_DIR}/paths.env" ]; then
    echo "ERROR: ${SCRIPT_DIR}/paths.env not found"
    exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

: "${RESULTS_ROOT:?RESULTS_ROOT not set in paths.env}"

module load conda
source activate lambda_aggregate  # or any env with python>=3.10, pandas, scikit-learn

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
echo "  RESULTS_ROOT = ${RESULTS_ROOT}"
echo "  OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "  Job ID       = ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/03_build_website_data.py" "${OPTS[@]}"

echo "============================================================"
echo "Done. Outputs:"
echo "  ${OUTPUT_DIR}/metrics_summary.csv"
echo "  ${OUTPUT_DIR}/website_data/{assembly}_{window}.json + index.json"
echo "============================================================"
