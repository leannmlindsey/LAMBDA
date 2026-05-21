#!/bin/bash
#SBATCH --job-name=lambda_baselines
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=lambda_baselines_%j.out
#SBATCH --error=lambda_baselines_%j.err
#
# CPU-only baselines (GC + k=4 + k=6 k-mer compositions, logistic regression)
# for the LAMBDA benchmark. Runs all (category × window) slices in one job.
#
# Output layout (auto-discovered by ../03_build_website_data.py):
#   RESULTS_ROOT/baseline_gc/{binary,error_bias,genome_wide}/{2k,4k,8k}/*_predictions.csv
#   RESULTS_ROOT/baseline_kmer4/...
#   RESULTS_ROOT/baseline_kmer6/...
#   RESULTS_ROOT/figures/gc_distribution.pdf
#
# Usage:
#   sbatch run_baselines.slurm
#
# Prerequisites:
#   1. paths.env populated in ../paths.env (DATASET_ROOT, RESULTS_ROOT)
#   2. A conda env with: pandas, scikit-learn, scipy, matplotlib
#      (the lambda_aggregate or any LAMBDA env will work)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source paths from ../paths.env
if [ ! -f "${SCRIPT_DIR}/../paths.env" ]; then
    echo "ERROR: ${SCRIPT_DIR}/../paths.env not found"
    exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../paths.env"

: "${DATASET_ROOT:?DATASET_ROOT not set in paths.env}"
: "${RESULTS_ROOT:?RESULTS_ROOT not set in paths.env}"

# Adjust the env name to match your cluster's setup.
module load conda
source activate "${BASELINES_ENV:-lambda_aggregate}"

echo "============================================================"
echo "LAMBDA sequence-feature baselines"
echo "  DATASET_ROOT = ${DATASET_ROOT}"
echo "  RESULTS_ROOT = ${RESULTS_ROOT}"
echo "  Job ID       = ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/run_baselines.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_root "${RESULTS_ROOT}" \
    --windows 2k 4k 8k \
    --plot

echo "============================================================"
echo "Done. Aggregate with:"
echo "  python ${SCRIPT_DIR}/../03_build_website_data.py \\"
echo "      --predictions ${RESULTS_ROOT} \\"
echo "      --output ${RESULTS_ROOT}/aggregated"
echo "Baseline rows will appear in metrics_summary.csv as:"
echo "  baseline_gc, baseline_kmer4, baseline_kmer6"
echo "GC figure: ${RESULTS_ROOT}/figures/gc_distribution.pdf"
echo "============================================================"
