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

# Where the per-model prediction CSVs live. Same as RESULTS_ROOT in
# run_baselines.slurm and 02_submit_inference_jobs.sh.
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/LAMBDA/results"

# Conda env (needs python>=3.10, pandas, scikit-learn).
AGGREGATE_ENV="gena_lm"

# Optional metadata for the per-genome website JSON files. Leave blank to
# skip — the metrics_summary.csv is produced regardless.
GROUND_TRUTH_CSV=""        # e.g. /gpfs/.../lambda_final/lambda_ground_truth.csv
TAXONOMY_CSV=""            # simple per-assembly taxonomy or GTDB-Tk TSV
CHECKV_TSV=""              # CheckV quality_summary.tsv (optional)
PHROG_CSV=""               # phage_annotated_segments.csv (optional)

# ============================================================

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Activate conda environment (tries modern `conda activate` then falls back
# to legacy `source activate`)
conda activate "${AGGREGATE_ENV}" 2>/dev/null || source activate "${AGGREGATE_ENV}" 2>/dev/null || true

OUTPUT_DIR="${RESULTS_ROOT}/aggregated"
mkdir -p "${OUTPUT_DIR}"

OPTS=(--predictions "${RESULTS_ROOT}" --output "${OUTPUT_DIR}")
[ -n "${GROUND_TRUTH_CSV}" ] && [ -f "${GROUND_TRUTH_CSV}" ] && OPTS+=(--ground-truth "${GROUND_TRUTH_CSV}")
[ -n "${TAXONOMY_CSV}" ]     && [ -f "${TAXONOMY_CSV}" ]     && OPTS+=(--taxonomy "${TAXONOMY_CSV}")
[ -n "${CHECKV_TSV}" ]       && [ -f "${CHECKV_TSV}" ]       && OPTS+=(--checkv "${CHECKV_TSV}")
[ -n "${PHROG_CSV}" ]        && [ -f "${PHROG_CSV}" ]        && OPTS+=(--phrog "${PHROG_CSV}")

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
