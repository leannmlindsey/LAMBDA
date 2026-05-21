#!/bin/bash
#SBATCH --job-name=lambda_baselines
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=lambda_baselines_%j.out
#SBATCH --error=lambda_baselines_%j.err
#
# CPU-only sequence-feature baselines (GC + k=4 + k=6 k-mer compositions,
# logistic regression) for the LAMBDA benchmark. Runs all (category × window)
# slices in one job; finishes in 10–15 minutes total.
#
# Output layout (auto-discovered by ../03_build_website_data.py):
#   RESULTS_ROOT/baseline_gc/{binary,error_bias,genome_wide}/{2k,4k,8k}/*_predictions.csv
#   RESULTS_ROOT/baseline_kmer4/...
#   RESULTS_ROOT/baseline_kmer6/...
#   RESULTS_ROOT/figures/gc_distribution.pdf
#
# Usage:
#   sbatch run_baselines.slurm

set -euo pipefail

# ============================================================
# CONFIGURATION — edit if your paths differ
# ============================================================

# Root of the LAMBDA dataset. Use the physical /gpfs path rather than the
# /home/lindseylm/lindseylm symlink — compute nodes resolve the physical
# path more reliably. The script auto-detects which subdir layout you have
# at this root (Zenodo-style binary_segments_2k/, or the existing Biowulf
# merged_datasets_filtered/2k/).
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"

# Where to write predictions, metrics, and the GC figure.
RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/LAMBDA/results"

# Conda env (needs pandas, scikit-learn, scipy, matplotlib).
# The gena_lm env created by GENA_LM_generic_sequence_classification/setup_lambda.sh
# already has all of these (scipy comes in transitively via scikit-learn),
# so it's the simplest one to reuse.
BASELINES_ENV="gena_lm"

# Which window sizes to process.
WINDOWS="2k 4k 8k"

# ============================================================

# Allow ../paths.env to override the above (if it exists)
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/replication/baselines"
# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
conda activate $BASELINES_ENV 2>/dev/null || source activate $BASELINES_ENV 2>/dev/null || true



echo "============================================================"
echo "LAMBDA sequence-feature baselines"
echo "  DATASET_ROOT = ${DATASET_ROOT}"
echo "  RESULTS_ROOT = ${RESULTS_ROOT}"
echo "  WINDOWS      = ${WINDOWS}"
echo "  Job ID       = ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

mkdir -p "${RESULTS_ROOT}"

python "${SCRIPT_DIR}/run_baselines.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_root "${RESULTS_ROOT}" \
    --windows ${WINDOWS} \
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
