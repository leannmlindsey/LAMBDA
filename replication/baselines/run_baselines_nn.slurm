#!/bin/bash
#SBATCH --job-name=lambda_baselines_nn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=lambda_baselines_nn_%j.out
#SBATCH --error=lambda_baselines_nn_%j.err
#
# Trains the SAME 3-Layer NN the gLM "3-Layer NN" column in paper Table 2 uses
# (verbatim ThreeLayerNN class from DNABERT2/NTv2/Caduceus
# embedding_analysis*.py: 256 -> 128 -> 2 with ReLU + Dropout(0.3), Adam lr=1e-3,
# full-batch, max 100 epochs, early stopping patience=10 on val loss, seed=42),
# but on the GC / k=4 / k=6 baseline feature vectors instead of gLM embeddings.
#
# Output:
#   ${RESULTS_ROOT}/baselines_nn/baseline_<feat>_<window>_nn_metrics.json
#   ${RESULTS_ROOT}/baselines_nn/baseline_nn_summary.csv
#
# Total runtime: ~15 minutes on an A100. GC is trivial (1 feature) so it gets
# skipped via --baselines (kept available via flag if needed for the supplement).

set -euo pipefail

# ============================================================
# CONFIGURATION -- edit if your paths differ
# ============================================================

# Canonical /gpfs path (not /home/lindseylm/lindseylm symlink) -- per
# project memory: compute nodes resolve the physical path more reliably.
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"

RESULTS_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/LAMBDA/results"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/replication/baselines"

# The gena_lm env from GENA_LM_generic_sequence_classification/setup_lambda.sh
# already has torch + sklearn + pandas. Same env as run_baselines.slurm.
BASELINES_ENV="gena_lm"

WINDOWS="2k 4k 8k"
BASELINES="kmer4 kmer6"   # GC has 1 feature; NN reduces to LP -- skipped.

# ============================================================

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi
conda activate $BASELINES_ENV 2>/dev/null || source activate $BASELINES_ENV 2>/dev/null || true

OUTPUT_DIR="${RESULTS_ROOT}/baselines_nn"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "LAMBDA 3-Layer NN baselines (Table 2 column)"
echo "  DATASET_ROOT = ${DATASET_ROOT}"
echo "  OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "  WINDOWS      = ${WINDOWS}"
echo "  BASELINES    = ${BASELINES}"
echo "  Job ID       = ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/run_baselines_nn.py" \
    --dataset_root "${DATASET_ROOT}" \
    --output_root "${OUTPUT_DIR}" \
    --windows ${WINDOWS} \
    --baselines ${BASELINES}

echo "============================================================"
echo "Done."
echo "Summary CSV: ${OUTPUT_DIR}/baseline_nn_summary.csv"
echo "Use the MCC column to fill in the '3-Layer NN Model' column of paper Table 2"
echo "for the k=4 and k=6 baseline rows."
echo "============================================================"
