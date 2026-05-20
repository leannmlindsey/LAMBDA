#!/bin/bash
#SBATCH --job-name=caduceus_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=caduceus_inf_%j.out
#SBATCH --error=caduceus_inf_%j.err

# Biowulf batch script for Caduceus inference
# Usage: sbatch run_inference.sh
#
# Required environment variables:
#   INPUT_CSV: Path to CSV file with 'sequence' column
#   CHECKPOINT_PATH: Path to fine-tuned Caduceus checkpoint (.ckpt file)
#   CONFIG_PATH: Path to Caduceus config JSON file

echo "============================================================"
echo "Caduceus Inference"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load cuda/12.8

# Activate conda environment
source activate caduceus_env

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-1024}
D_OUTPUT=${D_OUTPUT:-2}
THRESHOLD=${THRESHOLD:-0.5}
CONJOIN_TEST=${CONJOIN_TEST:-true}

# Validate required parameters
if [ -z "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV is not set"
    exit 1
fi

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set"
    exit 1
fi

if [ -z "${CONFIG_PATH}" ]; then
    echo "ERROR: CONFIG_PATH is not set"
    exit 1
fi

# ============================================================
# IMPORTANT: Update this path to your repo location on the cluster
# ============================================================
SCRIPT_DIR="/gpfs/gsfs12/users/lindseylm/GLM_EVALUATIONS/MODELS/CADUCEUS_GENERIC/Caduceus_generic_sequence_classification"

cd "${SCRIPT_DIR}" || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Set output path
if [ -z "${OUTPUT_CSV}" ]; then
    OUTPUT_CSV="${INPUT_CSV%.csv}_predictions.csv"
fi

# Build conjoin flag
CONJOIN_FLAG=""
if [ "${CONJOIN_TEST}" == "true" ]; then
    CONJOIN_FLAG="--conjoin_test"
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Config: ${CONFIG_PATH}"
echo "  Input CSV: ${INPUT_CSV}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  D output: ${D_OUTPUT}"
echo "  Threshold: ${THRESHOLD}"
echo "  Conjoin test: ${CONJOIN_TEST}"
echo "============================================================"
echo ""

# Run inference
python -m src.inference \
    --input_csv="${INPUT_CSV}" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --config_path="${CONFIG_PATH}" \
    --output_csv="${OUTPUT_CSV}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --d_output=${D_OUTPUT} \
    --threshold=${THRESHOLD} \
    ${CONJOIN_FLAG} \
    --save_metrics

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Predictions saved to: ${OUTPUT_CSV}"
echo "============================================================"
