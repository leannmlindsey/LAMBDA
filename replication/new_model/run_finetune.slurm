#!/bin/bash
#SBATCH --job-name=lambda_new_model_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=lambda_new_model_ft_%j.out
#SBATCH --error=lambda_new_model_ft_%j.err
#
# Fine-tune a new genomic language model on a LAMBDA train/dev/test split.
#
# Edit the configuration block below, then submit with:
#   sbatch run_finetune.slurm
#
# To fine-tune on all three window sizes, submit three separate jobs (or use
# a job array — see the bottom of this file for a template).

set -euo pipefail

#####################################################################
# CONFIGURATION — edit this section
#####################################################################

# === Model ===
# HuggingFace model identifier or local path. Examples:
#   AIRI-Institute/gena-lm-bert-base-t2t            (GENA-LM, 512 tokens)
#   InstaDeepAI/nucleotide-transformer-v2-500m-multi-species  (NT-v2, 2048 tokens)
#   zhihan1996/DNABERT-2-117M                        (DNABERT-2, requires --trust_remote_code)
MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"

# === LAMBDA data ===
# Directory containing train.csv / dev.csv / test.csv. The LAMBDA dataset on
# Zenodo provides three of these:
#   ${DATASET_ROOT}/binary_segments_2k
#   ${DATASET_ROOT}/binary_segments_4k
#   ${DATASET_ROOT}/binary_segments_8k
DATASET_DIR="/path/to/LAMBDA/binary_segments_2k"

# === Output ===
# Where to save the fine-tuned checkpoint. After fine-tuning is complete the
# script also writes test_results.json here.
OUTPUT_DIR="/path/to/checkpoints/gena_lm/2k"

# === Training hyperparameters ===
MAX_LENGTH=512        # Match your model's context window
BATCH_SIZE=16         # Reduce if OOM. Halve for 4k, quarter for 8k.
LEARNING_RATE=3e-5
NUM_EPOCHS=3
PRECISION="--bf16"    # "--bf16" on A100/H100; "--fp16" on V100; "" for fp32
EARLY_STOPPING_PATIENCE=3
SEED=42

# Set this to "--trust_remote_code" if your model requires it (DNABERT-2 does;
# GENA-LM and NT-v2 do not).
TRUST_REMOTE_CODE=""

#####################################################################
# END CONFIGURATION
#####################################################################

# Load environment. Edit the conda env name and module loads to match your cluster.
module load conda
module load CUDA/12.8
source activate lambda_new_model   # change to whatever env name you used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Fine-tuning ${MODEL_NAME} on ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/finetune_new_model.py" \
    --model_name "${MODEL_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --seed "${SEED}" \
    ${PRECISION} \
    ${TRUST_REMOTE_CODE}

echo "============================================================"
echo "Done. Checkpoint at: ${OUTPUT_DIR}"
echo "Run inference next with: sbatch run_inference.slurm"
echo "============================================================"

# ─── Job-array template (uncomment to fine-tune all 3 window sizes at once) ─
# To submit one job per window size:
#   1. Remove the single DATASET_DIR/OUTPUT_DIR lines above
#   2. Uncomment the array directive at the top: #SBATCH --array=0-2
#   3. Uncomment this block and remove the python call above
#
# WINDOWS=("2k" "4k" "8k")
# MAX_LENGTHS=(512 512 512)        # GENA-LM has fixed 512 context; for NT-v2 use (2048 4096 8192)
# BATCH_SIZES=(16 8 4)
# WINDOW="${WINDOWS[${SLURM_ARRAY_TASK_ID}]}"
# MAX_LEN="${MAX_LENGTHS[${SLURM_ARRAY_TASK_ID}]}"
# BS="${BATCH_SIZES[${SLURM_ARRAY_TASK_ID}]}"
# DATASET_DIR="${DATASET_ROOT}/binary_segments_${WINDOW}"
# OUTPUT_DIR="${CHECKPOINTS_ROOT}/${MODEL_SHORT}/${WINDOW}"
