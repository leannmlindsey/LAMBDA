#!/bin/bash
#SBATCH --job-name=prokbert_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=prokbert_inf_%j.out
#SBATCH --error=prokbert_inf_%j.err

# Biowulf batch script for ProkBERT inference
# Usage: sbatch run_inference.sh
#
# Required environment variables:
#   INPUT_CSV: Path to CSV file with 'sequence' column
#   MODEL_PATH: Path to fine-tuned model checkpoint directory

echo "============================================================"
echo "ProkBERT Inference"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (suppress errors for non-Biowulf systems)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true

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
BASE_MODEL=${BASE_MODEL:-neuralbioinfo/prokbert-mini}

# Validate required parameters
if [ -z "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV is not set"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set output path
if [ -z "${OUTPUT_CSV}" ]; then
    OUTPUT_CSV="${INPUT_CSV%.csv}_predictions.csv"
fi

# Set output directory (extract from OUTPUT_CSV path)
OUTPUT_DIR=$(dirname "${OUTPUT_CSV}")

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  Base model: ${BASE_MODEL}"
echo "  Input CSV: ${INPUT_CSV}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "============================================================"
echo ""

# Run inference using the existing inference_lambda.py script
python inference_lambda.py \
    --checkpoint_path="${MODEL_PATH}" \
    --base_model="${BASE_MODEL}" \
    --dataset_file="${INPUT_CSV}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --output_dir="${OUTPUT_DIR}" \
    --output_file="$(basename ${OUTPUT_CSV})" \
    --save_metrics

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Predictions saved to: ${OUTPUT_CSV}"
echo "============================================================"
