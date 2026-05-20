#!/bin/bash
#SBATCH --job-name=evo2_sae_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=evo2_sae_inf_%j.out
#SBATCH --error=evo2_sae_inf_%j.err

# Batch script for Evo2 SAE inference on a single input CSV
# Usage: sbatch run_sae_inference.sh
#
# Required environment variables:
#   INPUT_CSV: Path to CSV file with segment_id, sequence, label, source columns
#   OUTPUT_CSV: Path for output CSV

echo "============================================================"
echo "Evo2 SAE Inference"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
MODEL=${MODEL:-evo2_7b}
FEATURE_IDX=${FEATURE_IDX:-19746}
MAX_THRESHOLD=${MAX_THRESHOLD:-0.5}
MEAN_THRESHOLD=${MEAN_THRESHOLD:-0.1}
FRACTION_THRESHOLD=${FRACTION_THRESHOLD:-0.3}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_ACTIVATIONS=${SAVE_ACTIVATIONS:-true}

# Validate required parameters
if [ -z "${INPUT_CSV}" ]; then
    echo "ERROR: INPUT_CSV is not set"
    exit 1
fi

if [ -z "${OUTPUT_CSV}" ]; then
    # Default: same dir as input, with _sae_results suffix
    OUTPUT_CSV="${INPUT_CSV%.csv}_sae_results.csv"
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Create output directory if needed
mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL}"
echo "  Feature index: ${FEATURE_IDX}"
echo "  Input CSV: ${INPUT_CSV}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "  Max threshold: ${MAX_THRESHOLD}"
echo "  Mean threshold: ${MEAN_THRESHOLD}"
echo "  Fraction threshold: ${FRACTION_THRESHOLD}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Save activations: ${SAVE_ACTIVATIONS}"
echo "============================================================"
echo ""

# Build save_activations flag
SAVE_ACT_FLAG=""
if [ "${SAVE_ACTIVATIONS}" == "true" ]; then
    SAVE_ACT_FLAG="--save_activations"
fi

# Run SAE inference
python src/sae_inference.py \
    --input_csv "${INPUT_CSV}" \
    --output "${OUTPUT_CSV}" \
    --model "${MODEL}" \
    --feature_idx ${FEATURE_IDX} \
    --max_threshold ${MAX_THRESHOLD} \
    --mean_threshold ${MEAN_THRESHOLD} \
    --fraction_threshold ${FRACTION_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    ${SAVE_ACT_FLAG}

# Calculate and display metrics
if [ -f "${OUTPUT_CSV}" ]; then
    python src/calculate_metrics.py --input "${OUTPUT_CSV}"
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Predictions saved to: ${OUTPUT_CSV}"
echo "============================================================"
