#!/bin/bash
#SBATCH --job-name=lambda_new_model_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=lambda_new_model_inf_%j.out
#SBATCH --error=lambda_new_model_inf_%j.err
#
# Run inference with a fine-tuned new gLM on a LAMBDA category × window
# combination, producing the predictions CSV in the schema expected by
# ../03_build_website_data.py.
#
# Edit the configuration block below, then submit with:
#   sbatch run_inference.slurm
#
# To run inference across every (category, window) combination, see the
# input-list loop at the bottom of this file.

set -euo pipefail

#####################################################################
# CONFIGURATION — edit this section
#####################################################################

# === Fine-tuned checkpoint ===
# Path to the output_dir from run_finetune.slurm.
MODEL_PATH="/path/to/checkpoints/gena_lm/2k"

# === Inputs ===
# Single CSV to run inference on. For LAMBDA the typical paths are:
#   ${DATASET_ROOT}/binary_segments_2k/test.csv
#   ${DATASET_ROOT}/error_and_bias_2k/test.csv
#   ${DATASET_ROOT}/genome_wide_segments_2k_1k/<assembly>.csv
INPUT_CSV="/path/to/LAMBDA/binary_segments_2k/test.csv"

# === Output ===
# IMPORTANT: filename must end in `_predictions.csv` for the aggregator to
# pick it up. The directory layout RESULTS_ROOT/<model>/<category>/<window>/
# is what 03_build_website_data.py expects.
OUTPUT_CSV="/path/to/results/gena_lm/binary/2k/test_predictions.csv"

# === Inference settings ===
MAX_LENGTH=512
BATCH_SIZE=32
PRECISION="--bf16"            # "--bf16" or "--fp16" or ""
THRESHOLD=0.5                 # pred_label threshold

# Set if the model needs trust_remote_code (DNABERT-2 yes; GENA-LM no).
TRUST_REMOTE_CODE=""

#####################################################################
# END CONFIGURATION
#####################################################################

module load conda
module load CUDA/12.8
source activate lambda_new_model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "============================================================"
echo "Inference on ${INPUT_CSV}"
echo "Model:   ${MODEL_PATH}"
echo "Output:  ${OUTPUT_CSV}"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

python "${SCRIPT_DIR}/inference_new_model.py" \
    --model_path "${MODEL_PATH}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --threshold "${THRESHOLD}" \
    --save_metrics \
    ${PRECISION} \
    ${TRUST_REMOTE_CODE}

echo "============================================================"
echo "Done. After all categories × windows are run, aggregate with:"
echo "  python ../03_build_website_data.py --predictions \${RESULTS_ROOT} --output \${RESULTS_ROOT}/aggregated"
echo "============================================================"

# ─── Multi-input loop (uncomment to process every category × window at once) ─
# Replace the single python call above with this loop body, and set the
# variables RESULTS_ROOT, DATASET_ROOT, MODEL_KEY (e.g., "gena_lm") to drive it.
#
# for window in 2k 4k 8k; do
#   for category in binary error_bias genome_wide; do
#     case "${category}" in
#       binary)       subdir="binary_segments_${window}";      inputs=(test.csv) ;;
#       error_bias)   subdir="error_and_bias_${window}";        inputs=(test.csv) ;;
#       genome_wide)
#         case "${window}" in
#           2k) subdir="genome_wide_segments_2k_1k" ;;
#           4k) subdir="genome_wide_segments_4k_2k" ;;
#           8k) subdir="genome_wide_segments_8k_4k" ;;
#         esac
#         inputs=( "${DATASET_ROOT}/${subdir}"/*.csv ) ;;
#     esac
#     for input in "${inputs[@]}"; do
#       [ -e "${DATASET_ROOT}/${subdir}/${input}" ] || [ -e "${input}" ] || continue
#       in_path="${DATASET_ROOT}/${subdir}/${input}"
#       [ -e "${input}" ] && in_path="${input}"
#       base=$(basename "${in_path}" .csv)
#       out_path="${RESULTS_ROOT}/${MODEL_KEY}/${category}/${window}/${base}_predictions.csv"
#       python "${SCRIPT_DIR}/inference_new_model.py" \
#           --model_path "${CHECKPOINTS_ROOT}/${MODEL_KEY}/${window}" \
#           --input_csv "${in_path}" \
#           --output_csv "${out_path}" \
#           --max_length "${MAX_LENGTH}" --batch_size "${BATCH_SIZE}" \
#           --threshold "${THRESHOLD}" --save_metrics ${PRECISION} ${TRUST_REMOTE_CODE}
#     done
#   done
# done
