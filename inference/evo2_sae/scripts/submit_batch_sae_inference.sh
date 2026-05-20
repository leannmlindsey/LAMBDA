#!/bin/bash
#
# Batch SAE Inference Submission Script for Evo2
#
# This script submits multiple SLURM jobs for SAE inference on a list of input files.
#
# Usage:
#   ./submit_batch_sae_inference.sh \
#       --input_list /path/to/input_files.txt \
#       --output_dir /path/to/output_directory
#
# The input_list file should contain one input CSV path per line.
#

set -e

# Default values
MODEL="evo2_7b"
FEATURE_IDX="19746"
MAX_THRESHOLD="0.5"
MEAN_THRESHOLD="0.1"
FRACTION_THRESHOLD="0.3"
BATCH_SIZE="1"
SAVE_ACTIVATIONS="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_list)        INPUT_LIST="$2";        shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --model)             MODEL="$2";             shift 2 ;;
        --feature_idx)       FEATURE_IDX="$2";       shift 2 ;;
        --max_threshold)     MAX_THRESHOLD="$2";     shift 2 ;;
        --mean_threshold)    MEAN_THRESHOLD="$2";    shift 2 ;;
        --fraction_threshold) FRACTION_THRESHOLD="$2"; shift 2 ;;
        --batch_size)        BATCH_SIZE="$2";        shift 2 ;;
        --save_activations)  SAVE_ACTIVATIONS="$2";  shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --input_list FILE         Text file with one input CSV path per line"
            echo "  --output_dir DIR          Directory to store all output files"
            echo ""
            echo "Optional arguments:"
            echo "  --model NAME              Evo2 model: evo2_7b, evo2_40b (default: evo2_7b)"
            echo "  --feature_idx N           SAE feature index (default: 19746)"
            echo "  --max_threshold F         Max activation threshold (default: 0.5)"
            echo "  --mean_threshold F        Mean activation threshold (default: 0.1)"
            echo "  --fraction_threshold F    Fraction firing threshold (default: 0.3)"
            echo "  --batch_size N            Batch size (default: 1)"
            echo "  --save_activations BOOL   Save .npy activation arrays (default: true)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: --input_list is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: --output_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate input list file exists
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Get the directory of this script (for finding run_sae_inference.sh)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_sae_inference.sh"

if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    echo "ERROR: Inference script not found: ${INFERENCE_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo "Evo2 SAE Batch Inference Submission"
echo "============================================================"
echo "Input list:          ${INPUT_LIST}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "Model:               ${MODEL}"
echo "Feature index:       ${FEATURE_IDX}"
echo "Max threshold:       ${MAX_THRESHOLD}"
echo "Mean threshold:      ${MEAN_THRESHOLD}"
echo "Fraction threshold:  ${FRACTION_THRESHOLD}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Save activations:    ${SAVE_ACTIVATIONS}"
echo "============================================================"
echo ""

# Count input files (non-empty, non-comment lines)
NUM_FILES=$(grep -c -v -E '^[[:space:]]*$|^[[:space:]]*#' "${INPUT_LIST}" || echo 0)
echo "Found ${NUM_FILES} input files to process"
echo ""

# Track submitted jobs
SUBMITTED_JOBS=()

# Read input list and submit jobs
while IFS= read -r INPUT_CSV || [ -n "${INPUT_CSV}" ]; do
    # Skip empty lines and comments
    if [[ -z "${INPUT_CSV}" ]] || [[ "${INPUT_CSV}" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    INPUT_CSV=$(echo "${INPUT_CSV}" | xargs)

    # Validate input file exists
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "WARNING: Input file not found, skipping: ${INPUT_CSV}"
        continue
    fi

    # Generate output filename
    INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)
    OUTPUT_CSV="${OUTPUT_DIR}/${INPUT_BASENAME}_sae_results.csv"

    echo "Submitting job for: ${INPUT_BASENAME}"
    echo "  Input:  ${INPUT_CSV}"
    echo "  Output: ${OUTPUT_CSV}"

    # Submit SLURM job
    JOB_ID=$(sbatch \
        --job-name="sae_${INPUT_BASENAME}" \
        --output="${OUTPUT_DIR}/slurm_${INPUT_BASENAME}_%j.out" \
        --error="${OUTPUT_DIR}/slurm_${INPUT_BASENAME}_%j.err" \
        --export=ALL,INPUT_CSV="${INPUT_CSV}",OUTPUT_CSV="${OUTPUT_CSV}",MODEL="${MODEL}",FEATURE_IDX="${FEATURE_IDX}",MAX_THRESHOLD="${MAX_THRESHOLD}",MEAN_THRESHOLD="${MEAN_THRESHOLD}",FRACTION_THRESHOLD="${FRACTION_THRESHOLD}",BATCH_SIZE="${BATCH_SIZE}",SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS}" \
        "${INFERENCE_SCRIPT}" | awk '{print $NF}')

    echo "  Job ID: ${JOB_ID}"
    SUBMITTED_JOBS+=("${JOB_ID}")
    echo ""

done < "${INPUT_LIST}"

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"
