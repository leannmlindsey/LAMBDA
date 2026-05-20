#!/bin/bash
#
# 02_submit_inference_jobs.sh
#
# Submits SLURM inference jobs for every (model, category, window) combination
# selected in paths.env. For each combination this script:
#   1. Builds an input-files list pointing at the CSVs to process
#   2. Resolves the checkpoint path for that model + window
#   3. Calls the corresponding inference/{model}/slurm_scripts/submit_batch_inference.sh
#      with --input_list / --output_dir / --model_path
#
# Each `submit_batch_inference.sh` (and the SAE equivalent for evo2_sae) submits
# one SLURM job per input CSV. So a full run with all 7 models × 3 categories ×
# 3 windows × N input files = O(N × 60) jobs.
#
# Re-running is safe: existing prediction outputs are not overwritten (the
# downstream inference scripts skip CSVs whose `_predictions.csv` already
# exists). To force a re-run, delete the relevant RESULTS_ROOT subdirectory.
#
# Usage:
#   bash 02_submit_inference_jobs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${SCRIPT_DIR}/paths.env" ]; then
    echo "ERROR: ${SCRIPT_DIR}/paths.env not found."
    exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

: "${DATASET_ROOT:?DATASET_ROOT not set}"
: "${CHECKPOINTS_ROOT:?CHECKPOINTS_ROOT not set}"
: "${RESULTS_ROOT:?RESULTS_ROOT not set}"
: "${LAMBDA_REPO_ROOT:?LAMBDA_REPO_ROOT not set}"

MODELS="${MODELS:-dnabert2 nucleotide_transformer_v2 caduceus prokbert megadna generanno evo2_sae}"
CATEGORIES="${CATEGORIES:-binary error_bias genome_wide}"
WINDOWS="${WINDOWS:-2k 4k 8k}"
# Directory of source FASTAs to segment for the genome_wide category. Defaults
# to ${DATASET_ROOT}/genome_wide_fastas (what 01_download_zenodo.sh produces);
# override in paths.env to point at your own FASTAs.
FASTA_DIR="${FASTA_DIR:-${DATASET_ROOT}/genome_wide_fastas}"

mkdir -p "${RESULTS_ROOT}"
INPUT_LISTS_DIR="${RESULTS_ROOT}/_input_lists"
mkdir -p "${INPUT_LISTS_DIR}"

# ---------------------------------------------------------------------------
# Auto-segment FASTAs into genome-wide CSVs if needed.
#
# If the user has selected the genome_wide category but the segmented CSV
# subdirectories under DATASET_ROOT don't exist yet, we run 00_segment_fasta.py
# on the FASTAs in FASTA_DIR to produce them. This is a no-op when the CSVs
# already exist (either because the user pre-built them or this script ran
# previously).
# ---------------------------------------------------------------------------
maybe_segment_fastas() {
    case " ${CATEGORIES} " in
        *" genome_wide "*) ;;
        *) return 0 ;;
    esac

    local need_segment=0
    for window in ${WINDOWS}; do
        local subdir
        case "${window}" in
            2k) subdir="genome_wide_segments_2k_1k" ;;
            4k) subdir="genome_wide_segments_4k_2k" ;;
            8k) subdir="genome_wide_segments_8k_4k" ;;
            *)  continue ;;
        esac
        if ! compgen -G "${DATASET_ROOT}/${subdir}/*.csv" > /dev/null; then
            need_segment=1
            break
        fi
    done

    if [ "${need_segment}" -eq 0 ]; then
        echo "[segment] genome-wide CSVs already exist in ${DATASET_ROOT}, skipping segmentation."
        return 0
    fi

    if [ ! -d "${FASTA_DIR}" ]; then
        echo "ERROR: genome_wide selected but no segmented CSVs and FASTA_DIR not found."
        echo "       Expected FASTAs in: ${FASTA_DIR}"
        echo "       Either provide a directory of bacterial FASTAs there, or remove"
        echo "       'genome_wide' from CATEGORIES in paths.env."
        exit 1
    fi

    echo "[segment] running 00_segment_fasta.py on ${FASTA_DIR}"
    python "${SCRIPT_DIR}/00_segment_fasta.py" \
        --fasta-dir "${FASTA_DIR}" \
        --output-root "${DATASET_ROOT}" \
        --windows ${WINDOWS}
    echo "[segment] done. Segmented CSVs written under ${DATASET_ROOT}/genome_wide_segments_*."
    echo
}

maybe_segment_fastas

# ---------------------------------------------------------------------------
# Map a (category, window) pair to the directory under DATASET_ROOT that
# contains the CSVs (binary/error_bias) or per-genome CSV subdirectory
# (genome_wide).
# ---------------------------------------------------------------------------
dataset_subdir_for() {
    local category="$1"
    local window="$2"
    case "${category}" in
        binary)
            echo "binary_segments_${window}"
            ;;
        error_bias)
            echo "error_and_bias_${window}"
            ;;
        genome_wide)
            # Convention from the Zenodo layout: genome_wide_segments_{window}_{stride}
            case "${window}" in
                2k) echo "genome_wide_segments_2k_1k" ;;
                4k) echo "genome_wide_segments_4k_2k" ;;
                8k) echo "genome_wide_segments_8k_4k" ;;
                *)  echo ""; return 1 ;;
            esac
            ;;
        *)
            echo ""; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Build the input file list for a given (category, window). Writes a text
# file with one CSV path per line into INPUT_LISTS_DIR and echoes the path.
# ---------------------------------------------------------------------------
build_input_list() {
    local category="$1"
    local window="$2"
    local subdir
    subdir="$(dataset_subdir_for "${category}" "${window}")"
    local data_dir="${DATASET_ROOT}/${subdir}"
    local list_file="${INPUT_LISTS_DIR}/${category}_${window}.txt"

    if [ ! -d "${data_dir}" ]; then
        echo "" # signal "skip"
        return 0
    fi

    # binary / error_bias = a small set of CSVs at the top of the subdir.
    # genome_wide = one CSV per genome, possibly under per_genome/ or similar.
    if [ "${category}" = "genome_wide" ]; then
        find "${data_dir}" -maxdepth 2 -type f -name "*.csv" | sort > "${list_file}"
    else
        find "${data_dir}" -maxdepth 1 -type f -name "*.csv" | sort > "${list_file}"
    fi

    if [ ! -s "${list_file}" ]; then
        rm -f "${list_file}"
        echo ""
        return 0
    fi
    echo "${list_file}"
}

# ---------------------------------------------------------------------------
# Resolve the per-model checkpoint path for a given window. Conventions for
# CHECKPOINTS_ROOT layout (matches what 01_download_zenodo.sh unpacks):
#   CHECKPOINTS_ROOT/<model>/<window>/{checkpoint,config files}
# ---------------------------------------------------------------------------
checkpoint_path_for() {
    local model="$1"
    local window="$2"
    echo "${CHECKPOINTS_ROOT}/${model}/${window}"
}

# ---------------------------------------------------------------------------
# Submit one (model, category, window) combination
# ---------------------------------------------------------------------------
submit_combination() {
    local model="$1"
    local category="$2"
    local window="$3"

    local input_list
    input_list="$(build_input_list "${category}" "${window}")"
    if [ -z "${input_list}" ]; then
        echo "[skip] ${model} / ${category} / ${window} — no input CSVs found"
        return 0
    fi

    local output_dir="${RESULTS_ROOT}/${model}/${category}/${window}"
    mkdir -p "${output_dir}"

    local checkpoint_dir
    checkpoint_dir="$(checkpoint_path_for "${model}" "${window}")"

    local model_dir="${LAMBDA_REPO_ROOT}/inference/${model}"
    local slurm_dir
    if [ -d "${model_dir}/slurm_scripts" ]; then
        slurm_dir="${model_dir}/slurm_scripts"
    elif [ -d "${model_dir}/scripts" ]; then
        slurm_dir="${model_dir}/scripts"
    else
        echo "[error] ${model}: no slurm_scripts/ or scripts/ in ${model_dir}"
        return 1
    fi

    echo "[submit] ${model} / ${category} / ${window}"
    echo "         input_list = ${input_list}"
    echo "         output_dir = ${output_dir}"
    echo "         checkpoint = ${checkpoint_dir}"

    case "${model}" in
        evo2_sae)
            # Evo2+SAE uses a fixed SAE feature (19746) — no model_path needed.
            local opts=(--input_list "${input_list}" --output_dir "${output_dir}")
            [ -n "${BATCH_SIZE:-}" ] && opts+=(--batch_size "${BATCH_SIZE}")
            bash "${slurm_dir}/submit_batch_sae_inference.sh" "${opts[@]}"
            ;;
        caduceus)
            # Caduceus needs --checkpoint and --config. The checkpoint_dir is
            # expected to contain `last.ckpt` and `model_config.json`.
            local opts=(
                --input_list "${input_list}"
                --output_dir "${output_dir}"
                --checkpoint "${checkpoint_dir}/last.ckpt"
                --config "${checkpoint_dir}/model_config.json"
            )
            [ -n "${BATCH_SIZE:-}" ] && opts+=(--batch_size "${BATCH_SIZE}")
            [ -n "${THRESHOLD:-}" ] && opts+=(--threshold "${THRESHOLD}")
            bash "${slurm_dir}/submit_batch_inference.sh" "${opts[@]}"
            ;;
        *)
            # DNABERT2 / NTv2 / ProkBERT / megaDNA / GENERanno all use
            # --model_path pointing at the checkpoint directory.
            local opts=(
                --input_list "${input_list}"
                --output_dir "${output_dir}"
                --model_path "${checkpoint_dir}"
            )
            [ -n "${BATCH_SIZE:-}" ] && opts+=(--batch_size "${BATCH_SIZE}")
            [ -n "${THRESHOLD:-}" ] && opts+=(--threshold "${THRESHOLD}")
            bash "${slurm_dir}/submit_batch_inference.sh" "${opts[@]}"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "============================================================"
echo "LAMBDA inference job submission"
echo "============================================================"
echo "DATASET_ROOT     = ${DATASET_ROOT}"
echo "CHECKPOINTS_ROOT = ${CHECKPOINTS_ROOT}"
echo "RESULTS_ROOT     = ${RESULTS_ROOT}"
echo "MODELS           = ${MODELS}"
echo "CATEGORIES       = ${CATEGORIES}"
echo "WINDOWS          = ${WINDOWS}"
echo

for model in ${MODELS}; do
    for category in ${CATEGORIES}; do
        for window in ${WINDOWS}; do
            submit_combination "${model}" "${category}" "${window}"
            echo
        done
    done
done

echo "============================================================"
echo "All combinations submitted. Track progress with:"
echo "  squeue -u \$USER"
echo "Outputs will appear in: ${RESULTS_ROOT}"
echo "============================================================"
