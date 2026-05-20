#!/bin/bash
#
# 01_download_zenodo.sh
#
# Downloads the LAMBDA benchmark from Zenodo (10.5281/zenodo.19236553) and
# unpacks it into the directories configured in paths.env.
#
# Zenodo deposit layout (what this script expects to find):
#
#   lambda_dataset.tar.gz       — pre-built segment CSVs + ground truth:
#       binary_segments_{2k,4k,8k}/{train,dev,test}.csv
#       error_and_bias_{2k,4k,8k}/test.csv
#       ground_truth.csv                 (prophage start/end per assembly)
#       taxonomy.csv                     (per-assembly taxonomy, optional)
#
#   lambda_fastas.tar.gz        — source FASTAs for the genome-wide test set:
#       genome_wide_fastas/<assembly>.fna
#       (Step 02 slides windows over these to produce genome_wide_segments_*.)
#
#   lambda_checkpoints.tar.gz   — fine-tuned model checkpoints (one per model × window):
#       <model>/<window>/...
#       e.g., dnabert2/2k/, nucleotide_transformer_v2/4k/, evo2_sae/8k/, ...
#
# After running, DATASET_ROOT will look like:
#   DATASET_ROOT/
#       binary_segments_{2k,4k,8k}/{train,dev,test}.csv
#       error_and_bias_{2k,4k,8k}/test.csv
#       genome_wide_fastas/<assembly>.fna
#       ground_truth.csv
#       taxonomy.csv
#
# and CHECKPOINTS_ROOT:
#   CHECKPOINTS_ROOT/<model>/<window>/...
#
# Usage:
#   cp paths.env.template paths.env       (if you haven't already)
#   vim paths.env                          (fill in DATASET_ROOT, CHECKPOINTS_ROOT)
#   bash 01_download_zenodo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${SCRIPT_DIR}/paths.env" ]; then
    echo "ERROR: ${SCRIPT_DIR}/paths.env not found."
    echo "Copy paths.env.template to paths.env and fill in the values."
    exit 1
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

: "${DATASET_ROOT:?DATASET_ROOT not set in paths.env}"
: "${CHECKPOINTS_ROOT:?CHECKPOINTS_ROOT not set in paths.env}"

ZENODO_RECORD="19236553"
ZENODO_BASE="https://zenodo.org/record/${ZENODO_RECORD}/files"

# Adjust these filenames to match the actual uploads when the deposit is created.
DATASET_ARCHIVE="lambda_dataset.tar.gz"
FASTAS_ARCHIVE="lambda_fastas.tar.gz"
CHECKPOINTS_ARCHIVE="lambda_checkpoints.tar.gz"

mkdir -p "${DATASET_ROOT}" "${CHECKPOINTS_ROOT}"

download_and_extract() {
    local archive_name="$1"
    local dest_root="$2"
    local marker="$3"     # File or directory that signals "already extracted"
    local url="${ZENODO_BASE}/${archive_name}"
    local archive_path="${dest_root}/${archive_name}"

    if [ -n "${marker}" ] && [ -e "${dest_root}/${marker}" ]; then
        echo "[skip] ${archive_name} — ${marker} already exists in ${dest_root}."
        return 0
    fi

    echo "[download] ${url}"
    echo "           -> ${archive_path}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 5 --continue-at - -o "${archive_path}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue -O "${archive_path}" "${url}"
    else
        echo "ERROR: neither curl nor wget is available"
        exit 1
    fi

    echo "[extract] ${archive_path} -> ${dest_root}"
    tar -xzf "${archive_path}" -C "${dest_root}"
    rm -f "${archive_path}"
}

echo "============================================================"
echo "Downloading LAMBDA dataset (pre-built CSVs + ground truth)"
echo "============================================================"
download_and_extract "${DATASET_ARCHIVE}" "${DATASET_ROOT}" "ground_truth.csv"

echo
echo "============================================================"
echo "Downloading LAMBDA test-set FASTAs (genome-wide source data)"
echo "============================================================"
download_and_extract "${FASTAS_ARCHIVE}" "${DATASET_ROOT}" "genome_wide_fastas"

echo
echo "============================================================"
echo "Downloading LAMBDA fine-tuned checkpoints"
echo "============================================================"
download_and_extract "${CHECKPOINTS_ARCHIVE}" "${CHECKPOINTS_ROOT}" ""

echo
echo "============================================================"
echo "Done. Contents:"
echo "============================================================"
echo "DATASET_ROOT     = ${DATASET_ROOT}"
ls -1 "${DATASET_ROOT}" | sed 's/^/  /'
echo "CHECKPOINTS_ROOT = ${CHECKPOINTS_ROOT}"
ls -1 "${CHECKPOINTS_ROOT}" | sed 's/^/  /'
echo
echo "Next: bash 02_submit_inference_jobs.sh"
echo "(genome-wide CSVs will be auto-generated from the FASTAs on first run.)"
