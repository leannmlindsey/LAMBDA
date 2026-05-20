#!/bin/bash
# Split an input list file into N roughly equal parts
# Usage: bash scripts/split_list.sh <input_file> <num_splits>
# Example: bash scripts/split_list.sh scripts/genome_list_2k.txt 5

INPUT_FILE="${1}"
NUM_SPLITS="${2:-5}"

if [ -z "${INPUT_FILE}" ] || [ ! -f "${INPUT_FILE}" ]; then
    echo "Usage: bash split_list.sh <input_file> <num_splits>"
    exit 1
fi

BASENAME=$(basename "${INPUT_FILE}" .txt)
DIR=$(dirname "${INPUT_FILE}")

# Filter out empty lines and comments
TOTAL=$(grep -c -v -E '^[[:space:]]*$|^[[:space:]]*#' "${INPUT_FILE}")
PER_SPLIT=$(( (TOTAL + NUM_SPLITS - 1) / NUM_SPLITS ))

echo "Input:  ${INPUT_FILE}"
echo "Total lines: ${TOTAL}"
echo "Splits: ${NUM_SPLITS} (${PER_SPLIT} per split)"
echo ""

grep -v -E '^[[:space:]]*$|^[[:space:]]*#' "${INPUT_FILE}" | \
    split -l ${PER_SPLIT} -d -a 1 - "${DIR}/${BASENAME}_"

# Rename to .txt
for f in "${DIR}/${BASENAME}_"[0-9]; do
    mv "$f" "${f}.txt"
    COUNT=$(wc -l < "${f}.txt" | tr -d ' ')
    echo "  Created: ${f}.txt  (${COUNT} lines)"
done
