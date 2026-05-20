#!/bin/bash
# Create training-ready CSV files for all merged datasets
#
# This combines the merged FASTA files with labels to create
# CSV files with columns: segment_id, sequence, label (0/1), source
#
# Usage: bash create_all_training_csvs.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

echo "=========================================="
echo "Create Training CSV Files"
echo "=========================================="
echo ""

# Process both filtered and unfiltered
for DATASET_TYPE in filtered unfiltered; do
    echo "============================================"
    echo "Processing ${DATASET_TYPE} datasets"
    echo "============================================"
    echo ""

    MERGED_DIR="${SCRIPT_DIR}/merged_datasets_${DATASET_TYPE}"

    # Process each segment length
    for SEG_LEN in 2k 4k 8k; do
        echo "--- ${SEG_LEN} segments ---"

        INPUT_DIR="${MERGED_DIR}/${SEG_LEN}"

        if [ ! -d "${INPUT_DIR}" ]; then
            echo "  Skipping: ${INPUT_DIR} not found"
            echo ""
            continue
        fi

        # Process each split
        for SPLIT in train dev test; do
            FASTA="${INPUT_DIR}/${SPLIT}_merged.fasta"
            LABELS="${INPUT_DIR}/${SPLIT}_labels.tsv"
            OUTPUT="${INPUT_DIR}/${SPLIT}.csv"

            if [ ! -f "${FASTA}" ]; then
                echo "  Skipping ${SPLIT}: FASTA not found"
                continue
            fi

            if [ ! -f "${LABELS}" ]; then
                echo "  Skipping ${SPLIT}: Labels not found"
                continue
            fi

            echo "  Creating ${SPLIT}.csv..."

            python ${SCRIPT_DIR}/create_training_csv.py \
                --fasta "${FASTA}" \
                --labels "${LABELS}" \
                --output "${OUTPUT}"

            if [ $? -eq 0 ]; then
                # Show file size and line count
                LINES=$(wc -l < "${OUTPUT}")
                SIZE=$(ls -lh "${OUTPUT}" | awk '{print $5}')
                echo "    Done: ${LINES} rows, ${SIZE}"
            else
                echo "    ERROR: Failed to create ${OUTPUT}"
            fi
        done

        echo ""
    done
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Created CSV files in:"
echo ""

for DATASET_TYPE in filtered unfiltered; do
    MERGED_DIR="${SCRIPT_DIR}/merged_datasets_${DATASET_TYPE}"
    echo "${DATASET_TYPE}:"

    for SEG_LEN in 2k 4k 8k; do
        INPUT_DIR="${MERGED_DIR}/${SEG_LEN}"

        if [ -d "${INPUT_DIR}" ]; then
            echo "  ${SEG_LEN}:"
            for SPLIT in train dev test; do
                CSV="${INPUT_DIR}/${SPLIT}.csv"
                if [ -f "${CSV}" ]; then
                    LINES=$(wc -l < "${CSV}")
                    SIZE=$(ls -lh "${CSV}" | awk '{print $5}')
                    printf "    %-8s %8d rows  %8s\n" "${SPLIT}.csv" "$((LINES-1))" "${SIZE}"
                fi
            done
        fi
    done
    echo ""
done

echo "=========================================="
echo "Done!"
echo "=========================================="
