#!/bin/bash
# Create CSV files for GC-content control (shuffled) datasets
#
# These are shuffled versions of the test segments - same GC content
# but destroyed sequence patterns. Used to test if models are using
# sequence patterns vs just GC content.
#
# Usage: bash create_gc_control_csvs.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
GC_DIR="${SCRIPT_DIR}/shuffled_gc_control"

echo "=========================================="
echo "Create GC Control CSV Files"
echo "=========================================="
echo ""

# Function to convert FASTA to CSV
fasta_to_csv() {
    local FASTA="$1"
    local CSV="$2"
    local LABEL="$3"
    local SOURCE="$4"

    if [ ! -f "${FASTA}" ]; then
        echo "  Skipping: ${FASTA} not found"
        return
    fi

    echo "  Converting $(basename ${FASTA}) -> $(basename ${CSV})"

    # Use awk to convert FASTA to CSV
    awk -v label="${LABEL}" -v source="${SOURCE}" '
    BEGIN {
        print "segment_id,sequence,label,source"
    }
    /^>/ {
        if (seq != "") {
            print id "," seq "," label "," source
        }
        id = substr($1, 2)  # Remove > and take first word
        seq = ""
        next
    }
    {
        seq = seq $0
    }
    END {
        if (seq != "") {
            print id "," seq "," label "," source
        }
    }
    ' "${FASTA}" > "${CSV}"

    LINES=$(wc -l < "${CSV}")
    SIZE=$(ls -lh "${CSV}" | awk '{print $5}')
    echo "    Done: $((LINES-1)) sequences, ${SIZE}"
}

# Process each segment length
for SEG_LEN in 2k 4k 8k; do
    echo ""
    echo "=== ${SEG_LEN} segments ==="

    # Shuffled phage (original label would be 1, but these are shuffled controls)
    PHAGE_FASTA="${GC_DIR}/inphared_${SEG_LEN}_test_shuffled.fasta"
    PHAGE_CSV="${GC_DIR}/inphared_${SEG_LEN}_test_shuffled.csv"
    fasta_to_csv "${PHAGE_FASTA}" "${PHAGE_CSV}" "1" "inphared_shuffled"

    # Shuffled bacteria (original label would be 0, but these are shuffled controls)
    BACTERIA_FASTA="${GC_DIR}/gtdb_${SEG_LEN}_test_shuffled.fasta"
    BACTERIA_CSV="${GC_DIR}/gtdb_${SEG_LEN}_test_shuffled.csv"
    fasta_to_csv "${BACTERIA_FASTA}" "${BACTERIA_CSV}" "0" "gtdb_shuffled"

    # Also create a combined CSV for this segment length
    COMBINED_CSV="${GC_DIR}/gc_control_${SEG_LEN}_test.csv"
    echo "  Creating combined: gc_control_${SEG_LEN}_test.csv"

    # Header
    echo "segment_id,sequence,label,source" > "${COMBINED_CSV}"

    # Append phage (skip header)
    if [ -f "${PHAGE_CSV}" ]; then
        tail -n +2 "${PHAGE_CSV}" >> "${COMBINED_CSV}"
    fi

    # Append bacteria (skip header)
    if [ -f "${BACTERIA_CSV}" ]; then
        tail -n +2 "${BACTERIA_CSV}" >> "${COMBINED_CSV}"
    fi

    LINES=$(wc -l < "${COMBINED_CSV}")
    SIZE=$(ls -lh "${COMBINED_CSV}" | awk '{print $5}')
    echo "    Done: $((LINES-1)) sequences, ${SIZE}"
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Created files in ${GC_DIR}:"
echo ""
ls -lh ${GC_DIR}/*.csv 2>/dev/null | awk '{print "  " $9 " " $5}' | sed "s|${GC_DIR}/||"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
