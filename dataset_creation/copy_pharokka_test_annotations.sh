#!/bin/bash
# Copy Pharokka annotations for test set phages to a separate directory
# This prepares the Phage-Only benchmark dataset for Zenodo upload
#
# Usage: bash copy_pharokka_test_annotations.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Input
TEST_ACCESSIONS="${SCRIPT_DIR}/inphared_dataset/test_accessions.txt"
PHAROKKA_DIR="/projects/bfzj/llindsey1/black_and_white/data/inphared/pharokka"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/phage_only_annotations"

echo "=========================================="
echo "Copy Pharokka Annotations for Test Set"
echo "=========================================="
echo ""
echo "Test accessions: ${TEST_ACCESSIONS}"
echo "Pharokka source: ${PHAROKKA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check inputs exist
if [ ! -f "${TEST_ACCESSIONS}" ]; then
    echo "Error: Test accessions file not found: ${TEST_ACCESSIONS}"
    exit 1
fi

if [ ! -d "${PHAROKKA_DIR}" ]; then
    echo "Error: Pharokka directory not found: ${PHAROKKA_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Count accessions
TOTAL=$(wc -l < "${TEST_ACCESSIONS}")
echo "Total test accessions: ${TOTAL}"
echo ""

# Copy annotations for each accession
COPIED=0
MISSING=0

while read -r ACCESSION; do
    # Skip empty lines
    [ -z "${ACCESSION}" ] && continue

    # Remove version number if present (e.g., MH572370.1 -> MH572370)
    ACCESSION_BASE=$(echo "${ACCESSION}" | sed 's/\.[0-9]*$//')

    # Look for files matching {ACCESSION}_genome.* pattern
    FILES=$(ls "${PHAROKKA_DIR}/${ACCESSION_BASE}_genome."* 2>/dev/null)

    if [ -n "${FILES}" ]; then
        cp ${PHAROKKA_DIR}/${ACCESSION_BASE}_genome.* "${OUTPUT_DIR}/"
        ((COPIED++))
    else
        # Also try with full accession (with version)
        FILES=$(ls "${PHAROKKA_DIR}/${ACCESSION}_genome."* 2>/dev/null)
        if [ -n "${FILES}" ]; then
            cp ${PHAROKKA_DIR}/${ACCESSION}_genome.* "${OUTPUT_DIR}/"
            ((COPIED++))
        else
            echo "  Missing: ${ACCESSION}"
            ((MISSING++))
        fi
    fi
done < "${TEST_ACCESSIONS}"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total test accessions: ${TOTAL}"
echo "Copied: ${COPIED}"
echo "Missing: ${MISSING}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Count files copied
if [ -d "${OUTPUT_DIR}" ]; then
    FILE_COUNT=$(ls "${OUTPUT_DIR}" | wc -l)
    echo "Total files in output: ${FILE_COUNT}"
    echo ""
    echo "Sample files:"
    ls "${OUTPUT_DIR}" | head -10
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
