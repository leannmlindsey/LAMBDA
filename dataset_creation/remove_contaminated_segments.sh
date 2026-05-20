#!/bin/bash
# Remove segments from contaminated genomes from final datasets
#
# Usage: bash remove_contaminated_segments.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Genomes to remove (discovered via validation BLAST)
CONTAMINATED_GENOMES=(
    "GCF_039536325.1"
    "GCF_900142275.1"
)

# Dataset directories
DATASETS=(
    "${SCRIPT_DIR}/gtdb_dataset_filtered_v3"
    "${SCRIPT_DIR}/gtdb_dataset_filtered_v3_4k"
    "${SCRIPT_DIR}/gtdb_dataset_filtered_v3_8k"
)

echo "=========================================="
echo "Remove Contaminated Segments"
echo "=========================================="
echo ""

for DATASET_DIR in "${DATASETS[@]}"; do
    echo "Processing: ${DATASET_DIR}"

    for SPLIT in train dev test; do
        FASTA="${DATASET_DIR}/${SPLIT}_segments.fasta"
        TSV="${DATASET_DIR}/${SPLIT}_segments.tsv"

        if [ -f "${FASTA}" ]; then
            BEFORE=$(grep -c "^>" "${FASTA}")

            # Create backup
            cp "${FASTA}" "${FASTA}.bak"

            # Remove sequences from contaminated genomes
            # Uses awk to handle multi-line FASTA properly
            awk -v genomes="${CONTAMINATED_GENOMES[*]}" '
            BEGIN {
                split(genomes, arr, " ")
                for (i in arr) contaminated[arr[i]] = 1
            }
            /^>/ {
                # Check if header contains any contaminated genome
                skip = 0
                for (g in contaminated) {
                    if (index($0, g) > 0) {
                        skip = 1
                        break
                    }
                }
                printing = !skip
            }
            printing { print }
            ' "${FASTA}.bak" > "${FASTA}"

            AFTER=$(grep -c "^>" "${FASTA}")
            REMOVED=$((BEFORE - AFTER))

            echo "  ${SPLIT}: ${BEFORE} -> ${AFTER} (removed ${REMOVED})"

            # Clean up backup
            rm "${FASTA}.bak"
        fi

        # Also filter TSV if it exists
        if [ -f "${TSV}" ]; then
            cp "${TSV}" "${TSV}.bak"

            for GENOME in "${CONTAMINATED_GENOMES[@]}"; do
                grep -v "${GENOME}" "${TSV}.bak" > "${TSV}.tmp"
                mv "${TSV}.tmp" "${TSV}.bak"
            done
            mv "${TSV}.bak" "${TSV}"
        fi
    done
    echo ""
done

# Also update the accession lists
echo "Updating accession lists..."
ACCESSION_DIR="${SCRIPT_DIR}/gtdb_dataset_filtered_v3"

for SPLIT in train dev test; do
    ACCESSION_FILE="${ACCESSION_DIR}/${SPLIT}_accessions.txt"

    if [ -f "${ACCESSION_FILE}" ]; then
        BEFORE=$(wc -l < "${ACCESSION_FILE}")

        for GENOME in "${CONTAMINATED_GENOMES[@]}"; do
            grep -v "^${GENOME}$" "${ACCESSION_FILE}" > "${ACCESSION_FILE}.tmp"
            mv "${ACCESSION_FILE}.tmp" "${ACCESSION_FILE}"
        done

        AFTER=$(wc -l < "${ACCESSION_FILE}")
        REMOVED=$((BEFORE - AFTER))

        if [ ${REMOVED} -gt 0 ]; then
            echo "  ${SPLIT}_accessions.txt: removed ${REMOVED}"
        fi
    fi
done

# Add to contaminated list
CONTAMINATED_FILE="${ACCESSION_DIR}/contaminated_accessions.txt"
echo ""
echo "Updating contaminated_accessions.txt..."
for GENOME in "${CONTAMINATED_GENOMES[@]}"; do
    if ! grep -q "^${GENOME}$" "${CONTAMINATED_FILE}" 2>/dev/null; then
        echo "${GENOME}" >> "${CONTAMINATED_FILE}"
        echo "  Added ${GENOME}"
    fi
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
