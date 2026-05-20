#!/bin/bash
# Extract full genomes for Phage-Only and Bacteria-Only benchmark datasets
#
# Usage: bash extract_benchmark_genomes.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Source data
INPHARED_FASTA="/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa"
GTDB_GENOMES="/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Accession lists
PHAGE_TEST_ACCESSIONS="${SCRIPT_DIR}/inphared_dataset/test_accessions.txt"
# Use prophage-free bacteria (ZERO phage BLAST hits)
BACTERIA_100_ACCESSIONS="${SCRIPT_DIR}/bacteria_only_prophage_free/prophage_free_100_accessions.txt"

# Output directories
PHAGE_OUTPUT_DIR="${SCRIPT_DIR}/phage_only_genomes"
BACTERIA_OUTPUT_DIR="${SCRIPT_DIR}/bacteria_only_genomes"

echo "=========================================="
echo "Extract Benchmark Genomes"
echo "=========================================="
echo ""

# ============================================
# Phage-Only: Extract 869 test genomes
# ============================================
echo "=== Phage-Only Genomes (869 test set) ==="
echo ""

mkdir -p "${PHAGE_OUTPUT_DIR}"

if [ ! -f "${PHAGE_TEST_ACCESSIONS}" ]; then
    echo "Error: Phage test accessions not found: ${PHAGE_TEST_ACCESSIONS}"
else
    echo "Extracting phage genomes from INPHARED..."
    echo "  Source: ${INPHARED_FASTA}"
    echo "  Accessions: ${PHAGE_TEST_ACCESSIONS}"

    # Use seqtk if available, otherwise use awk
    if command -v seqtk &> /dev/null; then
        seqtk subseq "${INPHARED_FASTA}" "${PHAGE_TEST_ACCESSIONS}" > "${PHAGE_OUTPUT_DIR}/phage_only_869_genomes.fasta"
    else
        # Use awk to extract sequences (handles multi-line FASTA)
        awk 'NR==FNR {accessions[$1]=1; next}
             /^>/ {
                 # Extract accession from header (first word after >)
                 header = substr($0, 2)
                 split(header, parts, " ")
                 acc = parts[1]
                 # Also try without version number
                 split(acc, acc_parts, ".")
                 acc_base = acc_parts[1]
                 printing = (acc in accessions || acc_base in accessions)
             }
             printing {print}' "${PHAGE_TEST_ACCESSIONS}" "${INPHARED_FASTA}" > "${PHAGE_OUTPUT_DIR}/phage_only_869_genomes.fasta"
    fi

    # Count extracted sequences
    COUNT=$(grep -c "^>" "${PHAGE_OUTPUT_DIR}/phage_only_869_genomes.fasta" 2>/dev/null || echo "0")
    echo "  Extracted: ${COUNT} genomes"

    # Show file size
    SIZE=$(ls -lh "${PHAGE_OUTPUT_DIR}/phage_only_869_genomes.fasta" | awk '{print $5}')
    echo "  File size: ${SIZE}"
fi

echo ""

# ============================================
# Bacteria-Only: Copy 100 balanced genomes
# ============================================
echo "=== Bacteria-Only Genomes (100 balanced) ==="
echo ""

mkdir -p "${BACTERIA_OUTPUT_DIR}"

if [ ! -f "${BACTERIA_100_ACCESSIONS}" ]; then
    echo "Error: Bacteria accessions not found: ${BACTERIA_100_ACCESSIONS}"
else
    echo "Copying bacterial genomes..."
    echo "  Source: ${GTDB_GENOMES}"
    echo "  Accessions: ${BACTERIA_100_ACCESSIONS}"

    COPIED=0
    MISSING=0

    # Also create a combined FASTA
    > "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"

    while read -r ACCESSION; do
        [ -z "${ACCESSION}" ] && continue

        # GTDB genomes are in subdirectories based on accession prefix
        # e.g., GCF_000123456.1 -> GCF/000/123/456/GCF_000123456.1_genomic.fna.gz
        # But they might also be flat or in different structures

        # Try common patterns
        FOUND=0

        # Pattern 1: Flat directory with .fna.gz
        GENOME_FILE="${GTDB_GENOMES}/${ACCESSION}_genomic.fna.gz"
        if [ -f "${GENOME_FILE}" ]; then
            zcat "${GENOME_FILE}" >> "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
            FOUND=1
        fi

        # Pattern 2: Flat directory with .fna
        if [ ${FOUND} -eq 0 ]; then
            GENOME_FILE="${GTDB_GENOMES}/${ACCESSION}_genomic.fna"
            if [ -f "${GENOME_FILE}" ]; then
                cat "${GENOME_FILE}" >> "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
                FOUND=1
            fi
        fi

        # Pattern 3: Nested directory structure
        if [ ${FOUND} -eq 0 ]; then
            # Extract parts: GCF_000123456.1 -> GCF/000/123/456
            PREFIX=$(echo "${ACCESSION}" | cut -d'_' -f1)
            NUMBERS=$(echo "${ACCESSION}" | cut -d'_' -f2 | cut -d'.' -f1)
            if [ ${#NUMBERS} -ge 9 ]; then
                DIR1=$(echo "${NUMBERS}" | cut -c1-3)
                DIR2=$(echo "${NUMBERS}" | cut -c4-6)
                DIR3=$(echo "${NUMBERS}" | cut -c7-9)
                NESTED_PATH="${GTDB_GENOMES}/${PREFIX}/${DIR1}/${DIR2}/${DIR3}"

                GENOME_FILE="${NESTED_PATH}/${ACCESSION}_genomic.fna.gz"
                if [ -f "${GENOME_FILE}" ]; then
                    zcat "${GENOME_FILE}" >> "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
                    FOUND=1
                fi
            fi
        fi

        # Pattern 4: Search for file
        if [ ${FOUND} -eq 0 ]; then
            GENOME_FILE=$(find "${GTDB_GENOMES}" -name "${ACCESSION}*.fna*" -type f 2>/dev/null | head -1)
            if [ -n "${GENOME_FILE}" ]; then
                if [[ "${GENOME_FILE}" == *.gz ]]; then
                    zcat "${GENOME_FILE}" >> "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
                else
                    cat "${GENOME_FILE}" >> "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
                fi
                FOUND=1
            fi
        fi

        if [ ${FOUND} -eq 1 ]; then
            ((COPIED++))
        else
            echo "  Missing: ${ACCESSION}"
            ((MISSING++))
        fi

    done < "${BACTERIA_100_ACCESSIONS}"

    echo ""
    echo "  Copied: ${COPIED} genomes"
    echo "  Missing: ${MISSING} genomes"

    # Count sequences in combined file
    COUNT=$(grep -c "^>" "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta" 2>/dev/null || echo "0")
    echo "  Total contigs in combined file: ${COUNT}"

    # Show file size
    SIZE=$(ls -lh "${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta" | awk '{print $5}')
    echo "  File size: ${SIZE}"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Phage-Only genomes: ${PHAGE_OUTPUT_DIR}/phage_only_869_genomes.fasta"
echo "Bacteria-Only genomes: ${BACTERIA_OUTPUT_DIR}/bacteria_only_100_genomes.fasta"
echo ""
echo "Now run create_lambda_final.sh to include these in the archives."
echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
