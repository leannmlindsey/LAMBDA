#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_bakta
#SBATCH --output=test_bakta_%j.out
#SBATCH --error=test_bakta_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g

# Test Bakta on a single genome before running full array job
#
# Usage:
#   conda activate bakta_env
#   sbatch test_bakta_single.slurm

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Test accessions list (will use first genome)
ACCESSION_LIST="${SCRIPT_DIR}/gtdb_dataset_filtered_v3/test_accessions.txt"

# GTDB genome directory
GTDB_BASE="/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Bakta database path
BAKTA_DB="/u/llindsey1/scratch/bakta_db/db"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/bacteria_only_annotations_test"

THREADS=8

# ============================================
# TEST RUN
# ============================================

mkdir -p "${OUTPUT_DIR}"

# Get the first accession for testing
ACCESSION=$(head -1 "${ACCESSION_LIST}")

echo "=========================================="
echo "Bakta Test Run"
echo "=========================================="
echo "Start time: $(date)"
echo "Test accession: ${ACCESSION}"
echo "Bakta DB: ${BAKTA_DB}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Function to find genome file
find_genome_file() {
    local acc=$1
    local base_dir=$2

    local clean_acc="${acc#GB_}"
    clean_acc="${clean_acc#RS_}"

    local prefix="${clean_acc:0:3}"
    local num="${clean_acc:4:9}"
    local part1="${num:0:3}"
    local part2="${num:3:3}"
    local part3="${num:6:3}"

    local genome_path="${base_dir}/database/${prefix}/${part1}/${part2}/${part3}"
    local genome_file=$(find "${genome_path}" -name "${clean_acc}*_genomic.fna.gz" 2>/dev/null | head -1)

    echo "${genome_file}"
}

# Find genome file
echo "Step 1: Finding genome file..."
GENOME_FILE=$(find_genome_file "${ACCESSION}" "${GTDB_BASE}")

if [ -z "${GENOME_FILE}" ] || [ ! -f "${GENOME_FILE}" ]; then
    echo "ERROR: Genome file not found for ${ACCESSION}"
    exit 1
fi
echo "  Found: ${GENOME_FILE}"
echo ""

# Decompress
echo "Step 2: Decompressing genome..."
TEMP_DIR="${OUTPUT_DIR}/temp_test"
mkdir -p "${TEMP_DIR}"
GENOME_BASENAME=$(basename "${GENOME_FILE}" .gz)
gunzip -c "${GENOME_FILE}" > "${TEMP_DIR}/${GENOME_BASENAME}"
echo "  Decompressed to: ${TEMP_DIR}/${GENOME_BASENAME}"
echo ""

# Check Bakta is available
echo "Step 3: Checking Bakta installation..."
which bakta
bakta --version
echo ""

# Check Bakta database
echo "Step 4: Checking Bakta database..."
if [ -d "${BAKTA_DB}" ]; then
    echo "  Database exists: ${BAKTA_DB}"
    ls -la "${BAKTA_DB}" | head -5
else
    echo "ERROR: Bakta database not found: ${BAKTA_DB}"
    exit 1
fi
echo ""

# Run Bakta
echo "Step 5: Running Bakta..."
GENOME_OUTPUT="${OUTPUT_DIR}/${ACCESSION}"
mkdir -p "${GENOME_OUTPUT}"

bakta \
    --db "${BAKTA_DB}" \
    --output "${GENOME_OUTPUT}" \
    --prefix "${ACCESSION}" \
    --threads ${THREADS} \
    --keep-contig-headers \
    --force \
    "${TEMP_DIR}/${GENOME_BASENAME}"

BAKTA_EXIT=$?

# Clean up
rm -rf "${TEMP_DIR}"

echo ""
if [ ${BAKTA_EXIT} -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS! Bakta test completed."
    echo "=========================================="
    echo ""
    echo "Output files:"
    ls -la "${GENOME_OUTPUT}/"
    echo ""
    echo "You can now run the full array job."
else
    echo "=========================================="
    echo "FAILED! Bakta test failed with exit code ${BAKTA_EXIT}"
    echo "=========================================="
    echo "Check the error output above for details."
    exit 1
fi
