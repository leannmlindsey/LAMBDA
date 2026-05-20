#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=bakta_bacteria
#SBATCH --output=bakta_bacteria_%A_%a.out
#SBATCH --error=bakta_bacteria_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --array=1-100%20

# Run Bakta annotation on prophage-free bacterial subset
#
# This is an array job that processes each genome in parallel.
# Array size = 100 (prophage-free bacteria with ZERO phage BLAST hits)
# %20 limits to 20 concurrent jobs
#
# Prerequisites: Run select_prophage_free_bacteria.sh first to create the accession list
#
# Usage:
#   conda activate bakta_env
#   sbatch run_bakta_bacteria.slurm

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Prophage-free 100 bacteria accession list (ZERO phage BLAST hits)
ACCESSION_LIST="${SCRIPT_DIR}/bacteria_only_prophage_free/prophage_free_100_accessions.txt"

# GTDB genome directory
GTDB_BASE="/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Bakta database path
BAKTA_DB="/u/llindsey1/scratch/bakta_db/db"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/bacteria_only_annotations"

# Number of threads per job
THREADS=8

# ============================================
# END CONFIGURATION
# ============================================

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Get the accession for this array task
ACCESSION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ACCESSION_LIST}")

if [ -z "${ACCESSION}" ]; then
    echo "Error: No accession found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "=========================================="
echo "Bakta Annotation - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Start time: $(date)"
echo "Accession: ${ACCESSION}"

# Function to find genome file in GTDB nested structure
find_genome_file() {
    local acc=$1
    local base_dir=$2

    # Remove GB_ or RS_ prefix if present
    local clean_acc="${acc#GB_}"
    clean_acc="${clean_acc#RS_}"

    # Extract parts: GCA_000000000.1 -> GCA, 000, 000, 000
    local prefix="${clean_acc:0:3}"  # GCA or GCF
    local num="${clean_acc:4:9}"     # 000000000
    local part1="${num:0:3}"         # 000
    local part2="${num:3:3}"         # 000
    local part3="${num:6:3}"         # 000

    # Construct path
    local genome_path="${base_dir}/database/${prefix}/${part1}/${part2}/${part3}"

    # Find the .fna.gz file
    local genome_file=$(find "${genome_path}" -name "${clean_acc}*_genomic.fna.gz" 2>/dev/null | head -1)

    echo "${genome_file}"
}

# Find the genome file
GENOME_FILE=$(find_genome_file "${ACCESSION}" "${GTDB_BASE}")

if [ -z "${GENOME_FILE}" ] || [ ! -f "${GENOME_FILE}" ]; then
    echo "Error: Genome file not found for ${ACCESSION}"
    echo "Searched in: ${GTDB_BASE}"
    exit 1
fi

echo "Genome file: ${GENOME_FILE}"

# Create temporary directory for this job
TEMP_DIR="${OUTPUT_DIR}/temp_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TEMP_DIR}"

# Decompress genome to temp directory
GENOME_BASENAME=$(basename "${GENOME_FILE}" .gz)
gunzip -c "${GENOME_FILE}" > "${TEMP_DIR}/${GENOME_BASENAME}"

# Output directory for this genome
GENOME_OUTPUT="${OUTPUT_DIR}/${ACCESSION}"
mkdir -p "${GENOME_OUTPUT}"

echo "Running Bakta..."

# Note: Conda environment is inherited from submission environment on Delta
# Submit with: conda activate bakta_env && sbatch run_bakta_bacteria.slurm

# Run Bakta (--force to overwrite if exists)
bakta \
    --db "${BAKTA_DB}" \
    --output "${GENOME_OUTPUT}" \
    --prefix "${ACCESSION}" \
    --threads ${THREADS} \
    --keep-contig-headers \
    --force \
    "${TEMP_DIR}/${GENOME_BASENAME}"

BAKTA_EXIT=$?

# Clean up temp directory
rm -rf "${TEMP_DIR}"

if [ ${BAKTA_EXIT} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Bakta completed successfully!"
    echo "Output: ${GENOME_OUTPUT}"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo "Error: Bakta failed for ${ACCESSION}"
    exit 1
fi
