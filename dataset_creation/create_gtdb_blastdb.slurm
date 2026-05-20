#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gtdb_blastdb
#SBATCH --output=gtdb_blastdb_%j.out
#SBATCH --error=gtdb_blastdb_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g

# Create a BLAST database from GTDB bacterial genomes
# Handles nested directory structure and gzipped files
#
# Usage: sbatch create_gtdb_blastdb.slurm

# ============================================
# CONFIGURATION
# ============================================

# GTDB base directory (contains database/GCA/... and database/GCF/... structure)
GTDB_BASE="/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Output directory for BLAST database (using /work/hdd for space)
OUTPUT_DIR="/work/hdd/bfzj/llindsey1/gtdb_blastdb"

# Database name
DB_NAME="gtdb_bacteria"

# Temporary combined FASTA file (will be large - needs sufficient disk space)
COMBINED_FASTA="${OUTPUT_DIR}/gtdb_all_genomes.fna"

# Keep combined FASTA after database creation? (true/false)
KEEP_COMBINED="true"

# ============================================

echo "=========================================="
echo "Create GTDB Bacterial BLAST Database"
echo "=========================================="
echo "Start time: $(date)"
echo "GTDB base directory: ${GTDB_BASE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Database name: ${DB_NAME}"
echo "=========================================="

# Check input directory exists
if [ ! -d "${GTDB_BASE}/database" ]; then
    echo "Error: GTDB database directory not found: ${GTDB_BASE}/database"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Activate conda environment with BLAST installed
# First install blast if needed: conda install -c bioconda blast
source ~/.bashrc
conda activate lambda_tools

# Find all gzipped FASTA files
echo ""
echo "Finding all gzipped FASTA files..."
FASTA_FILES=$(find ${GTDB_BASE}/database -name "*.fna.gz" -type f)
NUM_FILES=$(echo "${FASTA_FILES}" | wc -l)
echo "  Found ${NUM_FILES} gzipped FASTA files"

if [ "${NUM_FILES}" -eq 0 ]; then
    echo "Error: No .fna.gz files found in ${GTDB_BASE}/database"
    exit 1
fi

# Show sample files
echo ""
echo "Sample files:"
echo "${FASTA_FILES}" | head -3

# Concatenate all gzipped files into one FASTA
echo ""
echo "Concatenating all genomes (this may take a while)..."
echo "Output: ${COMBINED_FASTA}"

# Remove existing combined file if present
rm -f ${COMBINED_FASTA}

# Use parallel zcat for speed (if available), otherwise sequential
COUNTER=0
for FNA_GZ in ${FASTA_FILES}; do
    zcat "${FNA_GZ}" >> ${COMBINED_FASTA}
    COUNTER=$((COUNTER + 1))
    if [ $((COUNTER % 1000)) -eq 0 ]; then
        echo "  Processed ${COUNTER}/${NUM_FILES} files..."
    fi
done

echo "  Concatenation complete!"
echo "  Combined file size: $(ls -lh ${COMBINED_FASTA} | awk '{print $5}')"
echo "  Total sequences: $(grep -c "^>" ${COMBINED_FASTA})"

# Create BLAST database
echo ""
echo "Running makeblastdb..."
makeblastdb \
    -in ${COMBINED_FASTA} \
    -dbtype nucl \
    -out ${OUTPUT_DIR}/${DB_NAME} \
    -title "GTDB Bacterial Representative Genomes" \
    -parse_seqids

if [ $? -eq 0 ]; then
    echo ""
    echo "BLAST database created successfully!"

    # Optionally remove combined FASTA to save space
    if [ "${KEEP_COMBINED}" != "true" ]; then
        echo "Removing combined FASTA file to save space..."
        rm -f ${COMBINED_FASTA}
    fi

    echo ""
    echo "=========================================="
    echo "Done!"
    echo "End time: $(date)"
    echo "=========================================="
    echo ""
    echo "Database files:"
    ls -lh ${OUTPUT_DIR}/${DB_NAME}.*
    echo ""
    echo "To use this database with BLAST:"
    echo "  blastn -db ${OUTPUT_DIR}/${DB_NAME} -query <your_query.fasta> -out results.txt"
else
    echo ""
    echo "=========================================="
    echo "BLAST database creation failed!"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi
