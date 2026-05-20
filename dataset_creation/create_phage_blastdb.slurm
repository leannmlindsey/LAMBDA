#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=phage_blastdb
#SBATCH --output=phage_blastdb_%j.out
#SBATCH --error=phage_blastdb_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g

# Create a BLAST database from INPHARED phage genomes
#
# Usage: sbatch create_phage_blastdb.slurm

# ============================================
# CONFIGURATION
# ============================================

# Input FASTA file with all INPHARED phage genomes
INPUT_FASTA="/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa"

# Output directory for BLAST database
OUTPUT_DIR="/u/llindsey1/llindsey/black_and_white/data/inphared/blastdb"

# Database name
DB_NAME="inphared_phages"

# ============================================

echo "=========================================="
echo "Create INPHARED Phage BLAST Database"
echo "=========================================="
echo "Start time: $(date)"
echo "Input FASTA: ${INPUT_FASTA}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Database name: ${DB_NAME}"
echo "=========================================="

# Check input exists
if [ ! -f "${INPUT_FASTA}" ]; then
    echo "Error: Input FASTA not found: ${INPUT_FASTA}"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Activate conda environment with BLAST installed
# First install blast if needed: conda install -c bioconda blast
source ~/.bashrc
conda activate lambda_tools

# Show input file info
echo ""
echo "Input file info:"
echo "  Size: $(ls -lh ${INPUT_FASTA} | awk '{print $5}')"
echo "  Sequences: $(grep -c "^>" ${INPUT_FASTA})"
echo ""

# Create BLAST database
echo "Running makeblastdb..."
makeblastdb \
    -in ${INPUT_FASTA} \
    -dbtype nucl \
    -out ${OUTPUT_DIR}/${DB_NAME} \
    -title "INPHARED Phage Genomes" \
    -parse_seqids

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "BLAST database created successfully!"
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
