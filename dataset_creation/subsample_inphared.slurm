#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=inphared_subsample
#SBATCH --output=inphared_subsample_%j.out
#SBATCH --error=inphared_subsample_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g

# Subsample segments from INPHARED phage genomes for train/dev/test splits
#
# Usage: sbatch subsample_inphared.slurm

# ============================================
# CONFIGURATION - Modify these paths as needed
# ============================================

# Directory containing the scripts
SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Input: INPHARED data directory (contains the genomes FASTA file)
INPHARED_DATA="/u/llindsey1/llindsey/black_and_white/data/inphared"

# Output: INPHARED dataset directory (contains accession lists and will receive output)
INPHARED_DATASET="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset"

# Path to the INPHARED genomes FASTA file
GENOMES_FASTA="${INPHARED_DATA}/14Apr2025_genomes.fa"

# Subsampling parameters
SEGMENT_LENGTH=2000
MIN_LENGTH=5000
SAMPLE_PER_BP=10000
SEED=42

# ============================================
# END CONFIGURATION
# ============================================

echo "=========================================="
echo "INPHARED Segment Subsampling"
echo "=========================================="
echo "Start time: $(date)"
echo "Script directory: ${SCRIPT_DIR}"
echo "INPHARED data: ${INPHARED_DATA}"
echo "Output directory: ${INPHARED_DATASET}"
echo "Genomes FASTA: ${GENOMES_FASTA}"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "Min genome length: ${MIN_LENGTH}"
echo "Sample per bp: ${SAMPLE_PER_BP}"
echo "Seed: ${SEED}"
echo "=========================================="

# Check that required files exist
if [ ! -f "${GENOMES_FASTA}" ]; then
    echo "Error: Genomes FASTA file not found: ${GENOMES_FASTA}"
    exit 1
fi

# Load any required modules (uncomment/modify as needed)
# module load python/3.9
# conda activate lambda_tools

# Process each split
for SPLIT in train dev test; do
    echo ""
    echo "=========================================="
    echo "Processing ${SPLIT} split..."
    echo "=========================================="

    ACCESSION_FILE="${INPHARED_DATASET}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${INPHARED_DATASET}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${INPHARED_DATASET}/${SPLIT}_segments.tsv"

    if [ ! -f "${ACCESSION_FILE}" ]; then
        echo "Warning: Accession file not found: ${ACCESSION_FILE}"
        echo "Skipping ${SPLIT} split..."
        continue
    fi

    echo "Accession file: ${ACCESSION_FILE}"
    echo "Output FASTA: ${OUTPUT_FASTA}"
    echo "Output metadata: ${OUTPUT_METADATA}"

    python ${SCRIPT_DIR}/subsample_segments.py \
        --input-fasta "${GENOMES_FASTA}" \
        --accession-list "${ACCESSION_FILE}" \
        --output "${OUTPUT_FASTA}" \
        --output-metadata "${OUTPUT_METADATA}" \
        --segment-length ${SEGMENT_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --sample-per-bp ${SAMPLE_PER_BP} \
        --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "${SPLIT} split completed successfully!"
    else
        echo "Error: ${SPLIT} split failed!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All splits completed!"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -lh ${INPHARED_DATASET}/*_segments.fasta ${INPHARED_DATASET}/*_segments.tsv 2>/dev/null
