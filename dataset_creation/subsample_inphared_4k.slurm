#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=inphared_4k
#SBATCH --output=inphared_subsample_4k_%j.out
#SBATCH --error=inphared_subsample_4k_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g

# Subsample 4k segments from INPHARED phage genomes for train/dev/test splits
#
# Usage: sbatch subsample_inphared_4k.slurm

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
INPHARED_DATA="/u/llindsey1/llindsey/black_and_white/data/inphared"
INPHARED_ACCESSIONS="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset"
OUTPUT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset_4k"
GENOMES_FASTA="${INPHARED_DATA}/14Apr2025_genomes.fa"

# Subsampling parameters
SEGMENT_LENGTH=4000
MIN_LENGTH=5000
SAMPLE_PER_BP=10000
SEED=42

# ============================================

echo "=========================================="
echo "INPHARED 4k Segment Subsampling"
echo "=========================================="
echo "Start time: $(date)"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

if [ ! -f "${GENOMES_FASTA}" ]; then
    echo "Error: Genomes FASTA file not found: ${GENOMES_FASTA}"
    exit 1
fi

for SPLIT in train dev test; do
    echo ""
    echo "Processing ${SPLIT} split..."

    ACCESSION_FILE="${INPHARED_ACCESSIONS}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${OUTPUT_DIR}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${OUTPUT_DIR}/${SPLIT}_segments.tsv"

    if [ ! -f "${ACCESSION_FILE}" ]; then
        echo "Warning: Accession file not found: ${ACCESSION_FILE}"
        continue
    fi

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
        echo "${SPLIT} completed!"
    else
        echo "Error: ${SPLIT} failed!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Done! End time: $(date)"
echo "=========================================="
ls -lh ${OUTPUT_DIR}/*_segments.fasta
