#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gtdb_4k
#SBATCH --output=gtdb_subsample_4k_%j.out
#SBATCH --error=gtdb_subsample_4k_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g

# Subsample 4k segments from GTDB bacterial genomes for train/dev/test splits
#
# Usage: sbatch subsample_gtdb_4k.slurm

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
GTDB_BASE="/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"
GTDB_ACCESSIONS="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset"
OUTPUT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_4k"

THREADS=16
SEGMENT_LENGTH=4000
MIN_LENGTH=5000
SEED=42

# Match bacterial segments to phage segment counts
# UPDATE THESE after running INPHARED 4k subsampling
TOTAL_SEGMENTS_TRAIN=27000
TOTAL_SEGMENTS_DEV=3400
TOTAL_SEGMENTS_TEST=3400

# ============================================

echo "=========================================="
echo "GTDB 4k Segment Subsampling"
echo "=========================================="
echo "Start time: $(date)"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "Target segments - Train: ${TOTAL_SEGMENTS_TRAIN}, Dev: ${TOTAL_SEGMENTS_DEV}, Test: ${TOTAL_SEGMENTS_TEST}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

if [ ! -d "${GTDB_BASE}/database" ]; then
    echo "Error: GTDB database directory not found: ${GTDB_BASE}/database"
    exit 1
fi

for SPLIT in train dev test; do
    echo ""
    echo "Processing ${SPLIT} split..."

    ACCESSION_FILE="${GTDB_ACCESSIONS}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${OUTPUT_DIR}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${OUTPUT_DIR}/${SPLIT}_segments.tsv"

    if [ "${SPLIT}" == "train" ]; then
        TARGET_SEGMENTS=${TOTAL_SEGMENTS_TRAIN}
    elif [ "${SPLIT}" == "dev" ]; then
        TARGET_SEGMENTS=${TOTAL_SEGMENTS_DEV}
    else
        TARGET_SEGMENTS=${TOTAL_SEGMENTS_TEST}
    fi

    if [ ! -f "${ACCESSION_FILE}" ]; then
        echo "Warning: Accession file not found: ${ACCESSION_FILE}"
        continue
    fi

    echo "Target segments: ${TARGET_SEGMENTS}"

    python ${SCRIPT_DIR}/subsample_gtdb_segments.py \
        --gtdb-dir "${GTDB_BASE}" \
        --accession-list "${ACCESSION_FILE}" \
        --output "${OUTPUT_FASTA}" \
        --output-metadata "${OUTPUT_METADATA}" \
        --segment-length ${SEGMENT_LENGTH} \
        --min-length ${MIN_LENGTH} \
        --threads ${THREADS} \
        --seed ${SEED} \
        --total-segments ${TARGET_SEGMENTS}

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
