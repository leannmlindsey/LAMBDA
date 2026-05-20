#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=phage_vs_gtdb_v3
#SBATCH --output=phage_vs_gtdb_v3_%A_%a.out
#SBATCH --error=phage_vs_gtdb_v3_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --array=0-9

# BLAST INPHARED phages against GTDB selected bacteria
# Version 3: max_target_seqs=2000 (increased from 500)
#
# Usage: sbatch blast_phage_vs_gtdb_selected_v3.slurm

# ============================================
# CONFIGURATION
# ============================================

# INPHARED phage genomes (query)
PHAGE_FASTA="/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa"

# GTDB selected bacteria database (subject)
BACTERIA_DB="/work/hdd/bfzj/llindsey1/gtdb_blastdb_selected/gtdb_bacteria_selected"

# Output directory (new directory for v3)
OUTPUT_DIR="/work/hdd/bfzj/llindsey1/prophage_blast_results_v3"

# Number of splits (must match --array above)
NUM_SPLITS=10

# BLAST parameters
MAX_TARGET_SEQS=2000    # Increased from 500
MIN_ALIGNMENT_LENGTH=200
NUM_THREADS=16

# ============================================

echo "=========================================="
echo "BLAST Phages vs GTDB Bacteria (v3)"
echo "=========================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Max target seqs: ${MAX_TARGET_SEQS}"
echo "Min alignment length: ${MIN_ALIGNMENT_LENGTH} bp"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate lambda_tools

# Create output and temp directories
mkdir -p ${OUTPUT_DIR}
SPLIT_DIR="${OUTPUT_DIR}/splits"
mkdir -p ${SPLIT_DIR}

# Step 1: Split the input FASTA (only task 0 does this, others wait)
SPLIT_DONE="${OUTPUT_DIR}/.split_done"

if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    echo "Task 0: Splitting input FASTA into ${NUM_SPLITS} chunks..."

    # Count sequences
    NUM_SEQS=$(grep -c "^>" ${PHAGE_FASTA})
    echo "  Total sequences: ${NUM_SEQS}"

    # Calculate sequences per split
    SEQS_PER_SPLIT=$(( (NUM_SEQS + NUM_SPLITS - 1) / NUM_SPLITS ))
    echo "  Sequences per split: ~${SEQS_PER_SPLIT}"

    # Split using awk
    awk -v n=${SEQS_PER_SPLIT} -v dir="${SPLIT_DIR}" '
        BEGIN { file_num = 0; seq_count = 0 }
        /^>/ {
            if (seq_count >= n) {
                file_num++
                seq_count = 0
            }
            seq_count++
        }
        { print > (dir "/split_" file_num ".fasta") }
    ' ${PHAGE_FASTA}

    echo "  Split complete!"
    ls -lh ${SPLIT_DIR}/

    # Signal that splitting is done
    touch ${SPLIT_DONE}
else
    # Wait for task 0 to finish splitting
    echo "Waiting for split to complete..."
    while [ ! -f ${SPLIT_DONE} ]; do
        sleep 10
    done
    echo "Split complete, proceeding..."
fi

# Step 2: Run BLAST on this task's chunk
SPLIT_FILE="${SPLIT_DIR}/split_${SLURM_ARRAY_TASK_ID}.fasta"
RAW_OUTPUT="${OUTPUT_DIR}/blast_raw_${SLURM_ARRAY_TASK_ID}.tsv"
FILTERED_OUTPUT="${OUTPUT_DIR}/blast_filtered_${SLURM_ARRAY_TASK_ID}.tsv"

if [ ! -f "${SPLIT_FILE}" ]; then
    echo "No split file for task ${SLURM_ARRAY_TASK_ID}, exiting."
    exit 0
fi

NUM_QUERIES=$(grep -c "^>" ${SPLIT_FILE})
echo ""
echo "Running BLAST on ${NUM_QUERIES} sequences..."
echo "  Input: ${SPLIT_FILE}"
echo "  Raw output: ${RAW_OUTPUT}"
echo "  Max target seqs: ${MAX_TARGET_SEQS}"

blastn \
    -db ${BACTERIA_DB} \
    -query ${SPLIT_FILE} \
    -out ${RAW_OUTPUT} \
    -outfmt "6 std" \
    -max_target_seqs ${MAX_TARGET_SEQS} \
    -evalue 1e-5 \
    -num_threads ${NUM_THREADS}

echo "  Raw hits: $(wc -l < ${RAW_OUTPUT})"

# Step 3: Filter by alignment length (column 4 >= MIN_ALIGNMENT_LENGTH)
echo ""
echo "Filtering for alignment length >= ${MIN_ALIGNMENT_LENGTH} bp..."
awk -v min_len=${MIN_ALIGNMENT_LENGTH} '$4 >= min_len' ${RAW_OUTPUT} > ${FILTERED_OUTPUT}

echo "  Filtered hits: $(wc -l < ${FILTERED_OUTPUT})"

echo ""
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
echo "End time: $(date)"
echo "=========================================="
