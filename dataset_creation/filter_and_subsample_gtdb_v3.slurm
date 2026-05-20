#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=filter_subsample_v3
#SBATCH --output=filter_subsample_v3_%j.out
#SBATCH --error=filter_subsample_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g

# Filter prophage-contaminated bacteria and subsample segments at 2k, 4k, 8k
# Version 3: Uses BLAST results with max_target_seqs=2000
#
# Prerequisites:
#   1. BLAST v3 results must be complete
#   2. Contig mapping exists (will copy from v2 if needed)
#
# Usage: sbatch filter_and_subsample_gtdb_v3.slurm

# ============================================
# CONFIGURATION
# ============================================

# Script directory
SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# BLAST results from v3 (max_target_seqs=2000)
BLAST_RESULTS_DIR="/work/hdd/bfzj/llindsey1/prophage_blast_results_v3"
BLAST_RESULTS="${BLAST_RESULTS_DIR}/blast_filtered_combined.tsv"

# Contig-to-genome mapping (copy from v2 if not present)
CONTIG_MAP="${BLAST_RESULTS_DIR}/contig_to_genome_map.tsv"
CONTIG_MAP_V2="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv"

# Original GTDB accession lists
GTDB_ACCESSIONS="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset"

# Base output directory for filtered datasets (v3)
OUTPUT_BASE="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# GTDB genomes directory
GTDB_BASE="/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Filtering thresholds
MIN_IDENTITY=90
MIN_ALIGNMENT_LENGTH=200

# Subsampling parameters
THREADS=16
MIN_LENGTH=5000
SEED=42

# Target segment counts (should match INPHARED counts)
TOTAL_SEGMENTS_TRAIN=27000
TOTAL_SEGMENTS_DEV=3400
TOTAL_SEGMENTS_TEST=3400

# ============================================

echo "=========================================="
echo "Filter and Subsample GTDB (Prophage-Free) v3"
echo "=========================================="
echo "Start time: $(date)"
echo "Using BLAST v3 results (max_target_seqs=2000)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate lambda_tools

# ============================================
# STEP 0: Prepare files
# ============================================
echo ""
echo "Step 0: Preparing files..."

# Combine BLAST results if not already done
if [ ! -f "${BLAST_RESULTS}" ]; then
    echo "  Combining filtered results from all array tasks..."
    cat ${BLAST_RESULTS_DIR}/blast_filtered_*.tsv > ${BLAST_RESULTS}
fi
echo "  BLAST results: $(wc -l < ${BLAST_RESULTS}) hits"

# Copy contig mapping from v2 if not present
if [ ! -f "${CONTIG_MAP}" ]; then
    if [ -f "${CONTIG_MAP_V2}" ]; then
        echo "  Copying contig mapping from v2..."
        cp ${CONTIG_MAP_V2} ${CONTIG_MAP}
    else
        echo "Error: Contig mapping not found!"
        echo "Run create_contig_map.slurm first."
        exit 1
    fi
fi
echo "  Contig mapping: $(wc -l < ${CONTIG_MAP}) entries"

# ============================================
# STEP 1: Compare with v2 results
# ============================================
echo ""
echo "Step 1: Comparing v3 vs v2 BLAST results..."

V2_RESULTS="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/blast_filtered_combined.tsv"
if [ -f "${V2_RESULTS}" ]; then
    V2_HITS=$(wc -l < ${V2_RESULTS})
    V3_HITS=$(wc -l < ${BLAST_RESULTS})
    echo "  v2 hits (max_target_seqs=500):  ${V2_HITS}"
    echo "  v3 hits (max_target_seqs=2000): ${V3_HITS}"
    echo "  Additional hits found: $((V3_HITS - V2_HITS))"
fi

# ============================================
# STEP 2: Filter contaminated accessions
# ============================================
echo ""
echo "Step 2: Filtering contaminated accessions..."

# Output directory for filtered accession lists (v3)
FILTERED_ACCESSIONS="${OUTPUT_BASE}/gtdb_dataset_filtered_v3"
mkdir -p ${FILTERED_ACCESSIONS}

python ${SCRIPT_DIR}/filter_prophage_contaminated_v2.py \
    --blast-results "${BLAST_RESULTS}" \
    --contig-map "${CONTIG_MAP}" \
    --input-accessions-dir "${GTDB_ACCESSIONS}" \
    --output-accessions-dir "${FILTERED_ACCESSIONS}" \
    --min-identity ${MIN_IDENTITY} \
    --min-alignment-length ${MIN_ALIGNMENT_LENGTH} \
    --output-contaminated "${FILTERED_ACCESSIONS}/contaminated_accessions.txt" \
    --output-summary "${FILTERED_ACCESSIONS}/filtering_summary.txt"

if [ $? -ne 0 ]; then
    echo "Error: Filtering failed!"
    exit 1
fi

# Check results
NUM_CLEAN=$(wc -l < "${FILTERED_ACCESSIONS}/train_accessions.txt")
NUM_CONTAMINATED=$(wc -l < "${FILTERED_ACCESSIONS}/contaminated_accessions.txt")
echo ""
echo "  Clean genomes in train set: ${NUM_CLEAN}"
echo "  Total contaminated genomes: ${NUM_CONTAMINATED}"

# Compare with v2
V2_CLEAN="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered/train_accessions.txt"
if [ -f "${V2_CLEAN}" ]; then
    V2_NUM=$(wc -l < ${V2_CLEAN})
    echo "  v2 clean genomes: ${V2_NUM}"
    echo "  Additional contaminated found in v3: $((V2_NUM - NUM_CLEAN))"
fi

# ============================================
# STEP 3: Subsample at 2000 bp
# ============================================
echo ""
echo "=========================================="
echo "Step 3: Subsampling at 2000 bp..."
echo "=========================================="

SEGMENT_LENGTH=2000

for SPLIT in train dev test; do
    echo ""
    echo "Processing ${SPLIT} split (${SEGMENT_LENGTH} bp)..."

    ACCESSION_FILE="${FILTERED_ACCESSIONS}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${FILTERED_ACCESSIONS}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${FILTERED_ACCESSIONS}/${SPLIT}_segments.tsv"

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

    N_ACCESSIONS=$(wc -l < "${ACCESSION_FILE}")
    echo "  Accessions: ${N_ACCESSIONS}"
    echo "  Target segments: ${TARGET_SEGMENTS}"

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
        ACTUAL=$(grep -c "^>" "${OUTPUT_FASTA}" 2>/dev/null || echo "0")
        echo "  ${SPLIT} complete: ${ACTUAL} segments"
    else
        echo "  Error: ${SPLIT} failed!"
    fi
done

# ============================================
# STEP 4: Subsample at 4000 bp
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Subsampling at 4000 bp..."
echo "=========================================="

SEGMENT_LENGTH=4000
OUTPUT_4K="${OUTPUT_BASE}/gtdb_dataset_filtered_v3_4k"
mkdir -p ${OUTPUT_4K}

for SPLIT in train dev test; do
    echo ""
    echo "Processing ${SPLIT} split (${SEGMENT_LENGTH} bp)..."

    ACCESSION_FILE="${FILTERED_ACCESSIONS}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${OUTPUT_4K}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${OUTPUT_4K}/${SPLIT}_segments.tsv"

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
        ACTUAL=$(grep -c "^>" "${OUTPUT_FASTA}" 2>/dev/null || echo "0")
        echo "  ${SPLIT} complete: ${ACTUAL} segments"
    else
        echo "  Error: ${SPLIT} failed!"
    fi
done

# ============================================
# STEP 5: Subsample at 8000 bp
# ============================================
echo ""
echo "=========================================="
echo "Step 5: Subsampling at 8000 bp..."
echo "=========================================="

SEGMENT_LENGTH=8000
OUTPUT_8K="${OUTPUT_BASE}/gtdb_dataset_filtered_v3_8k"
mkdir -p ${OUTPUT_8K}

for SPLIT in train dev test; do
    echo ""
    echo "Processing ${SPLIT} split (${SEGMENT_LENGTH} bp)..."

    ACCESSION_FILE="${FILTERED_ACCESSIONS}/${SPLIT}_accessions.txt"
    OUTPUT_FASTA="${OUTPUT_8K}/${SPLIT}_segments.fasta"
    OUTPUT_METADATA="${OUTPUT_8K}/${SPLIT}_segments.tsv"

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
        ACTUAL=$(grep -c "^>" "${OUTPUT_FASTA}" 2>/dev/null || echo "0")
        echo "  ${SPLIT} complete: ${ACTUAL} segments"
    else
        echo "  Error: ${SPLIT} failed!"
    fi
done

# ============================================
# SUMMARY
# ============================================
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

echo ""
echo "Filtered accession lists (${FILTERED_ACCESSIONS}):"
for SPLIT in train dev test; do
    if [ -f "${FILTERED_ACCESSIONS}/${SPLIT}_accessions.txt" ]; then
        COUNT=$(wc -l < "${FILTERED_ACCESSIONS}/${SPLIT}_accessions.txt")
        echo "  ${SPLIT}: ${COUNT} clean accessions"
    fi
done

echo ""
echo "2k segments (${FILTERED_ACCESSIONS}):"
for SPLIT in train dev test; do
    if [ -f "${FILTERED_ACCESSIONS}/${SPLIT}_segments.fasta" ]; then
        COUNT=$(grep -c "^>" "${FILTERED_ACCESSIONS}/${SPLIT}_segments.fasta" 2>/dev/null || echo "0")
        echo "  ${SPLIT}: ${COUNT} segments"
    fi
done

echo ""
echo "4k segments (${OUTPUT_4K}):"
for SPLIT in train dev test; do
    if [ -f "${OUTPUT_4K}/${SPLIT}_segments.fasta" ]; then
        COUNT=$(grep -c "^>" "${OUTPUT_4K}/${SPLIT}_segments.fasta" 2>/dev/null || echo "0")
        echo "  ${SPLIT}: ${COUNT} segments"
    fi
done

echo ""
echo "8k segments (${OUTPUT_8K}):"
for SPLIT in train dev test; do
    if [ -f "${OUTPUT_8K}/${SPLIT}_segments.fasta" ]; then
        COUNT=$(grep -c "^>" "${OUTPUT_8K}/${SPLIT}_segments.fasta" 2>/dev/null || echo "0")
        echo "  ${SPLIT}: ${COUNT} segments"
    fi
done

echo ""
echo "=========================================="
echo "Done!"
echo "End time: $(date)"
echo "=========================================="
