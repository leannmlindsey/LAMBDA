#!/bin/bash
# Create tar.gz archives of all LAMBDA datasets for Zenodo upload
#
# Usage: bash create_lambda_final.sh

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
OUTPUT_DIR="${SCRIPT_DIR}/lambda_final"

# Source data locations
INPHARED_DATA="/u/llindsey1/llindsey/black_and_white/data/inphared"
GTDB_METADATA="/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata"
INPHARED_VCLUST="/projects/bfzj/llindsey1/black_and_white/data/inphared/vclust_info.tsv"

echo "=========================================="
echo "Create LAMBDA Final Dataset Archives"
echo "=========================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to create tar.gz and report size
create_archive() {
    local SOURCE_DIR="$1"
    local ARCHIVE_NAME="$2"

    if [ -d "${SOURCE_DIR}" ]; then
        echo "Creating ${ARCHIVE_NAME}.tar.gz..."
        tar -czf "${OUTPUT_DIR}/${ARCHIVE_NAME}.tar.gz" -C "$(dirname ${SOURCE_DIR})" "$(basename ${SOURCE_DIR})"
        SIZE=$(ls -lh "${OUTPUT_DIR}/${ARCHIVE_NAME}.tar.gz" | awk '{print $5}')
        echo "  Done: ${SIZE}"
    else
        echo "  Skipping ${ARCHIVE_NAME} - directory not found: ${SOURCE_DIR}"
    fi
}

# ============================================
# Step 0: Create metadata directory
# ============================================
echo ""
echo "=== Creating Metadata Directory ==="

METADATA_DIR="${SCRIPT_DIR}/lambda_metadata"
mkdir -p "${METADATA_DIR}"

# Copy INPHARED metadata
echo "Copying INPHARED metadata..."
cp "${INPHARED_DATA}/14Apr2025_data.tsv" "${METADATA_DIR}/inphared_metadata.tsv"
cp "${INPHARED_VCLUST}" "${METADATA_DIR}/inphared_vclust_clusters.tsv"

# Copy GTDB metadata
echo "Copying GTDB metadata..."
cp "${GTDB_METADATA}/bac120_metadata.tsv" "${METADATA_DIR}/gtdb_bac120_metadata.tsv"

# Copy contaminated accessions list
echo "Copying contaminated accessions..."
cp "${SCRIPT_DIR}/gtdb_dataset_filtered_v3/contaminated_accessions.txt" "${METADATA_DIR}/gtdb_contaminated_accessions.txt"

# Copy accession lists
echo "Copying accession lists..."
mkdir -p "${METADATA_DIR}/accessions"
for SPLIT in train dev test; do
    cp "${SCRIPT_DIR}/inphared_dataset/${SPLIT}_accessions.txt" "${METADATA_DIR}/accessions/inphared_${SPLIT}_accessions.txt"
    cp "${SCRIPT_DIR}/gtdb_dataset_filtered_v3/${SPLIT}_accessions.txt" "${METADATA_DIR}/accessions/gtdb_filtered_${SPLIT}_accessions.txt"
    cp "${SCRIPT_DIR}/gtdb_dataset/${SPLIT}_accessions.txt" "${METADATA_DIR}/accessions/gtdb_unfiltered_${SPLIT}_accessions.txt" 2>/dev/null || true
done

echo "Metadata directory created: ${METADATA_DIR}"
ls -la "${METADATA_DIR}"

# ============================================
# Phage Segments (INPHARED)
# ============================================
echo ""
echo "=== Phage Segments (INPHARED) ==="
create_archive "${SCRIPT_DIR}/inphared_dataset" "inphared_segments_2k"
create_archive "${SCRIPT_DIR}/inphared_dataset_4k" "inphared_segments_4k"
create_archive "${SCRIPT_DIR}/inphared_dataset_8k" "inphared_segments_8k"

# ============================================
# Bacteria Segments - Filtered (prophage-free)
# ============================================
echo ""
echo "=== Bacteria Segments (Filtered - Prophage-free) ==="
create_archive "${SCRIPT_DIR}/gtdb_dataset_filtered_v3" "gtdb_segments_filtered_2k"
create_archive "${SCRIPT_DIR}/gtdb_dataset_filtered_v3_4k" "gtdb_segments_filtered_4k"
create_archive "${SCRIPT_DIR}/gtdb_dataset_filtered_v3_8k" "gtdb_segments_filtered_8k"

# ============================================
# Bacteria Segments - Unfiltered (for comparison)
# ============================================
echo ""
echo "=== Bacteria Segments (Unfiltered - for comparison) ==="
create_archive "${SCRIPT_DIR}/gtdb_dataset" "gtdb_segments_unfiltered_2k"
create_archive "${SCRIPT_DIR}/gtdb_dataset_4k" "gtdb_segments_unfiltered_4k"
create_archive "${SCRIPT_DIR}/gtdb_dataset_8k" "gtdb_segments_unfiltered_8k"

# ============================================
# Merged Datasets - Filtered (INPHARED + GTDB prophage-free)
# ============================================
echo ""
echo "=== Merged Datasets (Filtered) ==="
create_archive "${SCRIPT_DIR}/merged_datasets_filtered" "merged_filtered"

# ============================================
# Merged Datasets - Unfiltered (INPHARED + GTDB with prophage)
# ============================================
echo ""
echo "=== Merged Datasets (Unfiltered) ==="
create_archive "${SCRIPT_DIR}/merged_datasets_unfiltered" "merged_unfiltered"

# ============================================
# GC-Content Control (Shuffled)
# ============================================
echo ""
echo "=== GC-Content Control (Shuffled) ==="
create_archive "${SCRIPT_DIR}/shuffled_gc_control" "gc_control_shuffled"

# ============================================
# Benchmark Datasets
# ============================================
echo ""
echo "=== Benchmark Datasets ==="
create_archive "${SCRIPT_DIR}/phage_only_genomes" "benchmark_phage_only_genomes"
create_archive "${SCRIPT_DIR}/phage_only_annotations" "benchmark_phage_only_annotations"
create_archive "${SCRIPT_DIR}/bacteria_only_genomes" "benchmark_bacteria_only_genomes"
create_archive "${SCRIPT_DIR}/bacteria_only_prophage_free" "benchmark_bacteria_only_metadata"
create_archive "${SCRIPT_DIR}/bacteria_only_annotations" "benchmark_bacteria_only_annotations"

# ============================================
# Metadata
# ============================================
echo ""
echo "=== Metadata ==="
create_archive "${METADATA_DIR}" "lambda_metadata"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Archives created in: ${OUTPUT_DIR}"
echo ""

# List all archives with sizes
echo "Archive                                    Size"
echo "------------------------------------------ --------"
for f in "${OUTPUT_DIR}"/*.tar.gz; do
    if [ -f "$f" ]; then
        NAME=$(basename "$f")
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        printf "%-42s %s\n" "$NAME" "$SIZE"
    fi
done

# Calculate total size
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | awk '{print $1}')
echo ""
echo "Total size: ${TOTAL_SIZE}"

echo ""
echo "=========================================="
echo "Contents Summary"
echo "=========================================="
echo ""
echo "Segment Datasets (for fine-tuning):"
echo "  - inphared_segments_{2k,4k,8k}.tar.gz    Phage segments"
echo "  - gtdb_segments_filtered_{2k,4k,8k}.tar.gz   Bacteria (prophage-free)"
echo "  - gtdb_segments_unfiltered_{2k,4k,8k}.tar.gz Bacteria (with prophage)"
echo ""
echo "Merged Datasets (phage + bacteria combined):"
echo "  - merged_filtered.tar.gz                 INPHARED + GTDB filtered"
echo "  - merged_unfiltered.tar.gz               INPHARED + GTDB unfiltered"
echo ""
echo "Control Datasets:"
echo "  - gc_control_shuffled.tar.gz             Shuffled test segments"
echo ""
echo "Benchmark Datasets (full genomes + annotations):"
echo "  - benchmark_phage_only_genomes.tar.gz        869 test phage full genomes"
echo "  - benchmark_phage_only_annotations.tar.gz    Pharokka annotations"
echo "  - benchmark_bacteria_only_genomes.tar.gz     100 prophage-free bacteria genomes"
echo "  - benchmark_bacteria_only_metadata.tar.gz    100 prophage-free bacteria metadata"
echo "  - benchmark_bacteria_only_annotations.tar.gz Bakta annotations"
echo ""
echo "Metadata:"
echo "  - lambda_metadata.tar.gz                 All metadata files"
echo "    - inphared_metadata.tsv                INPHARED taxonomy + info"
echo "    - inphared_vclust_clusters.tsv         INPHARED clustering (95% ANI)"
echo "    - gtdb_bac120_metadata.tsv             GTDB taxonomy + quality"
echo "    - gtdb_contaminated_accessions.txt     Removed prophage genomes"
echo "    - accessions/                          Train/dev/test accession lists"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
