#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gtdb_sel_blastdb
#SBATCH --output=gtdb_selected_blastdb_%j.out
#SBATCH --error=gtdb_selected_blastdb_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

# Create a BLAST database from SELECTED GTDB bacterial genomes only
# (Uses the accession list from gtdb_dataset, ~16k genomes)
#
# Usage: sbatch create_gtdb_selected_blastdb.slurm

# ============================================
# CONFIGURATION
# ============================================

# GTDB base directory
GTDB_BASE="/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"

# Accession list (all selected genomes)
ACCESSION_LIST="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset/all_accessions.txt"

# Output directory for BLAST database (using /work/hdd for space)
OUTPUT_DIR="/work/hdd/bfzj/llindsey1/gtdb_blastdb_selected"

# Database name
DB_NAME="gtdb_bacteria_selected"

# Combined FASTA file
COMBINED_FASTA="${OUTPUT_DIR}/gtdb_selected_genomes.fna"

# Keep combined FASTA after database creation?
KEEP_COMBINED="true"

# ============================================

echo "=========================================="
echo "Create GTDB Selected BLAST Database"
echo "=========================================="
echo "Start time: $(date)"
echo "GTDB base directory: ${GTDB_BASE}"
echo "Accession list: ${ACCESSION_LIST}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Database name: ${DB_NAME}"
echo "=========================================="

# Check inputs
if [ ! -d "${GTDB_BASE}/database" ]; then
    echo "Error: GTDB database directory not found: ${GTDB_BASE}/database"
    exit 1
fi

if [ ! -f "${ACCESSION_LIST}" ]; then
    echo "Error: Accession list not found: ${ACCESSION_LIST}"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Activate conda environment with BLAST installed
source ~/.bashrc
conda activate lambda_tools

# Count accessions
NUM_ACCESSIONS=$(wc -l < ${ACCESSION_LIST})
echo ""
echo "Processing ${NUM_ACCESSIONS} selected genomes..."

# Function to convert accession to file path
accession_to_path() {
    local acc=$1

    # Strip GB_ or RS_ prefix if present
    acc=${acc#GB_}
    acc=${acc#RS_}

    # Extract prefix (GCA or GCF) and number
    local prefix=${acc%%_*}
    local rest=${acc#*_}
    local number=${rest%%.*}

    # Pad to 9 digits and split into chunks
    # Use 10# to force decimal interpretation (avoid octal issues with leading zeros)
    number=$(printf "%09d" $((10#${number})))
    local chunk1=${number:0:3}
    local chunk2=${number:3:3}
    local chunk3=${number:6:3}

    # Build path
    echo "${GTDB_BASE}/database/${prefix}/${chunk1}/${chunk2}/${chunk3}/${acc}_genomic.fna.gz"
}

# Remove existing combined file
rm -f ${COMBINED_FASTA}

# Process each accession
COUNTER=0
FOUND=0
NOT_FOUND=0

while IFS= read -r accession; do
    [ -z "$accession" ] && continue

    COUNTER=$((COUNTER + 1))

    # Get file path
    fna_path=$(accession_to_path "$accession")

    if [ -f "$fna_path" ]; then
        zcat "$fna_path" >> ${COMBINED_FASTA}
        FOUND=$((FOUND + 1))
    else
        # Try alternate prefix (GCA <-> GCF)
        alt_accession=${accession#GB_}
        alt_accession=${alt_accession#RS_}
        if [[ $alt_accession == GCA_* ]]; then
            alt_accession=${alt_accession/GCA_/GCF_}
        else
            alt_accession=${alt_accession/GCF_/GCA_}
        fi
        alt_path=$(accession_to_path "$alt_accession")

        if [ -f "$alt_path" ]; then
            zcat "$alt_path" >> ${COMBINED_FASTA}
            FOUND=$((FOUND + 1))
        else
            echo "  Not found: $accession"
            NOT_FOUND=$((NOT_FOUND + 1))
        fi
    fi

    if [ $((COUNTER % 1000)) -eq 0 ]; then
        echo "  Processed ${COUNTER}/${NUM_ACCESSIONS} accessions..."
    fi

done < ${ACCESSION_LIST}

echo ""
echo "Concatenation complete!"
echo "  Found: ${FOUND}"
echo "  Not found: ${NOT_FOUND}"
echo "  Combined file size: $(ls -lh ${COMBINED_FASTA} | awk '{print $5}')"
echo "  Total sequences: $(grep -c "^>" ${COMBINED_FASTA})"

# Create BLAST database
echo ""
echo "Running makeblastdb..."
makeblastdb \
    -in ${COMBINED_FASTA} \
    -dbtype nucl \
    -out ${OUTPUT_DIR}/${DB_NAME} \
    -title "GTDB Selected Bacterial Genomes" \
    -parse_seqids

if [ $? -eq 0 ]; then
    echo ""
    echo "BLAST database created successfully!"

    if [ "${KEEP_COMBINED}" != "true" ]; then
        echo "Removing combined FASTA file..."
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
else
    echo ""
    echo "=========================================="
    echo "BLAST database creation failed!"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi
