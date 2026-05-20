#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=contig_map
#SBATCH --output=contig_map_%j.out
#SBATCH --error=contig_map_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g

# Create mapping from contig IDs to genome accessions
#
# Usage: sbatch create_contig_map.slurm

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
GTDB_BASE="/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"
ACCESSION_LIST="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset/all_accessions.txt"
OUTPUT_FILE="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv"

echo "=========================================="
echo "Create Contig to Genome Mapping"
echo "=========================================="
echo "Start time: $(date)"

source ~/.bashrc
conda activate lambda_tools

python ${SCRIPT_DIR}/create_contig_to_genome_map.py \
    --gtdb-dir "${GTDB_BASE}" \
    --accession-list "${ACCESSION_LIST}" \
    --output "${OUTPUT_FILE}" \
    --threads 16

echo ""
echo "=========================================="
echo "Done! End time: $(date)"
echo "=========================================="
