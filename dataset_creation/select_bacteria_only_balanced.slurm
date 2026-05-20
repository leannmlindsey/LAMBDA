#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=select_bact_100
#SBATCH --output=select_bact_100_%j.out
#SBATCH --error=select_bact_100_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g

# Select 100 taxonomically balanced bacteria for Bacteria-Only benchmark
#
# Usage: sbatch select_bacteria_only_balanced.slurm

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Input
ACCESSIONS="${SCRIPT_DIR}/gtdb_dataset_filtered_v3/test_accessions.txt"
METADATA="/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata/bac120_metadata.tsv"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/bacteria_only_dataset"

python ${SCRIPT_DIR}/select_bacteria_only_balanced.py \
    --accessions "${ACCESSIONS}" \
    --metadata "${METADATA}" \
    --output "${OUTPUT_DIR}" \
    --n-genomes 100 \
    --seed 42
