#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=analyze_blast
#SBATCH --output=analyze_blast_%j.out
#SBATCH --error=analyze_blast_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g

# Analyze BLAST filtering steps
#
# Usage: sbatch analyze_blast_filtering.slurm

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
bash ${SCRIPT_DIR}/analyze_blast_filtering.sh
