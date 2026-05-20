#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=investigate_hits
#SBATCH --output=investigate_hits_%j.out
#SBATCH --error=investigate_hits_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g

# Investigate why 2 hits slipped through v3 filtering
#
# Usage: sbatch investigate_missed_hits.slurm

SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

bash ${SCRIPT_DIR}/investigate_missed_hits.sh
