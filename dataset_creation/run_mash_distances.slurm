#!/bin/bash
#SBATCH --job-name=mash_dist
#SBATCH --output=mash_dist_%j.out
#SBATCH --error=mash_dist_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#
# NOTE: Add your account and partition, e.g.:
# #SBATCH --account=your_account
# #SBATCH --partition=your_partition

# Compute all-vs-all MASH distances
#
# Usage: Set MSH_FILE and OUTPUT_FILE environment variables, then submit:
#   export MSH_FILE=/path/to/genomes.fa.msh
#   export OUTPUT_FILE=/path/to/output_distances.tsv
#   sbatch run_mash_distances.slurm

# Check for required environment variables
if [ -z "${MSH_FILE}" ]; then
    echo "Error: MSH_FILE environment variable not set"
    echo "Usage: export MSH_FILE=/path/to/genomes.fa.msh && sbatch run_mash_distances.slurm"
    exit 1
fi

if [ -z "${OUTPUT_FILE}" ]; then
    echo "Error: OUTPUT_FILE environment variable not set"
    echo "Usage: export OUTPUT_FILE=/path/to/distances.tsv && sbatch run_mash_distances.slurm"
    exit 1
fi

echo "=========================================="
echo "MASH All-vs-All Distance Calculation"
echo "=========================================="
echo "Start time: $(date)"
echo "Input: ${MSH_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Threads: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

# Load mash module if needed (uncomment/modify as needed)
# module load mash

# Get number of genomes for progress estimation
NUM_GENOMES=$(mash info -t ${MSH_FILE} | tail -n +2 | wc -l)
echo "Number of genomes: ${NUM_GENOMES}"
echo "Expected comparisons: $((NUM_GENOMES * NUM_GENOMES))"
echo ""

# Start progress monitor in background
(
    while true; do
        if [ -f "${OUTPUT_FILE}" ]; then
            LINES=$(wc -l < "${OUTPUT_FILE}" 2>/dev/null || echo 0)
            SIZE=$(ls -lh "${OUTPUT_FILE}" 2>/dev/null | awk '{print $5}')
            # Each genome produces NUM_GENOMES lines of output
            GENOMES_DONE=$((LINES / NUM_GENOMES))
            PERCENT=$((GENOMES_DONE * 100 / NUM_GENOMES))
            echo "[$(date '+%H:%M:%S')] Progress: ${GENOMES_DONE}/${NUM_GENOMES} genomes (${PERCENT}%) - Output size: ${SIZE}"
        fi
        sleep 60
    done
) &
MONITOR_PID=$!

# Run MASH distance calculation
# -p: number of threads
# -t: output in tabular format (easier to parse)
mash dist -p ${SLURM_CPUS_PER_TASK} -t ${MSH_FILE} ${MSH_FILE} > ${OUTPUT_FILE}
MASH_EXIT=$?

# Kill the monitor
kill ${MONITOR_PID} 2>/dev/null

# Check exit status
if [ ${MASH_EXIT} -eq 0 ]; then
    echo "=========================================="
    echo "MASH distance calculation completed!"
    echo "Output file: ${OUTPUT_FILE}"
    echo "Output size: $(ls -lh ${OUTPUT_FILE} | awk '{print $5}')"
    echo "Number of pairwise comparisons: $(wc -l < ${OUTPUT_FILE})"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "MASH distance calculation failed!"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi
