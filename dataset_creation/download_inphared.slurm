#!/bin/bash
#SBATCH --job-name=inphared_download
#SBATCH --output=inphared_download_%j.out
#SBATCH --error=inphared_download_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#
# NOTE: Add your account and partition, e.g.:
# #SBATCH --account=your_account
# #SBATCH --partition=your_partition

# Download INPHARED phage genome database
#
# Usage: Set OUTPUT_DIR and SCRIPT_DIR environment variables, then submit:
#   export OUTPUT_DIR=/path/to/output
#   export SCRIPT_DIR=/path/to/scripts
#   sbatch download_inphared.slurm

# Check for required environment variables
if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: OUTPUT_DIR environment variable not set"
    echo "Usage: export OUTPUT_DIR=/path/to/output && sbatch download_inphared.slurm"
    exit 1
fi

if [ -z "${SCRIPT_DIR}" ]; then
    echo "Error: SCRIPT_DIR environment variable not set"
    echo "Usage: export SCRIPT_DIR=/path/to/scripts && sbatch download_inphared.slurm"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "INPHARED Database Download"
echo "=========================================="
echo "Start time: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
echo "=========================================="

# Load any required modules (uncomment/modify as needed)
# module load python/3.9

# Run the download script
python ${SCRIPT_DIR}/download_inphared.py \
    --output ${OUTPUT_DIR} \
    --category all \
    --include-phrogs \
    --use-wget \
    --decompress

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Download completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Download failed with errors"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi

# List downloaded files
echo ""
echo "Downloaded files:"
ls -lh ${OUTPUT_DIR}
