#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=shuffle_segments
#SBATCH --output=shuffle_segments_%j.out
#SBATCH --error=shuffle_segments_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g

# Shuffle test segments for GC-content control benchmark
# This creates shuffled versions of test_segments.fasta files
# preserving GC content but destroying sequence patterns

module load anaconda3
conda activate phage_env

BASE_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"
SCRIPT_DIR="$BASE_DIR"

# Output directory for shuffled segments
OUTPUT_DIR="$BASE_DIR/shuffled_gc_control"
mkdir -p "$OUTPUT_DIR"

echo "Creating GC-content control datasets by shuffling test segments"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Define all test segment files to shuffle
declare -A DATASETS=(
    # Phage datasets
    ["inphared_2k"]="$BASE_DIR/inphared_dataset/test_segments.fasta"
    ["inphared_4k"]="$BASE_DIR/inphared_dataset_4k/test_segments.fasta"
    ["inphared_8k"]="$BASE_DIR/inphared_dataset_8k/test_segments.fasta"
    # Bacteria datasets (filtered v3)
    ["gtdb_2k"]="$BASE_DIR/gtdb_dataset_filtered_v3/test_segments.fasta"
    ["gtdb_4k"]="$BASE_DIR/gtdb_dataset_filtered_v3_4k/test_segments.fasta"
    ["gtdb_8k"]="$BASE_DIR/gtdb_dataset_filtered_v3_8k/test_segments.fasta"
)

# Process each dataset
for name in "${!DATASETS[@]}"; do
    input_file="${DATASETS[$name]}"
    output_file="$OUTPUT_DIR/${name}_test_shuffled.fasta"

    echo "Processing: $name"
    echo "  Input:  $input_file"
    echo "  Output: $output_file"

    if [ -f "$input_file" ]; then
        python "$SCRIPT_DIR/shuffle_segments.py" \
            --input "$input_file" \
            --output "$output_file" \
            --seed 42 \
            --label-suffix "_shuffled"

        if [ $? -eq 0 ]; then
            echo "  Status: SUCCESS"
        else
            echo "  Status: FAILED"
        fi
    else
        echo "  Status: SKIPPED (input file not found)"
    fi
    echo ""
done

echo "=============================================================="
echo "Shuffling complete!"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.fasta 2>/dev/null || echo "No output files found"
