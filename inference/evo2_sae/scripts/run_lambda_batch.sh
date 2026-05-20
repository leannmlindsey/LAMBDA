#!/bin/bash
# Run LAMBDA batch processing with Evo2 SAE

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evo2-sae

# Set paths
FASTA_DIR="/net/intdev/metagut/lindseylm/LAMBDA_DATA/lambda_genomes/FASTA"
GROUND_TRUTH="/net/intdev/metagut/lindseylm/LAMBDA_DATA/lambda_genomes/Lambda_Genome_Wide_Evaluation_Test_Set.csv"
OUTPUT_DIR="./lambda_results_7b"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run batch processing
echo "Starting LAMBDA batch processing..."
echo "FASTA dir: $FASTA_DIR"
echo "Ground truth: $GROUND_TRUTH"
echo "Output: $OUTPUT_DIR"
echo ""

python src/run_lambda_batch.py \
    --fasta_dir $FASTA_DIR \
    --ground_truth $GROUND_TRUTH \
    --output_dir $OUTPUT_DIR \
    --model evo2_7b \
    --window_size 50000

echo ""
echo "Done! Results in $OUTPUT_DIR"
