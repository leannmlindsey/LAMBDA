#!/bin/bash
#SBATCH --job-name=dnabert2_lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=./out/dnabert2_lambda_%j.out
#SBATCH --error=./out/dnabert2_lambda_%j.err

# Biowulf-specific batch script for DNABERT2 finetuning

seed=$1

if [ -z "$seed" ]; then
    echo "Error: Please provide seed as an argument"
    exit 1
fi

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Seed: $seed"

# Load modules
module load conda
module load cuda/12.8

# Activate conda environment
source activate dna

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "Python environment:"
which python
python --version

data_path="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/4k"
lr=3e-5
f="filtered"
len="4k"

# Validate input
if [ -z "$data_path" ]; then
    echo "Error: Please provide data_path as an argument"
    echo "Usage: $0 /path/to/your/data"
    exit 1
fi

echo "The provided data_path is $data_path"

# Check if data path exists
if [ ! -d "$data_path" ]; then
    echo "Error: Data path $data_path does not exist"
    exit 1
fi

# Run training for different seeds (add more seeds for robustness)
echo "Training with seed: $seed"
    
python ../train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path $data_path \
        --kmer -1 \
        --run_name DNABERT2_lambda_${f}_${len}_lr${lr}_seed${seed} \
        --model_max_length 128 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 3 \
        --fp16 \
	--save_strategy steps \
        --save_steps 200 \
        --output_dir output/dnabert2_lambda_${f}_${len}_lr${lr}_seed${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 100000 \
        --overwrite_output_dir \
        --log_level info \
        --seed $seed \
        --load_best_model_at_end True \
	--metric_for_best_model eval_loss \
	--do_train \
	--do_eval 

echo "Training complete! Models saved to output/dnabert2_lambda_${f}_${len}_lr${lr}_seed${seed} "
