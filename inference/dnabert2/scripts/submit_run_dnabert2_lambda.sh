#!/bin/bash

# Submission script for DNABERT2 finetuning
# Submits 10 jobs with different random seeds

for seed in {1..10}
do
    echo "Submitting job with seed: $seed"
    sbatch run_dnabert2_lambda.sh $seed
done

echo "All 10 jobs submitted!"
