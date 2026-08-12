#!/bin/bash
#SBATCH --job-name=gomoe_lambda_embed
#SBATCH --account=nstaff_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=gomoe_lambda_embed_%j.out
#
# GenomeOcean-MoE embedding analysis on a LAMBDA window (linear probe + 3-layer NN).
# Perlmutter (NERSC), run inside the go-embed podman-hpc container.
#
# Edit the three paths below, then:  sbatch run_embedding_analysis.slurm
# Run once per window (2k / 4k / 8k) by pointing CSV_DIR / OUT_DIR at that window.

set -euo pipefail

# ---- paths (edit these) --------------------------------------------------
IMAGE="docker.io/lmlindsey/go-embed:1.0"
CKPT_DIR="/global/cfs/cdirs/jgirnd/projects/LLMs/genomeocean-moe/100m-juhno/checkpoints"
MODEL="hf_moe_upcycled"                                   # subdir under CKPT_DIR
DATA_ROOT="$SCRATCH/LAMBDA_v1"                            # unpacked LAMBDA_v1
WINDOW="2k"                                               # 2k | 4k | 8k
CSV_DIR="$DATA_ROOT/train_val_test/$WINDOW"              # train.csv,val.csv,test.csv
OUT_DIR="$SCRATCH/go_moe_lambda/$WINDOW"                 # results land here
# --------------------------------------------------------------------------

mkdir -p "$OUT_DIR"

# The container sees the repo at /work, checkpoints at /ckpts (read-only),
# and the LAMBDA data + output dir at their real $SCRATCH paths (bind-mounted).
podman-hpc run --gpu --rm \
    -v "$PWD":/work \
    -v "$CKPT_DIR":/ckpts:ro \
    -v "$SCRATCH":"$SCRATCH" \
    -w /work \
    -e KMP_DUPLICATE_LIB_OK=TRUE \
    "$IMAGE" \
    python embedding_analysis.py \
        --csv_dir    "$CSV_DIR" \
        --model_path "/ckpts/$MODEL" \
        --output_dir "$OUT_DIR" \
        --batch_size 32 \
        --max_length 2048 \
        --pooling    mean \
        --seed       42

echo "Done. Results in $OUT_DIR/embedding_analysis_results.json"
