# Evaluating a New Genomic Language Model on LAMBDA

This directory provides templates for **use case 2** in `../README.md`:
fine-tune your new gLM on the LAMBDA training data, run inference on the
LAMBDA test data, and aggregate into metrics directly comparable to the
published models.

The templates here work for any model compatible with HuggingFace's
`AutoModelForSequenceClassification` interface — most BERT-style encoder
models, including [GENA-LM](https://github.com/AIRI-Institute/GENA_LM),
DNABERT-2, ProkBERT, Nucleotide Transformer v2, and many others. For models
that need a custom training loop (Caduceus, Evo2, megaDNA), follow the
per-model training repository linked in [`../../inference/README.md`](../../inference/README.md).

## Files

| File | Purpose |
|------|---------|
| `finetune_new_model.py` | Generic HF fine-tune template (train/dev/test CSVs → fine-tuned checkpoint). |
| `inference_new_model.py` | Generic HF inference template (sequences CSV → predictions CSV in the schema the aggregator expects). |
| `run_finetune.slurm` | SLURM submission script for fine-tuning. Edit the configuration block, then `sbatch run_finetune.slurm`. |
| `run_inference.slurm` | SLURM submission script for inference. Edit the configuration block, then `sbatch run_inference.slurm`. |

## End-to-end workflow

```bash
# 1. Set up your conda environment (one-time)
conda create -n lambda_new_model python=3.10 -y
conda activate lambda_new_model
pip install torch transformers datasets accelerate scikit-learn pandas numpy

# 2. Download the LAMBDA dataset (you only need the data, not the checkpoints)
cd ..
cp paths.env.template paths.env && vim paths.env
bash 01_download_zenodo.sh   # downloads dataset; checkpoints are also fetched
                             # but you can ignore them for use case 2

# 3. Fine-tune your model on the LAMBDA training data
cd new_model
vim run_finetune.slurm       # edit MODEL_NAME, DATASET_DIR, OUTPUT_DIR
sbatch run_finetune.slurm    # fine-tune on binary_segments_2k

# Repeat for the 4k and 8k splits to evaluate on all window sizes.

# 4. Run inference on the LAMBDA test data
vim run_inference.slurm      # edit MODEL_PATH, INPUT_CSV, OUTPUT_CSV
sbatch run_inference.slurm

# 5. Aggregate metrics (use the same aggregator the published models use)
cd ..
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --output ${RESULTS_ROOT}/aggregated
```

Your new model will appear in `metrics_summary.csv` alongside the published
models, as long as you wrote the predictions to:

```
${RESULTS_ROOT}/<your_model_name>/<category>/<window>/<input_basename>_predictions.csv
```

For example: `${RESULTS_ROOT}/gena_lm/binary/2k/test_predictions.csv`.

## CSV schema your inference must produce

If you write your own inference (instead of using `inference_new_model.py`),
the predictions CSV must contain these columns. See
[`../README.md#schemas`](../README.md#schemas) for the full reference.

```csv
sequence,label,prob_0,prob_1,pred_label
ACGT...,0,0.92,0.08,0
TGCA...,1,0.12,0.88,1
```

`segment_id`, `seq_id`, `start`, and `end` columns from the input are
preserved if present (needed for the `genome_wide` category).

## Worked example: GENA-LM

[GENA-LM](https://github.com/AIRI-Institute/GENA_LM) is a BERT-style
genomic language model from AIRI. It's a standard
`AutoModelForSequenceClassification`-compatible model, so the templates
above work directly. Recommended starting settings:

```bash
# Fine-tune
sbatch --export=ALL run_finetune.slurm
# inside run_finetune.slurm:
MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"
DATASET_DIR="${DATASET_ROOT}/binary_segments_2k"
OUTPUT_DIR="${CHECKPOINTS_ROOT}/gena_lm/2k"
MAX_LENGTH=512              # GENA-LM context window
BATCH_SIZE=16
LEARNING_RATE=3e-5
NUM_EPOCHS=3
PRECISION="--bf16"
TRUST_REMOTE_CODE=""        # GENA-LM doesn't need this
```

```bash
# Inference (after fine-tuning completes)
sbatch run_inference.slurm
# inside run_inference.slurm:
MODEL_PATH="${CHECKPOINTS_ROOT}/gena_lm/2k"
INPUT_CSV="${DATASET_ROOT}/binary_segments_2k/test.csv"
OUTPUT_CSV="${RESULTS_ROOT}/gena_lm/binary/2k/test_predictions.csv"
MAX_LENGTH=512
BATCH_SIZE=32
PRECISION="--bf16"
```

Repeat for `4k` and `8k` window sizes (and for the `error_and_bias` /
`genome_wide` categories if you want full coverage), then aggregate with
`../03_build_website_data.py`.

## Supported / unsupported gLMs

| Model family | Works with these templates? |
|--------------|:--:|
| GENA-LM | ✓ |
| DNABERT-2 | ✓ (set `--trust_remote_code`) |
| Nucleotide Transformer v2 | ✓ |
| ProkBERT | ✓ |
| HyenaDNA (HF version) | ✓ |
| GENERanno | ✓ |
| Caduceus | ✗ — use the [Caduceus training repo](https://github.com/leannmlindsey/Caduceus_generic_sequence_classification) |
| megaDNA | ✗ — embeddings-based; use the [megaDNA repo](https://github.com/leannmlindsey/megaDNA) |
| Evo2 | ✗ — embeddings/SAE-based; use the [Evo2_SAE repo](https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment) |
