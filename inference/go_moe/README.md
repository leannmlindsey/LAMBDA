# GenomeOcean-MoE (GO-MoE) Inference

Run **GenomeOcean-MoE** on the LAMBDA benchmark via the **embedding-probe** pathway
(frozen embeddings → linear probe + 3-layer NN), the same evaluation used for the
other embedding-based models in LAMBDA.

**Upstream repository (model, training, checkpoints):**
[jgi-genomeocean/genomeocean-moe](https://github.com/jgi-genomeocean/genomeocean-moe) — GO-MoE is a
Mixtral-style sparse Mixture-of-Experts DNA language model (12 layers, hidden 768,
8 experts, top-2 routing; ~714M total / ~205M active params), drop-upcycled from
GenomeOcean-100M.

> **Loading note (important).** GO-MoE is loaded **natively as a Mixtral model with
> `trust_remote_code=False`**. The checkpoint's vendored remote-code loading path is
> broken and returns incorrect weights on modern `transformers`; the native
> `MixtralModel`/`MixtralForCausalLM` path reproduces the checkpoint bit-for-bit.
> All loading in `embedding_analysis.py` already uses `trust_remote_code=False`.

## Files

| Path | Purpose |
|------|---------|
| `embedding_analysis.py` | Extract mean-pooled embeddings → linear probe (LogisticRegression) + 3-layer NN, plus silhouette and PCA. Writes `embedding_analysis_results.json` and per-window predictions. |
| `requirements.txt` | Python dependencies (already in the go-embed container). |
| `slurm_scripts/run_embedding_analysis.slurm` | Perlmutter (NERSC) SLURM wrapper that runs the analysis inside the `go-embed` podman-hpc container. |

The probe / NN / metric code is identical to `inference/generanno/…/embedding_analysis.py`,
so GO-MoE's linear-probe and 3-layer-NN numbers are directly comparable to the other
LAMBDA models. Only the model-loading was changed (native Mixtral, `trust_remote_code=False`).

## Input

A LAMBDA `train_val_test/<window>/` directory containing `train.csv`, `val.csv`, and
`test.csv`, each with at least `sequence` and `label` columns (the standard LAMBDA
`segment_id,sequence,label,source` schema works as-is).

## Quick start (Perlmutter)

```bash
# 1. Unpack the dataset to $SCRATCH (one time)
tar xzf LAMBDA_v1.tar.gz -C $SCRATCH

# 2. From this directory, submit one job per window (edit WINDOW in the script)
cd inference/go_moe
sbatch slurm_scripts/run_embedding_analysis.slurm
```

Results are written to `$SCRATCH/go_moe_lambda/<window>/`:
- `embedding_analysis_results.json` — linear-probe and 3-layer-NN metrics
  (accuracy, precision, recall, F1, MCC, AUC, sensitivity, specificity) + silhouette + PCA variance.
- `embeddings_pretrained.npz` — cached train/val/test embeddings (delete to re-extract).
- `test_predictions_pretrained.csv`, `three_layer_nn_pretrained.pt`, `pca_visualization_pretrained.png`.

## Run directly (interactive node / container)

```bash
salloc -N 1 -C gpu -q interactive -t 01:00:00 -A nstaff_g -G 1
podman-hpc run --gpu --rm -v "$PWD":/work \
    -v /global/cfs/cdirs/jgirnd/projects/LLMs/genomeocean-moe/100m-juhno/checkpoints:/ckpts:ro \
    -v "$SCRATCH":"$SCRATCH" -w /work -e KMP_DUPLICATE_LIB_OK=TRUE \
    docker.io/lmlindsey/go-embed:1.0 \
    python embedding_analysis.py \
        --csv_dir  $SCRATCH/LAMBDA_v1/train_val_test/2k \
        --model_path /ckpts/hf_moe_upcycled \
        --output_dir $SCRATCH/go_moe_lambda/2k \
        --pooling mean --max_length 2048 --seed 42
```

## Notes

- **Pooling:** `mean` (attention-mask-weighted mean of the final hidden state) is the
  default and matches the GO-MoE PHROG analysis. `cls`/`last` are also available.
- **Windows:** run once per LAMBDA window (2k / 4k / 8k). GO-MoE's BPE tokenizer is
  ~4.89 bp/token, so `--max_length 2048` (tokens) covers all three without truncation.
- **Randomized control** is *not* run by default (no `--include_random_baseline`),
  matching the scope agreed for this evaluation (embedding probes only, no fine-tune).
- **Fine-tuning** GO-MoE for LAMBDA (Category 2) is out of scope here; if needed later,
  follow the `replication/new_model/` template.
