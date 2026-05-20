# megaDNA Inference

Run megaDNA inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [megaDNA](https://github.com/leannmlindsey/megaDNA)

megaDNA is a causal (generative) model, so classification is performed in two stages:
1. **Embedding extraction** — run the LAMBDA training CSVs through megaDNA to extract per-sequence embeddings, and train a linear probe + 3-layer NN classifier on them (this step is in the [upstream repo](https://github.com/leannmlindsey/megaDNA), not here).
2. **Inference** — use the saved classifier (`three_layer_nn_pretrained.pt`) + scaler (`three_layer_nn_pretrained_scaler.pkl`) to predict on new sequences. That's what the script in this directory does.

## Files

| Path | Purpose |
|------|---------|
| `inference_megadna.py` | Extract embeddings, run classifier, output predictions |
| `aggregate_results.py`, `calculate_metrics.py` | Aggregate / metric utilities |
| `inspect_checkpoint.py` | Sanity-check a megaDNA checkpoint |
| `megaDNA/` | megaDNA model package |
| `requirements.txt`, `setup.py` | Python dependencies + install |
| `slurm_scripts/` | SLURM wrappers (single, batch, genome-wide) |

## Quick start

```bash
pip install -e .

python inference_megadna.py \
    --input_csv /path/to/lambda_test.csv \
    --model_path /path/to/megaDNA_phage_145M.pt \
    --classifier_path /path/to/three_layer_nn_pretrained.pt \
    --scaler_path /path/to/three_layer_nn_pretrained_scaler.pkl \
    --layer middle \
    --pooling mean \
    --save_metrics
```

`--layer` and `--pooling` must match the values used when the classifier was trained. See the [upstream README](https://github.com/leannmlindsey/megaDNA) for embedding extraction and classifier training.
