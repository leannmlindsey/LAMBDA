# Caduceus Inference

Run Caduceus inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [Caduceus_generic_sequence_classification](https://github.com/leannmlindsey/Caduceus_generic_sequence_classification)

Caduceus's inference uses the framework's Hydra-based config system, so this subdirectory includes the full `src/` package, `configs/`, and `train.py` (used by the framework even at inference time) in addition to the entry-point inference scripts. The training-only finetune/pretrain scripts are omitted.

## Files

| Path | Purpose |
|------|---------|
| `src/inference.py` | Single-CSV inference |
| `src/batch_inference.py` | Batch inference over many CSVs |
| `src/dataloaders/`, `src/models/`, `src/callbacks/`, `src/tasks/`, `src/utils/`, `src/ops/` | Caduceus framework package — required by inference |
| `configs/` | Hydra configs (csv_binary, model variants, callbacks, etc.) |
| `train.py` | Framework entry point — imported by inference scripts |
| `caduceus_env.yml`, `setup_env.sh` | Conda env spec |
| `slurm_scripts/run_inference.sh`, `wrapper_run_batch_inference.sh`, etc. | SLURM wrappers |

## Quick start

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env

python -m src.inference \
    --input_csv=/path/to/lambda_test.csv \
    --checkpoint_path=/path/to/finetuned.ckpt \
    --config_path=/path/to/model_config.json \
    --output_csv=predictions.csv \
    --conjoin_test \
    --save_metrics
```

Use `--conjoin_test` for Caduceus-Ph (post-hoc reverse-complement averaging). For Caduceus-PS, omit it. See the [upstream README](https://github.com/leannmlindsey/Caduceus_generic_sequence_classification) for variant-specific configuration.
