# ProkBERT Inference

Run ProkBERT inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [ProkBERT_generic_sequence_classification](https://github.com/leannmlindsey/ProkBERT_generic_sequence_classification)

ProkBERT's inference scripts import from the `prokbert` Python package, so the full `src/prokbert/` package source is included here (with `pyproject.toml`) so it can be installed locally with `pip install -e .`.

## Files

| Path | Purpose |
|------|---------|
| `inference_lambda.py` | Inference from a local fine-tuned checkpoint |
| `inference_hf.py` | Inference from a HuggingFace-hosted ProkBERT model |
| `analyze_predictions.py`, `analyze_threshold.py`, `reeval_with_threshold.py`, `fix_prediction_columns.py` | Post-processing utilities |
| `src/prokbert/` | ProkBERT package source (LCA tokenizer, model definitions, dataset utilities) |
| `pyproject.toml`, `MANIFEST.in` | Package install config |
| `slurm_scripts/` | SLURM wrappers for single and batch inference (both local and HF variants) |

## Quick start

```bash
# Install the ProkBERT package
pip install -e .

# Local checkpoint inference
python inference_lambda.py \
    --checkpoint_path /path/to/finetuned/prokbert-mini \
    --base_model neuralbioinfo/prokbert-mini \
    --dataset_file /path/to/lambda_test.csv \
    --save_metrics

# Or use a HuggingFace-hosted model directly:
python inference_hf.py \
    --model_name neuralbioinfo/prokbert-mini-c-phage \
    --dataset_file /path/to/lambda_test.csv \
    --save_metrics
```

Pick the model variant by its k-mer / shift / context-length tradeoff — see the [upstream README](https://github.com/leannmlindsey/ProkBERT_generic_sequence_classification) for guidance.
