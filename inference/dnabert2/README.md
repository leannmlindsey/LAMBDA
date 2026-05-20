# DNABERT-2 Inference

Run DNABERT-2 inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [DNABERT2_generic_sequence_classification](https://github.com/leannmlindsey/DNABERT2_generic_sequence_classification)

## Files

| File | Purpose |
|------|---------|
| `inference_dnabert2.py` | Main inference script — CSV in, predictions CSV out |
| `requirements.txt` | Python dependencies |
| `scripts/run_inference.sh` / `wrapper_run_inference.sh` | SLURM single-file inference |
| `scripts/run_batch_inference_interactive.sh` / `submit_batch_inference.sh` / `wrapper_run_batch_inference.sh` | Batch over many input CSVs |
| `scripts/analyze_genome_wide_results.py` | Aggregate per-segment predictions across a genome |
| `scripts/analyze_phage_only_results.py` | Restrict analysis to phage-positive segments |

## Quick start

```bash
python inference_dnabert2.py \
    --input_csv /path/to/lambda_test.csv \
    --model_path /path/to/finetuned/dnabert2 \
    --output_csv predictions.csv \
    --save_metrics
```

A fine-tuned DNABERT-2 checkpoint is required — see the [upstream repo](https://github.com/leannmlindsey/DNABERT2_generic_sequence_classification) for fine-tuning instructions on LAMBDA training splits.
