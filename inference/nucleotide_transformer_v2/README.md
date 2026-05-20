# Nucleotide Transformer v2 Inference

Run Nucleotide Transformer v2 inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [NTv2_generic_sequence_classification](https://github.com/leannmlindsey/NTv2_generic_sequence_classification)

## Files

| File | Purpose |
|------|---------|
| `inference_nt.py` | Single-CSV inference |
| `inference_nt_dir.py` | Directory-wide inference (loads model once, processes every CSV) |
| `summarize_inference_results.py` | Aggregate metrics across multiple prediction CSVs |
| `requirements.txt`, `setup.sh` | Python dependencies and conda setup |
| `slurm_scripts/` | SLURM wrappers for single, batch, and directory inference |

## Quick start

```bash
python inference_nt.py \
    --input_csv /path/to/lambda_test.csv \
    --model_path /path/to/finetuned/ntv2_500m \
    --output_csv predictions.csv \
    --fp16 \
    --save_metrics
```

Use `--fp16` for ~2.6× speedup on A100/V100 GPUs. For long sequences (4k / 8k), see the [upstream README](https://github.com/leannmlindsey/NTv2_generic_sequence_classification) for memory-tuning options.
