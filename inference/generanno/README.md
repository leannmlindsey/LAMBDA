# GENERanno Inference

Run GENERanno inference on the LAMBDA benchmark CSVs.

**Upstream repository (training + full source):** [Generanno_generic_sequence_classification](https://github.com/leannmlindsey/Generanno_generic_sequence_classification)

GENERanno's inference is invoked as a Python module (`python -m src.tasks.downstream.inference`), so the relevant `src/tasks/downstream/` tree is preserved here.

## Files

| Path | Purpose |
|------|---------|
| `src/tasks/downstream/inference.py` | Main inference script (run as a module) |
| `src/tasks/downstream/analyze_predictions.py` | Post-processing |
| `requirements.txt` | Python dependencies |
| `slurm_scripts/` | SLURM wrappers for batch inference |

## Quick start

```bash
pip install -r requirements.txt

python -m src.tasks.downstream.inference \
    --input_csv=/path/to/lambda_test.csv \
    --model_path=/path/to/finetuned/generanno \
    --output_csv=predictions.csv \
    --threshold=0.5 \
    --save_metrics
```

See the [upstream README](https://github.com/leannmlindsey/Generanno_generic_sequence_classification) for fine-tuning on LAMBDA training splits.
