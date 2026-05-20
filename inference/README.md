# Inference Scripts

This directory contains the **inference-only** scripts for running the LAMBDA benchmark dataset through each of the genomic language models evaluated in the paper. The scripts here take a fine-tuned (or pretrained, for embedding-based models) checkpoint and a CSV of DNA sequences, and produce predictions with per-class probabilities.

> **For training:** Each model's full training, fine-tuning, and embedding-analysis code lives in a separate upstream repository — see the per-model README and the table below. The scripts in this directory are extracted from those repos and reproduce only the inference pathway.

## Models

| Model | Subdirectory | Training repo |
|-------|--------------|---------------|
| DNABERT-2 | [`dnabert2/`](dnabert2/) | [DNABERT2_generic_sequence_classification](https://github.com/leannmlindsey/DNABERT2_generic_sequence_classification) |
| Nucleotide Transformer v2 | [`nucleotide_transformer_v2/`](nucleotide_transformer_v2/) | [NTv2_generic_sequence_classification](https://github.com/leannmlindsey/NTv2_generic_sequence_classification) |
| Caduceus | [`caduceus/`](caduceus/) | [Caduceus_generic_sequence_classification](https://github.com/leannmlindsey/Caduceus_generic_sequence_classification) |
| ProkBERT | [`prokbert/`](prokbert/) | [ProkBERT_generic_sequence_classification](https://github.com/leannmlindsey/ProkBERT_generic_sequence_classification) |
| megaDNA | [`megadna/`](megadna/) | [megaDNA](https://github.com/leannmlindsey/megaDNA) |
| GENERanno | [`generanno/`](generanno/) | [Generanno_generic_sequence_classification](https://github.com/leannmlindsey/Generanno_generic_sequence_classification) |
| Evo2 / Evo2+SAE | [`evo2_sae/`](evo2_sae/) | [Evo2_SAE_LAMBDA_assessment](https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment) (Evo2 itself: [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)) |

## Common input format

All inference scripts accept a CSV file with at minimum a `sequence` column (DNA string of A/C/G/T/N). If a `label` column is present, classification metrics are computed and saved alongside predictions.

```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
```

## Output

Each script writes a predictions CSV with `prob_0`, `prob_1`, and `pred_label` columns appended, and (with `--save_metrics`) a JSON file with accuracy, precision, recall, F1, MCC, AUC, sensitivity, and specificity.

## Hardware

All transformer inference scripts assume a CUDA-capable GPU. The Evo2+SAE genome-wide scan requires an A100 (or larger) for the 7B parameter model. Post-processing (clustering, metric calculation, plotting) runs on CPU.

## LAMBDA dataset

Download the benchmark CSVs from Zenodo: [10.5281/zenodo.19236553](https://doi.org/10.5281/zenodo.19236553). See the top-level [README](../README.md) for the directory layout.
