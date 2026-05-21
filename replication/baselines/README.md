# Simple sequence-feature baselines

Three CPU-only classifiers fit on hand-crafted DNA sequence features. These
establish a reference floor for the gLM evaluations in the LAMBDA paper —
"how much of the phage / bacterial distinction is captured by classical
compositional features alone?"

| Baseline | Features | Classifier |
|----------|----------|-----------|
| `baseline_gc`    | GC content (1 feature) | Logistic regression |
| `baseline_kmer4` | Normalized tetranucleotide frequencies (256 features) | Logistic regression |
| `baseline_kmer6` | Normalized hexanucleotide frequencies (4 096 features) | Logistic regression |

The classifier protocol is identical to the gLM linear probes:
`StandardScaler` → `LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')`.
This is intentional — it lets the paper say "the same linear-probe protocol
applied to a 4 096-dimensional gLM embedding and to a 4 096-dimensional k=6
k-mer composition gives the following MCC values."

## How it slots into the pipeline

All three baselines are trained on `binary_segments_<window>/train.csv` and
evaluated on every test split at the same window size:

| Category | Evaluated on |
|----------|--------------|
| `binary` | `binary_segments_<window>/test.csv` |
| `error_bias` | `error_and_bias_<window>/test.csv` |
| `genome_wide` | each `*.csv` under `genome_wide_segments_<window>_<stride>/` |

Outputs are written in the same layout `02_submit_inference_jobs.sh` uses,
so `03_build_website_data.py` auto-discovers them. Baseline rows appear in
`metrics_summary.csv` alongside the gLM rows with no aggregator changes.

## Running

```bash
# Locally (a few minutes on CPU):
python run_baselines.py \
    --dataset_root ${DATASET_ROOT} \
    --output_root ${RESULTS_ROOT} \
    --windows 2k 4k 8k \
    --plot

# On Biowulf:
sbatch run_baselines.slurm
```

## Outputs

```
RESULTS_ROOT/
├── baseline_gc/
│   ├── binary/{2k,4k,8k}/test_predictions.csv  (+ _metrics.json)
│   ├── error_bias/{2k,4k,8k}/test_predictions.csv
│   └── genome_wide/{2k,4k,8k}/<assembly>_predictions.csv
├── baseline_kmer4/  (same structure)
├── baseline_kmer6/  (same structure)
└── figures/
    └── gc_distribution.pdf      (3-panel KDE plot, with --plot)
```

## The GC figure

With `--plot`, the script produces a 3-panel figure (`figures/gc_distribution.pdf`)
showing the distribution of GC content for phage vs. bacterial sequences at
each window size (2k, 4k, 8k). Each panel:

- Filled KDE curves for **phage** (pink-red, `#f35f73`) and **bacteria** (teal, `#02afbd`)
- A dashed vertical line at the GC threshold that maximizes MCC on the test set
- Annotated MCC and threshold value in the panel title

The colors match the LAMBDA project palette (ProkBERT-family pink-red used
here as a phage class color; DNABERT2 teal used as a bacteria class color).

## Reference numbers (paper-side rough priors)

These are typical magnitudes — not actual results. Use to spot something
obviously wrong:

| Baseline | Typical MCC on 2k binary test |
|----------|-------------------------------|
| `baseline_gc`    | 0.3 – 0.5 |
| `baseline_kmer4` | 0.65 – 0.80 |
| `baseline_kmer6` | 0.70 – 0.85 |

If `baseline_gc` is above 0.6 something is wrong (GC alone shouldn't do that
well on a class-balanced benchmark). If `baseline_kmer6` is below 0.5
something is wrong (the training set isn't being seen properly).
