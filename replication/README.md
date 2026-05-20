# Reproducing LAMBDA Results & Running the Pipeline on Your Own Data

This directory provides clear, scripted workflows for three use cases:

1. **[Reproduce the manuscript results](#use-case-1-reproduce-the-manuscript-results)** — download dataset + fine-tuned checkpoints from Zenodo, run inference for all 8 models on all LAMBDA splits, produce the per-genome JSON files consumed by the [LAMBDA visualization dashboard](https://github.com/leannmlindsey/lambda-benchmark).
2. **[Evaluate a new genomic language model on LAMBDA](#use-case-2-evaluate-a-new-genomic-language-model-on-lambda)** — download the LAMBDA dataset, run your own model's inference following the documented CSV schemas, and use this repo's aggregator to produce metrics directly comparable to the published results.
3. **[Predict prophage locations in your own bacterial genomes](#use-case-3-predict-prophages-in-your-own-data)** — point the pipeline at one or more bacterial FASTAs; segment them into 2k/4k/8k windows; run any/all of the pretrained gLMs; aggregate into predicted prophage regions; optionally annotate with Pharokka.

All three use cases share the same scripts; what changes is which steps you run and how `paths.env` is populated.

## Files in this directory

| File | Purpose |
|------|---------|
| `paths.env.template` | Configuration template. Copy to `paths.env` and fill in. |
| `00_segment_fasta.py` | Sliding-window FASTA → LAMBDA segment CSV. Called automatically by step 02 for the genome-wide category; can also be run manually. |
| `01_download_zenodo.sh` | **Use cases 1 & 2.** Download pre-built CSVs, source FASTAs, ground truth, and fine-tuned checkpoints from Zenodo. |
| `02_submit_inference_jobs.sh` | **Use cases 1 & 3.** Submit SLURM jobs for every (model, category, window) combination. Auto-runs `00_segment_fasta.py` on `FASTA_DIR` first if genome-wide CSVs don't exist. |
| `03_build_website_data.py` | **All use cases.** Aggregate per-model predictions into a metrics CSV and per-genome JSON files. |
| `run_aggregate.slurm` | SLURM wrapper around step 03 for large result sets. Optional — running step 03 on a login node is usually fine. |
| `04_annotate_predicted_prophages.sh` | **Optional (use case 3).** Annotate predicted prophages with Pharokka PHROG categories. |
| `new_model/` | **Use case 2.** Templates for fine-tuning + running inference with a new gLM on LAMBDA. See [`new_model/README.md`](new_model/README.md). |

---

## Prerequisites

- **SLURM cluster** with GPU partitions. The default `#SBATCH` directives in each per-model `run_inference.sh` request 1× A100. Adjust in `inference/{model}/slurm_scripts/run_inference.sh` if needed.
- **Conda environments**, one per model. Env files live with each model:
  - `inference/caduceus/caduceus_env.yml`
  - `inference/evo2_sae/environment.yml` (Evo2 itself must be installed separately — see [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2))
  - `inference/{dnabert2,nucleotide_transformer_v2,prokbert,megadna,generanno}/requirements.txt`
- **Python 3.10+** for the aggregation scripts in this directory.
- **Disk space** — ~100 GB for the LAMBDA dataset, ~50 GB for the checkpoints, ~20 GB for prediction outputs.

---

## Use case 1: Reproduce the manuscript results

```bash
# 1. Configure
cp paths.env.template paths.env
vim paths.env          # set DATASET_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT, LAMBDA_REPO_ROOT

# 2. Download dataset + FASTAs + checkpoints from Zenodo
bash 01_download_zenodo.sh
# After this, DATASET_ROOT contains:
#   binary_segments_{2k,4k,8k}/{train,dev,test}.csv     pre-built CSVs
#   error_and_bias_{2k,4k,8k}/test.csv                  pre-built CSVs
#   genome_wide_fastas/<assembly>.fna                   source FASTAs
#   ground_truth.csv, taxonomy.csv

# 3. Submit every (model × category × window) inference job to SLURM
bash 02_submit_inference_jobs.sh
# On first run, this auto-segments the FASTAs in genome_wide_fastas/ into
# 2k/4k/8k segment CSVs (subdirs genome_wide_segments_2k_1k, _4k_2k, _8k_4k).
# Monitor jobs with: squeue -u $USER
# Outputs land in RESULTS_ROOT/{model}/{category}/{window}/*_predictions.csv

# 4. Once all jobs finish, aggregate
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --ground-truth ${DATASET_ROOT}/ground_truth.csv \
    --taxonomy ${DATASET_ROOT}/taxonomy.csv \
    --output ${RESULTS_ROOT}/aggregated
```

You'll get:

- `${RESULTS_ROOT}/aggregated/metrics_summary.csv` — one row per (model, category, window) with accuracy / precision / recall / F1 / MCC / AUC / sensitivity / specificity. These are the numbers in the manuscript's tables.
- `${RESULTS_ROOT}/aggregated/website_data/{assembly}_{window}.json` plus an `index.json` — the per-genome files consumed by the dashboard.

To view the dashboard locally:

```bash
git clone https://github.com/leannmlindsey/lambda-benchmark.git
cd lambda-benchmark
npm install
ln -s ${RESULTS_ROOT}/aggregated/website_data public/data
npm run dev
# Open http://localhost:5173
```

---

## Use case 2: Evaluate a new genomic language model on LAMBDA

You have a new gLM and want metrics directly comparable to the manuscript's results. The workflow is **fine-tune your model on the LAMBDA training split → run inference on the LAMBDA test split → aggregate with the same script we use for the published models.**

The [`new_model/`](new_model/) subdirectory contains templates that work for any HuggingFace `AutoModelForSequenceClassification`-compatible model (GENA-LM, DNABERT-2, NT-v2, ProkBERT, HyenaDNA, GENERanno, etc.). For models that need a custom training loop (Caduceus, Evo2, megaDNA), see the per-model training repository linked in [`../inference/README.md`](../inference/README.md).

```bash
# 1. Set up a conda env for your new model
conda create -n lambda_new_model python=3.10 -y
conda activate lambda_new_model
pip install torch transformers datasets accelerate scikit-learn pandas numpy

# 2. Configure paths and download the LAMBDA dataset
cp paths.env.template paths.env
vim paths.env                         # set DATASET_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT, LAMBDA_REPO_ROOT
bash 01_download_zenodo.sh

# 3. Fine-tune your model on the LAMBDA training data (one window size per job)
cd new_model
vim run_finetune.slurm                # set MODEL_NAME, DATASET_DIR, OUTPUT_DIR
sbatch run_finetune.slurm
# Repeat editing DATASET_DIR + OUTPUT_DIR for binary_segments_4k and _8k.

# 4. Run inference with the fine-tuned checkpoint(s)
vim run_inference.slurm               # set MODEL_PATH, INPUT_CSV, OUTPUT_CSV
sbatch run_inference.slurm
# Repeat for each (category, window) you want to evaluate.
# The multi-input loop at the bottom of run_inference.slurm can drive
# everything in one job.

# 5. Aggregate metrics (same aggregator the published models use)
cd ..
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --ground-truth ${DATASET_ROOT}/ground_truth.csv \
    --taxonomy ${TAXONOMY_CSV} \
    --output ${RESULTS_ROOT}/aggregated
```

**Output layout** — your model writes to the same convention the aggregator already understands:

```
${RESULTS_ROOT}/<your_model_name>/<category>/<window>/<input_basename>_predictions.csv
```

For example: `${RESULTS_ROOT}/gena_lm/binary/2k/test_predictions.csv`. Your model will appear in `metrics_summary.csv` and the per-genome JSONs alongside the published models. If you've already run use case 1, just drop your model's predictions into the same `RESULTS_ROOT` and re-run step 5 to add your model to the comparison.

**If you only want top-level metrics** (no genome-wide JSONs), omit `--ground-truth` and `--taxonomy`:

```bash
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --output ${RESULTS_ROOT}/aggregated
# → only metrics_summary.csv is produced
```

### Worked example: GENA-LM

[GENA-LM](https://github.com/AIRI-Institute/GENA_LM) is a BERT-style genomic LM from AIRI; it works directly with the templates. Settings to use in `new_model/run_finetune.slurm`:

```bash
MODEL_NAME="AIRI-Institute/gena-lm-bert-base-t2t"
DATASET_DIR="${DATASET_ROOT}/binary_segments_2k"
OUTPUT_DIR="${CHECKPOINTS_ROOT}/gena_lm/2k"
MAX_LENGTH=512                # GENA-LM context window
BATCH_SIZE=16
LEARNING_RATE=3e-5
NUM_EPOCHS=3
PRECISION="--bf16"            # on A100
TRUST_REMOTE_CODE=""          # GENA-LM does NOT need this
```

Then in `new_model/run_inference.slurm`:

```bash
MODEL_PATH="${CHECKPOINTS_ROOT}/gena_lm/2k"
INPUT_CSV="${DATASET_ROOT}/binary_segments_2k/test.csv"
OUTPUT_CSV="${RESULTS_ROOT}/gena_lm/binary/2k/test_predictions.csv"
MAX_LENGTH=512
BATCH_SIZE=32
PRECISION="--bf16"
```

Repeat for the 4k and 8k splits and for the `error_and_bias` / `genome_wide` categories, then aggregate. See [`new_model/README.md`](new_model/README.md) for the full reference including a table of which model families are supported by the templates.

---

## Use case 3: Predict prophages in your own data (use the published gLMs)

You have one or more bacterial FASTAs and want to run our pretrained gLMs over them to predict prophage locations.

```bash
# 1. Configure
cp paths.env.template paths.env
vim paths.env
# Set: CHECKPOINTS_ROOT, RESULTS_ROOT, LAMBDA_REPO_ROOT
# Set: CATEGORIES="genome_wide"     (skip binary / error_bias — those are for evaluation, not prediction)
# Set: FASTA_DIR=/path/to/your/bacterial_fastas
# Set: DATASET_ROOT=/path/to/working/dir   (where the segmented CSVs will be written)

# 2. Download fine-tuned checkpoints from Zenodo
bash 01_download_zenodo.sh
# (The dataset archive is also downloaded but you can ignore it — your FASTAs
# override the Zenodo FASTAs because FASTA_DIR is set.)

# 3. Submit inference jobs
#    Step 02 will auto-segment your FASTAs into 2k/4k/8k CSVs on first run.
bash 02_submit_inference_jobs.sh

# 4. Aggregate into per-genome JSON files
#    --taxonomy is optional; if you have a taxonomy CSV in the simple format,
#    pass it for richer dashboard display.
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --output ${RESULTS_ROOT}/aggregated

# 5. (Optional) Annotate the predicted prophages with Pharokka PHROG categories
bash 04_annotate_predicted_prophages.sh \
    --website-data ${RESULTS_ROOT}/aggregated/website_data \
    --fasta-dir ${FASTA_DIR} \
    --pharokka-db /path/to/pharokka_db \
    --output-dir ${RESULTS_ROOT}/pharokka

# 6. (Optional) Re-aggregate with PHROG annotations folded in
python 03_build_website_data.py \
    --predictions ${RESULTS_ROOT} \
    --phrog ${RESULTS_ROOT}/pharokka/annotations.csv \
    --output ${RESULTS_ROOT}/aggregated
```

> **Tip:** if you'd rather see exactly how the FASTAs are split (e.g., to use a non-default stride), you can run `00_segment_fasta.py` explicitly before step 02; the auto-segmentation in step 02 is skipped when the segmented CSVs already exist.

---

## Schemas

### Segment input CSV (one per genome × window size)

Used by all gLM inference scripts and produced by `00_segment_fasta.py`. The minimum required columns are `sequence` and `label`; the rest are needed for the genome-wide pipeline.

| Column | Type | Required | Description |
|--------|------|:---:|-------------|
| `segment_id` | string | recommended | Unique identifier for the segment (e.g., `GCF_000007845.1_seg_00001`) |
| `seq_id` | string | genome-wide only | Contig name from the source FASTA |
| `start` | int | genome-wide only | 0-based start position in the contig |
| `end` | int | genome-wide only | 0-based end position in the contig (half-open) |
| `sequence` | string | ✓ | DNA sequence (A/C/G/T/N) |
| `label` | int | classification eval only | Ground-truth label (0 = bacteria, 1 = prophage); set to 0 if unknown |

### Predictions output CSV (one per input CSV per model)

This is what every gLM inference script produces, and what use-case-2 users must produce for their new model. Filename convention: `<input_basename>_predictions.csv`.

| Column | Type | Required | Description |
|--------|------|:---:|-------------|
| `segment_id` or `seq_id` | string | ✓ | Identifier, copied from the input CSV |
| `sequence` | string | optional | Copied from input; can be omitted to keep files small |
| `label` | int | optional | Ground-truth label from input (preserved if present) |
| `start`, `end` | int | genome-wide only | Copied from input |
| `prob_0` | float | ✓ | Predicted probability of class 0 (bacteria) |
| `prob_1` | float | ✓ | Predicted probability of class 1 (prophage) |
| `pred_label` | int | ✓ | Predicted class (0 or 1; threshold defaults to 0.5) |

### Ground-truth CSV

| Column | Type | Description |
|--------|------|-------------|
| `Assembly` | string | Assembly accession (e.g., `GCF_000007845.1`) |
| `NCBI Id` | string | Contig accession |
| `start` | int | Prophage start in the contig (0-based) |
| `end` | int | Prophage end in the contig (0-based half-open) |
| `Organism Name` | string | Human-readable organism name |

### Taxonomy CSV (two supported formats)

**Simple format (recommended for your-own-data):**

| Column | Description |
|--------|-------------|
| `assembly` | Assembly accession matching `Assembly` in ground truth |
| `phylum`, `class`, `order`, `family`, `genus`, `species` | One column per rank; leave blank if unknown |

**GTDB-Tk format:** the original `gtdbtk.bac120.summary.tsv` from a GTDB-Tk classification run. Parsed columns: `user_genome` and `classification` (the semicolon-delimited `p__...;c__...` string).

The script auto-detects which format you've provided by inspecting the columns.

### Output layout (what `02_submit_inference_jobs.sh` writes, what `03_build_website_data.py` reads)

```
RESULTS_ROOT/
└── <model>/                     # dnabert2, nucleotide_transformer_v2, ...
    └── <category>/              # binary, error_bias, genome_wide
        └── <window>/            # 2k, 4k, 8k
            ├── <input1>_predictions.csv
            ├── <input2>_predictions.csv
            └── ...
```

Any extra `<model>` you drop in here (use case 2) will be auto-discovered by step 03 — no code changes needed.

### Per-genome JSON output (what the dashboard reads)

`{assembly}_{window}.json`:

```jsonc
{
  "assembly": "GCF_000007845.1",
  "organism": "...",
  "genome_length": 2814816,
  "window_size": "2k",
  "taxonomy": {"phylum": "...", "class": "...", ...},
  "ground_truth": [{"start": 100000, "end": 145000}, ...],
  "metrics": {
    "DNABERT2 2k": {"mcc": 0.71, "precision": 0.82, "recall": 0.69, "f1": 0.75, "auc": 0.91},
    ...
  },
  "clustered_predictions": {
    "DNABERT2 2k": [{"start": 99500, "end": 146200}, ...],
    ...
  },
  "per_segment": {
    "DNABERT2 2k": [{"start": 0, "end": 2000, "prob_1": 0.04, "pred_label": 0, "label": 0}, ...],
    ...
  },
  "phrog_annotations": [],
  "checkv_quality": [],
  "model_categories": {"DNABERT2 2k": "genomic_lm", ...}
}
```

The dashboard also expects `index.json` next to the per-genome files; `03_build_website_data.py` writes it automatically.

---

## Coverage matrix

| Model | Binary (2k/4k/8k) | Error & bias | Genome-wide |
|-------|:-:|:-:|:-:|
| DNABERT-2 | ✓ | ✓ | ✓ (segment-based) |
| NT v2 | ✓ | ✓ | ✓ (segment-based) |
| Caduceus | ✓ | ✓ | ✓ (segment-based) |
| ProkBERT | ✓ | ✓ | ✓ (segment-based) |
| megaDNA | ✓ | ✓ | ✓ (segment-based) |
| GENERanno | ✓ | ✓ | ✓ (segment-based) |
| Evo2 (LP/NN) | ✓ | ✓ | ✓ (segment-based) |
| Evo2 + SAE | ✓ | ✓ | ✓ (native windowed scan) |

*Segment-based* genome-wide means: the genome is pre-split into overlapping segments (2k/4k/8k with 1k/2k/4k stride respectively), each segment is classified, and the per-segment predictions are then stitched back into a per-genome track in step 03. Evo2+SAE has a native windowed-scan pipeline (50 kb windows with MAX pooling) that produces per-position activation tracks directly — see `inference/evo2_sae/src/run_lambda_batch.py`.
