# LAMBDA simple-baseline results (response to reviewer)

This document records the results of three sequence-feature baselines
(GC content, k=4 k-mer composition, k=6 k-mer composition) on the full
LAMBDA benchmark, prepared in response to the reviewer request: *"Please
add stronger simple baselines in the first results section. GC-content
and k-mer based baselines would be very helpful. Use a classifier
trained on these seq features."*

Every baseline number reported here was produced by running the **exact
same scripts** that produced the gLM rows in paper Tables 2 and 3, so
the baselines and gLMs are directly comparable.

---

## Methodology

Three baselines, all using `sklearn.linear_model.LogisticRegression(max_iter=1000,
solver='lbfgs', random_state=42)` on `StandardScaler`-normalized features
— the same protocol as the gLM linear probes in the paper:

| Baseline | Features | # features |
|----------|----------|:-:|
| `baseline_gc`    | GC content                          | 1 |
| `baseline_kmer4` | Normalized tetranucleotide frequencies (`AAAA`...`TTTT`) | 256 |
| `baseline_kmer6` | Normalized hexanucleotide frequencies                    | 4 096 |

Implementation: `replication/baselines/run_baselines.py`. Trained on
`binary_segments_<window>/train.csv` for each window size (2k / 4k / 8k),
then evaluated on every test split.

## Reproduction map (which script produces which number)

| Paper column (Table 3) | Script that produced the gLM rows | Script + inputs for baselines |
|---|---|---|
| GC-bias (Shuffled MCC) | per-model JSON metrics on `gc_control_<window>_test` | `outputs/results/aggregated/metrics_summary.csv` (produced by `replication/baselines/run_baselines.py` + `03_build_website_data.py`) |
| FPR Bacteria-only      | per-model JSON metrics on `bacterial_cds_<window>`         | same source, `1 - specificity` of bacterial_cds row |
| FNR Phage-only         | per-model JSON metrics on `phage_segments_<window>_<stride>` | same source, `1 - recall` of phage_segments row |
| Peak MCC               | max of Fine-Tuned MCC / 3-Layer NN MCC                     | binary MCC of the LR baseline (no fine-tuning equivalent — see Table 3 note below) |
| **Raw FPR/Recall/MCC** | `FINAL_SCRIPTS/analyze_genome_wide_results.py` on `DATA/per_segment_<w>/<MODEL>/` → `<MODEL>_<w>_summary.csv` row `type=raw, method="average (mean per-genome)"` | same script run on `outputs/reproduction/staged_baselines/<baseline>_<w>/` → `outputs/reproduction/baseline_summaries/<baseline>_<w>_summary.csv` |
| **Filtered FPR/Recall/MCC** | `FINAL_SCRIPTS/grid_search_clustering.py` (v2 grid) → `grid_search_results_v2/grid_search_best.csv` (2k) and `grid_search_results_v2_4k8k/grid_search_best.csv` (4k/8k) | same script + identical grid run on `outputs/reproduction/grid_search_baselines_data/` → `outputs/reproduction/grid_search_baselines_v2grid/grid_search_best.csv` |

**Reproduction verified**: my v2 grid-search reproduction matches the
paper exactly for every regular gLM (DNABERT2 2k: macro_mcc=0.3914,
recall=0.4673, FPR=1.78% — same numbers, same hyperparameters as the
paper). EVO2_LP and EVO2_SAE differ because they have dedicated grid
scripts (`grid_search_evo2_lp.py`, `grid_search_evo2_sae.py`) and are
not searched by `grid_search_clustering.py`; the baselines do not use
those scripts.

## v2 grid used (matches paper)

```
--norm-methods   zscore
--smooth-windows 3 5 7
--thresholds     0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0
--min-sizes      8000 10000 12000 15000 18000 20000
--merge-gaps     1000 3000 5000 8000
```
576 combinations × 9 (baseline × window) = 5 184 evaluations.

---

## 1. Binary classification (Table 2 input — Linear Probe column)

| Baseline | 2k MCC | 4k MCC | 8k MCC | 8k AUC |
|----------|:------:|:------:|:------:|:------:|
| GC content   | 0.126 | 0.127 | 0.082 | 0.57 |
| $k=4$ k-mer  | 0.680 | 0.748 | 0.797 | 0.96 |
| $k=6$ k-mer  | 0.730 | 0.809 | **0.856** | **0.98** |

GC content alone barely beats chance. $k=6$ k-mer composition is the
strongest classical baseline; the gLM Linear Probe column in Table 2
must clear this line to claim added value at the binary task. Many do
(NTv2 0.846/0.924/0.951, GENERanno 0.932/0.967/0.979, EVO2 0.951/0.964/0.974);
DNABERT2 does not at the windows it covers (0.724/0.734).

## 2. Diagnostic splits (Table 3 input — fragment-classification columns)

| Baseline | Window | GC-bias (Shuffled MCC) | FPR Bacteria-only | FNR Phage-only |
|----------|:-:|:-:|:-:|:-:|
| GC content | 2k | 0.126 | 46.4% | 39.6% |
| GC content | 4k | 0.127 | 45.4% | 39.5% |
| GC content | 8k | 0.082 | 36.8% | 51.1% |
| $k=4$ k-mer | 2k | 0.115 | 16.7% | 17.2% |
| $k=4$ k-mer | 4k | 0.113 | 16.6% | 14.6% |
| $k=4$ k-mer | 8k | 0.109 | 19.2% | 14.1% |
| $k=6$ k-mer | 2k | 0.077 | 14.7% | 14.0% |
| $k=6$ k-mer | 4k | 0.078 | 14.8% | 10.7% |
| $k=6$ k-mer | 8k | 0.075 | 13.6% | 9.7% |

GC-shuffled MCC for all three baselines sits near 0.1 — the GC-only
ceiling on this dataset. Bacterial-CDS false-positive rate and
phage-only false-negative rate both drop sharply moving from GC → k=4
→ k=6, as expected. The k=6 baseline misses ~10% of phage segments
at 8k and wrongly calls ~14% of bacterial CDS regions phage.

**Peak MCC for baselines (Table 3 input).** Baselines have no
fine-tuning or 3-layer NN equivalent; the LR result is the strongest
classifier the hand-crafted features afford. We therefore report the
LR (binary) MCC in the Peak column (= same value as the Linear Probe
column for these rows): 0.126/0.127/0.082 for GC, 0.680/0.748/0.797
for k=4, 0.730/0.809/0.856 for k=6.

## 3. Genome-wide Raw (Table 3 input — paper's `analyze_genome_wide_results.py`)

Macro-averaged across 81–82 genomes (matches paper's "average (mean
per-genome)" row):

| Baseline | Window | Raw FPR (%) | Raw Recall (%) | Raw MCC |
|----------|:-:|:-:|:-:|:-:|
| GC content | 2k | 38.83 | 40.35 | 0.015 |
| GC content | 4k | 38.85 | 40.38 | 0.012 |
| GC content | 8k | 31.85 | 32.91 | 0.010 |
| $k=4$ k-mer | 2k | 32.03 | 46.63 | 0.059 |
| $k=4$ k-mer | 4k | 30.73 | 49.10 | 0.080 |
| $k=4$ k-mer | 8k | 26.80 | 49.41 | 0.113 |
| $k=6$ k-mer | 2k | 29.79 | 55.72 | 0.104 |
| $k=6$ k-mer | 4k | 27.24 | 58.92 | 0.138 |
| $k=6$ k-mer | 8k | 23.46 | 59.80 | 0.178 |

For comparison, paper Table 3 gLM Raw MCC at 2k: DNABERT-2 0.138,
NTv2 0.295, Caduceus 0.232, ProkBERT-mini 0.310, megaDNA 0.240,
GENERanno 0.314, **EVO2 0.439**. The k=6 baseline (0.104) is at the
floor; even DNABERT-2 (0.138) sits between k=6 2k (0.104) and k=6 4k
(0.138).

## 4. Genome-wide Filtered (Table 3 input — paper's `grid_search_clustering.py` v2 grid)

| Baseline | Window | Filtered FPR (%) | Filtered Recall (%) | Filtered MCC | norm | smooth | threshold | min_size | merge_gap |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GC content | 2k | 8.73 | 34.43 | 0.138 | zscore | 7 | 0.5 | 20 000 | 5 000 |
| GC content | 4k | 9.89 | 35.70 | 0.135 | zscore | 5 | 0.5 | 20 000 | 5 000 |
| GC content | 8k | 8.27 | 32.30 | 0.142 | zscore | 5 | 0.75 | 18 000 | 3 000 |
| $k=4$ k-mer | 2k | 2.88 | 30.82 | 0.205 | zscore | 7 | 0.5 | 20 000 | 1 000 |
| $k=4$ k-mer | 4k | 3.50 | 31.25 | 0.200 | zscore | 5 | 0.75 | 20 000 | 5 000 |
| $k=4$ k-mer | 8k | 4.26 | 33.64 | 0.210 | zscore | 3 | 0.75 | 20 000 | 1 000 |
| $k=6$ k-mer | 2k | 1.91 | 41.41 | 0.343 | zscore | 7 | 0.5 | 20 000 | 1 000 |
| $k=6$ k-mer | 4k | 1.78 | 42.02 | 0.367 | zscore | 7 | 0.75 | 20 000 | 1 000 |
| $k=6$ k-mer | 8k | 1.19 | 37.69 | **0.367** | zscore | 5 | 1.25 | 20 000 | 1 000 |

For comparison, paper Table 3 gLM Filtered MCC at 2k:
DNABERT-2 0.391, Caduceus 0.577, GENERanno 0.648, NTv2 0.647,
megaDNA 0.602, ProkBERT-mini 0.658, **EVO2 0.680**.

**The k=6 baseline (0.343–0.367 Filtered MCC) is close to DNABERT-2
(0.391) but well below the other gLMs (0.51–0.68).** GC content
remains near the binary-only floor (0.135–0.142). The same filtering
pipeline that lifts the gLMs by 0.2–0.3 MCC also lifts the baselines —
the k=6 baseline jumps from Raw 0.104 → Filtered 0.343 at 2k — so
"better than nothing" should not be confused with "competitive with
gLMs". The Filtered gap between the strongest baseline (k=6, ≤0.367)
and the strongest gLM (EVO2, 0.680) is the right operational margin to
quote in the manuscript.

---

## Suggested reviewer-response text

> Following the reviewer's suggestion, we added three sequence-feature
> baselines --- GC content, tetranucleotide ($k=4$) composition, and
> hexanucleotide ($k=6$) composition --- each fit with logistic
> regression using the same `StandardScaler` + `LogisticRegression`
> (`max_iter=1000`, `solver='lbfgs'`, `random_state=42`) protocol used
> for our gLM linear probes. Baseline rows have been added to
> Tables~\ref{tab:linear_probe_nn} and~\ref{tab:combined} and pass
> through the identical post-processing pipeline used for the gLM
> rows (`FINAL_SCRIPTS/analyze_genome_wide_results.py` for Raw
> genome-wide metrics; `FINAL_SCRIPTS/grid_search_clustering.py` for
> the post-processed/Filtered metrics, with the same hyperparameter
> grid: zscore normalization; smoothing $\in\{3,5,7\}$; threshold $\in\{0.5, 0.75, 1.0,
> 1.25, 1.5, 2.0, 2.5, 3.0\}$ standard deviations; minimum region size
> $\in\{8, 10, 12, 15, 18, 20\}$ kb; merge gap $\in\{1, 3, 5, 8\}$ kb).
>
> On the binary classification task, the $k=6$ k-mer baseline reaches
> $0.73 / 0.81 / 0.86$ MCC at 2k / 4k / 8k, comparable to the
> strongest gLM linear probes (NTv2 $0.85$--$0.95$, EVO2
> $0.95$--$0.97$, GENERanno $0.93$--$0.98$). GC content alone is
> barely better than chance (MCC $0.08$--$0.13$), so the k-mer
> baseline is the stronger reference floor.
>
> The picture inverts on the genome-wide prophage detection task. Raw
> macro-averaged MCC (paper Table~3, Raw columns) for the $k=6$ baseline is
> $0.10 / 0.14 / 0.18$ at 2k / 4k / 8k, comparable to or below the
> weakest gLM (DNABERT-2 at $0.138$ at 2k) and far below the
> strongest (EVO2 at $0.439$ at 2k). After the same post-processing
> applied to the gLMs (Filtered columns), the $k=6$ baseline reaches a
> ceiling of $0.343$--$0.367$ MCC, closing the gap to DNABERT-2
> ($0.391$) but leaving a $0.13$--$0.31$ MCC margin between the best
> baseline and the rest of the gLMs (Caduceus $0.577$, GENERanno
> $0.648$, NTv2 $0.647$, megaDNA $0.602$, ProkBERT-mini $0.658$, EVO2
> $0.680$). This is the right operational margin: simple
> compositional features can classify isolated phage segments in a
> class-balanced binary setting but cannot localize prophages within
> real bacterial chromosomes, which is the biologically meaningful
> task and the regime where the gLMs' added structure pays off.

---

## Files produced

```
outputs/results/aggregated/
├── metrics_summary.csv                          (baseline binary + diagnostic rows; produced on Biowulf)
└── baseline_pooled_genome_wide.csv              (initial pooled-MCC analysis -- superseded by the macro values below)

outputs/reproduction/
├── staged_baselines/                            (gLM-schema baseline JSONs + CSVs, ready for analyze_genome_wide_results.py)
├── baseline_summaries/
│   ├── <baseline>_<window>_summary.csv          (raw "average (mean per-genome)" rows = Table 3 Raw)
│   ├── <baseline>_<window>_individual.csv       (per-genome metrics)
│   └── baseline_raw_table3.csv                  (consolidated Table 3 Raw values)
├── grid_search_baselines_data/
│   └── per_segment_<w>/<baseline>/*.csv         (input layout for grid_search_clustering.py)
├── grid_search_baselines_v2grid/
│   ├── grid_search_results.csv                  (5184 rows, full v2 grid search)
│   └── grid_search_best.csv                     (best per baseline x window = Table 3 Filtered)
├── grid_search_glm_v2grid/                      (gLM 2k reproduction -- confirms my v2 grid matches paper)
│   ├── grid_search_results.csv
│   └── grid_search_best.csv
├── table2_baseline_rows.tex                     (paper-ready LaTeX rows for Table 2)
└── table3_baseline_rows.tex                     (paper-ready LaTeX rows for Table 3)
```

## Reproduction commands

```bash
# 1. Pool tp/tn/fp/fn into a single table (initial sanity check).
python replication/baselines/compute_pooled_genome_wide.py \
    --results_root outputs/results \
    --out_dir outputs/results/aggregated

# 2. Stage baseline data in the layouts the paper scripts expect.
python replication/baselines/stage_baselines_for_paper_scripts.py \
    --results_root outputs/results \
    --staged_root outputs/reproduction/staged_baselines
python replication/baselines/stage_baselines_for_grid_search.py \
    --results_root outputs/results \
    --staged_root outputs/reproduction/grid_search_baselines_data

# 3. Raw genome-wide metrics via the paper's analyze_genome_wide_results.py.
for b in baseline_gc baseline_kmer4 baseline_kmer6; do
  for w in 2k 4k 8k; do
    python /Users/leannmlindsey/WORK/CLAUDE_LAMBDA_VISUALIZE_RESULTS/FINAL_SCRIPTS/analyze_genome_wide_results.py \
        -d outputs/reproduction/staged_baselines/${b}_${w} \
        -m ${b}_${w} \
        -r outputs/reproduction/baseline_summaries
  done
done
python replication/baselines/extract_baseline_raw_table.py \
    --summaries_dir outputs/reproduction/baseline_summaries \
    --out_csv outputs/reproduction/baseline_summaries/baseline_raw_table3.csv

# 4. Filtered genome-wide metrics via the paper's grid_search_clustering.py
#    (identical v2 grid).
python /Users/leannmlindsey/WORK/CLAUDE_LAMBDA_VISUALIZE_RESULTS/FINAL_SCRIPTS/grid_search_clustering.py \
    --data-dir outputs/reproduction/grid_search_baselines_data \
    --gt /Users/leannmlindsey/WORK/CLAUDE_LAMBDA_VISUALIZE_RESULTS/DATA/lambda_ground_truth.csv \
    --output-dir outputs/reproduction/grid_search_baselines_v2grid \
    --no-heatmaps \
    --norm-methods zscore \
    --smooth-windows 3 5 7 \
    --thresholds 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0 \
    --min-sizes 8000 10000 12000 15000 18000 20000 \
    --merge-gaps 1000 3000 5000 8000
```
