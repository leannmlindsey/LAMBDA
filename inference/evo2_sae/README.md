# Evo2 / Evo2+SAE Inference

Run Evo2-based inference on the LAMBDA benchmark — three pathways are supported:
1. **Evo2 + linear probe** (`evo2_lp_inference.py`) — trained linear classifier on Evo2 embeddings.
2. **Evo2 + 3-layer NN** (`evo2_nn_inference.py`) — trained NN classifier on Evo2 embeddings.
3. **Evo2 + SAE feature** (`sae_inference.py`, `run_lambda_batch.py`) — zero-shot prophage detection using the Goodfire SAE feature f/19746 on Evo2 layer-26 activations.

**Upstream repositories:**
- Evo2 itself: [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)
- LAMBDA-specific inference + SAE: [Evo2_SAE_LAMBDA_assessment](https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment)

## Files

| Path | Purpose |
|------|---------|
| `src/sae_inference.py` | SAE feature extraction on short (~2 kb) segments |
| `src/run_lambda_batch.py` | Genome-wide windowed SAE scanning (1–10 Mb genomes) |
| `src/cluster_activations.py` | Convert per-position activations into discrete predicted prophage regions |
| `src/nucleotide_evaluation.py` | Nucleotide-level evaluation from stitched segment activations |
| `src/evo2_lp_inference.py`, `src/evo2_nn_inference.py` | Trained-classifier inference on Evo2 embeddings |
| `src/batch_inference.py` | Batch driver |
| `src/calculate_metrics.py` | Metrics from prediction CSVs |
| `src/generate_lambda_plots.py`, `src/analyze_performance_factors.py`, `src/create_categorized_pdfs.py`, `src/analyze_segment_activations.py` | Visualization and analysis |
| `environment.yml` | Analysis-only conda env (no GPU). For inference itself, install Evo2 separately. |
| `scripts/` | Bash wrappers and SLURM scripts |
| `METHODS_SUMMARY.md` | Technical notes on SAE extraction and post-processing |

## Quick start (SAE on short segments)

```bash
python src/sae_inference.py \
    --input_csv /path/to/lambda_2k_test.csv \
    --output sae_results.csv \
    --max_threshold 0.5 \
    --save_activations
```

## Genome-wide scan (full pipeline)

```bash
# 1. Scan genomes (needs GPU + Evo2)
python src/run_lambda_batch.py \
    --fasta_dir /path/to/LAMBDA/FASTA \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./lambda_results_7b

# 2. Cluster activations into prophage regions (CPU)
python src/cluster_activations.py \
    --results_dir ./lambda_results_7b \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./clustering_results \
    --normalize zscore --threshold 7.0 \
    --max_gap 300 --merge_distance 5000 --min_region_size 1000
```

The SAE weights (`sae-layer26-mixed-expansion_8-k_64.pt`) are downloaded automatically from [Goodfire/Evo-2-Layer-26-Mixed](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed) on first run. See the [upstream README](https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment) for the full pipeline including PDF reports and performance-factor analysis.
