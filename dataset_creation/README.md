# LAMBDA Dataset Construction

Scripts used to build the LAMBDA benchmark dataset from public sources (GTDB representative bacterial genomes + INPHARED phage genomes).

These are the working scripts; the final assembled dataset is hosted on Zenodo: [10.5281/zenodo.19236553](https://doi.org/10.5281/zenodo.19236553). Use the scripts here if you want to reproduce the dataset construction or apply the same pipeline to a different source corpus.

**Upstream repository:** [lambda_dataset_tools](https://github.com/leannmlindsey/lambda_dataset_tools)

## What's included

The scripts construct the LAMBDA (LAnguage Model Bacteriophage Detection Assessment) benchmark datasets:
- Downloading and processing the INPHARED phage genome database
- Selecting and filtering GTDB bacterial representative genomes
- **Removing prophage-contaminated bacterial genomes** (BLAST-based filtering)
- Creating balanced phage/bacterial datasets with cluster-aware train/dev/test splits
- Subsampling genome segments at 2k, 4k, and 8k lengths
- Generating GC-content-matched control sequences (shuffled nucleotides)
- Creating taxonomically balanced annotation datasets (Phage-Only, Bacteria-Only)

> See [`PATHS.md`](PATHS.md) for the full reference of data paths and file locations, and [`METHODS_DATA_SELECTION.md`](METHODS_DATA_SELECTION.md) for a narrative description of the selection methodology.

## Pipeline overview

1. **Source download** — fetch GTDB bacterial representatives and INPHARED phage genomes
   - `download_inphared.py` / `download_inphared.slurm`
   - `select_gtdb_representatives.py` / `extract_representatives.py`

2. **Prophage identification & removal** — flag bacterial genomes that already contain prophages, and remove contaminated regions
   - `find_prophage_free_bacteria.py` / `select_prophage_free_bacteria.sh`
   - `validate_prophage_removal_v3.slurm`
   - `remove_contaminated_segments.sh`

3. **BLAST cross-validation** — confirm phage-vs-bacteria sequence separation by BLAST
   - `create_phage_blastdb.slurm`, `create_gtdb_blastdb.slurm`, `create_gtdb_selected_blastdb.slurm`
   - `blast_phage_vs_gtdb_selected_v3.slurm`
   - `analyze_blast_filtering.sh` / `.slurm`
   - `investigate_missed_hits.sh` / `.slurm`

4. **Segment extraction & subsampling** — slice genomes into 2k / 4k / 8k segments and subsample for class balance
   - `subsample_gtdb.slurm`, `subsample_gtdb_4k.slurm`, `subsample_gtdb_8k.slurm`
   - `subsample_inphared.slurm`, `subsample_inphared_4k.slurm`, `subsample_inphared_8k.slurm`
   - `subsample_gtdb_segments.py`, `subsample_segments.py`
   - `select_bacteria_only_balanced.py` / `.slurm`

5. **Diversity filtering** — Mash distances to ensure genome-level diversity
   - `run_mash_distances.slurm`
   - `cluster_mash_distances.py`

6. **CSV assembly** — produce train/dev/test CSVs and metadata maps
   - `create_training_csv.py` / `create_all_training_csvs.sh`
   - `create_contig_to_genome_map.py` / `create_contig_map.slurm`
   - `merge_and_shuffle.py`, `merge_datasets.slurm`, `shuffle_segments.py`, `shuffle_test_segments.slurm`
   - `create_gc_control_csvs.sh` — build the GC-matched control split
   - `create_lambda_final.sh` — orchestration

7. **Annotation (for the genome-wide split)** — Bakta annotation of test genomes
   - `run_bakta_bacteria.slurm`, `test_bakta_single.slurm`
   - `extract_bacterial_cds.py` / `.sh`

8. **Metrics & QC** — final prophage-detection metric calculation and timing analyses
   - `calculate_prophage_metrics.py`
   - `plot_processing_time.py`

## Quick start

```bash
# Environment
conda env create -f environment.yml
pip install -r requirements.txt

# 1. Create phage segments
sbatch subsample_inphared.slurm

# 2. Create prophage-filtered bacterial segments
sbatch blast_phage_vs_gtdb_selected_v3.slurm
sbatch filter_and_subsample_gtdb_v3.slurm

# 3. Validate prophage removal
sbatch validate_prophage_removal_v3.slurm

# 4. Assemble final CSVs
bash create_lambda_final.sh
```
