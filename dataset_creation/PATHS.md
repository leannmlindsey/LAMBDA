# LAMBDA Project - Important Paths Reference

> **Note on path prefixes:**
> - `/projects/bfzj/` - Shared project storage (persistent)
> - `/work/hdd/bfzj/` - High-capacity work storage (for large intermediate files)
> - `/u/llindsey1/` - User home/scratch directories

---

## 1. Source Data

### INPHARED (Phage Data)
| Description | Path |
|-------------|------|
| Base directory | `/u/llindsey1/llindsey/black_and_white/data/inphared/` |
| Genomes FASTA | `/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa` |
| vclust clustering | `/projects/bfzj/llindsey1/black_and_white/data/inphared/vclust_info.tsv` |
| Pharokka annotations (all) | `/projects/bfzj/llindsey1/black_and_white/data/inphared/pharokka/` |

### GTDB (Bacteria Data)
| Description | Path |
|-------------|------|
| Base directory | `/projects/bfzj/llindsey1/black_and_white/data/gtdb/` |
| Metadata file | `/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata/bac120_metadata.tsv` |
| Representative genomes | `/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226/` |

### External Tools
| Description | Path |
|-------------|------|
| Bakta database | `/u/llindsey1/scratch/bakta_db/db` |

---

## 2. BLAST Databases

| Description | Path | Created by |
|-------------|------|------------|
| INPHARED phages | `/u/llindsey1/llindsey/black_and_white/data/inphared/blastdb/inphared_phages` | `create_phage_blastdb.slurm` |
| GTDB selected bacteria | `/work/hdd/bfzj/llindsey1/gtdb_blastdb_selected/gtdb_bacteria_selected` | `create_gtdb_selected_blastdb.slurm` |
| GTDB full bacteria | `/work/hdd/bfzj/llindsey1/gtdb_blastdb/gtdb_bacteria` | `create_gtdb_blastdb.slurm` |

---

## 3. Scripts Directory

| Description | Path |
|-------------|------|
| Lambda dataset tools | `/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/` |

---

## 4. Generated Datasets for Fine-Tuning

> All dataset directories are under: `/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/`

### 4.1 Phage Segments (INPHARED)
| Length | Directory | Created by |
|--------|-----------|------------|
| 2k | `inphared_dataset/` | `subsample_inphared.slurm` |
| 4k | `inphared_dataset_4k/` | `subsample_inphared_4k.slurm` |
| 8k | `inphared_dataset_8k/` | `subsample_inphared_8k.slurm` |

**Contents per directory:**
- `train_accessions.txt`, `dev_accessions.txt`, `test_accessions.txt`
- `train_segments.fasta`, `dev_segments.fasta`, `test_segments.fasta`
- `train_segments.tsv`, `dev_segments.tsv`, `test_segments.tsv`

### 4.2 Bacteria Segments - Filtered v3 (prophage-free) ✓ RECOMMENDED
| Length | Directory | Created by |
|--------|-----------|------------|
| 2k | `gtdb_dataset_filtered_v3/` | `filter_and_subsample_gtdb_v3.slurm` |
| 4k | `gtdb_dataset_filtered_v3_4k/` | `filter_and_subsample_gtdb_v3.slurm` |
| 8k | `gtdb_dataset_filtered_v3_8k/` | `filter_and_subsample_gtdb_v3.slurm` |

**Additional files in 2k directory:**
- `contaminated_accessions.txt` - Genomes removed due to prophage hits (973 total)
- `train_accessions.txt`, `dev_accessions.txt`, `test_accessions.txt` - Clean genomes only

**Post-processing:** Run `remove_contaminated_segments.sh` to remove 2 additional genomes discovered via validation.

### 4.3 Bacteria Segments - Unfiltered (with prophage contamination)
| Length | Directory | Created by |
|--------|-----------|------------|
| 2k | `gtdb_dataset/` | `subsample_gtdb.slurm` |
| 4k | `gtdb_dataset_4k/` | `subsample_gtdb_4k.slurm` |
| 8k | `gtdb_dataset_8k/` | `subsample_gtdb_8k.slurm` |

> ⚠️ **FOR COMPARISON ONLY** - Contains prophage sequences. Use filtered v3 for training.

---

## 5. Control Datasets

### 5.1 GC-Content Control (Shuffled Test Segments)
| Description | Path | Created by |
|-------------|------|------------|
| Output directory | `shuffled_gc_control/` | `shuffle_test_segments.slurm` |

**Output files:**
- `inphared_2k_test_shuffled.fasta`, `inphared_4k_test_shuffled.fasta`, `inphared_8k_test_shuffled.fasta`
- `gtdb_2k_test_shuffled.fasta`, `gtdb_4k_test_shuffled.fasta`, `gtdb_8k_test_shuffled.fasta`

---

## 6. Benchmark-Specific Datasets (Full Genomes + Annotations)

> These datasets are for benchmarking, not fine-tuning. They use full genomes with functional annotations.

### 6.1 Phage-Only (869 Test Set Genomes)
| Description | Path | Created by |
|-------------|------|------------|
| Accession list | `inphared_dataset/test_accessions.txt` | `subsample_inphared.slurm` |
| Full genomes | `/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa` | - |
| Pharokka annotations (test only) | `phage_only_annotations/` | `copy_pharokka_test_annotations.sh` |

**For Zenodo upload:** Run `copy_pharokka_test_annotations.sh` to copy only test set annotations.

### 6.2 Bacteria-Only (100 Taxonomically Balanced)
| Description | Path | Created by |
|-------------|------|------------|
| Accession list | `bacteria_only_dataset/bacteria_only_100_accessions.txt` | `select_bacteria_only_balanced.slurm` |
| Metadata | `bacteria_only_dataset/bacteria_only_100_metadata.tsv` | `select_bacteria_only_balanced.slurm` |
| Full genomes | `/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226/` | - |
| Bakta annotations | `bacteria_only_annotations/` | `run_bakta_bacteria.slurm` |

### 6.3 Bacterial CDS Benchmark (Verified Non-Phage)
| Description | Path | Created by |
|-------------|------|------------|
| CDS 500-2000bp | `bacteria_cds_benchmark/bacterial_cds_2k.csv` | `extract_bacterial_cds.sh` |
| CDS 1000-4000bp | `bacteria_cds_benchmark/bacterial_cds_4k.csv` | `extract_bacterial_cds.sh` |
| CDS 2000-8000bp | `bacteria_cds_benchmark/bacterial_cds_8k.csv` | `extract_bacterial_cds.sh` |

**Filtering criteria:**
- Excludes CDS with phage-related annotations (capsid, tail, integrase, etc.)
- Excludes CDS overlapping BLAST hits (≥90% identity AND ≥200bp)
- CSV format: `segment_id, sequence, label (0), source, product, length`

---

## 7. Merged Datasets

| Description | Path | Created by |
|-------------|------|------------|
| Output directory | `merged_datasets/` | `merge_datasets.slurm` |

---

## 8. Intermediate/Working Files

### 8.1 BLAST Results
| Description | Path | Created by |
|-------------|------|------------|
| Prophage BLAST v3 results | `/work/hdd/bfzj/llindsey1/prophage_blast_results_v3/` | `blast_phage_vs_gtdb_selected_v3.slurm` |
| Combined filtered results | `/work/hdd/bfzj/llindsey1/prophage_blast_results_v3/blast_filtered_combined.tsv` | `blast_phage_vs_gtdb_selected_v3.slurm` |
| Contig-to-genome map | `/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv` | `create_contig_map.slurm` |

### 8.2 Validation Results
| Description | Path | Created by |
|-------------|------|------------|
| Validation directory | `/work/hdd/bfzj/llindsey1/prophage_validation_v3/` | `validate_prophage_removal_v3.slurm` |
| Validation BLAST raw | `/work/hdd/bfzj/llindsey1/prophage_validation_v3/validation_blast_raw.tsv` | `validate_prophage_removal_v3.slurm` |
| Validation BLAST filtered | `/work/hdd/bfzj/llindsey1/prophage_validation_v3/validation_blast_filtered.tsv` | `validate_prophage_removal_v3.slurm` |

---

## 9. Key Files Reference

### INPHARED vclust_info.tsv
- **Path:** `/projects/bfzj/llindsey1/black_and_white/data/inphared/vclust_info.tsv`
- Contains cluster assignments for all phage genomes (95% ANI clustering)
- First genome in each cluster is the representative
- Used by `extract_representatives.py` to create train/dev/test splits
- **Total genomes:** 28,665 → **8,683 clusters**

### GTDB bac120_metadata.tsv
- **Path:** `/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata/bac120_metadata.tsv`
- Contains metadata for all bacterial genomes
- Includes taxonomy, CheckM2 quality scores, genome size
- Used by `select_gtdb_representatives.py` to select high-quality representatives
- **Selected representatives:** 15,865 → **14,892 after prophage filtering**

---

## 10. Quick Reference - Common Paths

```bash
# Scripts
SCRIPT_DIR="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools"

# Source data
INPHARED_FASTA="/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa"
INPHARED_VCLUST="/projects/bfzj/llindsey1/black_and_white/data/inphared/vclust_info.tsv"
GTDB_GENOMES="/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226"
GTDB_METADATA="/projects/bfzj/llindsey1/black_and_white/data/gtdb/metadata/bac120_metadata.tsv"

# BLAST databases
PHAGE_BLASTDB="/u/llindsey1/llindsey/black_and_white/data/inphared/blastdb/inphared_phages"
BACTERIA_BLASTDB="/work/hdd/bfzj/llindsey1/gtdb_blastdb_selected/gtdb_bacteria_selected"

# Recommended datasets for fine-tuning (prophage-free)
PHAGE_2K="${SCRIPT_DIR}/inphared_dataset"
PHAGE_4K="${SCRIPT_DIR}/inphared_dataset_4k"
PHAGE_8K="${SCRIPT_DIR}/inphared_dataset_8k"
BACTERIA_2K="${SCRIPT_DIR}/gtdb_dataset_filtered_v3"
BACTERIA_4K="${SCRIPT_DIR}/gtdb_dataset_filtered_v3_4k"
BACTERIA_8K="${SCRIPT_DIR}/gtdb_dataset_filtered_v3_8k"

# Unfiltered bacteria (for comparison testing)
BACTERIA_2K_UNFILTERED="${SCRIPT_DIR}/gtdb_dataset"
BACTERIA_4K_UNFILTERED="${SCRIPT_DIR}/gtdb_dataset_4k"
BACTERIA_8K_UNFILTERED="${SCRIPT_DIR}/gtdb_dataset_8k"

# GC-content control
GC_CONTROL="${SCRIPT_DIR}/shuffled_gc_control"

# Benchmark datasets
PHAGE_ONLY_ANNOTATIONS="${SCRIPT_DIR}/phage_only_annotations"
BACTERIA_ONLY_ANNOTATIONS="${SCRIPT_DIR}/bacteria_only_annotations"

# Bakta
BAKTA_DB="/u/llindsey1/scratch/bakta_db/db"
```

---

## 11. Dataset Statistics Summary

| Dataset | Train | Dev | Test | Total Genomes |
|---------|------:|----:|-----:|--------------:|
| **Phage (INPHARED)** | 27,661 | 3,703 | 3,427 | 7,044 |
| **Bacteria (filtered v3)** | 27,000 | 3,400 | 3,400 | 14,892 |
| **Bacteria (unfiltered)** | 27,000 | 3,400 | 3,400 | 15,865 |

| Benchmark Dataset | Genomes |
|-------------------|--------:|
| **Phage-Only** | 869 (test set) |
| **Bacteria-Only** | 100 (balanced) |
