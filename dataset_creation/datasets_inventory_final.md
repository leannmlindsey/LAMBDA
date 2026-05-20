# LAMBDA Datasets - Final Inventory

## Dataset Summary

### Source Data
| Source | Version | Total Genomes | Description |
|--------|---------|---------------|-------------|
| INPHARED | Apr 2025 | ~8,683 accessions | Phage genomes |
| GTDB | r226 | 15,865 selected | Taxonomically balanced bacteria |
| GTDB Filtered | r226 (v3) | 14,894 clean | Prophage-free bacteria (971 removed) |

---

## Phage Datasets (INPHARED)

| Length | Directory | Train | Dev | Test | Genomes Used |
|--------|-----------|------:|----:|-----:|-------------:|
| **2k** | `inphared_dataset/` | 27,661 | 3,703 | 3,427 | 5,608 / 716 / 720 |
| **4k** | `inphared_dataset_4k/` | ✓ | ✓ | ✓ | same |
| **8k** | `inphared_dataset_8k/` | ✓ | ✓ | ✓ | same |

**Base path:** `/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/`

**Files per directory:**
- `{train,dev,test}_segments.fasta` - Segment sequences
- `{train,dev,test}_segments.tsv` - Segment metadata
- `{train,dev,test}_accessions.txt` - Genome accessions used

---

## Bacteria Datasets - Original (WITH prophage contamination)

| Length | Directory | Train | Dev | Test | Genomes Used |
|--------|-----------|------:|----:|-----:|-------------:|
| **2k** | `gtdb_dataset/` | 27,661 | 3,703 | 3,427 | 12,692 / 1,586 / 1,587 |
| **4k** | `gtdb_dataset_4k/` | ✓ | ✓ | ✓ | same |
| **8k** | `gtdb_dataset_8k/` | ✓ | ✓ | ✓ | same |

**⚠️ NOT RECOMMENDED FOR BENCHMARK** - Contains prophage sequences

**Base path:** `/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/`

---

## Bacteria Datasets - Filtered v3 (Prophage-free) ✓ RECOMMENDED

| Length | Directory | Train | Dev | Test | Genomes Used |
|--------|-----------|------:|----:|-----:|-------------:|
| **2k** | `gtdb_dataset_filtered_v3/` | 27,000 | 3,400 | 3,400 | 11,927 / 1,474 / 1,493 |
| **4k** | `gtdb_dataset_filtered_v3_4k/` | TBD | TBD | TBD | same accessions |
| **8k** | `gtdb_dataset_filtered_v3_8k/` | TBD | TBD | TBD | same accessions |

**Base path:** `/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/`

**Filtering Statistics (v3):**
- Original genomes: 15,865
- Contaminated (removed): 971 (6.1%)
- Clean (retained): 14,894 (93.9%)
- BLAST hits (≥200bp, ≥90% identity): 52,503
- Unique phages with bacterial hits: 3,632

---

## Prophage Filtering Parameters

| Parameter | Value |
|-----------|-------|
| Min alignment length | ≥200 bp |
| Min identity | ≥90% |
| BLAST max_target_seqs | 2,000 |
| E-value threshold | 1e-5 |

---

## Full Paths Reference

```
# Phage segments
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset_4k/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/inphared_dataset_8k/

# Bacteria segments - Original (with prophage)
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_4k/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_8k/

# Bacteria segments - Filtered v3 (prophage-free) ✓ USE THESE
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3_4k/
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3_8k/

# Accession lists
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3/train_accessions.txt
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3/contaminated_accessions.txt

# GC-Content Control (shuffled test segments)
/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/shuffled_gc_control/

# Source data
/u/llindsey1/llindsey/black_and_white/data/inphared/14Apr2025_genomes.fa
/u/llindsey1/llindsey/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226/
```

---

## GC-Content Control Dataset

| Length | Input Dataset | Output File | Status |
|--------|---------------|-------------|--------|
| **2k** | `inphared_dataset/test_segments.fasta` | `shuffled_gc_control/inphared_2k_test_shuffled.fasta` | Ready to run |
| **2k** | `gtdb_dataset_filtered_v3/test_segments.fasta` | `shuffled_gc_control/gtdb_2k_test_shuffled.fasta` | Ready to run |
| **4k** | `inphared_dataset_4k/test_segments.fasta` | `shuffled_gc_control/inphared_4k_test_shuffled.fasta` | Ready to run |
| **4k** | `gtdb_dataset_filtered_v3_4k/test_segments.fasta` | `shuffled_gc_control/gtdb_4k_test_shuffled.fasta` | Ready to run |
| **8k** | `inphared_dataset_8k/test_segments.fasta` | `shuffled_gc_control/inphared_8k_test_shuffled.fasta` | Ready to run |
| **8k** | `gtdb_dataset_filtered_v3_8k/test_segments.fasta` | `shuffled_gc_control/gtdb_8k_test_shuffled.fasta` | Ready to run |

**Script:** `shuffle_test_segments.slurm` (uses `shuffle_segments.py`)
**Method:** Shuffles nucleotides within each segment, preserving GC content but destroying sequence patterns

---

## Phage-Only Dataset (Full Genomes with Annotations)

| Description | Path | Status |
|-------------|------|--------|
| Accession list | `inphared_dataset/test_accessions.txt` | EXISTS (869 genomes) |
| Full genome sequences | `/projects/bfzj/llindsey1/black_and_white/data/inphared/14Apr2025_genomes.fa` | EXISTS |
| Pharokka annotations | `/projects/bfzj/llindsey1/black_and_white/data/inphared/pharokka/` | EXISTS |

**Method:** Uses fine-tuning test set genomes (869) with Pharokka functional annotations
**Purpose:** Analyze which functional categories carry strongest phage-associated signal

---

## Bacteria-Only Dataset (Full Genomes with Annotations)

| Description | Path | Status |
|-------------|------|--------|
| Accession list (100 balanced) | `bacteria_only_dataset/bacteria_only_100_accessions.txt` | Script ready |
| Metadata | `bacteria_only_dataset/bacteria_only_100_metadata.tsv` | Script ready |
| Full genome sequences | `/projects/bfzj/llindsey1/black_and_white/data/gtdb/fna/gtdb_genomes_reps_r226/` | EXISTS |
| Bakta annotations | `bacteria_only_annotations/` | Script ready |

**Scripts:**
1. `select_bacteria_only_balanced.slurm` - Select 100 taxonomically balanced bacteria
2. `test_bakta_single.slurm` - Test Bakta on 1 genome first
3. `run_bakta_bacteria.slurm` - Run Bakta on all 100 (array job)

**Method:** Taxonomically balanced subset (100 genomes) from test set with Bakta annotations
**Purpose:** Analyze which bacterial functional categories are most frequently misclassified as phage

---

## Datasets Still Needed

| Dataset | Status | Notes |
|---------|--------|-------|
| GTDB filtered 4k | Running? | Check `gtdb_dataset_filtered_v3_4k/` |
| GTDB filtered 8k | Running? | Check `gtdb_dataset_filtered_v3_8k/` |
| Shuffled GC Control | Script ready | Run `shuffle_test_segments.slurm` |
| Bacteria-Only annotations | Script ready | Run `run_bakta_bacteria.slurm` (need Bakta DB) |
| Prophage Boundary Detection | Not defined | Need to specify data source |
| Validation v3 | Running | `validate_prophage_removal_v3.slurm` |
