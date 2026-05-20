# LAMBDA Benchmark Dataset Construction Methods

## Overview

The LAMBDA benchmark datasets were constructed to train and evaluate genomic language models for prophage detection. The datasets consist of phage genomic sequences (positive class) and bacterial genomic sequences (negative class), balanced at a 50:50 ratio to enable effective discrimination learning.

## Data Sources

### Phage Genomes (INPHARED)
- **Source**: INPHARED database (https://github.com/RyanCook94/inphared)
- **Version**: 14 April 2025 release
- **Initial size**: ~47,000 complete phage genomes
- **Clustering**: Pre-computed using vclust at approximately 95% average nucleotide identity (ANI)
- **Representatives**: 8,684 cluster representative genomes
- **Annotations**: Gene predictions from Pharokka for all representative genomes (GenBank format)

### Bacterial Genomes (GTDB)
- **Source**: Genome Taxonomy Database (GTDB) Release 220
- **Version**: bac120 (Bacterial) metadata
- **Initial size**: 715,230 bacterial genomes
- **Representatives**: 136,646 species cluster representatives

## Phage Dataset Selection (INPHARED)

### Clustering
- **Method**: vclust clustering at approximately 95% average nucleotide identity (ANI)
- **Representatives**: 8,684 cluster representatives (first genome in each cluster designated as representative)
- **No quality filtering**: INPHARED genomes are curated complete phage genomes; no additional CheckM-based filtering was applied

### Inclusion Criteria
- Minimum genome length for segment sampling: 5,000 bp
- All 8,684 cluster representatives were retained to maximize phage diversity

### Rationale
Cluster-based selection ensures that highly similar phages (>95% ANI) are represented by a single genome, preventing redundancy while maintaining broad coverage of phage sequence diversity.

## Bacterial Dataset Selection (GTDB)

### Quality Filtering
Unlike the curated INPHARED phage database, GTDB contains genomes of varying quality. Quality filtering was applied:
- **Completeness**: ≥95% (CheckM2)
- **Contamination**: ≤5% (CheckM2)
- **Representative status**: Only GTDB species cluster representatives were considered
- **Result**: 67,193 high-quality representative genomes passed quality filters

### Taxonomic Subsampling
To ensure broad taxonomic representation while creating a balanced dataset:
1. Genomes were grouped by GTDB genus assignment
2. One genome was randomly selected per genus
3. **Result**: 15,865 genomes representing 15,865 unique genera

### Segment Balancing
To achieve 50:50 balance between phage and bacterial segments:
1. Segment sampling rate was adjusted for bacterial genomes
2. Each bacterial genome contributes 1-2 segments (vs. ~3 segments per phage genome)
3. This reflects the larger size of bacterial genomes (~3-5 Mb) compared to phage genomes (~30 kb average)

## Train/Dev/Test Splitting

### Cluster-Aware Splitting
To prevent data leakage, splits were performed at the cluster level (phage) or genus level (bacterial), ensuring that similar sequences do not appear in multiple splits:

- **Train**: 80% of clusters/genera
- **Development**: 10% of clusters/genera
- **Test**: 10% of clusters/genera

### Verification
No cluster/genus overlap exists between splits, verified programmatically after selection.

## Segment Subsampling

### Rationale
Genomic language models typically operate on fixed-length input sequences. To create training examples while fairly representing genomes of varying sizes:

### Phage Segment Sampling
- **Segment length**: 2,000 nucleotides
- **Sampling rate**: 1 segment per 10,000 bp of genome length
- **Minimum segments**: 1 per genome (for genomes 5-10 kb)
- **Exclusion**: Genomes < 5,000 bp excluded
- **Placement**: Random, non-overlapping positions within each genome
- **Coverage**: ~20% of each genome sampled

| Genome Size | Segments | Coverage |
|-------------|----------|----------|
| 5-10 kb | 1 | ~20% |
| 30 kb (avg) | 3 | ~20% |
| 100 kb | 10 | ~20% |
| 200 kb (jumbo) | 20 | ~20% |

### Bacterial Segment Sampling
- **Segment length**: 2,000 nucleotides
- **Total segments**: Matched to phage segment count for 50:50 balance
- **Distribution**: Evenly distributed across all genera (~1-2 segments per genome)
- **Placement**: Random, non-overlapping positions

## Shuffled Control Dataset

### Purpose
To create negative controls that preserve sequence composition (GC content, length) but destroy functional sequence features.

### Method
1. Parse CDS (coding sequence) annotations from Pharokka GenBank files
2. For each CDS region, shuffle nucleotides randomly within the CDS boundaries
3. Optionally shuffle intergenic regions
4. Maintain original genome structure (gene order, gene boundaries preserved)

### Properties Preserved
- Genome length
- GC content (identical to original)
- Gene boundaries and order
- Coding density

### Properties Destroyed
- Sequence similarity to any known sequence
- Codon structure and reading frames
- Functional motifs and regulatory elements

## Dataset Statistics

### Phage (INPHARED)
| Split | Genomes | Clusters | Segments (est.) |
|-------|---------|----------|-----------------|
| Train | ~6,947 | ~6,947 | ~20,800 |
| Dev | ~868 | ~868 | ~2,600 |
| Test | ~869 | ~869 | ~2,600 |
| **Total** | **8,684** | **8,684** | **~26,000** |

### Bacterial (GTDB)
| Split | Genomes | Genera | Segments (est.) |
|-------|---------|--------|-----------------|
| Train | ~12,692 | ~12,692 | ~20,800 |
| Dev | ~1,586 | ~1,586 | ~2,600 |
| Test | ~1,587 | ~1,587 | ~2,600 |
| **Total** | **15,865** | **15,865** | **~26,000** |

### Combined Dataset
| Split | Phage Segments | Bacterial Segments | Total | Balance |
|-------|----------------|-------------------|-------|---------|
| Train | ~20,800 | ~20,800 | ~41,600 | 50:50 |
| Dev | ~2,600 | ~2,600 | ~5,200 | 50:50 |
| Test | ~2,600 | ~2,600 | ~5,200 | 50:50 |

## Data Leakage Prevention

Different strategies were employed for each dataset to prevent data leakage between training and evaluation sets:

### Phage Dataset
1. **Sequence clustering**: Genomes pre-clustered at 95% ANI using vclust
2. **Representative selection**: Only one genome per cluster included in dataset
3. **Cluster-aware splitting**: Entire clusters assigned to a single split (train, dev, or test)
4. **Verification**: Zero cluster overlap between splits verified programmatically

### Bacterial Dataset
1. **Taxonomic grouping**: Genomes grouped by GTDB genus assignment
2. **One-per-genus selection**: Only one genome per genus included in dataset
3. **Genus-aware splitting**: Entire genera assigned to a single split (train, dev, or test)
4. **Verification**: Zero genus overlap between splits verified programmatically

### Cross-Dataset Considerations
Phage and bacterial datasets are inherently distinct sequence types, eliminating concerns about cross-contamination between positive and negative classes.

## Reproducibility

- **Random seed**: 42 (used for all random operations)
- **Scripts**: Available at [repository URL]
- **Data versions**: INPHARED 14Apr2025, GTDB r220

## Software and Tools

| Tool | Version | Purpose |
|------|---------|---------|
| vclust | - | Phage genome clustering |
| Pharokka | - | Phage genome annotation |
| CheckM2 | - | Bacterial genome quality assessment |
| GTDB-Tk | r220 | Bacterial taxonomy assignment |
| BioPython | ≥1.79 | Sequence parsing and manipulation |

## References

1. Cook R, et al. (2021). INfrastructure for a PHAge REference Database: Identification of Large-Scale Biases in the Current Collection of Cultured Phage Genomes. PHAGE.
2. Parks DH, et al. (2022). GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. Nucleic Acids Research.
3. Chklovski A, et al. (2023). CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning. Nature Methods.
