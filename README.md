# LAMBDA: A Prophage Detection Benchmark for Genomic Language Models

## Abstract

Transformer-based genomic sequence models represent an emerging frontier in computational biology. Yet, their embeddings have not yet shown the same level of predictive power as natural and protein language models, indicating a gap between current implementations and theoretical promise. Existing benchmarks for DNA language models primarily focus on classifying regulatory elements in eukaryotic genomes, leaving open the fundamental question of whether these models learn sequence-level features across whole genomes. We introduce LAMBDA, a benchmark designed to rigorously evaluate genome language model embeddings through phage-bacteria sequence discrimination across four categories of increasing complexity: probing tasks, fine-tuning assessments, diagnostic tests, and genome-wide prophage detection. Our comprehensive analysis of current genomic language models provides novel insights into the importance of training data quality relative to model size, the need for domain-specific training, and the application of genomic language models for detecting prophage sequences. This benchmark represents a challenging genomic annotation task in the bacterial domain and addresses a key computational problem with direct relevance to microbiology and medicine.

## Interactive Results

Explore and compare all model results on the LAMBDA benchmark:
[https://leannmlindsey.github.io/lambda-benchmark/](https://leannmlindsey.github.io/lambda-benchmark/)

## Download the Dataset

The LAMBDA benchmark dataset is available on Zenodo:
[https://doi.org/10.5281/zenodo.19236553](https://doi.org/10.5281/zenodo.19236553)

### Dataset Structure

The download contains the following directories:

| Directory | Description |
|-----------|-------------|
| `binary_segments_2k` | Binary classification segments (2k) |
| `binary_segments_4k` | Binary classification segments (4k) |
| `binary_segments_8k` | Binary classification segments (8k) |
| `error_and_bias_2k` | Error and bias diagnostic tests (2k) |
| `error_and_bias_4k` | Error and bias diagnostic tests (4k) |
| `error_and_bias_8k` | Error and bias diagnostic tests (8k) |
| `genome_wide_segments_2k_1k` | Genome-wide prophage detection segments (2k window, 1k stride) |
| `genome_wide_segments_4k_2k` | Genome-wide prophage detection segments (4k window, 2k stride) |
| `genome_wide_segments_8k_4k` | Genome-wide prophage detection segments (8k window, 4k stride) |
| `phrog_annotation_analysis` | PHROG annotation analysis data |

## Models Benchmarked

The following genomic language models were evaluated on the LAMBDA benchmark:

| Model | Repository |
|-------|------------|
| DNABERT-2 | [DNABERT2_generic_sequence_classification](https://github.com/leannmlindsey/DNABERT2_generic_sequence_classification) |
| Nucleotide Transformer v2 | [NTv2_generic_sequence_classification](https://github.com/leannmlindsey/NTv2_generic_sequence_classification) |
| Caduceus | [Caduceus_generic_sequence_classification](https://github.com/leannmlindsey/Caduceus_generic_sequence_classification) |
| ProkBERT | [ProkBERT_generic_sequence_classification](https://github.com/leannmlindsey/ProkBERT_generic_sequence_classification) |
| megaDNA | [megaDNA](https://github.com/leannmlindsey/megaDNA) |
| GENERanno | [Generanno_generic_sequence_classification](https://github.com/leannmlindsey/Generanno_generic_sequence_classification) |
| EVO2 | [evo2](https://github.com/ArcInstitute/evo2) (Arc Institute) |
| EVO2+SAE | [Evo2_SAE_LAMBDA_assessment](https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment) |

Each repository contains benchmarking instructions and inference scripts for running the corresponding model on the LAMBDA dataset.

## Authors

Leann M. Lindsey, Nicole L. Pershing, Keith Dufault-Thompson, Ho-jin Gwak, Anisa Habib, Aaron Schindler, Arjun Rakheja, June Round, W. Zac Stephens, Anne J. Blaschke, Hari Sundar, Xiaofang Jiang

**Affiliations:**
1. National Library of Medicine, National Institutes of Health, Bethesda, MD, USA
2. Kahlert School of Computing, University of Utah, Salt Lake City, UT, USA
3. Department of Pediatrics, School of Medicine, University of Utah, Salt Lake City, UT, USA
4. Department of Pathology, School of Medicine, University of Utah, Salt Lake City, UT, USA
5. Department of Computer Science, Tufts University, Medford, MA, USA
6. Division of Computer Engineering, Hankuk University of Foreign Studies, Yongin, Republic of Korea

## Citation

> *bioRxiv preprint coming soon — citation will be added here.*

## License

See [LICENSE](LICENSE) for details.
