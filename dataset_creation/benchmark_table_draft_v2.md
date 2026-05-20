# LAMBDA Benchmark Table (Draft v2)

## For Review - Will Convert to LaTeX

| Benchmark Test | Dataset | Evaluation Rationale |
|----------------|---------|---------------------|
| **Embedding Strength** | Phage segments (INPHARED) vs Bacteria segments (GTDB filtered) at 2k, 4k, 8k bp | Assesses whether the model has learned to distinguish phage from bacterial sequences as a fundamental feature of its sequence representation |
| **Linear Probe Accuracy** | Phage segments (INPHARED) vs Bacteria segments (GTDB filtered) at 2k, 4k, 8k bp | Quantifies discriminative signal in embeddings using a minimal classifier architecture |
| **Fine-Tuning Performance** | Phage segments (INPHARED) vs Bacteria segments (GTDB filtered) at 2k, 4k, 8k bp | Establishes upper-bound performance when model parameters are optimized for prophage detection |
| **Phage-Only Performance** | INPHARED phage segments with Pharokka functional annotations | Identifies which functional categories carry strongest phage-associated signal; characterizes true positive and false negative patterns |
| **Bacteria-Only Performance** | GTDB filtered bacteria segments with functional annotations | Identifies bacterial functional categories most frequently misclassified as phage; characterizes true negative and false positive patterns |
| **GC-Content Control** | Shuffled CDS sequences (nucleotide composition preserved) | Detects models relying on GC content or other compositional biases rather than sequence-specific features |
| **Prophage Boundary Detection** | ? (Full bacterial genomes with known prophage insertions?) | Evaluates real-world prophage detection performance including integration site resolution |

---

## Questions:

1. **Prophage Boundary Detection** - What dataset is used for this? Known prophage insertions from a database?

2. **Segment lengths** - Should each test specify all three lengths (2k, 4k, 8k) or are certain tests only run at specific lengths?

3. **Train/Test split info** - Should the table include segment counts (e.g., "Train: 27k, Test: 3.4k")?

4. **Dataset statistics** - Should there be a separate companion table with dataset statistics (genome counts, segment counts, etc.)?

---

## Draft LaTeX (pending your feedback):

```latex
\begin{table}[ht]
\centering
\caption{LAMBDA benchmark tests for evaluating genomic language models on prophage detection tasks.}
\label{tab:lambda_benchmark}
\begin{tabular}{p{3cm} p{4.5cm} p{6.5cm}}
\toprule
\textbf{Benchmark Test} & \textbf{Dataset} & \textbf{Evaluation Rationale} \\
\midrule
Embedding Strength & Phage (INPHARED) vs Bacteria (GTDB filtered); 2k, 4k, 8k bp segments & Assesses whether the model has learned to distinguish phage from bacterial sequences as a fundamental feature of its sequence representation \\
\addlinespace
Linear Probe Accuracy & Phage (INPHARED) vs Bacteria (GTDB filtered); 2k, 4k, 8k bp segments & Quantifies discriminative signal in embeddings using a minimal classifier architecture \\
\addlinespace
Fine-Tuning Performance & Phage (INPHARED) vs Bacteria (GTDB filtered); 2k, 4k, 8k bp segments & Establishes upper-bound performance when model parameters are optimized for prophage detection \\
\addlinespace
Phage-Only Performance & INPHARED phage segments with Pharokka annotations & Identifies which functional categories carry strongest phage-associated signal; characterizes true positive and false negative patterns \\
\addlinespace
Bacteria-Only Performance & GTDB filtered bacteria segments with functional annotations & Identifies bacterial functional categories most frequently misclassified as phage; characterizes true negative and false positive patterns \\
\addlinespace
GC-Content Control & Shuffled CDS sequences (composition preserved) & Detects models relying on GC content or other compositional biases rather than sequence-specific features \\
\addlinespace
Prophage Boundary Detection & [TBD - bacterial genomes with known prophage sites?] & Evaluates real-world prophage detection performance including integration site resolution \\
\bottomrule
\end{tabular}
\end{table}
```
