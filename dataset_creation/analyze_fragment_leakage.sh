#!/bin/bash
#SBATCH --job-name=lambda_leakage
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=lambda_leakage_%j.out
#SBATCH --error=lambda_leakage_%j.err
#
# Fragment-level leakage analysis for the LAMBDA dataset.
#
# For each window size (2k, 4k, 8k) and each class (phage, bacteria):
#   1. Splits train/dev/test CSVs into per-class FASTA files
#   2. Builds a BLAST database from train fragments
#   3. BLASTs dev and test fragments against the train DB
#   4. Runs a within-train background control (each train fragment vs all
#      other train fragments, excluding self-matches)
#   5. Calls analyze_fragment_leakage.py to summarize and plot
#
# Output structure:
#   ${OUTPUT_ROOT}/<window>/<class>/{train_db, blast_<split>_vs_train.tsv,
#                                    blast_within_train.tsv}
#   ${OUTPUT_ROOT}/summary/{cumulative_pident.png, summary_stats.csv,
#                          high_similarity_examples.tsv}
#
# Usage:
#   sbatch dataset_creation/analyze_fragment_leakage.sh

# ─── Conda activation (verbatim from run_train_gena_lm.sh — do not edit) ────
module load conda 2>/dev/null || true
module load BLAST+/2.16.0 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi
conda activate gena_lm 2>/dev/null || source activate gena_lm 2>/dev/null || true

# ─── Configuration ──────────────────────────────────────────────────────────
DATASET_ROOT="/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final"
OUTPUT_ROOT="/data/lindseylm/GLM_EVALUATIONS/MODELS/LAMBDA/results/fragment_leakage_analysis"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/LAMBDA/dataset_creation"

WINDOWS=("2k" "4k" "8k")
CLASSES=("phage" "bacteria")        # phage = label 1, bacteria = label 0
SPLITS=("dev" "test")               # query splits (compared against train)

# BLAST parameters tuned for fragment-vs-fragment comparison
BLAST_EVALUE="1e-5"
BLAST_WORD_SIZE="11"                # default for blastn; catches moderate divergence
BLAST_THREADS="16"
BLAST_OUTFMT="6 qseqid sseqid pident length qlen slen qcovs evalue bitscore"

# Within-train background control: sample size
BG_SAMPLE_SIZE="2000"               # subsample train for the background BLAST

# ─── Helpers ────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}" || { echo "ERROR: cannot cd to ${SCRIPT_DIR}"; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

# Split a CSV with sequence + label columns into per-class FASTA
csv_to_class_fasta() {
    local csv_file="$1"
    local out_phage="$2"
    local out_bact="$3"

    python3 - <<PYTHON
import csv, sys
csv.field_size_limit(sys.maxsize)
phage_out = open("${out_phage}", "w")
bact_out  = open("${out_bact}", "w")
n_p, n_b = 0, 0
with open("${csv_file}") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        seq = row["sequence"].strip().upper()
        if not seq:
            continue
        label = int(row["label"])
        if label == 1:
            phage_out.write(f">p_{i}\n{seq}\n")
            n_p += 1
        else:
            bact_out.write(f">b_{i}\n{seq}\n")
            n_b += 1
phage_out.close(); bact_out.close()
print(f"  {n_p} phage and {n_b} bacterial fragments written")
PYTHON
}

# ─── Main loop ──────────────────────────────────────────────────────────────
for window in "${WINDOWS[@]}"; do
    csv_dir="${DATASET_ROOT}/merged_datasets_filtered/${window}"
    if [ ! -d "${csv_dir}" ]; then
        echo "SKIP ${window}: dataset dir not found at ${csv_dir}"
        continue
    fi

    work_dir="${OUTPUT_ROOT}/${window}"
    mkdir -p "${work_dir}"

    echo ""
    echo "============================================================"
    echo "Window ${window}"
    echo "============================================================"

    # Step 1: split each split's CSV into per-class FASTA
    for split in train dev test; do
        echo "[${window}/${split}] splitting CSV into per-class FASTA"
        csv_to_class_fasta \
            "${csv_dir}/${split}.csv" \
            "${work_dir}/${split}_phage.fasta" \
            "${work_dir}/${split}_bacteria.fasta"
    done

    # Step 2-4: per class, build train DB, BLAST dev/test, and run background
    for class in "${CLASSES[@]}"; do
        class_dir="${work_dir}/${class}"
        mkdir -p "${class_dir}"

        train_fasta="${work_dir}/train_${class}.fasta"
        if [ ! -s "${train_fasta}" ]; then
            echo "[${window}/${class}] SKIP — train FASTA empty"
            continue
        fi

        # 2. Build BLAST db from train
        echo "[${window}/${class}] building BLAST db from train"
        makeblastdb -in "${train_fasta}" -dbtype nucl \
            -out "${class_dir}/train_db" >/dev/null

        # 3. BLAST dev and test queries against train
        for split in "${SPLITS[@]}"; do
            query_fasta="${work_dir}/${split}_${class}.fasta"
            out_tsv="${class_dir}/blast_${split}_vs_train.tsv"
            if [ ! -s "${query_fasta}" ]; then
                echo "[${window}/${class}/${split}] SKIP — query FASTA empty"
                continue
            fi
            echo "[${window}/${class}/${split}] BLAST query vs train"
            blastn -query "${query_fasta}" -db "${class_dir}/train_db" \
                -evalue "${BLAST_EVALUE}" \
                -word_size "${BLAST_WORD_SIZE}" \
                -num_threads "${BLAST_THREADS}" \
                -max_target_seqs 1 -max_hsps 1 \
                -outfmt "${BLAST_OUTFMT}" \
                -out "${out_tsv}"
            n_hits=$(wc -l < "${out_tsv}")
            n_queries=$(grep -c "^>" "${query_fasta}")
            echo "    ${n_hits} hits for ${n_queries} queries"
        done

        # 4. Within-train background: subsample train, BLAST vs full train,
        #    exclude self-matches in the analysis step
        echo "[${window}/${class}] within-train background BLAST"
        bg_fasta="${class_dir}/bg_sample.fasta"
        python3 - <<PYTHON
from Bio import SeqIO
import random
random.seed(42)
records = list(SeqIO.parse("${train_fasta}", "fasta"))
sample = random.sample(records, min(${BG_SAMPLE_SIZE}, len(records)))
SeqIO.write(sample, "${bg_fasta}", "fasta")
print(f"  Sampled {len(sample)} of {len(records)} train fragments")
PYTHON

        blastn -query "${bg_fasta}" -db "${class_dir}/train_db" \
            -evalue "${BLAST_EVALUE}" \
            -word_size "${BLAST_WORD_SIZE}" \
            -num_threads "${BLAST_THREADS}" \
            -max_target_seqs 2 -max_hsps 1 \
            -outfmt "${BLAST_OUTFMT}" \
            -out "${class_dir}/blast_within_train.tsv"
        n_hits=$(wc -l < "${class_dir}/blast_within_train.tsv")
        echo "    ${n_hits} hits (will exclude self-matches in analysis)"
    done
done

# ─── Analysis + plots ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Running analysis + plotting"
echo "============================================================"
python3 "${SCRIPT_DIR}/analyze_fragment_leakage.py" \
    --input-root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/summary"

echo ""
echo "Done. Results: ${OUTPUT_ROOT}/summary/"
