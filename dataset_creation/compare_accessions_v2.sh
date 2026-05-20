#!/bin/bash

# Compare original accessions with BLAST hit accessions
# Uses contig-to-genome mapping to convert BLAST hits to genome accessions

# ============================================
# CONFIGURATION
# ============================================

# Original accession list (the selected genomes)
ORIGINAL_ACCESSIONS="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset/all_accessions.txt"

# Combined BLAST results from v2
BLAST_RESULTS="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/blast_filtered_combined.tsv"

# Contig to genome mapping
CONTIG_MAP="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv"

# Output directory
OUTPUT_DIR="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2"

# ============================================

echo "=========================================="
echo "Compare Original vs BLAST Hit Accessions (v2)"
echo "=========================================="

# Check that mapping file exists
if [ ! -f "${CONTIG_MAP}" ]; then
    echo "Error: Contig mapping file not found: ${CONTIG_MAP}"
    echo "Run create_contig_map.slurm first!"
    exit 1
fi

# Combine BLAST results if not already done
if [ ! -f "${BLAST_RESULTS}" ]; then
    echo "Combining BLAST filtered results..."
    cat ${OUTPUT_DIR}/blast_filtered_*.tsv > ${BLAST_RESULTS}
fi

echo "BLAST results: ${BLAST_RESULTS}"
echo "Contig mapping: ${CONTIG_MAP}"
echo "Total BLAST hits: $(wc -l < ${BLAST_RESULTS})"
echo "Total contig mappings: $(wc -l < ${CONTIG_MAP})"
echo ""

# Step 1: Extract unique contig IDs from BLAST hits
echo "Step 1: Extracting unique contig IDs from BLAST hits..."
BLAST_CONTIGS="${OUTPUT_DIR}/blast_hit_contigs.txt"

cut -f2 ${BLAST_RESULTS} | sort -u > ${BLAST_CONTIGS}

NUM_BLAST_CONTIGS=$(wc -l < ${BLAST_CONTIGS})
echo "  Unique contigs with phage hits: ${NUM_BLAST_CONTIGS}"

# Step 2: Map contig IDs to genome accessions
echo ""
echo "Step 2: Mapping contigs to genome accessions..."
BLAST_GENOMES="${OUTPUT_DIR}/blast_hit_genomes.txt"

# Join BLAST contigs with mapping (skip header line in mapping)
# mapping format: contig_id\tgenome_accession
tail -n +2 ${CONTIG_MAP} | sort -k1,1 > ${OUTPUT_DIR}/contig_map_sorted.tmp
sort ${BLAST_CONTIGS} > ${OUTPUT_DIR}/blast_contigs_sorted.tmp

join -1 1 -2 1 -o 2.2 \
    ${OUTPUT_DIR}/blast_contigs_sorted.tmp \
    ${OUTPUT_DIR}/contig_map_sorted.tmp | \
    sort -u > ${BLAST_GENOMES}

NUM_BLAST_GENOMES=$(wc -l < ${BLAST_GENOMES})
echo "  Unique genomes with phage hits: ${NUM_BLAST_GENOMES}"

# Check for unmapped contigs
UNMAPPED="${OUTPUT_DIR}/unmapped_contigs.txt"
join -v 1 \
    ${OUTPUT_DIR}/blast_contigs_sorted.tmp \
    ${OUTPUT_DIR}/contig_map_sorted.tmp > ${UNMAPPED}

NUM_UNMAPPED=$(wc -l < ${UNMAPPED})
if [ ${NUM_UNMAPPED} -gt 0 ]; then
    echo "  WARNING: ${NUM_UNMAPPED} contigs could not be mapped!"
    echo "  First 5 unmapped:"
    head -5 ${UNMAPPED} | sed 's/^/    /'
fi

# Cleanup temp files
rm -f ${OUTPUT_DIR}/*.tmp

# Step 3: Normalize original accessions
echo ""
echo "Step 3: Normalizing original accession list..."
ORIGINAL_NORMALIZED="${OUTPUT_DIR}/original_accessions_normalized.txt"

sed 's/^GB_//; s/^RS_//' ${ORIGINAL_ACCESSIONS} | sort -u > ${ORIGINAL_NORMALIZED}

NUM_ORIGINAL=$(wc -l < ${ORIGINAL_NORMALIZED})
echo "  Original accessions: ${NUM_ORIGINAL}"

# Step 4: Find clean genomes (in original but NOT in BLAST hits)
echo ""
echo "Step 4: Finding clean genomes..."
CLEAN_ACCESSIONS="${OUTPUT_DIR}/clean_accessions.txt"

comm -23 ${ORIGINAL_NORMALIZED} ${BLAST_GENOMES} > ${CLEAN_ACCESSIONS}

NUM_CLEAN=$(wc -l < ${CLEAN_ACCESSIONS})
echo "  Clean accessions (no phage hits): ${NUM_CLEAN}"

# Step 5: Find contaminated genomes (in both lists)
CONTAMINATED="${OUTPUT_DIR}/contaminated_accessions.txt"
comm -12 ${ORIGINAL_NORMALIZED} ${BLAST_GENOMES} > ${CONTAMINATED}

NUM_CONTAMINATED=$(wc -l < ${CONTAMINATED})
echo "  Contaminated accessions: ${NUM_CONTAMINATED}"

# Step 6: Summary
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Original genomes:        ${NUM_ORIGINAL}"
echo "Contaminated (with hits): ${NUM_CONTAMINATED}"
echo "Clean (no hits):         ${NUM_CLEAN}"
echo ""

if [ ${NUM_ORIGINAL} -gt 0 ]; then
    PERCENT_CLEAN=$(echo "scale=1; ${NUM_CLEAN} * 100 / ${NUM_ORIGINAL}" | bc)
    PERCENT_CONTAMINATED=$(echo "scale=1; ${NUM_CONTAMINATED} * 100 / ${NUM_ORIGINAL}" | bc)
    echo "Percent clean:           ${PERCENT_CLEAN}%"
    echo "Percent contaminated:    ${PERCENT_CONTAMINATED}%"
fi

echo ""
if [ ${NUM_CLEAN} -eq 0 ]; then
    echo "WARNING: All genomes have at least one phage hit!"
    echo "You may need to adjust filtering thresholds."
elif [ ${NUM_CLEAN} -lt 1000 ]; then
    echo "WARNING: Only ${NUM_CLEAN} clean genomes - may not be enough for balanced dataset."
else
    echo "You have ${NUM_CLEAN} clean genomes to work with."
fi

echo ""
echo "Output files:"
echo "  ${BLAST_CONTIGS} - unique contigs with hits"
echo "  ${BLAST_GENOMES} - unique genomes with hits"
echo "  ${CLEAN_ACCESSIONS} - genomes without hits"
echo "  ${CONTAMINATED} - genomes with hits"
echo "=========================================="
