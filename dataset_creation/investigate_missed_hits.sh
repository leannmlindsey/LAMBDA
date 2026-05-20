#!/bin/bash
# Investigate why 2 hits slipped through v3 filtering
#
# Usage: bash investigate_missed_hits.sh

# ============================================
# CONFIGURATION
# ============================================

# Original v3 BLAST results
V3_BLAST="/work/hdd/bfzj/llindsey1/prophage_blast_results_v3/blast_filtered_combined.tsv"

# Validation BLAST results
VALIDATION_BLAST="/work/hdd/bfzj/llindsey1/prophage_validation_v3/blast_results_validation.tsv"

# Contig to genome mapping
CONTIG_MAP="/work/hdd/bfzj/llindsey1/prophage_blast_results_v2/contig_to_genome_map.tsv"

# Contaminated accessions list
CONTAMINATED="/projects/bfzj/llindsey1/black_and_white/scripts/lambda_dataset_tools/gtdb_dataset_filtered_v3/contaminated_accessions.txt"

# The problem phage
PROBLEM_PHAGE="MZ092003"

# The missed contigs
MISSED_CONTIG_1="NZ_FRAI01000004.1"
MISSED_CONTIG_2="NZ_BAAAYO010000022.1"

# ============================================
# INVESTIGATION
# ============================================

echo "=========================================="
echo "Investigating Missed Hits in v3 Filtering"
echo "=========================================="
echo ""

# 1. Check how many hits MZ092003 had in original v3 BLAST
echo "1. How many hits did ${PROBLEM_PHAGE} have in original v3 BLAST?"
echo "   (max_target_seqs was set to 2000)"
echo ""
TOTAL_HITS=$(grep "^${PROBLEM_PHAGE}" "${V3_BLAST}" 2>/dev/null | wc -l)
echo "   Total hits for ${PROBLEM_PHAGE}: ${TOTAL_HITS}"

if [ "${TOTAL_HITS}" -ge 2000 ]; then
    echo "   >>> HIT THE LIMIT! This phage has ≥2000 hits."
    echo "   >>> Some hits were likely not reported due to max_target_seqs=2000"
else
    echo "   Under the limit - all hits should have been reported."
fi
echo ""

# 2. Check if the missed contigs were in the original v3 BLAST
echo "2. Were the missed contigs in the original v3 BLAST?"
echo ""
echo "   Checking ${MISSED_CONTIG_1}:"
grep "${MISSED_CONTIG_1}" "${V3_BLAST}" 2>/dev/null | head -5
CONTIG1_IN_BLAST=$(grep "${MISSED_CONTIG_1}" "${V3_BLAST}" 2>/dev/null | wc -l)
echo "   Hits found: ${CONTIG1_IN_BLAST}"
echo ""

echo "   Checking ${MISSED_CONTIG_2}:"
grep "${MISSED_CONTIG_2}" "${V3_BLAST}" 2>/dev/null | head -5
CONTIG2_IN_BLAST=$(grep "${MISSED_CONTIG_2}" "${V3_BLAST}" 2>/dev/null | wc -l)
echo "   Hits found: ${CONTIG2_IN_BLAST}"
echo ""

# 3. Map contigs to genomes
echo "3. Which genomes do these contigs belong to?"
echo ""
echo "   ${MISSED_CONTIG_1}:"
GENOME1=$(grep "${MISSED_CONTIG_1}" "${CONTIG_MAP}" 2>/dev/null | cut -f2 | head -1)
if [ -n "${GENOME1}" ]; then
    echo "   Genome: ${GENOME1}"
else
    echo "   NOT FOUND in contig map!"
fi
echo ""

echo "   ${MISSED_CONTIG_2}:"
GENOME2=$(grep "${MISSED_CONTIG_2}" "${CONTIG_MAP}" 2>/dev/null | cut -f2 | head -1)
if [ -n "${GENOME2}" ]; then
    echo "   Genome: ${GENOME2}"
else
    echo "   NOT FOUND in contig map!"
fi
echo ""

# 4. Check if these genomes are in the contaminated list
echo "4. Are these genomes in the contaminated list?"
echo ""
if [ -n "${GENOME1}" ]; then
    if grep -q "${GENOME1}" "${CONTAMINATED}" 2>/dev/null; then
        echo "   ${GENOME1}: YES (should have been removed!)"
    else
        echo "   ${GENOME1}: NO (was not flagged as contaminated)"
    fi
fi

if [ -n "${GENOME2}" ]; then
    if grep -q "${GENOME2}" "${CONTAMINATED}" 2>/dev/null; then
        echo "   ${GENOME2}: YES (should have been removed!)"
    else
        echo "   ${GENOME2}: NO (was not flagged as contaminated)"
    fi
fi
echo ""

# 5. Check the longest hit in validation
echo "5. What is the longest hit in validation BLAST?"
echo "   (Looking for the 15214 bp alignment)"
echo ""
echo "   Top 10 longest alignments (showing identity):"
echo "   query | subject | identity | length | ..."
awk -F'\t' '{print $0}' "${VALIDATION_BLAST}" 2>/dev/null | sort -t$'\t' -k4 -nr | head -10
echo ""

# 6. Check identity distribution of long hits
echo "6. Identity distribution of hits ≥1000bp:"
echo ""
awk -F'\t' '$4 >= 1000 {
    if ($3 >= 90) print "≥90%: " $0
    else if ($3 >= 80) bin["80-89%"]++
    else if ($3 >= 70) bin["70-79%"]++
    else bin["<70%"]++
}
END {
    for (b in bin) print "   " b ": " bin[b] " hits"
}' "${VALIDATION_BLAST}" 2>/dev/null
echo ""

# 7. Summary of phages with most hits (to understand max_target_seqs impact)
echo "7. Top 10 phages by hit count in original v3 BLAST:"
echo "   (Shows which phages might be hitting the 2000 limit)"
echo ""
cut -f1 "${V3_BLAST}" 2>/dev/null | sort | uniq -c | sort -rn | head -10
echo ""

echo "=========================================="
echo "CONCLUSION"
echo "=========================================="
echo ""
if [ "${TOTAL_HITS}" -ge 2000 ]; then
    echo "ROOT CAUSE: Phage ${PROBLEM_PHAGE} has ${TOTAL_HITS} hits, hitting the"
    echo "max_target_seqs=2000 limit. The hits to ${MISSED_CONTIG_1}"
    echo "and ${MISSED_CONTIG_2} were ranked >2000 and not reported."
    echo ""
    echo "SOLUTION OPTIONS:"
    echo "  1. Accept 2 remaining hits (very clean dataset)"
    echo "  2. Manually add the 2 genomes to contaminated list"
    echo "  3. Increase max_target_seqs further (diminishing returns)"
fi
echo ""
