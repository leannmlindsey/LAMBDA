#!/bin/bash
# Analyze BLAST filtering steps in detail
#
# Usage: bash analyze_blast_filtering.sh

# Validation BLAST results (raw, before filtering)
BLAST_FILE="/work/hdd/bfzj/llindsey1/prophage_validation_v3/validation_blast_raw.tsv"

echo "=========================================="
echo "BLAST Filtering Analysis"
echo "=========================================="
echo ""
echo "Input: ${BLAST_FILE}"
echo ""

# Total raw hits
TOTAL=$(wc -l < "${BLAST_FILE}")
echo "1. Total raw BLAST hits: ${TOTAL}"
echo ""

# Step 1: Length filter only (≥200bp)
echo "2. After LENGTH filter (≥200bp):"
LENGTH_FILTERED=$(awk -F'\t' '$4 >= 200' "${BLAST_FILE}" | wc -l)
echo "   Hits ≥200bp: ${LENGTH_FILTERED}"
echo ""

# Step 2: Identity filter only (≥90%)
echo "3. After IDENTITY filter (≥90%):"
IDENTITY_FILTERED=$(awk -F'\t' '$3 >= 90' "${BLAST_FILE}" | wc -l)
echo "   Hits ≥90% identity: ${IDENTITY_FILTERED}"
echo ""

# Step 3: Both filters (≥200bp AND ≥90%)
echo "4. After BOTH filters (≥200bp AND ≥90%):"
BOTH_FILTERED=$(awk -F'\t' '$4 >= 200 && $3 >= 90' "${BLAST_FILE}" | wc -l)
echo "   Hits passing both: ${BOTH_FILTERED}"
echo ""

# Detailed breakdown
echo "=========================================="
echo "Detailed Breakdown"
echo "=========================================="
echo ""

echo "Length distribution (all hits):"
awk -F'\t' '{
    if ($4 < 100) bin["<100bp"]++
    else if ($4 < 200) bin["100-199bp"]++
    else if ($4 < 500) bin["200-499bp"]++
    else if ($4 < 1000) bin["500-999bp"]++
    else bin["≥1000bp"]++
}
END {
    print "   <100bp:     " bin["<100bp"]+0
    print "   100-199bp:  " bin["100-199bp"]+0
    print "   200-499bp:  " bin["200-499bp"]+0
    print "   500-999bp:  " bin["500-999bp"]+0
    print "   ≥1000bp:    " bin["≥1000bp"]+0
}' "${BLAST_FILE}"
echo ""

echo "Identity distribution (hits ≥200bp only):"
awk -F'\t' '$4 >= 200 {
    if ($3 < 70) bin["<70%"]++
    else if ($3 < 80) bin["70-79%"]++
    else if ($3 < 85) bin["80-84%"]++
    else if ($3 < 90) bin["85-89%"]++
    else if ($3 < 95) bin["90-94%"]++
    else bin["≥95%"]++
}
END {
    print "   <70%:   " bin["<70%"]+0
    print "   70-79%: " bin["70-79%"]+0
    print "   80-84%: " bin["80-84%"]+0
    print "   85-89%: " bin["85-89%"]+0
    print "   90-94%: " bin["90-94%"]+0
    print "   ≥95%:   " bin["≥95%"]+0
}' "${BLAST_FILE}"
echo ""

echo "=========================================="
echo "The 2 remaining hits (≥200bp AND ≥90%):"
echo "=========================================="
awk -F'\t' '$4 >= 200 && $3 >= 90 {print}' "${BLAST_FILE}"
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Raw hits:                    ${TOTAL}"
echo "After ≥200bp filter:         ${LENGTH_FILTERED} ($(echo "scale=2; ${LENGTH_FILTERED}*100/${TOTAL}" | bc)%)"
echo "After ≥90% identity filter:  ${IDENTITY_FILTERED} ($(echo "scale=2; ${IDENTITY_FILTERED}*100/${TOTAL}" | bc)%)"
echo "After BOTH filters:          ${BOTH_FILTERED} ($(echo "scale=4; ${BOTH_FILTERED}*100/${TOTAL}" | bc)%)"
echo ""
