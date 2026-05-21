#!/bin/bash
# check_error_bias_files.sh
#
# Diagnose what's actually in each error_and_bias_<window>/ directory and
# whether the test.csv files in those directories are content-identical to
# the binary_segments / merged_datasets_filtered test.csv files at the
# same window size.
#
# Symptom that triggered this check: in metrics_summary.csv, the
# `error_bias` rows for every baseline had identical numbers to the
# `binary` rows — suggesting the script picked up the wrong file as
# "the" error_and_bias test split.
#
# Usage:
#   bash check_error_bias_files.sh
# (Edit DATASET_ROOT below if your path differs.)

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/gpfs/gsfs12/users/Irp-jiang/share/lindseylm/lambda_final}"

echo "============================================================"
echo "Inspecting LAMBDA dataset at: ${DATASET_ROOT}"
echo "============================================================"

for window in 2k 4k 8k; do
    echo
    echo "─── window ${window} ───────────────────────────────────────"

    binary_dir=""
    for cand in "binary_segments_${window}" "merged_datasets_filtered/${window}" "${window}"; do
        if [ -d "${DATASET_ROOT}/${cand}" ]; then
            binary_dir="${DATASET_ROOT}/${cand}"
            break
        fi
    done

    err_dir=""
    for cand in "error_and_bias_${window}" "error_and_bias/${window}"; do
        if [ -d "${DATASET_ROOT}/${cand}" ]; then
            err_dir="${DATASET_ROOT}/${cand}"
            break
        fi
    done

    echo "binary subdir     : ${binary_dir:-NOT FOUND}"
    echo "error_bias subdir : ${err_dir:-NOT FOUND}"

    if [ -z "${err_dir}" ]; then
        echo "[skip] no error_and_bias subdir for window ${window}"
        continue
    fi

    echo
    echo "Contents of ${err_dir}:"
    ls -lh "${err_dir}" | sed 's/^/  /'

    bin_test="${binary_dir}/test.csv"
    err_test="${err_dir}/test.csv"

    if [ -f "${bin_test}" ] && [ -f "${err_test}" ]; then
        bin_md5=$(md5sum "${bin_test}" | awk '{print $1}')
        err_md5=$(md5sum "${err_test}" | awk '{print $1}')
        bin_lines=$(wc -l < "${bin_test}")
        err_lines=$(wc -l < "${err_test}")

        echo
        echo "test.csv comparison:"
        printf "  binary     %s  %s rows  %s\n" "${bin_md5}" "${bin_lines}" "${bin_test}"
        printf "  error_bias %s  %s rows  %s\n" "${err_md5}" "${err_lines}" "${err_test}"

        if [ "${bin_md5}" = "${err_md5}" ]; then
            echo "  >>> IDENTICAL FILES <<<"
            echo "  The error_and_bias test set is the same as the binary test set."
            echo "  Either it's a symlink/copy placeholder, or the actual diagnostic"
            echo "  splits live under a different filename in ${err_dir}."
        else
            echo "  Files differ (good — they are distinct test sets)."
        fi
    else
        echo
        echo "Missing one of the two test.csv files:"
        [ -f "${bin_test}" ] || echo "  binary test.csv NOT FOUND at ${bin_test}"
        [ -f "${err_test}" ] || echo "  error_and_bias test.csv NOT FOUND at ${err_test}"
    fi

    # List any non-test.csv files in error_and_bias — these might be the
    # actual diagnostic splits we're missing.
    echo
    echo "Other CSV files in ${err_dir} (potential diagnostic splits):"
    find "${err_dir}" -maxdepth 1 -type f -name "*.csv" ! -name "test.csv" 2>/dev/null \
        | sed 's/^/  /' || echo "  (none)"
done

echo
echo "============================================================"
echo "Done."
echo "============================================================"
