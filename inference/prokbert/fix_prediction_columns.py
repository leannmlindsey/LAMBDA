#!/usr/bin/env python3
"""
Fix column names in prediction CSV files produced by the old inference scripts
so they are compatible with analyze_genome_wide_results.py.

Old inference output columns:
    segment_id, predicted_label, prob_class_0, prob_class_1, true_label, correct, [metadata...]

Required by analysis script:
    segment_id, label, prob_class_0, prob_1, pred_label, correct, [metadata...]

Renames:
    predicted_label -> pred_label
    true_label      -> label
    prob_class_1    -> prob_1

Also verifies that pred_label == argmax(prob_class_0, prob_1) for every row.

Usage:
    # Fix a single file (overwrites in place):
    python fix_prediction_columns.py /path/to/predictions.csv

    # Fix all prediction CSVs in a directory tree:
    python fix_prediction_columns.py /path/to/results/ --recursive

    # Dry run (report issues without overwriting):
    python fix_prediction_columns.py /path/to/results/ --recursive --dry-run
"""

import argparse
import glob
import os
import sys

import pandas as pd


# Column renames: old name -> new name
RENAME_MAP = {
    'predicted_label': 'pred_label',
    'true_label': 'label',
    'prob_class_1': 'prob_1',
}


def fix_csv(filepath, dry_run=False):
    """Fix column names in a single prediction CSV. Returns True if file was fixed."""
    try:
        df = pd.read_csv(filepath)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"  SKIP (cannot read: {e}): {filepath}")
        return False

    if df.empty:
        print(f"  SKIP (empty file): {filepath}")
        return False

    cols = list(df.columns)

    # Check if this is a prediction CSV at all
    if 'prob_class_0' not in cols and 'prob_1' not in cols:
        print(f"  SKIP (not a prediction CSV): {filepath}")
        return False

    # Check if already fixed (has the new column names)
    if 'pred_label' in cols and 'label' in cols and 'prob_1' in cols:
        mismatches = verify_predictions(df, pred_col='pred_label', prob_col='prob_1')
        if mismatches > 0:
            print(f"  WARNING: {mismatches} prediction mismatches in {filepath}")
        else:
            print(f"  OK (already has correct column names): {filepath}")
        return False

    # Determine which renames are needed
    renames_needed = {old: new for old, new in RENAME_MAP.items() if old in cols}

    if not renames_needed:
        print(f"  SKIP (no columns to rename): {filepath}")
        return False

    # Verify predictions before renaming
    pred_col = 'predicted_label' if 'predicted_label' in cols else 'pred_label'
    prob_col = 'prob_class_1' if 'prob_class_1' in cols else 'prob_1'
    if pred_col in cols and prob_col in cols:
        mismatches = verify_predictions(df, pred_col=pred_col, prob_col=prob_col)
        if mismatches > 0:
            print(f"  WARNING: {mismatches} rows where prediction != argmax(probabilities)")

    if dry_run:
        print(f"  WOULD FIX: {filepath}")
        print(f"    Renames: {renames_needed}")
        return True

    # Apply renames
    df = df.rename(columns=renames_needed)

    # Reorder: segment_id, label, prob_class_0, prob_1, pred_label, correct, [metadata]
    new_cols = list(df.columns)
    priority = ['segment_id', 'label', 'prob_class_0', 'prob_1', 'pred_label', 'correct']
    ordered = [c for c in priority if c in new_cols]
    ordered += [c for c in new_cols if c not in ordered]
    df = df[ordered]

    df.to_csv(filepath, index=False)
    print(f"  FIXED: {filepath}")
    return True


def verify_predictions(df, pred_col='pred_label', prob_col='prob_1'):
    """Verify predictions match argmax of probabilities. Returns count of mismatches."""
    expected = (df[prob_col] >= 0.5).astype(int)
    actual = df[pred_col].astype(int)
    return int((expected != actual).sum())


def main():
    parser = argparse.ArgumentParser(
        description='Fix column names in prediction CSVs for analyze_genome_wide_results.py')
    parser.add_argument('path', help='Path to a CSV file or directory')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Recursively find all *predictions*.csv files in directory')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Report what would be fixed without modifying files')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        pattern = '**/*predictions*.csv' if args.recursive else '*predictions*.csv'
        files = sorted(glob.glob(os.path.join(args.path, pattern), recursive=args.recursive))
    else:
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)

    if not files:
        print(f"No prediction CSV files found in {args.path}")
        sys.exit(0)

    print(f"Found {len(files)} prediction CSV file(s)")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    fixed = 0
    for f in files:
        if fix_csv(f, dry_run=args.dry_run):
            fixed += 1

    print(f"\n{'Would fix' if args.dry_run else 'Fixed'} {fixed}/{len(files)} files")


if __name__ == '__main__':
    main()
