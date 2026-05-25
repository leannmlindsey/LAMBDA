#!/usr/bin/env python3
"""
aggregate_gena_finetune.py

Aggregate the GENA-LM fine-tuning replicate runs (10 seeds x 2 variants x 3
windows = 60 jobs) into mean +/- std test MCC per (variant, window) -- the
fine-tuned column for the LAMBDA tables.

Each run directory written by finetune_gena_lm_phage.py contains:
    test_results.json      -> {"mcc": ..., "f1": ..., "accuracy": ..., ...}
    training_summary.json  -> {"model_name": "...gena-lm-bigbird-base-t2t" |
                                              "...moderngena-base", "seed": ...}

Run dirs live under:
    <base>/<window>/gena_lm_lambda_filtered_<window>_<seed>_<lr>_<timestamp>/

Usage (on Biowulf):
    python aggregate_gena_finetune.py \\
        --base /data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/GENA_LM_generic_sequence_classification/output/filtered
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def variant_from_model_name(name: str) -> str:
    n = name.lower()
    if "bigbird" in n:
        return "GENA-LM (BigBird)"
    if "moderngena" in n:
        return "ModernGENA"
    if "bert-base" in n:
        return "GENA-LM (BERT)"
    return name  # fall back to the raw id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="/data/lindseylm/GLM_EVALUATIONS/MODELS/GENA-LM/"
                "GENA_LM_generic_sequence_classification/output/filtered",
        help="Directory containing <window>/<run_dir>/ subfolders",
    )
    ap.add_argument("--out_csv", default="gena_finetune_summary.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If given, keep only these seeds (e.g. --seeds 1 2 3 4 5 6 7 8 9 10). "
                         "Use to drop stray runs like the default seed 42.")
    args = ap.parse_args()

    keep_seeds = set(args.seeds) if args.seeds is not None else None
    base = Path(args.base)

    # First pass: collect every run, keyed by (variant, window, seed), keeping
    # only the latest timestamp so stale earlier runs can't double-count.
    # latest[(variant, window, seed)] = (timestamp_str, mcc)
    latest: dict[tuple[str, str, object], tuple] = {}
    dup_count = 0

    for summ in base.glob("*/*/training_summary.json"):
        run_dir = summ.parent
        window = run_dir.parent.name            # 2k / 4k / 8k
        test_json = run_dir / "test_results.json"
        if not test_json.exists():
            print(f"[warn] no test_results.json in {run_dir}")
            continue
        summary = json.loads(summ.read_text())
        test = json.loads(test_json.read_text())
        variant = variant_from_model_name(summary.get("model_name", ""))
        mcc = test.get("mcc", test.get("test_mcc", test.get("eval_mcc")))
        seed = summary.get("seed")
        if mcc is None:
            print(f"[warn] no mcc in {test_json}")
            continue
        if keep_seeds is not None and seed not in keep_seeds:
            continue   # drop stray seeds (e.g. the default 42)
        # trailing _YYYYMMDD_HHMMSS in the run-dir name (lexicographically sortable)
        ts = "_".join(run_dir.name.split("_")[-2:])
        key = (variant, window, seed)
        if key in latest:
            dup_count += 1
            if ts <= latest[key][0]:
                continue   # keep the newer run already stored
        latest[key] = (ts, float(mcc))

    if dup_count:
        print(f"[note] {dup_count} duplicate (variant,window,seed) run(s) found; "
              f"kept the latest timestamp for each.")

    # (variant, window) -> list of (seed, mcc)
    groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for (variant, window, seed), (ts, mcc) in latest.items():
        groups[(variant, window)].append((seed, mcc))

    rows = []
    win_order = {"2k": 0, "4k": 1, "8k": 2}
    print(f"\n{'variant':<20}{'window':>7}{'n':>4}{'mean MCC':>11}{'std':>9}")
    for (variant, window) in sorted(groups, key=lambda k: (k[0], win_order.get(k[1], 9))):
        vals = [m for _, m in groups[(variant, window)]]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        n = len(vals)
        print(f"{variant:<20}{window:>7}{n:>4}{mean:>11.4f}{std:>9.4f}")
        rows.append({
            "variant": variant, "window": window, "n": n,
            "mean_mcc": round(mean, 4), "std_mcc": round(std, 4),
            "seeds": ",".join(str(s) for s, _ in sorted(groups[(variant, window)])),
        })

    if rows:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out_csv} ({len(rows)} rows)")
        print("\nFine-tuned column (mean +/- std) for the tables:")
        for r in rows:
            print(f"  {r['variant']} {r['window']}: {r['mean_mcc']:.3f} \\pm {r['std_mcc']:.3f}  (n={r['n']})")
    else:
        print("\nNo runs found -- check --base path.")


if __name__ == "__main__":
    main()
