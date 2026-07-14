#!/usr/bin/env python3
"""gather_checkpoints.py -- assemble the released checkpoints/probe heads for the
best-performing models on the LAMBDA genome-wide task into `lambda_best_checkpoints/`
for Zenodo (`lambda_best_checkpoints.tar.gz`).

Only the top genome-wide models are released; the rest are available on request.
The three released models differ in what "checkpoint" means, so this gathers each
from the right place:

  EVO2      -- zero-shot; its winning head is a linear probe on frozen EVO2
               embeddings. Ships the small probe heads (LP + 3-layer NN + scalers)
               from <results-root>/EVO2/<w>/embedding/evo2/. The base EVO2 model is
               obtained from Arc's repo (https://github.com/ArcInstitute/evo2).
  EVO2+SAE  -- zero-shot SAE feature; no trained head of ours to ship. A pointer
               file is written referencing the Evo2_SAE_LAMBDA_assessment repo.
  ProkBERT  -- fine-tuned (small). Ships the best-seed checkpoint per variant
               (mini/mini-c/mini-long) from <prokbert-root>/2k/finetune/<v>/seed-<N>/.
               These weights live on Delta, so pass --prokbert-root to a location
               that has them (run on Delta, or a copied tree).

Dry-run by default (prints what it would gather + sizes). Re-run with --apply, then:
    tar czf lambda_best_checkpoints.tar.gz -C <out> lambda_best_checkpoints

Usage:
    python3 gather_checkpoints.py \
        --results-root /path/to/INPUTS/LAMBDA_v1_results_to_visualize \
        --prokbert-root /path/to/prokbert/outputs_or_DATA_v1 \
        --out $PWD/best_ckpt_stage [--apply]
"""
import argparse
import os
import shutil

EVO2_PROBE_FILES = ["linear_probe.pkl", "linear_probe_scaler.pkl",
                    "three_layer_nn.pt", "three_layer_nn_scaler.pkl",
                    "embedding_analysis_results.json"]
EVO2_WINDOWS = ["2k", "4k", "8k"]
PROKBERT_BEST = {"prokbert-mini": 1, "prokbert-mini-c": 4, "prokbert-mini-long": 2}  # best FT seed

SAE_NOTE = """EVO2+SAE checkpoint
===================
EVO2+SAE is a ZERO-SHOT method: prophage scores come directly from a sparse
autoencoder (SAE) feature over frozen EVO2 embeddings, with no trained head in this
benchmark. There is therefore no checkpoint to ship here.

To reproduce EVO2+SAE:
  1. Obtain the base EVO2 model from Arc Institute: https://github.com/ArcInstitute/evo2
  2. Use the SAE + scoring code from: https://github.com/leannmlindsey/Evo2_SAE_LAMBDA_assessment
"""

EVO2_NOTE = """EVO2 checkpoint
===============
EVO2 is ZERO-SHOT: its winning LAMBDA head is a linear probe on frozen EVO2
embeddings. This directory ships only the small trained probe heads
(linear_probe.pkl + scaler; three_layer_nn.pt + scaler) per window.

To use them, obtain the base EVO2 model from Arc Institute
(https://github.com/ArcInstitute/evo2), extract embeddings, then apply the probe.
"""


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    t = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            fp = os.path.join(r, f)
            if not os.path.islink(fp):
                t += os.path.getsize(fp)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", help="INPUTS/LAMBDA_v1_results_to_visualize (for EVO2 probes)")
    ap.add_argument("--prokbert-root", help="tree with 2k/finetune/<variant>/seed-N/ (ProkBERT weights; Delta)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    dest = os.path.join(args.out, "lambda_best_checkpoints")
    total, missing = 0, []
    print(f"{'artifact':40s} {'size':>9s}  source")
    print("-" * 92)

    # EVO2 probe heads
    if args.results_root:
        for w in EVO2_WINDOWS:
            src = os.path.join(args.results_root, "EVO2", w, "embedding", "evo2")
            files = [f for f in EVO2_PROBE_FILES if os.path.isfile(os.path.join(src, f))]
            if not files:
                missing.append(f"EVO2 {w}: no probe files in {src}")
                print(f"{'evo2/'+w:40s} {'--':>9s}  MISSING {src}")
                continue
            sz = sum(size_of(os.path.join(src, f)) for f in files)
            total += sz
            print(f"{'evo2/'+w+'/ ('+str(len(files))+' files)':40s} {human(sz):>9s}  {src}")
            if args.apply:
                d = os.path.join(dest, "evo2", w)
                os.makedirs(d, exist_ok=True)
                for f in files:
                    shutil.copy2(os.path.join(src, f), os.path.join(d, f))
    else:
        print("  (skip EVO2: no --results-root)")

    # ProkBERT best-seed checkpoints
    if args.prokbert_root:
        for v, seed in PROKBERT_BEST.items():
            src = os.path.join(args.prokbert_root, "2k", "finetune", v, f"seed-{seed}")
            has_weights = os.path.isdir(src) and any(
                f.endswith((".bin", ".safetensors")) for f in os.listdir(src)) if os.path.isdir(src) else False
            if not has_weights:
                missing.append(f"ProkBERT {v} seed-{seed}: no model weights at {src}")
                print(f"{'prokbert/'+v:40s} {'--':>9s}  MISSING weights {src}")
                continue
            sz = size_of(src)
            total += sz
            print(f"{'prokbert/'+v+' (seed-'+str(seed)+')':40s} {human(sz):>9s}  {src}")
            if args.apply:
                d = os.path.join(dest, "prokbert", v)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(src, d, symlinks=False)
    else:
        print("  (skip ProkBERT: no --prokbert-root)")

    print("-" * 92)
    print(f"TOTAL: {human(total)}")
    if missing:
        print("\nUNRESOLVED:\n  " + "\n  ".join(missing))

    if args.apply:
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "EVO2_README.txt"), "w") as fh:
            fh.write(EVO2_NOTE)
        with open(os.path.join(dest, "EVO2_SAE_README.txt"), "w") as fh:
            fh.write(SAE_NOTE)
        print(f"\nAPPLIED -> {dest}  (wrote EVO2 + EVO2+SAE pointer READMEs)")
        print(f"Next: tar czf lambda_best_checkpoints.tar.gz -C {args.out} lambda_best_checkpoints")
    else:
        print("\nDRY RUN -- nothing copied. Re-run with --apply once paths/sizes look right.")


if __name__ == "__main__":
    main()
