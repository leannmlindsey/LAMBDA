#!/usr/bin/env python3
"""
extract_gena_table_values.py

Assemble the complete GENA-LM rows for paper Table 2 (Linear Probe + 3-Layer NN)
and Table 3 (Diagnostics + Genome-wide), pulling from whatever is on disk and
flagging anything missing so it can be followed up.

Run on Biowulf, where FINAL_RESULTS lives. For each variant x window it reads:

  Table 3 diagnostics (already on disk):
    <base>/<variant>/error_bias/<w>/gc_control_<w>_test_metrics.json   -> GC-bias = mcc
    <base>/<variant>/error_bias/<w>/bacterial_cds_<w>_metrics.json     -> FPR = 1 - specificity
    <base>/<variant>/error_bias/<w>/phage_segments_<w>_*_metrics.json  -> FNR = 1 - recall

  Table 2 linear probe / 3-layer NN (may be missing until the embedding jobs finish):
    <base>/<variant>/embedding_analysis/<w>/embedding_analysis_results.json
      -> linear_probe_mcc, random LP mcc, 3-layer NN mcc, random NN mcc

  Fine-tuned binary MCC and genome-wide Raw/Filtered are LOCKED CONSTANTS below
  (computed earlier: FT = aggregate_gena_finetune.py n=10 seeds 1-10; genome-wide =
  combine_baseline_grids.py on the GENA grid searches). Edit them only if those
  upstream numbers change.

Peak (Table 3) = max(LP_model, NN_model, FT_mean), with an L/N/F marker for which won.
If LP/NN are missing, Peak is reported as max(FT) only, flagged "(FT; LP/NN pending)".

Usage (Biowulf):
    python extract_gena_table_values.py \\
        --base /data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS
"""

from __future__ import annotations

import argparse
import glob
import json
import os

VARIANTS = {
    "gena_lm_bigbird": "GENA-LM (BigBird)",
    "gena_lm_moderngena": "ModernGENA",
}
WINDOWS = ["2k", "4k", "8k"]

# ── Locked constants (computed earlier) ──────────────────────────────────────
# Fine-tuned binary MCC: mean, std over 10 seeds (aggregate_gena_finetune.py).
FINETUNED = {
    ("gena_lm_bigbird", "2k"): (0.675, 0.019),
    ("gena_lm_bigbird", "4k"): (0.722, 0.019),
    ("gena_lm_bigbird", "8k"): (0.757, 0.036),
    ("gena_lm_moderngena", "2k"): (0.810, 0.032),
    ("gena_lm_moderngena", "4k"): (0.864, 0.028),
    ("gena_lm_moderngena", "8k"): (0.911, 0.009),
}
# Genome-wide, combined grid: (raw_fpr%, raw_recall%, raw_mcc, filt_fpr%, filt_recall%, filt_mcc).
GENOME_WIDE = {
    ("gena_lm_bigbird", "2k"): (45.83, 65.90, 0.080,  2.11, 31.28, 0.260),
    ("gena_lm_bigbird", "4k"): (27.20, 58.29, 0.141,  2.67, 43.95, 0.340),
    ("gena_lm_bigbird", "8k"): (22.68, 48.60, 0.169,  2.03, 36.94, 0.339),
    ("gena_lm_moderngena", "2k"): (27.82, 61.42, 0.142,  1.77, 48.52, 0.402),
    ("gena_lm_moderngena", "4k"): (14.73, 60.24, 0.257,  1.75, 62.20, 0.541),
    ("gena_lm_moderngena", "8k"): (19.16, 72.86, 0.299,  2.70, 67.35, 0.568),
}


def first_key(d: dict, candidates: list[str]):
    """Return the first present key's value (case-insensitive), else None."""
    lower = {k.lower(): v for k, v in d.items()}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_diagnostics(base: str, variant: str, w: str) -> dict:
    d = os.path.join(base, variant, "error_bias", w)
    out = {"gc_bias": None, "fpr_bact": None, "fnr_phage": None, "missing": []}

    gc = load_json(os.path.join(d, f"gc_control_{w}_test_metrics.json"))
    if gc is not None:
        out["gc_bias"] = first_key(gc, ["mcc"])
    else:
        out["missing"].append("gc_control")

    bc = load_json(os.path.join(d, f"bacterial_cds_{w}_metrics.json"))
    if bc is not None:
        spec = first_key(bc, ["specificity"])
        out["fpr_bact"] = (1 - spec) * 100 if spec is not None else None
    else:
        out["missing"].append("bacterial_cds")

    ph_matches = glob.glob(os.path.join(d, f"phage_segments_{w}_*_metrics.json"))
    if ph_matches:
        ph = load_json(ph_matches[0])
        rec = first_key(ph, ["recall", "sensitivity"]) if ph else None
        out["fnr_phage"] = (1 - rec) * 100 if rec is not None else None
    else:
        out["missing"].append("phage_segments")
    return out


def read_embedding(base: str, variant: str, w: str) -> dict:
    """Return LP/NN model+random MCC, or note the file is missing."""
    path = os.path.join(base, variant, "embedding_analysis", w, "embedding_analysis_results.json")
    r = load_json(path)
    out = {"lp_model": None, "lp_random": None, "nn_model": None, "nn_random": None,
           "present": r is not None, "unresolved_keys": False, "available_keys": None}
    if r is None:
        return out
    out["lp_model"] = first_key(r, ["pretrained_linear_probe_mcc", "linear_probe_mcc",
                                    "linear_probe_model_mcc", "lp_mcc"])
    out["lp_random"] = first_key(r, ["random_linear_probe_mcc", "linear_probe_random_mcc",
                                     "random_baseline_linear_mcc", "random_lp_mcc"])
    out["nn_model"] = first_key(r, ["pretrained_nn_mcc", "three_layer_nn_mcc", "nn_mcc",
                                    "three_layer_mcc", "3_layer_nn_mcc", "mlp_mcc"])
    out["nn_random"] = first_key(r, ["random_nn_mcc", "random_three_layer_nn_mcc",
                                     "three_layer_nn_random_mcc", "random_baseline_nn_mcc"])
    # Nested fallbacks (e.g., {"random_baseline_linear": {"mcc": ...}})
    if out["lp_random"] is None and isinstance(r.get("random_baseline_linear"), dict):
        out["lp_random"] = r["random_baseline_linear"].get("mcc")
    if out["nn_random"] is None and isinstance(r.get("random_baseline_nn"), dict):
        out["nn_random"] = r["random_baseline_nn"].get("mcc")
    if any(v is None for v in (out["lp_model"], out["nn_model"])):
        out["unresolved_keys"] = True
        out["available_keys"] = sorted(r.keys())
    return out


def fmt(v, pct=False, nd=3):
    if v is None:
        return "[MISSING]"
    return f"{v:.2f}\\%" if pct else f"{v:.{nd}f}"


def peak_and_marker(lp, nn, ft):
    """Peak = max(LP, NN, FT) with marker L/N/F. Handles missing LP/NN."""
    cands = []
    if lp is not None: cands.append((lp, "LP"))
    if nn is not None: cands.append((nn, "NN"))
    if ft is not None: cands.append((ft, "FT"))
    if not cands:
        return None, ""
    best_val, best_mark = max(cands, key=lambda x: x[0])
    note = "" if (lp is not None and nn is not None) else "  (LP/NN pending)"
    return best_val, best_mark + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS")
    ap.add_argument("--out_csv", default="gena_table_values.csv")
    args = ap.parse_args()

    rows = []
    missing_report = []

    print("=" * 92)
    print("GENA-LM table values  (LP/NN read from disk; FT + genome-wide are locked constants)")
    print("=" * 92)

    for variant, disp in VARIANTS.items():
        for w in WINDOWS:
            diag = read_diagnostics(args.base, variant, w)
            emb = read_embedding(args.base, variant, w)
            ft_mean, ft_std = FINETUNED[(variant, w)]
            gw = GENOME_WIDE[(variant, w)]

            peak_val, peak_mark = peak_and_marker(emb["lp_model"], emb["nn_model"], ft_mean)

            rows.append({
                "variant": disp, "window": w,
                # Table 2
                "lp_model": emb["lp_model"], "lp_random": emb["lp_random"],
                "nn_model": emb["nn_model"], "nn_random": emb["nn_random"],
                # Table 3 diagnostics
                "gc_bias": diag["gc_bias"], "fpr_bact_pct": diag["fpr_bact"], "fnr_phage_pct": diag["fnr_phage"],
                # Table 3 peak / fine-tuned
                "ft_mean": ft_mean, "ft_std": ft_std,
                "peak": peak_val, "peak_marker": peak_mark.strip(),
                # Table 3 genome-wide
                "raw_fpr": gw[0], "raw_recall": gw[1], "raw_mcc": gw[2],
                "filt_fpr": gw[3], "filt_recall": gw[4], "filt_mcc": gw[5],
            })

            if diag["missing"]:
                missing_report.append(f"  {disp} {w}: diagnostics missing -> {diag['missing']}")
            if not emb["present"]:
                missing_report.append(f"  {disp} {w}: embedding_analysis_results.json MISSING (LP/NN pending)")
            elif emb["unresolved_keys"]:
                missing_report.append(f"  {disp} {w}: LP/NN keys not recognized; keys present = {emb['available_keys']}")

    # ── Table 2 block ────────────────────────────────────────────────────────
    print("\n--- TABLE 2 (Linear Probe + 3-Layer NN MCC) ---")
    print(f"{'variant':<20}{'win':>4}{'LP-model':>11}{'LP-rand':>10}{'NN-model':>11}{'NN-rand':>10}")
    for r in rows:
        print(f"{r['variant']:<20}{r['window']:>4}{fmt(r['lp_model']):>11}{fmt(r['lp_random']):>10}"
              f"{fmt(r['nn_model']):>11}{fmt(r['nn_random']):>10}")

    # ── Table 3 block ────────────────────────────────────────────────────────
    print("\n--- TABLE 3 (Diagnostics + Peak + Genome-wide) ---")
    print(f"{'variant':<20}{'win':>4}{'GC-bias':>9}{'FPR-b':>8}{'FNR-p':>8}"
          f"{'Peak':>14}{'rawMCC':>9}{'filtMCC':>9}")
    for r in rows:
        peak_str = (f"{r['peak']:.3f}^{r['peak_marker']}" if r['peak'] is not None else "[MISSING]")
        print(f"{r['variant']:<20}{r['window']:>4}{fmt(r['gc_bias']):>9}"
              f"{fmt(r['fpr_bact_pct'], pct=True):>8}{fmt(r['fnr_phage_pct'], pct=True):>8}"
              f"{peak_str:>14}{r['raw_mcc']:>9.3f}{r['filt_mcc']:>9.3f}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    import csv
    with open(args.out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows)")

    # ── Missing / follow-up report ─────────────────────────────────────────────
    print("\n--- FOLLOW-UP (missing data) ---")
    if missing_report:
        for m in missing_report:
            print(m)
    else:
        print("  none -- all GENA table values resolved.")


if __name__ == "__main__":
    main()
