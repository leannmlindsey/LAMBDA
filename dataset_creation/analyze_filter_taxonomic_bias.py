#!/usr/bin/env python3
"""
Taxonomic bias check for the bacterial prophage-contamination filter.

The filter excluded 971 of 15,865 GTDB representatives whose segments had
BLAST hits to INPHARED phage references (>=90% identity, >=200 bp alignment).
This analysis tests whether that exclusion was taxonomically biased.

For each accession, look up the GTDB phylum and class assignment. Compute
the proportion of excluded vs retained genomes per phylum. Run a chi-squared
goodness-of-fit test comparing the excluded distribution to the retained
distribution.

Outputs:
  - filter_taxonomic_bias.csv          per-phylum counts and proportions
  - filter_taxonomic_bias.png          comparison bar chart at phylum level
  - filter_taxonomic_bias_summary.txt  chi-squared result + one-line verdict
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chisquare


def extract_taxonomy_level(taxonomy_string, level):
    """Extract a specific taxonomy level from GTDB taxonomy string.

    GTDB format: d__Bacteria;p__Phylum;c__Class;o__Order;f__Family;g__Genus;s__Species
    (Same helper as select_gtdb_representatives.py:66-71)
    """
    pattern = f'{level}__([^;]*)'
    match = re.search(pattern, taxonomy_string)
    return match.group(1) if match else "Unknown"


def load_accessions(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether the prophage filter introduced taxonomic bias"
    )
    parser.add_argument(
        "--contaminated", required=True,
        help="contaminated_accessions.txt (excluded by filter)",
    )
    parser.add_argument(
        "--retained-files", required=True, nargs="+",
        help="One or more retained accession lists (typically "
             "train_accessions.txt dev_accessions.txt test_accessions.txt)",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="GTDB bac120_metadata.tsv",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for the CSV, PNG, and summary text",
    )
    parser.add_argument(
        "--min-phylum-count", type=int, default=5,
        help="Phyla with fewer than this many total genomes are pooled into "
             "an 'Other' bucket for the chi-squared test (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load accession lists
    excluded = set(load_accessions(args.contaminated))
    retained = set()
    for f in args.retained_files:
        retained.update(load_accessions(f))

    print(f"Excluded accessions: {len(excluded)}")
    print(f"Retained accessions: {len(retained)}")
    overlap = excluded & retained
    if overlap:
        print(f"  WARNING: {len(overlap)} accessions appear in both lists. "
              "Treating as retained (intended behaviour for the analysis).")
        excluded -= overlap

    # Load GTDB metadata
    print(f"\nLoading GTDB metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata, sep="\t", low_memory=False)

    # GTDB metadata accessions may have RS_ / GB_ prefixes; the saved
    # accession lists may not. Normalize on both sides for the lookup.
    def normalize(a):
        if isinstance(a, str) and (a.startswith("RS_") or a.startswith("GB_")):
            return a[3:]
        return a

    df["accession_norm"] = df["accession"].astype(str).apply(normalize)
    df["phylum"] = df["gtdb_taxonomy"].apply(lambda x: extract_taxonomy_level(x, "p"))
    df["class"]  = df["gtdb_taxonomy"].apply(lambda x: extract_taxonomy_level(x, "c"))

    excluded_norm = {normalize(a) for a in excluded}
    retained_norm = {normalize(a) for a in retained}

    df_excluded = df[df["accession_norm"].isin(excluded_norm)]
    df_retained = df[df["accession_norm"].isin(retained_norm)]
    print(f"  Matched excluded in metadata: {len(df_excluded)} / {len(excluded_norm)}")
    print(f"  Matched retained in metadata: {len(df_retained)} / {len(retained_norm)}")

    # Per-phylum counts
    counts_excl = Counter(df_excluded["phylum"])
    counts_ret  = Counter(df_retained["phylum"])
    phyla = sorted(set(counts_excl.keys()) | set(counts_ret.keys()))

    # Build the per-phylum table
    rows = []
    for p in phyla:
        n_excl = counts_excl.get(p, 0)
        n_ret  = counts_ret.get(p, 0)
        n_total = n_excl + n_ret
        rows.append({
            "phylum": p,
            "n_excluded": n_excl,
            "n_retained": n_ret,
            "n_total": n_total,
            "frac_excluded_of_phylum": (n_excl / n_total) if n_total else 0.0,
            "frac_of_excluded_set":    (n_excl / max(1, sum(counts_excl.values()))),
            "frac_of_retained_set":    (n_ret  / max(1, sum(counts_ret.values()))),
        })
    tbl = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    csv_path = out_dir / "filter_taxonomic_bias.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Chi-squared goodness-of-fit: are excluded proportions consistent with
    # retained proportions at the phylum level?
    # Pool small phyla into "Other" to avoid low-expected-count issues.
    keep = tbl["n_total"] >= args.min_phylum_count
    pooled = tbl[~keep][["n_excluded", "n_retained"]].sum()
    grouped = tbl[keep][["phylum", "n_excluded", "n_retained"]].copy()
    if pooled["n_excluded"] + pooled["n_retained"] > 0:
        grouped = pd.concat([grouped, pd.DataFrame([{
            "phylum": "Other (pooled)",
            "n_excluded": int(pooled["n_excluded"]),
            "n_retained": int(pooled["n_retained"]),
        }])], ignore_index=True)

    obs = grouped["n_excluded"].values.astype(float)
    ret = grouped["n_retained"].values.astype(float)
    total_excl = obs.sum()
    total_ret  = ret.sum()
    if total_ret > 0 and total_excl > 0:
        # Expected exclusions per phylum if exclusion proportions matched
        # the retained-set proportions exactly
        exp = (ret / total_ret) * total_excl
        # Guard against zero-expected entries (shouldn't happen post-pool)
        mask = exp > 0
        chi2, pval = chisquare(f_obs=obs[mask], f_exp=exp[mask])
    else:
        chi2, pval = float("nan"), float("nan")

    # Summary text
    summary_lines = [
        f"Excluded genomes: {len(df_excluded)}",
        f"Retained genomes: {len(df_retained)}",
        f"Phyla represented overall: {len(phyla)}",
        f"Phyla used in chi-squared test (after pooling phyla with n_total < {args.min_phylum_count}): {len(grouped)}",
        f"chi-squared statistic: {chi2:.3f}",
        f"chi-squared p-value:   {pval:.4g}",
        "",
        ("VERDICT: No statistically significant taxonomic bias at the phylum level "
         f"(p = {pval:.4g})." if pval > 0.05 else
         f"VERDICT: Possible taxonomic bias at the phylum level (p = {pval:.4g}). "
         "Inspect filter_taxonomic_bias.csv for the per-phylum breakdown."),
    ]
    summary_path = out_dir / "filter_taxonomic_bias_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print()
    for line in summary_lines:
        print(line)

    # Bar chart
    plot_tbl = grouped.copy()
    plot_tbl["frac_excluded"] = plot_tbl["n_excluded"] / max(1, total_excl)
    plot_tbl["frac_retained"] = plot_tbl["n_retained"] / max(1, total_ret)
    plot_tbl = plot_tbl.sort_values("frac_retained", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_tbl) * 0.5), 5))
    x = np.arange(len(plot_tbl))
    width = 0.4
    ax.bar(x - width / 2, plot_tbl["frac_retained"], width,
           label=f"Retained (n={int(total_ret)})", color="steelblue")
    ax.bar(x + width / 2, plot_tbl["frac_excluded"], width,
           label=f"Excluded (n={int(total_excl)})", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_tbl["phylum"], rotation=45, ha="right")
    ax.set_ylabel("Proportion of set")
    ax.set_title("Phylum-level taxonomic distribution: excluded vs retained "
                 f"({pval:.3g} chi-squared)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / "filter_taxonomic_bias.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png_path}")


if __name__ == "__main__":
    main()
