"""
Add COG category column to predictions CSV using Bakta JSON annotations.

The predictions CSV contains CDS from multiple genomes. The script extracts
the genome ID from each row's ID (e.g., GB_GCA_021108215.1) and looks up
the corresponding Bakta JSON file in the annotations directory.

For genes without COG annotation:
- If product contains "hypothetical", assign "hypothetical_protein"
- Otherwise assign "OTHER"

Usage:
    python -m src.add_cog_column \
        --input_csv /path/to/bacterial_cds_predictions.csv \
        --annotations_dir /path/to/bacteria_only_annotations \
        --output_csv /path/to/predictions_with_cog.csv

The annotations_dir should have structure:
    annotations_dir/
        GB_GCA_021108215.1/
            GB_GCA_021108215.1.json
        GB_GCA_040755155.1/
            GB_GCA_040755155.1.json
        ...
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


COG_DESCRIPTIONS = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure",
    "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "Z": "Cytoskeleton",
    "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, vesicular transport",
    "O": "Posttranslational modification, protein turnover, chaperones",
    "C": "Energy production and conversion",
    "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport, catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
    "X": "Mobilome: prophages, transposons",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Add COG category column to predictions CSV"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input predictions CSV file",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        required=True,
        help="Path to directory containing Bakta annotation folders",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV (default: input with _with_cog suffix)",
    )
    return parser.parse_args()


def extract_genome_id(id_string):
    """
    Extract genome ID (e.g., GB_GCA_021108215.1) from a CDS ID string.

    Example ID formats:
        GB_GCA_021108215.1_EDIBNM_00025_50933_51550
        GB_GCA_040755155.1_ABCDEF_00123_1000_2000
    """
    # Pattern: GB_GCA_XXXXXXXXX.X or RS_GCF_XXXXXXXXX.X
    match = re.search(r'((?:GB_GCA|RS_GCF)_\d+\.\d+)', str(id_string))
    if match:
        return match.group(1)
    return None


def extract_locus_tag(id_string):
    """
    Extract locus tag from a CDS ID string.

    Example: GB_GCA_021108215.1_EDIBNM_00025_50933_51550 -> EDIBNM_00025
    """
    # Pattern: genome_id followed by locus tag (letters + underscore + numbers)
    match = re.search(r'(?:GB_GCA|RS_GCF)_\d+\.\d+_([A-Z]+_\d+)', str(id_string))
    if match:
        return match.group(1)
    return None


def load_bakta_json(json_path):
    """Load Bakta JSON and extract locus_tag -> annotation mapping."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = {}
    for feature in data.get('features', []):
        if feature.get('type') != 'cds':
            continue

        locus_tag = feature.get('locus')
        if not locus_tag:
            continue

        annotations[locus_tag] = {
            'product': feature.get('product', ''),
            'cog_category': feature.get('cog_category', ''),
            'cog_id': feature.get('cog_id', ''),
        }

    return annotations


def assign_category(cog_category, product):
    """
    Assign a category based on COG or product.

    Returns COG category if available, otherwise:
    - "hypothetical_protein" if product contains "hypothetical"
    - "OTHER" otherwise
    """
    if cog_category and cog_category.strip():
        return cog_category

    if product and "hypothetical" in product.lower():
        return "hypothetical_protein"

    return "OTHER"


def main():
    args = parse_arguments()

    print("=" * 70)
    print("Adding COG Category Column")
    print("=" * 70)

    # Load input CSV
    print(f"\nLoading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  Rows: {len(df)}")

    # Find ID column (first column)
    id_column = df.columns[0]
    print(f"  ID column: {id_column}")

    # Get unique genome IDs
    genome_ids = set()
    for id_val in df[id_column]:
        gid = extract_genome_id(id_val)
        if gid:
            genome_ids.add(gid)

    print(f"  Unique genomes: {len(genome_ids)}")

    # Load Bakta annotations for each genome
    print(f"\nLoading Bakta annotations from: {args.annotations_dir}")
    genome_annotations = {}
    missing_genomes = []

    for genome_id in tqdm(genome_ids, desc="Loading JSON files"):
        json_path = os.path.join(args.annotations_dir, genome_id, f"{genome_id}.json")

        if os.path.exists(json_path):
            genome_annotations[genome_id] = load_bakta_json(json_path)
        else:
            missing_genomes.append(genome_id)

    print(f"  Loaded: {len(genome_annotations)} genomes")
    if missing_genomes:
        print(f"  Missing: {len(missing_genomes)} genomes")
        if len(missing_genomes) <= 5:
            for g in missing_genomes:
                print(f"    - {g}")

    # Process each row
    print("\nProcessing rows...")

    cog_categories = []
    cog_ids = []
    products = []
    categories = []

    matched = 0
    unmatched_genomes = 0
    unmatched_locus = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding COG"):
        id_value = row[id_column]

        genome_id = extract_genome_id(id_value)
        locus_tag = extract_locus_tag(id_value)

        cog_cat = ''
        cog_id = ''
        product = ''

        if genome_id and genome_id in genome_annotations:
            annotations = genome_annotations[genome_id]

            if locus_tag and locus_tag in annotations:
                matched += 1
                ann = annotations[locus_tag]
                cog_cat = ann['cog_category']
                cog_id = ann['cog_id']
                product = ann['product']
            else:
                unmatched_locus += 1
                # Try to get product from existing column
                product = row.get('annotation', row.get('product', ''))
        else:
            unmatched_genomes += 1
            product = row.get('annotation', row.get('product', ''))

        cog_categories.append(cog_cat)
        cog_ids.append(cog_id)
        products.append(product)
        categories.append(assign_category(cog_cat, product))

    # Add columns
    df['cog_category'] = cog_categories
    df['cog_id'] = cog_ids
    df['bakta_product'] = products
    df['category'] = categories

    print(f"\n  Matched: {matched} / {len(df)} ({100*matched/len(df):.1f}%)")
    print(f"  Unmatched (missing genome): {unmatched_genomes}")
    print(f"  Unmatched (missing locus): {unmatched_locus}")

    # Summary of categories
    print("\nCategory distribution:")
    cat_counts = df['category'].value_counts()
    for cat, count in cat_counts.head(20).items():
        desc = COG_DESCRIPTIONS.get(cat, cat)
        pct = 100 * count / len(df)
        print(f"  {cat:<25} {count:>6} ({pct:>5.1f}%)  {desc}")

    if len(cat_counts) > 20:
        print(f"  ... and {len(cat_counts) - 20} more categories")

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_with_cog{ext}"

    # Save
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to: {args.output_csv}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
