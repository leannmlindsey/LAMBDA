#!/usr/bin/env python3
"""
Download INPHARED phage genome database files.

INPHARED: https://github.com/RyanCook94/inphared
Data hosted at: https://millardlab-inphared.s3.climb.ac.uk/
"""

import argparse
import os
import gzip
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error


# S3 bucket base URL
S3_BASE_URL = "https://millardlab-inphared.s3.climb.ac.uk/"

# All available file suffixes (without date prefix)
FILE_SUFFIXES = [
    # Core data files
    "data.tsv.gz",
    "data_excluding_refseq.tsv.gz",
    "genomes.fa.gz",
    "genomes_excluding_refseq.fa.gz",
    "refseq_genomes.fa.gz",
    "genomes.db.gz",
    "genomes.fa.msh.gz",
    # GenBank records
    "phages_downloaded_from_genbank.gb.gz",
    # IToL annotation files
    "itol_family_annotations.txt.gz",
    "itol_genus_annotations.txt.gz",
    "itol_host_annotations.txt.gz",
    "itol_length_annotations.txt.gz",
    "itol_lowest_taxa_annotations.txt.gz",
    "itol_node_label_annotations.txt.gz",
    "itol_subfamily_annotations.txt.gz",
    # vConTACT2 files
    "vConTACT2_proteins.faa.gz",
    "vConTACT2_gene_to_genome.csv.gz",
    "vConTACT2_family_annotations.tsv.gz",
    "vConTACT2_genus_annotations.tsv.gz",
    "vConTACT2_host_annotations.tsv.gz",
    "vConTACT2_lowest_taxa_annotations.tsv.gz",
    "vConTACT2_subfamily_annotations.tsv.gz",
]

# Additional standalone files
ADDITIONAL_FILES = {
    "phrogs_hmm": "https://warwick.s3.climb.ac.uk/ADM_share/all_phrogs.hmm.gz",
    "genomesdb_archive": "https://millardlab-inphared.s3.climb.ac.uk/GenomesDB12102022.tar.gz",
}

# File categories for selective downloads
FILE_CATEGORIES = {
    "core": [
        "data.tsv.gz",
        "genomes.fa.gz",
    ],
    "metadata": [
        "data.tsv.gz",
        "data_excluding_refseq.tsv.gz",
    ],
    "genomes": [
        "genomes.fa.gz",
        "genomes_excluding_refseq.fa.gz",
        "refseq_genomes.fa.gz",
    ],
    "databases": [
        "genomes.db.gz",
        "genomes.fa.msh.gz",
    ],
    "itol": [
        "itol_family_annotations.txt.gz",
        "itol_genus_annotations.txt.gz",
        "itol_host_annotations.txt.gz",
        "itol_length_annotations.txt.gz",
        "itol_lowest_taxa_annotations.txt.gz",
        "itol_node_label_annotations.txt.gz",
        "itol_subfamily_annotations.txt.gz",
    ],
    "vcontact2": [
        "vConTACT2_proteins.faa.gz",
        "vConTACT2_gene_to_genome.csv.gz",
        "vConTACT2_family_annotations.tsv.gz",
        "vConTACT2_genus_annotations.tsv.gz",
        "vConTACT2_host_annotations.tsv.gz",
        "vConTACT2_lowest_taxa_annotations.tsv.gz",
        "vConTACT2_subfamily_annotations.tsv.gz",
    ],
    "genbank": [
        "phages_downloaded_from_genbank.gb.gz",
    ],
    "all": FILE_SUFFIXES,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download INPHARED phage genome database files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download core files (metadata + genomes) with latest date
  python download_inphared.py -o ./inphared_data --category core

  # Download all files for a specific date
  python download_inphared.py -o ./inphared_data --date 14Apr2025 --category all

  # Download only genome sequences
  python download_inphared.py -o ./inphared_data --category genomes

  # Download and decompress
  python download_inphared.py -o ./inphared_data --category core --decompress

Categories available: core, metadata, genomes, databases, itol, vcontact2, genbank, all
        """
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date string for files (e.g., '14Apr2025'). If not specified, tries common recent dates."
    )
    parser.add_argument(
        "--category", "-c",
        default="core",
        choices=list(FILE_CATEGORIES.keys()),
        help="Category of files to download (default: core)"
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .gz files after downloading"
    )
    parser.add_argument(
        "--include-phrogs",
        action="store_true",
        help="Also download PHROGs HMM database"
    )
    parser.add_argument(
        "--include-genomesdb",
        action="store_true",
        help="Also download GenomesDB archive (large file)"
    )
    parser.add_argument(
        "--use-wget",
        action="store_true",
        help="Use wget instead of Python urllib (may be faster for large files)"
    )
    parser.add_argument(
        "--use-curl",
        action="store_true",
        help="Use curl instead of Python urllib (may be faster for large files)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    return parser.parse_args()


def check_url_exists(url):
    """Check if a URL exists without downloading the full file."""
    try:
        request = urllib.request.Request(url, method='HEAD')
        response = urllib.request.urlopen(request, timeout=10)
        return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def find_latest_date():
    """Try to find the latest available date by checking common patterns."""
    # Generate potential dates (current month going back several months)
    now = datetime.now()
    potential_dates = []

    for months_back in range(12):
        year = now.year
        month = now.month - months_back
        while month <= 0:
            month += 12
            year -= 1

        # Try different day options (1st and 14th are common release days)
        for day in [1, 14, 15]:
            try:
                date = datetime(year, month, day)
                date_str = date.strftime("%d%b%Y")
                potential_dates.append(date_str)
            except ValueError:
                continue

    # Also add some known dates
    known_dates = ["14Apr2025", "1Apr2025", "14Mar2025", "1Mar2025", "14Feb2025"]
    potential_dates = known_dates + potential_dates

    print("Searching for latest available dataset...")
    for date_str in potential_dates:
        test_url = f"{S3_BASE_URL}{date_str}_data.tsv.gz"
        if check_url_exists(test_url):
            print(f"Found dataset: {date_str}")
            return date_str

    return None


def download_with_urllib(url, output_path):
    """Download a file using urllib with progress indication."""
    try:
        print(f"  Downloading: {os.path.basename(output_path)}")

        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()  # New line after progress
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  Error: HTTP {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  Error: {e.reason}")
        return False


def download_with_wget(url, output_path):
    """Download a file using wget."""
    print(f"  Downloading with wget: {os.path.basename(output_path)}")
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", output_path, url],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("  Error: wget not found. Install wget or use --use-curl or default urllib.")
        return False


def download_with_curl(url, output_path):
    """Download a file using curl."""
    print(f"  Downloading with curl: {os.path.basename(output_path)}")
    try:
        result = subprocess.run(
            ["curl", "-L", "-#", "-o", output_path, url],
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("  Error: curl not found. Install curl or use --use-wget or default urllib.")
        return False


def decompress_gzip(gz_path):
    """Decompress a .gz file using gunzip (memory-efficient for large files)."""
    print(f"  Decompressing: {os.path.basename(gz_path)}")
    try:
        # Use gunzip which is memory-efficient (streams data)
        result = subprocess.run(
            ["gunzip", "-f", gz_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error decompressing with gunzip: {e.stderr}")
        return False
    except FileNotFoundError:
        # Fallback to Python gzip if gunzip not available
        print(f"  gunzip not found, using Python gzip (may use more memory)...")
        output_path = gz_path[:-3]
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1024*1024)  # 1MB chunks
            os.remove(gz_path)
            return True
        except Exception as e:
            print(f"  Error decompressing: {e}")
            return False


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine date
    date_str = args.date
    if date_str is None:
        date_str = find_latest_date()
        if date_str is None:
            print("Error: Could not find available dataset. Please specify --date manually.")
            print("Example: --date 14Apr2025")
            return 1

    # Get files to download
    files_to_download = FILE_CATEGORIES[args.category]

    # Select download method
    if args.use_wget:
        download_func = download_with_wget
    elif args.use_curl:
        download_func = download_with_curl
    else:
        download_func = download_with_urllib

    print(f"\nINPHARED Data Download")
    print("=" * 60)
    print(f"Date: {date_str}")
    print(f"Category: {args.category}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Files to download: {len(files_to_download)}")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - Files that would be downloaded:")
        for suffix in files_to_download:
            filename = f"{date_str}_{suffix}"
            url = f"{S3_BASE_URL}{filename}"
            print(f"  {url}")
        if args.include_phrogs:
            print(f"  {ADDITIONAL_FILES['phrogs_hmm']}")
        if args.include_genomesdb:
            print(f"  {ADDITIONAL_FILES['genomesdb_archive']}")
        return 0

    # Download main files
    successful = 0
    failed = 0

    print(f"\nDownloading {args.category} files...")
    for suffix in files_to_download:
        filename = f"{date_str}_{suffix}"
        url = f"{S3_BASE_URL}{filename}"
        output_path = output_dir / filename

        if output_path.exists():
            print(f"  Skipping (exists): {filename}")
            successful += 1
            continue

        if download_func(url, str(output_path)):
            successful += 1
            if args.decompress and str(output_path).endswith('.gz'):
                decompress_gzip(str(output_path))
        else:
            failed += 1

    # Download additional files if requested
    if args.include_phrogs:
        print("\nDownloading PHROGs HMM database...")
        url = ADDITIONAL_FILES['phrogs_hmm']
        output_path = output_dir / "all_phrogs.hmm.gz"
        if not output_path.exists():
            if download_func(url, str(output_path)):
                successful += 1
                if args.decompress:
                    decompress_gzip(str(output_path))
            else:
                failed += 1
        else:
            print(f"  Skipping (exists): all_phrogs.hmm.gz")

    if args.include_genomesdb:
        print("\nDownloading GenomesDB archive (this may take a while)...")
        url = ADDITIONAL_FILES['genomesdb_archive']
        output_path = output_dir / "GenomesDB12102022.tar.gz"
        if not output_path.exists():
            if download_func(url, str(output_path)):
                successful += 1
            else:
                failed += 1
        else:
            print(f"  Skipping (exists): GenomesDB12102022.tar.gz")

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir.absolute()}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
