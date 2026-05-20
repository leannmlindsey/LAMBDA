#!/usr/bin/env python3
"""
Batch process all LAMBDA genomes with Evo2 SAE prophage feature f/19746.

This script:
1. Processes all FASTA files in the LAMBDA directory
2. Extracts feature 19746 activations for each genome
3. Compares to ground truth prophage regions
4. Generates summary statistics and per-genome results

Usage:
    python run_lambda_batch.py \
        --fasta_dir /path/to/FASTA \
        --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
        --output_dir ./lambda_results
"""

import os
import sys
import csv
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import prod
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import hf_hub_download
from evo2 import Evo2

torch.set_grad_enabled(False)


# ============================================================
# Model components (same as visualize_prophage_feature.py)
# ============================================================

class ModelScope:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self._build_module_dict()

    def _build_module_dict(self):
        self._module_dict = {}
        def recurse(module, prefix=''):
            for name, child in module.named_children():
                self._module_dict[prefix + name] = child
                recurse(child, prefix=prefix + name + '-')
        recurse(self.model)

    def add_hook(self, hook_fn, module_str, hook_name):
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle

    def remove_all_hooks(self):
        for hook_name in list(self.hooks.keys()):
            self.hooks[hook_name].remove()
            del self.hooks[hook_name]

    def clear_all_caches(self):
        for key in self.activations_cache:
            self.activations_cache[key] = []


class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, k, device, dtype, tiebreaker_epsilon=1e-6):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(device, dtype)

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)

    def _batch_topk(self, f, k, tiebreak=False):
        if tiebreak:
            f = f + self.tiebreaker.to(f.device).broadcast_to(f.shape)
        *input_shape, _ = f.shape
        numel = k * prod(input_shape)
        f_flat = f.flatten().float()
        f_topk = torch.topk(f_flat, numel, dim=-1)
        result = torch.zeros_like(f_flat).scatter(-1, f_topk.indices, f_topk.values)
        return result.reshape(f.shape).to(f.dtype)

    def decode(self, f):
        return f @ self.W.T + self.b_dec


def load_topk_sae(sae_path, d_hidden, device, dtype, expansion_factor=8):
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")
    new_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sae_dict.items()}
    cached_sae = BatchTopKTiedSAE(d_hidden, d_hidden * expansion_factor, 64, device, dtype)
    cached_sae.load_state_dict(new_dict)
    return cached_sae


class ObservableEvo2:
    def __init__(self, model_name):
        self.evo_model = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self):
        return next(self.evo_model.model.parameters()).device

    def forward(self, toks, cache_activations_at=None):
        if not cache_activations_at:
            cache_activations_at = []
        output_cache = {}

        for layer in cache_activations_at:
            def _intervene(model, input, output, layer=layer):
                acts = output[0] if isinstance(output, tuple) else output
                output_cache[layer] = acts.detach()
                return (acts, output[1]) if isinstance(output, tuple) else acts
            self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        try:
            model_outputs = self.model(toks)
            cached_activations = {layer: act.clone() for layer, act in output_cache.items()}
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()

        return model_outputs[0], cached_activations


SAE_LAYER_NAME = 'blocks-26'
PROPHAGE_FEATURE = 19746


def get_feature_ts(model, sae, seq):
    """Extract feature activations for a sequence."""
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    logits, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    feats = sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


# ============================================================
# Data loading
# ============================================================

def load_fasta(fasta_path):
    """Load sequence from FASTA file."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_name:
            sequences[current_name] = ''.join(current_seq)

    return sequences


def load_ground_truth(csv_path):
    """Load all ground truth prophage regions."""
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['Assembly']
            if assembly not in gt:
                gt[assembly] = []
            gt[assembly].append({
                'ncbi_id': row['NCBI Id'],
                'start': int(row['start']),
                'end': int(row['end']),
                'organism': row['Organism Name'],
            })
    return gt


def get_assembly_id(fasta_filename):
    """Extract assembly ID from FASTA filename."""
    # Handle different naming conventions
    name = Path(fasta_filename).stem

    # NC_000913.fasta -> NC_000913
    if name.startswith('NC_'):
        return name

    # GCF_000006665.1_ASM666v1_genomic.fna -> GCF_000006665.1
    # GCA_000284555.1_ASM28455v1_genomic.fna -> GCA_000284555.1
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] in ['GCF', 'GCA']:
        return f"{parts[0]}_{parts[1]}"

    return name


# ============================================================
# Processing
# ============================================================

def process_genome(model, sae, fasta_path, gt_regions, output_dir, window_size=50000, startup_trim=10):
    """Process a single genome and return results.

    Args:
        startup_trim: Number of positions to discard at start of each window
                     to avoid model startup artifacts (default: 10)
    """

    assembly_id = get_assembly_id(fasta_path)
    results = {
        'assembly': assembly_id,
        'fasta': str(fasta_path),
        'ground_truth_count': len(gt_regions),
        'ground_truth_regions': gt_regions,
    }

    # Load sequence
    sequences = load_fasta(fasta_path)
    if not sequences:
        results['error'] = 'No sequences found in FASTA'
        return results

    # Use first/main sequence
    seq_name = list(sequences.keys())[0]
    full_seq = sequences[seq_name]
    results['sequence_name'] = seq_name
    results['sequence_length'] = len(full_seq)

    # Extract activations for full genome in windows
    overlap = 1000
    all_acts = np.zeros(len(full_seq))

    positions = list(range(0, len(full_seq), window_size - overlap))
    for win_idx, win_start in enumerate(tqdm(positions, desc=f"  {assembly_id}", leave=False)):
        win_end = min(win_start + window_size, len(full_seq))
        win_seq = full_seq[win_start:win_end]
        if len(win_seq) < 100:
            continue
        try:
            win_feats = get_feature_ts(model, sae, win_seq)
            win_acts = win_feats[:, PROPHAGE_FEATURE]

            # Trim startup artifact (first N positions have artificially high activations)
            # For the first window, keep all positions; for subsequent windows, trim startup
            if win_idx > 0 and startup_trim > 0:
                win_acts[:startup_trim] = 0.0

            actual_len = min(len(win_acts), win_end - win_start)
            all_acts[win_start:win_start+actual_len] = np.maximum(
                all_acts[win_start:win_start+actual_len],
                win_acts[:actual_len]
            )
        except Exception as e:
            pass  # Skip failed windows

    # Global stats
    results['total_positions_above_threshold'] = int(sum(all_acts > 0.5))
    results['max_activation'] = float(all_acts.max())
    results['mean_activation'] = float(all_acts.mean())

    # Per-region stats
    region_stats = []
    for i, r in enumerate(gt_regions):
        start, end = r['start'], r['end']
        if start >= len(full_seq) or end > len(full_seq):
            continue
        region_acts = all_acts[start:end]

        stats = {
            'region_idx': i,
            'start': start,
            'end': end,
            'length': end - start,
            'max_activation': float(region_acts.max()),
            'mean_activation': float(region_acts.mean()),
            'positions_above_05': int(sum(region_acts > 0.5)),
            'positions_above_01': int(sum(region_acts > 0.1)),
            'density_above_05': float(sum(region_acts > 0.5) / len(region_acts)) if len(region_acts) > 0 else 0,
        }
        region_stats.append(stats)

    results['region_stats'] = region_stats

    # Calculate what fraction of all firing positions are in ground truth regions
    total_in_gt = sum(s['positions_above_05'] for s in region_stats)
    total_firing = results['total_positions_above_threshold']
    results['fraction_in_ground_truth'] = total_in_gt / total_firing if total_firing > 0 else 0

    # Save activations
    np.save(output_dir / f"{assembly_id}_activations.npy", all_acts)

    return results


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch process LAMBDA genomes with Evo2 SAE")
    parser.add_argument("--fasta_dir", type=str, required=True, help="Directory containing FASTA files")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth CSV file")
    parser.add_argument("--output_dir", type=str, default="./lambda_results", help="Output directory")
    parser.add_argument("--model", type=str, default="evo2_7b", help="Model name (evo2_7b or evo2_40b)")
    parser.add_argument("--window_size", type=int, default=50000, help="Window size for processing")
    parser.add_argument("--startup_trim", type=int, default=10, help="Positions to trim from window start to remove artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LAMBDA Batch Processing with Evo2 SAE")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"FASTA dir: {args.fasta_dir}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")

    # Load model and SAE
    print("\nLoading Evo2 model...")
    model = ObservableEvo2(model_name=args.model)
    print(f"  Device: {model.device}")

    print("Loading SAE...")
    sae_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model"
    )
    sae = load_topk_sae(sae_path, d_hidden=model.d_hidden, device=model.device,
                        dtype=torch.bfloat16, expansion_factor=8)
    print(f"  Loaded SAE: d_in={sae.d_in}, d_hidden={sae.d_hidden}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  Found {len(gt)} assemblies with ground truth")
    total_regions = sum(len(regions) for regions in gt.values())
    print(f"  Total prophage regions: {total_regions}")

    # Find FASTA files
    fasta_dir = Path(args.fasta_dir)
    fasta_files = list(fasta_dir.glob("*.fna")) + list(fasta_dir.glob("*.fasta"))
    print(f"\nFound {len(fasta_files)} FASTA files")

    # Process each genome
    all_results = []

    for fasta_path in tqdm(fasta_files, desc="Processing genomes"):
        assembly_id = get_assembly_id(fasta_path)

        # Get ground truth for this assembly
        gt_regions = gt.get(assembly_id, [])

        # Also try matching by NCBI ID in the ground truth
        if not gt_regions:
            for gt_assembly, regions in gt.items():
                if regions and regions[0].get('ncbi_id') == assembly_id:
                    gt_regions = regions
                    break

        results = process_genome(model, sae, fasta_path, gt_regions, output_dir, args.window_size, args.startup_trim)
        all_results.append(results)

        # Print progress
        if gt_regions:
            frac = results.get('fraction_in_ground_truth', 0)
            print(f"  {assembly_id}: {len(gt_regions)} regions, {results.get('total_positions_above_threshold', 0)} firing, {frac:.1%} in GT")

    # Save all results
    results_file = output_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    genomes_with_gt = [r for r in all_results if r.get('ground_truth_count', 0) > 0]
    print(f"Genomes with ground truth: {len(genomes_with_gt)} / {len(all_results)}")

    if genomes_with_gt:
        avg_fraction = np.mean([r.get('fraction_in_ground_truth', 0) for r in genomes_with_gt])
        print(f"Average fraction of firing positions in ground truth: {avg_fraction:.1%}")

        # Per-region stats
        all_region_stats = []
        for r in genomes_with_gt:
            all_region_stats.extend(r.get('region_stats', []))

        if all_region_stats:
            avg_density = np.mean([s['density_above_05'] for s in all_region_stats])
            avg_max = np.mean([s['max_activation'] for s in all_region_stats])
            print(f"Average density (>0.5) in GT regions: {avg_density:.2%}")
            print(f"Average max activation in GT regions: {avg_max:.2f}")

    # Save summary CSV
    summary_data = []
    for r in all_results:
        row = {
            'assembly': r.get('assembly'),
            'sequence_length': r.get('sequence_length'),
            'ground_truth_count': r.get('ground_truth_count', 0),
            'total_firing': r.get('total_positions_above_threshold', 0),
            'max_activation': r.get('max_activation', 0),
            'fraction_in_gt': r.get('fraction_in_ground_truth', 0),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    print(f"\nEnd time: {datetime.now()}")
    print("Done!")


if __name__ == "__main__":
    main()
