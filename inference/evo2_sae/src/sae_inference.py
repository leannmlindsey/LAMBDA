#!/usr/bin/env python3
"""
Standalone Evo2 SAE inference on short DNA segments.

Runs SAE feature extraction (default: f/19746 prophage detector) on a CSV
of ~2kb DNA segments and outputs per-segment activation metrics.

Required input CSV columns: sequence (and optionally segment_id)
All other input columns are passed through to the output CSV unchanged.

If segment_id is not present in the input, it is auto-generated from
{seq_id}_{start}_{end} (if those columns exist) or as seg_0, seg_1, etc.

Output CSV columns: segment_id, <all input columns except segment_id and sequence>,
                    sequence, max_activation, mean_activation, fraction_firing, pred_label

Usage:
    python sae_inference.py \
        --input_csv gc_control_2k_test.csv \
        --output gc_control_2k_results.csv \
        --threshold 0.5 \
        --save_activations
"""

import os
import sys
import csv
import json
import argparse
import torch
import numpy as np
import pandas as pd
from math import prod
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from evo2 import Evo2

torch.set_grad_enabled(False)


# ============================================================
# Model components (from run_lambda_batch.py)
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


def get_feature_ts(model, sae, seq):
    """Extract feature activations for a sequence."""
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    logits, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    feats = sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


# ============================================================
# Inference
# ============================================================

def process_segment(model, sae, sequence, feature_idx,
                    max_threshold, mean_threshold, fraction_threshold):
    """Run SAE inference on a single segment and return activation metrics.

    pred_label is 1 if ANY of the following conditions is met:
      - max_activation > max_threshold
      - mean_activation > mean_threshold
      - fraction_firing > fraction_threshold
    """
    feats = get_feature_ts(model, sae, sequence)
    activations = feats[:, feature_idx]

    max_act = float(activations.max())
    mean_act = float(activations.mean())
    fraction = float((activations > 0).sum() / len(activations)) if len(activations) > 0 else 0.0

    pred_label = 1 if (max_act > max_threshold
                       or mean_act > mean_threshold
                       or fraction > fraction_threshold) else 0

    return {
        'max_activation': max_act,
        'mean_activation': mean_act,
        'fraction_firing': fraction,
        'pred_label': pred_label,
    }, activations


def main():
    parser = argparse.ArgumentParser(
        description="Evo2 SAE inference on short DNA segments"
    )
    parser.add_argument("--input_csv", required=True,
                        help="Input CSV with required column: sequence. All other columns are passed through.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. results.csv)")
    parser.add_argument("--model", default="evo2_7b",
                        help="Evo2 model name (default: evo2_7b)")
    parser.add_argument("--feature_idx", type=int, default=19746,
                        help="SAE feature index (default: 19746)")
    parser.add_argument("--max_threshold", type=float, default=0.5,
                        help="Max activation threshold for pred_label (default: 0.5)")
    parser.add_argument("--mean_threshold", type=float, default=0.1,
                        help="Mean activation threshold for pred_label (default: 0.1)")
    parser.add_argument("--fraction_threshold", type=float, default=0.3,
                        help="Fraction firing threshold for pred_label (default: 0.3)")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save per-segment .npy activation arrays")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1, reserved for future use)")
    parser.add_argument("--save_metrics", action="store_true",
                        help="If 'label' column present, calculate and save metrics to JSON")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_prefix = output_path.stem

    # Read input CSV
    print(f"Reading {args.input_csv} ...")
    rows = []
    with open(args.input_csv, 'r') as f:
        reader = csv.DictReader(f)
        input_columns = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} segments")
    print(f"  Input columns: {input_columns}")

    # Auto-generate segment_id if not present
    has_segment_id = 'segment_id' in input_columns
    if not has_segment_id:
        has_coords = 'seq_id' in input_columns and 'start' in input_columns and 'end' in input_columns
        for i, row in enumerate(rows):
            if has_coords:
                row['segment_id'] = f"{row['seq_id']}_{row['start']}_{row['end']}"
            else:
                row['segment_id'] = f"seg_{i}"
        print(f"  Auto-generated segment_id" + (" from seq_id/start/end" if has_coords else " as seg_N"))

    # Determine passthrough columns (all input columns except segment_id and sequence)
    passthrough_columns = [c for c in input_columns if c not in ('segment_id', 'sequence')]

    # Load model
    print(f"\nLoading Evo2 model ({args.model}) ...")
    model = ObservableEvo2(model_name=args.model)
    print(f"  Device: {model.device}")

    # Load SAE
    print("Loading SAE ...")
    sae_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model"
    )
    sae = load_topk_sae(sae_path, d_hidden=model.d_hidden, device=model.device,
                        dtype=torch.bfloat16, expansion_factor=8)
    print(f"  SAE loaded: d_in={sae.d_in}, d_hidden={sae.d_hidden}")

    # Prepare activations directory
    act_dir = None
    if args.save_activations:
        act_dir = output_path.parent / f"{output_prefix}_activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Activation arrays will be saved to {act_dir}/")

    # Process segments
    print(f"\nRunning inference (feature={args.feature_idx}) ...")
    print(f"  Thresholds — max: {args.max_threshold}, mean: {args.mean_threshold}, fraction: {args.fraction_threshold}")
    print(f"  pred_label=1 if ANY threshold is exceeded")
    computed_fields = ['max_activation', 'mean_activation', 'fraction_firing', 'pred_label']
    output_fields = ['segment_id'] + passthrough_columns + ['sequence'] + computed_fields

    with open(output_path, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields)
        writer.writeheader()

        for row in tqdm(rows, desc="Segments"):
            segment_id = row['segment_id']
            sequence = row['sequence']

            metrics, activations = process_segment(
                model, sae, sequence, args.feature_idx,
                args.max_threshold, args.mean_threshold, args.fraction_threshold
            )

            out_row = {'segment_id': segment_id, 'sequence': sequence}
            # Pass through all extra columns
            for col in passthrough_columns:
                out_row[col] = row.get(col, '')
            out_row['max_activation'] = f"{metrics['max_activation']:.6f}"
            out_row['mean_activation'] = f"{metrics['mean_activation']:.6f}"
            out_row['fraction_firing'] = f"{metrics['fraction_firing']:.6f}"
            out_row['pred_label'] = metrics['pred_label']
            writer.writerow(out_row)

            if act_dir is not None:
                np.save(act_dir / f"{segment_id}.npy", activations)

    print(f"\nResults written to {output_path}")
    if act_dir is not None:
        print(f"Activation arrays saved to {act_dir}/")

    # Calculate and save aggregate metrics if labels present
    has_labels = 'label' in input_columns
    if has_labels and args.save_metrics:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef, roc_auc_score, confusion_matrix as sk_confusion_matrix,
        )
        labels = np.array([int(row['label']) for row in rows])
        preds = np.array([int(row.get('_pred_label', 0)) for row in rows])

        # Re-read predictions from the output CSV to get pred_label
        result_df = pd.read_csv(output_path)
        labels = result_df['label'].values.astype(int)
        preds = result_df['pred_label'].values.astype(int)

        tn, fp, fn, tp = sk_confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics_dict = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, preds)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "num_samples": len(labels),
            "input_csv": args.input_csv,
            "feature_idx": args.feature_idx,
            "max_threshold": args.max_threshold,
            "mean_threshold": args.mean_threshold,
            "fraction_threshold": args.fraction_threshold,
        }

        print(f"\n{'=' * 50}")
        print(f"METRICS")
        print(f"{'=' * 50}")
        print(f"  Accuracy:    {metrics_dict['accuracy']:.4f}")
        print(f"  Precision:   {metrics_dict['precision']:.4f}")
        print(f"  Recall:      {metrics_dict['recall']:.4f}")
        print(f"  F1 Score:    {metrics_dict['f1']:.4f}")
        print(f"  MCC:         {metrics_dict['mcc']:.4f}")
        print(f"  Sensitivity: {metrics_dict['sensitivity']:.4f}")
        print(f"  Specificity: {metrics_dict['specificity']:.4f}")
        print(f"{'=' * 50}")

        metrics_path = str(output_path).replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

    print("Done!")


if __name__ == "__main__":
    main()
