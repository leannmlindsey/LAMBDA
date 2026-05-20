#!/usr/bin/env python3
"""
Batch inference script that loads Evo2 ONCE and runs SAE, NN, and LP
inference on all input CSV files.

Usage:
    python src/batch_inference.py \
        --input_list scripts/input_files.txt \
        --output_dir results/inference/2k \
        --model_dir results/embedding_analysis/2k \
        --model evo2_7b \
        --run_sae --run_nn --run_lp
"""

import argparse
import csv
import json
import os
import pickle
import time
from math import prod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

torch.set_grad_enabled(False)


# ============================================================
# SAE model components (from sae_inference.py)
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
        from evo2 import Evo2
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


# ============================================================
# NN classifier
# ============================================================

class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Shared utilities
# ============================================================

def extract_embeddings(evo_model, sequences, layer_name, pooling, batch_size, max_length):
    """Extract embeddings using the Evo2 return_embeddings API."""
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="  Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        for seq in batch_seqs:
            if max_length is not None and len(seq) > max_length:
                seq = seq[:max_length]
            input_ids = torch.tensor(
                evo_model.tokenizer.tokenize(seq), dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')
            with torch.no_grad():
                outputs, embeddings = evo_model(
                    input_ids, return_embeddings=True, layer_names=[layer_name]
                )
                layer_emb = embeddings[layer_name]
                if pooling == "mean":
                    pooled = layer_emb.mean(dim=1)
                elif pooling == "first":
                    pooled = layer_emb[:, 0, :]
                elif pooling == "last":
                    pooled = layer_emb[:, -1, :]
                elif pooling == "max":
                    pooled = layer_emb.max(dim=1)[0]
                all_embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(all_embeddings)


def calculate_metrics(labels, predictions, prob_positive):
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, prob_positive))
    except ValueError:
        metrics["auc"] = 0.0
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    return metrics


def print_metrics(metrics, label=""):
    print(f"\n  {'=' * 50}")
    print(f"  METRICS {label}")
    print(f"  {'=' * 50}")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    F1 Score:    {metrics['f1']:.4f}")
    print(f"    MCC:         {metrics['mcc']:.4f}")
    print(f"    AUC:         {metrics['auc']:.4f}")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"  {'=' * 50}")


# ============================================================
# SAE inference for one file
# ============================================================

def run_sae_on_file(obs_model, sae, input_csv, output_csv, args):
    """Run SAE inference on a single CSV file."""
    rows = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        input_columns = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    # Auto-generate segment_id if not present
    has_segment_id = 'segment_id' in input_columns
    if not has_segment_id:
        has_coords = 'seq_id' in input_columns and 'start' in input_columns and 'end' in input_columns
        for i, row in enumerate(rows):
            if has_coords:
                row['segment_id'] = f"{row['seq_id']}_{row['start']}_{row['end']}"
            else:
                row['segment_id'] = f"seg_{i}"

    passthrough_columns = [c for c in input_columns if c not in ('segment_id', 'sequence')]
    computed_fields = ['max_activation', 'mean_activation', 'fraction_firing', 'pred_label']
    output_fields = ['segment_id'] + passthrough_columns + ['sequence'] + computed_fields

    output_path = Path(output_csv)
    act_dir = None
    if args.save_activations:
        act_dir = output_path.parent / f"{output_path.stem}_activations"
        act_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields)
        writer.writeheader()

        for row in tqdm(rows, desc="  SAE segments"):
            segment_id = row['segment_id']
            sequence = row['sequence']

            # Get SAE feature activations
            toks = obs_model.tokenizer.tokenize(sequence)
            toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(obs_model.device)
            logits, acts = obs_model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
            feats = sae.encode(acts[SAE_LAYER_NAME][0])
            feats_np = feats.cpu().detach().float().numpy()
            activations = feats_np[:, args.feature_idx]

            max_act = float(activations.max())
            mean_act = float(activations.mean())
            fraction = float((activations > 0).sum() / len(activations)) if len(activations) > 0 else 0.0
            pred_label = 1 if (max_act > args.sae_max_threshold
                               or mean_act > args.sae_mean_threshold
                               or fraction > args.sae_fraction_threshold) else 0

            out_row = {'segment_id': segment_id, 'sequence': sequence}
            for col in passthrough_columns:
                out_row[col] = row.get(col, '')
            out_row['max_activation'] = f"{max_act:.6f}"
            out_row['mean_activation'] = f"{mean_act:.6f}"
            out_row['fraction_firing'] = f"{fraction:.6f}"
            out_row['pred_label'] = pred_label
            writer.writerow(out_row)

            if act_dir is not None:
                np.save(act_dir / f"{segment_id}.npy", activations)

    print(f"  SAE results saved to: {output_csv}")


# ============================================================
# NN inference for one file
# ============================================================

def run_nn_on_file(embeddings, df, nn_clf, nn_scaler, output_csv, threshold, has_labels):
    """Run NN inference given pre-extracted embeddings."""
    device = next(nn_clf.parameters()).device
    embeddings_scaled = nn_scaler.transform(embeddings)
    embeddings_tensor = torch.FloatTensor(embeddings_scaled).to(device)

    with torch.no_grad():
        logits = nn_clf(embeddings_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    if threshold != 0.5:
        preds = (probs[:, 1] >= threshold).astype(int)

    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds
    output_df.to_csv(output_csv, index=False)
    print(f"  NN predictions saved to: {output_csv}")

    if has_labels:
        metrics = calculate_metrics(df["label"].values, preds, probs[:, 1])
        print_metrics(metrics, "(NN)")
        metrics_path = output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  NN metrics saved to: {metrics_path}")


# ============================================================
# LP inference for one file
# ============================================================

def run_lp_on_file(embeddings, df, lp_clf, lp_scaler, output_csv, threshold, has_labels):
    """Run linear probe inference given pre-extracted embeddings."""
    embeddings_scaled = lp_scaler.transform(embeddings)
    preds = lp_clf.predict(embeddings_scaled)
    probs = lp_clf.predict_proba(embeddings_scaled)

    if threshold != 0.5:
        preds = (probs[:, 1] >= threshold).astype(int)

    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds
    output_df.to_csv(output_csv, index=False)
    print(f"  LP predictions saved to: {output_csv}")

    if has_labels:
        metrics = calculate_metrics(df["label"].values, preds, probs[:, 1])
        print_metrics(metrics, "(LP)")
        metrics_path = output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  LP metrics saved to: {metrics_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference: loads Evo2 once, runs SAE/NN/LP on all input files"
    )
    parser.add_argument("--input_list", type=str, required=True,
                        help="Text file with one input CSV path per line")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all results")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with trained NN/LP artifacts (required if --run_nn or --run_lp)")

    # Methods to run
    parser.add_argument("--run_sae", action="store_true", help="Run SAE inference")
    parser.add_argument("--run_nn", action="store_true", help="Run 3-layer NN inference")
    parser.add_argument("--run_lp", action="store_true", help="Run linear probe inference")

    # Model config
    parser.add_argument("--model", type=str, default="evo2_7b")
    parser.add_argument("--layer", type=str, default="blocks.28.mlp.l3")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "first", "last", "max"])
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)

    # SAE config
    parser.add_argument("--feature_idx", type=int, default=19746)
    parser.add_argument("--sae_max_threshold", type=float, default=0.5)
    parser.add_argument("--sae_mean_threshold", type=float, default=0.1)
    parser.add_argument("--sae_fraction_threshold", type=float, default=0.3)
    parser.add_argument("--save_activations", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read input file list
    input_files = []
    with open(args.input_list, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                input_files.append(line)

    print("\n" + "=" * 60)
    print("Evo2 Batch Inference")
    print("=" * 60)
    print(f"  Input files:  {len(input_files)}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Methods:      SAE={args.run_sae}  NN={args.run_nn}  LP={args.run_lp}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Evo2 model (once)
    # ------------------------------------------------------------------
    print(f"\nLoading Evo2 model: {args.model}")

    if args.run_sae:
        # SAE needs ObservableEvo2 for hooks
        obs_model = ObservableEvo2(args.model)
        # For NN/LP embedding extraction, use the underlying Evo2 model
        evo_model = obs_model.evo_model
        print(f"  Loaded ObservableEvo2 (SAE + embedding extraction)")
    else:
        from evo2 import Evo2
        evo_model = Evo2(args.model)
        obs_model = None
        print(f"  Loaded Evo2 (embedding extraction only)")

    # ------------------------------------------------------------------
    # 2. Load SAE if needed
    # ------------------------------------------------------------------
    sae = None
    if args.run_sae:
        print("\nLoading SAE...")
        sae_path = hf_hub_download(
            repo_id="Goodfire/Evo-2-Layer-26-Mixed",
            filename="sae-layer26-mixed-expansion_8-k_64.pt",
            repo_type="model"
        )
        sae = load_topk_sae(sae_path, d_hidden=obs_model.d_hidden,
                            device=obs_model.device, dtype=torch.bfloat16,
                            expansion_factor=8)
        print(f"  SAE loaded: d_in={sae.d_in}, d_hidden={sae.d_hidden}")

    # ------------------------------------------------------------------
    # 3. Load NN/LP classifiers if needed
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_clf = None
    nn_scaler = None
    lp_clf = None
    lp_scaler = None

    if args.run_nn:
        if not args.model_dir:
            print("ERROR: --model_dir required for --run_nn")
            return
        print(f"\nLoading NN classifier from: {args.model_dir}")
        checkpoint = torch.load(
            os.path.join(args.model_dir, "three_layer_nn.pt"),
            map_location=device, weights_only=True
        )
        nn_clf = ThreeLayerNN(checkpoint["input_dim"], checkpoint["hidden_dim"]).to(device)
        nn_clf.load_state_dict(checkpoint["model_state_dict"])
        nn_clf.eval()
        with open(os.path.join(args.model_dir, "three_layer_nn_scaler.pkl"), "rb") as f:
            nn_scaler = pickle.load(f)
        print(f"  NN loaded (input_dim={checkpoint['input_dim']}, hidden_dim={checkpoint['hidden_dim']})")

    if args.run_lp:
        if not args.model_dir:
            print("ERROR: --model_dir required for --run_lp")
            return
        print(f"\nLoading LP classifier from: {args.model_dir}")
        with open(os.path.join(args.model_dir, "linear_probe.pkl"), "rb") as f:
            lp_clf = pickle.load(f)
        with open(os.path.join(args.model_dir, "linear_probe_scaler.pkl"), "rb") as f:
            lp_scaler = pickle.load(f)
        print(f"  LP loaded")

    # ------------------------------------------------------------------
    # 4. Process each input file
    # ------------------------------------------------------------------
    for file_idx, input_csv in enumerate(input_files):
        basename = os.path.splitext(os.path.basename(input_csv))[0]

        print(f"\n{'#' * 60}")
        print(f"File {file_idx + 1}/{len(input_files)}: {basename}")
        print(f"  Input: {input_csv}")
        print(f"{'#' * 60}")

        if not os.path.isfile(input_csv):
            print(f"  WARNING: File not found, skipping: {input_csv}")
            continue

        # --- SAE ---
        if args.run_sae:
            sae_output = os.path.join(args.output_dir, f"{basename}_sae_results.csv")
            print(f"\n  --- SAE Inference ---")
            run_sae_on_file(obs_model, sae, input_csv, sae_output, args)

        # --- Extract embeddings once for NN + LP ---
        embeddings = None
        df = None
        if args.run_nn or args.run_lp:
            print(f"\n  --- Extracting embeddings (layer={args.layer}) ---")
            df = pd.read_csv(input_csv)
            has_labels = "label" in df.columns
            sequences = df["sequence"].tolist()

            embeddings = extract_embeddings(
                evo_model, sequences,
                args.layer, args.pooling,
                args.batch_size, args.max_length,
            )
            print(f"  Embeddings shape: {embeddings.shape}")

        # --- NN ---
        if args.run_nn and embeddings is not None:
            nn_output = os.path.join(args.output_dir, f"{basename}_nn_predictions.csv")
            print(f"\n  --- NN Inference ---")
            run_nn_on_file(embeddings, df, nn_clf, nn_scaler,
                           nn_output, args.threshold, has_labels)

        # --- LP ---
        if args.run_lp and embeddings is not None:
            lp_output = os.path.join(args.output_dir, f"{basename}_lp_predictions.csv")
            print(f"\n  --- LP Inference ---")
            run_lp_on_file(embeddings, df, lp_clf, lp_scaler,
                           lp_output, args.threshold, has_labels)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Batch Inference Complete")
    print(f"  Processed {len(input_files)} files in {elapsed:.1f}s")
    print(f"  Results in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
