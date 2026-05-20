"""
Inference Script for Caduceus

This script performs inference on a CSV file using a fine-tuned Caduceus model.
It replicates the exact forward pass used during training.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or threshold-based)

Usage:
    python -m src.inference \
        --input_csv /path/to/test.csv \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --config_path /path/to/config.json \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

# Caduceus-specific imports
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import Caduceus
from transformers import AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with fine-tuned Caduceus model"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned Caduceus checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Caduceus config JSON file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--d_output",
        type=int,
        default=2,
        help="Number of output classes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for prob_1 (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    parser.add_argument(
        "--conjoin_test",
        action="store_true",
        help="Use post-hoc reverse complement conjoining (for Caduceus-Ph models)",
    )
    return parser.parse_args()


def get_complement_map():
    """
    Get the complement map for DNA tokenization.
    Maps token IDs to their reverse complement token IDs.
    """
    complement_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
        7: 10,  # A -> T
        8: 9,   # C -> G
        9: 8,   # G -> C
        10: 7,  # T -> A
        11: 11, # N -> N
    }
    return complement_map


class DNAEmbeddingModelCaduceus(nn.Module):
    """
    Caduceus backbone model - matches training exactly.
    From src/models/sequence/dna_embedding.py
    """

    def __init__(self, config: CaduceusConfig, conjoin_train=False, conjoin_test=False):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.caduceus = Caduceus(config=config)
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None):
        """Caduceus backbone-specific forward pass - matches training exactly."""
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            hidden_states = self.caduceus(input_ids, return_dict=False)
            num_chan = hidden_states.shape[-1]
            return torch.stack(
                [hidden_states[..., :num_chan // 2], torch.flip(hidden_states[..., num_chan // 2:], dims=[1, 2])],
                dim=-1
            ), None
        if self.conjoin_train or (self.conjoin_test and not self.training):
            assert input_ids.ndim == 3, "Input must be 3D tensor for conjoin mode"
            hidden_states = self.caduceus(input_ids[..., 0], return_dict=False)
            hidden_states_rc = self.caduceus(input_ids[..., 1], return_dict=False)
            return torch.stack([hidden_states, hidden_states_rc], dim=-1), None

        return self.caduceus(input_ids, return_dict=False), None

    @property
    def d_output(self):
        return self.d_model


class SequenceDecoder(nn.Module):
    """
    Sequence decoder - matches training exactly.
    From src/tasks/decoders.py
    """

    def __init__(self, d_model, d_output, l_output=0, mode="pool", conjoin_train=True, conjoin_test=False):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)

        if l_output == 0:
            self.l_output = 1
            self.squeeze = True
        else:
            self.l_output = l_output
            self.squeeze = False

        self.mode = mode
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model) or (n_batch, l_seq, d_model, 2) if using conjoin/RCPS
        Returns: (n_batch, d_output)
        """
        l_output = self.l_output
        squeeze = self.squeeze

        # Restrict along sequence dimension (dim=1)
        if self.mode == "last":
            x = x[:, -l_output:, ...]
        elif self.mode == "first":
            x = x[:, :l_output, ...]
        elif self.mode == "pool":
            x = x.mean(dim=1, keepdim=True)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        # conjoin_train=True means always chunk and average (even at inference)
        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x.squeeze(-1))
            x_rc = self.output_transform(x_rc.squeeze(-1))
            x = (x + x_rc) / 2
        else:
            x = self.output_transform(x)

        return x


class CaduceusForInference(nn.Module):
    """
    Full model for inference - backbone + decoder.
    Matches the training architecture exactly.
    """

    def __init__(self, config: CaduceusConfig, d_output: int = 2, conjoin_test: bool = False):
        super().__init__()
        self.config = config

        # Backbone (model in training)
        self.model = DNAEmbeddingModelCaduceus(
            config=config,
            conjoin_train=False,
            conjoin_test=conjoin_test,
        )

        # Decoder - matches training config:
        # mode='pool', conjoin_train=True, conjoin_test=False
        self.decoder = SequenceDecoder(
            d_model=config.d_model,
            d_output=d_output,
            l_output=0,  # means squeeze to single output
            mode="pool",
            conjoin_train=True,
            conjoin_test=False,
        )

    def forward(self, input_ids):
        """Forward pass matching training."""
        hidden_states, _ = self.model(input_ids)
        logits = self.decoder(hidden_states)
        return logits


def load_model(
    config_path: str,
    checkpoint_path: str,
    d_output: int,
    conjoin_test: bool,
    device: torch.device,
) -> CaduceusForInference:
    """Load Caduceus model from checkpoint."""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Handle nested Hydra config format
    if 'config' in config_dict and isinstance(config_dict['config'], dict):
        print("  Detected nested Hydra config format, extracting inner config...")
        inner_config = config_dict['config']
        inner_config = {k: v for k, v in inner_config.items() if not k.startswith('_')}
        config_dict = inner_config

    # Handle complement_map being null or missing
    if not config_dict.get('complement_map'):
        print("  Setting complement_map for DNA tokenization...")
        config_dict['complement_map'] = get_complement_map()

    print(f"  d_model: {config_dict.get('d_model', 'not specified')}")
    print(f"  n_layer: {config_dict.get('n_layer', 'not specified')}")
    print(f"  rcps: {config_dict.get('rcps', 'not specified')}")

    config = CaduceusConfig(**config_dict)
    model = CaduceusForInference(config, d_output=d_output, conjoin_test=conjoin_test)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Map state dict keys to match our model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # model.caduceus.* -> model.caduceus.*
        if key.startswith('model.caduceus.'):
            new_key = key[6:]  # Remove 'model.' prefix -> caduceus.*
            new_key = 'model.' + new_key  # Add back as model.caduceus.*
            new_state_dict[new_key] = value
            continue

        # decoder.0.output_transform.* -> decoder.output_transform.*
        if key.startswith('decoder.0.output_transform.'):
            new_key = 'decoder.output_transform.' + key[27:]
            new_state_dict[new_key] = value
            continue

        # Skip other keys
        if 'decoder' in key or key.startswith('model.'):
            continue

    # Debug: print what we're loading
    print(f"  Mapped {len(new_state_dict)} keys from checkpoint")

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    return model


def get_tokenizer():
    """Get the character tokenizer for DNA sequences."""
    tokenizer = AutoTokenizer.from_pretrained(
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        trust_remote_code=True,
    )
    return tokenizer


def string_reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def run_inference(
    model: CaduceusForInference,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    conjoin_test: bool,
    device: torch.device,
) -> tuple:
    """
    Run inference on sequences.
    """
    model.eval()
    all_probs = []
    all_preds = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Running inference"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]

        # For conjoin_test with non-RCPS models, stack forward and RC
        if conjoin_test and not model.config.rcps:
            batch_seqs_rc = [string_reverse_complement(s) for s in batch_seqs]
            inputs_rc = tokenizer(
                batch_seqs_rc,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = torch.stack([input_ids, inputs_rc["input_ids"]], dim=-1)

        input_ids = input_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)

    probs_array = np.vstack(all_probs)
    preds_array = np.array(all_preds)

    return probs_array, preds_array


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values (both short and long names)
    metrics["TP"] = int(tp)
    metrics["TN"] = int(tn)
    metrics["FP"] = int(fp)
    metrics["FN"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Caduceus Inference")
    print("=" * 60)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Load model and tokenizer
    model = load_model(
        args.config_path,
        args.checkpoint_path,
        args.d_output,
        args.conjoin_test,
        device,
    )
    tokenizer = get_tokenizer()

    # Run inference
    sequences = df["sequence"].tolist()
    probs, preds = run_inference(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, args.conjoin_test, device,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"\nApplying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe (exclude sequence column to save space)
    output_df = df.drop(columns=['sequence']).copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)

        metrics["checkpoint_path"] = args.checkpoint_path
        metrics["config_path"] = args.config_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["conjoin_test"] = args.conjoin_test
        metrics["num_samples"] = len(df)

        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

    elif has_labels:
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
