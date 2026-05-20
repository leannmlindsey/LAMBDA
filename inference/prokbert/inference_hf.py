#!/usr/bin/env python3
"""
Inference script using a HuggingFace-hosted ProkBERT model.
Instead of loading from a local checkpoint, this script loads a fine-tuned
model directly from the HuggingFace Hub.

Usage:
    python inference_hf.py --model_name neuralbioinfo/prokbert-mini-c-phage --dataset_file data/test.csv

Examples:
    # Using a local CSV file
    python inference_hf.py --model_name neuralbioinfo/prokbert-mini-c-phage --dataset_file data/test.csv

    # Using a HuggingFace dataset
    python inference_hf.py --model_name neuralbioinfo/prokbert-mini-c-phage --dataset leannmlindsey/lambda --split test

    # Prediction-only mode (no labels)
    python inference_hf.py --model_name neuralbioinfo/prokbert-mini-c-phage --dataset_file data/sequences.csv --no_labels
"""

import argparse
import json
import os
import re
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

from transformers import MegatronBertForSequenceClassification
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.training_utils import get_torch_data_from_segmentdb_classification
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT
from torch.utils.data import DataLoader


# Map known base model names to their tokenization parameters.
# Fine-tuned variants (e.g. prokbert-mini-c-phage) are matched by checking
# whether the model name starts with a known base name.
BASE_MODEL_TOKENIZATION_PARAMS = {
    'prokbert-mini-long': {'kmer': 6, 'shift': 2},
    'prokbert-mini-c':    {'kmer': 1, 'shift': 1},
    'prokbert-mini':      {'kmer': 6, 'shift': 1},
}


def infer_tokenization_params(model_name):
    """
    Infer kmer and shift tokenization parameters from a HuggingFace model name.

    Checks the normalized model name against known base models. Fine-tuned
    variants like 'prokbert-mini-c-phage' match their base 'prokbert-mini-c'.

    Falls back to parsing a 'kNsM' pattern from the name.

    Args:
        model_name: HuggingFace model identifier (e.g. 'neuralbioinfo/prokbert-mini-c-phage')

    Returns:
        dict: {'kmer': int, 'shift': int}

    Raises:
        ValueError: If tokenization parameters cannot be determined.
    """
    normalized = model_name.replace('neuralbioinfo/', '')

    # Check longest base names first to avoid 'prokbert-mini' matching before 'prokbert-mini-c'
    for base_name, params in BASE_MODEL_TOKENIZATION_PARAMS.items():
        if normalized.startswith(base_name):
            return params

    # Try regex pattern like k6s1
    match = re.search(r'k(\d+)s(\d+)', normalized)
    if match:
        kmer, shift = map(int, match.groups())
        return {'kmer': kmer, 'shift': shift}

    raise ValueError(
        f"Cannot infer tokenization parameters from model name '{model_name}'. "
        f"Use --kmer and --shift to specify them explicitly."
    )


def prepare_dataframe_from_dataset(dataset_split, max_length=1024, preserve_metadata=True):
    """
    Convert dataset split to the format expected by ProkBERT.

    Args:
        dataset_split: HuggingFace dataset split or pandas DataFrame
        max_length: Maximum sequence length
        preserve_metadata: Whether to preserve additional metadata columns

    Returns:
        pd.DataFrame: Prepared dataframe with required columns
    """
    if hasattr(dataset_split, 'to_pandas'):
        df = dataset_split.to_pandas()
    else:
        df = dataset_split.copy()

    # Store original metadata columns if present
    if preserve_metadata:
        core_cols = {'sequence', 'label', 'segment', 'segment_id', 'y'}
        metadata_cols = [col for col in df.columns if col not in core_cols]
        if metadata_cols:
            print(f"  Preserving metadata columns: {metadata_cols}")

    # Truncate sequences that are too long
    df['sequence'] = df['sequence'].apply(lambda x: x[:max_length] if len(x) > max_length else x)

    # Rename 'sequence' to 'segment' if needed
    if 'sequence' in df.columns and 'segment' not in df.columns:
        df['segment'] = df['sequence']

    # Create segment_id as a unique identifier per row.
    # IMPORTANT: segment_ids MUST be unique because the tokenization pipeline
    # stores results in a dict keyed by segment_id — duplicate IDs cause
    # sequences to be silently overwritten.
    if 'segment_id' not in df.columns:
        df['segment_id'] = [f"seg_{i}" for i in range(len(df))]

    # Create 'y' column (same as label for binary classification)
    if 'label' in df.columns and 'y' not in df.columns:
        df['y'] = df['label']

    # Print truncation statistics
    original_lengths = df['sequence'].apply(len) if 'sequence' in df.columns else df['segment'].apply(len)
    truncated_count = (original_lengths > max_length).sum()
    if truncated_count > 0:
        print(f"  Truncated {truncated_count} sequences from max length {original_lengths.max()} to {max_length}")

    return df


def load_inference_dataset(args):
    """
    Load dataset for inference from either HuggingFace or local file.

    Args:
        args: Argument namespace with dataset parameters

    Returns:
        pd.DataFrame: Prepared dataset
    """
    if args.dataset_file:
        print(f"Loading dataset from local file: {args.dataset_file}")
        if args.dataset_file.endswith('.csv'):
            df = pd.read_csv(args.dataset_file)
        elif args.dataset_file.endswith('.tsv'):
            df = pd.read_csv(args.dataset_file, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {args.dataset_file}")

        if 'sequence' not in df.columns:
            raise ValueError("Dataset must have a 'sequence' column")
        if 'label' not in df.columns and not args.no_labels:
            print("Warning: No 'label' column found. Running in prediction-only mode.")
            args.no_labels = True
            df['label'] = 0  # Dummy labels

        return prepare_dataframe_from_dataset(df, max_length=args.max_length)

    else:
        print(f"Loading dataset from HuggingFace: {args.dataset}")
        dataset = load_dataset(args.dataset)

        if args.split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{args.split}' not found. Available splits: {available_splits}")

        dataset_split = dataset[args.split]
        return prepare_dataframe_from_dataset(dataset_split, max_length=args.max_length)


def perform_inference(model, dataloader, device, show_progress=True):
    """
    Perform inference on the dataset.

    Args:
        model: Fine-tuned model
        dataloader: DataLoader for inference
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        tuple: (predictions, probabilities)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []

    iterator = tqdm(dataloader, desc="Running inference") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        dict: Dictionary of metrics (all values are JSON-serializable)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def save_results(predictions, probabilities, labels, segment_ids, output_path, metadata_df=None):
    """
    Save inference results to file.

    Args:
        predictions: Predicted labels
        probabilities: Prediction probabilities
        labels: True labels (if available)
        segment_ids: Sequence identifiers
        output_path: Path to save results
        metadata_df: Optional dataframe with metadata columns to preserve
    """
    results = pd.DataFrame({
        'segment_id': segment_ids,
    })

    if labels is not None:
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        results['label'] = labels

    results['prob_class_0'] = probabilities[:, 0]
    results['prob_1'] = probabilities[:, 1]
    results['pred_label'] = predictions

    if labels is not None:
        results['correct'] = (predictions == labels).astype(int)

    if metadata_df is not None and not metadata_df.empty:
        results['segment_id'] = results['segment_id'].astype(str)
        if 'segment_id' in metadata_df.columns:
            metadata_df['segment_id'] = metadata_df['segment_id'].astype(str)
            results = pd.merge(results, metadata_df, on='segment_id', how='left')
        elif 'seq_id' in metadata_df.columns:
            metadata_df['segment_id'] = metadata_df['seq_id'].astype(str)
            metadata_cols = [col for col in metadata_df.columns if col not in results.columns or col == 'segment_id']
            results = pd.merge(results, metadata_df[metadata_cols], on='segment_id', how='left')

    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference using a HuggingFace-hosted ProkBERT model'
    )

    # Model arguments
    parser.add_argument('--model_name', type=str, default='neuralbioinfo/prokbert-mini-c-phage',
                        help='HuggingFace model name (default: neuralbioinfo/prokbert-mini-c-phage)')

    # Tokenizer arguments (override auto-detection)
    parser.add_argument('--kmer', type=int, default=None,
                        help='K-mer size for tokenizer (auto-detected from model name if not specified)')
    parser.add_argument('--shift', type=int, default=None,
                        help='Shift size for tokenizer (auto-detected from model name if not specified)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace dataset name or path')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='Path to local dataset file (CSV or TSV format). Overrides --dataset if provided')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (default: test)')
    parser.add_argument('--no_labels', action='store_true',
                        help='Run inference without labels (prediction only mode)')
    parser.add_argument('--save_metrics', action='store_true',
                        help='If labels are present, calculate and save metrics to JSON')

    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length (default: 1024)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results (default: inference_results)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output filename (default: auto-generated based on dataset and model)')

    args = parser.parse_args()

    # Validate that at least one data source is provided
    if args.dataset is None and args.dataset_file is None:
        parser.error("Either --dataset or --dataset_file must be provided")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine tokenization parameters
    if args.kmer is not None and args.shift is not None:
        tokenization_params = {'kmer': args.kmer, 'shift': args.shift}
    else:
        tokenization_params = infer_tokenization_params(args.model_name)

    print("=" * 60)
    print("PROKBERT HUGGINGFACE INFERENCE")
    print("=" * 60)
    print(f"Model:       {args.model_name}")
    print(f"Tokenizer:   kmer={tokenization_params['kmer']}, shift={tokenization_params['shift']}")
    print(f"Dataset:     {args.dataset_file if args.dataset_file else args.dataset}")
    print(f"Device:      {device}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Max length:  {args.max_length}")
    print("=" * 60)

    start_time = time.time()

    # Load tokenizer
    print(f"\n1. Creating tokenizer (kmer={tokenization_params['kmer']}, shift={tokenization_params['shift']})...")
    tokenizer = ProkBERTTokenizer(
        tokenization_params=tokenization_params,
        operation_space='sequence'
    )

    # Load model from HuggingFace
    print(f"\n2. Loading model from HuggingFace: {args.model_name}...")
    try:
        model = MegatronBertForSequenceClassification.from_pretrained(
            args.model_name, trust_remote_code=True
        )

        # Validate max_length against model's max_position_embeddings
        max_pos = model.config.max_position_embeddings
        if args.max_length > max_pos:
            print(f"   WARNING: --max_length ({args.max_length}) exceeds model's "
                  f"max_position_embeddings ({max_pos}). Clamping to {max_pos}.")
            args.max_length = max_pos
        print(f"   Model max_position_embeddings: {max_pos}")
        print(f"   Using max_length: {args.max_length}")
        print(f"   Number of labels: {model.config.num_labels}")

        model = model.to(device)
        model.eval()
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load dataset
    print(f"\n3. Loading dataset...")
    test_df = load_inference_dataset(args)
    print(f"   Loaded {len(test_df)} sequences")

    # Store metadata columns for later
    core_cols = {'sequence', 'label', 'segment', 'segment_id', 'y'}
    metadata_cols = [col for col in test_df.columns if col not in core_cols]
    metadata_df = test_df[metadata_cols + ['segment_id']] if metadata_cols else pd.DataFrame()

    # Prepare data for ProkBERT
    print(f"\n4. Preparing data for inference...")

    # Use model's max_position_embeddings to set token limit (minus 2 for CLS/SEP)
    # The default token_limit (4096) can exceed the model's position embeddings (1024),
    # causing CUDA out-of-bounds errors.
    max_tokens = model.config.max_position_embeddings - 2
    print(f"   Max tokens per sequence: {max_tokens} (max_position_embeddings={model.config.max_position_embeddings})")

    [X_test, y_test, torchdb_test] = get_torch_data_from_segmentdb_classification(
        tokenizer, test_df, L=max_tokens, randomize=False
    )

    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    print(f"   torchdb_test size: {len(torchdb_test)}")

    # Validate token IDs against model vocabulary size
    max_token_id = int(X_test.max())
    vocab_size = model.config.vocab_size
    print(f"   Model vocab_size: {vocab_size}")
    print(f"   Max token ID in data: {max_token_id}")
    if max_token_id >= vocab_size:
        print(f"\n   ERROR: Token ID {max_token_id} exceeds model vocab_size {vocab_size}!")
        print(f"   This means the tokenizer (kmer={tokenization_params['kmer']}, shift={tokenization_params['shift']}) "
              f"does not match the model.")
        sys.exit(1)

    # Ensure X and y have the same first dimension
    if X_test.shape[0] != y_test.shape[0]:
        print(f"   WARNING: Size mismatch! X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}")
        min_size = min(X_test.shape[0], y_test.shape[0])
        X_test = X_test[:min_size]
        y_test = y_test[:min_size]
        print(f"   Truncated to size: {min_size}")

    test_ds = ProkBERTTrainingDatasetPT(X_test, y_test, AddAttentionMask=True)
    print(f"   Dataset size: {len(test_ds)}")

    test_dataloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Perform inference
    print(f"\n5. Running inference on {len(test_ds)} sequences...")
    predictions, probabilities = perform_inference(model, test_dataloader, device)

    # Calculate metrics if labels are available
    if not args.no_labels:
        print(f"\n6. Calculating metrics...")
        metrics = calculate_metrics(y_test, predictions, probabilities)

        print("\n" + "=" * 60)
        print("METRICS")
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

        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              0     1")
        print(f"Actual 0  {metrics['true_negatives']:5d} {metrics['false_positives']:5d}")
        print(f"       1  {metrics['false_negatives']:5d} {metrics['true_positives']:5d}")

    # Save results
    model_short = os.path.basename(args.model_name.rstrip('/'))
    args.output_dir = os.path.join(args.output_dir, model_short)
    print(f"\n7. Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_file:
        output_path = os.path.join(args.output_dir, args.output_file)
    else:
        dataset_name = args.dataset.replace('/', '_') if args.dataset else os.path.basename(args.dataset_file).split('.')[0]
        output_path = os.path.join(args.output_dir, f"predictions_{dataset_name}_{model_short}.csv")

    # Use segment_ids from torchdb_test which matches the processed data
    if 'segment_id' in torchdb_test.columns:
        segment_ids = torchdb_test['segment_id'].values[:len(predictions)]
    elif 'segment_id' in test_df.columns:
        segment_ids = test_df['segment_id'].values[:len(predictions)]
    else:
        segment_ids = [f"seq_{i}" for i in range(len(predictions))]

    # Ensure metadata_df is aligned with the predictions
    if not metadata_df.empty and 'segment_id' in metadata_df.columns:
        metadata_df = metadata_df[metadata_df['segment_id'].isin(segment_ids)]

    save_results(predictions, probabilities, y_test[:len(predictions)] if not args.no_labels else None,
                 segment_ids, output_path, metadata_df)

    # Save metrics to JSON if requested
    if not args.no_labels and args.save_metrics:
        metrics["model_name"] = args.model_name
        metrics["input_file"] = args.dataset_file if args.dataset_file else args.dataset
        metrics["num_samples"] = len(predictions)
        metrics["batch_size"] = args.batch_size
        metrics["max_length"] = args.max_length
        metrics["kmer"] = tokenization_params['kmer']
        metrics["shift"] = tokenization_params['shift']

        metrics_path = output_path.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

    # Print timing info
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Throughput: {len(predictions) / elapsed:.1f} sequences/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
