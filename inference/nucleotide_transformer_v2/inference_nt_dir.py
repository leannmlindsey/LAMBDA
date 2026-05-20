#!/usr/bin/env python3
"""
Directory-based Inference Script for Nucleotide Transformer v2

This script performs inference on ALL CSV files in a directory using a fine-tuned
NT-v2 model. The model is loaded ONCE and reused for all files, making it much
faster than running separate jobs for each file.

Input:
    - Directory containing CSV files with 'sequence' column (and optionally 'label')

Output:
    - For each input CSV, creates {basename}_predictions.csv in output directory
    - Optionally saves metrics JSON for each file (if labels present and --save_metrics)

Usage:
    python inference_nt_dir.py \
        --input_dir /path/to/csv_directory \
        --output_dir /path/to/output_directory \
        --model_path /path/to/finetuned/model \
        --fp16
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from safetensors.torch import load_file as load_safetensors


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on all CSV files in a directory with fine-tuned NT-v2 model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing CSV files with 'sequence' column",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction CSV files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
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
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input files (default: *.csv)",
    )

    # Optimization flags
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (recommended for A100 GPUs)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision (use if bf16 not supported)",
    )
    return parser.parse_args()


def run_inference(
    model,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    amp_dtype: torch.dtype = None,
    desc: str = "Running inference",
) -> tuple:
    """
    Run inference on sequences.

    Returns:
        Tuple of (probabilities array shape (n, 2), predictions array)
    """
    model.eval()
    all_probs = []
    all_preds = []

    use_amp = amp_dtype is not None and device.type == "cuda"

    for i in tqdm(range(0, len(sequences), batch_size), desc=desc):
        batch_seqs = sequences[i:i + batch_size]

        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(**inputs)
                    logits = outputs.logits
            else:
                outputs = model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
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
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Positive Rate
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0  # False Negative Rate
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def load_model(model_path: str, device: torch.device):
    """Load model and tokenizer."""
    base_model = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

    print(f"  Loading tokenizer from base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    checkpoint_has_model_code = os.path.exists(os.path.join(model_path, "modeling_esm.py"))

    if checkpoint_has_model_code:
        print(f"  Loading model directly from checkpoint")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
    else:
        print(f"  Loading model architecture from base model: {base_model}")
        print(f"  Fine-tuned checkpoint: {model_path}")

        # NOTE: The following from_pretrained() call will print a warning about:
        #   - "lm_head.*" weights not being used (expected - base model has LM head, not classifier)
        #   - "classifier.*" weights being newly initialized (expected - will be overwritten below)
        # This warning is expected and can be safely ignored.
        print("  (Note: The following HuggingFace warning about 'newly initialized' weights is expected)")

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            trust_remote_code=True,
            num_labels=2,
        )

        # Load the fine-tuned weights from checkpoint (overwrites the randomly initialized classifier)
        checkpoint_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(checkpoint_path):
            state_dict = load_safetensors(checkpoint_path)
            model.load_state_dict(state_dict)
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
            print(f"  Loaded fine-tuned weights from: {checkpoint_path}")
            print(f"  Classifier head weights loaded: {classifier_keys}")
        else:
            checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict)
                classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
                print(f"  Loaded fine-tuned weights from: {checkpoint_path}")
                print(f"  Classifier head weights loaded: {classifier_keys}")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {model_path}. "
                    "Expected 'model.safetensors' or 'pytorch_model.bin'"
                )

    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def process_single_file(
    csv_path: str,
    output_dir: str,
    model,
    tokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    args,
) -> Dict:
    """Process a single CSV file and return results."""
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(output_dir, f"{basename}_predictions.csv")

    # Load CSV
    df = pd.read_csv(csv_path)

    if "sequence" not in df.columns:
        print(f"  WARNING: Skipping {basename} - no 'sequence' column")
        return None

    has_labels = "label" in df.columns
    sequences = df["sequence"].tolist()

    # Run inference
    file_start = time.time()
    probs, preds = run_inference(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, device,
        amp_dtype=amp_dtype,
        desc=f"  {basename}",
    )
    file_elapsed = time.time() - file_start

    # Apply threshold
    if args.threshold != 0.5:
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Save predictions
    output_df.to_csv(output_csv, index=False)

    result = {
        "file": basename,
        "samples": len(df),
        "time_seconds": file_elapsed,
        "throughput": len(df) / file_elapsed,
        "output_csv": output_csv,
    }

    # Calculate metrics if labels present
    if has_labels:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)
        result["metrics"] = metrics

        # Save metrics JSON if requested
        if args.save_metrics:
            metrics_path = os.path.join(output_dir, f"{basename}_metrics.json")
            metrics_to_save = metrics.copy()
            metrics_to_save["file"] = basename
            metrics_to_save["samples"] = len(df)
            with open(metrics_path, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            result["metrics_path"] = metrics_path

    return result


def main():
    """Main function."""
    args = parse_arguments()

    print("\n" + "=" * 70)
    print("Nucleotide Transformer v2 - Directory Inference")
    print("=" * 70)

    total_start = time.time()

    # Validate directories
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all CSV files
    csv_pattern = os.path.join(args.input_dir, args.pattern)
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        raise ValueError(f"No files matching '{args.pattern}' found in {args.input_dir}")

    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files to process: {len(csv_files)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model ONCE
    print(f"\nLoading model from: {args.model_path}")
    model_start = time.time()
    model, tokenizer = load_model(args.model_path, device)
    model_load_time = time.time() - model_start
    print(f"Model loaded in {model_load_time:.1f} seconds")

    # Determine mixed precision dtype
    if args.bf16 and args.fp16:
        print("WARNING: Both --bf16 and --fp16 specified, using bf16")
        args.fp16 = False

    amp_dtype = None
    if args.bf16:
        print("Using bfloat16 mixed precision")
        amp_dtype = torch.bfloat16
    elif args.fp16:
        print("Using float16 mixed precision")
        amp_dtype = torch.float16

    # Process settings
    print(f"\nInference settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Precision: {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")

    # Process all files
    print("\n" + "=" * 70)
    print("Processing files...")
    print("=" * 70)

    results = []
    total_samples = 0
    total_inference_time = 0

    for i, csv_path in enumerate(csv_files, 1):
        basename = os.path.basename(csv_path)
        print(f"\n[{i}/{len(csv_files)}] {basename}")

        result = process_single_file(
            csv_path, args.output_dir,
            model, tokenizer, device, amp_dtype, args
        )

        if result:
            results.append(result)
            total_samples += result["samples"]
            total_inference_time += result["time_seconds"]

            # Print per-file summary
            print(f"    Samples: {result['samples']}, "
                  f"Time: {result['time_seconds']:.1f}s, "
                  f"Throughput: {result['throughput']:.1f} seq/s")

            # Print metrics if available
            if "metrics" in result:
                m = result["metrics"]
                print(f"    Acc: {m['accuracy']:.4f}, "
                      f"Prec: {m['precision']:.4f}, "
                      f"Rec: {m['recall']:.4f}, "
                      f"MCC: {m['mcc']:.4f}, "
                      f"F1: {m['f1']:.4f}, "
                      f"FPR: {m['fpr']:.4f}, "
                      f"FNR: {m['fnr']:.4f}")

    # Summary
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(results)}/{len(csv_files)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Model load time: {model_load_time:.1f}s")
    print(f"Total inference time: {total_inference_time:.1f}s")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print(f"Overall throughput: {total_samples / total_inference_time:.1f} seq/s")

    # Average time per file
    if len(results) > 0:
        avg_time_per_file = total_inference_time / len(results)
        avg_samples_per_file = total_samples / len(results)
        print(f"Average time per file: {avg_time_per_file:.2f}s")
        print(f"Average samples per file: {avg_samples_per_file:.0f}")

    # Print average metrics if available
    if any("metrics" in r for r in results):
        metrics_results = [r["metrics"] for r in results if "metrics" in r]
        print(f"\nMean metrics across {len(metrics_results)} files:")
        print(f"  Acc: {np.mean([m['accuracy'] for m in metrics_results]):.4f}, "
              f"Prec: {np.mean([m['precision'] for m in metrics_results]):.4f}, "
              f"Rec: {np.mean([m['recall'] for m in metrics_results]):.4f}, "
              f"MCC: {np.mean([m['mcc'] for m in metrics_results]):.4f}, "
              f"F1: {np.mean([m['f1'] for m in metrics_results]):.4f}, "
              f"FPR: {np.mean([m['fpr'] for m in metrics_results]):.4f}, "
              f"FNR: {np.mean([m['fnr'] for m in metrics_results]):.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "files_processed": len(results),
        "total_samples": total_samples,
        "model_load_time_s": model_load_time,
        "total_inference_time_s": total_inference_time,
        "total_wall_time_s": total_elapsed,
        "overall_throughput": total_samples / total_inference_time if total_inference_time > 0 else 0,
        "avg_time_per_file_s": total_inference_time / len(results) if len(results) > 0 else 0,
        "avg_samples_per_file": total_samples / len(results) if len(results) > 0 else 0,
        "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
        "batch_size": args.batch_size,
        "results": results,
    }

    # Add average metrics if available
    if any("metrics" in r for r in results):
        metrics_results = [r["metrics"] for r in results if "metrics" in r]
        summary["mean_metrics"] = {
            "accuracy": float(np.mean([m["accuracy"] for m in metrics_results])),
            "precision": float(np.mean([m["precision"] for m in metrics_results])),
            "recall": float(np.mean([m["recall"] for m in metrics_results])),
            "mcc": float(np.mean([m["mcc"] for m in metrics_results])),
            "f1": float(np.mean([m["f1"] for m in metrics_results])),
            "fpr": float(np.mean([m["fpr"] for m in metrics_results])),
            "fnr": float(np.mean([m["fnr"] for m in metrics_results])),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Print GPU memory if available
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")

    print("=" * 70)


if __name__ == "__main__":
    main()
