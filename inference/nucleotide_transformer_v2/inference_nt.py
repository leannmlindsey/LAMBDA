#!/usr/bin/env python3
"""
Inference Script for Nucleotide Transformer v2

This script performs inference on a CSV file using a fine-tuned NT-v2 model.
It outputs predictions with probability scores for threshold analysis.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or thresholded)

Usage:
    python inference_nt.py \
        --input_csv /path/to/test.csv \
        --model_path /path/to/finetuned/model \
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
        description="Run inference on CSV file with fine-tuned NT-v2 model"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
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
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (NT-v2 supports up to 2048)",
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
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: uses model_path, or falls back to base model)",
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
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for potential speedup (first run will be slow due to compilation)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode: default (recommended), reduce-overhead (may fail with some models), max-autotune",
    )

    # Profiling flags
    parser.add_argument(
        "--profile_torch",
        action="store_true",
        help="Enable profiling with torch.profiler (no special permissions needed)",
    )
    parser.add_argument(
        "--profile_warmup",
        type=int,
        default=3,
        help="Number of warmup batches before profiling (default: 3)",
    )
    parser.add_argument(
        "--profile_batches",
        type=int,
        default=10,
        help="Number of batches to profile (default: 10)",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="./profile_traces",
        help="Directory to save profiling traces (default: ./profile_traces)",
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
) -> tuple:
    """
    Run inference on sequences.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        sequences: List of DNA sequences
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        amp_dtype: If set, use automatic mixed precision with this dtype (torch.float16 or torch.bfloat16)

    Returns:
        Tuple of (probabilities array shape (n, 2), predictions array)
    """
    model.eval()
    all_probs = []
    all_preds = []

    # Set up autocast context if using mixed precision
    use_amp = amp_dtype is not None and device.type == "cuda"

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Running inference"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
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

            # Apply softmax to get probabilities (in fp32 for numerical stability)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)

    probs_array = np.vstack(all_probs)
    preds_array = np.array(all_preds)

    return probs_array, preds_array


def run_profiled_inference(
    model,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    amp_dtype: torch.dtype = None,
    warmup_batches: int = 3,
    profile_batches: int = 10,
    output_dir: str = "./profile_traces",
) -> Dict:
    """
    Run profiled inference for performance analysis.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        sequences: List of DNA sequences
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        amp_dtype: If set, use automatic mixed precision with this dtype
        warmup_batches: Number of warmup batches before profiling
        profile_batches: Number of batches to profile
        output_dir: Directory to save profiling traces

    Returns:
        Dict with profiling results and statistics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"

    # Prepare batches
    all_batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        all_batches.append(inputs)

    total_batches_needed = warmup_batches + profile_batches
    if len(all_batches) < total_batches_needed:
        print(f"WARNING: Only {len(all_batches)} batches available, "
              f"need {total_batches_needed} for warmup + profiling")
        # Cycle through batches if needed
        while len(all_batches) < total_batches_needed:
            all_batches.extend(all_batches[:total_batches_needed - len(all_batches)])

    # Warmup runs (not profiled)
    print(f"\nRunning {warmup_batches} warmup batches...")
    with torch.no_grad():
        for i in range(warmup_batches):
            inputs = all_batches[i]
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(**inputs)
            else:
                _ = model(**inputs)
    torch.cuda.synchronize()

    # Profiled runs
    print(f"Profiling {profile_batches} batches...")

    # Determine precision string for filenames
    if amp_dtype == torch.float16:
        precision_str = "fp16"
    elif amp_dtype == torch.bfloat16:
        precision_str = "bf16"
    else:
        precision_str = "fp32"

    trace_filename = f"trace_bs{batch_size}_{precision_str}"
    chrome_trace_path = os.path.join(output_dir, f"{trace_filename}_chrome.json")

    # Configure profiler (without on_trace_ready to allow manual export)
    # Also measure wall-clock time for accurate throughput calculation
    torch.cuda.synchronize()
    profile_start_time = time.time()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for i in range(profile_batches):
                inputs = all_batches[warmup_batches + i]
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        outputs = model(**inputs)
                        logits = outputs.logits
                else:
                    outputs = model(**inputs)
                    logits = outputs.logits

                # Include softmax in profiling
                _ = torch.softmax(logits.float(), dim=-1)

                prof.step()

    torch.cuda.synchronize()
    profile_wall_time = time.time() - profile_start_time

    # Export Chrome trace
    prof.export_chrome_trace(chrome_trace_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print(f"Batch size: {batch_size}, Precision: {precision_str}")
    print(f"Profiled batches: {profile_batches}")
    print("=" * 80)

    # Summary sorted by CUDA time
    print("\nTop 20 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Summary sorted by CPU time
    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Memory summary
    print("\nTop 10 operations by CUDA memory:")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # Calculate aggregate statistics
    key_averages = prof.key_averages()

    # Helper to get CUDA time from an event (try multiple attribute names)
    def get_cuda_time(item):
        for attr in ['self_cuda_time_total', 'cuda_time_total', 'self_cuda_time', 'cuda_time']:
            val = getattr(item, attr, None)
            if val is not None and val > 0:
                return val
        return 0

    def get_cpu_time(item):
        for attr in ['self_cpu_time_total', 'cpu_time_total', 'self_cpu_time', 'cpu_time']:
            val = getattr(item, attr, None)
            if val is not None and val > 0:
                return val
        return 0

    total_cuda_time = sum(get_cuda_time(item) for item in key_averages)
    total_cpu_time = sum(get_cpu_time(item) for item in key_averages)

    # If CUDA time is still 0, estimate from CPU time (inference was clearly running)
    if total_cuda_time == 0 and total_cpu_time > 0:
        print("  Note: CUDA time not captured, using CPU time as estimate")
        total_cuda_time = total_cpu_time

    # Get FLOPS if available
    total_flops = sum(getattr(item, 'flops', 0) or 0 for item in key_averages)

    # Calculate model size and memory bandwidth estimation
    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if amp_dtype in [torch.float16, torch.bfloat16] else 4
    model_size_bytes = num_params * bytes_per_param
    model_size_gb = model_size_bytes / (1024**3)

    # Estimate memory traffic per batch (model weights + activations)
    # For inference: we load model weights once per forward pass
    # Activations depend on batch_size, seq_len, hidden_dim
    # This is a simplified estimate
    estimated_bytes_per_batch = model_size_bytes  # At minimum, load all weights

    # Use wall-clock time for accurate throughput/bandwidth calculations
    # (profiler CUDA time only captures kernel-specific time, not full execution)
    time_seconds = profile_wall_time
    total_bytes_transferred = estimated_bytes_per_batch * profile_batches

    # Memory bandwidth achieved (GB/s)
    achieved_bandwidth_gbs = (total_bytes_transferred / (1024**3)) / time_seconds if time_seconds > 0 else 0

    # A100 peak bandwidth for reference
    a100_peak_bandwidth_gbs = 2039  # GB/s for A100 80GB HBM2e
    bandwidth_utilization = (achieved_bandwidth_gbs / a100_peak_bandwidth_gbs) * 100 if a100_peak_bandwidth_gbs > 0 else 0

    # Arithmetic intensity (FLOPs / Bytes)
    arithmetic_intensity = total_flops / total_bytes_transferred if total_bytes_transferred > 0 else 0

    # Calculate throughput
    sequences_profiled = profile_batches * batch_size
    throughput = sequences_profiled / time_seconds if time_seconds > 0 else 0

    stats = {
        "batch_size": batch_size,
        "precision": precision_str,
        "profile_batches": profile_batches,
        "wall_time_seconds": profile_wall_time,
        "wall_time_ms": profile_wall_time * 1000,
        "avg_batch_time_ms": (profile_wall_time * 1000) / profile_batches,
        "profiler_cuda_time_ms": total_cuda_time / 1000,
        "profiler_cpu_time_ms": total_cpu_time / 1000,
        "total_flops": total_flops,
        "chrome_trace_path": chrome_trace_path,
        "model_params": num_params,
        "model_size_gb": model_size_gb,
        "bytes_per_param": bytes_per_param,
        "estimated_bytes_per_batch": estimated_bytes_per_batch,
        "achieved_bandwidth_gbs": achieved_bandwidth_gbs,
        "bandwidth_utilization_pct": bandwidth_utilization,
        "arithmetic_intensity": arithmetic_intensity,
        "throughput_seq_per_sec": throughput,
        "metric_sources": {
            "measured": [
                "wall_time_seconds",
                "wall_time_ms",
                "avg_batch_time_ms",
                "throughput_seq_per_sec",
                "total_flops (estimated by PyTorch from op shapes)"
            ],
            "estimated": [
                "achieved_bandwidth_gbs (assumes full model load per batch)",
                "bandwidth_utilization_pct (vs theoretical A100 peak)",
                "arithmetic_intensity (FLOPs / estimated bytes)",
                "compute_utilization_pct (vs theoretical A100 peak)"
            ],
            "not_available": [
                "L1/L2 cache hit/miss rates (requires ncu)",
                "actual HBM bandwidth (requires ncu)",
                "SM occupancy (requires ncu)",
                "warp efficiency (requires ncu)"
            ]
        },
    }

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"  Wall-clock time: {stats['wall_time_ms']:.2f} ms ({profile_batches} batches)")
    print(f"  Avg batch time: {stats['avg_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_seq_per_sec']:.1f} sequences/second")

    print("\n  --- Model Info ---")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Model size: {model_size_gb:.2f} GB ({precision_str})")

    print("\n  --- Memory Bandwidth Analysis ---")
    print(f"  Estimated bytes/batch: {estimated_bytes_per_batch / (1024**2):.1f} MB")
    print(f"  Achieved bandwidth: {achieved_bandwidth_gbs:.1f} GB/s")
    print(f"  A100 peak bandwidth: {a100_peak_bandwidth_gbs} GB/s")
    print(f"  Bandwidth utilization: {bandwidth_utilization:.1f}%")

    if total_flops > 0:
        tflops = total_flops / 1e12
        tflops_per_sec = tflops / time_seconds if time_seconds > 0 else 0
        print("\n  --- Compute Analysis ---")
        print(f"  Total TFLOPs: {tflops:.2f}")
        print(f"  Achieved TFLOPS/s: {tflops_per_sec:.2f}")
        # A100 peak compute
        a100_peak_tflops_fp16 = 312  # TFLOPS for fp16/bf16
        a100_peak_tflops_fp32 = 19.5  # TFLOPS for fp32
        peak_tflops = a100_peak_tflops_fp16 if amp_dtype else a100_peak_tflops_fp32
        compute_utilization = (tflops_per_sec / peak_tflops) * 100
        print(f"  A100 peak TFLOPS ({precision_str}): {peak_tflops}")
        print(f"  Compute utilization: {compute_utilization:.1f}%")
        stats["tflops"] = tflops
        stats["tflops_per_sec"] = tflops_per_sec
        stats["compute_utilization_pct"] = compute_utilization

        print("\n  --- Roofline Analysis ---")
        print(f"  Arithmetic intensity: {arithmetic_intensity:.2f} FLOPs/Byte")
        # Ridge point for A100: peak_compute / peak_bandwidth
        ridge_point = (peak_tflops * 1e12) / (a100_peak_bandwidth_gbs * 1e9)
        print(f"  A100 ridge point: {ridge_point:.1f} FLOPs/Byte")
        if arithmetic_intensity < ridge_point:
            print(f"  Status: MEMORY BOUND (below ridge point)")
        else:
            print(f"  Status: COMPUTE BOUND (above ridge point)")
        stats["ridge_point"] = ridge_point
        stats["is_memory_bound"] = arithmetic_intensity < ridge_point

    print("=" * 80)

    print("\n  --- Metric Sources ---")
    print("  MEASURED (torch.profiler):")
    print("    - Wall-clock time, throughput")
    print("    - Per-operation timing breakdown")
    print("    - Memory allocation")
    print("    - FLOPS (estimated by PyTorch based on operation shapes)")
    print("  ESTIMATED (calculated):")
    print("    - Memory bandwidth (assumes full model load per batch)")
    print("    - Bandwidth/compute utilization (vs theoretical peak)")
    print("    - Arithmetic intensity (FLOPs / estimated bytes)")
    print("  NOT AVAILABLE (requires ncu with GPU perf counter access):")
    print("    - L1/L2 cache hit/miss rates")
    print("    - Actual HBM bandwidth")
    print("    - SM occupancy and warp efficiency")
    print("=" * 80)

    print("\n  --- For Detailed Hardware Metrics (requires ncu) ---")
    print("  Run with NVIDIA Nsight Compute (requires GPU perf counter permissions):")
    print(f"    ncu --set full -o profile python inference_nt.py --input_csv ... --{precision_str}")
    print("  Key ncu metrics:")
    print("    - l2_tex_read_hit_rate: L2 cache hit rate")
    print("    - dram_read_throughput: Actual HBM bandwidth")
    print("    - sm_efficiency: Streaming multiprocessor utilization")
    print("    - achieved_occupancy: Warp occupancy")
    print("=" * 80)

    print(f"\nTraces saved to:")
    print(f"  Chrome trace: {chrome_trace_path}")
    print(f"\nTo view Chrome trace: Open chrome://tracing and load the JSON file")

    # Save stats to JSON
    stats_path = os.path.join(output_dir, f"{trace_filename}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")

    return stats


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

    # AUC
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Nucleotide Transformer v2 Inference")
    print("=" * 60)

    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Load model and tokenizer
    print(f"\nLoading fine-tuned weights from: {args.model_path}")

    # The base model provides the architecture and tokenizer
    # Fine-tuning only changes weights, not the tokenizer or model code
    base_model = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

    # Always load tokenizer from base model (fine-tuning doesn't change it)
    print(f"  Loading tokenizer from base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load model: use base model for architecture/code, checkpoint for weights
    # Check if checkpoint has the custom model files
    checkpoint_has_model_code = os.path.exists(os.path.join(args.model_path, "modeling_esm.py"))

    if checkpoint_has_model_code:
        # Checkpoint has everything, load directly
        print(f"  Loading model directly from checkpoint")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    else:
        # Checkpoint only has weights - load architecture from base, weights from checkpoint
        print(f"  Loading model architecture from base model: {base_model}")
        print(f"  Fine-tuned checkpoint: {args.model_path}")

        # NOTE: The following from_pretrained() call will print a warning about:
        #   - "lm_head.*" weights not being used (expected - base model has LM head, not classifier)
        #   - "classifier.*" weights being newly initialized (expected - will be overwritten below)
        # This warning is expected and can be safely ignored.
        print("  (Note: The following HuggingFace warning about 'newly initialized' weights is expected)")

        # Load the base model with classification head (num_labels=2 for binary classification)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            trust_remote_code=True,
            num_labels=2,
        )

        # Load the fine-tuned weights from checkpoint (overwrites the randomly initialized classifier)
        checkpoint_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(checkpoint_path):
            state_dict = load_safetensors(checkpoint_path)
            model.load_state_dict(state_dict)
            # Verify classifier weights were loaded
            classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
            print(f"  Loaded fine-tuned weights from: {checkpoint_path}")
            print(f"  Classifier head weights loaded: {classifier_keys}")
        else:
            # Try pytorch format
            checkpoint_path = os.path.join(args.model_path, "pytorch_model.bin")
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict)
                classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
                print(f"  Loaded fine-tuned weights from: {checkpoint_path}")
                print(f"  Classifier head weights loaded: {classifier_keys}")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {args.model_path}. "
                    "Expected 'model.safetensors' or 'pytorch_model.bin'"
                )

    model = model.to(device)

    # Determine mixed precision dtype
    if args.bf16 and args.fp16:
        print("WARNING: Both --bf16 and --fp16 specified, using bf16")
        args.fp16 = False

    amp_dtype = None
    if args.bf16:
        print("  Using bfloat16 mixed precision (autocast)")
        amp_dtype = torch.bfloat16
    elif args.fp16:
        print("  Using float16 mixed precision (autocast)")
        amp_dtype = torch.float16

    # Apply torch.compile() if requested
    if args.compile:
        print(f"  Compiling model with torch.compile(mode='{args.compile_mode}')")
        print("  Note: First inference will be slow due to compilation...")
        if args.compile_mode == "reduce-overhead":
            print("  WARNING: reduce-overhead mode may fail with models using rotary embeddings")
            print("           Use --compile_mode default if you encounter CUDA graph errors")
        # Use dynamic=False since sequence lengths are typically fixed
        # This avoids recompilation overhead for different shapes
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run inference (or profiled inference)
    sequences = df["sequence"].tolist()

    if args.profile_torch:
        # Run profiled inference mode with torch.profiler
        print("\n" + "=" * 60)
        print("PROFILING MODE (torch.profiler)")
        print("=" * 60)
        profile_stats = run_profiled_inference(
            model, tokenizer, sequences,
            args.batch_size, args.max_length, device,
            amp_dtype=amp_dtype,
            warmup_batches=args.profile_warmup,
            profile_batches=args.profile_batches,
            output_dir=args.profile_output,
        )
        # After profiling, still run full inference for predictions
        print("\nRunning full inference for predictions...")

    probs, preds = run_inference(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, device,
        amp_dtype=amp_dtype,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"\nApplying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
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

        # Add metadata
        metrics["model_path"] = args.model_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["num_samples"] = len(df)

        # Save metrics
        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Print metrics
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
        # Just print basic accuracy even if not saving
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    # Print timing and performance stats
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")

    # Print precision used
    if args.bf16:
        print(f"Precision: bfloat16")
    elif args.fp16:
        print(f"Precision: float16")
    else:
        print(f"Precision: float32 (default)")

    # Print compile status
    if args.compile:
        print(f"torch.compile: enabled (mode={args.compile_mode})")
    else:
        print(f"torch.compile: disabled")

    # Print GPU memory usage if available
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")


if __name__ == "__main__":
    main()
