"""
Batch Inference Script for Caduceus

Loads the model once and runs inference on multiple CSV files sequentially.
This avoids the overhead of reloading the model for each file.

Usage:
    python -m src.batch_inference \
        --input_list /path/to/input_files.txt \
        --output_dir /path/to/output_directory \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --config_path /path/to/config.json
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.inference import (
    load_model,
    get_tokenizer,
    run_inference,
    calculate_metrics,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference on multiple CSV files with fine-tuned Caduceus model"
    )
    parser.add_argument(
        "--input_list",
        type=str,
        required=True,
        help="Path to text file with one input CSV path per line",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store all output files",
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


def main():
    """Main function to run batch inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Caduceus Batch Inference")
    print("=" * 60)

    total_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer ONCE
    print("\nLoading model and tokenizer...")
    model_start_time = time.time()
    model = load_model(
        args.config_path,
        args.checkpoint_path,
        args.d_output,
        args.conjoin_test,
        device,
    )
    tokenizer = get_tokenizer()
    model_elapsed = time.time() - model_start_time
    print(f"Model loaded in {model_elapsed:.2f} seconds")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input list
    with open(args.input_list, 'r') as f:
        input_files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    num_files = len(input_files)
    print(f"\nFound {num_files} input files to process")
    print("=" * 60)

    # Process each file
    total_sequences = 0
    files_processed = 0

    for i, input_csv in enumerate(input_files):
        if not os.path.isfile(input_csv):
            print(f"\nWARNING: Input file not found, skipping: {input_csv}")
            continue

        input_basename = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(args.output_dir, f"{input_basename}_predictions.csv")

        print(f"\n{'=' * 60}")
        print(f"Processing file {i + 1}/{num_files}: {input_basename}")
        print(f"  Input:  {input_csv}")
        print(f"  Output: {output_csv}")
        print(f"{'=' * 60}")

        file_start_time = time.time()

        # Load input CSV
        df = pd.read_csv(input_csv)

        if "sequence" not in df.columns:
            print(f"  ERROR: Input CSV must have a 'sequence' column, skipping: {input_csv}")
            continue

        has_labels = "label" in df.columns
        print(f"  Samples: {len(df)}")
        print(f"  Has labels: {has_labels}")

        # Run inference
        sequences = df["sequence"].tolist()
        probs, preds = run_inference(
            model, tokenizer, sequences,
            args.batch_size, args.max_length, args.conjoin_test, device,
        )

        # Apply custom threshold if specified
        if args.threshold != 0.5:
            preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
        else:
            preds_thresholded = preds

        # Create output dataframe (exclude sequence column to save space)
        output_df = df.drop(columns=['sequence']).copy()
        output_df["prob_0"] = probs[:, 0]
        output_df["prob_1"] = probs[:, 1]
        output_df["pred_label"] = preds_thresholded

        # Save predictions
        output_df.to_csv(output_csv, index=False)
        print(f"  Saved predictions to: {output_csv}")

        # Calculate and save metrics if labels present
        if has_labels and args.save_metrics:
            labels = df["label"].values
            metrics = calculate_metrics(labels, preds_thresholded, probs)

            metrics["checkpoint_path"] = args.checkpoint_path
            metrics["config_path"] = args.config_path
            metrics["input_csv"] = input_csv
            metrics["threshold"] = args.threshold
            metrics["conjoin_test"] = args.conjoin_test
            metrics["num_samples"] = len(df)

            metrics_path = output_csv.replace(".csv", "_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved metrics to: {metrics_path}")

            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  MCC:         {metrics['mcc']:.4f}")
            print(f"  AUC:         {metrics['auc']:.4f}")

        elif has_labels:
            from sklearn.metrics import accuracy_score
            labels = df["label"].values
            acc = accuracy_score(labels, preds_thresholded)
            print(f"  Accuracy: {acc:.4f}")

        file_elapsed = time.time() - file_start_time
        print(f"  Completed in {file_elapsed:.2f} seconds ({len(df) / file_elapsed:.1f} seq/s)")

        total_sequences += len(df)
        files_processed += 1

    total_elapsed = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print("Batch Inference Complete")
    print(f"{'=' * 60}")
    print(f"  Files processed: {files_processed}")
    print(f"  Total sequences: {total_sequences}")
    print(f"  Total time: {total_elapsed:.2f} seconds")
    if total_sequences > 0:
        print(f"  Overall throughput: {total_sequences / total_elapsed:.1f} seq/s")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
