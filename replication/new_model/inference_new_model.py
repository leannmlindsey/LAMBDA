#!/usr/bin/env python3
"""
inference_new_model.py

Generic inference template for evaluating a new genomic language model on
LAMBDA. Pairs with finetune_new_model.py — once you've fine-tuned, use this
script to produce predictions on the LAMBDA test splits (or your own data)
in the exact CSV schema consumed by ../03_build_website_data.py.

Works for any HuggingFace `AutoModelForSequenceClassification`-compatible
checkpoint.

INPUT:
    --input_csv: A CSV with at least a `sequence` column. If a `label` column
                 is present, metrics will be computed and saved to a JSON.
                 Other columns (segment_id, seq_id, start, end) are passed
                 through to the output if present.

OUTPUT:
    --output_csv: A CSV with the same columns as input plus prob_0, prob_1,
                  and pred_label. Filename convention for use with the
                  aggregator:  <input_basename>_predictions.csv

Usage:
    python inference_new_model.py \\
        --model_path /path/to/checkpoints/gena_lm/2k \\
        --input_csv /path/to/LAMBDA/binary_segments_2k/test.csv \\
        --output_csv /path/to/results/gena_lm/binary/2k/test_predictions.csv \\
        --max_length 512 \\
        --batch_size 32 \\
        --bf16 \\
        --save_metrics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a fine-tuned HF gLM.")
    p.add_argument("--model_path", required=True, help="Path to fine-tuned checkpoint directory")
    p.add_argument("--input_csv", required=True, help="Input CSV with a 'sequence' column")
    p.add_argument("--output_csv", required=True, help="Where to write the predictions CSV")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="prob_1 threshold for pred_label (default 0.5)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None, help="Force cuda or cpu (default: auto)")
    p.add_argument("--save_metrics", action="store_true",
                   help="If labels are present, save accuracy/F1/MCC/AUC etc. to a _metrics.json")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], tokenizer, max_length: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.sequences[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )


def collate(batch):
    return {
        k: torch.stack([b[k].squeeze(0) for b in batch]) for k in batch[0].keys()
    }


def main():
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    print(f"Loading model from {args.model_path} (device={device}, dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, torch_dtype=dtype,
    ).to(device).eval()

    print(f"Loading input from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if "sequence" not in df.columns:
        raise ValueError(f"{args.input_csv} must contain a 'sequence' column")

    dataset = SequenceDataset(df["sequence"].astype(str).tolist(), tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(**batch)
            all_logits.append(out.logits.float().cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    df["prob_0"] = probs[:, 0]
    df["prob_1"] = probs[:, 1]
    df["pred_label"] = (df["prob_1"] >= args.threshold).astype(int)

    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(df)} predictions)")

    if args.save_metrics and "label" in df.columns:
        labels = df["label"].astype(int).to_numpy()
        preds = df["pred_label"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics = {
            "n": int(len(labels)),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, preds)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
        try:
            metrics["auc"] = float(roc_auc_score(labels, df["prob_1"]))
        except ValueError:
            metrics["auc"] = None
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {metrics_path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
