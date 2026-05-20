#!/usr/bin/env python3
"""
finetune_new_model.py

Generic fine-tuning template for evaluating a new genomic language model on
LAMBDA. Works for any model compatible with HuggingFace's
`AutoModelForSequenceClassification` interface — i.e., most BERT-style and
related encoder models, including:

  - GENA-LM (AIRI-Institute/gena-lm-bert-base-t2t and variants)
  - DNABERT-2 (zhihan1996/DNABERT-2-117M)
  - Nucleotide Transformer v2 (InstaDeepAI/nucleotide-transformer-v2-*)
  - ProkBERT (neuralbioinfo/prokbert-*)
  - any other AutoModelForSequenceClassification-compatible HF model

For models that need custom training loops (e.g., Caduceus, Evo2, megaDNA),
follow the per-model training repository linked in inference/README.md
instead.

INPUT: A directory containing `train.csv`, `dev.csv`, and `test.csv`.
       Each CSV must have `sequence` and `label` columns (label = 0 or 1).
       The LAMBDA binary_segments_{2k,4k,8k} directories on Zenodo follow
       this layout.

OUTPUT: A fine-tuned model checkpoint + test_results.json saved to --output_dir.

Usage:
    python finetune_new_model.py \\
        --model_name AIRI-Institute/gena-lm-bert-base-t2t \\
        --dataset_dir /path/to/LAMBDA/binary_segments_2k \\
        --output_dir /path/to/checkpoints/gena_lm/2k \\
        --max_length 512 \\
        --per_device_train_batch_size 16 \\
        --learning_rate 3e-5 \\
        --num_train_epochs 3 \\
        --bf16 \\
        --early_stopping_patience 3

See run_finetune.slurm in this directory for a SLURM submission template.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune any HuggingFace AutoModelForSequenceClassification-compatible "
                    "genomic language model on a LAMBDA train/dev/test split."
    )
    p.add_argument("--model_name", required=True,
                   help="HuggingFace model identifier or local path "
                        "(e.g., AIRI-Institute/gena-lm-bert-base-t2t)")
    p.add_argument("--dataset_dir", required=True,
                   help="Directory containing train.csv / dev.csv / test.csv")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the fine-tuned checkpoint + test_results.json")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max tokens per sequence (default 512). Set to match your model's "
                        "context window: GENA-LM = 512, NT-v2 = 2048, ProkBERT-mini = 1024.")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (V100/older).")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (A100/H100).")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade compute for memory (needed for 4k/8k sequences).")
    p.add_argument("--eval_strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--early_stopping_patience", type=int, default=3,
                   help="Stop if no improvement in eval metric for N evaluations.")
    p.add_argument("--metric_for_best_model", default="eval_mcc")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Set if the HF model requires custom code (DNABERT-2 needs this; "
                        "GENA-LM does not).")
    return p.parse_args()


def load_csvs(dataset_dir: Path) -> DatasetDict:
    """Load train/dev/test CSVs, keeping only `sequence` and `label` columns."""
    splits = {}
    for split_name, file_candidates in [
        ("train", ["train.csv"]),
        ("validation", ["dev.csv", "val.csv", "validation.csv"]),
        ("test", ["test.csv"]),
    ]:
        path = None
        for candidate in file_candidates:
            if (dataset_dir / candidate).exists():
                path = dataset_dir / candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"Could not find {' / '.join(file_candidates)} in {dataset_dir}"
            )
        df = pd.read_csv(path)[["sequence", "label"]]
        df["label"] = df["label"].astype(int)
        splits[split_name] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(splits)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, trust_remote_code=args.trust_remote_code
    )

    print(f"Loading data from: {args.dataset_dir}")
    datasets = load_csvs(Path(args.dataset_dir))
    print(f"  train: {len(datasets['train'])}")
    print(f"  validation: {len(datasets['validation'])}")
    print(f"  test: {len(datasets['test'])}")

    def tokenize(batch):
        return tokenizer(
            batch["sequence"], truncation=True, max_length=args.max_length, padding=False
        )
    datasets = datasets.map(tokenize, batched=True, remove_columns=["sequence"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        logging_steps=100,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to="none",
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")

    # Save best model + tokenizer to the top of output_dir for easy inference
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "test_results.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nTest metrics saved to {output_dir / 'test_results.json'}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
