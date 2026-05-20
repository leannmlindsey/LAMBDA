"""Callback to save test predictions and compute metrics using scikit-learn.

This callback collects all predictions during testing and computes comprehensive
metrics on the full test dataset at the end, avoiding issues with batch-level
metric averaging (especially for MCC).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


class TestResultsCallback(pl.Callback):
    """Callback to collect test predictions and compute metrics using scikit-learn.

    This avoids the issue of averaging metrics like MCC across batches, which can
    produce incorrect results when batches have imbalanced class distributions.
    """

    def __init__(
        self,
        output_dir: str = None,
        output_filename: str = "test_results.json",
        save_predictions: bool = True,
        predictions_filename: str = "test_predictions.npz",
    ):
        super().__init__()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.save_predictions = save_predictions
        self.predictions_filename = predictions_filename

        # Storage for predictions
        self._all_logits = []
        self._all_labels = []
        self._all_losses = []
        self._test_start_time = None
        self._num_samples = 0

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset storage at the start of testing."""
        self._all_logits = []
        self._all_labels = []
        self._all_losses = []
        self._test_start_time = time.time()
        self._num_samples = 0

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions from each test batch."""
        # Get the batch data
        x, y, *_ = batch

        # Forward pass to get logits
        with torch.no_grad():
            logits, labels, _ = pl_module.forward(batch)

        # Store logits and labels
        self._all_logits.append(logits.detach().cpu())
        self._all_labels.append(labels.detach().cpu())

        # Store loss if available
        if outputs is not None:
            if isinstance(outputs, torch.Tensor):
                self._all_losses.append(outputs.detach().cpu().item())
            elif isinstance(outputs, dict) and 'loss' in outputs:
                self._all_losses.append(outputs['loss'].detach().cpu().item())

        self._num_samples += labels.shape[0]

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute metrics on full test set and save results."""
        test_runtime = time.time() - self._test_start_time

        # Concatenate all predictions
        all_logits = torch.cat(self._all_logits, dim=0)
        all_labels = torch.cat(self._all_labels, dim=0)

        # Reshape if needed
        all_logits = all_logits.view(-1, all_logits.shape[-1])
        all_labels = all_labels.view(-1)

        # Get predictions and probabilities
        all_probs = F.softmax(all_logits, dim=-1).numpy()
        all_preds = np.argmax(all_probs, axis=-1)
        all_labels_np = all_labels.numpy()

        # For binary classification, get probability of positive class
        if all_probs.shape[1] == 2:
            prob_positive = all_probs[:, 1]
        else:
            prob_positive = all_probs  # For multiclass, use full prob matrix

        # Compute cross-entropy loss on full dataset
        loss_fn = torch.nn.CrossEntropyLoss()
        eval_loss = loss_fn(all_logits, all_labels).item()

        # Compute metrics using scikit-learn
        results = {}

        # Basic metrics
        results['eval_loss'] = eval_loss
        results['eval_accuracy'] = accuracy_score(all_labels_np, all_preds)

        # Precision, Recall, F1 (binary or weighted for multiclass)
        if len(np.unique(all_labels_np)) == 2:
            # Binary classification
            results['eval_precision'] = precision_score(all_labels_np, all_preds, zero_division=0)
            results['eval_recall'] = recall_score(all_labels_np, all_preds, zero_division=0)
            results['eval_f1'] = f1_score(all_labels_np, all_preds, zero_division=0)

            # MCC - computed on full dataset to avoid batch averaging issues
            results['eval_mcc'] = matthews_corrcoef(all_labels_np, all_preds)

            # Sensitivity = Recall (true positive rate)
            results['eval_sensitivity'] = results['eval_recall']

            # Specificity (true negative rate)
            tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds).ravel()
            results['eval_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # AUC
            try:
                results['eval_auc'] = roc_auc_score(all_labels_np, prob_positive)
            except ValueError:
                # If only one class present
                results['eval_auc'] = 0.0
        else:
            # Multiclass - use weighted average
            results['eval_precision'] = precision_score(all_labels_np, all_preds, average='weighted', zero_division=0)
            results['eval_recall'] = recall_score(all_labels_np, all_preds, average='weighted', zero_division=0)
            results['eval_f1'] = f1_score(all_labels_np, all_preds, average='weighted', zero_division=0)
            results['eval_mcc'] = matthews_corrcoef(all_labels_np, all_preds)
            results['eval_sensitivity'] = results['eval_recall']
            results['eval_specificity'] = None  # Not well-defined for multiclass
            try:
                results['eval_auc'] = roc_auc_score(all_labels_np, all_probs, multi_class='ovr', average='weighted')
            except ValueError:
                results['eval_auc'] = 0.0

        # Runtime metrics
        results['eval_runtime'] = round(test_runtime, 4)
        results['eval_samples_per_second'] = round(self._num_samples / test_runtime, 3)
        results['eval_steps_per_second'] = round(len(self._all_logits) / test_runtime, 3)

        # Epoch (from trainer)
        results['epoch'] = float(trainer.current_epoch)

        # Add paths for traceability
        # Checkpoint path - try to get the actual checkpoint used for testing
        checkpoint_path = None
        # First check if trainer has ckpt_path (set when trainer.test(ckpt_path=...) is called)
        if hasattr(trainer, 'ckpt_path') and trainer.ckpt_path is not None:
            checkpoint_path = str(trainer.ckpt_path)
        # Fall back to the pretrained_model_path from config
        elif hasattr(pl_module, 'hparams') and hasattr(pl_module.hparams, 'train'):
            if hasattr(pl_module.hparams.train, 'pretrained_model_path'):
                checkpoint_path = str(pl_module.hparams.train.pretrained_model_path)
        # Also try to get best model checkpoint path from callbacks
        if checkpoint_path is None:
            for callback in trainer.callbacks:
                if hasattr(callback, 'best_model_path') and callback.best_model_path:
                    checkpoint_path = str(callback.best_model_path)
                    break
        if checkpoint_path:
            results['checkpoint_path'] = checkpoint_path

        # Dataset paths
        if hasattr(pl_module, 'hparams') and hasattr(pl_module.hparams, 'dataset'):
            dataset_cfg = pl_module.hparams.dataset
            if hasattr(dataset_cfg, 'data_dir'):
                data_dir = Path(dataset_cfg.data_dir)
                results['train_data_path'] = str(data_dir / 'train.csv')
                results['dev_data_path'] = str(data_dir / 'dev.csv')
                results['test_data_path'] = str(data_dir / 'test.csv')

        # Determine output directory
        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
        elif trainer.log_dir is not None:
            output_dir = Path(trainer.log_dir)
        else:
            output_dir = Path('.')

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_path = output_dir / self.output_filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nTest results saved to: {results_path}")

        # Save predictions if requested
        if self.save_predictions:
            predictions_path = output_dir / self.predictions_filename
            np.savez(
                predictions_path,
                logits=all_logits.numpy(),
                probabilities=all_probs,
                predictions=all_preds,
                labels=all_labels_np,
            )
            print(f"Predictions saved to: {predictions_path}")

        # Print results summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        for key, value in results.items():
            if value is not None and not key.endswith('_path'):
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        print("="*60 + "\n")
