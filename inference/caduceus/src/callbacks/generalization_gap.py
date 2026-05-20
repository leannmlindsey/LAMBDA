"""Early stopping based on train/val accuracy gap to prevent overfitting."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class GeneralizationGapStopping(Callback):
    """Stop training when the gap between train and val accuracy exceeds a threshold.

    This helps prevent overfitting by stopping when the model starts memorizing
    the training data rather than learning generalizable patterns.

    Args:
        max_gap: Maximum allowed gap between train and val accuracy (default: 0.03)
        train_metric: Name of training accuracy metric (default: "train/accuracy")
        val_metric: Name of validation accuracy metric (default: "val/accuracy")
        min_epochs: Minimum epochs before gap checking starts (default: 5)
        patience: Number of consecutive epochs gap can exceed threshold before stopping (default: 3)
        verbose: Whether to print messages (default: True)
    """

    def __init__(
        self,
        max_gap: float = 0.03,
        train_metric: str = "train/accuracy",
        val_metric: str = "val/accuracy",
        min_epochs: int = 5,
        patience: int = 3,
        verbose: bool = True,
    ):
        super().__init__()
        self.max_gap = max_gap
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.min_epochs = min_epochs
        self.patience = patience
        self.verbose = verbose
        self.gap_exceeded_count = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Skip if we haven't reached minimum epochs
        if trainer.current_epoch < self.min_epochs:
            return

        # Get metrics from the logged values
        logs = trainer.callback_metrics

        train_acc = logs.get(self.train_metric)
        val_acc = logs.get(self.val_metric)

        if train_acc is None or val_acc is None:
            if self.verbose:
                print(f"GeneralizationGapStopping: Could not find metrics "
                      f"'{self.train_metric}' and/or '{self.val_metric}'")
            return

        # Convert to float if tensor
        train_acc = float(train_acc)
        val_acc = float(val_acc)
        gap = train_acc - val_acc

        if self.verbose:
            print(f"Epoch {trainer.current_epoch}: train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}, gap={gap:.4f} (max={self.max_gap})")

        if gap > self.max_gap:
            self.gap_exceeded_count += 1
            if self.verbose:
                print(f"  Gap exceeded threshold! Count: {self.gap_exceeded_count}/{self.patience}")

            if self.gap_exceeded_count >= self.patience:
                if self.verbose:
                    print(f"\nStopping training: generalization gap ({gap:.4f}) "
                          f"exceeded {self.max_gap} for {self.patience} consecutive epochs.")
                trainer.should_stop = True
        else:
            # Reset counter if gap is acceptable
            self.gap_exceeded_count = 0
