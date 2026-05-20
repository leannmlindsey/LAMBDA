"""
Embedding Analysis Script for Caduceus

This script:
1. Extracts embeddings from a trained/pretrained Caduceus model for sequences in CSV files
2. Trains a linear probe (logistic regression) classifier
3. Calculates silhouette score to measure embedding quality
4. Creates PCA visualization showing class separation
5. Trains a simple 3-layer neural network classifier

Usage:
    python -m src.embedding_analysis \
        --csv_dir /path/to/csv/data \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --config_path /path/to/config.json \
        --output_dir ./outputs/embedding_analysis
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Caduceus-specific imports
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import Caduceus
from transformers import AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings and perform embedding analysis for Caduceus"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Path to directory containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Caduceus checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Caduceus config JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/embedding_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "first", "last"],
        help="Pooling strategy for embeddings (mean, first, last)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--nn_epochs",
        type=int,
        default=100,
        help="Number of epochs for 3-layer NN training",
    )
    parser.add_argument(
        "--nn_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for 3-layer NN",
    )
    parser.add_argument(
        "--nn_lr",
        type=float,
        default=1e-3,
        help="Learning rate for 3-layer NN",
    )
    parser.add_argument(
        "--include_random_baseline",
        action="store_true",
        help="Include random baseline (randomly initialized model) for comparison",
    )
    return parser.parse_args()


def load_csv_data(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test CSV files."""
    train_path = os.path.join(csv_dir, "train.csv")
    test_path = os.path.join(csv_dir, "test.csv")

    # Check for dev.csv or val.csv
    dev_path = os.path.join(csv_dir, "dev.csv")
    val_path = os.path.join(csv_dir, "val.csv")
    if os.path.exists(dev_path):
        validation_path = dev_path
    elif os.path.exists(val_path):
        validation_path = val_path
    else:
        raise FileNotFoundError(f"No validation file found in {csv_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_complement_map():
    """
    Create the complement map for DNA tokens, matching the tokenizer's vocabulary.

    The vocabulary is:
        "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3, "[PAD]": 4,
        "[RESERVED]": 5, "[UNK]": 6, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11

    Complements: A<->T, C<->G, N<->N, special tokens map to themselves
    """
    complement_map = {
        0: 0,   # [CLS] -> [CLS]
        1: 1,   # [SEP] -> [SEP]
        2: 2,   # [BOS] -> [BOS]
        3: 3,   # [MASK] -> [MASK]
        4: 4,   # [PAD] -> [PAD]
        5: 5,   # [RESERVED] -> [RESERVED]
        6: 6,   # [UNK] -> [UNK]
        7: 10,  # A -> T
        8: 9,   # C -> G
        9: 8,   # G -> C
        10: 7,  # T -> A
        11: 11, # N -> N
    }
    return complement_map


def load_caduceus_model(config_path: str, checkpoint_path: str, device: torch.device):
    """
    Load Caduceus model from config and checkpoint.

    Args:
        config_path: Path to config JSON file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded Caduceus model
    """
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Handle nested Hydra-style config where actual params are under 'config' key
    # The JSON might look like: {"_name_": "...", "config": {"d_model": 256, ...}}
    if 'config' in config_dict and isinstance(config_dict['config'], dict):
        print(f"  Found nested 'config' key, extracting inner config")
        inner_config = config_dict['config']
        # Remove Hydra-specific keys like '_target_'
        inner_config = {k: v for k, v in inner_config.items() if not k.startswith('_')}
        config_dict = inner_config

    # Remove any Hydra-specific keys from top level
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}

    # Always add complement_map (required by Caduceus when rcps=True)
    # Adding it even when rcps=False doesn't cause issues
    # Check for None value too, not just missing key
    if not config_dict.get('complement_map'):
        print(f"  Adding complement_map for Caduceus model")
        config_dict['complement_map'] = get_complement_map()

    print(f"  Config: d_model={config_dict.get('d_model')}, n_layer={config_dict.get('n_layer')}, rcps={config_dict.get('rcps')}")

    config = CaduceusConfig(**config_dict)
    model = Caduceus(config)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove prefixes from state dict keys (from PyTorch Lightning)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Try different key patterns
        if key.startswith('model.caduceus.'):
            new_key = key.replace('model.caduceus.', '')
        elif key.startswith('model.backbone.'):
            new_key = key.replace('model.backbone.', '')
        elif key.startswith('model.'):
            new_key = key.replace('model.', '')
        elif key.startswith('caduceus.'):
            new_key = key.replace('caduceus.', '')
        else:
            new_key = key

        # Skip decoder/head weights
        if 'decoder' in new_key or 'head' in new_key or 'lm_head' in new_key:
            continue

        new_state_dict[new_key] = value

    # Load state dict with strict=False to handle missing keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys (will use random init): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

    model = model.to(device)
    model.eval()

    return model, config


def create_random_model(config_path: str, checkpoint_path: str, device: torch.device, seed: int = 42):
    """
    Create a randomly initialized Caduceus model (same architecture, no pretrained weights).

    Args:
        config_path: Path to config JSON file
        checkpoint_path: Path to checkpoint file (unused, kept for API compatibility)
        device: Device to load model on
        seed: Random seed for reproducibility

    Returns:
        Randomly initialized Caduceus model
    """
    print("\n" + "=" * 60)
    print("Creating Randomly Initialized Baseline Model")
    print("=" * 60)

    # Set seed for reproducible random initialization
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Handle nested Hydra-style config where actual params are under 'config' key
    if 'config' in config_dict and isinstance(config_dict['config'], dict):
        print(f"  Found nested 'config' key, extracting inner config")
        inner_config = config_dict['config']
        # Remove Hydra-specific keys like '_target_'
        inner_config = {k: v for k, v in inner_config.items() if not k.startswith('_')}
        config_dict = inner_config

    # Remove any Hydra-specific keys from top level
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}

    # Always add complement_map (required by Caduceus when rcps=True)
    # Check for None value too, not just missing key
    if not config_dict.get('complement_map'):
        print(f"  Adding complement_map for Caduceus model")
        config_dict['complement_map'] = get_complement_map()

    print(f"  Config: d_model={config_dict.get('d_model')}, n_layer={config_dict.get('n_layer')}, rcps={config_dict.get('rcps')}")

    config = CaduceusConfig(**config_dict)

    # Create model with random initialization (no checkpoint loading)
    model = Caduceus(config)
    model = model.to(device)
    model.eval()

    print("  Model initialized with random weights")

    return model, config


def get_tokenizer():
    """Get the character tokenizer for DNA sequences."""
    # Caduceus uses a character-level tokenizer
    # The vocab is typically: [PAD, A, C, G, T, N, ...]
    tokenizer = AutoTokenizer.from_pretrained(
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        trust_remote_code=True,
    )
    return tokenizer


def extract_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    labels: List[int],
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from Caduceus model for given sequences.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            # Get hidden states from Caduceus
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Direct output is the hidden states
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

            # Apply pooling
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if pooling == "mean":
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif pooling == "first":
                embeddings = hidden_states[:, 0, :]
            elif pooling == "last":
                if attention_mask is not None:
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    embeddings = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
                else:
                    embeddings = hidden_states[:, -1, :]

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch_labels)

    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)

    return embeddings_array, labels_array


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """Train a linear probe (logistic regression) classifier."""
    print("\n" + "=" * 60)
    print("Training Linear Probe (Logistic Regression)")
    print("=" * 60)

    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver='lbfgs',
        n_jobs=-1,
    )
    clf.fit(train_scaled, train_labels)

    # Predict
    test_preds = clf.predict(test_scaled)
    test_probs = clf.predict_proba(test_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        "linear_probe_accuracy": float(accuracy_score(test_labels, test_preds)),
        "linear_probe_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    try:
        metrics["linear_probe_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["linear_probe_auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["linear_probe_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["linear_probe_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"  Accuracy: {metrics['linear_probe_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['linear_probe_f1']:.4f}")
    print(f"  MCC: {metrics['linear_probe_mcc']:.4f}")
    print(f"  AUC: {metrics['linear_probe_auc']:.4f}")

    # Return metrics and predictions
    predictions = {
        "test_preds": test_preds,
        "test_probs": test_probs,
    }
    return metrics, predictions


def calculate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score for embeddings."""
    print("\n" + "=" * 60)
    print("Calculating Silhouette Score")
    print("=" * 60)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    score = silhouette_score(scaled_embeddings, labels)
    print(f"  Silhouette Score: {score:.4f}")
    print(f"  Interpretation: ", end="")
    if score > 0.5:
        print("Strong structure (embeddings well-separated by class)")
    elif score > 0.25:
        print("Reasonable structure")
    elif score > 0:
        print("Weak structure (some overlap between classes)")
    else:
        print("No apparent structure (classes highly overlapped)")

    return float(score)


def create_pca_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "PCA Visualization of Embeddings",
) -> Dict[str, float]:
    """Create PCA visualization of embeddings colored by class."""
    print("\n" + "=" * 60)
    print("Creating PCA Visualization")
    print("=" * 60)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(scaled_embeddings)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]*100:.2f}% of variance")
    print(f"  PC2 explains {explained_var[1]*100:.2f}% of variance")
    print(f"  Total: {sum(explained_var)*100:.2f}%")

    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e']
    class_names = ['Class 0', 'Class 1']

    for class_idx in [0, 1]:
        mask = labels == class_idx
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[class_idx],
            label=f'{class_names[class_idx]} (n={mask.sum()})',
            alpha=0.6,
            s=30,
        )

    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")

    return {
        "pca_explained_variance_pc1": float(explained_var[0]),
        "pca_explained_variance_pc2": float(explained_var[1]),
        "pca_total_explained_variance": float(sum(explained_var)),
    }


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
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


def train_three_layer_nn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], nn.Module, StandardScaler]:
    """Train a 3-layer neural network classifier on embeddings."""
    print("\n" + "=" * 60)
    print("Training 3-Layer Neural Network")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    train_X = torch.FloatTensor(train_scaled).to(device)
    train_y = torch.LongTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_scaled).to(device)
    val_y = torch.LongTensor(val_labels).to(device)
    test_X = torch.FloatTensor(test_scaled).to(device)
    test_y = torch.LongTensor(test_labels).to(device)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = train_embeddings.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )

    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

    metrics = {
        "nn_accuracy": float(accuracy_score(test_labels, test_preds)),
        "nn_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "nn_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "nn_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "nn_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    try:
        metrics["nn_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["nn_auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["nn_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["nn_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"\n  Final Test Results:")
    print(f"  Accuracy: {metrics['nn_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['nn_f1']:.4f}")
    print(f"  MCC: {metrics['nn_mcc']:.4f}")
    print(f"  AUC: {metrics['nn_auc']:.4f}")

    # Return metrics, model, scaler, and predictions
    predictions = {
        "test_preds": test_preds,
        "test_probs": test_probs,
    }
    return metrics, model, scaler, predictions


def run_analysis_on_embeddings(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_sequences: List[str],
    output_dir: str,
    prefix: str,
    nn_hidden_dim: int,
    nn_epochs: int,
    nn_lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run all embedding analyses (linear probe, silhouette, PCA, 3-layer NN).

    Args:
        test_sequences: List of test sequences (for saving predictions CSV)
        prefix: Prefix for output files and metric keys (e.g., "pretrained" or "random")

    Returns:
        Dictionary of all metrics
    """
    results = {}

    # 1. Train linear probe
    linear_metrics, linear_preds = train_linear_probe(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        seed,
    )
    results.update(linear_metrics)

    # 2. Calculate silhouette score
    silhouette = calculate_silhouette(test_embeddings, test_labels)
    results["silhouette_score"] = silhouette

    # 3. Create PCA visualization
    pca_path = os.path.join(output_dir, f"pca_visualization_{prefix}.png")
    pca_metrics = create_pca_visualization(
        test_embeddings, test_labels,
        pca_path,
        title=f"PCA of Test Embeddings ({prefix})\n(Silhouette: {silhouette:.3f})",
    )
    results.update(pca_metrics)

    # 4. Train 3-layer NN
    nn_metrics, nn_model, nn_scaler, nn_preds = train_three_layer_nn(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        nn_hidden_dim, nn_epochs, nn_lr,
        seed, device,
    )
    results.update(nn_metrics)

    # 5. Save test predictions to CSV
    predictions_df = pd.DataFrame({
        "sequence": test_sequences,
        "label": test_labels,
        "linear_probe_pred": linear_preds["test_preds"],
        "linear_probe_prob": linear_preds["test_probs"],
        "nn_pred": nn_preds["test_preds"],
        "nn_prob": nn_preds["test_probs"],
    })
    predictions_path = os.path.join(output_dir, f"test_predictions_{prefix}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved test predictions to: {predictions_path}")

    # Save NN model
    nn_model_path = os.path.join(output_dir, f"three_layer_nn_{prefix}.pt")
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "input_dim": test_embeddings.shape[1],
        "hidden_dim": nn_hidden_dim,
    }, nn_model_path)
    print(f"\nSaved 3-layer NN to: {nn_model_path}")

    return results


def main():
    """Main function to run embedding analysis."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Caduceus Embedding Analysis")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df, val_df, test_df = load_csv_data(args.csv_dir)
    tokenizer = get_tokenizer()

    # ========== PRETRAINED MODEL ==========
    print("\n" + "#" * 60)
    print("# PRETRAINED MODEL ANALYSIS")
    print("#" * 60)

    model, config = load_caduceus_model(args.config_path, args.checkpoint_path, device)

    # Extract embeddings from pretrained model
    print("\nExtracting train embeddings (pretrained)...")
    train_embeddings, train_labels = extract_embeddings(
        model, tokenizer,
        train_df["sequence"].tolist(),
        train_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print("\nExtracting validation embeddings (pretrained)...")
    val_embeddings, val_labels = extract_embeddings(
        model, tokenizer,
        val_df["sequence"].tolist(),
        val_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print("\nExtracting test embeddings (pretrained)...")
    test_embeddings, test_labels = extract_embeddings(
        model, tokenizer,
        test_df["sequence"].tolist(),
        test_df["label"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    print(f"\nEmbedding shape: {test_embeddings.shape}")

    # Save pretrained embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings_pretrained.npz")
    np.savez(
        embeddings_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    print(f"\nSaved pretrained embeddings to: {embeddings_path}")

    # Run analysis on pretrained embeddings
    pretrained_results = run_analysis_on_embeddings(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        test_df["sequence"].tolist(),
        args.output_dir, "pretrained",
        args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
        args.seed, device,
    )

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========== RANDOM BASELINE (if requested) ==========
    random_results = None
    if args.include_random_baseline:
        print("\n" + "#" * 60)
        print("# RANDOM BASELINE MODEL ANALYSIS")
        print("#" * 60)

        random_model, _ = create_random_model(args.config_path, args.checkpoint_path, device, seed=args.seed + 1000)

        # Extract embeddings from random model
        print("\nExtracting train embeddings (random)...")
        train_embeddings_rand, _ = extract_embeddings(
            random_model, tokenizer,
            train_df["sequence"].tolist(),
            train_df["label"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )

        print("\nExtracting validation embeddings (random)...")
        val_embeddings_rand, _ = extract_embeddings(
            random_model, tokenizer,
            val_df["sequence"].tolist(),
            val_df["label"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )

        print("\nExtracting test embeddings (random)...")
        test_embeddings_rand, _ = extract_embeddings(
            random_model, tokenizer,
            test_df["sequence"].tolist(),
            test_df["label"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )

        # Save random embeddings
        embeddings_path_rand = os.path.join(args.output_dir, "embeddings_random.npz")
        np.savez(
            embeddings_path_rand,
            train_embeddings=train_embeddings_rand,
            train_labels=train_labels,
            val_embeddings=val_embeddings_rand,
            val_labels=val_labels,
            test_embeddings=test_embeddings_rand,
            test_labels=test_labels,
        )
        print(f"\nSaved random embeddings to: {embeddings_path_rand}")

        # Run analysis on random embeddings
        random_results = run_analysis_on_embeddings(
            train_embeddings_rand, train_labels,
            val_embeddings_rand, val_labels,
            test_embeddings_rand, test_labels,
            test_df["sequence"].tolist(),
            args.output_dir, "random",
            args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
            args.seed, device,
        )

        # Free memory
        del random_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========== COMPILE FINAL RESULTS ==========
    results = {
        "checkpoint_path": args.checkpoint_path,
        "config_path": args.config_path,
        "csv_dir": args.csv_dir,
        "pooling": args.pooling,
        "embedding_dim": int(test_embeddings.shape[1]),
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "include_random_baseline": args.include_random_baseline,
    }

    # Add pretrained results with prefix
    for key, value in pretrained_results.items():
        results[f"pretrained_{key}"] = value

    # Add random results and compute embedding power if available
    if random_results is not None:
        for key, value in random_results.items():
            results[f"random_{key}"] = value

        # Compute embedding power (pretrained - random)
        print("\n" + "=" * 60)
        print("Computing Embedding Power (Pretrained - Random)")
        print("=" * 60)

        embedding_power = {}
        metrics_to_compare = [
            "linear_probe_accuracy", "linear_probe_f1", "linear_probe_mcc", "linear_probe_auc",
            "nn_accuracy", "nn_f1", "nn_mcc", "nn_auc",
            "silhouette_score",
        ]

        for metric in metrics_to_compare:
            pretrained_val = pretrained_results.get(metric, 0)
            random_val = random_results.get(metric, 0)
            power = pretrained_val - random_val
            embedding_power[f"embedding_power_{metric}"] = power
            print(f"  {metric}: {pretrained_val:.4f} - {random_val:.4f} = {power:+.4f}")

        results.update(embedding_power)

    # Save results
    results_path = os.path.join(args.output_dir, "embedding_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - PRETRAINED MODEL")
    print("=" * 60)
    print(f"\nLinear Probe Results:")
    print(f"  Accuracy: {pretrained_results['linear_probe_accuracy']:.4f}")
    print(f"  MCC: {pretrained_results['linear_probe_mcc']:.4f}")
    print(f"  AUC: {pretrained_results['linear_probe_auc']:.4f}")
    print(f"\n3-Layer NN Results:")
    print(f"  Accuracy: {pretrained_results['nn_accuracy']:.4f}")
    print(f"  MCC: {pretrained_results['nn_mcc']:.4f}")
    print(f"  AUC: {pretrained_results['nn_auc']:.4f}")
    print(f"\nEmbedding Quality:")
    print(f"  Silhouette Score: {pretrained_results['silhouette_score']:.4f}")
    print(f"  PCA Variance Explained: {pretrained_results['pca_total_explained_variance']*100:.1f}%")

    if random_results is not None:
        print("\n" + "=" * 60)
        print("SUMMARY - RANDOM BASELINE")
        print("=" * 60)
        print(f"\nLinear Probe Results:")
        print(f"  Accuracy: {random_results['linear_probe_accuracy']:.4f}")
        print(f"  MCC: {random_results['linear_probe_mcc']:.4f}")
        print(f"  AUC: {random_results['linear_probe_auc']:.4f}")
        print(f"\n3-Layer NN Results:")
        print(f"  Accuracy: {random_results['nn_accuracy']:.4f}")
        print(f"  MCC: {random_results['nn_mcc']:.4f}")
        print(f"  AUC: {random_results['nn_auc']:.4f}")
        print(f"\nEmbedding Quality:")
        print(f"  Silhouette Score: {random_results['silhouette_score']:.4f}")
        print(f"  PCA Variance Explained: {random_results['pca_total_explained_variance']*100:.1f}%")

        print("\n" + "=" * 60)
        print("EMBEDDING POWER (Pretrained - Random)")
        print("=" * 60)
        print(f"\nLinear Probe:")
        print(f"  Accuracy: {embedding_power['embedding_power_linear_probe_accuracy']:+.4f}")
        print(f"  MCC: {embedding_power['embedding_power_linear_probe_mcc']:+.4f}")
        print(f"  AUC: {embedding_power['embedding_power_linear_probe_auc']:+.4f}")
        print(f"\n3-Layer NN:")
        print(f"  Accuracy: {embedding_power['embedding_power_nn_accuracy']:+.4f}")
        print(f"  MCC: {embedding_power['embedding_power_nn_mcc']:+.4f}")
        print(f"  AUC: {embedding_power['embedding_power_nn_auc']:+.4f}")
        print(f"\nSilhouette Score: {embedding_power['embedding_power_silhouette_score']:+.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
