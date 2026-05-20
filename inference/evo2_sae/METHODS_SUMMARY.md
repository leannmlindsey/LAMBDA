# SAE Activation Extraction for Prophage Detection

## Overview

We use a Sparse Autoencoder (SAE) trained on Evo2 to detect prophage regions in bacterial genomes. The SAE was trained by Goodfire/Arc Institute and contains interpretable features, including feature **f/19746** which activates specifically on prophage sequences.

## SAE Source

- **Repository:** [Goodfire/Evo-2-Layer-26-Mixed](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed)
- **File:** `sae-layer26-mixed-expansion_8-k_64.pt`
- **Architecture:** BatchTopK Tied-weight SAE
  - Input dimension: 4,096 (Evo2 hidden dimension)
  - SAE dimension: 32,768 (8× expansion)
  - TopK: 64 (only top 64 features active per position)

## Model

- **Base model:** Evo2 7B (`evo2_7b`)
- **Hook location:** Layer 26 (`blocks-26`)

## Activation Extraction Process

1. **Tokenize** the DNA sequence using Evo2's tokenizer
2. **Forward pass** through Evo2, capturing hidden states at layer 26 via PyTorch hooks
3. **Encode** the layer-26 activations through the SAE: `features = sae.encode(hidden_states)`
4. **Extract** prophage feature f/19746: `prophage_signal = features[:, 19746]`

## Windowing (for long genomes)

- **Window size:** 50,000 bp
- **Overlap:** 1,000 bp
- **Stride:** 49,000 bp
- Overlapping regions use **MAX** to preserve sparse signal
- First 10 positions of each window (except first) zeroed to remove startup artifacts

## Post-processing Pipeline

1. **Artifact removal:** Zero out window boundary artifacts
2. **Z-score normalization:** `(x - mean) / std` per genome
3. **Thresholding:** Positions with z-score > 7.0
4. **Clustering:** Group nearby positions (max_gap=300 bp)
5. **Size filtering:** Remove regions < 1,000 bp
6. **Merging:** Combine regions within 5,000 bp

## Code

```python
from huggingface_hub import hf_hub_download

# Download SAE weights
sae_path = hf_hub_download(
    repo_id="Goodfire/Evo-2-Layer-26-Mixed",
    filename="sae-layer26-mixed-expansion_8-k_64.pt",
    repo_type="model"
)

# Extract features (simplified)
toks = model.tokenizer.tokenize(sequence)
toks = torch.tensor(toks).unsqueeze(0).to(device)
logits, acts = model.forward(toks, cache_activations_at=['blocks-26'])
features = sae.encode(acts['blocks-26'][0])
prophage_signal = features[:, 19746].cpu().numpy()
```

## Best Parameters (from optimization)

| Parameter | Value |
|-----------|-------|
| Normalization | z-score |
| Threshold | 7.0 |
| Max gap | 300 bp |
| Merge distance | 5,000 bp |
| Min region size | 1,000 bp |

## Performance

- **MCC:** 0.599
- **Precision:** 71.9%
- **Recall:** 52.0%
