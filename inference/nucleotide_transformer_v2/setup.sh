#!/bin/bash
# Setup conda environment for Nucleotide Transformer fine-tuning on Biowulf
# Run this once to create the environment

echo "Setting up NT fine-tuning environment..."

# Load modules
module load conda
module load cuda/12.8

# Create conda environment
conda create -n nt python=3.10 -y
source activate nt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace and dependencies
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# Install additional dependencies
pip install scikit-learn pandas numpy einops

echo ""
echo "Environment setup complete!"
echo "Activate with: source activate nt"
echo ""
echo "To test, run:"
echo "  python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', trust_remote_code=True); print('Success!')\""
