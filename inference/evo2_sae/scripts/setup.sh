#!/bin/bash
# ============================================================
# Evo2 SAE Setup for H200 Node (No SLURM)
# ============================================================
#
# Run this script to set up the environment:
#   bash setup.sh
#
# Or with custom install location:
#   EVO2_BASE_DIR=/path/to/your/dir bash setup.sh
#
# ============================================================

set -e  # Exit on error

# ============================================================
# USER CONFIGURATION - Set your install directory here
# ============================================================
# Default is $HOME, override with environment variable or edit this line
EVO2_BASE_DIR="${EVO2_BASE_DIR:-$HOME}"

echo "============================================================"
echo "Evo2 SAE Prophage Detection - Setup"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Install directory: ${EVO2_BASE_DIR}"
echo ""

# Check for GPUs
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# ============================================================
# Step 1: Create conda environment
# ============================================================
echo "Step 1: Creating conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda/anaconda first."
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "evo2-sae"; then
    echo "Environment 'evo2-sae' already exists. Activating..."
else
    echo "Creating new environment 'evo2-sae'..."
    conda create -n evo2-sae python=3.12 -y
fi

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evo2-sae

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================
# Step 2: Install CUDA dependencies
# ============================================================
echo "Step 2: Installing CUDA dependencies..."

conda install -c nvidia cuda-nvcc cuda-cudart-dev -y
conda install -c conda-forge transformer-engine-torch=2.3.0 -y

# ============================================================
# Step 3: Install Flash Attention
# ============================================================
echo "Step 3: Installing Flash Attention..."

# Install psutil first (required for flash-attn build)
pip install psutil

pip install flash-attn==2.8.0.post2 --no-build-isolation

# ============================================================
# Step 4: Install Evo2
# ============================================================
echo "Step 4: Installing Evo2..."

pip install evo2

# Additional dependencies
pip install huggingface_hub pandas matplotlib seaborn tqdm biopython

# ============================================================
# Step 5: Clone Evo2 repo for notebooks
# ============================================================
echo "Step 5: Cloning Evo2 repository..."

cd "${EVO2_BASE_DIR}"
if [ -d "evo2" ]; then
    echo "evo2 directory already exists, pulling latest..."
    cd evo2 && git pull && cd -
else
    git clone https://github.com/arcinstitute/evo2
fi

# ============================================================
# Step 6: Download SAE weights
# ============================================================
echo "Step 6: Downloading SAE weights from HuggingFace..."

python << EOF
from huggingface_hub import snapshot_download, list_repo_files
import os

repo_id = "Goodfire/Evo-2-Layer-26-Mixed"
local_dir = "${EVO2_BASE_DIR}/sae_weights"

print(f"Downloading {repo_id} to {local_dir}...")

# First, list the files to see what's available
print("\\nFiles in repository:")
files = list_repo_files(repo_id)
for f in files:
    print(f"  {f}")

# Download all files
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    repo_type="model"
)

print(f"\\nDownload complete! Files saved to: {local_dir}")
EOF

# ============================================================
# Step 7: Verify installation
# ============================================================
echo ""
echo "Step 7: Verifying installation..."

python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Test evo2 import
try:
    from evo2 import Evo2
    print("\\n✓ Evo2 imported successfully")
except Exception as e:
    print(f"\\n✗ Evo2 import failed: {e}")

# Check SAE weights
import os
sae_dir = "${EVO2_BASE_DIR}/sae_weights"
if os.path.exists(sae_dir):
    print(f"\\n✓ SAE weights directory exists: {sae_dir}")
    for f in os.listdir(sae_dir):
        fpath = os.path.join(sae_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024  # MB
            print(f"    {f}: {size:.1f} MB")
else:
    print(f"\\n✗ SAE weights not found")
EOF

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment:  conda activate evo2-sae"
echo "  2. Test Evo2 model:       python -m evo2.test.test_evo2_generation --model_name evo2_7b"
echo "  3. Run checkpoint inspector: python exploratory/inspect_sae_checkpoint.py"
echo "  4. Run prophage detection:   python exploratory/prophage_detection.py --help"
echo ""
echo "SAE weights installed to: ${EVO2_BASE_DIR}/sae_weights"
echo ""
