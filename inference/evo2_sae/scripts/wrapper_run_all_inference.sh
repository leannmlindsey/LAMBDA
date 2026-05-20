#!/bin/bash

# Wrapper script for running batch inference with all methods (SAE, NN, LP)
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash scripts/run_all_inference_interactive.sh [this_wrapper.sh]

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input Files ===
# Path to text file containing one input CSV path per line
# Example contents of input_files.txt:
#   /path/to/dataset1.csv
#   /path/to/dataset2.csv
INPUT_LIST="/path/to/input_files.txt"

# === REQUIRED: Output Directory ===
OUTPUT_DIR="/path/to/output_directory"

# === REQUIRED: Trained Model Artifacts ===
# Directory containing: three_layer_nn.pt, three_layer_nn_scaler.pkl,
#                       linear_probe.pkl, linear_probe_scaler.pkl
MODEL_DIR="/path/to/results/embedding_analysis/2k"

# === OPTIONAL: Which methods to run (true/false) ===
RUN_SAE="true"
RUN_NN="true"
RUN_LP="true"

# === OPTIONAL: Model Configuration ===
MODEL="evo2_7b"
LAYER="blocks.28.mlp.l3"
POOLING="mean"
BATCH_SIZE="16"

# === OPTIONAL: SAE Parameters ===
FEATURE_IDX="19746"
SAE_MAX_THRESHOLD="0.5"
SAE_MEAN_THRESHOLD="0.1"
SAE_FRACTION_THRESHOLD="0.3"
SAVE_ACTIVATIONS="true"

#####################################################################
# END CONFIGURATION
#####################################################################
