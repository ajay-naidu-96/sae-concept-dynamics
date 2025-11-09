#!/bin/bash

# SAE Neuron Visualization Runner Script
# This script runs the visualization for SAE neurons on different datasets

set -e  # Exit on error

# ============================================
# Configuration - Update these paths
# ============================================

# Dataset selection: "mnist" or "cifar10"
DATASET="cifar10"

ORACLE_PATH="./logs/cifar10expt5/oracle.pt"
SAE_PATH="./logs/cifar10expt5/Seed1/vanilla_sae/vanilla_sae_512_18/best_loss.pth"

# Data directory
DATA_DIR="./Data/"

# Output directory
LOG_DIR="./visualizations_512/${DATASET}_$(date +%Y%m%d_%H%M%S)"

# Visualization parameters
NUM_NEURONS=10
SELECTION_METHOD="top_max_activation"  # Options: top_max_activation, top_mean_activation, least_sparse, most_sparse
NUM_SAMPLES=16
BATCH_SIZE=256

# Device (cuda or cpu)
DEVICE="cuda"

# Use training set instead of test set (uncomment to enable)
# USE_TRAIN="--is_train"
USE_TRAIN=""

# ============================================
# Run the visualization script
# ============================================

echo "================================================"
echo "SAE Neuron Visualization"
echo "================================================"
echo "Dataset: ${DATASET}"
echo "Oracle Model: ${ORACLE_PATH}"
echo "SAE Model: ${SAE_PATH}"
echo "Output Directory: ${LOG_DIR}"
echo "Number of Neurons: ${NUM_NEURONS}"
echo "Selection Method: ${SELECTION_METHOD}"
echo "Samples per Neuron: ${NUM_SAMPLES}"
echo "================================================"
echo ""

# Create output directory
mkdir -p "${LOG_DIR}"

# Run the Python script
python src/visualize_sae_activations.py \
    --dataset "${DATASET}" \
    --oracle_path "${ORACLE_PATH}" \
    --sae_path "${SAE_PATH}" \
    --data_dir "${DATA_DIR}" \
    --log_dir "${LOG_DIR}" \
    --num_neurons ${NUM_NEURONS} \
    --selection_method "${SELECTION_METHOD}" \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --device "${DEVICE}" \
    ${USE_TRAIN}

echo ""
echo "================================================"
echo "✓ Visualization Complete!"
echo "Results saved to: ${LOG_DIR}"
echo "================================================"

# Optional: Open the output directory
# Uncomment the appropriate line for your OS
# xdg-open "${LOG_DIR}"  # Linux
# open "${LOG_DIR}"      # macOS
# explorer "${LOG_DIR}"  # Windows (Git Bash)