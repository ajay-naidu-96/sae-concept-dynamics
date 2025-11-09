#!/bin/bash
# combined_infer.sh
# This script runs infer.sh followed by infer_cifar.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================"
echo "Starting infer.sh..."
echo "========================================"
bash infer.sh

echo
echo "========================================"
echo "infer.sh completed successfully."
echo "Starting infer_cifar.sh..."
echo "========================================"
bash infer_cifar.sh

echo
echo "========================================"
echo "All inference tasks completed successfully!"
echo "========================================"
