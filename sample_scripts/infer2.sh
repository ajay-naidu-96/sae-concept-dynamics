#!/bin/bash
set -euo pipefail  

GPU_ID=2

# Define datasets and corresponding log dirs
declare -A DATASETS
DATASETS["./logs/cifar10expt5/test_activations/"]="
    ./logs/cifar10expt5/sae_train_logs/vanilla_sae/vanilla_sae/
"
# DATASETS["./logs/mnist_activation/test_activations/"]="
#     ./logs/mnist_activation/sae_train_logs/vanilla_sae/
# "

run_inference() {
    local data_dir="$1"
    local log_dir="$2"

    echo "========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Running inference"
    echo "DATA_DIR: $data_dir"
    echo "LOG_DIR : $log_dir"
    echo "========================================="

    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py --data_dir "$data_dir" --log_dir "$log_dir" --run_regression
    # CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py --data_dir "$data_dir" --log_dir "$log_dir" --run_concept_path
    # CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py --data_dir "$data_dir" --log_dir "$log_dir" --run_stable_rank --stable_rank_thresholds 0.01 0.05 0.1 --use_frobenius_definition

    echo "✅ Completed: $log_dir"
    echo ""
}

# Iterate over datasets and log dirs
for data_dir in "${!DATASETS[@]}"; do
    for log_dir in ${DATASETS[$data_dir]}; do
        run_inference "$data_dir" "$log_dir"
    done
done

echo "🎉 All inference runs completed successfully!"

