#!/bin/bash
set -euo pipefail

GPU_ID=2

DATA_DIR="./logs/cifar10expt5/test_activations/"
LOG_DIR="./logs/cifar10expt5/Seed1/vanilla_sae/"

run_inference() {
    local data_dir="$1"
    local log_dir="$2"
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local logfile="${log_dir%/}/inference_${timestamp}.log"

    echo "========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Running inference"
    echo "DATA_DIR: $data_dir"
    echo "LOG_DIR : $log_dir"
    echo "LOG_FILE: $logfile"
    echo "========================================="

    mkdir -p "$log_dir"

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py --data_dir "$data_dir" --log_dir "$log_dir" --run_regression
        CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py --data_dir "$data_dir" --log_dir "$log_dir" --run_stable_rank --stable_rank_thresholds 0.01 0.05 0.1 --use_frobenius_definition
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully"
    } >> "$logfile" 2>&1

    echo "✅ Completed: $log_dir (log saved to $logfile)"
    echo ""
}

# Run once
run_inference "$DATA_DIR" "$LOG_DIR"

echo "🎉 Inference run completed successfully!"
