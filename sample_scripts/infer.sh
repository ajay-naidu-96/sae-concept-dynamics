#!/bin/bash
set -euo pipefail

GPU_ID=2
DATA_DIR="./logs/mnist_activation/test_activations/"

LOG_DIRS=(
    "./logs/mnist_activation/Seed1/vanilla_sae/"
    "./logs/mnist_activation/Seed2/vanilla_sae/"
    "./logs/mnist_activation/Seed3/vanilla_sae/"
    "./logs/mnist_activation/Seed1/topk_sae/"
    "./logs/mnist_activation/Seed2/topk_sae/"
    "./logs/mnist_activation/Seed3/topk_sae/"
)

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
        # CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py \
        #     --data_dir "$data_dir" \
        #     --log_dir "$log_dir" \
        #     --run_regression
        
        # CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py \
        #     --data_dir "$data_dir" \
        #     --log_dir "$log_dir" \
        #     --run_stable_rank \
        #     --stable_rank_thresholds 0.01 0.05 0.1 \
        #     --use_frobenius_definition

        CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/infer_v2.py \
        --run_concept_path  \
        --log_dir "$log_dir" \
        --data_dir "$data_dir"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully"
    } >> "$logfile" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $log_dir (log saved to $logfile)"
    else
        echo "❌ Failed: $log_dir (check $logfile for errors)"
        return 1
    fi
    echo ""
}

echo "🚀 Starting inference for ${#LOG_DIRS[@]} directories..."
echo ""

failed_dirs=()
for log_dir in "${LOG_DIRS[@]}"; do
    if ! run_inference "$DATA_DIR" "$log_dir"; then
        failed_dirs+=("$log_dir")
    fi
done

echo "========================================="
echo "🎉 All inference runs completed!"
echo "Total directories: ${#LOG_DIRS[@]}"
echo "Successful: $((${#LOG_DIRS[@]} - ${#failed_dirs[@]}))"
echo "Failed: ${#failed_dirs[@]}"

if [ ${#failed_dirs[@]} -gt 0 ]; then
    echo ""
    echo "Failed directories:"
    for dir in "${failed_dirs[@]}"; do
        echo "  - $dir"
    done
    exit 1
fi
echo "========================================="