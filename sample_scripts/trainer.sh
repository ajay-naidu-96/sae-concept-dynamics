#!/bin/bash

set -e
set -u

DATA_DIR="./logs/mnist_activation/train_activations/"
LR=1e-3
BATCH_SIZE=1024
EPOCHS=25
DEVICE="cuda"
DEVICE_ID=0
NB_CONCEPTS_LIST=(256 512)

SEEDS=(42 69 128)

VANILLA_LOG_DIRS=(
    "./logs/mnist_activation/Seed1/vanilla_sae/"
    "./logs/mnist_activation/Seed2/vanilla_sae/"
    "./logs/mnist_activation/Seed3/vanilla_sae/"
)

TOPK_LOG_DIRS=(
    "./logs/mnist_activation/Seed1/topk_sae/"
    "./logs/mnist_activation/Seed2/topk_sae/"
    "./logs/mnist_activation/Seed3/topk_sae/"
)

REANIM_FLAGS=("" "--reanim")

echo "========================================"
echo "Starting SAE training across seeds"
echo "Concepts list: ${NB_CONCEPTS_LIST[*]}"
echo "Using device ID: ${DEVICE_ID}"
echo "========================================"

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    echo ""
    echo ">>> Training for Seed ${SEED}"
    echo "----------------------------------------"

    for FLAG in "${REANIM_FLAGS[@]}"; do
        if [[ "$FLAG" == "--reanim" ]]; then
            FLAG_NAME="reanim"
        else
            FLAG_NAME="no_reanim"
        fi

        echo "Running Vanilla SAE (${FLAG_NAME}, Seed ${SEED})..."
        CUDA_VISIBLE_DEVICES=${DEVICE_ID} python src/sae_trainer.py \
            --sae_type vanilla \
            --data_dir "$DATA_DIR" \
            --log_dir "${VANILLA_LOG_DIRS[$i]}" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --device "$DEVICE" \
            --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
            --seed "$SEED" \
            $FLAG

        echo "Running Top-K SAE (${FLAG_NAME}, Seed ${SEED})..."
        CUDA_VISIBLE_DEVICES=${DEVICE_ID} python src/sae_trainer.py \
            --sae_type top_k \
            --data_dir "$DATA_DIR" \
            --log_dir "${TOPK_LOG_DIRS[$i]}" \
            --lr "$LR" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --device "$DEVICE" \
            --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
            --seed "$SEED" \
            $FLAG

        echo "Completed both SAE types (${FLAG_NAME}) for Seed ${SEED}."
        echo "----------------------------------------"
    done
done

echo "All trainings complete!"
