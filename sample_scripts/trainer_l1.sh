#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

DATA_DIR="./logs/cifar10expt5/train_activations/"

LR=1e-3
BATCH_SIZE=1024
EPOCHS=25
DEVICE="cuda"
NB_CONCEPTS_LIST=(256 512)

echo "Starting SAE training..."
echo "Concepts list: ${NB_CONCEPTS_LIST[*]}"
echo "----------------------------------------"

LOG_DIR="./logs/cifar10expt/sae_train_logs_l1_v2/vanilla_sae/"

CUDA_VISIBLE_DEVICES=3 python src/sae_trainer.py \
    --sae_type vanilla \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --nb_concepts_list "${NB_CONCEPTS_LIST[@]}"

CUDA_VISIBLE_DEVICES=3 python3 src/sae_trainer.py \
    --sae_type vanilla \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
    --reanim

# LOG_DIR="./logs/cifar10expt5/sae_train_logs_run3/topk_sae/"

# CUDA_VISIBLE_DEVICES=3 python src/sae_trainer.py \
#     --sae_type top_k \
#     --data_dir "$DATA_DIR" \
#     --log_dir "$LOG_DIR" \
#     --lr "$LR" \
#     --batch_size "$BATCH_SIZE" \
#     --epochs "$EPOCHS" \
#     --device "$DEVICE" \
#     --nb_concepts_list "${NB_CONCEPTS_LIST[@]}"

# CUDA_VISIBLE_DEVICES=3 python3 src/sae_trainer.py \
#     --sae_type top_k \
#     --data_dir "$DATA_DIR" \
#     --log_dir "$LOG_DIR" \
#     --lr "$LR" \
#     --batch_size "$BATCH_SIZE" \
#     --epochs "$EPOCHS" \
#     --device "$DEVICE" \
#     --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
#     --reanim

echo "----------------------------------------"
echo "All training runs completed successfully."

DATA_DIR="./logs/mnist_activation/train_activations/"

LR=1e-3
BATCH_SIZE=1024
EPOCHS=25
DEVICE="cuda"
NB_CONCEPTS_LIST=(256 512)

echo "Starting SAE training..."
echo "Concepts list: ${NB_CONCEPTS_LIST[*]}"
echo "----------------------------------------"

LOG_DIR="./logs/mnist_activation/sae_train_logs_l1_v2/vanilla_sae/"

CUDA_VISIBLE_DEVICES=3 python src/sae_trainer.py \
    --sae_type vanilla \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --nb_concepts_list "${NB_CONCEPTS_LIST[@]}"

CUDA_VISIBLE_DEVICES=3 python3 src/sae_trainer.py \
    --sae_type vanilla \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
    --reanim

# LOG_DIR="./logs/mnist_activation/sae_train_logs_run3/topk_sae/"

# CUDA_VISIBLE_DEVICES=3 python src/sae_trainer.py \
#     --sae_type top_k \
#     --data_dir "$DATA_DIR" \
#     --log_dir "$LOG_DIR" \
#     --lr "$LR" \
#     --batch_size "$BATCH_SIZE" \
#     --epochs "$EPOCHS" \
#     --device "$DEVICE" \
#     --nb_concepts_list "${NB_CONCEPTS_LIST[@]}"

# CUDA_VISIBLE_DEVICES=3 python3 src/sae_trainer.py \
#     --sae_type top_k \
#     --data_dir "$DATA_DIR" \
#     --log_dir "$LOG_DIR" \
#     --lr "$LR" \
#     --batch_size "$BATCH_SIZE" \
#     --epochs "$EPOCHS" \
#     --device "$DEVICE" \
#     --nb_concepts_list "${NB_CONCEPTS_LIST[@]}" \
#     --reanim

echo "----------------------------------------"
echo "All training runs completed successfully."