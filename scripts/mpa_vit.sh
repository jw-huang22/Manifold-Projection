#!/bin/bash

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="60"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

GPUS="0"
DATASET="cifar_10"
WEIGHT_DIR="results/train_results"
WEIGHT_DIR_TSQP="results/tsqp_results"
RESTORE_DIR="results/our_results"
RECOVER_DATA_DIR="data/recover_data"
OUTPUT_DIR="tmp/output_results"
OBFUS_DIR="tmp/obfus_results"
OBFUS="translinkguard"
BS="32"
RECOVER_EPOCHS="10"
RECOVER_LR="5e-5"
OPT="adam"
LR_SCHEDULER_TYPE="linear"
LR_SCHEDULER_WARMUP_RATIO="0.1"
WEIGHT_DECAY="0.05"
RANK_R="8"
VIT_MODEL="vit_base_patch16_224.orig_in21k"
MODEL_NAME="ViT"
RECOVER_DATA_RATIO="0.01"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --weight_dir) WEIGHT_DIR="$2"; shift ;;
        --weight_dir_tsqp) WEIGHT_DIR_TSQP="$2"; shift ;;
        --restore_dir) RESTORE_DIR="$2"; shift ;;
        --recover_data_dir) RECOVER_DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --obfus_dir) OBFUS_DIR="$2"; shift ;;
        --obfus) OBFUS="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --recover_epochs) RECOVER_EPOCHS="$2"; shift ;;
        --recover_lr) RECOVER_LR="$2"; shift ;;
        --opt) OPT="$2"; shift ;;
        --lr_scheduler_type) LR_SCHEDULER_TYPE="$2"; shift ;;
        --lr_scheduler_warmup_ratio) LR_SCHEDULER_WARMUP_RATIO="$2"; shift ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift ;;
        --rank_r) RANK_R="$2"; shift ;;
        --vit_model) VIT_MODEL="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --recover_data_ratio) RECOVER_DATA_RATIO="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python code/mpa_vit.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --weight_dir "$WEIGHT_DIR" \
    --weight_dir_tsqp "$WEIGHT_DIR_TSQP" \
    --restore_dir "$RESTORE_DIR" \
    --recover_data_dir "$RECOVER_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --obfus_dir "$OBFUS_DIR" \
    --obfus "$OBFUS" \
    --bs "$BS" \
    --recover_epochs "$RECOVER_EPOCHS" \
    --recover_lr "$RECOVER_LR" \
    --opt "$OPT" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --lr_scheduler_warmup_ratio "$LR_SCHEDULER_WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --rank_r "$RANK_R" \
    --vit_model "$VIT_MODEL" \
    --model_name "$MODEL_NAME" \
    --recover_data_ratio "$RECOVER_DATA_RATIO"
