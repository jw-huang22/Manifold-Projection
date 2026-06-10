#!/bin/bash

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

DATASET="cifar_100"
OUTPUT_DIR="results/train_results/ViT"
TSQP="false"
OPT="adam"
LR="5e-5"
BS="128"
N_EPOCHS="30"
LR_SCHEDULER_TYPE="cosine"
LR_SCHEDULER_WARMUP_RATIO="0.1"
WEIGHT_DECAY="0.01"
VIT_MODEL="vit_base_patch16_224.orig_in21k"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --tsqp) TSQP="$2"; shift ;;
        --opt) OPT="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --n_epochs) N_EPOCHS="$2"; shift ;;
        --lr_scheduler_type) LR_SCHEDULER_TYPE="$2"; shift ;;
        --lr_scheduler_warmup_ratio) LR_SCHEDULER_WARMUP_RATIO="$2"; shift ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift ;;
        --vit_model) VIT_MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python train/train_vit.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tsqp "$TSQP" \
    --opt "$OPT" \
    --lr "$LR" \
    --bs "$BS" \
    --n_epochs "$N_EPOCHS" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --lr_scheduler_warmup_ratio "$LR_SCHEDULER_WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --vit_model "$VIT_MODEL"
