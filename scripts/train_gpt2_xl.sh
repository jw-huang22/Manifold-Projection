#!/bin/bash

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="0,1"
DATASET="sst2"
OUTPUT_DIR="results/train_results"
TSQP="false"
FULL_FINETUNE="true"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --tsqp) TSQP="$2"; shift ;;
        --full_finetune) FULL_FINETUNE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python train/train_gpt2_xl.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tsqp "$TSQP" \
    --full_finetune "$FULL_FINETUNE"
