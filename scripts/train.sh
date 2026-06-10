#!/bin/bash

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="0,1"
DATASET="mnli"
MODEL="bert-base-cased"
OUTPUT_DIR="results/train_results"
TSQP="false"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --tsqp) TSQP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python train/train.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tsqp "$TSQP"
