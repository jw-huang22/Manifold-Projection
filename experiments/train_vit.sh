#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

VIT_MODEL="${VIT_MODEL:-vit_base_patch16_224.augreg_in21k}"
VIT_DATASETS=(cifar_10 cifar_100 food101)

for dataset in "${VIT_DATASETS[@]}"; do
  ./scripts/train_vit.sh \
    --dataset "$dataset" \
    --output_dir results/train_results/ViT \
    --vit_model "$VIT_MODEL" \
    --bs "${VIT_BS:-64}" \
    --n_epochs "${VIT_EPOCHS:-10}"
done
