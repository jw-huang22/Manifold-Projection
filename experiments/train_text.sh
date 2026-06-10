#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="${GPUS:-0,1}"
TEXT_DATASETS=(mnli qqp qnli sst2)

for dataset in "${TEXT_DATASETS[@]}"; do
  ./scripts/train.sh --gpus "$GPUS" --model bert-base-cased --dataset "$dataset" --output_dir results/train_results
  ./scripts/train.sh --gpus "$GPUS" --model gpt2 --dataset "$dataset" --output_dir results/train_results
done

./scripts/train_gpt2_xl.sh --gpus "$GPUS" --dataset sst2 --output_dir results/train_results
