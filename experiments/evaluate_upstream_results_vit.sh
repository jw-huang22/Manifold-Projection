#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

VIT_DATASETS=(cifar_10 cifar_100 food101)

for dataset in "${VIT_DATASETS[@]}"; do
  python code/evaluate_model_vit.py --dataset "$dataset" --result_of_arrowcloak_model false
done
