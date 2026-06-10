#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="${GPUS:-0,1}"
TEXT_DATASETS=(mnli qqp qnli sst2)

for dataset in "${TEXT_DATASETS[@]}"; do
  python code/evaluate_model.py --dataset "$dataset" --gpus "$GPUS" --result_of_arrowcloak_model false
  python code/evaluate_model_gpt2.py --dataset "$dataset" --gpus "$GPUS" --result_of_arrowcloak_model false
done

python code/evaluate_model_gpt2_xl.py --dataset sst2 --gpus "$GPUS" --result_of_arrowcloak_model false
