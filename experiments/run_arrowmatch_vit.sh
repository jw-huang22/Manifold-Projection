#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="${GPUS:-0}"
RESTORE_DIR="${RESTORE_DIR:-results/arrowmatch_results}"
VIT_DATASETS=(cifar_10 cifar_100 food101)
OBFUSCATIONS=(black translinkguard tempo soter shadownet LoRO obfuscatune groupcover twinshield arrowcloak)

for dataset in "${VIT_DATASETS[@]}"; do
  for obfus in "${OBFUSCATIONS[@]}"; do
    ./scripts/arrowmatch_vit.sh --gpus "$GPUS" --dataset "$dataset" --obfus "$obfus" --restore_dir "$RESTORE_DIR"
  done
done
