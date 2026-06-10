#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="${GPUS:-0,1}"
RESTORE_DIR="${RESTORE_DIR:-results/arrowmatch_results}"
TEXT_DATASETS=(mnli qqp qnli sst2)
OBFUSCATIONS=(black translinkguard tempo soter shadownet LoRO obfuscatune groupcover twinshield arrowcloak)

for dataset in "${TEXT_DATASETS[@]}"; do
  for obfus in "${OBFUSCATIONS[@]}"; do
    ./scripts/arrowmatch.sh --gpus "$GPUS" --dataset "$dataset" --obfus "$obfus" --restore_dir "$RESTORE_DIR"
    ./scripts/arrowmatch_gpt2.sh --gpus "$GPUS" --dataset "$dataset" --obfus "$obfus" --restore_dir "$RESTORE_DIR"
  done
done

for obfus in "${OBFUSCATIONS[@]}"; do
  ./scripts/arrowmatch_gpt2_xl.sh --gpus "$GPUS" --dataset sst2 --obfus "$obfus" --restore_dir "$RESTORE_DIR"
done
