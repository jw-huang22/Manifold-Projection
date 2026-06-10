#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

GPUS="${GPUS:-0,1}"
RESTORE_DIR="${RESTORE_DIR:-results/mpa_results}"
RANKS=(${RANKS:-32 64 128})
TEXT_DATASETS=(mnli qqp qnli sst2)
OBFUSCATIONS=(black translinkguard tempo soter shadownet LoRO obfuscatune groupcover twinshield arrowcloak)

for dataset in "${TEXT_DATASETS[@]}"; do
  for obfus in "${OBFUSCATIONS[@]}"; do
    ./scripts/mpa.sh --gpus "$GPUS" --dataset "$dataset" --obfus "$obfus" --restore_dir "$RESTORE_DIR"
    ./scripts/mpa_gpt2.sh --gpus "$GPUS" --dataset "$dataset" --obfus "$obfus" --restore_dir "$RESTORE_DIR"
  done
  for rank in "${RANKS[@]}"; do
    ./scripts/mpa.sh --gpus "$GPUS" --dataset "$dataset" --obfus LRAO --restore_dir "$RESTORE_DIR" --rank_r "$rank"
    ./scripts/mpa_gpt2.sh --gpus "$GPUS" --dataset "$dataset" --obfus LRAO --restore_dir "$RESTORE_DIR" --rank_r "$rank"
  done
done

for obfus in "${OBFUSCATIONS[@]}"; do
  ./scripts/mpa_gpt2_xl.sh --gpus "$GPUS" --dataset sst2 --obfus "$obfus" --restore_dir "$RESTORE_DIR"
done
for rank in "${RANKS[@]}"; do
  ./scripts/mpa_gpt2_xl.sh --gpus "$GPUS" --dataset sst2 --obfus LRAO --restore_dir "$RESTORE_DIR" --rank_r "$rank"
done
