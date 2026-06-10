#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

VIT_DATASETS=(cifar_10 cifar_100 food101)
OBFUSCATIONS=(translinkguard tempo soter shadownet LoRO obfuscatune groupcover twinshield arrowcloak LRAO)

for obfus in "${OBFUSCATIONS[@]}"; do
  python code/measure_adp.py --models vit --datasets "${VIT_DATASETS[@]}" --obfus "$obfus"
done
