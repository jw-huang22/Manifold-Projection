#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

TEXT_DATASETS=(mnli qqp qnli sst2)
OBFUSCATIONS=(translinkguard tempo soter shadownet LoRO obfuscatune groupcover twinshield arrowcloak LRAO)

for obfus in "${OBFUSCATIONS[@]}"; do
  python code/measure_adp.py --models bert gpt2 --datasets "${TEXT_DATASETS[@]}" --obfus "$obfus"
  python code/measure_adp.py --models gpt2_xl --datasets sst2 --obfus "$obfus"
done
