# Manifold Projection

This is the artifact for the paper **"Manifold Projection: A Unified View to Break and Fix Weight Obfuscation in TEE-Shielded LLM Partition"**.

The code evaluates public-prior attacks against lightweight weight obfuscation for TEE-shielded LLM partitioning. It includes:

- **MPA**: Manifold Projection Attack against per-vector and coupled obfuscation schemes.
- **ADP**: adaptation-directional projection measurements used to explain why lightweight, adaptation-agnostic obfuscation fails.
- **LRAO**: Low-Rank Adaptation-Aware Obfuscation, a retraining-free defense component.
- **ArrowMatch** as the public-prior attack baseline.

## Implementation Base

This repository was developed on top of the public codebase for **Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition Algorithms**: <https://github.com/qsxltss/Game-of-Arrows>

```bibtex
@inproceedings{gameofarrows2025,
  title = {Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition Algorithms},
  booktitle = {USENIX Security Symposium},
  year = {2025}
}
```

We keep the training/evaluation scaffold and ArrowMatch implementation, and add the unified MPA solvers, coupled-scheme attacks, ADP analysis, and LRAO defense experiments.

## Repository Layout

- `code/`: experiment entry points for MPA, ArrowMatch, evaluation, ADP, plotting, and the random-weight sanity test.
- `utils/`: obfuscation, attack, recovery, ADP, and model utility functions.
- `train/`: private-model fine-tuning scripts for BERT, GPT-2, GPT-2 XL, and ViT.
- `scripts/`: single-run wrappers.
- `experiments/`: experiment pipelines for training, MPA, attack/defense comparisons,evaluation, and ADP.
- `install1.sh`, `install2.sh`: pinned setup scripts for the text-model and ViT environments.

## Setup

For BERT, GPT-2, and GPT-2 XL:

```bash
./install1.sh
conda activate mpa-env
```

For ViT:

```bash
./install2.sh
conda activate mpa-vit-env
```

The install scripts create the conda environments and install the pinned dependencies used by the artifact. Set `HF_ENDPOINT` before running scripts if you need a Hugging Face mirror. Set `HF_HOME` if you want model caches outside `.cache/huggingface`.

## Reproducing Experiments

The text models and ViT use different dependency stacks. Run the two groups in separate conda environments.

### Text Models: BERT, GPT-2, GPT-2 XL

Activate the text environment:

```bash
conda activate mpa-env
```

Train private models:

```bash
GPUS=0,1 ./experiments/train_text.sh
```

Remember to rename the last checkpoint to `final_checkpoint`.

Run the MPA experiments:

```bash
GPUS=0,1 ./experiments/run_mpa_text.sh
```

Run the ArrowMatch comparison experiments:

```bash
GPUS=0,1 ./experiments/run_arrowmatch_text.sh
```

Measure ADP:

```bash
./experiments/measure_adp_text.sh
```

### Vision Model: ViT

Activate the ViT environment:

```bash
conda activate mpa-vit-env
```

Train private models:

```bash
GPUS=0 ./experiments/train_vit.sh
```

Run the MPA experiments:

```bash
GPUS=0 ./experiments/run_mpa_vit.sh
```

Run the ArrowMatch comparison experiments:

```bash
GPUS=0 ./experiments/run_arrowmatch_vit.sh
```

Measure ADP:

```bash
./experiments/measure_adp_vit.sh
```


For a single run, use the wrappers in `scripts/`, for example:

```bash
./scripts/mpa.sh --gpus 0,1 --dataset sst2 --obfus arrowcloak --restore_dir results/mpa_results
./scripts/mpa.sh --gpus 0,1 --dataset sst2 --obfus LRAO --rank_r 32 --restore_dir results/mpa_results
./scripts/mpa_vit.sh --gpus 0 --dataset cifar_10 --obfus twinshield --restore_dir results/mpa_results
```

## Experimental Scope

The main scripts cover BERT-Base, GPT-2 Base, GPT-2 XL, and ViT-Base; GLUE-style text tasks `mnli`, `qqp`, `qnli`, `sst2`; and vision tasks `cifar_10`, `cifar_100`, `food101`.

Supported obfuscation names include:

```text
translinkguard, tempo, soter, shadownet, LoRO, obfuscatune, groupcover, twinshield, arrowcloak, LRAO
```

`LRAO` corresponds to the paper's Low-Rank Adaptation-Aware Obfuscation. `--rank_r` controls its retained adaptation rank.
