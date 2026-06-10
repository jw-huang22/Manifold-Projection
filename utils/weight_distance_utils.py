import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TARGET_MODULE_PATTERNS = (
    "query.weight",
    "key.weight",
    "value.weight",
    "output.dense.weight",
    "intermediate.dense.weight",
)


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_data(task, model, validation_key, sentence1_key, sentence2_key, max_length=512):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("glue", task)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[validation_key]
    return train_dataset, eval_dataset, tokenizer


def prepare_recover_data(model, trainset, batch_size, path, ratio=1):
    set_seed()
    all_indices = list(range(len(trainset)))
    num_samples = int(len(trainset) * ratio)
    indices = random.sample(all_indices, num_samples)

    recover_data = {key: [] for key in trainset[0].keys() if key != "label"}
    recover_data["label"] = []
    true_labels = []

    for idx in indices:
        item = trainset[idx]
        for key in recover_data.keys():
            if key != "label":
                recover_data[key].append(item[key])
        true_labels.append(item["label"])

    subset = torch.utils.data.Subset(trainset, indices)

    def collate_fn(batch):
        payload = {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor(
                [item["attention_mask"] for item in batch], dtype=torch.long
            ),
        }
        if "token_type_ids" in batch[0]:
            payload["token_type_ids"] = torch.tensor(
                [item["token_type_ids"] for item in batch], dtype=torch.long
            )
        return payload

    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            predictions = torch.argmax(logits, dim=1)
            for idx in range(inputs["input_ids"].size(0)):
                predicted_label = predictions[idx].cpu().tolist()
                recover_data["label"].append(predicted_label)
                total += 1
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1

    accuracy = correct / total
    print(f"accuracy between predicted labels and true labels: {accuracy:.4f}")

    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)


def _iter_float_state_dict_keys(state_dict_a, state_dict_b):
    keys = []
    for key, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b.get(key)
        if tensor_b is None:
            continue
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            continue
        if tensor_a.shape != tensor_b.shape:
            continue
        if not torch.is_floating_point(tensor_a) or not torch.is_floating_point(tensor_b):
            continue
        keys.append(key)
    return keys


def is_target_module_key(key, target_patterns=TARGET_MODULE_PATTERNS):
    return any(pattern in key for pattern in target_patterns)


def clone_state_dict_to_cpu(state_dict):
    cloned_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cloned_state_dict[key] = value.detach().cpu().clone()
        else:
            cloned_state_dict[key] = value
    return cloned_state_dict


def build_state_dict_with_reference(reference_state_dict, updated_tensors):
    merged_state_dict = clone_state_dict_to_cpu(reference_state_dict)
    for key, value in updated_tensors.items():
        if isinstance(value, torch.Tensor):
            merged_state_dict[key] = value.detach().cpu().clone()
        else:
            merged_state_dict[key] = value
    return merged_state_dict


def interpolate_state_dict(state_dict_a, state_dict_b, alpha):
    interpolated_state_dict = {}
    for key, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b.get(key)
        if (
            tensor_b is not None
            and isinstance(tensor_a, torch.Tensor)
            and isinstance(tensor_b, torch.Tensor)
            and tensor_a.shape == tensor_b.shape
            and torch.is_floating_point(tensor_a)
            and torch.is_floating_point(tensor_b)
        ):
            tensor_a_cpu = tensor_a.detach().cpu()
            tensor_b_cpu = tensor_b.detach().cpu().to(dtype=tensor_a_cpu.dtype)
            interpolated_state_dict[key] = (1 - alpha) * tensor_a_cpu + alpha * tensor_b_cpu
        elif isinstance(tensor_a, torch.Tensor):
            interpolated_state_dict[key] = tensor_a.detach().cpu().clone()
        else:
            interpolated_state_dict[key] = tensor_a
    return interpolated_state_dict


def compute_weight_distance(state_dict_a, state_dict_b, metric="l2", eps=1e-12):
    if metric != "l2":
        raise ValueError(f"Unsupported distance metric: {metric}")

    total = 0.0
    for key in _iter_float_state_dict_keys(state_dict_a, state_dict_b):
        diff = state_dict_a[key].detach().cpu() - state_dict_b[key].detach().cpu()
        total += torch.sum(diff * diff).item()
    return float(np.sqrt(max(total, eps)))


def sample_orthogonal_direction(state_dict_a, state_dict_b, seed=42, eps=1e-12):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = {}
    dot = 0.0
    delta_norm_sq = 0.0

    for key in _iter_float_state_dict_keys(state_dict_a, state_dict_b):
        delta = state_dict_b[key].detach().cpu() - state_dict_a[key].detach().cpu()
        random_tensor = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
        direction[key] = random_tensor
        dot += torch.sum(random_tensor * delta).item()
        delta_norm_sq += torch.sum(delta * delta).item()

    if delta_norm_sq <= eps:
        raise ValueError("The two state_dicts are identical; cannot sample an orthogonal direction.")

    projection_scale = dot / delta_norm_sq
    orthogonal_norm_sq = 0.0
    for key in direction:
        delta = state_dict_b[key].detach().cpu() - state_dict_a[key].detach().cpu()
        orthogonal_tensor = direction[key] - projection_scale * delta
        direction[key] = orthogonal_tensor
        orthogonal_norm_sq += torch.sum(orthogonal_tensor * orthogonal_tensor).item()

    orthogonal_norm = float(np.sqrt(max(orthogonal_norm_sq, eps)))
    for key in direction:
        direction[key] = direction[key] / orthogonal_norm
    return direction


def sample_random_direction(base_state_dict, seed=42, eps=1e-12):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = {}
    norm_sq = 0.0

    for key, base_tensor in base_state_dict.items():
        if not isinstance(base_tensor, torch.Tensor) or not torch.is_floating_point(base_tensor):
            continue
        random_tensor = torch.randn(base_tensor.shape, generator=generator, dtype=base_tensor.dtype)
        direction[key] = random_tensor
        norm_sq += torch.sum(random_tensor * random_tensor).item()

    norm = float(np.sqrt(max(norm_sq, eps)))
    for key in direction:
        direction[key] = direction[key] / norm
    return direction


def apply_direction_to_state_dict(base_state_dict, direction, radius):
    updated_state_dict = {}
    for key, base_tensor in base_state_dict.items():
        if key in direction:
            updated_state_dict[key] = base_tensor.detach().cpu() + radius * direction[key].to(
                dtype=base_tensor.dtype
            )
        elif isinstance(base_tensor, torch.Tensor):
            updated_state_dict[key] = base_tensor.detach().cpu().clone()
        else:
            updated_state_dict[key] = base_tensor
    return updated_state_dict


def interpolate_target_modules(
    pretrained_state_dict,
    victim_state_dict,
    alpha,
    target_patterns=TARGET_MODULE_PATTERNS,
):
    updated_tensors = {}
    for key, pretrained_tensor in pretrained_state_dict.items():
        victim_tensor = victim_state_dict.get(key)
        if (
            victim_tensor is not None
            and isinstance(pretrained_tensor, torch.Tensor)
            and isinstance(victim_tensor, torch.Tensor)
            and pretrained_tensor.shape == victim_tensor.shape
            and torch.is_floating_point(pretrained_tensor)
            and torch.is_floating_point(victim_tensor)
            and is_target_module_key(key, target_patterns)
        ):
            pre_cpu = pretrained_tensor.detach().cpu()
            vic_cpu = victim_tensor.detach().cpu().to(dtype=pre_cpu.dtype)
            updated_tensors[key] = (1 - alpha) * pre_cpu + alpha * vic_cpu
    return build_state_dict_with_reference(pretrained_state_dict, updated_tensors)


def sample_orthogonal_direction_on_target_modules(
    pretrained_state_dict,
    victim_state_dict,
    seed=42,
    eps=1e-12,
    target_patterns=TARGET_MODULE_PATTERNS,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = {}
    dot = 0.0
    delta_norm_sq = 0.0

    for key in _iter_float_state_dict_keys(pretrained_state_dict, victim_state_dict):
        if not is_target_module_key(key, target_patterns):
            continue
        delta = victim_state_dict[key].detach().cpu() - pretrained_state_dict[key].detach().cpu()
        random_tensor = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
        direction[key] = random_tensor
        dot += torch.sum(random_tensor * delta).item()
        delta_norm_sq += torch.sum(delta * delta).item()

    if delta_norm_sq <= eps:
        raise ValueError("The two state_dicts are identical on target modules; cannot sample an orthogonal direction.")

    projection_scale = dot / delta_norm_sq
    orthogonal_norm_sq = 0.0
    for key in direction:
        delta = victim_state_dict[key].detach().cpu() - pretrained_state_dict[key].detach().cpu()
        orthogonal_tensor = direction[key] - projection_scale * delta
        direction[key] = orthogonal_tensor
        orthogonal_norm_sq += torch.sum(orthogonal_tensor * orthogonal_tensor).item()

    orthogonal_norm = float(np.sqrt(max(orthogonal_norm_sq, eps)))
    for key in direction:
        direction[key] = direction[key] / orthogonal_norm
    return direction


def sample_random_direction_on_target_modules(
    base_state_dict,
    seed=42,
    eps=1e-12,
    target_patterns=TARGET_MODULE_PATTERNS,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = {}

    for key, base_tensor in base_state_dict.items():
        if not is_target_module_key(key, target_patterns):
            continue
        if not isinstance(base_tensor, torch.Tensor) or not torch.is_floating_point(base_tensor):
            continue
        random_tensor = torch.randn(base_tensor.shape, generator=generator, dtype=base_tensor.dtype)
        module_norm = float(torch.norm(random_tensor).item())
        direction[key] = random_tensor / max(module_norm, eps)
    return direction


def compute_target_module_reference_scales(
    pretrained_state_dict,
    victim_state_dict,
    target_patterns=TARGET_MODULE_PATTERNS,
    eps=1e-12,
):
    module_scales = {}
    for key in _iter_float_state_dict_keys(pretrained_state_dict, victim_state_dict):
        if not is_target_module_key(key, target_patterns):
            continue
        delta = victim_state_dict[key].detach().cpu() - pretrained_state_dict[key].detach().cpu()
        module_scales[key] = max(float(torch.norm(delta).item()), eps)
    return module_scales


def apply_direction_with_module_scales(base_state_dict, direction, module_scales, radius_scale):
    updated_state_dict = {}
    for key, base_tensor in base_state_dict.items():
        if key in direction:
            updated_state_dict[key] = base_tensor.detach().cpu() + (
                radius_scale * module_scales[key] * direction[key].to(dtype=base_tensor.dtype)
            )
        elif isinstance(base_tensor, torch.Tensor):
            updated_state_dict[key] = base_tensor.detach().cpu().clone()
        else:
            updated_state_dict[key] = base_tensor
    return updated_state_dict
