import argparse
import csv
import importlib
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, GPT2ForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.weight_distance_utils import (
    apply_direction_to_state_dict,
    apply_direction_with_module_scales,
    compute_weight_distance,
    compute_target_module_reference_scales,
    interpolate_state_dict,
    interpolate_target_modules,
    is_target_module_key,
    prepare_data,
    prepare_recover_data,
    sample_random_direction_on_target_modules,
    sample_orthogonal_direction_on_target_modules,
    set_seed,
)

try:
    import timm
    from utils.methods_vit import attack_finetune as vit_attack_finetune
    from utils.utils_vit import (
        eval as vit_eval,
        load_finetune_dataloader as vit_load_finetune_dataloader,
        prepare_data as vit_prepare_data,
        prepare_recover_data as vit_prepare_recover_data,
    )
except ImportError:
    timm = None
    vit_attack_finetune = None
    vit_eval = None
    vit_load_finetune_dataloader = None
    vit_prepare_data = None
    vit_prepare_recover_data = None


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TEXT_ALL_DATASETS = (
    "mnli",
    "qqp",
    "qnli",
    "sst2",
)

VIT_ALL_DATASETS = (
    "cifar_10",
    "cifar_100",
    "food101",
)

TEXT_MODEL_TARGET_PATTERNS = {
    "bert": (
        "query.weight",
        "key.weight",
        "value.weight",
        "output.dense.weight",
        "intermediate.dense.weight",
    ),
    "gpt2_base": (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ),
}

VIT_TARGET_PATTERNS = (
    "qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def resolve_model_config(raw_model, vit_model):
    normalized = raw_model.lower().replace("_", "-")
    if normalized in {"bert", "bert-base", "bert-base-cased"}:
        return {
            "family": "text",
            "model_name": "bert",
            "pretrained_id": "bert-base-cased" if normalized == "bert" else raw_model,
            "target_patterns": TEXT_MODEL_TARGET_PATTERNS["bert"],
        }
    if normalized in {"gpt2", "gpt2-base"}:
        return {
            "family": "text",
            "model_name": "gpt2_base",
            "pretrained_id": "gpt2",
            "target_patterns": TEXT_MODEL_TARGET_PATTERNS["gpt2_base"],
        }
    if normalized in {"vit", "vit-base", "vit-base-patch16-224"}:
        return {
            "family": "vit",
            "model_name": "ViT",
            "pretrained_id": vit_model,
            "target_patterns": VIT_TARGET_PATTERNS,
        }
    raise ValueError(
        "Unsupported --model. Use bert-base-cased, gpt2/gpt2-base, or vit/vit-base."
    )


def get_all_datasets_for_model(model_family):
    if model_family == "vit":
        return VIT_ALL_DATASETS
    return TEXT_ALL_DATASETS


def build_dataset_command(argv, dataset):
    command = [sys.executable, os.path.abspath(__file__)]
    replaced = False
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--dataset":
            command.extend(["--dataset", dataset])
            index += 2
            replaced = True
            continue
        if item.startswith("--dataset="):
            command.append(f"--dataset={dataset}")
            index += 1
            replaced = True
            continue
        command.append(item)
        index += 1
    if not replaced:
        command.extend(["--dataset", dataset])
    return command


def build_summary_plot_command(model_name, output_dir):
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plot_weight_distance_adp_accuracy.py",
    )
    output_root = os.path.abspath(output_dir)
    if os.path.basename(output_root) == model_name:
        output_root = os.path.dirname(output_root)
    return [
        sys.executable,
        script_path,
        "--model",
        model_name,
        "--results-root",
        output_root,
        "--output-root",
        output_root,
    ]


def parse_float_list(raw_value):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def build_scale_list_from_step(step, eps=1e-12):
    if step <= 0 or step > 1:
        raise ValueError(f"line_scale_step must be in (0, 1], got {step}")

    scales = []
    current = 0.0
    while current < 1.0 - eps:
        scales.append(round(current, 10))
        current += step
    if not scales or abs(scales[-1] - 1.0) > eps:
        scales.append(1.0)
    return scales


def format_ratio_tag(value):
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def get_primary_metric(metrics):
    if "eval_accuracy" in metrics:
        return "eval_accuracy", float(metrics["eval_accuracy"])
    for key, value in metrics.items():
        if key.startswith("eval_") and key != "eval_loss":
            return key, float(value)
    raise ValueError(f"Unable to determine primary metric from metrics: {metrics}")


def load_glue_metric(task_name):
    try:
        evaluate_module = importlib.import_module("evaluate")
        return evaluate_module.load("glue", task_name)
    except ModuleNotFoundError:
        pass

    try:
        from datasets import load_metric
    except ImportError as exc:
        raise ImportError(
            "Neither the `evaluate` package nor `datasets.load_metric` is available."
        ) from exc
    return load_metric("glue", task_name)


def compute_pearson(x_values, y_values, eps=1e-12):
    if len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    x_std = x.std()
    y_std = y.std()
    if x_std <= eps or y_std <= eps:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_spearman(x_values, y_values):
    if len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    x_ranks = np.argsort(np.argsort(x)).astype(np.float64)
    y_ranks = np.argsort(np.argsort(y)).astype(np.float64)
    return compute_pearson(x_ranks, y_ranks)


def compute_target_module_delta_norm(
    pretrained_state_dict,
    victim_state_dict,
    target_patterns,
    eps=1e-12,
):
    total = 0.0
    for key, pretrained_tensor in pretrained_state_dict.items():
        victim_tensor = victim_state_dict.get(key)
        if (
            victim_tensor is None
            or not isinstance(pretrained_tensor, torch.Tensor)
            or not isinstance(victim_tensor, torch.Tensor)
            or pretrained_tensor.shape != victim_tensor.shape
            or not torch.is_floating_point(pretrained_tensor)
            or not torch.is_floating_point(victim_tensor)
            or not is_target_module_key(key, target_patterns)
        ):
            continue
        delta = pretrained_tensor.detach().cpu() - victim_tensor.detach().cpu()
        total += torch.sum(delta * delta).item()
    return float(np.sqrt(max(total, eps)))


def compute_target_module_adp(
    sampled_state_dict,
    pretrained_state_dict,
    victim_state_dict,
    target_patterns,
    eps=1e-12,
):
    numerator = 0.0
    denominator = 0.0
    for key, sampled_tensor in sampled_state_dict.items():
        pretrained_tensor = pretrained_state_dict.get(key)
        victim_tensor = victim_state_dict.get(key)
        if (
            pretrained_tensor is None
            or victim_tensor is None
            or not isinstance(sampled_tensor, torch.Tensor)
            or not isinstance(pretrained_tensor, torch.Tensor)
            or not isinstance(victim_tensor, torch.Tensor)
            or sampled_tensor.shape != pretrained_tensor.shape
            or sampled_tensor.shape != victim_tensor.shape
            or not torch.is_floating_point(sampled_tensor)
            or not torch.is_floating_point(pretrained_tensor)
            or not torch.is_floating_point(victim_tensor)
            or not is_target_module_key(key, target_patterns)
        ):
            continue
        sampled_cpu = sampled_tensor.detach().cpu().to(torch.float64)
        pretrained_cpu = pretrained_tensor.detach().cpu().to(torch.float64)
        victim_cpu = victim_tensor.detach().cpu().to(torch.float64)
        direction = pretrained_cpu - victim_cpu
        numerator += torch.sum((sampled_cpu - victim_cpu) * direction).item()
        denominator += torch.sum(direction * direction).item()
    if denominator <= eps:
        return None
    return numerator / denominator


def summarize_records(records, metric_key):
    if not records:
        return {}

    distances = [record["distance_to_victim"] for record in records]
    metrics = [record["post_attack_metric"] for record in records]
    summary = {
        "num_runs": len(records),
        "metric_key": metric_key,
        "distance_metric": "l2",
        "pearson": compute_pearson(distances, metrics),
        "spearman": compute_spearman(distances, metrics),
    }

    line_records = [record for record in records if record["mode"] == "line"]
    if line_records:
        summary["line"] = []
        for recover_ratio in sorted({record["recover_ratio"] for record in line_records}):
            ratio_records = [
                record for record in line_records if record["recover_ratio"] == recover_ratio
            ]
            for line_scale in sorted({record["radius_scale"] for record in ratio_records}):
                scale_records = [
                    record for record in ratio_records if record["radius_scale"] == line_scale
                ]
                scale_metrics = [record["post_attack_metric"] for record in scale_records]
                scale_pre_metrics = [
                    get_primary_metric(record["pre_attack_metrics"])[1] for record in scale_records
                ]
                summary["line"].append(
                    {
                        "recover_ratio": recover_ratio,
                        "line_scale": line_scale,
                        "alpha": scale_records[0]["alpha"],
                        "distance_to_victim": scale_records[0]["distance_to_victim"],
                        "mean_pre_metric": float(np.mean(scale_pre_metrics)),
                        "mean_metric": float(np.mean(scale_metrics)),
                        "std_metric": float(np.std(scale_metrics)),
                        "num_runs": len(scale_records),
                    }
                )

    offline_records = [record for record in records if record["mode"] == "offline"]
    if offline_records:
        summary["offline"] = []
        for recover_ratio in sorted({record["recover_ratio"] for record in offline_records}):
            ratio_records = [
                record for record in offline_records if record["recover_ratio"] == recover_ratio
            ]
            for radius_scale in sorted({record["radius_scale"] for record in ratio_records}):
                radius_records = [
                    record for record in ratio_records if record["radius_scale"] == radius_scale
                ]
                radius_metrics = [record["post_attack_metric"] for record in radius_records]
                radius_distances = [record["distance_to_victim"] for record in radius_records]
                summary["offline"].append(
                    {
                        "recover_ratio": recover_ratio,
                        "radius_scale": radius_scale,
                        "mean_distance_to_victim": float(np.mean(radius_distances)),
                        "std_distance_to_victim": float(np.std(radius_distances)),
                        "mean_metric": float(np.mean(radius_metrics)),
                        "std_metric": float(np.std(radius_metrics)),
                        "num_runs": len(radius_records),
                    }
                )

    random_records = [record for record in records if record["mode"] == "random"]
    if random_records:
        summary["random"] = []
        for recover_ratio in sorted({record["recover_ratio"] for record in random_records}):
            ratio_records = [
                record for record in random_records if record["recover_ratio"] == recover_ratio
            ]
            for radius_scale in sorted({record["radius_scale"] for record in ratio_records}):
                radius_records = [
                    record for record in ratio_records if record["radius_scale"] == radius_scale
                ]
                radius_metrics = [record["post_attack_metric"] for record in radius_records]
                radius_distances = [record["distance_to_victim"] for record in radius_records]
                summary["random"].append(
                    {
                        "recover_ratio": recover_ratio,
                        "radius_scale": radius_scale,
                        "mean_distance_to_victim": float(np.mean(radius_distances)),
                        "std_distance_to_victim": float(np.std(radius_distances)),
                        "mean_metric": float(np.mean(radius_metrics)),
                        "std_metric": float(np.std(radius_metrics)),
                        "num_runs": len(radius_records),
                    }
                )

    adp_records = [record for record in records if record["mode"] == "adp_control"]
    if adp_records:
        adp_values = [record["constructed_adp"] for record in adp_records]
        adp_metrics = [record["post_attack_metric"] for record in adp_records]
        summary["adp_control"] = {
            "pearson": compute_pearson(adp_values, adp_metrics),
            "spearman": compute_spearman(adp_values, adp_metrics),
            "grid": [],
        }
        for recover_ratio in sorted({record["recover_ratio"] for record in adp_records}):
            ratio_records = [
                record for record in adp_records if record["recover_ratio"] == recover_ratio
            ]
            for adp_value in sorted({record["constructed_adp"] for record in ratio_records}):
                adp_value_records = [
                    record for record in ratio_records if record["constructed_adp"] == adp_value
                ]
                for orthogonal_scale in sorted(
                    {record["orthogonal_scale"] for record in adp_value_records}
                ):
                    cell_records = [
                        record
                        for record in adp_value_records
                        if record["orthogonal_scale"] == orthogonal_scale
                    ]
                    cell_metrics = [record["post_attack_metric"] for record in cell_records]
                    cell_distances = [record["distance_to_victim"] for record in cell_records]
                    measured_adps = [
                        record["measured_adp"]
                        for record in cell_records
                        if record["measured_adp"] is not None
                    ]
                    summary["adp_control"]["grid"].append(
                        {
                            "recover_ratio": recover_ratio,
                            "constructed_adp": adp_value,
                            "orthogonal_scale": orthogonal_scale,
                            "mean_measured_adp": (
                                float(np.mean(measured_adps)) if measured_adps else None
                            ),
                            "std_measured_adp": (
                                float(np.std(measured_adps)) if measured_adps else None
                            ),
                            "mean_distance_to_victim": float(np.mean(cell_distances)),
                            "std_distance_to_victim": float(np.std(cell_distances)),
                            "mean_metric": float(np.mean(cell_metrics)),
                            "std_metric": float(np.std(cell_metrics)),
                            "num_runs": len(cell_records),
                        }
                    )

    adp_curve_records = [record for record in records if record["mode"] == "adp_curve"]
    if adp_curve_records:
        adp_values = [record["constructed_adp"] for record in adp_curve_records]
        adp_metrics = [record["post_attack_metric"] for record in adp_curve_records]
        summary["adp_curve"] = {
            "pearson": compute_pearson(adp_values, adp_metrics),
            "spearman": compute_spearman(adp_values, adp_metrics),
            "points": [],
        }
        for recover_ratio in sorted({record["recover_ratio"] for record in adp_curve_records}):
            ratio_records = [
                record for record in adp_curve_records if record["recover_ratio"] == recover_ratio
            ]
            for adp_value in sorted({record["constructed_adp"] for record in ratio_records}):
                cell_records = [
                    record for record in ratio_records if record["constructed_adp"] == adp_value
                ]
                cell_metrics = [record["post_attack_metric"] for record in cell_records]
                cell_distances = [record["distance_to_victim"] for record in cell_records]
                measured_adps = [
                    record["measured_adp"]
                    for record in cell_records
                    if record["measured_adp"] is not None
                ]
                summary["adp_curve"]["points"].append(
                    {
                        "recover_ratio": recover_ratio,
                        "constructed_adp": adp_value,
                        "orthogonal_scale": 0.0,
                        "mean_measured_adp": (
                            float(np.mean(measured_adps)) if measured_adps else None
                        ),
                        "std_measured_adp": (
                            float(np.std(measured_adps)) if measured_adps else None
                        ),
                        "mean_distance_to_victim": float(np.mean(cell_distances)),
                        "std_distance_to_victim": float(np.std(cell_distances)),
                        "mean_metric": float(np.mean(cell_metrics)),
                        "std_metric": float(np.std(cell_metrics)),
                        "num_runs": len(cell_records),
                    }
                )
    return summary


def make_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, compute_metrics):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def clone_state_dict_to_cpu(model):
    state_dict = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.detach().cpu().clone()
        else:
            state_dict[key] = value
    return state_dict


def load_text_model(model_config, checkpoint_or_model_id, num_labels, tokenizer=None):
    model_cls = (
        GPT2ForSequenceClassification
        if model_config["model_name"] == "gpt2_base"
        else AutoModelForSequenceClassification
    )
    try:
        model = model_cls.from_pretrained(
            checkpoint_or_model_id,
            num_labels=num_labels,
            use_safetensors=True,
        )
    except Exception as exc:
        print(
            f"Warning: failed to load {checkpoint_or_model_id} with safetensors "
            f"({type(exc).__name__}: {exc}). Retrying without safetensors."
        )
        model = model_cls.from_pretrained(
            checkpoint_or_model_id,
            num_labels=num_labels,
            use_safetensors=False,
        )
    if tokenizer is not None and model_config["model_name"] == "gpt2_base":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_state_dict_strict_with_context(model, state_dict, context):
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        raise RuntimeError(
            f"Failed to load state_dict for {context}. "
            f"missing_keys={missing[:20]}, unexpected_keys={unexpected[:20]}"
        ) from exc
    return model


def load_vit_model(model_config, num_classes, checkpoint_dir=None):
    if timm is None:
        raise ImportError("ViT support requires timm and the ViT utility modules.")
    model = timm.create_model(
        model_config["pretrained_id"],
        pretrained=True,
        num_classes=num_classes,
    )
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt.t7")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    return model


def run_attack(
    args,
    run_name,
    init_state_dict,
    train_dataset,
    eval_dataset,
    tokenizer,
    num_labels,
    compute_metrics,
):
    set_seed(args.seed)
    model = load_text_model(args.model_config, args.pretrained_model_id, num_labels, tokenizer)
    load_state_dict_strict_with_context(model, init_state_dict, run_name)
    run_output_dir = os.path.join(args.output_dir, "runs", run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=args.recover_lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.recover_epochs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
    )
    trainer = make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    pre_attack_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.train()
    post_attack_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pre_attack_metrics, post_attack_metrics


def run_vit_attack(
    args,
    run_name,
    init_state_dict,
    recover_dataloader,
    eval_dataloader,
    num_classes,
    criterion,
    device,
):
    set_seed(args.seed)
    model = load_vit_model(args.model_config, num_classes)
    load_state_dict_strict_with_context(model, init_state_dict, run_name)
    run_output_dir = os.path.join(args.output_dir, "runs", run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    pre_loss, pre_acc = vit_eval(model, eval_dataloader, criterion, device)
    recovered_model = vit_attack_finetune(
        model,
        recover_dataloader,
        eval_dataloader,
        num_classes,
        save_path=run_output_dir,
        device=device,
        size=args.image_size,
        epochs=args.recover_epochs,
        lr=args.recover_lr,
        opt=args.opt,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_warmup_ratio=args.lr_scheduler_warmup_ratio,
        weight_decay=args.weight_decay,
    )
    post_loss, post_acc = vit_eval(recovered_model, eval_dataloader, criterion, device)
    del recovered_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        {"eval_loss": pre_loss, "eval_accuracy": pre_acc / 100.0},
        {"eval_loss": post_loss, "eval_accuracy": post_acc / 100.0},
    )


def run_attack_for_family(
    args,
    run_name,
    init_state_dict,
    train_dataset,
    eval_dataset,
    tokenizer,
    num_labels,
    compute_metrics,
):
    if args.model_family == "vit":
        return run_vit_attack(
            args=args,
            run_name=run_name,
            init_state_dict=init_state_dict,
            recover_dataloader=train_dataset,
            eval_dataloader=eval_dataset,
            num_classes=num_labels,
            criterion=compute_metrics,
            device=tokenizer,
        )
    return run_attack(
        args=args,
        run_name=run_name,
        init_state_dict=init_state_dict,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        num_labels=num_labels,
        compute_metrics=compute_metrics,
    )


def plot_line_curve(records, metric_key, output_dir):
    line_records = [record for record in records if record["mode"] == "line"]
    if not line_records:
        return None

    plt.figure(figsize=(8, 5))
    for recover_ratio in sorted({record["recover_ratio"] for record in line_records}):
        ratio_records = sorted(
            [record for record in line_records if record["recover_ratio"] == recover_ratio],
            key=lambda record: record["radius_scale"],
        )
        line_scales = [record["radius_scale"] for record in ratio_records]
        accuracies = [record["post_attack_metric"] for record in ratio_records]
        plt.plot(
            line_scales,
            accuracies,
            marker="o",
            linewidth=2,
            label=f"recover_ratio={recover_ratio}",
        )
    plt.xlabel("line_scale")
    plt.ylabel("accuracy" if metric_key == "eval_accuracy" else metric_key)
    plt.title("Line Scale vs Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "line_scale_vs_accuracy.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def plot_adp_accuracy(records, metric_key, output_dir):
    adp_records = [record for record in records if record["mode"] == "adp_curve"]
    if not adp_records:
        return None

    plt.figure(figsize=(8, 5))
    grouped = []
    for adp_value in sorted({record["constructed_adp"] for record in adp_records}):
        cell_records = [
            record for record in adp_records if record["constructed_adp"] == adp_value
        ]
        grouped.append(
            (
                adp_value,
                float(np.mean([record["post_attack_metric"] for record in cell_records])),
            )
        )

    plt.plot(
        [item[0] for item in grouped],
        [item[1] for item in grouped],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("ADP")
    plt.ylabel("model-stealing accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "constructed_adp_vs_accuracy.pdf")
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()
    return plot_path


def write_adp_grid_csv(records, metric_key, output_dir):
    adp_records = [record for record in records if record["mode"] == "adp_control"]
    if not adp_records:
        return None

    csv_path = os.path.join(output_dir, "adp_control_grid.csv")
    fieldnames = [
        "recover_ratio",
        "constructed_adp",
        "orthogonal_scale",
        "mean_accuracy",
        "std_accuracy",
        "num_runs",
        "mean_measured_adp",
        "std_measured_adp",
        "mean_distance_to_victim",
        "std_distance_to_victim",
        "metric_key",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for recover_ratio in sorted({record["recover_ratio"] for record in adp_records}):
            ratio_records = [
                record for record in adp_records if record["recover_ratio"] == recover_ratio
            ]
            for adp_value in sorted({record["constructed_adp"] for record in ratio_records}):
                adp_value_records = [
                    record for record in ratio_records if record["constructed_adp"] == adp_value
                ]
                for orthogonal_scale in sorted(
                    {record["orthogonal_scale"] for record in adp_value_records}
                ):
                    cell_records = [
                        record
                        for record in adp_value_records
                        if record["orthogonal_scale"] == orthogonal_scale
                    ]
                    metrics = [record["post_attack_metric"] for record in cell_records]
                    distances = [record["distance_to_victim"] for record in cell_records]
                    measured_adps = [
                        record["measured_adp"]
                        for record in cell_records
                        if record["measured_adp"] is not None
                    ]
                    writer.writerow(
                        {
                            "recover_ratio": recover_ratio,
                            "constructed_adp": adp_value,
                            "orthogonal_scale": orthogonal_scale,
                            "mean_accuracy": float(np.mean(metrics)),
                            "std_accuracy": float(np.std(metrics)),
                            "num_runs": len(cell_records),
                            "mean_measured_adp": (
                                float(np.mean(measured_adps)) if measured_adps else ""
                            ),
                            "std_measured_adp": (
                                float(np.std(measured_adps)) if measured_adps else ""
                            ),
                            "mean_distance_to_victim": float(np.mean(distances)),
                            "std_distance_to_victim": float(np.std(distances)),
                            "metric_key": metric_key,
                        }
                    )
    return csv_path


def write_adp_table_csv(records, output_dir):
    adp_records = [record for record in records if record["mode"] == "adp_control"]
    if not adp_records:
        return None

    orthogonal_scales = sorted({record["orthogonal_scale"] for record in adp_records})
    csv_path = os.path.join(output_dir, "adp_control_table.csv")
    fieldnames = ["constructed_adp", "adp"] + [
        f"b={orthogonal_scale:g}" for orthogonal_scale in orthogonal_scales
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for adp_value in sorted({record["constructed_adp"] for record in adp_records}):
            row = {
                "constructed_adp": adp_value,
                "adp": adp_value,
            }
            adp_value_records = [
                record for record in adp_records if record["constructed_adp"] == adp_value
            ]
            for orthogonal_scale in orthogonal_scales:
                cell_records = [
                    record
                    for record in adp_value_records
                    if record["orthogonal_scale"] == orthogonal_scale
                ]
                if cell_records:
                    mean_metric = float(
                        np.mean([record["post_attack_metric"] for record in cell_records])
                    )
                    row[f"b={orthogonal_scale:g}"] = f"{100.0 * mean_metric:.2f}%"
                else:
                    row[f"b={orthogonal_scale:g}"] = ""
            writer.writerow(row)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Weight distance vs fine-tuning attack evaluation")
    parser.add_argument("--dataset", default="sst2", type=str, help="dataset")
    parser.add_argument("--model", default="bert-base-cased", type=str, help="base pretrained model: bert-base-cased, gpt2-base, or vit-base")
    parser.add_argument(
        "--vit_model",
        default="vit_base_patch16_224.orig_in21k",
        type=str,
        help="timm ViT checkpoint used when --model is vit/vit-base",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size for ViT inputs")
    parser.add_argument("--dataset_dir", default="./data/datasets", type=str, help="dataset root for ViT")
    parser.add_argument("--max_length", default=512, type=int, help="max sequence length")
    parser.add_argument(
        "--recover_lr",
        default=None,
        type=float,
        help="learning rate for attack finetune; defaults to 5e-5 for ViT and 1e-5 otherwise",
    )
    parser.add_argument(
        "--recover_epochs",
        default=None,
        type=int,
        help="epochs for attack finetune; defaults to 10 for ViT and 3 otherwise",
    )
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument(
        "--weight_decay",
        default=None,
        type=float,
        help="weight decay; defaults to 0.05 for ViT and 1e-4 otherwise",
    )
    parser.add_argument("--opt", default="adam", choices=["adam", "adamw", "sgd"], help="optimizer for ViT attack finetune")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str, help="LR scheduler for ViT attack finetune")
    parser.add_argument("--lr_scheduler_warmup_ratio", default=0.1, type=float, help="warmup ratio for ViT attack finetune")
    parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids")
    parser.add_argument(
        "--output_dir",
        default="evaluate_weight_distance_results",
        type=str,
        help="output directory",
    )
    parser.add_argument("--weight_dir", default="results/train_results", type=str, help="victim model directory")
    parser.add_argument(
        "--recover_data_dir",
        default="data/recover_data",
        type=str,
        help="recover data directory",
    )
    parser.add_argument(
        "--experiment_mode",
        default="adp",
        choices=[
            "baseline",
            "line",
            "offline",
            "random",
            "adp_control",
            "adp_curve",
            "adp",
            "all",
        ],
        help="which experiments to run",
    )
    parser.add_argument(
        "--line_scales",
        default="0.0,0.25,0.5,0.75,1.0",
        type=str,
        help="comma separated line scales measured from the victim side; 0.0 means victim, 1.0 means farthest point toward pretrained",
    )
    parser.add_argument(
        "--line_scale_step",
        default=None,
        type=float,
        help="if set, ignore --line_scales and generate line scales from 0.0 to 1.0 with this step",
    )
    parser.add_argument(
        "--offline_radii",
        default="0.25,0.5,0.75,1.0",
        type=str,
        help="comma separated radius scales relative to ||victim-pretrain||",
    )
    parser.add_argument(
        "--offline_num_dirs",
        default=5,
        type=int,
        help="number of random orthogonal directions",
    )
    parser.add_argument(
        "--offline_base_alpha",
        default=1.0,
        type=float,
        help="line position used as offline experiment base point; 1.0 means victim model",
    )
    parser.add_argument(
        "--random_radii",
        default="0.25,0.5,0.75,1.0",
        type=str,
        help="comma separated radius scales for unconstrained random directions",
    )
    parser.add_argument(
        "--random_num_dirs",
        default=3,
        type=int,
        help="number of unconstrained random directions",
    )
    parser.add_argument(
        "--random_base_alpha",
        default=1.0,
        type=float,
        help="line position used as random-direction experiment base point; 1.0 means victim model",
    )
    parser.add_argument(
        "--adp_values",
        default="0.0,0.25,0.5,0.75,1.0",
        type=str,
        help="comma separated constructed ADP values a_j; 0.0 means victim, 1.0 means public",
    )
    parser.add_argument(
        "--orthogonal_scales",
        default="0.0,0.25,0.5,0.75,1.0",
        type=str,
        help="comma separated orthogonal residual scales b_j",
    )
    parser.add_argument(
        "--orthogonal_num_dirs",
        default=10,
        type=int,
        help="number of orthogonal random directions per nonzero b_j",
    )
    parser.add_argument(
        "--adp_curve_step",
        default=0.05,
        type=float,
        help="step size for the extra b=0 ADP-accuracy curve",
    )
    parser.add_argument(
        "--recover_ratio",
        default=0.01,
        type=float,
        help="recover data ratio sampled from the training split",
    )
    parser.add_argument(
        "--recover_ratios",
        default=None,
        type=str,
        help="comma separated recover ratios; if set, runs all listed ratios and ignores --recover_ratio",
    )
    parser.add_argument(
        "--distance_metric",
        default="l2",
        choices=["l2"],
        help="distance metric between weights",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="when --dataset all is used, keep running remaining datasets after one fails",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_config = resolve_model_config(args.model, args.vit_model)
    args.model_config = model_config
    args.model_family = model_config["family"]
    args.pretrained_model_id = model_config["pretrained_id"]
    target_patterns = model_config["target_patterns"]
    model_name = model_config["model_name"]
    if args.recover_lr is None:
        args.recover_lr = 5e-5 if args.model_family == "vit" else 1e-5
    if args.recover_epochs is None:
        args.recover_epochs = 10 if args.model_family == "vit" else 3
    if args.weight_decay is None:
        args.weight_decay = 0.05 if args.model_family == "vit" else 1e-4
    print(
        "Attack finetune config: "
        f"epochs={args.recover_epochs}, lr={args.recover_lr}, "
        f"weight_decay={args.weight_decay}, opt={args.opt}, "
        f"scheduler={args.lr_scheduler_type}, warmup_ratio={args.lr_scheduler_warmup_ratio}"
    )
    if args.model_family == "vit" and timm is None:
        raise ImportError("ViT support requires timm and the repository ViT utility modules.")

    if args.dataset.lower() == "all":
        datasets_to_run = get_all_datasets_for_model(args.model_family)
        print(
            f"Running all datasets for {model_name}: "
            f"{', '.join(datasets_to_run)}"
        )
        failed_datasets = []
        for dataset in datasets_to_run:
            command = build_dataset_command(sys.argv[1:], dataset)
            print("=" * 80)
            print(f"Starting dataset={dataset}")
            print("Command:", " ".join(command))
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                message = (
                    f"Dataset {dataset} failed with exit code {completed.returncode}."
                )
                if args.continue_on_error:
                    print(message)
                    failed_datasets.append(dataset)
                    continue
                raise RuntimeError(message)
        print("=" * 80)
        print(f"Finished all datasets for {model_name}.")
        if failed_datasets:
            print(
                "Skipping summary plot because these datasets failed: "
                f"{', '.join(failed_datasets)}"
            )
        else:
            plot_command = build_summary_plot_command(model_name, args.output_dir)
            print("Generating summary plot:", " ".join(plot_command))
            completed = subprocess.run(plot_command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Summary plot failed with exit code {completed.returncode}."
                )
        return

    args.output_dir = f"{args.output_dir}/{model_name}/{args.dataset}"
    if args.model_family == "vit":
        args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}"
    else:
        args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}/final_checkpoint"
    args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.recover_data_dir, exist_ok=True)

    set_seed(args.seed)
    if args.model_family == "vit":
        if args.dataset == "cifar_10":
            num_labels = 10
        elif args.dataset == "cifar_100":
            num_labels = 100
        elif args.dataset == "food101":
            num_labels = 101
        elif args.dataset == "pretrained":
            num_labels = 1000
        else:
            raise ValueError("ViT supports cifar_10, cifar_100, food101, or pretrained.")

        print("Preparing ViT data..")
        trainset, evalset, _ = vit_prepare_data(args.dataset_dir, args, args.image_size)
        tokenizer = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        compute_metrics = nn.CrossEntropyLoss()

        print("Loading ViT victim and pretrained models..")
        victim_model = load_vit_model(model_config, num_labels, args.weight_dir)
        set_seed(args.seed)
        pretrained_model = load_vit_model(model_config, num_labels)
    else:
        actual_task = args.dataset
        num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
        validation_key = "validation_matched" if args.dataset == "mnli" else "validation"
        sentence1_key, sentence2_key = TASK_TO_KEYS[args.dataset]

        print("Preparing data..")
        trainset, evalset, tokenizer = prepare_data(
            actual_task,
            args.pretrained_model_id,
            validation_key,
            sentence1_key,
            sentence2_key,
            args.max_length,
        )
        if model_name == "gpt2_base" and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading metric..")
        metric = load_glue_metric(actual_task)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        print("Loading victim and pretrained models..")
        victim_model = load_text_model(model_config, args.weight_dir, num_labels, tokenizer)
        pretrained_model = load_text_model(
            model_config,
            args.pretrained_model_id,
            num_labels,
            tokenizer,
        )

    victim_state_dict = clone_state_dict_to_cpu(victim_model)
    pretrained_state_dict = clone_state_dict_to_cpu(pretrained_model)

    recover_ratios = (
        parse_float_list(args.recover_ratios)
        if args.recover_ratios is not None
        else [args.recover_ratio]
    )

    base_distance = compute_weight_distance(
        pretrained_state_dict,
        victim_state_dict,
        metric=args.distance_metric,
    )
    print(f"Base distance ({args.distance_metric}) between pretrained and victim: {base_distance:.6f}")
    target_delta_norm = compute_target_module_delta_norm(
        pretrained_state_dict,
        victim_state_dict,
        target_patterns,
    )
    print(f"Target-module ||Delta W||_F: {target_delta_norm:.6f}")

    results = []

    for recover_ratio in recover_ratios:
        ratio_tag = format_ratio_tag(recover_ratio)
        recover_data_path = os.path.join(
            args.recover_data_dir,
            f"recover_data_ratio_{ratio_tag}.json",
        )
        if args.model_family == "vit":
            ratio_recover_dir = os.path.join(args.recover_data_dir, f"ratio_{ratio_tag}")
            if not os.path.exists(os.path.join(ratio_recover_dir, "recover_dataset.pth")):
                vit_prepare_recover_data(
                    victim_model,
                    trainset,
                    num_labels,
                    tokenizer,
                    ratio_recover_dir,
                    ratio=recover_ratio,
                )
            recover_dataset = vit_load_finetune_dataloader(
                ratio_recover_dir,
                batch_size=args.bs,
            )
        else:
            if not os.path.exists(recover_data_path):
                prepare_recover_data(victim_model, trainset, args.bs, recover_data_path, ratio=recover_ratio)
            recover_dataset = load_dataset("json", data_files=recover_data_path)["train"]
        print(f"recover_data prepared for ratio={recover_ratio}!")

        if args.experiment_mode in {"adp_control", "adp", "all"}:
            adp_values = parse_float_list(args.adp_values)
            orthogonal_scales = parse_float_list(args.orthogonal_scales)
            for adp_value in adp_values:
                base_alpha = 1.0 - adp_value
                adp_base_state_dict = interpolate_target_modules(
                    pretrained_state_dict,
                    victim_state_dict,
                    base_alpha,
                    target_patterns=target_patterns,
                )
                for orthogonal_scale in orthogonal_scales:
                    if orthogonal_scale == 0.0:
                        direction_ids = [None]
                    else:
                        direction_ids = list(range(args.orthogonal_num_dirs))

                    for direction_id in direction_ids:
                        if direction_id is None:
                            sampled_state_dict = adp_base_state_dict
                            direction_seed = None
                        else:
                            direction_seed = args.seed + 20000 + direction_id
                            direction = sample_orthogonal_direction_on_target_modules(
                                pretrained_state_dict,
                                victim_state_dict,
                                seed=direction_seed,
                                target_patterns=target_patterns,
                            )
                            sampled_state_dict = apply_direction_to_state_dict(
                                adp_base_state_dict,
                                direction,
                                orthogonal_scale * target_delta_norm,
                            )

                        distance_to_victim = compute_weight_distance(
                            sampled_state_dict,
                            victim_state_dict,
                            metric=args.distance_metric,
                        )
                        measured_adp = compute_target_module_adp(
                            sampled_state_dict,
                            pretrained_state_dict,
                            victim_state_dict,
                            target_patterns,
                        )
                        dir_tag = "none" if direction_id is None else str(direction_id)
                        run_name = (
                            f"ratio_{ratio_tag}_adp_{str(adp_value).replace('.', '_')}"
                            f"_b_{str(orthogonal_scale).replace('.', '_')}_dir_{dir_tag}"
                        )
                        measured_adp_text = (
                            "none" if measured_adp is None else f"{measured_adp:.6f}"
                        )
                        print(
                            f"Running ADP control recover_ratio={recover_ratio}, "
                            f"a={adp_value}, b={orthogonal_scale}, dir={dir_tag}, "
                            f"measured_adp={measured_adp_text}, "
                            f"distance={distance_to_victim:.6f}"
                        )
                        pre_attack_metrics, post_attack_metrics = run_attack_for_family(
                            args=args,
                            run_name=run_name,
                            init_state_dict=sampled_state_dict,
                            train_dataset=recover_dataset,
                            eval_dataset=evalset,
                            tokenizer=tokenizer,
                            num_labels=num_labels,
                            compute_metrics=compute_metrics,
                        )
                        metric_key, metric_value = get_primary_metric(post_attack_metrics)
                        results.append(
                            {
                                "mode": "adp_control",
                                "recover_ratio": recover_ratio,
                                "constructed_adp": adp_value,
                                "measured_adp": measured_adp,
                                "orthogonal_scale": orthogonal_scale,
                                "orthogonal_radius_abs": orthogonal_scale * target_delta_norm,
                                "direction_id": direction_id,
                                "direction_seed": direction_seed,
                                "distance_to_victim": distance_to_victim,
                                "pre_attack_metrics": pre_attack_metrics,
                                "post_attack_metrics": post_attack_metrics,
                                "post_attack_metric_key": metric_key,
                                "post_attack_metric": metric_value,
                            }
                        )

        if args.experiment_mode in {"adp_curve", "adp", "all"}:
            adp_curve_values = build_scale_list_from_step(args.adp_curve_step)
            for adp_value in adp_curve_values:
                base_alpha = 1.0 - adp_value
                sampled_state_dict = interpolate_target_modules(
                    pretrained_state_dict,
                    victim_state_dict,
                    base_alpha,
                    target_patterns=target_patterns,
                )
                distance_to_victim = compute_weight_distance(
                    sampled_state_dict,
                    victim_state_dict,
                    metric=args.distance_metric,
                )
                measured_adp = compute_target_module_adp(
                    sampled_state_dict,
                    pretrained_state_dict,
                    victim_state_dict,
                    target_patterns,
                )
                run_name = (
                    f"ratio_{ratio_tag}_adp_curve_{str(adp_value).replace('.', '_')}"
                )
                measured_adp_text = (
                    "none" if measured_adp is None else f"{measured_adp:.6f}"
                )
                print(
                    f"Running ADP curve recover_ratio={recover_ratio}, "
                    f"a={adp_value}, b=0.0, measured_adp={measured_adp_text}, "
                    f"distance={distance_to_victim:.6f}"
                )
                pre_attack_metrics, post_attack_metrics = run_attack_for_family(
                    args=args,
                    run_name=run_name,
                    init_state_dict=sampled_state_dict,
                    train_dataset=recover_dataset,
                    eval_dataset=evalset,
                    tokenizer=tokenizer,
                    num_labels=num_labels,
                    compute_metrics=compute_metrics,
                )
                metric_key, metric_value = get_primary_metric(post_attack_metrics)
                results.append(
                    {
                        "mode": "adp_curve",
                        "recover_ratio": recover_ratio,
                        "constructed_adp": adp_value,
                        "measured_adp": measured_adp,
                        "orthogonal_scale": 0.0,
                        "orthogonal_radius_abs": 0.0,
                        "direction_id": None,
                        "direction_seed": None,
                        "distance_to_victim": distance_to_victim,
                        "pre_attack_metrics": pre_attack_metrics,
                        "post_attack_metrics": post_attack_metrics,
                        "post_attack_metric_key": metric_key,
                        "post_attack_metric": metric_value,
                    }
                )

        if args.experiment_mode in {"baseline", "line", "all"}:
            if args.experiment_mode == "baseline":
                line_scales = [1.0]
            elif args.line_scale_step is not None:
                line_scales = build_scale_list_from_step(args.line_scale_step)
            else:
                line_scales = parse_float_list(args.line_scales)

            for line_scale in line_scales:
                alpha = 1.0 - line_scale
                sampled_state_dict = interpolate_target_modules(
                    pretrained_state_dict,
                    victim_state_dict,
                    alpha,
                    target_patterns=target_patterns,
                )
                distance_to_victim = compute_weight_distance(
                    sampled_state_dict,
                    victim_state_dict,
                    metric=args.distance_metric,
                )
                run_name = (
                    f"ratio_{ratio_tag}_line_scale_{str(line_scale).replace('.', '_')}"
                )
                print(
                    f"Running line experiment recover_ratio={recover_ratio}, "
                    f"line_scale={line_scale}, alpha={alpha}, "
                    f"distance={distance_to_victim:.6f}"
                )
                pre_attack_metrics, post_attack_metrics = run_attack_for_family(
                    args=args,
                    run_name=run_name,
                    init_state_dict=sampled_state_dict,
                    train_dataset=recover_dataset,
                    eval_dataset=evalset,
                    tokenizer=tokenizer,
                    num_labels=num_labels,
                    compute_metrics=compute_metrics,
                )
                metric_key, metric_value = get_primary_metric(post_attack_metrics)
                results.append(
                    {
                        "mode": "line",
                        "recover_ratio": recover_ratio,
                        "alpha": alpha,
                        "radius_scale": line_scale,
                        "radius_abs": None,
                        "direction_id": None,
                        "distance_to_victim": distance_to_victim,
                        "pre_attack_metrics": pre_attack_metrics,
                        "post_attack_metrics": post_attack_metrics,
                        "post_attack_metric_key": metric_key,
                        "post_attack_metric": metric_value,
                    }
                )

        if args.experiment_mode in {"offline", "all"}:
            offline_radii = parse_float_list(args.offline_radii)
            offline_base_state_dict = interpolate_target_modules(
                pretrained_state_dict,
                victim_state_dict,
                args.offline_base_alpha,
                target_patterns=target_patterns,
            )
            for direction_id in range(args.offline_num_dirs):
                direction_seed = args.seed + direction_id
                direction = sample_orthogonal_direction_on_target_modules(
                    pretrained_state_dict,
                    victim_state_dict,
                    seed=direction_seed,
                    target_patterns=target_patterns,
                )
                for radius_scale in offline_radii:
                    radius_abs = radius_scale * base_distance
                    sampled_state_dict = apply_direction_to_state_dict(
                        offline_base_state_dict,
                        direction,
                        radius_abs,
                    )
                    distance_to_victim = compute_weight_distance(
                        sampled_state_dict,
                        victim_state_dict,
                        metric=args.distance_metric,
                    )
                    run_name = (
                        f"ratio_{ratio_tag}_offline_base_{str(args.offline_base_alpha).replace('.', '_')}"
                        f"_dir_{direction_id}_radius_{str(radius_scale).replace('.', '_')}"
                    )
                    print(
                        f"Running offline experiment recover_ratio={recover_ratio}, "
                        f"base_alpha={args.offline_base_alpha}, dir={direction_id}, "
                        f"radius_scale={radius_scale}, distance={distance_to_victim:.6f}"
                    )
                    pre_attack_metrics, post_attack_metrics = run_attack_for_family(
                        args=args,
                        run_name=run_name,
                        init_state_dict=sampled_state_dict,
                        train_dataset=recover_dataset,
                        eval_dataset=evalset,
                        tokenizer=tokenizer,
                        num_labels=num_labels,
                        compute_metrics=compute_metrics,
                    )
                    metric_key, metric_value = get_primary_metric(post_attack_metrics)
                    results.append(
                        {
                            "mode": "offline",
                            "recover_ratio": recover_ratio,
                            "alpha": args.offline_base_alpha,
                            "radius_scale": radius_scale,
                            "radius_abs": radius_abs,
                            "direction_id": direction_id,
                            "distance_to_victim": distance_to_victim,
                            "pre_attack_metrics": pre_attack_metrics,
                            "post_attack_metrics": post_attack_metrics,
                            "post_attack_metric_key": metric_key,
                            "post_attack_metric": metric_value,
                        }
                    )

        if args.experiment_mode in {"random", "all"}:
            random_radii = parse_float_list(args.random_radii)
            random_base_state_dict = interpolate_target_modules(
                pretrained_state_dict,
                victim_state_dict,
                args.random_base_alpha,
                target_patterns=target_patterns,
            )
            random_module_scales = compute_target_module_reference_scales(
                pretrained_state_dict,
                victim_state_dict,
                target_patterns=target_patterns,
            )
            for direction_id in range(args.random_num_dirs):
                direction_seed = args.seed + 10000 + direction_id
                direction = sample_random_direction_on_target_modules(
                    random_base_state_dict,
                    seed=direction_seed,
                    target_patterns=target_patterns,
                )
                for radius_scale in random_radii:
                    sampled_state_dict = apply_direction_with_module_scales(
                        random_base_state_dict,
                        direction,
                        random_module_scales,
                        radius_scale,
                    )
                    distance_to_victim = compute_weight_distance(
                        sampled_state_dict,
                        victim_state_dict,
                        metric=args.distance_metric,
                    )
                    run_name = (
                        f"ratio_{ratio_tag}_random_base_{str(args.random_base_alpha).replace('.', '_')}"
                        f"_dir_{direction_id}_radius_{str(radius_scale).replace('.', '_')}"
                    )
                    print(
                        f"Running random experiment recover_ratio={recover_ratio}, "
                        f"base_alpha={args.random_base_alpha}, dir={direction_id}, "
                        f"radius_scale={radius_scale}, distance={distance_to_victim:.6f}"
                    )
                    pre_attack_metrics, post_attack_metrics = run_attack_for_family(
                        args=args,
                        run_name=run_name,
                        init_state_dict=sampled_state_dict,
                        train_dataset=recover_dataset,
                        eval_dataset=evalset,
                        tokenizer=tokenizer,
                        num_labels=num_labels,
                        compute_metrics=compute_metrics,
                    )
                    metric_key, metric_value = get_primary_metric(post_attack_metrics)
                    results.append(
                        {
                            "mode": "random",
                            "recover_ratio": recover_ratio,
                            "alpha": args.random_base_alpha,
                            "radius_scale": radius_scale,
                            "radius_abs": None,
                            "direction_id": direction_id,
                            "distance_to_victim": distance_to_victim,
                            "pre_attack_metrics": pre_attack_metrics,
                            "post_attack_metrics": post_attack_metrics,
                            "post_attack_metric_key": metric_key,
                            "post_attack_metric": metric_value,
                        }
                    )

    if not results:
        raise ValueError("No experiment runs were executed. Please check experiment_mode and argument values.")

    metric_key = results[0]["post_attack_metric_key"]
    summary = summarize_records(results, metric_key)
    summary["experiment_mode"] = args.experiment_mode
    summary["base_distance"] = base_distance
    summary["target_delta_norm"] = target_delta_norm
    summary["offline_base_alpha"] = args.offline_base_alpha
    summary["random_base_alpha"] = args.random_base_alpha
    summary["recover_ratios"] = recover_ratios
    if any(record["mode"] == "adp_control" for record in results):
        summary["adp_values"] = parse_float_list(args.adp_values)
        summary["orthogonal_scales"] = parse_float_list(args.orthogonal_scales)
        summary["orthogonal_num_dirs"] = args.orthogonal_num_dirs
    if any(record["mode"] == "adp_curve" for record in results):
        summary["adp_curve_values"] = build_scale_list_from_step(args.adp_curve_step)
        summary["adp_curve_step"] = args.adp_curve_step
    summary["line_scales"] = (
        build_scale_list_from_step(args.line_scale_step)
        if args.line_scale_step is not None
        else parse_float_list(args.line_scales)
    )
    summary["offline_radii"] = parse_float_list(args.offline_radii)
    summary["offline_num_dirs"] = args.offline_num_dirs
    summary["random_radii"] = parse_float_list(args.random_radii)
    summary["random_num_dirs"] = args.random_num_dirs

    details_path = os.path.join(args.output_dir, "weight_distance_details.json")
    summary_path = os.path.join(args.output_dir, "weight_distance_summary.json")
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    line_plot_path = plot_line_curve(results, metric_key, args.output_dir)
    adp_plot_path = plot_adp_accuracy(results, metric_key, args.output_dir)
    adp_csv_path = write_adp_grid_csv(results, metric_key, args.output_dir)
    adp_table_path = write_adp_table_csv(results, args.output_dir)
    if line_plot_path is not None:
        summary["line_plot_path"] = line_plot_path
    if adp_plot_path is not None:
        summary["adp_plot_path"] = adp_plot_path
    if adp_csv_path is not None:
        summary["adp_csv_path"] = adp_csv_path
    if adp_table_path is not None:
        summary["adp_table_path"] = adp_table_path
    if (
        line_plot_path is not None
        or adp_plot_path is not None
        or adp_csv_path is not None
        or adp_table_path is not None
    ):
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved detailed results to {details_path}")
    print(f"Saved summary results to {summary_path}")
    if line_plot_path is not None:
        print(f"Saved line curve plot to {line_plot_path}")
    if adp_plot_path is not None:
        print(f"Saved ADP curve plot to {adp_plot_path}")
    if adp_csv_path is not None:
        print(f"Saved ADP control grid to {adp_csv_path}")
    if adp_table_path is not None:
        print(f"Saved ADP paper table to {adp_table_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
