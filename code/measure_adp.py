import argparse
import copy
import csv
import os
import re
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TEXT_DATASETS = ("mnli", "qqp", "qnli", "sst2")
VIT_DATASETS = ("cifar_10", "cifar_100", "food101")
OBFUS_METHODS = (
    "translinkguard",
    "tempo",
    "soter",
    "shadownet",
    "tsqp",
    "LoRO",
    "LRAO",
    "LRAO+arrowcloak",
    "obfuscatune",
    "groupcover",
    "twinshield",
    "arrowcloak",
)

MODEL_DIR = {
    "bert": "bert",
    "gpt2": "gpt2_base",
    "gpt2_xl": "gpt2_xl",
    "vit": "ViT",
}


def text_num_labels(dataset):
    return 3 if dataset in {"mnli", "mnli-mm"} else 2


def vit_num_classes(dataset):
    if dataset == "cifar_10":
        return 10
    if dataset == "cifar_100":
        return 100
    if dataset == "food101":
        return 101
    raise ValueError(f"Unknown ViT dataset: {dataset}")


def is_attacked_weight(model_name, name, tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        return False
    if model_name == "bert":
        return (
            "query.weight" in name
            or "key.weight" in name
            or "value.weight" in name
            or "output.dense.weight" in name
            or "intermediate.dense.weight" in name
        )
    if model_name in {"gpt2", "gpt2_xl"}:
        return (
            "attn.c_attn.weight" in name
            or "attn.c_proj.weight" in name
            or "mlp.c_fc.weight" in name
            or "mlp.c_proj.weight" in name
        )
    if model_name == "vit":
        return (
            "qkv.weight" in name
            or "attn.proj.weight" in name
            or "mlp.fc1.weight" in name
            or "mlp.fc2.weight" in name
        )
    return False


def split_shapes(model_name, name, shape):
    n, m = shape
    if model_name == "vit" and "qkv.weight" in name and n % 3 == 0:
        return [(n // 3, m), (n // 3, m), (n // 3, m)]
    if model_name in {"gpt2", "gpt2_xl"} and "attn.c_attn.weight" in name and m % 3 == 0:
        return [(n, m // 3), (n, m // 3), (n, m // 3)]
    return [(n, m)]


def packed_qkv(model_name, name):
    return (model_name == "vit" and "qkv.weight" in name) or (
        model_name in {"gpt2", "gpt2_xl"} and "attn.c_attn.weight" in name
    )


def qkv_split_dim(model_name, name):
    if model_name == "vit" and "qkv.weight" in name:
        return 0
    if model_name in {"gpt2", "gpt2_xl"} and "attn.c_attn.weight" in name:
        return 1
    return None


def split_qkv_tensors(model_name, name, wp, wv, ws):
    dim = qkv_split_dim(model_name, name)
    if dim is None:
        return [(name, wp, wv, ws)]
    if wp.shape[dim] % 3 != 0:
        return [(name, wp, wv, ws)]
    return [
        (f"{name}:{tag}", p, v, s)
        for tag, p, v, s in zip(
            ("q", "k", "v"),
            wp.chunk(3, dim=dim),
            wv.chunk(3, dim=dim),
            ws.chunk(3, dim=dim),
        )
    ]


def orthogonal_dof(d):
    return d * (d - 1) // 2


def continuous_param_count_piece(model_name, obfus, name, shape, rank_r, group_size):
    n, m = shape
    if obfus in {"black", "translinkguard"}:
        return 0
    if obfus in {"tsqp", "soter"}:
        return 1
    if obfus == "LoRO":
        return 8 * (n + m)
    if obfus == "LRAO":
        return rank_r * (n + m)
    if obfus == "LRAO+arrowcloak":
        return rank_r * (n + m) + (n + 2 * m if model_name in {"gpt2", "gpt2_xl"} else m + 2 * n)
    if obfus == "obfuscatune":
        return orthogonal_dof(m)
    if obfus == "groupcover":
        return n * group_size
    if obfus == "twinshield":
        return n
    if obfus == "tempo":
        if model_name in {"gpt2", "gpt2_xl"}:
            return m
        return n
    if obfus == "shadownet":
        if model_name in {"gpt2", "gpt2_xl"} and "attn.c_attn.weight" in name:
            return n
        return m
    if obfus == "arrowcloak":
        if model_name in {"gpt2", "gpt2_xl"}:
            return n + 2 * m
        return m + 2 * n
    raise ValueError(f"Unsupported obfuscation method: {obfus}")


def continuous_param_count(model_name, obfus, name, shape, rank_r, group_size):
    """Count continuous obfuscation parameters for one attacked weight.

    Permutations are deliberately excluded. Counts follow the local obfuscation
    implementations in utils/methods*.py.
    """
    pieces = split_shapes(model_name, name, shape)
    return sum(
        continuous_param_count_piece(model_name, obfus, name, (pn, pm), rank_r, group_size)
        for pn, pm in pieces
    )


def load_hf_state(model_class, model_path, num_labels, local_files_only):
    kwargs = {"num_labels": num_labels, "local_files_only": local_files_only}
    try:
        model = model_class.from_pretrained(model_path, use_safetensors=True, **kwargs)
    except TypeError:
        model = model_class.from_pretrained(model_path, **kwargs)
    except Exception:
        model = model_class.from_pretrained(model_path, **kwargs)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return state


def load_text_states(args, model_name, dataset, obfus):
    from transformers import AutoModelForSequenceClassification, GPT2ForSequenceClassification

    num_labels = 2 if model_name == "gpt2_xl" else text_num_labels(dataset)
    model_dir = MODEL_DIR[model_name]
    if model_name == "bert":
        public_model = args.bert_public_model
        model_class = AutoModelForSequenceClassification
    elif model_name == "gpt2":
        public_model = args.gpt2_public_model
        model_class = GPT2ForSequenceClassification
    else:
        public_model = args.gpt2_xl_public_model
        model_class = GPT2ForSequenceClassification

    victim_dir = Path(args.weight_dir) / model_dir / dataset / "final_checkpoint"
    tsqp_dir = Path(args.weight_dir_tsqp) / model_dir / dataset / "final_checkpoint"
    if obfus == "tsqp" and tsqp_dir.exists():
        victim_dir = tsqp_dir

    restore_dir = Path(args.restore_dir) / model_dir / obfus / dataset / args.restore_stage
    public_state = load_hf_state(model_class, public_model, num_labels, args.local_files_only)

    if model_name == "gpt2_xl" and victim_dir.exists():
        victim_state = load_gpt2_xl_victim_state(args, victim_dir, num_labels)
    else:
        victim_state = load_hf_state(model_class, victim_dir, num_labels, args.local_files_only)

    restore_state = load_hf_state(model_class, restore_dir, num_labels, args.local_files_only)
    return public_state, victim_state, restore_state, str(victim_dir), str(restore_dir)


def load_gpt2_xl_victim_state(args, victim_dir, num_labels):
    from peft import LoraConfig
    from safetensors.torch import load_file
    from utils.utils_gpt2_xl import adjust_lora_model

    shard1 = Path(victim_dir) / "model-00001-of-00002.safetensors"
    shard2 = Path(victim_dir) / "model-00002-of-00002.safetensors"
    if not shard1.exists() or not shard2.exists():
        raise FileNotFoundError(f"Missing GPT2-XL safetensor shards under {victim_dir}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_fc", "c_attn", "c_proj"],
    )
    model = adjust_lora_model(
        args.gpt2_xl_public_model,
        lora_config=lora_config,
        num_labels=num_labels,
        weight1=load_file(str(shard1)),
        weight2=load_file(str(shard2)),
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return state


def load_vit_ckpt(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    return {k: v.detach().cpu() for k, v in obj.items() if isinstance(v, torch.Tensor)}


def load_vit_states(args, dataset, obfus):
    import timm

    model_dir = MODEL_DIR["vit"]
    num_classes = vit_num_classes(dataset)
    public_model = timm.create_model(
        args.vit_model,
        pretrained=True,
        num_classes=num_classes,
    )
    public_state = {k: v.detach().cpu() for k, v in public_model.state_dict().items()}
    del public_model

    victim_ckpt = Path(args.weight_dir) / model_dir / dataset / "ckpt.t7"
    tsqp_ckpt = Path(args.weight_dir_tsqp) / model_dir / dataset / "ckpt.t7"
    if obfus == "tsqp" and tsqp_ckpt.exists():
        victim_ckpt = tsqp_ckpt

    restore_ckpt = Path(args.restore_dir) / model_dir / obfus / dataset / args.restore_stage / "ckpt.t7"
    victim_state = load_vit_ckpt(victim_ckpt)
    restore_state = load_vit_ckpt(restore_ckpt)
    return public_state, victim_state, restore_state, str(victim_ckpt), str(restore_ckpt)


def resolve_datasets(model_name, requested):
    if requested:
        return requested
    return VIT_DATASETS if model_name == "vit" else TEXT_DATASETS


def iter_jobs(args):
    models = ("bert", "gpt2", "gpt2_xl", "vit") if "all" in args.models else args.models
    obfus_list = OBFUS_METHODS if "all" in args.obfus else args.obfus
    for model_name in models:
        datasets = resolve_datasets(model_name, args.datasets)
        for dataset in datasets:
            for obfus in obfus_list:
                if obfus == "black":
                    continue
                yield model_name, dataset, obfus


def rank_restore_root(args, model_name, dataset, obfus):
    return Path(args.restore_dir) / MODEL_DIR[model_name] / obfus / dataset


def discover_rank_rs(args, model_name, dataset, obfus):
    root = rank_restore_root(args, model_name, dataset, obfus)
    ranks = []
    if not root.exists():
        raise FileNotFoundError(f"Restore root does not exist: {root}")
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"r(\d+)", child.name)
        if not match:
            continue
        stage_path = child / args.rank_restore_stage
        if stage_path.exists():
            ranks.append(int(match.group(1)))
    if not ranks:
        raise FileNotFoundError(
            f"No r*/{args.rank_restore_stage} directories found under {root}"
        )
    return sorted(ranks)


def iter_rank_configs(args, model_name, dataset, obfus):
    if args.all_rank_r:
        ranks = discover_rank_rs(args, model_name, dataset, obfus)
    elif args.rank_rs:
        ranks = sorted(args.rank_rs)
    else:
        yield args
        return

    for rank in ranks:
        rank_args = copy.copy(args)
        rank_args.rank_r = rank
        rank_args.restore_stage = f"r{rank}/{args.rank_restore_stage}"
        yield rank_args


def adp_for_layer(wp, wv, ws, eps):
    wp = wp.detach().cpu().to(torch.float64).reshape(-1)
    wv = wv.detach().cpu().to(torch.float64).reshape(-1)
    ws = ws.detach().cpu().to(torch.float64).reshape(-1)
    direction = wp - wv
    denom = torch.dot(direction, direction).item()
    if denom <= eps:
        return None, denom
    return torch.dot(ws - wv, direction).item() / denom, denom


def measure_one(args, model_name, dataset, obfus):
    if model_name == "vit":
        states = load_vit_states(args, dataset, obfus)
    else:
        states = load_text_states(args, model_name, dataset, obfus)

    public_state, victim_state, restore_state, victim_path, restore_path = states
    rows = []
    for name, wp in public_state.items():
        if not is_attacked_weight(model_name, name, wp):
            continue
        if name not in victim_state or name not in restore_state:
            continue
        wv = victim_state[name]
        ws = restore_state[name]
        if wp.shape != wv.shape or wp.shape != ws.shape:
            continue

        for layer_name, wp_part, wv_part, ws_part in split_qkv_tensors(model_name, name, wp, wv, ws):
            adp, denom = adp_for_layer(wp_part, wv_part, ws_part, args.eps)
            nm = wp_part.numel()
            k = continuous_param_count_piece(
                model_name,
                obfus,
                name,
                tuple(wp_part.shape),
                args.rank_r,
                args.group_size,
            )
            k_over_nm = k / nm
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "obfus": obfus,
                    "rank_r": args.rank_r,
                    "restore_stage": args.restore_stage,
                    "layer": layer_name,
                    "shape": "x".join(str(x) for x in wp_part.shape),
                    "nm": nm,
                    "k": k,
                    "k_over_nm": k_over_nm,
                    "adp": "" if adp is None else adp,
                    "adp_over_k_over_nm": "" if adp is None or k_over_nm == 0 else adp / k_over_nm,
                    "delta_norm_sq": denom,
                    "victim_path": victim_path,
                    "restore_path": restore_path,
                }
            )
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "obfus",
        "rank_r",
        "restore_stage",
        "layer",
        "shape",
        "nm",
        "k",
        "k_over_nm",
        "adp",
        "adp_over_k_over_nm",
        "delta_norm_sq",
        "victim_path",
        "restore_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure ADP(ws)=<ws-wv,wp-wv>/<wp-wv,wp-wv> and compare it with k/nm.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=["all", "bert", "gpt2", "gpt2_xl", "vit"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--obfus", nargs="+", default=["all"])
    parser.add_argument("--weight_dir", default="results/train_results")
    parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results")
    parser.add_argument("--restore_dir", default="results/our_results")
    parser.add_argument("--restore_stage", default="pre_finetune_checkpoint")
    parser.add_argument("--output", default="results/adp_results/adp.csv")
    parser.add_argument("--rank_r", type=int, default=32)
    parser.add_argument(
        "--rank_rs",
        nargs="+",
        type=int,
        default=None,
        help="measure multiple rank_r values, e.g. --rank_rs 1 2 4 8 16",
    )
    parser.add_argument(
        "--all_rank_r",
        action="store_true",
        help="discover and measure every r*/<rank_restore_stage> under restore_dir",
    )
    parser.add_argument(
        "--rank_restore_stage",
        default="pre_finetune_checkpoint",
        help="subdirectory inside each r{rank} directory used by --rank_rs/--all_rank_r",
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-20)
    parser.add_argument("--bert_public_model", default="bert-base-cased")
    parser.add_argument("--gpt2_public_model", default="gpt2")
    parser.add_argument("--gpt2_xl_public_model", default="gpt2-xl")
    parser.add_argument("--vit_model", default="vit_base_patch16_224.augreg_in21k")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []
    print("=" * 60)
    print("ADP measurement config")
    print("=" * 60)
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")
    print("=" * 60)

    for model_name, dataset, obfus in iter_jobs(args):
        try:
            rank_configs = list(iter_rank_configs(args, model_name, dataset, obfus))
        except Exception as exc:
            msg = f"[ADP] skip model={model_name} dataset={dataset} obfus={obfus}: {exc}"
            if args.strict:
                raise RuntimeError(msg) from exc
            print(msg)
            continue

        for rank_args in rank_configs:
            print(
                "[ADP] measuring "
                f"model={model_name} dataset={dataset} obfus={obfus} "
                f"rank_r={rank_args.rank_r} restore_stage={rank_args.restore_stage}"
            )
            try:
                rows = measure_one(rank_args, model_name, dataset, obfus)
            except Exception as exc:
                msg = (
                    "[ADP] skip "
                    f"model={model_name} dataset={dataset} obfus={obfus} "
                    f"rank_r={rank_args.rank_r}: {exc}"
                )
                if args.strict:
                    raise RuntimeError(msg) from exc
                print(msg)
                continue
            print(f"[ADP]   layers={len(rows)}")
            all_rows.extend(rows)

    output = Path(args.output)
    write_rows(output, all_rows)
    print(f"[ADP] wrote {len(all_rows)} rows to {output}")


if __name__ == "__main__":
    main()
