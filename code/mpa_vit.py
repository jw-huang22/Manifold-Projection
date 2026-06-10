import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HOME", ".cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import pickle
import sys

import numpy as np
import timm
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.methods_vit import *
from utils.utils_vit import *


parser = argparse.ArgumentParser(description="ViT MPA attack")
parser.add_argument("--dataset", default="cifar_10", type=str, help="dataset")
parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--bs", default=64, type=int, help="batch size")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--restore_dir", default="results/mpa_results", type=str, help="restore directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--obfus_dir", default="tmp/obfus_results", type=str, help="obfus directory")
parser.add_argument("--recover_epochs", default=10, type=int, help="epochs for recovering")
parser.add_argument("--recover_lr", default=5e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--opt", default="adam", choices=["adam", "adamw", "sgd"])
parser.add_argument("--lr_scheduler_type", default="linear", type=str)
parser.add_argument("--lr_scheduler_warmup_ratio", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--rank_r", default=8, type=int, help="rank used by LoRO")
parser.add_argument(
    "--groupcover_reuse_recovered",
    action="store_true",
    default=True,
    help="reuse recovered groupcover modules from current restore_dir when available",
)
parser.add_argument(
    "--no_groupcover_reuse_recovered",
    action="store_false",
    dest="groupcover_reuse_recovered",
    help="disable reusing recovered groupcover modules",
)
parser.add_argument(
    "--vit_model",
    default="vit_base_patch16_224.augreg_in21k",
    type=str,
    help="timm ViT checkpoint, aligned with scripts2/vit_trains.sh",
)
parser.add_argument(
    "--model_name",
    default="ViT",
    type=str,
)
parser.add_argument(
    "--recover_data_ratio",
    default=0.01,
    type=float,
)
parser.add_argument(
    "--public_non_target_seed",
    default=43,
    type=int,
    help="seed used to rebuild the public model before copying non-target modules",
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
RUN_HEAD_DIAGNOSTICS = False

model_name = args.model_name
args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}"
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
if "LRAO" in args.obfus:
    args.restore_dir = f"{args.restore_dir}/r{args.rank_r}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"
args.obfus_dir = f"{args.obfus_dir}/{model_name}/{args.obfus}/{args.dataset}"

os.makedirs(args.restore_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.obfus_dir, exist_ok=True)


def get_num_classes(dataset):
    if dataset == "cifar_10":
        return 10
    if dataset == "cifar_100":
        return 100
    if dataset == "food101":
        return 101
    if dataset == "pretrained":
        return 1000
    raise ValueError("Invalid dataset name")


def build_model(num_classes, device=None):
    model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


def load_trained_model(num_classes, device):
    tsqp_ckpt = f"{args.weight_dir_tsqp}/ckpt.t7"
    ckpt_dir = args.weight_dir_tsqp if args.obfus == "tsqp" and os.path.exists(tsqp_ckpt) else args.weight_dir
    checkpoint = torch.load(f"{ckpt_dir}/ckpt.t7", map_location=device)
    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["model"])
    return model


def save_restore_model_pre_finetune(restore_model, restore_dir, attack_extras=None):
    pre_dir = os.path.join(restore_dir, "pre_finetune_checkpoint")
    os.makedirs(pre_dir, exist_ok=True)
    torch.save(restore_model.state_dict(), os.path.join(pre_dir, "ckpt.t7"))
    if attack_extras is not None:
        with open(os.path.join(pre_dir, "attack_extras.pkl"), "wb") as f:
            pickle.dump(attack_extras, f)


def load_restore_model_pre_finetune(restore_dir, device):
    pre_dir = os.path.join(restore_dir, "pre_finetune_checkpoint")
    ckpt_path = os.path.join(pre_dir, "ckpt.t7")
    if not os.path.exists(ckpt_path):
        print(f"No saved recovery checkpoint found: {ckpt_path}")
        return None, {}
    state_dict = torch.load(ckpt_path, map_location=device)
    extras_path = os.path.join(pre_dir, "attack_extras.pkl")
    attack_extras = {}
    if os.path.exists(extras_path):
        with open(extras_path, "rb") as f:
            attack_extras = pickle.load(f)
    print(f"Loaded recovery checkpoint: {ckpt_path}")
    return state_dict, attack_extras


def is_vit_target_weight(name):
    return (
        "qkv.weight" in name
        or "attn.proj.weight" in name
        or "mlp.fc1.weight" in name
        or "mlp.fc2.weight" in name
    )


def copy_public_non_target_modules(restore_model, num_classes, device, seed):
    set_seed(seed)
    public_model = build_model(num_classes, device)
    restore_state = restore_model.state_dict()
    public_state = public_model.state_dict()
    copied = []
    skipped = []

    with torch.no_grad():
        for name, restore_tensor in restore_state.items():
            public_tensor = public_state.get(name)
            if public_tensor is None:
                skipped.append((name, "missing"))
                continue
            if is_vit_target_weight(name):
                continue
            if restore_tensor.shape != public_tensor.shape:
                skipped.append((name, "shape"))
                continue
            restore_tensor.copy_(public_tensor.to(device=restore_tensor.device, dtype=restore_tensor.dtype))
            copied.append(name)

    restore_model.load_state_dict(restore_state)
    print(
        f"Used seed={seed} public model to overwrite non-target modules in restore_model: "
        f"copied={len(copied)}, skipped={len(skipped)}"
    )
    head_copied = [name for name in copied if name.startswith("head.")]
    if head_copied:
        print(f"  copied classifier head params: {head_copied}")
    del public_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return restore_model


def snapshot_checkpoint_dir(checkpoint_dir):
    snapshot = {}
    for root, _, files in os.walk(checkpoint_dir):
        for filename in files:
            path = os.path.join(root, filename)
            stat = os.stat(path)
            snapshot[os.path.relpath(path, checkpoint_dir)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def assert_checkpoint_dir_unchanged(checkpoint_dir, before_snapshot):
    after_snapshot = snapshot_checkpoint_dir(checkpoint_dir)
    if after_snapshot != before_snapshot:
        raise RuntimeError(
            f"pre_finetune_checkpoint was modified unexpectedly: {checkpoint_dir}"
        )
    print(f"pre_finetune_checkpoint was not modified: {checkpoint_dir}")


def to_numpy_leaf(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accum_perm_recovery(p, rp):
    if isinstance(p, dict):
        matched = total = 0
        if not isinstance(rp, dict):
            return matched, total
        for key, value in p.items():
            if key not in rp:
                continue
            m, t = accum_perm_recovery(value, rp[key])
            matched += m
            total += t
        return matched, total
    return int(np.array_equal(to_numpy_leaf(p), to_numpy_leaf(rp))), 1


def rel_err_scaling(a, b):
    errs = []
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return errs
        for key, value in a.items():
            if key in b:
                errs.extend(rel_err_scaling(value, b[key]))
        return errs
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, (float, int)) or isinstance(b, (float, int)) or np.isscalar(a):
        errs.append(abs(float(a) - float(b)) / (abs(float(a)) + 1e-12))
        return errs
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size and av.size == bv.size:
        errs.append(np.mean(np.abs(av - bv) / (np.abs(av) + 1e-12)))
    return errs


def partitions_equal(clusters_a, clusters_b):
    set_a = {frozenset(map(int, c)) for c in clusters_a}
    set_b = {frozenset(map(int, c)) for c in clusters_b}
    return set_a == set_b


def accum_cluster_recovery(clusters_a, clusters_b):
    if isinstance(clusters_a, dict):
        matched = total = 0
        if not isinstance(clusters_b, dict):
            return matched, total
        for key, value in clusters_a.items():
            if key not in clusters_b:
                continue
            m, t = accum_cluster_recovery(value, clusters_b[key])
            matched += m
            total += t
        return matched, total
    return int(partitions_equal(clusters_a, clusters_b)), 1


def rel_err_vector_pairs(a, b):
    errs = []
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return errs
        for key, value in a.items():
            if key in b:
                errs.extend(rel_err_vector_pairs(value, b[key]))
        return errs
    av = np.asarray(to_numpy_leaf(a), dtype=np.float64).ravel()
    bv = np.asarray(to_numpy_leaf(b), dtype=np.float64).ravel()
    if av.size and av.size == bv.size:
        errs.append(np.mean(np.abs(av - bv) / (np.abs(av) + 1e-12)))
    return errs


def print_perm_recovery(name, obfus_perm, restore_perm):
    matched, total = accum_perm_recovery(obfus_perm, restore_perm)
    rate = matched / total if total else 0.0
    print(f"[{name}] permutation-structure recovery rate: {matched}/{total} = {rate:.4f}")


def print_scaling_error(name, obfus_scaling, restore_scaling):
    errs = rel_err_scaling(obfus_scaling, restore_scaling)
    if errs:
        print(f"[{name}] mean relative error of scaling factors: {float(np.mean(errs)):.6e}")
    else:
        print(f"[{name}] mean relative error of scaling factors: N/A")


def print_cluster_recovery(name, obfus_clusters, restore_clusters):
    matched, total = accum_cluster_recovery(obfus_clusters, restore_clusters)
    rate = matched / total if total else 0.0
    print(f"[{name}] cluster-partition recovery rate: {matched}/{total} = {rate:.4f}")


def print_vector_error(name, obfus_vectors, restore_vectors, label):
    errs = rel_err_vector_pairs(obfus_vectors, restore_vectors)
    if errs:
        print(f"[{name}] {label} mean relative error: {float(np.mean(errs)):.6e}")
    else:
        print(f"[{name}] {label} mean relative error: N/A")


def evaluate_and_print(label, model, testloader, criterion, device):
    loss, acc = eval(model, testloader, criterion, device)
    print(f"{label}:Loss: {loss:.4f} | Accuracy: {acc:.4f}%")
    return loss, acc


def print_classifier_head_source(restore_model, public_model, victim_state):
    print("=" * 60)
    print("Classifier head source check")
    print("=" * 60)
    if hasattr(restore_model, "get_classifier"):
        print(f"  classifier module: {restore_model.get_classifier()}")
    elif hasattr(restore_model, "head"):
        print(f"  classifier module: {restore_model.head}")
    restore_state = restore_model.state_dict()
    public_state = public_model.state_dict()
    head_keys = [
        k for k in restore_state.keys()
        if k.startswith("head.") and k in public_state and k in victim_state
    ]
    if not head_keys:
        print("  no head.* parameters found")
    for k in head_keys:
        restore_v = restore_state[k].detach().cpu()
        public_v = public_state[k].detach().cpu()
        victim_v = victim_state[k].detach().cpu()
        restore_public = torch.norm(restore_v - public_v).item()
        restore_victim = torch.norm(restore_v - victim_v).item()
        public_victim = torch.norm(public_v - victim_v).item()
        print(f"  {k}:")
        print(f"    ||restore - public_init|| = {restore_public:.6e}")
        print(f"    ||restore - victim||      = {restore_victim:.6e}")
        print(f"    ||public_init - victim||  = {public_victim:.6e}")
    print("=" * 60)


def build_restore_with_victim_head(restore_model, victim_state, num_classes, device):
    model = build_model(num_classes, device)
    state = restore_model.state_dict()
    model.load_state_dict(state)
    target_state = model.state_dict()
    copied = []
    for k in target_state:
        if k.startswith("head.") and k in victim_state:
            target_state[k].copy_(victim_state[k].to(target_state[k].device))
            copied.append(k)
    model.load_state_dict(target_state)
    print(f"  copied victim head params for diagnostic: {copied}")
    return model


def build_victim_with_public_head(victim_state, public_model, num_classes, device):
    model = build_model(num_classes, device)
    model.load_state_dict(victim_state)
    public_state = public_model.state_dict()
    target_state = model.state_dict()
    copied = []
    for k in target_state:
        if k.startswith("head.") and k in public_state:
            target_state[k].copy_(public_state[k].to(target_state[k].device))
            copied.append(k)
    model.load_state_dict(target_state)
    print(f"  copied public head params onto victim for diagnostic: {copied}")
    return model


def build_victim_with_fresh_random_head(victim_state, num_classes, device, seed=12345):
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        fresh_public = build_model(num_classes, device)

    model = build_model(num_classes, device)
    model.load_state_dict(victim_state)
    fresh_state = fresh_public.state_dict()
    target_state = model.state_dict()
    copied = []
    for k in target_state:
        if k.startswith("head.") and k in fresh_state:
            target_state[k].copy_(fresh_state[k].to(target_state[k].device))
            copied.append(k)
    model.load_state_dict(target_state)
    print(f"  copied fresh random head params onto victim for diagnostic: {copied}")
    return model


def build_victim_with_shuffled_public_head(victim_state, public_model, num_classes, device, seed=12345):
    model = build_model(num_classes, device)
    model.load_state_dict(victim_state)
    public_state = public_model.state_dict()
    target_state = model.state_dict()

    out_features = target_state["head.weight"].shape[0]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    perm = torch.randperm(out_features, generator=rng)

    copied = []
    if "head.weight" in target_state and "head.weight" in public_state:
        target_state["head.weight"].copy_(
            public_state["head.weight"][perm].to(target_state["head.weight"].device)
        )
        copied.append("head.weight")
    if "head.bias" in target_state and "head.bias" in public_state:
        target_state["head.bias"].copy_(
            public_state["head.bias"][perm].to(target_state["head.bias"].device)
        )
        copied.append("head.bias")
    model.load_state_dict(target_state)
    print(f"  copied shuffled public head params onto victim for diagnostic: {copied}")
    print(f"  shuffled public head first 10 class indices: {perm[:10].tolist()}")
    return model


def run_attack(model, num_classes, device, testloader, criterion):
    set_seed()
    init_model = build_model(num_classes, device)
    attack_extras = {}

    if args.obfus == "black":
        restore_model = init_model
        return restore_model, attack_extras

    if args.obfus == "translinkguard":
        obfus_model, permutations, rows = ob_translinkguard(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations = attack_translinkguard_our(obfus_model, init_model, rows)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {"obfus_perm": permutations, "restore_perm": restore_permutations}
        return restore_model, attack_extras

    if args.obfus == "tsqp":
        obfus_model, scaling_factors = ob_tsqp(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_scaling_factors = attack_tsqp(obfus_model, init_model)
        print_scaling_error(args.obfus, scaling_factors, restore_scaling_factors)
        attack_extras = {
            "obfus_scaling_factors": scaling_factors,
            "restore_scaling_factors": restore_scaling_factors,
        }
        return restore_model, attack_extras

    if args.obfus == "soter":
        obfus_model, scaling_factors, replaced_layers = ob_soter(model, init_model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_scaling_factors, layers_pretrained = attack_soter_our(obfus_model, init_model)
        print_scaling_error(args.obfus, scaling_factors, restore_scaling_factors)
        attack_extras = {
            "obfus_scaling_factors": scaling_factors,
            "restore_scaling_factors": restore_scaling_factors,
            "replaced_layers": replaced_layers,
            "layers_pretrained": layers_pretrained,
        }
        return restore_model, attack_extras

    if args.obfus == "tempo":
        obfus_model, permutations, scaling_factors = ob_tempo(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations = attack_tempo(obfus_model, init_model)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {
            "obfus_perm": permutations,
            "restore_perm": restore_permutations,
            "obfus_scaling_factors": scaling_factors,
        }
        return restore_model, attack_extras

    if args.obfus == "shadownet":
        obfus_model, permutations, scaling_factors = ob_shadownet(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations = attack_shadownet_our(obfus_model, init_model)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {
            "obfus_perm": permutations,
            "restore_perm": restore_permutations,
            "obfus_scaling_factors": scaling_factors,
        }
        return restore_model, attack_extras

    if args.obfus == "LoRO":
        obfus_model = ob_LoRO(model, r=args.rank_r, noise=1)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model = attack_LoRO(obfus_model, init_model, r=args.rank_r)
        attack_extras = {"rank_r": args.rank_r}
        return restore_model, attack_extras

    if args.obfus == "LRAO":
        obfus_model, R = ob_LRAO(model, init_model, r=args.rank_r)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model = attack_LRAO(obfus_model, init_model, r=args.rank_r)
        attack_extras = {"rank_r": args.rank_r, "obfus_R": R}
        return restore_model, attack_extras
    
    if args.obfus == "LRAO+shadownet":
        obfus_model, R = ob_LRAO(model, init_model, r=args.rank_r)
        obfus_model, permutations, scaling_factors = ob_shadownet(obfus_model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations = attack_shadownet_our(obfus_model, init_model)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {
            "rank_r": args.rank_r,
            "obfus_R": R,
            "obfus_perm": permutations,
            "restore_perm": restore_permutations,
            "obfus_scaling_factors": scaling_factors,
        }
        return restore_model, attack_extras
    
    if args.obfus == "LRAO+LoRO":
        obfus_model, R = ob_LRAO_LoRO(model, init_model, r=args.rank_r)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model = attack_LoRO(obfus_model, init_model, r=args.rank_r)
        attack_extras = {
            "rank_r": args.rank_r,
            "obfus_R": R,
        }
        return restore_model, attack_extras

    if args.obfus == "LRAO+arrowcloak":
        obfus_model, R = ob_LRAO(model, init_model, r=args.rank_r)
        obfus_model, permutations, masks, factors, weight_factors = ob_arrowcloak(obfus_model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(obfus_model, init_model)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {
            "rank_r": args.rank_r,
            "obfus_R": R,
            "obfus_perm": permutations,
            "restore_perm": restore_permutations,
            "obfus_masks": masks,
            "obfus_factors": factors,
            "obfus_weight_factors": weight_factors,
            "restore_L": restore_L,
            "restore_D": restore_D,
        }
        return restore_model, attack_extras

    if args.obfus == "obfuscatune":
        obfus_model = ob_obfuscatune(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model = attack_obfuscatune(obfus_model, init_model)
        return restore_model, attack_extras

    if args.obfus == "groupcover":
        obfus_model, ob_cluster_index, ob_random_coeff_list, ob_permutation = ob_groupcover(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        recovered_state_dict = None
        recovered_attack_extras = {}
        if args.groupcover_reuse_recovered:
            recovered_state_dict, recovered_attack_extras = load_restore_model_pre_finetune(
                args.restore_dir, device
            )
        restore_model, restore_permutation, restore_cluster_index = attack_groupcover(
            obfus_model,
            init_model,
            size=4,
            recovered_state_dict=recovered_state_dict,
            recovered_attack_extras=recovered_attack_extras,
            reuse_recovered=args.groupcover_reuse_recovered,
        )
        print_perm_recovery(args.obfus, ob_permutation, restore_permutation)
        print_cluster_recovery(args.obfus, ob_cluster_index, restore_cluster_index)
        attack_extras = {
            "obfus_cluster_index": ob_cluster_index,
            "obfus_random_coeff_list": ob_random_coeff_list,
            "obfus_permutation": ob_permutation,
            "restore_permutation": restore_permutation,
            "restore_cluster_index": restore_cluster_index,
        }
        return restore_model, attack_extras

    if args.obfus == "twinshield":
        obfus_model, obfus_permutations, obfus_d = ob_twinshield(model)
        print("TwinShield obfuscated results are stored as packed Wo1/Wo2 tensors; skipping intermediate model evaluation")
        restore_model, restore_permutations, restore_d = attack_twinshield(
            obfus_model, init_model, dataset=args.dataset
        )
        print_perm_recovery(args.obfus, obfus_permutations, restore_permutations)
        print_vector_error(args.obfus, obfus_d, restore_d, "d")
        attack_extras = {
            "obfus_permutations": obfus_permutations,
            "restore_permutations": restore_permutations,
            "obfus_d": obfus_d,
            "restore_d": restore_d,
        }
        return restore_model, attack_extras

    if args.obfus == "arrowcloak":
        obfus_model, permutations, masks, factors, weight_factors = ob_arrowcloak(model)
        evaluate_and_print("Obfuscated model results", obfus_model, testloader, criterion, device)
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(obfus_model, init_model)
        print_perm_recovery(args.obfus, permutations, restore_permutations)
        attack_extras = {
            "obfus_perm": permutations,
            "restore_perm": restore_permutations,
            "restore_L": restore_L,
            "restore_D": restore_D,
            "obfus_masks": masks,
            "obfus_factors": factors,
            "obfus_weight_factors": weight_factors,
        }
        return restore_model, attack_extras

    raise ValueError(
        "Invalid obfuscation method for ViT MPA. Supported: "
        "black, translinkguard, tsqp, soter, tempo, shadownet, LoRO, LRAO, "
        "obfuscatune, groupcover, twinshield, arrowcloak"
    )


set_seed()
num_classes = get_num_classes(args.dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
size = 224

print("=" * 60)
print("Run configuration (for experiments / reproducibility)")
print("=" * 60)
print(f"  argv: {sys.argv!r}")
print(f"  model_name: {model_name}")
print(f"  backbone: {args.vit_model}")
for key in sorted(vars(args)):
    print(f"  {key}: {getattr(args, key)}")
print("=" * 60)

print("==> Preparing data..")
trainloader, testloader, num_classes = prepare_data("./data/datasets", args, size)

print("==> Building victim model..")
model = load_trained_model(num_classes, device)
evaluate_and_print("White-box model results", model, testloader, criterion, device)

if not os.path.exists(f"{args.recover_data_dir}/recover_dataset.pth"):
    print("preparing recover data !")
    prepare_recover_data(model, trainloader, num_classes, device, args.recover_data_dir, ratio=args.recover_data_ratio)
recover_dataloader = load_finetune_dataloader(args.recover_data_dir, batch_size=args.bs)
set_seed()
print("recover_data prepared!")

final_ckpt = f"{args.restore_dir}/ckpt.t7"
pre_finetune_ckpt = f"{args.restore_dir}/pre_finetune_checkpoint/ckpt.t7"
if os.path.exists(pre_finetune_ckpt):
    pre_finetune_dir = f"{args.restore_dir}/pre_finetune_checkpoint"
    pre_finetune_snapshot = snapshot_checkpoint_dir(pre_finetune_dir)
    restore_model = build_model(num_classes, device)
    recovered_state_dict, recovered_attack_extras = load_restore_model_pre_finetune(
        args.restore_dir, device
    )
    restore_model.load_state_dict(recovered_state_dict)
    restore_model = copy_public_non_target_modules(
        restore_model,
        num_classes,
        device,
        seed=args.public_non_target_seed,
    )
    if args.obfus == "twinshield":
        obfus_permutations = recovered_attack_extras.get("obfus_permutations")
        restore_permutations = recovered_attack_extras.get("restore_permutations")
        if obfus_permutations is not None and restore_permutations is not None:
            print_perm_recovery(args.obfus, obfus_permutations, restore_permutations)
        else:
            print("[twinshield] Reused pre_finetune checkpoint, but permutation recovery metadata was not found")

        obfus_d = recovered_attack_extras.get("obfus_d")
        restore_d = recovered_attack_extras.get("restore_d")
        if obfus_d is not None and restore_d is not None:
            print_vector_error(args.obfus, obfus_d, restore_d, "d")
    new_restore_model = attack_finetune(
        restore_model,
        recover_dataloader,
        testloader,
        num_classes,
        save_path=args.restore_dir,
        device=device,
        size=size,
        epochs=args.recover_epochs,
        lr=args.recover_lr,
        opt=args.opt,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_warmup_ratio=args.lr_scheduler_warmup_ratio,
        weight_decay=args.weight_decay,
    )
    new_restore_loss, new_restore_acc = eval(new_restore_model, testloader, criterion, device)
    assert_checkpoint_dir_unchanged(pre_finetune_dir, pre_finetune_snapshot)
    print(f"Final recovered model results:Loss: {new_restore_loss:.4f} | Accuracy: {new_restore_acc:.4f}%")
elif os.path.exists(final_ckpt) and False:
    final_model = build_model(num_classes, device)
    final_model.load_state_dict(torch.load(final_ckpt, map_location=device))
    restore_loss, restore_acc = eval(final_model, testloader, criterion, device)
    print(f"Final recovered model results:Loss: {restore_loss:.4f} | Accuracy: {restore_acc:.4f}%")
else:
    model = model.to(device)
    victim_state_for_head_check = None
    if RUN_HEAD_DIAGNOSTICS:
        victim_state_for_head_check = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
    restore_model, attack_extras = run_attack(model, num_classes, device, testloader, criterion)
    restore_model = restore_model.to(device)
    restore_loss, restore_acc = eval(restore_model, testloader, criterion, device)
    print(f"Pre-finetune recovery results:Loss: {restore_loss:.4f} | Accuracy: {restore_acc:.4f}%")
    if RUN_HEAD_DIAGNOSTICS:
        set_seed()
        public_model_for_head_check = build_model(num_classes, device)
        print_classifier_head_source(
            restore_model, public_model_for_head_check, victim_state_for_head_check
        )
        public_loss, public_acc = eval(public_model_for_head_check, testloader, criterion, device)
        print(f"Public initialization model results:Loss: {public_loss:.4f} | Accuracy: {public_acc:.4f}%")
        restore_with_victim_head = build_restore_with_victim_head(
            restore_model, victim_state_for_head_check, num_classes, device
        )
        victim_head_loss, victim_head_acc = eval(restore_with_victim_head, testloader, criterion, device)
        print(
            f"Pre-finetune recovery results + victim-head diagnostic:Loss: {victim_head_loss:.4f} | "
            f"Accuracy: {victim_head_acc:.4f}%"
        )
        victim_with_public_head = build_victim_with_public_head(
            victim_state_for_head_check, public_model_for_head_check, num_classes, device
        )
        victim_public_head_loss, victim_public_head_acc = eval(
            victim_with_public_head, testloader, criterion, device
        )
        print(
            f"victim backbone + public-head diagnostic:Loss: {victim_public_head_loss:.4f} | "
            f"Accuracy: {victim_public_head_acc:.4f}%"
        )
        victim_with_fresh_head = build_victim_with_fresh_random_head(
            victim_state_for_head_check, num_classes, device
        )
        victim_fresh_head_loss, victim_fresh_head_acc = eval(
            victim_with_fresh_head, testloader, criterion, device
        )
        print(
            f"victim backbone + fresh-random-head diagnostic:Loss: {victim_fresh_head_loss:.4f} | "
            f"Accuracy: {victim_fresh_head_acc:.4f}%"
        )
        victim_with_shuffled_public_head = build_victim_with_shuffled_public_head(
            victim_state_for_head_check, public_model_for_head_check, num_classes, device
        )
        victim_shuffled_head_loss, victim_shuffled_head_acc = eval(
            victim_with_shuffled_public_head, testloader, criterion, device
        )
        print(
            f"victim backbone + shuffled-public-head diagnostic:Loss: {victim_shuffled_head_loss:.4f} | "
            f"Accuracy: {victim_shuffled_head_acc:.4f}%"
        )

    save_restore_model_pre_finetune(restore_model, args.restore_dir, attack_extras)
    new_restore_model = attack_finetune(
        restore_model,
        recover_dataloader,
        testloader,
        num_classes,
        save_path=args.restore_dir,
        device=device,
        size=size,
        epochs=args.recover_epochs,
        lr=args.recover_lr,
        opt=args.opt,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_warmup_ratio=args.lr_scheduler_warmup_ratio,
        weight_decay=args.weight_decay,
    )
    new_restore_loss, new_restore_acc = eval(new_restore_model, testloader, criterion, device)
    print(f"Final recovered model results:Loss: {new_restore_loss:.4f} | Accuracy: {new_restore_acc:.4f}%")
