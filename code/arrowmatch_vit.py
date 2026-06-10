import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HOME", ".cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import argparse
import timm
import torch
from pdb import set_trace as st
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.methods_vit import *
from utils.utils_vit import *
import torch.nn as nn

parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="cifar_10", type=str, help="dataset")
parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--bs", default=128, type=int, help="batch size")
parser.add_argument("--restore_dir", default="results/arrowmatch_results", type=str, help="restore directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--recover_lr", default=5e-6, type=float, help="Learning rate for recovering")
parser.add_argument("--opt", default="adam", choices=["adam", "adamw", "sgd"])
parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
parser.add_argument("--lr_scheduler_warmup_ratio", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--rank_r", default=8, type=int, help="Rank used by LRAO/LoRO obfuscation")
parser.add_argument("--full_finetune", default="true", type=str, help="Whether recovery fine-tuning updates all parameters")
parser.add_argument(
    "--vit_model",
    default="vit_base_patch16_224.orig_in21k",
    type=str,
    help="timm ViT checkpoint, aligned with scripts/mpa_vit.sh",
)
parser.add_argument("--model_name", default="ViT", type=str, help="Model name for logging and output naming")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_name = args.model_name

args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}"
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
if "LRAO" in args.obfus:
    args.restore_dir = f"{args.restore_dir}/r{args.rank_r}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

os.makedirs(args.restore_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)

if(args.dataset == "cifar_10"):
    num_classes = 10
elif(args.dataset == "cifar_100"):
    num_classes = 100
elif(args.dataset == "food101"):
    num_classes = 101
elif(args.dataset == "pretrained"):
    num_classes = 1000
else:
    raise ValueError("Invalid dataset name")

print("=" * 60)
print("Run configuration (for experiments / reproducibility)")
print("=" * 60)
print(f"  argv: {sys.argv!r}")
print(f"  model_name: {model_name}")
print(f"  backbone: {args.vit_model}")
print(f"  num_classes: {num_classes}")
for k in sorted(vars(args)):
    print(f"  {k}: {getattr(args, k)}")
print("=" * 60)

set_seed()
print("==> Building model..")
model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def str2bool(value):
    return str(value).lower() in ("true", "1", "yes", "y")


def load_checkpoint_state(ckpt_path, map_location=None):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model_weights(model, ckpt_path, map_location=None):
    state_dict = load_checkpoint_state(ckpt_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable_params / all_params if all_params else 0.0
    print(
        f"Trainable params: {trainable_params} || "
        f"All params: {all_params} || "
        f"Trainable ratio: {ratio:.4f}%"
    )


def prepare_restore_model_for_finetune(model):
    if str2bool(args.full_finetune):
        print("Recovery fine-tuning mode: full-parameter")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print("Recovery fine-tuning mode: existing trainable parameters")
    print_trainable_parameters(model)
    return model


def to_numpy_leaf(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accum_perm_recovery(obfus_perm, restore_perm):
    if isinstance(obfus_perm, dict):
        matched = total = 0
        if not isinstance(restore_perm, dict):
            return matched, total
        for key, value in obfus_perm.items():
            if key not in restore_perm:
                continue
            m, t = accum_perm_recovery(value, restore_perm[key])
            matched += m
            total += t
        return matched, total
    return int(np.array_equal(to_numpy_leaf(obfus_perm), to_numpy_leaf(restore_perm))), 1


def rel_err_vector_pairs(obfus_vectors, restore_vectors):
    errs = []
    if isinstance(obfus_vectors, dict):
        if not isinstance(restore_vectors, dict):
            return errs
        for key, value in obfus_vectors.items():
            if key in restore_vectors:
                errs.extend(rel_err_vector_pairs(value, restore_vectors[key]))
        return errs
    av = np.asarray(to_numpy_leaf(obfus_vectors), dtype=np.float64).ravel()
    bv = np.asarray(to_numpy_leaf(restore_vectors), dtype=np.float64).ravel()
    if av.size and av.size == bv.size:
        errs.append(np.mean(np.abs(av - bv) / (np.abs(av) + 1e-12)))
    return errs


def print_perm_recovery(name, obfus_perm, restore_perm):
    matched, total = accum_perm_recovery(obfus_perm, restore_perm)
    rate = matched / total if total else 0.0
    print(f"[{name}] permutation-structure recovery rate: {matched}/{total} = {rate:.4f}")


def print_vector_error(name, obfus_vectors, restore_vectors, label):
    errs = rel_err_vector_pairs(obfus_vectors, restore_vectors)
    if errs:
        print(f"[{name}] {label} mean relative error: {float(np.mean(errs)):.6e}")
    else:
        print(f"[{name}] {label} mean relative error: N/A")

# Data
print("==> Preparing data..")
size = 224
trainloader, testloader, num_classes = prepare_data("./data/datasets", args, size)

load_model_weights(model, f"{args.weight_dir}/ckpt.t7", map_location=device)


if not os.path.exists(f"{args.recover_data_dir}/recover_dataset.pth"):
    print("preparing recover data !")
    prepare_recover_data(model, trainloader,  num_classes, device, args.recover_data_dir, ratio=0.01)
recover_dataloader = load_finetune_dataloader(args.recover_data_dir, batch_size=args.bs)
set_seed()
print("recover_data prepared!")

if args.obfus == "tsqp":
    tsqp_ckpt = f"{args.weight_dir_tsqp}/ckpt.t7"
    if os.path.exists(tsqp_ckpt):
        ckpt_path = tsqp_ckpt
    else:
        ckpt_path = f"{args.weight_dir}/ckpt.t7"
else:
    ckpt_path = f"{args.weight_dir}/ckpt.t7"
load_model_weights(model, ckpt_path, map_location=device)


def build_public_vit():
    return timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)


def run_restore_finetune(restore_model):
    restore_model = prepare_restore_model_for_finetune(restore_model)
    new_restore_model = attack_finetune(
        restore_model,
        recover_dataloader,
        testloader,
        num_classes,
        save_path=args.restore_dir,
        device=device,
        size=224,
        epochs=args.recover_epochs,
        lr=args.recover_lr,
        opt=args.opt,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_warmup_ratio=args.lr_scheduler_warmup_ratio,
        weight_decay=args.weight_decay,
    )
    new_restore_loss, new_restore_acc = eval(new_restore_model, testloader, criterion, device)
    print(f"Final recovered model results:Loss: {new_restore_loss:.4f} | Accuracy: {new_restore_acc:.4f}%")


def build_obfus_model(obfus_name, target_model, init_model):
    rank_r = args.rank_r
    if obfus_name == "translinkguard":
        obfus_model, _, rows = ob_translinkguard(target_model)
        attack_meta = rows
    elif obfus_name == "tsqp":
        obfus_model, _ = ob_tsqp(target_model)
        attack_meta = None
    elif obfus_name == "soter":
        obfus_model, _, _ = ob_soter(target_model, init_model)
        attack_meta = None
    elif obfus_name == "tempo":
        obfus_model, _, _ = ob_tempo(target_model)
        attack_meta = None
    elif obfus_name == "shadownet":
        obfus_model, _, _ = ob_shadownet(target_model)
        attack_meta = None
    elif obfus_name == "LoRO":
        obfus_model = ob_LoRO(target_model, r=rank_r, noise=1)
        attack_meta = None
    elif obfus_name == "LRAO":
        obfus_model, _ = ob_LRAO(target_model, init_model, r=rank_r)
        attack_meta = None
    elif obfus_name == "LRAO+arrowcloak":
        obfus_model, _ = ob_LRAO(target_model, init_model, r=rank_r)
        obfus_model, _, _, _, _ = ob_arrowcloak(obfus_model)
        attack_meta = None
    elif obfus_name == "obfuscatune":
        obfus_model = ob_obfuscatune(target_model)
        attack_meta = None
    elif obfus_name == "groupcover":
        obfus_model, _, _, _ = ob_groupcover(target_model)
        attack_meta = None
    elif obfus_name == "twinshield":
        obfus_model, obfus_permutations, obfus_d = ob_twinshield(target_model)
        attack_meta = {
            "obfus_permutations": obfus_permutations,
            "obfus_d": obfus_d,
        }
    elif obfus_name == "arrowcloak":
        obfus_model, _, _, _, _ = ob_arrowcloak(target_model)
        attack_meta = None
    else:
        raise ValueError("Invalid obfuscation method")
    return obfus_model, attack_meta


def attack_obfus_model(obfus_name, obfus_model, init_model, attack_meta):
    if obfus_name == "translinkguard":
        return attack_translinkguard(obfus_model, init_model, attack_meta)
    if obfus_name == "tsqp":
        restore_model, _ = attack_tsqp(obfus_model, init_model)
        return restore_model
    if obfus_name == "soter":
        restore_model, _, _ = attack_soter(obfus_model, init_model)
        return restore_model
    if obfus_name == "tempo":
        restore_model, _ = attack_tempo(obfus_model, init_model)
        return restore_model
    if obfus_name == "shadownet":
        restore_model, _ = attack_shadownet(obfus_model, init_model)
        return restore_model
    if obfus_name == "twinshield":
        restore_model, restore_permutations, restore_d = attack_twinshield(
            obfus_model, init_model, dataset=args.dataset
        )
        print_perm_recovery(
            obfus_name,
            attack_meta["obfus_permutations"],
            restore_permutations,
        )
        print_vector_error(obfus_name, attack_meta["obfus_d"], restore_d, "d")
        return restore_model
    restore_model = attack_arrowcloak(obfus_model, init_model)
    print(f"[arrowmatch] {obfus_name} was not originally covered; using attack_arrowcloak for recovery")
    return restore_model


# obfuscation with TransLinkGuard
if os.path.exists(f"{args.restore_dir}/ckpt.t7") and False:
    final_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
    load_model_weights(final_model, f"{args.restore_dir}/ckpt.t7", map_location=device)
    restore_loss, restore_acc = eval(final_model, testloader, criterion, device)
    print(f"Final recovered model results:Loss: {restore_loss:.4f} | Accuracy: {restore_acc:.4f}%")
else:
    model = model.to(device)
    set_seed()
    init_model = build_public_vit().to(device)
    if args.obfus == "black":
        print("black baseline: no obfuscation, fine-tuning public init model directly")
        restore_model = init_model
    else:
        obfus_model, attack_meta = build_obfus_model(args.obfus, model, init_model)
        if args.obfus != "twinshield":
            ob_loss, ob_acc = eval(obfus_model, testloader, criterion, device)
            print(f"Obfuscated model results:Loss: {ob_loss:.4f} | Accuracy: {ob_acc:.4f}%")
        else:
            print("TwinShield obfuscated results are stored in column-packed form; skipping intermediate model evaluation")
        set_seed()
        restore_model = attack_obfus_model(args.obfus, obfus_model, init_model, attack_meta)
    run_restore_finetune(restore_model)
