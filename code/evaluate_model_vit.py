import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HOME", ".cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
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
parser.add_argument("--bs", default=128, type=int, help="batch size")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--output_dir", default="evaluate_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--recover_epochs", default=10, type=int, help="epochs for recovering")
parser.add_argument("--recover_lr", default=2e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--result_of_whitebox_model", default="false", type=str)
parser.add_argument("--result_of_blackbox_model", default="false", type=str)
parser.add_argument("--result_of_obfus_model", default="false", type=str)
parser.add_argument("--result_of_recover_model", default="false", type=str)
parser.add_argument("--result_of_arrowcloak_model", default="false", type=str)
parser.add_argument(
    "--vit_model",
    default="vit_base_patch16_224.orig_in21k",
    type=str,
    help=(
        "timm ViT checkpoint: default IN-21k (same as train_vit.py). "
        "Use vit_base_patch16_224 for IN-1k."
    ),
)
model_name = "ViT"

args = parser.parse_args()

args.output_dir = f"{args.output_dir}/{model_name}/{args.dataset}"
os.makedirs(f"{args.output_dir}", exist_ok=True)


args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

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

set_seed()
print("==> Building model..")
print(f"    backbone: {args.vit_model} (pretrained=True)")
model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Data
print("==> Preparing data..")
size = 224
trainloader, testloader, num_classes = prepare_data("./data/datasets", args, size)




if not os.path.exists(f"{args.recover_data_dir}/recover_dataset.pth"):
    print("preparing recover data !")
    prepare_recover_data(model, trainloader,  num_classes, device, args.recover_data_dir, ratio=0.01)
recover_dataloader = load_finetune_dataloader(args.recover_data_dir, batch_size=32)
set_seed()
print("recover_data prepared!")

if args.result_of_whitebox_model == "true":
    set_seed()
    whitebox_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
    loss, acc = eval(model, testloader, criterion, device)
    print(f"White-box(Whitebox model) evaluation results of {model_name} on {args.dataset}: {acc:.4f}%, {loss:.4f}")
    ## Saved under args.output_dir
    with open(f"{args.output_dir}/whitebox_results.txt", "w") as f:
        f.write(f"White-box(Whitebox model) evaluation results of {model_name} on {args.dataset}: {acc:.4f}%, {loss:.4f}")


if args.result_of_blackbox_model == "true":
    set_seed()
    init_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
    blackbox_model = attack_finetune(init_model, recover_dataloader, testloader, num_classes, save_path=args.output_dir, device = device, size=224, epochs=args.recover_epochs, lr=args.recover_lr)
    blackbox_loss, blackbox_acc = eval(blackbox_model, testloader, criterion, device)
    print(f"Black-box(Blackbox model) evaluation results of {model_name} on {args.dataset}: {blackbox_acc:.4f}%, {blackbox_loss:.4f}")
    with open(f"{args.output_dir}/blackbox_results.txt", "w") as f:
        f.write(f"Black-box(Blackbox model) evaluation results of {model_name} on {args.dataset}: {blackbox_acc:.4f}%, {blackbox_loss:.4f}")

if args.result_of_obfus_model == "true":
    for obfus in ["soter", "tsqp", "translinkguard", "tempo", "shadownet"]:
        set_seed()
        model = init_obfus_model(args, num_classes,obfus)
        init_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
        if obfus == "translinkguard":
            obfus_model, _, _ = ob_translinkguard(model)
        elif obfus == "tsqp":
            obfus_model, _ = ob_tsqp(model)
        elif obfus == "soter":
            obfus_model, _, _ = ob_soter(model, init_model)
        elif obfus == "shadownet":
            obfus_model, _, _ = ob_shadownet(model)
        elif obfus == "tempo":
            obfus_model, _, _ = ob_tempo(model)
        ob_loss, ob_acc = eval(obfus_model, testloader, criterion, device)
        print(f"Obfuscated model(Obfus_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {ob_acc:.4f}%, {ob_loss:.4f}")
        with open(f"{args.output_dir}/obfus_results.txt", "a") as f:
            f.write(f"Obfuscated model(Obfus_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {ob_acc:.4f}%, {ob_loss:.4f}\n")
    
if args.result_of_recover_model == "true":
    for obfus in ["soter", "tsqp", "translinkguard", "tempo", "shadownet"]:
        set_seed()
        recover_dir = f"results/arrowmatch_results/{model_name}/{obfus}/{args.dataset}"
        recover_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
        recover_checkpoint = torch.load(f"{recover_dir}/ckpt.t7")
        recover_model.load_state_dict(recover_checkpoint)
        recover_loss, recover_acc = eval(recover_model, testloader, criterion, device)
        print(f"Recovered model(Recover_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {recover_acc:.4f}%, {recover_loss:.4f}")
        with open(f"{args.output_dir}/recover_results.txt", "a") as f:
            f.write(f"Recovered model(Recover_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {recover_acc:.4f}%, {recover_loss:.4f}\n")

if args.result_of_arrowcloak_model == "true":
    set_seed()
    recover_dir = f"results/arrowcloak_results/{model_name}/arrowcloak/{args.dataset}"
    recover_model = timm.create_model(args.vit_model, pretrained=True, num_classes=num_classes)
    recover_checkpoint = torch.load(f"{recover_dir}/ckpt.t7")
    recover_model.load_state_dict(recover_checkpoint)
    recover_loss, recover_acc = eval(recover_model, testloader, criterion, device)
    print(f"Recovered model(Recover_model) evaluation results of {model_name}-arrowcloak on {args.dataset}: {recover_acc:.4f}%, {recover_loss:.4f}")
    with open(f"{args.output_dir}/arrowcloak_results.txt", "a") as f:
        f.write(f"Recovered model(Recover_model) evaluation results of {model_name}-arrowcloak on {args.dataset}: {recover_acc:.4f}%, {recover_loss:.4f}\n")

print("=====================================================================")
