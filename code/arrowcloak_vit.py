import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
parser.add_argument("--obfus", default="arrowcloak", type=str, help="obfuscation method")
parser.add_argument("--bs", default=128, type=int, help="batch size")
parser.add_argument("--restore_dir", default="results/arrowcloak_results", type=str, help="restore directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--recover_epochs", default=10, type=int, help="epochs for recovering")
parser.add_argument("--recover_lr", default=1e-3, type=float, help="Learning rate for recovering")
args = parser.parse_args()

model_name = "ViT"

args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}"
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

os.makedirs(args.restore_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)

# number of classes in the dataset
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
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Data
print("==> Preparing data..")
size = 224
trainloader, testloader, num_classes = prepare_data("./data/datasets", args, size)

checkpoint = torch.load(f"{args.weight_dir}/ckpt.t7")
model.load_state_dict(checkpoint["model"])

if not os.path.exists(f"{args.recover_data_dir}/recover_dataset.pth"):
    print("preparing recover data !")
    prepare_recover_data(model, trainloader,  num_classes, device, args.recover_data_dir, ratio=0.01)
recover_dataloader = load_finetune_dataloader(args.recover_data_dir, batch_size=32)
set_seed()
print("recover_data prepared!")


if os.path.exists(f"{args.restore_dir}/ckpt.t7") and False:
    final_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    checkpoint = torch.load(f"{args.restore_dir}/ckpt.t7")
    final_model.load_state_dict(checkpoint)
    restore_loss, restore_acc = eval(final_model, testloader, criterion, device)
    print(f"Final recovered model results:Loss: {restore_loss:.4f} | Accuracy: {restore_acc:.4f}%")
else:
    model = model.to(device)
    
    # copy model to orig_model
    orig_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    orig_model.load_state_dict(model.state_dict())
    orig_model = orig_model.to(device)
    
    obfus_model, permutations, _, _ ,_ = ob_arrowcloak(model)
    
    set_seed()
    init_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    restore_model, _, _, _ = attack_arrowcloak_our(obfus_model, init_model)
    new_restore_model = attack_finetune(restore_model, recover_dataloader, testloader, num_classes, save_path=args.restore_dir, device = device, size=224, epochs=args.recover_epochs, lr=args.recover_lr)
    new_restore_loss, new_restore_acc = eval(new_restore_model, testloader, criterion, device)
    print(f"Results after recovering arrowcloak:Loss: {new_restore_loss:.4f} | Accuracy: {new_restore_acc:.4f}%")
