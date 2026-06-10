import os
import sys
import time
import torch
import datasets
import torchvision
import torchvision.transforms as transforms
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal, getcontext
import timm
from pdb import set_trace as st

TOTAL_BAR_LENGTH = 65.0
term_width = 80
last_time = time.time()
begin_time = last_time

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)
    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()
    # No TTY under Slurm/file redirection: carriage-return progress bars cannot refresh correctly in logs, so print one line per update.
    if not sys.stderr.isatty():
        cur_time = time.time()
        if current == 0:
            last_time = cur_time
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time
        extra = " | " + msg if msg else ""
        sys.stderr.write(
            "[%d/%d] Step: %s | Tot: %s%s\n"
            % (current + 1, total, format_time(step_time), format_time(tot_time), extra)
        )
        sys.stderr.flush()
        return
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stderr.write(" [")
    for i in range(cur_len):
        sys.stderr.write("=")
    sys.stderr.write(">")
    for i in range(rest_len):
        sys.stderr.write(".")
    sys.stderr.write("]")
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)
    msg = "".join(L)
    sys.stderr.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stderr.write(" ")
    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stderr.write("\b")
    sys.stderr.write(" %d/%d " % (current + 1, total))
    if current < total - 1:
        sys.stderr.write("\r")
    else:
        sys.stderr.write("\n")
    sys.stderr.flush()


def prepare_data(dataset_dir, args, size):
    set_seed()
    bs = int(args.bs)
    if args.dataset == "cifar_10" or args.dataset == "cifar_100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if args.dataset == "cifar_10":
            trainset = torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_dir, "cifar_10"),
                train=True,
                download=True,
                transform=transform_train,
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=bs, shuffle=True, num_workers=8
            )
            testset = torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_dir, "cifar_10"),
                train=False,
                download=True,
                transform=transform_test,
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=bs, shuffle=False, num_workers=8
            )
            num_classes = 10
        if args.dataset == "cifar_100":
            trainset = torchvision.datasets.CIFAR100(
                root=os.path.join(dataset_dir, "cifar_100"),
                train=True,
                download=True,
                transform=transform_train,
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=bs, shuffle=True, num_workers=8
            )
            testset = torchvision.datasets.CIFAR100(
                root=os.path.join(dataset_dir, "cifar_100"),
                train=False,
                download=True,
                transform=transform_test,
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=bs, shuffle=False, num_workers=8
            )
            num_classes = 100
    elif args.dataset == "food101":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        trainset = torchvision.datasets.Food101(
            root=os.path.join(dataset_dir, "food101"),
            split="train",
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=8
        )
        testset = torchvision.datasets.Food101(
            root=os.path.join(dataset_dir, "food101"),
            split="test",
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=bs, shuffle=False, num_workers=8
        )
        num_classes = 101
    return trainloader, testloader, num_classes



def prepare_calibration_data(dataset_dir, args, size, subset_ratio):
    bs = int(args.bs)
    if args.dataset == "cifar_10" or args.dataset == "cifar_100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if args.dataset == "cifar_10":
            trainset = torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_dir, "cifar_10"),
                train=True,
                download=True,
                transform=transform_train,
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=bs, shuffle=True, num_workers=0
            )
            subset_size = int(subset_ratio * len(trainset))
            subset_indices = torch.randperm(len(trainset))[:subset_size]
            subset = torch.utils.data.Subset(trainset, subset_indices)
            
        if args.dataset == "cifar_100":
            trainset = torchvision.datasets.CIFAR100(
                root=os.path.join(dataset_dir, "cifar_100"),
                train=True,
                download=True,
                transform=transform_train,
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=bs, shuffle=True, num_workers=0
            )
            subset_size = int(subset_ratio * len(trainset))
            subset_indices = torch.randperm(len(trainset))[:subset_size]
            subset = torch.utils.data.Subset(trainset, subset_indices)
            
    elif args.dataset == "tiny_imgnet":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_dir = os.path.join(dataset_dir, "tiny-imagenet-200", "train")
        trainset = torchvision.datasets.ImageFolder(
            train_dir, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=8
        )
        subset_size = int(subset_ratio * len(trainset))
        subset_indices = torch.randperm(len(trainset))[:subset_size]
        subset = torch.utils.data.Subset(trainset, subset_indices)
        
    elif args.dataset == "food101":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        trainset = torchvision.datasets.Food101(
            root=os.path.join(dataset_dir, "food101"),
            split="train",
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=8
        )
        subset_size = int(subset_ratio * len(trainset))
        subset_indices = torch.randperm(len(trainset))[:subset_size]
        subset = torch.utils.data.Subset(trainset, subset_indices)
    elif args.dataset == "food101":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        trainset = torchvision.datasets.Food101(
            root=os.path.join(dataset_dir, "food101"),
            split="train",
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=8
        )
        subset_size = int(subset_ratio * len(trainset))
        subset_indices = torch.randperm(len(trainset))[:subset_size]
        subset = torch.utils.data.Subset(trainset, subset_indices)
    else:  # error
        raise ValueError("wrong dataset!")
        
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    return subset_loader


def loss1(model):
    loss = 0
    for name, param in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight"  in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            loss += torch.sum(param ** 2)
    return loss

def loss2(model, pre_model):
    loss = 0
    for name, param in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight"  in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            name = name.replace("module.", "")
            pre_data = pre_model.state_dict()[name]
            pre_data = pre_data.to(param.device)
            loss += torch.sum((param - pre_data) ** 2)
    return torch.sqrt(loss+1e-8)

def save_model_weights(model, filedir):
    filepath = os.path.join(filedir, "ckpt.t7")
    torch.save(model.state_dict(), filepath)


def load_finetune_dataloader(save_dir, batch_size=32, shuffle=True):
    set_seed()
    save_path = os.path.join(save_dir, "recover_dataset.pth")
    loaded_dataset = torch.load(save_path)
    finetune_dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=shuffle)
    return finetune_dataloader

def prepare_recover_data(model, trainloader, num_classes, device, save_dir, ratio=0.01):
    set_seed()
    model = model.to(device)
    model.eval()
    class_samples = {i: [] for i in range(num_classes)}
    for inputs, labels in trainloader:
        for input, label in zip(inputs, labels):
            label_value = label.item()
            if label_value in class_samples:
                class_samples[label_value].append(input)
            else:
                print(f"Warning: label {label_value} out of range.")
    selected_inputs = []
    selected_labels = []
    for class_label in class_samples:
        num = int(len(class_samples[class_label]) * ratio)
        samples = random.sample(class_samples[class_label], min(max(num, 1), 5))
        selected_inputs.extend(samples)
        selected_labels.extend([class_label] * len(samples))
    selected_inputs = torch.stack(selected_inputs)
    selected_labels = torch.tensor(selected_labels)

    batch_size = 32  
    selected_inputs = selected_inputs.to(device)
    selected_labels = selected_labels.to(device)
    predicted_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        total_iterations = (len(selected_inputs) + batch_size - 1) // batch_size
        progress_bar = tqdm(range(0, len(selected_inputs), batch_size), total=total_iterations, desc="Processing batches")
        for i in progress_bar:
            batch_inputs = selected_inputs[i:i+batch_size]
            batch_labels = selected_labels[i:i+batch_size]
            batch_outputs = model(batch_inputs)
            
            _, predicted = torch.max(batch_outputs, 1)
            # count correct predictions
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            predicted_labels.append(predicted)
    predicted_labels = torch.cat(predicted_labels, dim=0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    selected_inputs = selected_inputs.cpu()
    predicted_labels = predicted_labels.cpu()
    selected_labels = selected_labels.cpu()
    dataset = TensorDataset(selected_inputs, predicted_labels)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "recover_dataset.pth")
    torch.save(dataset, save_path)  
    print(f"Finetune dataset saved to {save_path}")



def row_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, row in enumerate(model_mat_cpu):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[perm[i]] = model_mat_cpu[i]
            success.append(perm[i])
    for i in range(len(restored_matrix)):
        if i not in success:
            restored_matrix[i] = pre_model_mat_cpu[i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def col_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu.T, pre_model_mat_cpu.T)
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, col in enumerate(model_mat_cpu.T):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[:,perm[i]] = model_mat_cpu[:,i]
            success.append(perm[i])
    for i in range(len(restored_matrix[0])):
        if i not in success:
            restored_matrix[:,i] = pre_model_mat_cpu[:,i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

from scipy.optimize import linear_sum_assignment
def col_restore_perm2(pre_model_mat, model_mat, threshold=0.0, perm = None):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = model_mat_cpu.T @ pre_model_mat_cpu
    row_ind, col_ind = linear_sum_assignment(-similarty_matrix)
    P = np.zeros_like(similarty_matrix)
    P[row_ind, col_ind] = 1
    similarty_matrix = P
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, col in enumerate(model_mat_cpu.T):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[:,perm[i]] = model_mat_cpu[:,i]
            success.append(perm[i])
    for i in range(len(restored_matrix[0])):
        if i not in success:
            restored_matrix[:,i] = pre_model_mat_cpu[:,i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def col_restore_perm_and_scale(pre_model_mat, model_mat, threshold=0.0, perm = None):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    A = model_mat_cpu
    B = pre_model_mat_cpu
    Mab = A.T @ B
    Maa = np.diag(A.T @ A)
    M = Mab**2 / Maa[:, np.newaxis]
    row_ind, col_ind = linear_sum_assignment(-M)
    K = np.zeros_like(M)
    for r, c in zip(row_ind, col_ind):
        K[r, c] = Mab[r, c] / Maa[r]
    restored_matrix = model_mat_cpu @ K
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return restored_matrix  

def restore_low_rank(pre_model_mat, model_mat, r):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    K = pre_model_mat_cpu - model_mat_cpu
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    S_r = np.zeros_like(S)
    S_r[:r] = S[:r]
    K_r = U @ np.diag(S_r) @ Vt
    restored_matrix = model_mat_cpu + K_r
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return restored_matrix

def restore_orthogonal(pre_model_mat, model_mat):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    U, _, Vt = np.linalg.svd(pre_model_mat_cpu.T @ model_mat_cpu)
    K = Vt.T @ U.T
    restored_matrix = model_mat_cpu @ K
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return restored_matrix

def eval(model, testloader, criterion, device):
    model.to(device)
    model.eval() 
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  
        progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc="Evaluating")
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)          
            outputs = model(inputs)
            loss = criterion(outputs, targets)  
            
            test_loss += loss.item()
            _, predicted = outputs.max(1) 
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()       
            avg_loss = test_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix(
                Loss=f"{avg_loss:.3f}", 
                Acc=f"{accuracy:.2f}%",
            )
    avg_loss = test_loss / len(testloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def fix_factor(num, mini=1.0, max=6.0):
    if(num < mini):
        return mini
    elif(num > max):
        return max
    else:
        return num

def init_obfus_model(args, num_classes, obfus):
    set_seed()
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    if obfus == "tsqp":
        if os.path.exists(args.weight_dir_tsqp):
            checkpoint = torch.load(f"{args.weight_dir_tsqp}/ckpt.t7")
        else:
            checkpoint = torch.load(f"{args.weight_dir}/ckpt.t7")
    else:
        checkpoint = torch.load(f"{args.weight_dir}/ckpt.t7")
    model.load_state_dict(checkpoint["model"])
    return model

"""
def load_peft_model(checkpoint, num_classes):
    lora_config = LoraConfig(
        target_modules=["qkv", "proj","fc1", "fc2"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        modules_to_save=["head"], 
    )
    
    base_model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
"""
