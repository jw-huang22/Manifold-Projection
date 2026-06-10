import sys
import time
import torch
import os
import datasets
import torchvision
import torchvision.transforms as transforms
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from pdb import set_trace as st
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, GPT2ForSequenceClassification, TrainerCallback
import torch.nn as nn
TOTAL_BAR_LENGTH = 65.0
term_width = 80
last_time = time.time()
begin_time = last_time


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def prepare_data(task,model,validation_key,sentence1_key,sentence2_key, max_length=512):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    dataset = load_dataset("glue", task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[validation_key]
    return train_dataset, eval_dataset, tokenizer

def prepare_recover_data(model, trainset, batch_size, path, ratio=0.01):
    set_seed()
    all_indices = list(range(len(trainset)))
    num_samples = int(len(trainset) * ratio)
    indices = random.sample(all_indices, num_samples)

    recover_data = {key: [] for key in trainset[0].keys() if key != 'label'}
    recover_data['label'] = []
    true_labels = [] 
    
    for idx in indices:
        item = trainset[idx]
        for key in recover_data.keys():
            if key != 'label':
                recover_data[key].append(item[key])
        true_labels.append(item['label']) 

    subset = torch.utils.data.Subset(trainset, indices)
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct, tot = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(outputs, dim=1)
            for i in range(inputs["input_ids"].size(0)):
                predicted_label = predictions[i].cpu().tolist()
                recover_data["label"].append(predicted_label) 
                tot += 1
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1
    accuracy = correct / tot
    print(f"accuracy between predicted labels and true labels: {accuracy:.4f}")
    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)

def loss1_gpt(model):
    loss = 0
    for name, param in model.named_parameters():
        if "attn.c_attn.weight" in name or "attn.c_proj.weight" in name or "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name:
            loss += torch.sum(param ** 2)
    return loss

#Compute the L2 distance between pre_model and model weights
def loss2_gpt(model, pre_model):
    loss = 0
    for name, param in model.named_parameters():
        if "attn.c_attn.weight" in name or "attn.c_proj.weight" in name or "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name:
            name = name.replace("module.", "")
            pre_data = pre_model.state_dict()[name]
            #Move pre_data to the device of param.data
            pre_data = pre_data.to(param.device)
            loss += torch.sum((param - pre_data) ** 2)
    return torch.sqrt(loss+1e-8)

class CustomTrainer_gpt(Trainer):
    def __init__(self, pre_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_model = pre_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss_fn = nn.CrossEntropyLoss()
        loss_1 = 1e-4*loss1_gpt(model)
        loss_2 = 1e-4*loss2_gpt(model, self.pre_model)
        loss = loss_fn(logits, labels)+loss_1-loss_2
        return (loss, outputs) if return_outputs else loss    

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
            ## If similarity is greater than or equal to the threshold, use the row corresponding to perm
            restored_matrix[perm[i]] = model_mat_cpu[i]
            success.append(perm[i])
    for i in range(len(restored_matrix)):
        if i not in success:
            print(f"index{i} is not in success")
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
    max_list = []
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


def _apply_inv_perm_scale(ob_w, inv_perm, inv_scales, axis="col"):
    """Apply inverse permutation and per-row/column scaling to ob_w on GPU to avoid __array__ errors from mixing numpy and CUDA tensors."""
    device, dtype = ob_w.device, ob_w.dtype
    ip = torch.as_tensor(inv_perm, device=device, dtype=torch.long)
    sc = torch.as_tensor(inv_scales, device=device, dtype=dtype)
    if axis == "col":
        return ob_w[:, ip] * sc.view(1, -1)
    return ob_w[ip, :] * sc.view(-1, 1)


def col_restore_perm_our(pre_model_mat, model_mat):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    M = model_mat_cpu.T @ pre_model_mat_cpu
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = row_ind[np.argsort(col_ind)]
    inv_perm = np.argsort(perm)
    return inv_perm


def row_restore_perm_our(pre_model_mat, model_mat):
    """Hungarian recovery under row permutation for GPT-2 TransLinkGuard row shuffling.

    The return value is compatible with argsort usage of the first
    return value from row_restore_perm.
    """
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    M = model_mat_cpu @ pre_model_mat_cpu.T
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = row_ind[np.argsort(col_ind)]
    inv_perm = np.argsort(perm)
    return inv_perm


def col_restore_perm_and_scale(pre_model_mat, model_mat):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    A = model_mat_cpu
    B = pre_model_mat_cpu
    Mab = B.T @ A
    Maa = np.diag(A.T @ A)
    Maa = np.maximum(Maa, 1e-20)
    M = (Mab**2) / Maa[np.newaxis, :]
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = col_ind[np.argsort(row_ind)]
    inv_perm = np.argsort(perm)
    scales = (Mab[row_ind, col_ind] / Maa[col_ind])[np.argsort(row_ind)]
    inv_scales = 1.0 / scales
    return inv_perm, inv_scales


def row_restore_perm_and_scale(pre_model_mat, model_mat):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    A = model_mat_cpu
    B = pre_model_mat_cpu
    Mab = B @ A.T
    Maa = np.diag(A @ A.T)
    Maa = np.maximum(Maa, 1e-20)
    M = (Mab**2) / Maa[np.newaxis, :]
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = col_ind[np.argsort(row_ind)]
    inv_perm = np.argsort(perm)
    scales = (Mab[row_ind, col_ind] / Maa[col_ind])[np.argsort(row_ind)]
    inv_scales = 1.0 / scales
    return inv_perm, inv_scales


def restore_low_rank(pre_model_mat, model_mat, r):
    target_device = model_mat.device
    target_dtype = model_mat.dtype
    model_mat_cpu = model_mat.detach().cpu().to(dtype=torch.float64)
    pre_model_mat_cpu = pre_model_mat.detach().cpu().to(dtype=torch.float64)
    K = pre_model_mat_cpu - model_mat_cpu
    if not torch.isfinite(K).all():
        raise ValueError("restore_low_rank received non-finite values in the weight difference")
    U, S, Vt = torch.linalg.svd(K, full_matrices=False)
    K_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
    restored_matrix = model_mat_cpu + K_r
    restored_matrix = restored_matrix.to(device=target_device, dtype=target_dtype)
    K_r_t = (-K_r).to(dtype=target_dtype)
    return restored_matrix, K_r_t


def restore_orthogonal(pre_model_mat, model_mat):
    target_device = model_mat.device
    target_dtype = model_mat.dtype
    model_mat_cpu = model_mat.detach().cpu().to(dtype=torch.float64)
    pre_model_mat_cpu = pre_model_mat.detach().cpu().to(dtype=torch.float64)
    U, _, Vt = torch.linalg.svd(pre_model_mat_cpu.T @ model_mat_cpu, full_matrices=False)
    K = Vt.T @ U.T
    restored_matrix = (model_mat_cpu @ K).to(device=target_device, dtype=target_dtype)
    K_inv = K.T.to(dtype=target_dtype)
    return restored_matrix, K_inv


def col_restore_perm_new(pre_model_mat, model_mat, threshold=0.0):
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
    return perm, similarty_matrix, restored_matrix

def fix_factor(num, mini=1.0, max=6.0):
    if(num < mini):
        return mini
    elif(num > max):
        return max
    else:
        return num

def init_obfus_model(args, num_labels, obfus):
    if obfus == "tsqp":
        if os.path.exists(args.weight_dir_tsqp):
            model = GPT2ForSequenceClassification.from_pretrained(args.weight_dir_tsqp, num_labels=num_labels, use_safetensors=True)
        else:
            model = GPT2ForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True)
    else:
        model = GPT2ForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True)
    return model
