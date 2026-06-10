import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils_vit import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from matplotlib import pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings


def _vit_obfus_weight(name):
    return (
        "qkv.weight" in name
        or "attn.proj.weight" in name
        or "mlp.fc1.weight" in name
        or "mlp.fc2.weight" in name
    )

def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            num_rows = w_q.shape[1]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation
            ob_w_q = w_q[:,permutation]
            ob_w_k = w_k[:,permutation]
            ob_w_v = w_v[:,permutation]
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        if "attn.proj.weight" in name:
            w_proj = module.data        
            layer_name = name.rsplit(".", 3)[0]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)    
            ob_proj = w_proj[inv_perm]
            module.data = ob_proj
        if "mlp.fc1.weight" in name:
            w_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            permutation = layer_permutations[layer_name]
            ob_fc1 = w_fc1[:,permutation]
            module.data = ob_fc1
        if "mlp.fc2.weight" in name:
            w_fc2 = module.data
            if(w_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]         
                permutation = layer_permutations[layer_name] 
                ob_fc2 = w_fc2[:,permutation]
                module.data = ob_fc2             
    return model, layer_permutations, rows

def attack_translinkguard(model, pre_model, rows):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            pre_wq, _, _ = pre_model.state_dict()[name].chunk(3, dim=0)
            perm, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wk = ob_wk[:, inv_perm]
            restore_wv = ob_wv[:, inv_perm]
            restore_w = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
            module.data = restore_w
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            restore_wo = ob_wo[perm]
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_fc1 = ob_fc1[:, inv_perm]
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            if(ob_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_fc2 = ob_fc2[:,inv_perm]
                module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data

    return model

def attack_translinkguard_our(model, pre_model, rows):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            pre_wq, _, _ = pre_model.state_dict()[name].chunk(3, dim=0)
            perm1, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            perm, _, restore_wq = col_restore_perm2(pre_wq, ob_wq)
            print(f"layer {layer_name} similarity between the two recovery methods: {np.mean(perm1==perm)}")
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wk = ob_wk[:, inv_perm]
            restore_wv = ob_wv[:, inv_perm]
            restore_w = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
            module.data = restore_w
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            restore_wo = ob_wo[perm]
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_fc1 = ob_fc1[:, inv_perm]
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            if(ob_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_fc2 = ob_fc2[:,inv_perm]
                module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data

    return model, restore_perm

def ob_tsqp(model):
    set_seed() 
    scaling_factors = {}  
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            scale_q = 1 + 5 * torch.rand(1).item()
            scale_k = 1 + 5 * torch.rand(1).item()
            scale_v = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            w_k *= scale_k
            w_v *= scale_v
            module.data = torch.cat([w_q, w_k, w_v], dim=0)
            scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
        elif "attn.proj.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            scale_fc1 = 1 + 5 * torch.rand(1).item()
            w_fc1 *= scale_fc1
            module.data = w_fc1
            scaling_factors[name] = scale_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            scale_fc2 = 1 + 5 * torch.rand(1).item()
            w_fc2 *= scale_fc2
            module.data = w_fc2
            scaling_factors[name] = scale_fc2
    return model, scaling_factors

def attack_tsqp(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_fc2).item()/torch.var(pre_fc2).item()))
            new_w_fc2 = w_fc2/k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_soter(model, pre_model):
    set_seed()
    scaling_factors = {}  
    init_layers = {}
    ## Randomly select some weight blocks and load pre_model weights
    ## ratio is 20%
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            init_layers[name] = module
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            scale_q = 1 + 5 * torch.rand(1).item()
            scale_k = 1 + 5 * torch.rand(1).item()
            scale_v = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            w_k *= scale_k
            w_v *= scale_v
            module.data = torch.cat([w_q, w_k, w_v], dim=0)
            scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
        elif "attn.proj.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            scale_fc1 = 1 + 5 * torch.rand(1).item()
            w_fc1 *= scale_fc1
            module.data = w_fc1
            scaling_factors[name] = scale_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            scale_fc2 = 1 + 5 * torch.rand(1).item()
            w_fc2 *= scale_fc2
            module.data = w_fc2
            scaling_factors[name] = scale_fc2
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    ## Record layers identical to pre_model
    layers_pretrained = []
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            pre_q = pre_q.to(w_q.device)
            ## If pre_q and w_q are identical, no recovery is needed
            if torch.allclose(w_q, pre_q, rtol=1e-5):
                layers_pretrained.append(name)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            if torch.allclose(w_proj, pre_proj, rtol=1e-5):
                layers_pretrained.append(name)
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            pre_fc1 = pre_fc1.to(w_fc1.device)
            if torch.allclose(w_fc1, pre_fc1, rtol=1e-5):
                layers_pretrained.append(name)
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            pre_fc2 = pre_fc2.to(w_fc2.device)
            if torch.allclose(w_fc2, pre_fc2, rtol=1e-5):
                layers_pretrained.append(name)
            k =  fix_factor(sqrt(torch.var(w_fc2).item()/torch.var(pre_fc2).item()))
            new_w_fc2 = w_fc2/k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors, layers_pretrained

def attack_soter_our(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    ## Record layers identical to pre_model
    layers_pretrained = []
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            pre_q = pre_q.to(w_q.device)
            pre_k = pre_k.to(w_k.device)
            pre_v = pre_v.to(w_v.device)
            ## If pre_q and w_q are identical, no recovery is needed
            if torch.allclose(w_q, pre_q, rtol=1e-5):
                layers_pretrained.append(name)
            k = 1/fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k1 = pre_q.flatten().dot(w_q.flatten()) / w_q.flatten().dot(w_q.flatten())
            print(f"Layer {name} k from var: {k}, k from dot: {k1}")
            k2 = pre_k.flatten().dot(w_k.flatten()) / w_k.flatten().dot(w_k.flatten())
            k3 = pre_v.flatten().dot(w_v.flatten()) / w_v.flatten().dot(w_v.flatten())
            new_w_q = w_q*k1
            new_w_k = w_k*k2
            new_w_v = w_v*k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            if torch.allclose(w_proj, pre_proj, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_proj.flatten().dot(w_proj.flatten()) / w_proj.flatten().dot(w_proj.flatten())
            new_w_proj = w_proj*k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            pre_fc1 = pre_fc1.to(w_fc1.device)
            if torch.allclose(w_fc1, pre_fc1, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_fc1.flatten().dot(w_fc1.flatten()) / w_fc1.flatten().dot(w_fc1.flatten())
            new_w_fc1 = w_fc1*k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            pre_fc2 = pre_fc2.to(w_fc2.device)
            if torch.allclose(w_fc2, pre_fc2, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_fc2.flatten().dot(w_fc2.flatten()) / w_fc2.flatten().dot(w_fc2.flatten())
            new_w_fc2 = w_fc2*k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors, layers_pretrained
def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            num_rows = w_q.shape[1]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_rows):
                ratio_q = 1 + 5 * torch.rand(1).item()
                ratio_k = 1 + 5 * torch.rand(1).item()
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_q[:,i] *= ratio_q
                w_k[:,i] *= ratio_k
                w_v[:,i] *= ratio_v
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
            scaling_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            permutation_q = torch.randperm(num_rows)
            permutation_k = torch.randperm(num_rows)
            permutation_v = torch.randperm(num_rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[:,permutation_q]
            ob_w_k = w_k[:,permutation_k]
            ob_w_v = w_v[:,permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        elif "attn.proj.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[:,permutation]
            module.data = ob_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            num_rows = w_fc1.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[:,permutation]
            module.data = ob_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            num_rows = w_fc2.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[:,permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, scaling_factors

def attack_shadownet(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq is wq after permutation recovery
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
                restore_wq[:,i] /= ratio_q
                restore_wk[:,i] /= ratio_k
                restore_wv[:,i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = col_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[1]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[:,i]).item()/torch.var(pre_wo[:,i]).item()))
                restore_wo[:,i] /= ratio_o
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = col_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[1]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[:,i]).item()/torch.var(pre_fc1[:,i]).item()))
                restore_fc1[:,i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = col_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item()))
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2   
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_shadownet_our(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq is wq after permutation recovery
            perm_q, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq1[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq1[:,i] /= ratio_q
            restore_wq = col_restore_perm_and_scale(pre_wq, ob_wq)
            print(f"layer {name} similarity between the two recovery methods: {np.mean(np.abs(restore_wq1.cpu().numpy().flatten()-restore_wq.cpu().numpy().flatten()) < 1e-4)}")
            perm_k, _, _ = col_restore_perm(pre_wk, ob_wk)
            perm_v, _, _ = col_restore_perm(pre_wv, ob_wv)
            restore_wk = col_restore_perm_and_scale(pre_wk, ob_wk)
            restore_wv = col_restore_perm_and_scale(pre_wv, ob_wv)
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
            restore_perm[name] = {"q": torch.tensor(perm_q), "k": torch.tensor(perm_k), "v": torch.tensor(perm_v)}
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            perm_o, _, _ = col_restore_perm(pre_wo, ob_wo)
            restore_wo = col_restore_perm_and_scale(pre_wo, ob_wo)
            module.data = restore_wo
            restore_perm[name] = torch.tensor(perm_o)
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            perm_fc1, _, _ = col_restore_perm(pre_fc1, ob_fc1)
            restore_fc1 = col_restore_perm_and_scale(pre_fc1, ob_fc1)
            module.data = restore_fc1
            restore_perm[name] = torch.tensor(perm_fc1)
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            perm_fc2, _, _ = col_restore_perm(pre_fc2, ob_fc2)
            restore_fc2 = col_restore_perm_and_scale(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item()))
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2   
            restore_perm[name] = torch.tensor(perm_fc2)
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm
def ob_LoRO(model, r=8, noise=1e-1):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            w = module.data
            d1, d2 = w.shape
            # Generate the two factors of the low-rank matrix
            A = torch.randn(d1, r) * noise
            A = A.to(w.device)
            B = torch.randn(r, d2) * noise
            B = B.to(w.device)
            low_rank_matrix = torch.matmul(A, B)
            # Add the low-rank matrix to the original weights
            w += low_rank_matrix
            module.data = w
    return model

def attack_LoRO(model, pre_model, vic_model=None, r=8):
    set_seed()
    for name, module in model.named_parameters():
        ob_w = module.data
        pre_w = pre_model.state_dict()[name].data
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            restore_w = restore_low_rank(pre_w, ob_w, r)
            module.data = restore_w
        else:
            module.data = pre_model.state_dict()[name].data
        restore_w = module.data
        if vic_model is not None:
            vic_w = vic_model.state_dict()[name].data
            print(f"layer {name} :")
            error = torch.norm(vic_w - pre_w, p='fro')
            print(f"error between victim model and public model: {error.item()}")
            error = torch.norm(restore_w.cpu() - pre_w, p='fro')
            print(f"error between recovered model and public model: {error.item()}")
            error = torch.norm(restore_w.cpu() - vic_w, p='fro')
            print(f"error between recovered model and victim model: {error.item()}")
    return model

def ob_obfuscatune(model):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            w = module.data
            d1, d2 = w.shape
            Q, _ = torch.linalg.qr(torch.randn(d2, d2))
            Q = Q.to(w.device)
            w = torch.matmul(w, Q)
            module.data = w
    return model
    

def attack_obfuscatune(model, pre_model, vic_model=None):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            restore_w = restore_orthogonal(pre_w, ob_w)
            if vic_model is not None:
                vic_w = vic_model.state_dict()[name].data
                print(f"layer {name} :")
                error = torch.norm(vic_w - pre_w, p='fro')
                print(f"error between victim model and public model: {error.item()}")
                error = torch.norm(restore_w.cpu() - pre_w, p='fro')
                print(f"error between recovered model and public model: {error.item()}")
                error = torch.norm(restore_w.cpu() - vic_w, p='fro')
                print(f"error between recovered model and victim model: {error.item()}")
            module.data = restore_w
        else:
            module.data = pre_model.state_dict()[name].data
    return model


def ob_LRAO(model, pre_model, r=8):
    set_seed()
    R = {}
    pre_state = pre_model.state_dict()
    with torch.no_grad():
        for name, module in model.named_parameters():
            if not _vit_obfus_weight(name):
                continue
            w = module.detach()
            wp = pre_state[name].detach()
            dw = wp.cpu().to(dtype=torch.float64) - w.cpu().to(dtype=torch.float64)
            if not torch.isfinite(dw).all():
                raise ValueError(f"ob_LRAO received non-finite values in {name}")
            U, S, Vt = torch.linalg.svd(dw, full_matrices=False)
            C = (U[:, :r] * S[:r]) @ Vt[:r, :]
            low_rank_matrix = C.to(dtype=w.dtype)
            module.copy_(w + low_rank_matrix.to(device=w.device, dtype=w.dtype))
            R[name] = low_rank_matrix.cpu()
    return model, R



def attack_LRAO(model, pre_model, vic_model=None, r=8):
    set_seed()
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None
    with torch.no_grad():
        for name, module in model.named_parameters():
            if _vit_obfus_weight(name):
                restore_w = module.detach().clone()
                if vic_model is not None:
                    pre_w = pre_state[name].detach()
                    vic_w = vic_state[name].detach()
                    print(f"layer {name} :")
                    print(f"error between obfuscated model and public model: {torch.norm(restore_w.cpu() - pre_w.cpu(), p='fro').item()}")
                    print(f"error between victim model and public model: {torch.norm(vic_w.cpu() - pre_w.cpu(), p='fro').item()}")
                    print(f"error between recovered model and public model: {torch.norm(restore_w.cpu() - pre_w.cpu(), p='fro').item()}")
                    print(f"error between recovered model and victim model: {torch.norm(restore_w.cpu() - vic_w.cpu(), p='fro').item()}")
                module.copy_(restore_w.to(device=module.device, dtype=module.dtype))
            else:
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
    return model


def cluster_vectors(vectors, cluster_size=4):
    index_pairs = np.array([np.array([i]) for i in range(len(vectors))])
    cur = 1
    while cur < cluster_size:
        cur *= 2
        cos_sim_matrix = cosine_similarity(vectors)
        sum_cos_sim_dis = np.mean(cos_sim_matrix, axis=0)
        sorted_indices = np.argsort(sum_cos_sim_dis)[::-1]
        np.fill_diagonal(cos_sim_matrix, np.inf)
        pairs = []
        index_pair = []
        repeat_index = []
        for i in sorted_indices:
            if i in repeat_index:
                continue
            j = np.argmin(cos_sim_matrix[i])
            repeat_index.append(j)
            cos_sim_matrix[i, :] = np.inf
            cos_sim_matrix[:, i] = np.inf
            cos_sim_matrix[j, :] = np.inf
            cos_sim_matrix[:, j] = np.inf
            index_pair.append(np.concatenate((index_pairs[i], index_pairs[j])))
            pairs.append(np.mean([vectors[i], vectors[j]], axis=0))
        vectors = pairs
        index_pairs = index_pair
    return index_pairs


def _torch_pinv_np(matrix):
    return np.linalg.pinv(matrix, rcond=1e-15)


def solve_AK_PB(A, B, max_iter=20):
    n, _ = A.shape
    U_A, _ = np.linalg.qr(A, mode="reduced")
    U_B, _ = np.linalg.qr(B, mode="reduced")

    lev_A = np.linalg.norm(U_A, axis=1) ** 2
    lev_B = np.linalg.norm(U_B, axis=1) ** 2
    cost_init = np.abs(lev_A[:, None] - lev_B[None, :])
    _, col_ind = linear_sum_assignment(cost_init)
    perm = col_ind.astype(np.int64, copy=False)

    for _ in range(max_iter):
        source = U_B[perm]
        Q, _ = orthogonal_procrustes(source, U_A)
        transformed_B = U_B @ Q
        cost = U_A @ transformed_B.T
        _, col_ind = linear_sum_assignment(cost, maximize=True)
        new_perm = col_ind.astype(np.int64, copy=False)
        if np.array_equal(new_perm, perm):
            break
        perm = new_perm

    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P


def recover_K_and_P2(A_obf, B, cluster_index, th=0.15, step=0.01, max_th=4.0, size=4):
    if step <= 0:
        raise ValueError("step must be positive for recover_K_and_P2")
    n, _ = A_obf.shape
    remaining = [i for i in range(n)]
    K_est = np.zeros((n, n))
    P2_est = np.zeros((n, n))
    idx = 0

    for idlist in cluster_index:
        perm = list(idlist)
        Ai = A_obf[perm, :]
        Bi = B[perm, :]
        Ki = Bi @ _torch_pinv_np(Ai)
        diff = np.linalg.norm(Ki @ Ai - Bi) / (np.linalg.norm(Bi) + 1e-12)
        if diff < th:
            K_est[idx : idx + size, idx : idx + size] = Ki
            P2_est[perm, idx : idx + size] = np.eye(size)
            remaining = [x for x in remaining if x not in perm]
            idx += size

    max_rounds = max(1, int(np.ceil((max_th - th) / step)) + 2)
    rounds = 0
    while True:
        rounds += 1
        found = False
        for i in list(remaining):
            if i not in remaining:
                continue
            bi = B[i, :]
            local_id = remaining.index(i)
            A = A_obf[remaining, :].T
            b = bi.T
            pre_select_n = size * 4
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scaler = StandardScaler()
            A = scaler.fit_transform(A)
            lasso = Lasso(alpha=0.00001, max_iter=1000)
            lasso.fit(A, b)
            importance = np.abs(lasso.coef_)
            candidate_indices = np.argsort(importance)[-pre_select_n:]
            if local_id not in candidate_indices:
                continue
            candidate_indices = [ci for ci in candidate_indices if ci != local_id]
            for indices in combinations(candidate_indices, size - 1):
                indices = list(indices)
                indices.append(local_id)
                perm = [remaining[x] for x in indices]
                Ai = A_obf[perm, :]
                Bi = B[perm, :]
                Ki = Bi @ _torch_pinv_np(Ai)
                diff = np.linalg.norm(Ki @ Ai - Bi) / (np.linalg.norm(Bi) + 1e-12)
                if diff < th:
                    K_est[idx : idx + size, idx : idx + size] = Ki
                    P2_est[perm, idx : idx + size] = np.eye(size)
                    remaining = [x for x in remaining if x not in perm]
                    idx += size
                    found = True
                    break
        if not found:
            th += step
        if len(remaining) == 0 or th > max_th:
            break

    if len(remaining) != 0:
        return None, None
    return K_est, P2_est


def recover_groupcover(A_obf, A_pub, size=4, th=0.1, max_th=0.5, step=0.1):
    n_rows, n_cols = A_obf.shape
    cluster_index = cluster_vectors(A_pub, cluster_size=size)

    P1_list = []
    perm_list = []
    best_diff = float("inf")
    for idlist in cluster_index:
        perm = list(idlist)
        Ai = A_obf[perm, :]
        Bi = A_pub[perm, :]
        P1 = solve_AK_PB(Bi.T, Ai.T, max_iter=20)
        Bi_perm = Bi @ P1
        Ki = Bi_perm @ _torch_pinv_np(Ai)
        diff = np.linalg.norm(Ki @ Ai - Bi_perm) / (np.linalg.norm(Bi_perm) + 1e-12)
        if diff < th * 2:
            P1_list.append(P1)
            perm_list.append(perm)
        if diff < th / 2:
            break
        best_diff = min(best_diff, diff)

    if len(P1_list) == 0:
        print(f"no suitable P1_est found; minimum candidate residual: {best_diff:.6f}")
        return None, None, None

    min_diff = float("inf")
    P1_est = None
    for P1 in P1_list:
        total_diff = 0
        for perm in perm_list:
            Ai = A_obf[perm, :]
            Bi = A_pub[perm, :] @ P1
            Ki = Bi @ _torch_pinv_np(Ai)
            diff = np.linalg.norm(Ki @ Ai - Bi) / (np.linalg.norm(Bi) + 1e-12)
            total_diff += diff
        if total_diff < min_diff:
            P1_est = P1.T
            min_diff = total_diff
    
    if min_diff / len(perm_list) > th * 2:
        print(f"selected P1_est has too large an average residual: {min_diff / len(perm_list):.6f} recovery may be incorrect")
        return None, None, None

    print(f"selected P1_est, residual: {min_diff / len(perm_list):.6f}")
    B_fix = A_pub @ P1_est.T
    K_est, P2_est = recover_K_and_P2(A_obf, B_fix, cluster_index, th=0.1, step=0.3, max_th=4, size=size)
    if K_est is None or P2_est is None:
        print("no suitable K_est and P2_est found")
        return None, None, None

    A_guess = P2_est @ K_est @ P2_est.T @ A_obf @ P1_est
    A = P2_est @ K_est @ P2_est.T @ A_obf
    cost_matrix = -(A.T @ A_pub)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P1 = np.zeros((n_cols, n_cols))
    P1[row_ind, col_ind] = 1
    A_guess2 = P2_est @ K_est @ P2_est.T @ A_obf @ P1

    if np.linalg.norm(A_guess2 - A_pub) < np.linalg.norm(A_guess - A_pub):
        print("updating P1")
        P1_est = P1
    B_fix = A_pub @ P1_est.T
    K_est, P2_est = recover_K_and_P2(A_obf, B_fix, cluster_index, th=th, step=step, max_th=max_th, size=size)
    if K_est is None or P2_est is None:
        print("no suitable K_est and P2_est found")
        return None, None, None
    A_guess = P2_est @ K_est @ P2_est.T @ A_obf @ P1_est

    def extract_p2_row_clusters(P2, m, cluster_size):
        clusters = []
        for c in range(0, m, cluster_size):
            block = P2[:, c : c + cluster_size]
            if block.size == 0 or np.max(np.abs(block)) < 1e-12:
                continue
            rows = np.where(np.any(np.abs(block) > 1e-12, axis=1))[0]
            if len(rows) == cluster_size:
                clusters.append(set(rows.tolist()))
        return clusters

    inv_perm = np.argmax(P1_est, axis=1)
    cluster_index_out = extract_p2_row_clusters(P2_est, n_rows, size)
    return A_guess, inv_perm, cluster_index_out


def _vit_ob_groupcover_apply_slice(w_slice, device, dtype, size):
    w_tensor = w_slice.detach()
    d1, d2 = w_tensor.shape
    w = w_tensor.cpu().numpy().astype(np.float64, copy=True)
    cluster_index = cluster_vectors(w, cluster_size=size)
    random_coeff_list = [[] for _ in range(d1)]
    new_w = np.zeros_like(w)
    for idlist in cluster_index:
        new_kernels = []
        for i in idlist:
            random_coeffs = np.random.randint(1, 100, size=size)
            random_coeff_list[i] = random_coeffs
            new_kernel = sum(coeff * w[idlist[j], :] for j, coeff in enumerate(random_coeffs))
            new_kernels.append(new_kernel)
        for idx, idi in enumerate(idlist):
            new_w[idi, :] = new_kernels[idx]
    permutation = torch.randperm(d2, device=device)
    new_w_t = torch.from_numpy(new_w).to(device=device, dtype=dtype)[:, permutation]
    ci_out = [set(map(int, idlist)) for idlist in cluster_index]
    return new_w_t, ci_out, random_coeff_list, permutation.cpu().numpy().tolist()


def ob_groupcover(model, size=4):
    set_seed()
    ob_cluster_index = {}
    ob_random_coeff_list = {}
    ob_permutation = {}
    with torch.no_grad():
        for name, module in model.named_parameters():
            if not _vit_obfus_weight(name):
                continue
            w_tensor = module.detach()
            device = w_tensor.device
            dtype = w_tensor.dtype
            if "qkv.weight" in name:
                outs = []
                ci_d, rl_d, pd = {}, {}, {}
                for tag, wsl in zip(("q", "k", "v"), w_tensor.chunk(3, dim=0)):
                    nw, ci, rnd, perm = _vit_ob_groupcover_apply_slice(wsl, device, dtype, size)
                    outs.append(nw)
                    ci_d[tag] = ci
                    rl_d[tag] = rnd
                    pd[tag] = perm
                module.copy_(torch.cat(outs, dim=0))
                ob_cluster_index[name] = ci_d
                ob_random_coeff_list[name] = rl_d
                ob_permutation[name] = pd
                continue

            nw, ci, rnd, perm = _vit_ob_groupcover_apply_slice(w_tensor, device, dtype, size)
            module.copy_(nw)
            ob_cluster_index[name] = ci
            ob_random_coeff_list[name] = rnd
            ob_permutation[name] = perm
    return model, ob_cluster_index, ob_random_coeff_list, ob_permutation


def attack_groupcover(
    model,
    pre_model,
    size=4,
    vic_model=None,
    th=0.1,
    max_th=0.5,
    recovered_state_dict=None,
    recovered_attack_extras=None,
    reuse_recovered=False,
):
    set_seed()
    restore_permutation = {}
    restore_cluster_index = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None
    recovered_attack_extras = recovered_attack_extras or {}
    recovered_restore_permutation = recovered_attack_extras.get("restore_permutation", {})
    recovered_restore_cluster_index = recovered_attack_extras.get("restore_cluster_index", {})

    def try_load_recovered(name, module):
        if not reuse_recovered or recovered_state_dict is None or name not in recovered_state_dict:
            return False
        try:
            if name not in recovered_restore_permutation or name not in recovered_restore_cluster_index:
                print(f"No successful recovery found; retrying recovery: {name}")
                return False

            recovered_weight = recovered_state_dict[name]
            if tuple(recovered_weight.shape) != tuple(module.shape):
                print(
                    f"Saved groupcover recovered module shape mismatch; retrying recovery: "
                    f"{name} cached={tuple(recovered_weight.shape)} current={tuple(module.shape)}"
                )
                return False
            module.copy_(recovered_weight.to(device=module.device, dtype=module.dtype))
            if name in recovered_restore_permutation:
                restore_permutation[name] = recovered_restore_permutation[name]
            if name in recovered_restore_cluster_index:
                restore_cluster_index[name] = recovered_restore_cluster_index[name]
            print(f"Loaded recovered groupcover module; skipping recovery: {name}")
            return True
        except Exception as exc:
            print(f"Failed to load recovered groupcover module; retrying recovery {name}: {exc}")
            return False

    with torch.no_grad():
        for name, module in model.named_parameters():
            if not _vit_obfus_weight(name):
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
                continue

            if try_load_recovered(name, module):
                continue

            ob_w = module.detach().clone()
            pre_w = pre_state[name].detach()

            def recover_and_report(A_guess_arr, rp, cci):
                restore_permutation[name] = rp
                restore_cluster_index[name] = cci
                module.copy_(torch.from_numpy(A_guess_arr).to(device=module.device, dtype=module.dtype))
                if vic_model is None:
                    return
                print(f"name: {name}")
                vic_w = vic_state[name].detach()
                print(f"    error between public model and victim model: {torch.norm(pre_w.cpu() - vic_w.cpu(), p='fro').item():.4e}")
                print(f"    error between recovered model and public model: {torch.norm(module.detach().cpu() - pre_w.cpu(), p='fro').item():.4e}")
                print(f"    error between recovered model and victim model: {torch.norm(module.detach().cpu() - vic_w.cpu(), p='fro').item():.4e}")

            if "qkv.weight" in name:
                guesses = []
                rp_d = {}
                ci_d = {}
                failed = False
                for tag, owl, pwl in zip(("q", "k", "v"), ob_w.chunk(3, dim=0), pre_w.chunk(3, dim=0)):
                    A_obf = owl.cpu().numpy().astype(np.float64, copy=False)
                    A_pub = pwl.cpu().numpy().astype(np.float64, copy=False)
                    A_guess, inv_perm, cluster_index = recover_groupcover(
                        A_obf, A_pub, size=size, th=th, max_th=max_th
                    )
                    if A_guess is None or inv_perm is None or cluster_index is None:
                        print(f"no suitable A_guess, inv_perm, cluster_index found for {name} ({tag})")
                        failed = True
                        break
                    guesses.append(A_guess)
                    rp_d[tag] = inv_perm
                    ci_d[tag] = cluster_index
                if failed:
                    module.copy_(pre_w.to(device=module.device, dtype=module.dtype))
                    continue
                recover_and_report(np.concatenate(guesses, axis=0), rp_d, ci_d)
                continue

            A_obf = ob_w.cpu().numpy().astype(np.float64, copy=False)
            A_pub = pre_w.cpu().numpy().astype(np.float64, copy=False)
            A_guess, inv_perm, cluster_index = recover_groupcover(
                A_obf, A_pub, size=size, th=th, max_th=max_th
            )
            if A_guess is None or inv_perm is None or cluster_index is None:
                print(f"no suitable A_guess, inv_perm, cluster_index found for {name}")
                module.copy_(pre_w.to(device=module.device, dtype=module.dtype))
                continue
            recover_and_report(A_guess, inv_perm, cluster_index)
    return model, restore_permutation, restore_cluster_index


def ob_twinshield(model):
    set_seed()
    ob_permutation = {}
    ob_d = {}
    with torch.no_grad():
        for name, module in model.named_parameters():
            if not _vit_obfus_weight(name):
                continue
            w = module.detach()
            device = w.device
            dtype = w.dtype

            def ob_one_slice(Wv):
                n, m = Wv.shape
                R = np.random.randn(n, m)
                d_true = np.random.randn(n)
                P_true_indices = np.random.permutation(2 * m)
                P = np.zeros((2 * m, 2 * m), dtype=np.float64)
                P[P_true_indices, np.arange(2 * m)] = 1.0
                return np.concatenate([Wv + R, d_true[:, None] * R], axis=1) @ P, P_true_indices, d_true

            if "qkv.weight" in name:
                packs = []
                ppm, pdd = {}, {}
                for tag, wl in zip(("q", "k", "v"), w.cpu().chunk(3, dim=0)):
                    packed, pid, dd = ob_one_slice(wl.numpy().astype(np.float64, copy=False))
                    packs.append(packed)
                    ppm[tag] = pid
                    pdd[tag] = dd
                module.data = torch.from_numpy(np.concatenate(packs, axis=0)).to(device=device, dtype=dtype)
                ob_permutation[name] = ppm
                ob_d[name] = pdd
                continue

            packed_w, P_true_indices, d_true = ob_one_slice(w.cpu().numpy().astype(np.float64, copy=False))
            module.data = torch.from_numpy(packed_w).to(device=device, dtype=dtype)
            ob_permutation[name] = P_true_indices
            ob_d[name] = d_true
    return model, ob_permutation, ob_d


def solve_D_P_alternating(Wo1, Wo2, Wp, max_iter=50, tol=1e-6):
    n, m = Wo1.shape
    d = np.zeros(n)
    P_indices = np.arange(m)
    prev_error = float("inf")

    for _ in range(max_iter):
        W_current = Wo1 + d[:, None] * Wo2
        W_curr_norm = np.sum(W_current ** 2, axis=0).reshape(-1, 1)
        Wp_norm = np.sum(Wp ** 2, axis=0).reshape(1, -1)
        cost_matrix = W_curr_norm + Wp_norm - 2 * (W_current.T @ Wp)
        _, col_ind = linear_sum_assignment(cost_matrix)
        P_indices = col_ind

        Wp_permuted = Wp[:, P_indices]
        target = Wp_permuted - Wo1
        num = np.sum(Wo2 * target, axis=1)
        den = np.sum(Wo2 ** 2, axis=1)
        safe_den = np.where(den < 1e-12, 1.0, den)
        d = np.where(den < 1e-12, 0.0, num / safe_den)

        current_error = np.linalg.norm(Wo1 + d[:, None] * Wo2 - Wp_permuted, "fro")
        if prev_error - current_error < tol:
            break
        prev_error = current_error
    return d, P_indices, current_error


def solve_twinshield(W_ob, Wp, sigma_R=1.0, eps=1e-12, min_abs_d=0.2, refine_iter=5):
    n, two_m = W_ob.shape
    assert two_m % 2 == 0
    m = two_m // 2
    assert Wp.shape == (n, m)

    row_mean_ob_sq = np.mean(W_ob ** 2, axis=1)
    row_mean_wp_sq = np.mean(Wp ** 2, axis=1)
    abs_d_sq = (2.0 * row_mean_ob_sq - row_mean_wp_sq - sigma_R ** 2) / (sigma_R ** 2)
    abs_d = np.sqrt(np.maximum(abs_d_sq, 1e-9))

    good_rows = abs_d > min_abs_d
    if np.sum(good_rows) < max(1, n // 4):
        good_rows = abs_d >= np.quantile(abs_d, 0.5)
    ad = abs_d[good_rows]

    X_abs = np.abs(W_ob[good_rows, :])
    Y_scaled_abs = X_abs / (ad[:, None] + eps)
    M = (
        np.sum(X_abs ** 2, axis=0)[:, None]
        + np.sum(Y_scaled_abs ** 2, axis=0)[None, :]
        - 2.0 * (X_abs.T @ Y_scaled_abs)
    )
    np.fill_diagonal(M, np.inf)

    N = two_m
    U = np.minimum(M, M.T)
    np.fill_diagonal(U, np.inf)
    triu = np.triu_indices(N, 1)
    order = np.argsort(U[triu])

    used = np.zeros(N, dtype=bool)
    pairs = []
    for idx in order:
        a = triu[0][idx]
        b = triu[1][idx]
        if used[a] or used[b]:
            continue
        u, v = (a, b) if M[a, b] <= M[b, a] else (b, a)
        pairs.append((u, v))
        used[a] = True
        used[b] = True
        if len(pairs) == m:
            break
    if len(pairs) != m:
        raise RuntimeError(f"Only found {len(pairs)} pairs, expected {m}.")

    c1 = np.array([p[0] for p in pairs], dtype=np.int64)
    c2 = np.array([p[1] for p in pairs], dtype=np.int64)
    A = W_ob[:, c1]
    B = W_ob[:, c2]

    d_calc = np.sum(A * B, axis=1) / (np.sum(A ** 2, axis=1) + eps)
    sign = np.sign(d_calc)
    sign[sign == 0] = 1.0
    d_calc = np.where(np.abs(d_calc) < 1e-8, sign * abs_d, d_calc)

    c1_sorted = None
    c2_sorted = None
    for _ in range(refine_iter + 1):
        safe_d = np.where(np.abs(d_calc) < 1e-8, np.sign(d_calc + eps) * 1e-8, d_calc)
        restored_pairs = A - B / safe_d[:, None]
        C = (
            np.sum(restored_pairs ** 2, axis=0)[:, None]
            + np.sum(Wp ** 2, axis=0)[None, :]
            - 2.0 * (restored_pairs.T @ Wp)
        )
        pair_idx, j_idx = linear_sum_assignment(C)

        new_c1_sorted = np.empty(m, dtype=np.int64)
        new_c2_sorted = np.empty(m, dtype=np.int64)
        new_c1_sorted[j_idx] = c1[pair_idx]
        new_c2_sorted[j_idx] = c2[pair_idx]

        X = W_ob[:, new_c1_sorted] - Wp
        Y = W_ob[:, new_c2_sorted]
        d_new = np.sum(X * Y, axis=1) / (np.sum(X ** 2, axis=1) + eps)

        c1_sorted = new_c1_sorted
        c2_sorted = new_c2_sorted
        if np.linalg.norm(d_new - d_calc) / (np.linalg.norm(d_calc) + eps) < 1e-8:
            d_calc = d_new
            break
        d_calc = d_new
        c1 = c1_sorted
        c2 = c2_sorted
        A = W_ob[:, c1]
        B = W_ob[:, c2]

    safe_d = np.where(np.abs(d_calc) < 1e-8, np.sign(d_calc + eps) * 1e-8, d_calc)
    restore_w = W_ob[:, c1_sorted] - W_ob[:, c2_sorted] / safe_d[:, None]
    C_final = (
        np.sum(restore_w ** 2, axis=0)[:, None]
        + np.sum(Wp ** 2, axis=0)[None, :]
        - 2.0 * (restore_w.T @ Wp)
    )
    row_ind, col_ind = linear_sum_assignment(C_final)

    final_c1 = np.empty(m, dtype=np.int64)
    final_c2 = np.empty(m, dtype=np.int64)
    final_c1[col_ind] = c1_sorted[row_ind]
    final_c2[col_ind] = c2_sorted[row_ind]

    X = W_ob[:, final_c1] - Wp
    Y = W_ob[:, final_c2]
    d_calc = np.sum(X * Y, axis=1) / (np.sum(X ** 2, axis=1) + eps)
    safe_d = np.where(np.abs(d_calc) < 1e-8, np.sign(d_calc + eps) * 1e-8, d_calc)
    restore_w = W_ob[:, final_c1] - W_ob[:, final_c2] / safe_d[:, None]

    p_calc = np.empty(2 * m, dtype=np.int64)
    p_calc[final_c1] = np.arange(m)
    p_calc[final_c2] = np.arange(m, 2 * m)

    return restore_w, d_calc, p_calc


def attack_twinshield(model, pre_model, vic_model=None, dataset="cifar_10"):
    set_seed()
    restore_permutation = {}
    restore_d = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None

    def recover_one_pack(Woo, Wpp):
        n_rows, n_cols = Wpp.shape
        if Woo.shape != (n_rows, 2 * n_cols):
            return None, None, None
        return solve_twinshield(Woo, Wpp)

    with torch.no_grad():
        for name, module in model.named_parameters():
            if not _vit_obfus_weight(name):
                module.data = pre_state[name].detach().to(device=module.device, dtype=module.dtype).contiguous()
                continue

            ob_w = module.detach()
            pre_w = pre_state[name].detach()
            vic_w = vic_state[name].detach() if vic_model is not None else None
            Wp = pre_w.cpu().numpy().astype(np.float64, copy=False)
            packed_w = ob_w.cpu().numpy().astype(np.float64, copy=False)

            if "qkv.weight" in name:
                n_rows, m_cols = Wp.shape
                slice_rows = n_rows // 3
                expected_shape = (n_rows, 2 * m_cols)
                if n_rows % 3 != 0 or packed_w.shape != expected_shape:
                    restore_w = pre_w.cpu().numpy()
                    print(f"[TwinShield] qkv packed weight shape is invalid; using public-model weights as the recovery result {name}")
                else:
                    guesses = []
                    rp_d, rd_d = {}, {}
                    failed = False
                    for tag, p_slice, ob_slice in zip(
                        ("q", "k", "v"),
                        np.split(Wp, 3, axis=0),
                        [packed_w[i * slice_rows : (i + 1) * slice_rows, :] for i in range(3)],
                    ):
                        rw, dd, pc = recover_one_pack(ob_slice, p_slice)
                        if rw is None:
                            failed = True
                            print(f"[TwinShield] {name} ({tag}) shape mismatch; using public weights")
                            break
                        guesses.append(rw)
                        rp_d[tag] = pc
                        rd_d[tag] = dd
                    restore_w = pre_w.cpu().numpy() if failed else np.concatenate(guesses, axis=0)
                    if not failed:
                        restore_permutation[name] = rp_d
                        restore_d[name] = rd_d
            else:
                rw, dd, pc = recover_one_pack(packed_w, Wp)
                if rw is None:
                    restore_w = pre_w.cpu().numpy()
                    print(f"TwinShield packed weight shape is invalid; using public-model weights as the recovery result {name}")
                else:
                    restore_w = rw
                    restore_permutation[name] = pc
                    restore_d[name] = dd

            restore_w = np.ascontiguousarray(restore_w)
            module.data = torch.from_numpy(restore_w).to(device=module.device, dtype=module.dtype).contiguous()
            if vic_w is not None:
                print(f"name: {name}")
                print(f"    error between recovered model and public model: {torch.norm(module.detach().cpu() - pre_w.cpu(), p='fro').item():.4e}")
                print(f"    error between public model and victim model: {torch.norm(pre_w.cpu() - vic_w.cpu(), p='fro').item():.4e}")
                print(f"    error between recovered model and victim model: {torch.norm(module.detach().cpu() - vic_w.cpu(), p='fro').item():.4e}")
    return model, restore_permutation, restore_d
    

def ob_tempo(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            num_cols = w_q.shape[0]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_cols):
                ratio_q = 1 + 5 * torch.rand(1).item()
                ratio_k = 1 + 5 * torch.rand(1).item()
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_q[i] *= ratio_q
                w_k[i] *= ratio_k
                w_v[i] *= ratio_v
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
            scaling_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            permutation_q = torch.randperm(num_cols)
            permutation_k = torch.randperm(num_cols)
            permutation_v = torch.randperm(num_cols)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        elif "attn.proj.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            num_cols = w_fc1.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            num_cols = w_fc2.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq is wq after permutation recovery
            perm_q, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            perm_k, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            perm_v, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            restore_perm[name] = {"q": torch.tensor(perm_q), "k": torch.tensor(perm_k), "v": torch.tensor(perm_v)}
            for i in range(ob_wq.shape[0]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item()))
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item()))
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item()))
                restore_wq[i] /= ratio_q
                restore_wk[i] /= ratio_k
                restore_wv[i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            perm_o, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            restore_perm[name] = torch.tensor(perm_o)
            for i in range(ob_wo.shape[0]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item()))
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            perm_fc1, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)
            restore_perm[name] = torch.tensor(perm_fc1)
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item()))
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1           
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            perm_fc2, _, restore_fc2 = row_restore_perm(pre_fc2, ob_fc2)
            restore_perm[name] = torch.tensor(perm_fc2)
            for i in range(ob_fc2.shape[0]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[i]).item()/torch.var(pre_fc2[i]).item()))
                restore_fc2[i] /= ratio_fc2
            module.data = restore_fc2   
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            device = w_q.device
            coeff_q, coeff_k, coeff_v = torch.randint(0,5,(w_q.shape[0],), device=device), torch.randint(0,5,(w_k.shape[0],), device=device), torch.randint(0,5,(w_v.shape[0],), device=device)
            mask_q, mask_k, mask_v = torch.matmul(w_q.T, coeff_q.float()), torch.matmul(w_k.T, coeff_k.float()), torch.matmul(w_v.T, coeff_v.float())
            layer_masks[name] = {"q": mask_q, "k": mask_k, "v": mask_v}
            ratios_q, ratios_k, ratios_v = [], [], []
            ratios_q2, ratios_k2, ratios_v2 = [], [], []
            for i in range(w_q.shape[0]):
                ratio_q, ratio_k, ratio_v = (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float()
                weight_q, weight_k, weight_v = (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= weight_q
                w_k[i] *= weight_k
                w_v[i] *= weight_v
                mask_qi, mask_ki, mask_vi = mask_q * ratio_q, mask_k * ratio_k, mask_v * ratio_v
                w_q[i] += mask_qi
                w_k[i] += mask_ki
                w_v[i] += mask_vi
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
                ratios_q2.append(weight_q)
                ratios_k2.append(weight_k)
                ratios_v2.append(weight_v)
            layer_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            weight_factors[name] = {"q": ratios_q2, "k": ratios_k2, "v": ratios_v2}
            rows = w_q.shape[0]
            permutation_q = torch.randperm(rows)
            permutation_k = torch.randperm(rows)
            permutation_v = torch.randperm(rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        if "attn.proj.weight" in name:
            w_proj = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_proj.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_proj.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        if "mlp.fc1.weight" in name:
            w_fc1 = module.data
            device = w_fc1.device
            coeff = torch.randint(0,5,(w_fc1.shape[0],), device=device)
            mask = torch.matmul(w_fc1.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc1.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc1[i] *= ratio2
                mask_i = mask * ratio 
                w_fc1[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc1.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        if "mlp.fc2.weight" in name:
            w_fc2 = module.data
            device = w_fc2.device
            coeff = torch.randint(0,5,(w_fc2.shape[0],), device=device)
            mask = torch.matmul(w_fc2.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc2.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc2[i] *= ratio2
                mask_i = mask * ratio 
                w_fc2[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc2.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def attack_arrowcloak(model, pre_model):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq is wq after permutation recovery
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wq[i] /= ratio_q
                restore_wk[i] /= ratio_k
                restore_wv[i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[0]):
                ratio_o = sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item())
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item())
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = row_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[0]):
                ratio_fc2 = sqrt(torch.var(restore_fc2[i]).item()/torch.var(pre_fc2[i]).item())
                restore_fc2[i] /= ratio_fc2
            module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model

from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds

def _prepare_right_ridge_operator(A, rho):
    n, m = A.shape
    if n <= m:
        system_inv = np.linalg.inv(A @ A.T + rho * np.eye(n, dtype=A.dtype))
        return "row", system_inv
    system_inv = np.linalg.inv(A.T @ A + rho * np.eye(m, dtype=A.dtype))
    return "col", system_inv

def _apply_right_ridge_operator(C, A, ridge_operator):
    mode, system_inv = ridge_operator
    if mode == "row":
        return (C @ A.T) @ system_inv
    return (C @ system_inv) @ A.T

def rank1_permuted_diagonal_decomposition(A, S_init=None, max_iter=50, tol=1e-6):
    m, n = A.shape
    
    if S_init is not None:
        S = S_init.copy()
    else:
        S = np.zeros_like(A)
    
    prev_error = np.inf
    
    for iteration in range(max_iter):
        R = A - S
        # SVD decomposition
        # U, s, Vt = svd(R, full_matrices=False, overwrite_a=True, check_finite=False)
        U, s, Vt = svds(R, k=1)
        # Take the largest singular value
        L = s[0] * np.outer(U[:, 0], Vt[0, :])
        
        E = A - L
        
        cost_matrix = -(E ** 2)
        
        # Use the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Construct the new S
        S_new = np.zeros_like(A)
        S_new[row_ind, col_ind] = E[row_ind, col_ind]
        
        # --- Check convergence ---
        error = np.linalg.norm(A - L - S_new, 'fro')
        if abs(prev_error - error) < tol:
            break
        prev_error = error
        S = S_new

    return L, S

def solve_permutation_projection(A, B, rho=1.0, alpha=1.0, inner_iter=30, inner_tol=1e-4):
    print(f"[Stage 1]Solving permutation projection: shape={A.shape}, rho = {rho}, alpha = {alpha}, iter = {inner_iter}, tol = {inner_tol}")
    ridge_operator = _prepare_right_ridge_operator(A, rho)
    T = alpha * _apply_right_ridge_operator(B, A, ridge_operator)
    L, S = rank1_permuted_diagonal_decomposition(
        T, max_iter=inner_iter, tol=inner_tol
    )
    loss = np.linalg.norm((L + S) @ A - B, "fro")
    print(f"[Stage1] Projection objective={loss:.6f}")
    P = np.zeros_like(S)
    row_ind, col_ind = linear_sum_assignment(-(S ** 2))
    P[row_ind, col_ind] = 1.0
    return P

def rank1_diagonal_decomposition(T, S_init=None, max_iter=30, tol=1e-6):
    if S_init is not None:
        S = S_init.copy()
    else:
        S = np.zeros_like(T)

    prev_err = np.inf
    for _ in range(max_iter):
        R = T - S
        U, s, Vt = svds(R, k=1)
        L = s[0] * np.outer(U[:, 0], Vt[0, :])

        E = T - L
        d = np.diag(E)
        S_new = np.diag(d)

        err = np.linalg.norm(T - (L + S_new), "fro")
        if abs(prev_err - err) < tol:
            S = S_new
            break
        prev_err = err
        S = S_new

    return L, S

def solve_diagonal_rank1_admm(A, B, rho=1.0, alpha=1.6, max_iter=200, tol=1e-6, inner_iter=20, inner_tol=1e-4):
    print(f"[Stage 2]Solving diagonal rank1 ADMM: shape={A.shape}, rho = {rho}, alpha = {alpha}, max_iter = {max_iter}, tol = {tol}, inner_iter = {inner_iter}, inner_tol = {inner_tol}")
    verbose_every = max(max_iter // 10, 1)
    n, _ = A.shape

    Z = np.zeros((n, n))
    Gamma = np.zeros((n, n))

    ridge_operator = _prepare_right_ridge_operator(A, rho)

    L = np.zeros((n, n))
    S = np.zeros((n, n))

    best_obj = np.inf
    best_L = L.copy()
    best_D = np.diag(np.diag(S)).copy()

    for it in range(max_iter):
        target_Z = (L + S) - (1.0 / rho) * Gamma
        Z_raw = target_Z + _apply_right_ridge_operator(B - target_Z @ A, A, ridge_operator)
        Z = alpha * Z_raw + (1.0 - alpha) * (L + S)

        T = Z + (1.0 / rho) * Gamma
        L, S = rank1_diagonal_decomposition(
            T,
            S_init=S,
            max_iter=inner_iter,
            tol=inner_tol,
        )

        X = L + S
        obj = np.linalg.norm(X @ A - B, "fro")
        primal_res = np.linalg.norm(Z - X, "fro")

        Gamma = Gamma + rho * (Z - X)

        if (it % verbose_every == 0) or (it == max_iter - 1):
            print(
                f"[Stage2] Iter {it:3d}: objective={obj:.6f}, primal_res={primal_res:.3e}"
            )

        if obj < best_obj:
            best_obj = obj
            best_L = L.copy()
            best_D = S.copy()

    return best_L, best_D

def attack_arrowcloak_our(model, pre_model, vic_model=None):
    set_seed()
    restore_perm = {}
    restore_L = {}
    restore_D = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None

    def solve_vit_forward_permutation(A_pub, A_obf):
        best_obj = np.inf
        best_P = None
        n = A_pub.shape[0]
        rows = np.arange(n)

        def make_support_init(T, perm):
            S = np.zeros_like(T)
            S[rows, perm] = T[rows, perm]
            return S

        for rho in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            ridge_operator = _prepare_right_ridge_operator(A_pub, rho)
            T = _apply_right_ridge_operator(A_obf, A_pub, ridge_operator)

            inits = [None]
            row_ind, col_ind = linear_sum_assignment(-(T ** 2))
            perm_abs = np.empty(n, dtype=np.int64)
            perm_abs[row_ind] = col_ind
            inits.append(make_support_init(T, perm_abs))

            rng = np.random.default_rng(42)
            for _ in range(4):
                inits.append(make_support_init(T, rng.permutation(n)))

            for S_init in inits:
                L, S = rank1_permuted_diagonal_decomposition(
                    T, S_init=S_init, max_iter=50, tol=1e-6
                )
                obj = np.linalg.norm((L + S) @ A_pub - A_obf, "fro")
                if obj < best_obj:
                    best_obj = obj
                    best_P = np.zeros_like(S)
                    r_ind, c_ind = linear_sum_assignment(-(S ** 2))
                    best_P[r_ind, c_ind] = 1.0

        return best_P

    def solve_shared_mask_rows(W_mix, W_pub, max_iter=100):
        row_norms = np.sum(W_pub * W_pub, axis=1) + 1e-20
        proj = np.sum(W_mix * W_pub, axis=1) / row_norms
        residual = W_mix - W_pub * proj[:, None]

        starts = []
        try:
            _, s, vt = svds(residual, k=1)
            starts.append(vt[0, :] * s[0])
        except Exception:
            _, s, vt = np.linalg.svd(residual, full_matrices=False)
            starts.append(vt[0, :] * s[0])
        starts.append(np.mean(residual, axis=0))
        starts.append(np.median(residual, axis=0))

        best = None
        for mask_init in starts:
            mask = mask_init.astype(np.float64, copy=True)
            if np.linalg.norm(mask) < 1e-20:
                continue

            scales = proj.copy()
            ratios = np.zeros(W_pub.shape[0], dtype=np.float64)
            pub_dot_mix = np.sum(W_pub * W_mix, axis=1)

            for _ in range(max_iter):
                mask_norm = np.dot(mask, mask) + 1e-20
                pub_dot_mask = W_pub @ mask
                mix_dot_mask = W_mix @ mask
                det = row_norms * mask_norm - pub_dot_mask * pub_dot_mask
                det = np.where(np.abs(det) < 1e-20, np.sign(det) * 1e-20 + 1e-20, det)

                scales = (pub_dot_mix * mask_norm - mix_dot_mask * pub_dot_mask) / det
                ratios = (mix_dot_mask * row_norms - pub_dot_mix * pub_dot_mask) / det

                denom = np.dot(ratios, ratios) + 1e-20
                mask = ratios @ (W_mix - W_pub * scales[:, None]) / denom

            safe_scales = np.where(
                np.abs(scales) < 1e-8,
                np.where(scales >= 0, 1e-8, -1e-8),
                scales,
            )
            W_rec = (W_mix - ratios[:, None] * mask[None, :]) / safe_scales[:, None]
            obj = np.linalg.norm(W_mix - (W_pub * scales[:, None] + ratios[:, None] * mask[None, :]), "fro")

            if best is None or obj < best[0]:
                best = (obj, W_rec, {"mask": mask, "scales": scales, "ratios": ratios})

        if best is None:
            return W_pub.copy(), {
                "mask": np.zeros(W_pub.shape[1]),
                "scales": np.ones(W_pub.shape[0]),
                "ratios": np.zeros(W_pub.shape[0]),
            }
        return best[1], best[2]

    def recover_one_vit_arrow(A_obf, A_pub):
        P_fwd = solve_vit_forward_permutation(A_pub, A_obf)
        rp = np.argmax(P_fwd, axis=1)
        A_mix = P_fwd.T @ A_obf
        A_rec, factors = solve_shared_mask_rows(A_mix, A_pub)
        return A_rec, rp, factors, None

    for name, module in model.named_parameters():
        if not _vit_obfus_weight(name):
            module.data = pre_state[name].data
            continue

        ob_w = module.data
        pre_w = pre_state[name].data

        if "qkv.weight" in name:
            pieces = []
            rpm, rLm, rDm = {}, {}, {}
            for tag, ob_slice, pre_slice in zip(
                ("q", "k", "v"),
                ob_w.chunk(3, dim=0),
                pre_w.chunk(3, dim=0),
            ):
                A_obf = ob_slice.cpu().numpy().astype(np.float64)
                A_pub = pre_slice.cpu().numpy().astype(np.float64)
                A_rec, rp, L_best, D_best = recover_one_vit_arrow(A_obf, A_pub)
                pieces.append(torch.from_numpy(A_rec).to(ob_slice.device).type_as(ob_slice))
                rpm[tag] = rp
                if L_best is not None:
                    rLm[tag] = L_best
                if D_best is not None:
                    rDm[tag] = D_best

            module.data = torch.cat(pieces, dim=0)
            restore_perm[name] = rpm
            restore_L[name] = rLm
            restore_D[name] = rDm

            if vic_model is not None:
                print(f"module name: {name}")
                A_rec_full = module.data.detach().cpu().numpy().astype(np.float64)
                A_pub_full = pre_w.cpu().numpy().astype(np.float64)
                A_vic_full = vic_state[name].data.cpu().numpy().astype(np.float64)
                for tag, pub, rec, vic in zip(
                    ("q", "k", "v"),
                    np.split(A_pub_full, 3, axis=0),
                    np.split(A_rec_full, 3, axis=0),
                    np.split(A_vic_full, 3, axis=0),
                ):
                    print(f"  [{tag}] error between public model and victim model: {np.linalg.norm(pub - vic, 'fro'):.4e}")
                    print(f"  [{tag}] error between recovered model and public model: {np.linalg.norm(rec - pub, 'fro'):.4e}")
                    print(f"  [{tag}] error between recovered model and victim model: {np.linalg.norm(rec - vic, 'fro'):.4e}")
            continue

        A_obf = ob_w.cpu().numpy().astype(np.float64)
        A_pub = pre_w.cpu().numpy().astype(np.float64)
        A_rec, rp, L_best, D_best = recover_one_vit_arrow(A_obf, A_pub)
        restore_perm[name] = rp
        if L_best is not None:
            restore_L[name] = L_best
        if D_best is not None:
            restore_D[name] = D_best
        module.data = torch.from_numpy(A_rec).to(ob_w.device).type_as(ob_w)

        if vic_model is not None:
            print(f"module name: {name}")
            A_vic = vic_state[name].data.cpu().numpy().astype(np.float64)
            print(f"error between public model and victim model: {np.linalg.norm(A_pub - A_vic, 'fro'):.4e}")
            print(f"error between recovered model and public model: {np.linalg.norm(A_rec - A_pub, 'fro'):.4e}")
            print(f"error between recovered model and victim model: {np.linalg.norm(A_rec - A_vic, 'fro'):.4e}")
    return model, restore_perm, restore_L, restore_D

def solve_admm_structured(A, B, rho=1.0, max_iter=500, alpha=1.6):
    m, n = A.shape
    print(f"ADMM Structured Solver: A shape = {A.shape}, rho = {rho}, max_iter = {max_iter}, alpha = {alpha}")
    
    # --- Initialize variables ---
    Z = np.zeros((m, m))
    L = np.zeros((m, m))
    S = np.zeros((m, m))
    Gamma = np.zeros((m, m))
    
    # Precompute cache to accelerate Z updates
    # (A^T A + rho*I) inverse
    AAt_rhoI_inv = np.linalg.inv(A @ A.T + rho * np.eye(m))
    BAt = B @ A.T
    
    min_loss = np.inf
    best_L, best_S = None, None
    for it in range(max_iter):
        # 1. Update Z (Global Step)
        Target_Z = (L + S) - (1/rho) * Gamma # D+uv^T is the structured term here
        Z = (BAt + rho * Target_Z) @ AAt_rhoI_inv
        Z = alpha * Z + (1.0 - alpha) * (L + S)
        
        # 2. Update D, u, v (Local Structure Step)
        # The objective is to fit T = Z + (1/rho)*Gamma at this step
        T = Z + (1/rho) * Gamma
        
        # 2
        L, S = rank1_permuted_diagonal_decomposition(T, S_init=S, max_iter=10, tol=1e-4)
        
        # 3. Update dual variable Gamma
        Residual_Constraint = Z - (L + S)
        Gamma = Gamma + rho * Residual_Constraint
        
        # Compute loss for monitoring
        curr_loss = np.linalg.norm((L + S) @ A - B, 'fro')
        prim_res = np.linalg.norm(Residual_Constraint, 'fro')
        
        if it % 100 == 0:
            print(f"Iter {it:2d}: Objective Loss = {curr_loss:.6f}, Primal Residual = {prim_res:.6f}")
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_L, best_S = L.copy(), S.copy()
        

    return best_L, best_S

def attack_finetune(
    model,
    trainloader,
    evalloader,
    num_classes,
    save_path,
    device,
    size=224,
    epochs=10,
    lr=5e-5,
    opt="adam",
    lr_scheduler_type="linear",
    lr_scheduler_warmup_ratio=0.1,
    weight_decay=0.01,
):
    set_seed()
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)

    def _evaluate_metrics():
        eval_loss, eval_acc = eval(model, evalloader, criterion, device)
        return eval_loss, eval_acc

    baseline_loss, baseline_acc = _evaluate_metrics()
    print(
        f"==> Attack finetune baseline: "
        f"Loss: {baseline_loss:.4f} | Accuracy: {baseline_acc:.4f}%"
    )

    if opt in ("adam", "adamw"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

    if lr_scheduler_type == "linear":
        warmup_steps = max(1, int(round(epochs * lr_scheduler_warmup_ratio)))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return max(0.0, 1.0 - float(step) / float(epochs))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Invalid lr scheduler type: {lr_scheduler_type}")

    print(
        f"==> Attack finetune: opt={opt} | lr={lr:.3e} | "
        f"weight_decay={weight_decay:.3e} | scheduler={lr_scheduler_type} | "
        f"warmup_ratio={lr_scheduler_warmup_ratio} | epochs={epochs}"
    )

    training_losses = []
    eval_losses = []
    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        epoch_progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training")
        tot_train_loss = 0  
        for batch_idx, (images, labels) in epoch_progress_bar:
            images, labels = images.to(device), labels.to(device)
            if images.size(-1) != size:
                images = torch.nn.functional.interpolate(images, size=(size, size))
            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            tot_train_loss += loss.item()
            epoch_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = tot_train_loss / len(trainloader)
        training_losses.append(avg_train_loss) 
           
        scheduler.step()  
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], Current LR: {current_lr:.6e}")
        
        avg_eval_loss, eval_acc = _evaluate_metrics()
        eval_losses.append(avg_eval_loss)
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Eval Loss: {avg_eval_loss:.6f} | Accuracy: {eval_acc:.4f}%"
        )
    final_loss, final_acc = _evaluate_metrics()
    print(
        f"Attack finetune final: Loss: {final_loss:.4f} | Accuracy: {final_acc:.4f}% "
        f"(baseline Loss: {baseline_loss:.4f} | Accuracy: {baseline_acc:.4f}%)"
    )
    save_model_weights(model, save_path)
    print(f"Model weights saved to {save_path}") 
    return model
