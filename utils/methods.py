from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt


def _apply_inv_perm_scale(ob_w, inv_perm, inv_scales, axis="col"):
    """Apply inverse permutation and per-row/column scaling to ob_w on GPU to avoid __array__ errors from mixing numpy and CUDA tensors."""
    device, dtype = ob_w.device, ob_w.dtype
    ip = torch.as_tensor(inv_perm, device=device, dtype=torch.long)
    sc = torch.as_tensor(inv_scales, device=device, dtype=dtype)
    if axis == "col":
        return ob_w[:, ip] * sc.view(1, -1)
    return ob_w[ip, :] * sc.view(-1, 1)

def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            layer_name = name.rsplit(".")[3]
            num_rows = w_q.shape[1]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation
            ob_w_q = w_q[:,permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_k = w_k[:,permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_v = w_v[:,permutation]
            module.data = ob_w_v
        elif "attention.output.dense.weight" in name:
            w_o = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)
            ob_o = w_o[inv_perm, :]
            module.data = ob_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            if(w_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                permutation = layer_permutations[layer_name]
                ob_proj = w_proj[:, permutation]
                module.data = ob_proj
    return model, layer_permutations, rows

def attack_translinkguard(model, pre_model, rows):
    set_seed()
    restore_perm = {}        
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            layer_name = name.rsplit(".")[3]
            pre_wq = pre_model.state_dict()[name]
            perm, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wq = ob_wq[:, inv_perm]
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wk = ob_wk[:, inv_perm]
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wv = ob_wv[:, inv_perm]
            module.data = restore_wv
        elif "attention.output.dense.weight" in name:
            ob_o = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            restore_o = ob_o[perm, :]
            module.data = restore_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            if(ob_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_proj = ob_proj[:, inv_perm]
                module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model        
    
def attack_translinkguard_our(model, pre_model, rows):
    set_seed()
    restore_perm = {}        
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            layer_name = name.rsplit(".")[3]
            pre_wq = pre_model.state_dict()[name]
            perm = col_restore_perm_our(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wq = ob_wq[:, inv_perm]
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wk = ob_wk[:, inv_perm]
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wv = ob_wv[:, inv_perm]
            module.data = restore_wv
        elif "attention.output.dense.weight" in name:
            ob_o = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            restore_o = ob_o[perm, :]
            module.data = restore_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            if(ob_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_proj = ob_proj[:, inv_perm]
                module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
        
    return model, restore_perm   

def ob_tsqp(model):
    set_seed()  
    scaling_factors = {}  
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            scale_q = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            module.data = w_q
            scaling_factors[name] = scale_q
        elif "key.weight" in name:
            w_k = module.data
            scale_k = 1 + 5 * torch.rand(1).item()
            w_k *= scale_k
            module.data = w_k
            scaling_factors[name] = scale_k
        elif "value.weight" in name:
            w_v = module.data
            scale_v = 1 + 5 * torch.rand(1).item()
            w_v *= scale_v
            module.data = w_v
            scaling_factors[name] = scale_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
    return model, scaling_factors

def attack_tsqp(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:      
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_soter(model, pre_model):
    set_seed()
    scaling_factors = {}  
    init_layers = {}
    # Randomly select some weight blocks and load pre_model weights,ratio is 20%
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            init_layers[name] = module.data
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            scale_q = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            module.data = w_q
            scaling_factors[name] = scale_q
        elif "key.weight" in name:
            w_k = module.data
            scale_k = 1 + 5 * torch.rand(1).item()
            w_k *= scale_k
            module.data = w_k
            scaling_factors[name] = scale_k
        elif "value.weight" in name:
            w_v = module.data
            scale_v = 1 + 5 * torch.rand(1).item()
            w_v *= scale_v
            module.data = w_v
            scaling_factors[name] = scale_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def attack_soter_our(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            k = pre_proj.flatten().dot(w_proj.flatten()) / w_proj.flatten().dot(w_proj.flatten())
            k = 1.0 / k
            k = fix_factor(k)
            new_w_proj = w_proj / k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors
    

def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[1]
            ratios_q = []
            for i in range(num_rows):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[:,i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_q = w_q[:, permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[1]
            ratios_k = []
            for i in range(num_rows):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[:,i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_k = w_k[:, permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[1]
            ratios_v = []
            for i in range(num_rows):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[:,i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_v = w_v[:, permutation]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
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
            ob_proj = w_proj[:, permutation]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_shadownet(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            ## restore_wq is wq after permutation recovery
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq[:,i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[1]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
                restore_wk[:,i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[1]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
                restore_wv[:,i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = col_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[1]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[:,i]).item()/torch.var(pre_proj[:,i]).item()))
                restore_proj[:,i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_shadownet_our(model, pre_model):
    set_seed()
    restore_perm = {}
    restore_scales = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            perm, scales = col_restore_perm_and_scale(pre_wq, ob_wq)
            restore_perm[name] = perm
            restore_scales[name] = scales
            inv_perm = np.argsort(perm)
            inv_scales = 1 / scales
            restore_wq = _apply_inv_perm_scale(ob_wq, inv_perm, inv_scales, axis="col")
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            perm, scales = col_restore_perm_and_scale(pre_wk, ob_wk)
            restore_perm[name] = perm
            restore_scales[name] = scales
            inv_perm = np.argsort(perm)
            inv_scales = 1 / scales
            restore_wk = _apply_inv_perm_scale(ob_wk, inv_perm, inv_scales, axis="col")
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            perm, scales = col_restore_perm_and_scale(pre_wv, ob_wv)
            restore_perm[name] = perm
            restore_scales[name] = scales
            inv_perm = np.argsort(perm)
            inv_scales = 1 / scales
            restore_wv = _apply_inv_perm_scale(ob_wv, inv_perm, inv_scales, axis="col")
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            perm, scales = col_restore_perm_and_scale(pre_proj, ob_proj)
            restore_perm[name] = perm
            restore_scales[name] = scales
            inv_perm = np.argsort(perm)
            inv_scales = 1 / scales
            restore_proj = _apply_inv_perm_scale(ob_proj, inv_perm, inv_scales, axis="col")
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm, restore_scales

def ob_LoRO(model, pre_model=None, r=8, noise=1e-1):
    set_seed()
    R = {}
    pre_state = pre_model.state_dict() if pre_model is not None else None
    printed_debug = False
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w = module.detach()
                d1, d2 = w.shape
                if pre_state is not None:
                    wp = pre_state[name].detach()
                    dw = wp.cpu().to(dtype=torch.float64) - w.cpu().to(dtype=torch.float64)
                    if not torch.isfinite(dw).all():
                        raise ValueError(f"ob_LoRO received non-finite values in {name}")
                    U, S, Vt = torch.linalg.svd(dw, full_matrices=False)
                    C = (U[:, :r] * S[:r]) @ Vt[:r, :]
                    low_rank_matrix = C.to(dtype=w.dtype)
                    if not printed_debug:
                        before = torch.linalg.norm(dw).item()
                        after = torch.linalg.norm(dw - C).item()
                        print(f"[LoRO debug] {name}: before={before:.6g}, after={after:.6g}, rank={r}, device={w.device}, dtype={w.dtype}")
                        printed_debug = True
                else:
                    A = torch.randn(d1, r, device=w.device, dtype=w.dtype) * noise
                    B = torch.randn(r, d2, device=w.device, dtype=w.dtype) * noise
                    low_rank_matrix = torch.matmul(A, B).detach().cpu()
                module.copy_(w + low_rank_matrix.to(device=w.device, dtype=w.dtype))
                R[name] = low_rank_matrix.cpu()
    return model, R

def ob_LRAO(model, pre_model, r=8):
    set_seed()
    R = {}
    pre_state = pre_model.state_dict() if pre_model is not None else None
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w = module.detach()
                d1, d2 = w.shape
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
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                ob_w = module.detach().clone()
                module.copy_(ob_w.to(device=module.device, dtype=module.dtype))
                pre_w = pre_state[name].detach().clone()
                if vic_model is not None:
                    vic_w = vic_state[name].detach()
                    print(f"layer {name} :")
                    error = torch.norm(ob_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between obfuscated model and public model: {error.item()}")
                    error = torch.norm(vic_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between victim model and public model: {error.item()}")
            else:
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
    return model

def attack_LoRO(model, pre_model, vic_model=None, r=8):
    set_seed()
    restore_R = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                ob_w = module.detach().clone()
                pre_w = pre_state[name].detach()
                restore_w, R = restore_low_rank(pre_w, ob_w, r)
                restore_R[name] = R.cpu()
                module.copy_(restore_w.to(device=module.device, dtype=module.dtype))
                if vic_model is not None:
                    vic_w = vic_state[name].detach()
                    print(f"layer {name} :")
                    error = torch.norm(ob_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between obfuscated model and public model: {error.item()}")
                    error = torch.norm(vic_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between victim model and public model: {error.item()}")
                    error = torch.norm(restore_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between recovered model and public model: {error.item()}")
                    error = torch.norm(restore_w.cpu() - vic_w.cpu(), p='fro')
                    print(f"error between recovered model and victim model: {error.item()}")
            else:
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
    return model, restore_R

def ob_obfuscatune(model):
    set_seed()
    ob_Q = {}
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w = module.detach()
                # Generate a CPU float64 orthogonal matrix to avoid recovery instability across GPU/BLAS backends.
                _, d2 = w.shape
                Q, _ = torch.linalg.qr(torch.randn(d2, d2, dtype=torch.float64))
                Q = Q.to(dtype=w.dtype)
                module.copy_(w @ Q.to(device=w.device, dtype=w.dtype))
                ob_Q[name] = Q.cpu()
    return model, ob_Q
    

def attack_obfuscatune(model, pre_model, vic_model=None):
    set_seed()
    restore_Q = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                ob_w = module.detach().clone()
                pre_w = pre_state[name].detach()
                restore_w, Q = restore_orthogonal(pre_w, ob_w)
                restore_Q[name] = Q.cpu()
                module.copy_(restore_w.to(device=module.device, dtype=module.dtype))
                if vic_model is not None:
                    vic_w = vic_state[name].detach()
                    print(f"layer {name} :")
                    error = torch.norm(vic_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between victim model and public model: {error.item()}")
                    error = torch.norm(restore_w.cpu() - pre_w.cpu(), p='fro')
                    print(f"error between recovered model and public model: {error.item()}")
                    error = torch.norm(restore_w.cpu() - vic_w.cpu(), p='fro')
                    print(f"error between recovered model and victim model: {error.item()}")
            else:
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
    return model, restore_Q
    

def ob_tempo(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_cols = w_q.shape[0]
            ratios_q = []
            for i in range(num_cols):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_q = w_q[permutation,:]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_cols = w_k.shape[0]
            ratios_k = []
            for i in range(num_cols):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_k = w_k[permutation,:]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_cols = w_v.shape[0]
            ratios_v = []
            for i in range(num_cols):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_v = w_v[permutation,:]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
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
            ob_proj = w_proj[permutation,:]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            ## restore_wq is wq after permutation recovery
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item()))
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item()))
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item()))
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item()))
                restore_proj[i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_tempo_our(model, pre_model):
    set_seed()
    restore_perm = {}
    restore_scales = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            perm, scales = row_restore_perm_and_scale(pre_w, ob_w)
            restore_perm[name] = perm
            restore_scales[name] = scales
            inv_perm = np.argsort(perm)
            inv_scales = 1 / scales
            # ob_tempo permutes rows as ob_w[perm, :]; recover by row and apply per-row scaling here.
            restore_w = _apply_inv_perm_scale(ob_w, inv_perm, inv_scales, axis="row")
            module.data = restore_w
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm, restore_scales

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[0]
            device = w_q.device
            coeff = torch.randint(0,5,(w_q.shape[0],), device=device)
            mask = torch.matmul(w_q.T, coeff.float())
            layer_masks[name] = mask
            ratios_q = []
            ratios_q2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= ratio2
                mask_qi = mask * ratio
                w_q[i] += mask_qi
                ratios_q.append(ratio)
                ratios_q2.append(ratio2)
            layer_factors[name] = ratios_q
            weight_factors[name] = ratios_q2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_q = w_q[permutation]
            module.data = w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[0]
            device = w_k.device
            coeff = torch.randint(0,5,(w_k.shape[0],), device=device)
            mask = torch.matmul(w_k.T, coeff.float())
            layer_masks[name] = mask
            ratios_k = []
            ratios_k2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_k[i] *= ratio2
                mask_ki = mask * ratio
                w_k[i] += mask_ki
                ratios_k.append(ratio)
                ratios_k2.append(ratio2)
            layer_factors[name] = ratios_k
            weight_factors[name] = ratios_k2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_k = w_k[permutation]
            module.data = w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[0]
            device = w_v.device
            coeff = torch.randint(0,5,(w_v.shape[0],), device=device)
            mask = torch.matmul(w_v.T, coeff.float())
            layer_masks[name] = mask
            ratios_v = []
            ratios_v2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_v[i] *= ratio2
                mask_vi = mask * ratio
                w_v[i] += mask_vi
                ratios_v.append(ratio)
                ratios_v2.append(ratio2)
            layer_factors[name] = ratios_v
            weight_factors[name] = ratios_v2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_v = w_v[permutation]
            module.data = w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[0]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_proj = w_proj[permutation]
            module.data = w_proj
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def attack_arrowcloak(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item())
                restore_proj[i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

            
            
                
                
                
                
            

def _prepare_right_ridge_operator(A, rho):
    """Precompute the small matrix inverse needed by the right-side ridge solve."""
    n, m = A.shape
    if n <= m:
        system_inv = np.linalg.inv(A @ A.T + rho * np.eye(n, dtype=A.dtype))
        return "row", system_inv

    system_inv = np.linalg.inv(A.T @ A + rho * np.eye(m, dtype=A.dtype))
    return "col", system_inv

def _apply_right_ridge_operator(C, A, ridge_operator):
    """
    Compute C @ A.T @ (A @ A.T + rho I)^-1 and automatically use the equivalent smaller-dimensional form.
    Here C has shape (*, A.shape[1])。
    """
    mode, system_inv = ridge_operator
    if mode == "row":
        return (C @ A.T) @ system_inv
    return (C @ system_inv) @ A.T

def rank1_permuted_diagonal_decomposition(A, S_init=None, max_iter=50, tol=1e-6):
    m, _ = A.shape
    if S_init is not None:
        S = S_init.copy()
    else:
        S = np.zeros_like(A)

    prev_error = np.inf
    P = np.eye(m)

    for _ in range(max_iter):
        # Under the fixed permutation-structure approximation, update the rank-1 term first
        R = A - S
        U, s, Vt = svds(R, k=1)
        L = s[0] * np.outer(U[:, 0], Vt[0, :])

        # then use Hungarian matching to update the diagonal term on the permutation structure
        E = A - L
        cost_matrix = -(E ** 2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        S_new = np.zeros_like(A)
        S_new[row_ind, col_ind] = E[row_ind, col_ind]

        error = np.linalg.norm(A - L - S_new, "fro")
        if abs(prev_error - error) < tol:
            S = S_new
            P = np.zeros((m, m))
            P[row_ind, col_ind] = 1.0
            break

        prev_error = error
        S = S_new
        P = np.zeros((m, m))
        P[row_ind, col_ind] = 1.0

    return L, S, P

def solve_permutation_projection(A, B, rho=1.0, alpha=1.0, inner_iter=30, inner_tol=1e-4):
    print(f"[Stage 1]Solving permutation projection: shape={A.shape}, rho = {rho}, alpha = {alpha}, iter = {inner_iter}, tol = {inner_tol}")
    ridge_operator = _prepare_right_ridge_operator(A, rho)
    ridge_operator = _prepare_right_ridge_operator(A, rho)
    T = alpha * _apply_right_ridge_operator(B, A, ridge_operator)
    L, S, P = rank1_permuted_diagonal_decomposition(
        T, max_iter=inner_iter, tol=inner_tol
    )
    loss = np.linalg.norm((L + S) @ A - B, "fro")
    print(f"[Stage1] Projection objective={loss:.6f}")
    return P

def rank1_diagonal_decomposition(T, S_init=None, max_iter=30, tol=1e-6):
    """
    Local structure projection: min ||T - (L + D)||_F subject to rank(L)=1 and diagonal D.
    """
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
    verbose_every = max_iter // 10
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

    def is_bert_arrowcloak_weight(name):
        return (
            "query.weight" in name
            or "key.weight" in name
            or "value.weight" in name
            or "output.dense.weight" in name
            or "intermediate.dense.weight" in name
        )

    def solve_bert_forward_permutation(A_pub, A_obf):
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
                L, S, _ = rank1_permuted_diagonal_decomposition(
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
            obj = np.linalg.norm(
                W_mix - (W_pub * scales[:, None] + ratios[:, None] * mask[None, :]),
                "fro",
            )

            if best is None or obj < best[0]:
                best = (obj, W_rec, {"mask": mask, "scales": scales, "ratios": ratios})

        if best is None:
            return W_pub.copy(), {
                "mask": np.zeros(W_pub.shape[1]),
                "scales": np.ones(W_pub.shape[0]),
                "ratios": np.zeros(W_pub.shape[0]),
            }
        return best[1], best[2]

    def recover_one_bert_arrow(A_obf, A_pub):
        P_fwd = solve_bert_forward_permutation(A_pub, A_obf)
        rp = np.argmax(P_fwd, axis=1)
        A_mix = P_fwd.T @ A_obf
        A_rec, factors = solve_shared_mask_rows(A_mix, A_pub)
        return A_rec, rp, factors, None

    for name, module in model.named_parameters():
        if not is_bert_arrowcloak_weight(name):
            module.data = pre_state[name].data
            continue

        ob_w = module.data
        pre_w = pre_state[name].data
        A_obf = ob_w.cpu().numpy().astype(np.float64)
        A_pub = pre_w.cpu().numpy().astype(np.float64)
        A_rec, rp, L_best, D_best = recover_one_bert_arrow(A_obf, A_pub)
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


from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
def rank1_permuted_diagonal_decomposition(A, S_init=None, max_iter=50, tol=1e-6):
    m, _ = A.shape
    if S_init is not None:
        S = S_init.copy()
    else:
        S = np.zeros_like(A)

    prev_error = np.inf
    P = np.eye(m)

    for _ in range(max_iter):
        # Under the fixed permutation-structure approximation, update the rank-1 term first
        R = A - S
        U, s, Vt = svds(R, k=1)
        L = s[0] * np.outer(U[:, 0], Vt[0, :])

        # then use Hungarian matching to update the diagonal term on the permutation structure
        E = A - L
        cost_matrix = -(E ** 2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        S_new = np.zeros_like(A)
        S_new[row_ind, col_ind] = E[row_ind, col_ind]

        error = np.linalg.norm(A - L - S_new, "fro")
        if abs(prev_error - error) < tol:
            S = S_new
            P = np.zeros((m, m))
            P[row_ind, col_ind] = 1.0
            break

        prev_error = error
        S = S_new
        P = np.zeros((m, m))
        P[row_ind, col_ind] = 1.0

    return L, S, P

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
        L, S, _ = rank1_permuted_diagonal_decomposition(T, S_init=S, max_iter=10, tol=1e-4)
        
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

from sklearn.metrics.pairwise import cosine_similarity
def cluster_vectors(vectors, cluster_size=4):
    index_pairs = np.array([np.array([i]) for i in range(len(vectors))])
    iter = 1
    while(iter<cluster_size):
        iter*=2
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
            pairs.append(np.mean([vectors[i], vectors[j]],axis=0))
        vectors = pairs
        index_pairs = index_pair 
    return index_pairs



def ob_groupcover(model, size=4):
    set_seed()
    ob_cluster_index = {}
    ob_random_coeff_list = {}
    ob_permutation = {}
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w_tensor = module.detach()
                device = w_tensor.device
                dtype = w_tensor.dtype
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

                    for index, idx in enumerate(idlist):
                        new_w[idx, :] = new_kernels[index]
                permutation = torch.randperm(d2)
                new_w = torch.from_numpy(new_w).to(device=device, dtype=dtype)[:, permutation.to(device)]
                module.copy_(new_w)

                ob_cluster_index[name] = [set(map(int, idlist)) for idlist in cluster_index]
                ob_random_coeff_list[name] = random_coeff_list
                ob_permutation[name] = permutation.cpu().numpy()

    return model, ob_cluster_index, ob_random_coeff_list, ob_permutation
    

def _torch_pinv_np(matrix):
    return np.linalg.pinv(matrix, rcond=1e-15)


from scipy.linalg import orthogonal_procrustes
def solve_AK_PB(A, B, max_iter=20):
    """
    Solve min || A K - P B ||
    where P is a row permutation matrix。
    
    Idea: 
    1. Eliminate K: convert to matching subspaces U_A approx P U_B Q
    2. use leverage scores initialize P
    3. alternate optimization of P (assignment) and Q (rotation)
    """
    n, m = A.shape

    U_A, _ = np.linalg.qr(A, mode='reduced')
    U_B, _ = np.linalg.qr(B, mode='reduced')

    lev_A = np.linalg.norm(U_A, axis=1)**2
    lev_B = np.linalg.norm(U_B, axis=1)**2
    
    Cost_init = np.abs(lev_A[:, None] - lev_B[None, :])
    _, col_ind = linear_sum_assignment(Cost_init)
    perm = col_ind.astype(np.int64, copy=False)
    
    for it in range(max_iter):
        Source = U_B[perm]
        Q, _ = orthogonal_procrustes(Source, U_A)

        Transformed_B = U_B @ Q
        
        Cost = U_A @ Transformed_B.T
        
        _, col_ind = linear_sum_assignment(Cost, maximize=True)
        new_perm = col_ind.astype(np.int64, copy=False)

        if np.array_equal(new_perm, perm):
            break
        perm = new_perm

    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from itertools import combinations
import warnings
from sklearn.exceptions import ConvergenceWarning
def recover_K_and_P2(A_obf, B, cluster_index, th=0.15, step=0.01, max_th=4.0, size=4):
    if step <= 0:
        raise ValueError("step must be positive for recover_K_and_P2")
    n, m = A_obf.shape
    
    l = [i for i in range(n)]
    K_est = np.zeros((n, n))
    P2_est = np.zeros((n, n))
    idx = 0 

    for idlist in cluster_index:
        
        perm = idlist
        Ai = A_obf[list(perm), :]
        Bi = B[list(perm), :]
        Ki = Bi @ _torch_pinv_np(Ai)
        diff = np.linalg.norm(Ki @ Ai - Bi) / np.linalg.norm(Bi)
        if diff < th:
            K_est[idx:idx+size, idx:idx+size] = Ki
            P2_est[list(perm), idx:idx+size] = np.eye(size)
            l = [x for x in l if x not in perm]
            idx += size
    
    max_rounds = max(1, int(np.ceil((max_th - th) / step)) + 2)
    rounds = 0
    while True:
        rounds += 1
        flag = False
        for i in l:
            if i not in l:
                continue
            bi = B[i, :]
            id = l.index(i)
            A = A_obf[l, :].T
            b = bi.T
            pre_select_n = size * 4
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scaler = StandardScaler()
            A = scaler.fit_transform(A)
            lasso = Lasso(alpha=0.00001, max_iter=1000)
            lasso.fit(A, b)
            importance = np.abs(lasso.coef_)
            candidate_indices = np.argsort(importance)[-pre_select_n:]
            
            if id not in candidate_indices:
                continue
            # remove id from candidate_indices
            candidate_indices = [idx for idx in candidate_indices if idx != id]
            
            for indices in combinations(candidate_indices, size-1):
                # add id to indices
                indices = list(indices)
                indices.append(id)
                perm = [l[x] for x in indices]
                Ai = A_obf[list(perm), :]
                Bi = B[list(perm), :]
                Ki = Bi @ _torch_pinv_np(Ai)
                diff = np.linalg.norm(Ki @ Ai - Bi) / np.linalg.norm(Bi)
                if diff < th:
                    K_est[idx:idx+size, idx:idx+size] = Ki
                    P2_est[list(perm), idx:idx+size] = np.eye(size)
                    l = [x for x in l if x not in perm]
                    idx += size
                    flag = True
                    break
        if not flag:
            th += step
        if len(l) == 0 or th > max_th:
            break
    
    if len(l) != 0:
        return None, None
    
    return K_est, P2_est

def _recover_groupcover_fast(
    A_obf,
    A_pub,
    cluster_index,
    size=4,
    th=0.5,
    p1_probe_limit=2,
    min_cluster_recover_ratio=0.05,
    max_p1_cols=1024,
):
    n_rows, n_cols = A_obf.shape
    if n_cols > max_p1_cols:
        print(f"[groupcover-fast] n_cols={n_cols} too large; skipping column-permutation estimation")
        return None, None, None

    P1_candidates = []
    scored_candidates = []
    best_diff = float("inf")
    for idlist in cluster_index[:p1_probe_limit]:
        perm = list(idlist)
        Ai = A_obf[perm, :]
        Bi = A_pub[perm, :]
        P1 = solve_AK_PB(Bi.T, Ai.T, max_iter=5)
        Bi_perm = Bi @ P1
        Ki = Bi_perm @ _torch_pinv_np(Ai)
        denom = np.linalg.norm(Bi_perm) + 1e-12
        diff = np.linalg.norm(Ki @ Ai - Bi_perm) / denom
        best_diff = min(best_diff, diff)
        if diff < max(0.2, th):
            P1_candidates.append(P1.T)
            scored_candidates.append(diff)

    if len(P1_candidates) == 0:
        print(f"[groupcover-fast] no suitable P1_est found; minimum candidate residual: {best_diff:.6f}")
        return None, None, None

    best = None
    best_score = float("inf")
    best_clusters = []
    for P1_est, p1_diff in zip(P1_candidates, scored_candidates):
        B = A_pub @ P1_est.T
        total_diff = 0.0
        recovered = []
        A_guess = A_pub.copy()
        for idlist in cluster_index:
            perm = list(idlist)
            Ai = A_obf[perm, :]
            Bi = B[perm, :]
            Ki = Bi @ _torch_pinv_np(Ai)
            denom = np.linalg.norm(Bi) + 1e-12
            diff = np.linalg.norm(Ki @ Ai - Bi) / denom
            total_diff += min(diff, th)
            if diff < th:
                A_guess[perm, :] = Ki @ Ai @ P1_est
                recovered.append(set(map(int, perm)))

        recover_ratio = len(recovered) / max(1, len(cluster_index))
        score = total_diff / max(1, len(cluster_index)) + 0.01 * p1_diff - 0.05 * recover_ratio
        if score < best_score:
            best = (A_guess, P1_est)
            best_score = score
            best_clusters = recovered

    if best is None:
        return None, None, None

    if len(best_clusters) / max(1, len(cluster_index)) < min_cluster_recover_ratio:
        print(
            f"[groupcover-fast] recovered-cluster ratio is too low: "
            f"{len(best_clusters)}/{len(cluster_index)}"
        )
        return None, None, None

    A_guess, P1_est = best
    inv_perm = np.argmax(P1_est, axis=1)
    print(
        f"[groupcover-fast] using fast cluster-level recovery: "
        f"{len(best_clusters)}/{len(cluster_index)} clusters, score={best_score:.6f}"
    )
    return A_guess, inv_perm, best_clusters


def _recover_groupcover_partial(
    A_obf,
    A_pub,
    P1_est,
    cluster_index,
    th=0.5,
    min_cluster_recover_ratio=0.05,
    max_pub_shift=0.05,
):
    B = A_pub @ P1_est.T
    A_guess = A_pub.copy()
    recovered = []
    diffs = []
    for idlist in cluster_index:
        perm = list(idlist)
        Ai = A_obf[perm, :]
        Bi = B[perm, :]
        Ki = Bi @ _torch_pinv_np(Ai)
        denom = np.linalg.norm(Bi) + 1e-12
        diff = np.linalg.norm(Ki @ Ai - Bi) / denom
        diffs.append(diff)
        if diff < th:
            A_guess[perm, :] = Ki @ Ai @ P1_est
            recovered.append(set(map(int, perm)))

    recover_ratio = len(recovered) / max(1, len(cluster_index))
    if recover_ratio < min_cluster_recover_ratio:
        print(
            f"[groupcover-partial] recovered-cluster ratio is too low: "
            f"{len(recovered)}/{len(cluster_index)}, best_diff={min(diffs) if diffs else float('inf'):.6f}"
        )
        return None, None
    pub_shift = np.linalg.norm(A_guess - A_pub) / (np.linalg.norm(A_pub) + 1e-12)
    if pub_shift > max_pub_shift:
        print(
            f"[groupcover-partial] recovery deviates too much from public weights: "
            f"{pub_shift:.6f} > {max_pub_shift:.6f}"
        )
        return None, None

    print(
        f"[groupcover-partial] using partial high-confidence recovery: "
        f"{len(recovered)}/{len(cluster_index)} clusters, pub_shift={pub_shift:.6f}"
    )
    return A_guess, recovered

















def recover_groupcover(A_obf, A_pub, size=4, th=0.1, max_th=0.5, step=0.1):
    n_rows, n_cols = A_obf.shape
    P1_est = None

    cluster_index = cluster_vectors(A_pub, cluster_size=size)

    P1_list = []
    perm_list = []
    best_diff = float("inf")
    for idlist in cluster_index:
        perm = idlist
        Ai = A_obf[list(perm), :]
        Bi = A_pub[list(perm), :]
        P1 = solve_AK_PB(Bi.T, Ai.T, max_iter=50)
        Bi = Bi @ P1
        Ki = Bi @ _torch_pinv_np(Ai)
        diff = np.linalg.norm(Ki @ Ai - Bi) / np.linalg.norm(Bi)

        if diff < th * 3:
            P1_list.append(P1)
            perm_list.append(perm)
        if diff < th / 2:
            break
        best_diff = min(best_diff, diff)

    if len(P1_list) == 0:
        print(f"no suitable P1_est found; minimum candidate residual: {best_diff:.6f}")
        return None, None, None

    min_diff = float("inf")
    for P1 in P1_list:
        total_diff = 0
        for perm in perm_list:
            Ai = A_obf[list(perm), :]
            Bi = A_pub[list(perm), :]
            Bi = Bi @ P1
            Ki = Bi @ _torch_pinv_np(Ai)
            diff = np.linalg.norm(Ki @ Ai - Bi) / np.linalg.norm(Bi)
            total_diff += diff
        if total_diff < min_diff:
            P1_est = P1.T
            min_diff = total_diff


    if min_diff / len(perm_list) > th * 3:
        print(f"selected P1_est has too large an average residual: {min_diff / len(perm_list):.6f}")
        return None, None, None

    print(f"selected P1_est, residual: {min_diff / len(perm_list):.6f}")

    B_fix = A_pub @ P1_est.T
    K_est, P2_est = recover_K_and_P2(A_obf, B_fix, cluster_index, th=0.1, step=0.3, max_th=4, size=size)


    A_guess = P2_est @ K_est @ P2_est.T @ A_obf @ P1_est

    A = P2_est @ K_est @ P2_est.T @ A_obf
    Bm = A_pub
    cost_matrix = - (A.T @ Bm)
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

    def extract_p2_row_clusters(P2_est, m, size):
        clusters = []
        for c in range(0, m, size):
            block = P2_est[:, c : c + size]
            if block.size == 0:
                continue
            if np.max(np.abs(block)) < 1e-12:
                continue
            rows = np.where(np.any(np.abs(block) > 1e-12, axis=1))[0]
            if len(rows) == size:
                clusters.append(set(rows.tolist()))
        return clusters

    inv_perm = np.argmax(P1_est, axis=1)
    cluster_index_out = extract_p2_row_clusters(P2_est, n_rows, size)

    return A_guess, inv_perm, cluster_index_out

def attack_groupcover(model, pre_model, size=4, vic_model=None, th=0.1, max_th=0.5, fast=False, fast_only=False, partial=False):
    set_seed()
    restore_permutation = {}
    restore_cluster_index = {}
    pre_state = pre_model.state_dict()
    vic_state = vic_model.state_dict() if vic_model is not None else None
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
                ob_w = module.detach().clone()
                pre_w = pre_state[name].detach()
                A_obf = ob_w.cpu().numpy().astype(np.float64, copy=False)
                A_pub = pre_w.cpu().numpy().astype(np.float64, copy=False)
                A_guess, inv_perm, cluster_index = recover_groupcover(
                    A_obf,
                    A_pub,
                    size=size,
                    th=th,
                    max_th=max_th,
                )
                if A_guess is None or inv_perm is None or cluster_index is None:
                    print(f"no suitable A_guess, inv_perm, cluster_index found for {name}")
                    module.copy_(pre_w.to(device=module.device, dtype=module.dtype))
                    continue
                restore_permutation[name] = inv_perm
                restore_cluster_index[name] = cluster_index
                module.copy_(torch.from_numpy(A_guess).to(device=module.device, dtype=module.dtype))
                if vic_model is not None:
                    print(f"name: {name}")
                    vic_w = vic_state[name].detach()
                    error = torch.norm(pre_w.cpu() - vic_w.cpu(), p='fro').item()
                    print(f"    error between public model and victim model: {error:.4e}")
                    error = torch.norm(module.detach().cpu() - pre_w.cpu(), p='fro').item()
                    print(f"    error between recovered model and public model: {error:.4e}")
                    error = torch.norm(module.detach().cpu() - vic_w.cpu(), p='fro').item()
                    print(f"    error between recovered model and victim model: {error:.4e}")
            else:
                module.copy_(pre_state[name].detach().to(device=module.device, dtype=module.dtype))
    return model, restore_permutation, restore_cluster_index

def ob_twinshield(model):
    set_seed()
    ob_permutation = {}
    ob_d = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w = module.data
            device = w.device
            dtype = w.dtype
            Wv = w.detach().cpu().numpy().astype(np.float64, copy=False)
            n, m = Wv.shape

            R = np.random.randn(n, m)
            d_true = np.random.randn(n)
            P_true_indices = np.random.permutation(2 * m)
            P = np.zeros((2 * m, 2 * m), dtype=np.float64)
            P[P_true_indices, np.arange(2 * m)] = 1.0

            Wo = np.concatenate([Wv + R, d_true[:, None] * R], axis=1) @ P

            module.data = torch.from_numpy(Wo).to(device=device, dtype=dtype)
            ob_permutation[name] = P_true_indices
            ob_d[name] = d_true

    return model, ob_permutation, ob_d

def solve_D_P_alternating(Wo1, Wo2, Wp, max_iter=50, tol=1e-6):
    """
    Solve min_{D,P} || Wo1 + D @ Wo2 - Wp @ P ||_F^2
    where D is a diagonal matrix acting on rows and P is a permutation matrix acting on columns.
    """
    n, m = Wo1.shape
    d = np.zeros(n)
    P_indices = np.arange(m)
    prev_error = float('inf')

    for _ in range(max_iter):
        W_current = Wo1 + d[:, None] * Wo2
        W_curr_norm = np.sum(W_current**2, axis=0).reshape(-1, 1)
        Wp_norm = np.sum(Wp**2, axis=0).reshape(1, -1)
        cost_matrix = W_curr_norm + Wp_norm - 2 * (W_current.T @ Wp)
        _, col_ind = linear_sum_assignment(cost_matrix)
        P_indices = col_ind

        Wp_permuted = Wp[:, P_indices]
        target = Wp_permuted - Wo1
        num = np.sum(Wo2 * target, axis=1)
        den = np.sum(Wo2**2, axis=1)
        safe_den = np.where(den < 1e-12, 1.0, den)
        d = np.where(den < 1e-12, 0.0, num / safe_den)

        current_error = np.linalg.norm(Wo1 + d[:, None] * Wo2 - Wp_permuted, 'fro')
        if prev_error - current_error < tol:
            break
        prev_error = current_error

    return d, P_indices, current_error

def solve_twinshield(W_ob, Wp, sigma_R=1.0, eps=1e-12, min_abs_d=0.2, refine_iter=5):
    """
    Attack:
        W_ob = [Wv + R, D @ R] @ P

    Given:
        W_ob: shape (n, 2m)
        Wp:   shape (n, m)

    Return:
        restore_w: recovered Wv, shape (n, m)
        d_calc:    recovered row-wise D diagonal, shape (n,)
        p_calc:    p_calc[observed_col] = original_col
                   consistent with:
                       P[P_true_indices, np.arange(2*m)] = 1
    """
    n, two_m = W_ob.shape
    assert two_m % 2 == 0
    m = two_m // 2
    assert Wp.shape == (n, m)

    # ============================================================
    # 1. Estimate |D| from row-wise second moments.
    #
    # Each row of W_ob contains half from Wv+R and half from d_i R.
    # Approximately:
    #   mean(W_ob_i^2) ≈ 1/2 * (mean(Wv_i^2) + sigma_R^2 + d_i^2 sigma_R^2)
    #
    # Therefore:
    #   d_i^2 ≈ (2 mean(W_ob_i^2) - mean(Wp_i^2) - sigma_R^2) / sigma_R^2
    # ============================================================
    row_mean_ob_sq = np.mean(W_ob ** 2, axis=1)
    row_mean_wp_sq = np.mean(Wp ** 2, axis=1)

    abs_d_sq = (2.0 * row_mean_ob_sq - row_mean_wp_sq - sigma_R ** 2) / (sigma_R ** 2)
    abs_d = np.sqrt(np.maximum(abs_d_sq, 1e-9))

    # Use stable rows for pairing.
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

    # This is an undirected disjoint pairing problem.
    # Use greedy matching on the symmetrized cost.
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

        # Orient the pair:
        # u should be Wv+R, v should be D R.
        if M[a, b] <= M[b, a]:
            u, v = a, b
        else:
            u, v = b, a

        pairs.append((u, v))
        used[a] = True
        used[b] = True

        if len(pairs) == m:
            break

    if len(pairs) != m:
        raise RuntimeError(f"Only found {len(pairs)} pairs, expected {m}.")

    c1 = np.array([p[0] for p in pairs], dtype=np.int64)  # Wv + R columns
    c2 = np.array([p[1] for p in pairs], dtype=np.int64)  # D R columns

    A = W_ob[:, c1]
    B = W_ob[:, c2]

    # ============================================================
    # 3. Initial signed D estimation.
    #
    # Before knowing the column order, estimate:
    #   B_i ≈ d_i A_i
    #
    # This is not exact because A_i = Wv_i + R_i,
    # but R dominates in your setting, so it gives a good sign and scale.
    # ============================================================
    d_calc = np.sum(A * B, axis=1) / (np.sum(A ** 2, axis=1) + eps)

    sign = np.sign(d_calc)
    sign[sign == 0] = 1.0
    d_calc = np.where(np.abs(d_calc) < 1e-8, sign * abs_d, d_calc)

    # ============================================================
    # 4. Reconstruct unordered Wv, then align to Wp.
    #
    # Important:
    #   Do NOT align pairs to Wp before reconstruction.
    #   The signal is weak because R dominates.
    #
    # Instead:
    # then match restored columns to Wp.
    # ============================================================
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

    # ============================================================
    # 5. Final reconstruction.
    # ============================================================
    safe_d = np.where(np.abs(d_calc) < 1e-8, np.sign(d_calc + eps) * 1e-8, d_calc)

    restore_w = W_ob[:, c1_sorted] - W_ob[:, c2_sorted] / safe_d[:, None]

    # One more final column alignment.
    C_final = (
        np.sum(restore_w ** 2, axis=0)[:, None]
        + np.sum(Wp ** 2, axis=0)[None, :]
        - 2.0 * (restore_w.T @ Wp)
    )

    row_ind, col_ind = linear_sum_assignment(C_final)

    final_c1 = np.empty(m, dtype=np.int64)
    final_c2 = np.empty(m, dtype=np.int64)
    final_restore = np.empty_like(restore_w)

    final_c1[col_ind] = c1_sorted[row_ind]
    final_c2[col_ind] = c2_sorted[row_ind]
    final_restore[:, col_ind] = restore_w[:, row_ind]

    # Final D refinement after final alignment.
    X = W_ob[:, final_c1] - Wp
    Y = W_ob[:, final_c2]
    d_calc = np.sum(X * Y, axis=1) / (np.sum(X ** 2, axis=1) + eps)

    safe_d = np.where(np.abs(d_calc) < 1e-8, np.sign(d_calc + eps) * 1e-8, d_calc)
    restore_w = W_ob[:, final_c1] - W_ob[:, final_c2] / safe_d[:, None]

    # ============================================================
    # 6. Build p_calc.
    #
    # Convention:
    #   P[P_true_indices, np.arange(2*m)] = 1
    # Therefore:
    #   P_true_indices[observed_col] = original_col
    # ============================================================
    p_calc = np.empty(2 * m, dtype=np.int64)
    p_calc[final_c1] = np.arange(m)
    p_calc[final_c2] = np.arange(m, 2 * m)

    return restore_w, d_calc, p_calc

def attack_twinshield(model, pre_model, vic_model=None):
    set_seed()
    restore_permutation = {}
    restore_d = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            vic_w = vic_model.state_dict()[name].data if vic_model is not None else None
            Wp = pre_w.detach().cpu().numpy().astype(np.float64, copy=False)
            n, m = Wp.shape
            Wo = ob_w.detach().cpu().numpy().astype(np.float64, copy=False)

            if Wo.shape != (n, 2*m):
                restore_w = pre_w.cpu().numpy()
                print(f"TwinShield packed weight shape is invalid; using public-model weights as the recovery result {name}")
            else:
                restore_w, d_calc, p_calc = solve_twinshield(Wo, Wp)

                restore_permutation[name] = p_calc
                restore_d[name] = d_calc

            restore_w = np.ascontiguousarray(restore_w)
            restore_w = torch.from_numpy(restore_w).to(device=ob_w.device, dtype=ob_w.dtype).contiguous()
            module.data = restore_w
            error = np.linalg.norm(module.data.cpu().numpy() - pre_w.cpu().numpy())
            if vic_w is not None:
                print(f"error between recovered model and public model: {error:.4e}")
                error = np.linalg.norm(pre_w.cpu().numpy() - vic_w.cpu().numpy())
                print(f"error between public model and victim model: {error:.4e}")
                error = np.linalg.norm(module.data.cpu().numpy() - vic_w.cpu().numpy())
                print(f"error between recovered model and victim model: {error:.4e}")
        else:
            module.data = pre_model.state_dict()[name].detach().to(
                device=module.device,
                dtype=module.dtype,
            ).contiguous()
    return model, restore_permutation, restore_d
