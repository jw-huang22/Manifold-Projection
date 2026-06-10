import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds


ALL_METHODS = [
    "translinkguard",
    "tsqp",
    "soter",
    "tempo",
    "shadownet",
    "LoRO",
    "LRAO",
    "obfuscatune",
    "groupcover",
    "twinshield",
    "arrowcloak",
]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_random_pair(size, delta, dtype, device):
    w_pub = torch.randn(size, size, dtype=dtype, device=device)
    d_w = torch.randn(size, size, dtype=dtype, device=device) * delta
    w_vic = torch.randn(size, size, dtype=dtype, device=device)
    return w_vic - d_w, d_w, w_vic


def as_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_torch(x, like):
    return torch.from_numpy(np.asarray(x)).to(device=like.device, dtype=like.dtype)


def row_perm_match(w_pub, w_obf):
    a = as_np(w_obf).astype(np.float64, copy=False)
    b = as_np(w_pub).astype(np.float64, copy=False)
    score = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-score)
    src_for_pub = row_ind[np.argsort(col_ind)]
    return src_for_pub


def col_perm_match(w_pub, w_obf):
    a = as_np(w_obf).astype(np.float64, copy=False)
    b = as_np(w_pub).astype(np.float64, copy=False)
    score = a.T @ b
    row_ind, col_ind = linear_sum_assignment(-score)
    src_for_pub = row_ind[np.argsort(col_ind)]
    return src_for_pub


def row_perm_scale_match(w_pub, w_obf):
    a = as_np(w_obf).astype(np.float64, copy=False)
    b = as_np(w_pub).astype(np.float64, copy=False)
    ab = b @ a.T
    aa = np.sum(a * a, axis=1) + 1e-20
    score = (ab * ab) / aa[None, :]
    row_ind, col_ind = linear_sum_assignment(-score)
    src_for_pub = col_ind[np.argsort(row_ind)]
    scales = (ab[row_ind, col_ind] / aa[col_ind])[np.argsort(row_ind)]
    return src_for_pub, scales


def col_perm_scale_match(w_pub, w_obf):
    a = as_np(w_obf).astype(np.float64, copy=False)
    b = as_np(w_pub).astype(np.float64, copy=False)
    ab = b.T @ a
    aa = np.sum(a * a, axis=0) + 1e-20
    score = (ab * ab) / aa[None, :]
    row_ind, col_ind = linear_sum_assignment(-score)
    src_for_pub = col_ind[np.argsort(row_ind)]
    scales = (ab[row_ind, col_ind] / aa[col_ind])[np.argsort(row_ind)]
    return src_for_pub, scales


def best_global_scale(w_pub, w_obf):
    num = torch.sum(w_pub.double() * w_obf.double())
    den = torch.sum(w_obf.double() * w_obf.double()) + 1e-20
    return (num / den).to(dtype=w_obf.dtype)


def restore_low_rank(w_pub, w_obf, rank):
    k = (w_pub - w_obf).double()
    u, s, vh = torch.linalg.svd(k, full_matrices=False)
    k_rank = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    return (w_obf.double() + k_rank).to(dtype=w_obf.dtype)


def restore_orthogonal(w_pub, w_obf):
    u, _, vh = torch.linalg.svd(w_pub.double().T @ w_obf.double(), full_matrices=False)
    q = vh.T @ u.T
    return (w_obf.double() @ q).to(dtype=w_obf.dtype)


def orthogonal_dof(d):
    return d * (d - 1) // 2


def continuous_param_count(method, n, m, args):
    if method == "translinkguard":
        return 0
    if method in {"tsqp", "soter"}:
        return 1
    if method == "LoRO":
        return 8 * (n + m)
    if method == "LRAO":
        return args.rank_r * (n + m)
    if method == "obfuscatune":
        return orthogonal_dof(m)
    if method == "groupcover":
        return n * args.group_size
    if method == "twinshield":
        return n
    if method == "tempo":
        return m
    if method == "shadownet":
        return n
    if method == "arrowcloak":
        return n + 2 * m
    raise ValueError(f"Unsupported obfuscation method: {method}")


def sci3(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(x) or np.isinf(x):
        return str(value)
    return f"{x:.3e}"


def to_numpy_int(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(x, dtype=np.int64)


def inverse_perm(perm):
    return np.argsort(to_numpy_int(perm))


def permutation_recovery_rate(obfus_meta, recover_meta):
    if "perm" not in obfus_meta or "src_for_pub" not in recover_meta:
        return np.nan
    true_src_for_pub = inverse_perm(obfus_meta["perm"])
    recovered_src_for_pub = to_numpy_int(recover_meta["src_for_pub"])
    if true_src_for_pub.shape != recovered_src_for_pub.shape:
        return np.nan
    return float(np.mean(true_src_for_pub == recovered_src_for_pub))


def ob_translinkguard(w):
    perm = torch.randperm(w.shape[0], device=w.device)
    return w[perm], {"perm": perm}


def attack_translinkguard_our(w_obf, w_pub, meta, args):
    src_for_pub = row_perm_match(w_pub, w_obf)
    rec = w_obf[torch.as_tensor(src_for_pub, device=w_obf.device)]
    return rec, {"src_for_pub": src_for_pub}


def ob_tsqp(w):
    scale = 1.0 + 5.0 * torch.rand((), device=w.device, dtype=w.dtype)
    return w * scale, {"scale": scale}


def attack_tsqp(w_obf, w_pub, meta, args):
    return w_obf * best_global_scale(w_pub, w_obf), {}


def ob_soter(w):
    return ob_tsqp(w)


def attack_soter_our(w_obf, w_pub, meta, args):
    return attack_tsqp(w_obf, w_pub, meta, args)


def ob_shadownet(w):
    scales = 1.0 + 5.0 * torch.rand(w.shape[0], device=w.device, dtype=w.dtype)
    perm = torch.randperm(w.shape[0], device=w.device)
    return (w * scales[:, None])[perm], {"scales": scales, "perm": perm}


def attack_shadownet_our(w_obf, w_pub, meta, args):
    src_for_pub, scales = row_perm_scale_match(w_pub, w_obf)
    rec = w_obf[torch.as_tensor(src_for_pub, device=w_obf.device)]
    return rec * as_torch(scales[:, None], w_obf), {"src_for_pub": src_for_pub}


def ob_tempo(w):
    scales = 1.0 + 5.0 * torch.rand(w.shape[1], device=w.device, dtype=w.dtype)
    perm = torch.randperm(w.shape[1], device=w.device)
    return (w * scales[None, :])[:, perm], {"scales": scales, "perm": perm}


def attack_tempo_our(w_obf, w_pub, meta, args):
    src_for_pub, scales = col_perm_scale_match(w_pub, w_obf)
    rec = w_obf[:, torch.as_tensor(src_for_pub, device=w_obf.device)]
    return rec * as_torch(scales[None, :], w_obf), {"src_for_pub": src_for_pub}


def ob_loro(w, rank):
    a = torch.randn(w.shape[0], rank, device=w.device, dtype=w.dtype) * 1e-1
    b = torch.randn(rank, w.shape[1], device=w.device, dtype=w.dtype) * 1e-1
    return w + a @ b, {"rank": rank}


def attack_loro(w_obf, w_pub, meta, args):
    return restore_low_rank(w_pub, w_obf, 8), {}


def ob_lrao(w, w_pub, rank):
    d = w_pub.double() - w.double()
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    d_rank = ((u[:, :rank] * s[:rank]) @ vh[:rank, :]).to(dtype=w.dtype)
    return w + d_rank, {"rank": rank}


def attack_lrao(w_obf, w_pub, meta, args):
    return w_obf, {}


def ob_obfuscatune(w):
    q, _ = torch.linalg.qr(torch.randn(w.shape[1], w.shape[1], device=w.device, dtype=torch.float64))
    q = q.to(dtype=w.dtype)
    return w @ q, {"q": q}


def attack_obfuscatune(w_obf, w_pub, meta, args):
    return restore_orthogonal(w_pub, w_obf), {}


def ob_groupcover(w, group_size):
    n, m = w.shape
    if n % group_size != 0:
        raise ValueError("groupcover requires size divisible by group_size")
    out = torch.empty_like(w)
    clusters = []
    for start in range(0, n, group_size):
        idx = torch.arange(start, start + group_size, device=w.device)
        coeff = torch.randint(1, 100, (group_size, group_size), device=w.device, dtype=w.dtype)
        out[idx] = coeff @ w[idx]
        clusters.append(idx.cpu().numpy())
    perm = torch.randperm(m, device=w.device)
    return out[:, perm], {"clusters": clusters, "perm": perm}


def attack_groupcover(w_obf, w_pub, meta, args):
    src_for_pub = col_perm_match(w_pub, w_obf)
    aligned = w_obf[:, torch.as_tensor(src_for_pub, device=w_obf.device)]
    rec = torch.empty_like(aligned)
    for idx in meta["clusters"]:
        rows = torch.as_tensor(idx, device=w_obf.device)
        a = aligned[rows].double().T
        b = w_pub[rows].double().T
        k = torch.linalg.lstsq(a, b).solution
        rec[rows] = (a @ k).T.to(dtype=w_obf.dtype)
    return rec, {"src_for_pub": src_for_pub}


def ob_twinshield(w):
    n, m = w.shape
    r = torch.randn(n, m, device=w.device, dtype=w.dtype)
    d = torch.randn(n, device=w.device, dtype=w.dtype)
    perm = torch.randperm(m, device=w.device)
    wo1 = (w + r)[:, perm]
    wo2 = (d[:, None] * r)[:, perm]
    return torch.cat([wo1, wo2], dim=0), {"perm": perm, "d": d}


def solve_twinshield(w_obf, w_pub, max_iter=50, tol=1e-6):
    n, m = w_pub.shape
    wo1 = as_np(w_obf[:n]).astype(np.float64, copy=False)
    wo2 = as_np(w_obf[n:]).astype(np.float64, copy=False)
    wp = as_np(w_pub).astype(np.float64, copy=False)
    d = np.zeros(n, dtype=np.float64)
    p_idx = np.arange(m)
    prev = np.inf
    for _ in range(max_iter):
        cur = wo1 + d[:, None] * wo2
        cost = (
            np.sum(cur * cur, axis=0)[:, None]
            + np.sum(wp * wp, axis=0)[None, :]
            - 2.0 * cur.T @ wp
        )
        _, p_idx = linear_sum_assignment(cost)
        target = wp[:, p_idx] - wo1
        den = np.sum(wo2 * wo2, axis=1)
        d = np.divide(np.sum(wo2 * target, axis=1), den, out=np.zeros_like(den), where=den > 1e-20)
        err = np.linalg.norm(wo1 + d[:, None] * wo2 - wp[:, p_idx], "fro")
        if prev - err < tol:
            break
        prev = err
    p = np.zeros((m, m), dtype=np.float64)
    p[p_idx, np.arange(m)] = 1.0
    rec = (wo1 + d[:, None] * wo2) @ p.T
    return rec, np.argsort(p_idx)


def attack_twinshield(w_obf, w_pub, meta, args):
    rec, src_for_pub = solve_twinshield(w_obf, w_pub)
    return as_torch(rec, w_pub), {"src_for_pub": src_for_pub}


def ob_arrowcloak(w):
    coeff = torch.randint(0, 5, (w.shape[1],), device=w.device, dtype=w.dtype)
    mask = w @ coeff
    scales = torch.randint(1, 3, (w.shape[1],), device=w.device, dtype=w.dtype)
    ratios = (torch.randint(0, 11, (w.shape[1],), device=w.device, dtype=w.dtype) - 5.0)
    mixed = w * scales[None, :] + mask[:, None] * ratios[None, :]
    perm = torch.randperm(w.shape[1], device=w.device)
    return mixed[:, perm], {"perm": perm, "mask": mask, "scales": scales, "ratios": ratios}


def rank1_permuted_diagonal_decomposition(t, max_iter=50, tol=1e-6):
    n = t.shape[0]
    s_mat = np.zeros_like(t)
    prev = np.inf
    p = np.eye(n)
    l_mat = np.zeros_like(t)
    for _ in range(max_iter):
        r_mat = t - s_mat
        u, s, vt = svds(r_mat, k=1)
        l_mat = s[0] * np.outer(u[:, 0], vt[0, :])
        e = t - l_mat
        row_ind, col_ind = linear_sum_assignment(-(e * e))
        new_s = np.zeros_like(t)
        new_s[row_ind, col_ind] = e[row_ind, col_ind]
        err = np.linalg.norm(t - l_mat - new_s, "fro")
        s_mat = new_s
        p = np.zeros((n, n))
        p[row_ind, col_ind] = 1.0
        if abs(prev - err) < tol:
            break
        prev = err
    return l_mat, s_mat, p


def solve_arrow_permutation(w_pub_t, w_obf_t):
    best_obj = np.inf
    best_p = None
    for rho in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        n, m = w_pub_t.shape
        if n <= m:
            inv = np.linalg.inv(w_pub_t @ w_pub_t.T + rho * np.eye(n))
            t = (w_obf_t @ w_pub_t.T) @ inv
        else:
            inv = np.linalg.inv(w_pub_t.T @ w_pub_t + rho * np.eye(m))
            t = (w_obf_t @ inv) @ w_pub_t.T
        l_mat, s_mat, p = rank1_permuted_diagonal_decomposition(t)
        obj = np.linalg.norm((l_mat + s_mat) @ w_pub_t - w_obf_t, "fro")
        if obj < best_obj:
            best_obj = obj
            best_p = p
    return best_p


def solve_arrow_shared_mask(w_mix, w_pub, max_iter=100):
    col_norms = np.sum(w_pub * w_pub, axis=0) + 1e-20
    scales = np.sum(w_mix * w_pub, axis=0) / col_norms
    residual = w_mix - w_pub * scales[None, :]
    try:
        u, s, _ = svds(residual, k=1)
        mask = u[:, 0] * s[0]
    except Exception:
        u, s, _ = np.linalg.svd(residual, full_matrices=False)
        mask = u[:, 0] * s[0]
    ratios = np.zeros(w_pub.shape[1], dtype=np.float64)
    pub_dot_mix = np.sum(w_pub * w_mix, axis=0)
    for _ in range(max_iter):
        mask_norm = np.dot(mask, mask) + 1e-20
        pub_dot_mask = w_pub.T @ mask
        mix_dot_mask = w_mix.T @ mask
        det = col_norms * mask_norm - pub_dot_mask * pub_dot_mask
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        scales = (pub_dot_mix * mask_norm - mix_dot_mask * pub_dot_mask) / det
        ratios = (mix_dot_mask * col_norms - pub_dot_mix * pub_dot_mask) / det
        mask = (w_mix - w_pub * scales[None, :]) @ ratios / (np.dot(ratios, ratios) + 1e-20)
    scales = np.where(np.abs(scales) < 1e-8, 1e-8, scales)
    return (w_mix - mask[:, None] * ratios[None, :]) / scales[None, :]


def attack_arrowcloak_our(w_obf, w_pub, meta, args):
    wo = as_np(w_obf).astype(np.float64, copy=False)
    wp = as_np(w_pub).astype(np.float64, copy=False)
    p = solve_arrow_permutation(wp.T, wo.T)
    src_for_pub = np.argmax(p, axis=0)
    w_mix = wo @ p
    return as_torch(solve_arrow_shared_mask(w_mix, wp), w_pub), {"src_for_pub": src_for_pub}


def obfuscate(method, w_vic, w_pub, args):
    if method == "translinkguard":
        return ob_translinkguard(w_vic)
    if method == "tsqp":
        return ob_tsqp(w_vic)
    if method == "soter":
        return ob_soter(w_vic)
    if method == "tempo":
        return ob_tempo(w_vic)
    if method == "shadownet":
        return ob_shadownet(w_vic)
    if method == "LoRO":
        return ob_loro(w_vic, 8)
    if method == "LRAO":
        return ob_lrao(w_vic, w_pub, args.rank_r)
    if method == "obfuscatune":
        return ob_obfuscatune(w_vic)
    if method == "groupcover":
        return ob_groupcover(w_vic, args.group_size)
    if method == "twinshield":
        return ob_twinshield(w_vic)
    if method == "arrowcloak":
        return ob_arrowcloak(w_vic)
    raise ValueError(f"Unsupported obfuscation method: {method}")


def recover(method, w_obf, w_pub, meta, args):
    if method == "translinkguard":
        return attack_translinkguard_our(w_obf, w_pub, meta, args)
    if method == "tsqp":
        return attack_tsqp(w_obf, w_pub, meta, args)
    if method == "soter":
        return attack_soter_our(w_obf, w_pub, meta, args)
    if method == "tempo":
        return attack_tempo_our(w_obf, w_pub, meta, args)
    if method == "shadownet":
        return attack_shadownet_our(w_obf, w_pub, meta, args)
    if method == "LoRO":
        return attack_loro(w_obf, w_pub, meta, args)
    if method == "LRAO":
        return attack_lrao(w_obf, w_pub, meta, args)
    if method == "obfuscatune":
        return attack_obfuscatune(w_obf, w_pub, meta, args)
    if method == "groupcover":
        return attack_groupcover(w_obf, w_pub, meta, args)
    if method == "twinshield":
        return attack_twinshield(w_obf, w_pub, meta, args)
    if method == "arrowcloak":
        return attack_arrowcloak_our(w_obf, w_pub, meta, args)
    raise ValueError(f"Unsupported obfuscation method: {method}")


def compute_metrics(w_rec, w_pub, w_vic):
    rec_minus_vic = (w_rec - w_vic).double().reshape(-1)
    pub_minus_vic = (w_pub - w_vic).double().reshape(-1)
    denom = torch.dot(pub_minus_vic, pub_minus_vic).item()
    rel_err = (torch.linalg.norm(rec_minus_vic).item()) / (np.sqrt(denom) + 1e-20)
    adp = torch.dot(rec_minus_vic, pub_minus_vic).item() / (denom + 1e-20)
    return rel_err, adp


def summarize(rows):
    print(
        "\nmethod, ok_trials, rel_err_mean, rel_err_std, adp_mean, adp_std, "
        "k_over_nm_mean, perm_recovery_rate_mean"
    )
    for method in ALL_METHODS:
        vals = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        if not vals:
            print(f"{method}, 0, nan, nan, nan, nan, nan, nan")
            continue
        rel = np.array([r["rel_err"] for r in vals], dtype=np.float64)
        adp = np.array([r["adp"] for r in vals], dtype=np.float64)
        k_over_nm = np.array([r["k_over_nm"] for r in vals], dtype=np.float64)
        perm_rates = np.array([r["perm_recovery_rate"] for r in vals], dtype=np.float64)
        perm_rate_mean = np.nanmean(perm_rates) if not np.all(np.isnan(perm_rates)) else np.nan
        print(
            f"{method}, {len(vals)}, {sci3(rel.mean())}, {sci3(rel.std(ddof=0))}, "
            f"{sci3(adp.mean())}, {sci3(adp.std(ddof=0))}, "
            f"{sci3(k_over_nm.mean())}, {sci3(perm_rate_mean)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone random-weight probe for obfuscation attacks."
    )
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank_r", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--obfus", nargs="+", default=["all"])
    parser.add_argument("--output", default="results/random_weight_attack.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = ALL_METHODS if "all" in args.obfus else args.obfus
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    rows = []

    print(
        f"Running random-weight probe: size={args.size}x{args.size}, "
        f"n={args.n}, delta={args.delta}, rank_r={args.rank_r}"
    )

    for trial in range(args.n):
        trial_seed = args.seed + trial
        set_all_seeds(trial_seed)
        w_pub, d_w, w_vic = make_random_pair(args.size, args.delta, dtype, device)
        actual_delta = (torch.linalg.norm(d_w) / torch.linalg.norm(w_pub)).item()

        for method in methods:
            set_all_seeds(trial_seed)
            n, m = w_pub.shape
            k = continuous_param_count(method, n, m, args)
            nm = n * m
            k_over_nm = k / nm
            try:
                w_obf, meta = obfuscate(method, w_vic.clone(), w_pub, args)
                w_rec, recover_meta = recover(method, w_obf, w_pub, meta, args)
                rel_err, adp = compute_metrics(w_rec, w_pub, w_vic)
                perm_rate = permutation_recovery_rate(meta, recover_meta)
                row = {
                    "method": method,
                    "trial": trial,
                    "seed": trial_seed,
                    "size": args.size,
                    "delta": sci3(args.delta),
                    "actual_delta": sci3(actual_delta),
                    "k": k,
                    "nm": nm,
                    "k_over_nm": sci3(k_over_nm),
                    "perm_recovery_rate": sci3(perm_rate),
                    "rel_err": sci3(rel_err),
                    "adp": sci3(adp),
                    "status": "ok",
                    "error": "",
                }
                print(
                    f"[ok] trial={trial:02d} method={method:14s} "
                    f"rel_err={sci3(rel_err)} adp={sci3(adp)} "
                    f"k/nm={sci3(k_over_nm)} perm={sci3(perm_rate)}"
                )
            except Exception as exc:
                row = {
                    "method": method,
                    "trial": trial,
                    "seed": trial_seed,
                    "size": args.size,
                    "delta": sci3(args.delta),
                    "actual_delta": sci3(actual_delta),
                    "k": k,
                    "nm": nm,
                    "k_over_nm": sci3(k_over_nm),
                    "perm_recovery_rate": "nan",
                    "rel_err": "nan",
                    "adp": "nan",
                    "status": "failed",
                    "error": repr(exc),
                }
                print(f"[failed] trial={trial:02d} method={method:14s} error={exc!r}")
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nSaved per-trial results to {output}")


if __name__ == "__main__":
    main()
