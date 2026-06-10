import argparse
import csv
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_DIR = {
    "bert": "bert",
    "gpt2": "gpt2_base",
    "gpt2_base": "gpt2_base",
    "gpt2_xl": "gpt2_xl",
    "vit": "ViT",
    "ViT": "ViT",
    "ViT_augreg": "ViT_augreg",
}

PERM_OBFUS = {
    "translinkguard",
    "tempo",
    "shadownet",
    "groupcover",
    "twinshield",
    "twinshield'",
    "arrowcloak",
    "LRAO+arrowcloak",
    "LRAO+shadownet",
}

PERM_KEY_PAIRS = (
    ("obfus_perm", "restore_perm"),
    ("obfus_permutations", "restore_permutations"),
    ("obfus_permutation", "restore_permutation"),
)


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def format_number(value):
    if isinstance(value, float):
        return f"{value:.2e}"
    return value


def rank_sort_value(rank):
    try:
        return int(rank)
    except (TypeError, ValueError):
        return -1


def to_numpy_leaf(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def count_perm_leaves(value):
    if isinstance(value, dict):
        return sum(count_perm_leaves(v) for v in value.values())
    return 1


def accum_perm_recovery(obfus_perm, restore_perm):
    if isinstance(obfus_perm, dict):
        if not isinstance(restore_perm, dict):
            return 0, count_perm_leaves(obfus_perm)
        matched = total = 0
        for key, value in obfus_perm.items():
            if key not in restore_perm:
                total += count_perm_leaves(value)
                continue
            m, t = accum_perm_recovery(value, restore_perm[key])
            matched += m
            total += t
        return matched, total
    return int(np.array_equal(to_numpy_leaf(obfus_perm), to_numpy_leaf(restore_perm))), 1


def obfus_has_permutation(obfus):
    if obfus in PERM_OBFUS:
        return True
    return any(part in PERM_OBFUS for part in str(obfus).split("+"))


def attack_extras_path(row, restore_dir):
    restore_path = row.get("restore_path")
    if restore_path:
        path = Path(restore_path)
        if not path.exists() and not path.is_absolute():
            mpa_path = Path("MPA") / path
            if mpa_path.exists():
                path = mpa_path
        if path.name == "pre_finetune_checkpoint":
            return path / "attack_extras.pkl"

    model_dir = MODEL_DIR.get(row["model"], row["model"])
    base = Path(restore_dir) / model_dir / row["obfus"] / row["dataset"]
    if not base.exists() and not base.is_absolute():
        mpa_base = Path("MPA") / base
        if mpa_base.exists():
            base = mpa_base
    rank_r = row.get("rank_r", "")
    if "LRAO" in row["obfus"] and rank_r not in ("", None):
        base = base / f"r{rank_r}"
    return base / "pre_finetune_checkpoint" / "attack_extras.pkl"


def load_perm_maps(row, restore_dir):
    path = attack_extras_path(row, restore_dir)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            extras = pickle.load(f)
    except Exception:
        return None

    for obfus_key, restore_key in PERM_KEY_PAIRS:
        if obfus_key not in extras or restore_key not in extras:
            continue
        return extras[obfus_key], extras[restore_key]
    return None


def layer_key_candidates(model, layer):
    base_layer, sep, tag = str(layer).partition(":")
    candidates = [base_layer, layer]

    if model == "bert":
        match = re.search(r"bert\.encoder\.layer\.(\d+)\.", base_layer)
        if match:
            candidates.append(match.group(1))
    elif model in {"gpt2", "gpt2_base", "gpt2_xl"}:
        match = re.search(r"transformer\.h\.(\d+)\.", base_layer)
        if match:
            candidates.append(match.group(1))
    elif model in {"vit", "ViT", "ViT_augreg"}:
        block_match = re.search(r"(blocks\.\d+)", base_layer)
        if block_match:
            block = block_match.group(1)
            candidates.extend([block, f"{block}.attn", f"{block}.mlp"])

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered, tag if sep else None


def pick_layer_perm(perm_map, model, layer):
    candidates, tag = layer_key_candidates(model, layer)
    for candidate in candidates:
        if not isinstance(perm_map, dict) or candidate not in perm_map:
            continue
        value = perm_map[candidate]
        if tag and isinstance(value, dict) and tag in value:
            return value[tag]
        return value
    return None


def permutation_restored_for_row(row, perm_maps):
    if perm_maps is None:
        return False
    obfus_map, restore_map = perm_maps
    obfus_perm = pick_layer_perm(obfus_map, row["model"], row.get("layer", ""))
    restore_perm = pick_layer_perm(restore_map, row["model"], row.get("layer", ""))
    if obfus_perm is None or restore_perm is None:
        return False
    matched, total = accum_perm_recovery(obfus_perm, restore_perm)
    return total > 0 and matched == total


def summarize(input_path, merge_datasets=False, group_by_rank="auto", restore_dir="results/our_results"):
    groups = defaultdict(lambda: {"adp_sum": 0.0, "adp_count": 0, "ratio_sum": 0.0, "ratio_count": 0, "adp_over_ratio_sum": 0.0, "adp_over_ratio_count": 0})
    perm_cache = {}
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        has_rank = reader.fieldnames is not None and "rank_r" in reader.fieldnames
        if group_by_rank == "auto":
            include_rank = has_rank
        else:
            include_rank = group_by_rank == "always" and has_rank
        for row in reader:
            if obfus_has_permutation(row["obfus"]):
                perm_key = (
                    row["model"],
                    row["dataset"],
                    row["obfus"],
                    row.get("rank_r", ""),
                    row.get("restore_stage", "pre_finetune_checkpoint"),
                    row.get("restore_path", ""),
                )
                if perm_key not in perm_cache:
                    perm_cache[perm_key] = load_perm_maps(row, restore_dir)
                if not permutation_restored_for_row(row, perm_cache[perm_key]):
                    continue
            if merge_datasets:
                key = (row["model"], row["obfus"])
            else:
                key = (row["model"], row["dataset"], row["obfus"])
            if include_rank:
                key = (*key, row.get("rank_r", ""))
            adp = parse_float(row.get("adp"))
            ratio = parse_float(row.get("k_over_nm"))
            if adp is not None:
                groups[key]["adp_sum"] += adp
                groups[key]["adp_count"] += 1
            if ratio is not None:
                groups[key]["ratio_sum"] += ratio
                groups[key]["ratio_count"] += 1
            if adp is not None and ratio is not None and ratio != 0:
                groups[key]["adp_over_ratio_sum"] += adp / ratio
                groups[key]["adp_over_ratio_count"] += 1
    rows = []
    def sort_key(item):
        key, _ = item
        rank = key[-1] if include_rank else None
        prefix = key[:-1] if include_rank else key
        return (*prefix, rank_sort_value(rank))

    for key, stat in sorted(groups.items(), key=sort_key):
        if merge_datasets:
            if include_rank:
                model, obfus, rank_r = key
            else:
                model, obfus = key
                rank_r = None
            dataset = "all"
        else:
            if include_rank:
                model, dataset, obfus, rank_r = key
            else:
                model, dataset, obfus = key
                rank_r = None
        adp_mean = stat["adp_sum"] / stat["adp_count"] if stat["adp_count"] else ""
        ratio_mean = stat["ratio_sum"] / stat["ratio_count"] if stat["ratio_count"] else ""
        adp_over_ratio_mean = stat["adp_over_ratio_sum"] / stat["adp_over_ratio_count"] if stat["adp_over_ratio_count"] else ""
        out = {
            "model": model,
            "dataset": dataset,
            "obfus": obfus,
            "num_adp_layers": stat["adp_count"],
            # "num_ratio_layers": stat["ratio_count"],
            "adp_mean": format_number(adp_mean),
            # "k_over_nm_mean": format_number(ratio_mean),
            # "adp_over_k_over_nm_mean": format_number(adp_over_ratio_mean),
            # "adp_mean_over_k_over_nm_mean": format_number(adp_mean / ratio_mean) if ratio_mean not in (0, "", None) else "",
        }
        if include_rank:
            out["rank_r"] = rank_r
        rows.append(out)
    return rows


def write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "obfus",
    ]
    if any("rank_r" in row for row in rows):
        fieldnames.append("rank_r")
    fieldnames.extend(
        [
            "num_adp_layers",
            "num_ratio_layers",
            "adp_mean",
            "k_over_nm_mean",
            "adp_over_k_over_nm_mean",
            "adp_mean_over_k_over_nm_mean",
        ]
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    if not rows:
        print("No ADP rows found.")
        return
    headers = ["model", "dataset", "obfus"]
    if any("rank_r" in row for row in rows):
        headers.append("rank_r")
    headers.extend(["num_adp_layers", "adp_mean", "k_over_nm_mean", "adp_over_k_over_nm_mean", "adp_mean_over_k_over_nm_mean"])
    widths = {h: len(h) for h in headers}
    printable = []
    for row in rows:
        item = {}
        for h in headers:
            value = str(row.get(h, ""))
            item[h] = value
            widths[h] = max(widths[h], len(value))
        printable.append(item)
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in printable:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ADP CSV by model and obfuscation method.")
    parser.add_argument("--input", default="results/adp_results/adp.csv")
    parser.add_argument("--output", default="results/adp_results/adp_summary_by_model_obfus.csv")
    parser.add_argument("--restore_dir", default="results/our_results", help="Directory that contains pre_finetune_checkpoint/attack_extras.pkl files.")
    parser.add_argument("--merge_datasets", action="store_true", help="Group only by model and obfuscation method.")
    parser.add_argument(
        "--group_by_rank",
        choices=["auto", "always", "never"],
        default="auto",
        help="Group by rank_r when the input CSV contains it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = summarize(input_path, merge_datasets=args.merge_datasets, group_by_rank=args.group_by_rank, restore_dir=args.restore_dir)
    write_rows(output_path, rows)
    print_table(rows)
    print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()
