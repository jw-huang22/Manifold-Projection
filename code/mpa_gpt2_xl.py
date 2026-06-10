import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import torch
import argparse
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import evaluate
from pdb import set_trace as st
from utils.utils_gpt2_xl import *
from peft import PeftModel, get_peft_model, LoraConfig
from safetensors.torch import load_file
from utils.methods_gpt2_xl import *
import json


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="sst2", type=str, help="dataset")
parser.add_argument("--model", default="gpt2-xl", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=4, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids")
parser.add_argument("--recover_lr", default=1e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--rank_r", default=32, type=int, help="Rank used by LRAO low-rank obfuscation/recovery")
parser.add_argument("--full_finetune", default="true", type=str, help="Whether recovery fine-tuning updates all parameters")
parser.add_argument(
    "--public_non_target_seed",
    default=43,
    type=int,
    help="seed used to rebuild the public model before copying non-target modules",
)

parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--restore_dir", default="results/our_results", type=str, help="restore directory")
parser.add_argument("--obfus_dir", default="tmp/obfus_results", type=str, help="obfus directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_labels = 2
if args.model == "gpt2-xl":
    model_name = "gpt2_xl"
else:
    raise ValueError("Invalid model name")

args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}/final_checkpoint"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}/final_checkpoint"
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
if "LRAO" in args.obfus:
    args.restore_dir = f"{args.restore_dir}/r{args.rank_r}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

os.makedirs(args.restore_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.obfus_dir, exist_ok=True)
set_seed()

task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

actual_task = args.dataset
validation_key = "validation_matched" if args.dataset == "mnli" else "validation"

print("=" * 60)
print("Run configuration (for experiments / reproducibility)")
print("=" * 60)
print(f"  argv: {sys.argv!r}")
print(f"  model_name: {model_name}")
print(f"  actual_task: {actual_task}  num_labels: {num_labels}  validation_key: {validation_key}")
for k in sorted(vars(args)):
    print(f"  {k}: {getattr(args, k)}")
print("=" * 60)

# Prepare data
print("Preparing data..")
sentence1_key, sentence2_key = task_to_keys[args.dataset]
trainset, evalset, tokenizer = prepare_data(actual_task, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def str2bool(value):
    return str(value).lower() in ("true", "1", "yes", "y")

def make_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_fc", "c_attn", "c_proj"],
    )

def _checkpoint_index(checkpoint_dir):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, "r") as f:
        return json.load(f)

def is_lora_checkpoint(checkpoint_dir):
    index = _checkpoint_index(checkpoint_dir)
    if index is None:
        return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    return any("lora_" in key or ".base_layer." in key for key in index.get("weight_map", {}))

def load_safetensor_state(checkpoint_dir):
    index = _checkpoint_index(checkpoint_dir)
    if index is not None:
        filenames = sorted(set(index.get("weight_map", {}).values()))
    elif os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        filenames = ["model.safetensors"]
    else:
        filenames = sorted(
            name for name in os.listdir(checkpoint_dir)
            if name.endswith(".safetensors") and name.startswith("model")
        )
    if not filenames:
        raise FileNotFoundError(f"No safetensors model files found in {checkpoint_dir}")
    state = {}
    for filename in filenames:
        state.update(load_file(os.path.join(checkpoint_dir, filename)))
    return state

def load_finetuned_gpt2_xl(checkpoint_dir):
    if is_lora_checkpoint(checkpoint_dir):
        print(f"Loading LoRA GPT-2 XL checkpoint from {checkpoint_dir}")
        state = load_safetensor_state(checkpoint_dir)
        model = adjust_lora_model(
            "gpt2-xl",
            lora_config=lora_config,
            num_labels=num_labels,
            weight1=state,
            weight2={},
        )
    else:
        print(f"Loading full-parameter GPT-2 XL checkpoint from {checkpoint_dir}")
        model = GPT2ForSequenceClassification.from_pretrained(
            checkpoint_dir,
            num_labels=num_labels,
            use_safetensors=True,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

def load_target_model(checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = args.weight_dir_tsqp if args.obfus == "tsqp" else args.weight_dir
    return load_finetuned_gpt2_xl(checkpoint_dir)

print("Loading metric..")
metric = evaluate.load('glue', actual_task)

# load model
set_seed()
lora_config = make_lora_config()
victim_model = load_finetuned_gpt2_xl(args.weight_dir)


path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    print("Preparing recover_data..")
    prepare_recover_data(victim_model, trainset, args.bs, path, ratio = 0.01)

with open(path, "r") as f:
    data = json.load(f)
recover_dataset = datasets.Dataset.from_dict({
    "sentence": data['sentence'],
    "idx": data['idx'],
    "input_ids": data['input_ids'],
    "attention_mask": data['attention_mask'],
    "label": data['label']
})
print("recover_data prepared!")


def mpa_gpt2_xl_to_numpy_leaf(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mpa_gpt2_xl_accum_perm_recovery(p, rp):
    if isinstance(p, dict):
        matched = total = 0
        for sk, pv in p.items():
            if sk not in rp:
                continue
            m, t = mpa_gpt2_xl_accum_perm_recovery(pv, rp[sk])
            matched += m
            total += t
        return matched, total
    return (
        int(np.array_equal(mpa_gpt2_xl_to_numpy_leaf(p), mpa_gpt2_xl_to_numpy_leaf(rp))),
        1,
    )


def mpa_gpt2_xl_rel_err_scaling(sf, rsf):
    errs = []
    if isinstance(sf, dict):
        for sk, sv in sf.items():
            if sk not in rsf:
                continue
            errs.extend(mpa_gpt2_xl_rel_err_scaling(sv, rsf[sk]))
        return errs
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    if isinstance(rsf, torch.Tensor):
        rsf = rsf.detach().cpu().numpy()
    if isinstance(sf, (float, int)) or isinstance(rsf, (float, int)) or np.isscalar(sf):
        errs.append(abs(float(sf) - float(rsf)) / (abs(float(sf)) + 1e-12))
        return errs
    a = np.asarray(sf, dtype=np.float64).ravel()
    b = np.asarray(rsf, dtype=np.float64).ravel()
    if a.size and b.size and a.size == b.size:
        errs.append(np.mean(np.abs(a - b) / (np.abs(a) + 1e-12)))
    return errs


def mpa_gpt2_xl_partitions_equal(clusters_a, clusters_b):
    if isinstance(clusters_a, dict):
        if not isinstance(clusters_b, dict):
            return False
        if set(clusters_a.keys()) != set(clusters_b.keys()):
            return False
        return all(
            mpa_gpt2_xl_partitions_equal(clusters_a[sk], clusters_b[sk])
            for sk in clusters_a
        )
    if len(clusters_a) != len(clusters_b):
        return False

    def canon(cs):
        return sorted(tuple(sorted(s)) for s in cs)

    return canon(clusters_a) == canon(clusters_b)


def mpa_gpt2_xl_accum_cluster_recovery(cl_a, cl_b):
    if isinstance(cl_a, dict):
        if not isinstance(cl_b, dict):
            return 0, 0
        matched = total = 0
        for sk, v in cl_a.items():
            if sk not in cl_b:
                continue
            m, t = mpa_gpt2_xl_accum_cluster_recovery(v, cl_b[sk])
            matched += m
            total += t
        return matched, total

    if not isinstance(cl_a, (list, tuple)) or not isinstance(cl_b, (list, tuple)):
        return int(mpa_gpt2_xl_partitions_equal(cl_a, cl_b)), 1

    def canon(cs):
        return [tuple(sorted(s)) for s in cs]

    canon_a = canon(cl_a)
    canon_b = canon(cl_b)
    matched = 0
    b_list = list(canon_b)
    for c in canon_a:
        if c in b_list:
            matched += 1
            b_list.remove(c)
    return matched, len(canon_a)


def mpa_gpt2_xl_rel_err_fro(a, b):
    errs = []
    if isinstance(a, dict):
        for sk, av in a.items():
            if sk in b:
                errs.extend(mpa_gpt2_xl_rel_err_fro(av, b[sk]))
        return errs
    if isinstance(a, torch.Tensor):
        a = a.detach()
    else:
        a = torch.as_tensor(np.asarray(a))
    if isinstance(b, torch.Tensor):
        b = b.detach()
    else:
        b = torch.as_tensor(np.asarray(b))
    errs.append((torch.norm(b - a, p="fro") / (torch.norm(a, p="fro") + 1e-12)).item())
    return errs


def mpa_gpt2_xl_rel_err_vector_pairs(d_obf, d_rec):
    errs = []
    if isinstance(d_obf, dict):
        for sk, v in d_obf.items():
            if sk in d_rec:
                errs.extend(mpa_gpt2_xl_rel_err_vector_pairs(v, d_rec[sk]))
        return errs
    da = np.asarray(d_obf, dtype=np.float64).ravel()
    rb = np.asarray(d_rec, dtype=np.float64).ravel()
    if da.size and da.size == rb.size:
        errs.append(np.mean(np.abs(rb - da) / (np.abs(da) + 1e-12)))
    return errs


def mpa_gpt2_xl_iter_leaf_pairs(a, b, prefix=""):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return
        for sk, av in a.items():
            if sk not in b:
                continue
            next_prefix = f"{prefix}.{sk}" if prefix else str(sk)
            yield from mpa_gpt2_xl_iter_leaf_pairs(av, b[sk], next_prefix)
        return
    yield prefix, a, b


def mpa_gpt2_xl_leaf_group(path):
    last = str(path).split(".")[-1]
    return last if last in {"q", "k", "v"} else "other"


def mpa_gpt2_xl_print_perm_stats(tag, ob_map, rec_map):
    totals = {"q": [0, 0], "k": [0, 0], "v": [0, 0], "other": [0, 0]}
    for path, ob_leaf, rec_leaf in mpa_gpt2_xl_iter_leaf_pairs(ob_map, rec_map):
        m, t = mpa_gpt2_xl_accum_perm_recovery(ob_leaf, rec_leaf)
        g = mpa_gpt2_xl_leaf_group(path)
        totals[g][0] += m
        totals[g][1] += t
        if m < t:
            print(f"[{tag}] {path}  permutation leaf does not fully match ({m}/{t})")
    matched = sum(v[0] for v in totals.values())
    total = sum(v[1] for v in totals.values())
    print(f"[{tag}] permutation-structure recovery rate: {matched}/{total} = {matched/total if total else 0.0:.4f}")
    for g in ("q", "k", "v", "other"):
        m, t = totals[g]
        if t:
            print(f"[{tag}] [{g}] permutation-structure recovery rate: {m}/{t} = {m/t:.4f}")


def mpa_gpt2_xl_print_scaling_stats(tag, ob_map, rec_map):
    grouped = {"q": [], "k": [], "v": [], "other": []}
    for path, ob_leaf, rec_leaf in mpa_gpt2_xl_iter_leaf_pairs(ob_map, rec_map):
        grouped[mpa_gpt2_xl_leaf_group(path)].extend(
            mpa_gpt2_xl_rel_err_scaling(ob_leaf, rec_leaf)
        )
    all_errs = [err for errs in grouped.values() for err in errs]
    print(f"[{tag}] mean relative error of scaling factors: {np.mean(all_errs) if all_errs else 0.0}")
    for g in ("q", "k", "v", "other"):
        if grouped[g]:
            print(f"[{tag}] [{g}] mean relative error of scaling factors: {np.mean(grouped[g])}")


def mpa_gpt2_xl_print_fro_stats(tag, ob_map, rec_map, name):
    grouped = {"q": [], "k": [], "v": [], "other": []}
    for path, ob_leaf, rec_leaf in mpa_gpt2_xl_iter_leaf_pairs(ob_map, rec_map):
        grouped[mpa_gpt2_xl_leaf_group(path)].extend(
            mpa_gpt2_xl_rel_err_fro(ob_leaf, rec_leaf)
        )
    all_errs = [err for errs in grouped.values() for err in errs]
    print(f"[{tag}] {name} relative error: {np.mean(all_errs) if all_errs else 0.0}")
    for g in ("q", "k", "v", "other"):
        if grouped[g]:
            print(f"[{tag}] [{g}] {name} relative error: {np.mean(grouped[g])}")


def mpa_gpt2_xl_print_vector_stats(tag, ob_map, rec_map, name):
    grouped = {"q": [], "k": [], "v": [], "other": []}
    for path, ob_leaf, rec_leaf in mpa_gpt2_xl_iter_leaf_pairs(ob_map, rec_map):
        grouped[mpa_gpt2_xl_leaf_group(path)].extend(
            mpa_gpt2_xl_rel_err_vector_pairs(ob_leaf, rec_leaf)
        )
    all_errs = [err for errs in grouped.values() for err in errs]
    print(f"[{tag}] {name} mean relative error: {np.mean(all_errs) if all_errs else 0.0}")
    for g in ("q", "k", "v", "other"):
        if grouped[g]:
            print(f"[{tag}] [{g}] {name} mean relative error: {np.mean(grouped[g])}")


def mpa_gpt2_xl_print_cluster_stats(tag, ob_map, rec_map):
    totals = {"q": [0, 0], "k": [0, 0], "v": [0, 0], "other": [0, 0]}
    for path, ob_leaf, rec_leaf in mpa_gpt2_xl_iter_leaf_pairs(ob_map, rec_map):
        m, t = mpa_gpt2_xl_accum_cluster_recovery(ob_leaf, rec_leaf)
        g = mpa_gpt2_xl_leaf_group(path)
        totals[g][0] += m
        totals[g][1] += t
        if m < t:
            print(f"[{tag}] {path}  cluster-partition leaf does not fully match ({m}/{t})")
    matched = sum(v[0] for v in totals.values())
    total = sum(v[1] for v in totals.values())
    print(f"[{tag}] cluster-partition recovery rate: {matched}/{total} = {matched/total if total else 0.0:.4f}")
    for g in ("q", "k", "v", "other"):
        m, t = totals[g]
        if t:
            print(f"[{tag}] [{g}] cluster-partition recovery rate: {m}/{t} = {m/t:.4f}")


def _prepare_model_for_hf_pretrained_save(model):
    m = model
    if isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    with torch.no_grad():
        for p in m.parameters():
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()
        for buf in m.buffers():
            if isinstance(buf, torch.Tensor) and not buf.is_contiguous():
                buf.set_(buf.contiguous())
    return m


def save_restore_model_pre_finetune(restore_model, restore_dir, tokenizer, attack_extras=None):
    pre_dir = os.path.join(restore_dir, "pre_finetune_checkpoint")
    os.makedirs(pre_dir, exist_ok=True)
    to_save = _prepare_model_for_hf_pretrained_save(restore_model)
    to_save.save_pretrained(pre_dir, safe_serialization=True)
    tokenizer.save_pretrained(pre_dir)
    if attack_extras is not None:
        with open(os.path.join(pre_dir, "attack_extras.pkl"), "wb") as f:
            pickle.dump(attack_extras, f)

def is_target_weight(name):
    return (
        "attn.c_attn.weight" in name
        or "attn.c_proj.weight" in name
        or "mlp.c_fc.weight" in name
        or "mlp.c_proj.weight" in name
    )


def build_public_model(seed):
    set_seed(seed)
    public_model = GPT2ForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )
    public_model.config.pad_token_id = tokenizer.pad_token_id
    return public_model


def copy_public_non_target_modules(restore_model, seed):
    public_model = build_public_model(seed)
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
            if is_target_weight(name):
                continue
            if restore_tensor.shape != public_tensor.shape:
                skipped.append((name, "shape"))
                continue
            restore_tensor.copy_(
                public_tensor.to(device=restore_tensor.device, dtype=restore_tensor.dtype)
            )
            copied.append(name)

    restore_model.load_state_dict(restore_state)
    print(
        f"Used seed={seed} public model to overwrite non-target modules in restore_model: "
        f"copied={len(copied)}, skipped={len(skipped)}"
    )
    classifier_copied = [
        name for name in copied if name.startswith("score.") or name.startswith("classifier.")
    ]
    if classifier_copied:
        print(f"  copied classifier params: {classifier_copied}")
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

target_checkpoint_dir = args.weight_dir_tsqp if args.obfus == "tsqp" else args.weight_dir
target_model_is_lora = is_lora_checkpoint(target_checkpoint_dir)
print(
    f"Target checkpoint format: {'LoRA' if target_model_is_lora else 'full-parameter'} "
    f"({target_checkpoint_dir})"
)
print(
    "Recovery fine-tuning mode: "
    f"{'full-parameter' if str2bool(args.full_finetune) else 'LoRA'}"
)

del victim_model

_vic_model = None


def get_vic_model():
    global _vic_model
    if args.obfus == "tsqp":
        return None
    if _vic_model is None:
        _vic_model = load_finetuned_gpt2_xl(args.weight_dir)
    return _vic_model

def prepare_restore_model_for_finetune(restore_model):
    if str2bool(args.full_finetune):
        print("Recovery fine-tuning all restored model parameters")
    else:
        print("Recovery fine-tuning with LoRA adapters")
        restore_model = get_peft_model(restore_model, lora_config).model
    restore_model.config.pad_token_id = tokenizer.pad_token_id
    return restore_model


def finetune_restore_model(restore_model):
    restore_args = TrainingArguments(
        output_dir=f"{args.restore_dir}",
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy="epoch",
        learning_rate=args.recover_lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.recover_epochs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4,
        seed=42,
    )
    restore_model = prepare_restore_model_for_finetune(restore_model)
    trainer = Trainer(
        model=restore_model,
        args=restore_args,
        train_dataset=recover_dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    set_seed()
    trainer.train()
    restore_results = trainer.evaluate(eval_dataset=evalset)
    print(f"Final recovered model results:{restore_results}")
    return restore_model, restore_results


def eval_obfus_model(obfus_model):
    obfus_args = TrainingArguments(
        output_dir=f"{args.obfus_dir}",
        eval_strategy='no',
        save_strategy="no",
        per_device_eval_batch_size=args.bs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4,
        do_train=False,
        seed=42,
    )
    trainer = Trainer(
        model=obfus_model,
        args=obfus_args,
        train_dataset=None,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    obfus_result = trainer.evaluate(eval_dataset=evalset)
    print(f"Obfuscated model results: {obfus_result}")
    return obfus_result

if os.path.exists(f"{args.restore_dir}/pre_finetune_checkpoint"):
    pre_finetune_dir = f"{args.restore_dir}/pre_finetune_checkpoint"
    pre_finetune_snapshot = snapshot_checkpoint_dir(pre_finetune_dir)
    set_seed()
    restore_model = load_finetuned_gpt2_xl(pre_finetune_dir)
    restore_model = copy_public_non_target_modules(
        restore_model,
        seed=args.public_non_target_seed,
    )
    finetune_restore_model(restore_model)
    assert_checkpoint_dir_unchanged(pre_finetune_dir, pre_finetune_snapshot)
elif os.path.exists(f"{args.restore_dir}/final_checkpoint"):
    set_seed()
    final_model = load_finetuned_gpt2_xl(f"{args.restore_dir}/final_checkpoint")
    restore_args = TrainingArguments(
        output_dir=f"{args.restore_dir}",
        eval_strategy='no',
        save_strategy="no", 
        per_device_eval_batch_size=args.bs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4, 
        do_train=False,
    )
    trainer = Trainer(
        model=final_model,
        args=restore_args,
        train_dataset=recover_dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    final_results = trainer.evaluate(eval_dataset=evalset)
    print(f"Final recovered model results:{final_results}")
else:
    if args.obfus == "black":
        set_seed()
        restore_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        restore_model.config.pad_token_id = tokenizer.pad_token_id
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer)
        finetune_restore_model(restore_model)

    elif args.obfus == "translinkguard":  
        set_seed()
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels) 
        model = load_target_model()
        obfus_model, permutations,rows = ob_translinkguard(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_perm = attack_translinkguard_our(obfus_model, init_model, rows)
        _perm_failed, _n_perm = [], len(permutations)
        for _ly in sorted(permutations.keys()):
            if _ly not in restore_perm:
                _perm_failed.append(f"layer-{_ly} (missing restore_perm)")
                continue
            _p = permutations[_ly].cpu().numpy()
            _r = restore_perm[_ly].cpu().numpy()
            if _p.size != _r.size:
                _perm_failed.append(f"layer-{_ly} (dimension {_p.size}!={_r.size})")
                continue
            if not np.array_equal(_r, _p):
                _nd = int(np.sum(_r != _p))
                _perm_failed.append(f"layer-{_ly} ({_nd}/{_p.size} permutation does not match ground truth)")
        if _perm_failed:
            print("[translinkguard] modules whose permutation was not recovered successfully: " + " | ".join(_perm_failed))
        else:
            print("[translinkguard] modules whose permutation was not recovered successfully: none")
        print(
            f"[translinkguard] permutation-structure recovery rate (matches ground truth): "
            f"{_n_perm - len(_perm_failed)}/{_n_perm} = "
            f"{(_n_perm - len(_perm_failed)) / _n_perm if _n_perm else 0.0:.4f}"
        )
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={"obfus_perm": permutations, "restore_perm": restore_perm},
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    
    elif args.obfus == "tsqp":    
        set_seed()
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        model = load_target_model()
        obfus_model, scaling_factors = ob_tsqp(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no', 
            save_strategy="no", 
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_scaling_factors = attack_tsqp(obfus_model,init_model)
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                relative_errors.extend(
                    mpa_gpt2_xl_rel_err_scaling(scaling_factors[k], restore_scaling_factors[k])
                )
            else:
                print(f"module {k} does not have restore_scaling_factors")
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[tsqp] scaling_factors and restore_scaling_factors mean relative error: {avg_rel_error}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_scaling_factors": scaling_factors,
                "restore_scaling_factors": restore_scaling_factors,
            },
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  
            logging_strategy='epoch',
            save_strategy="epoch", 
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
        set_seed()
        
    elif args.obfus == "soter":
        set_seed()
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        model = load_target_model()
        obfus_model, scaling_factors, _ = ob_soter(model,init_model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no", 
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_scaling_factors = attack_soter_our(obfus_model, init_model)
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                relative_errors.extend(
                    mpa_gpt2_xl_rel_err_scaling(scaling_factors[k], restore_scaling_factors[k])
                )
            else:
                print(f"module {k} does not have restore_scaling_factors")
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[soter] scaling_factors and restore_scaling_factors mean relative error: {avg_rel_error}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_scaling_factors": scaling_factors,
                "restore_scaling_factors": restore_scaling_factors,
            },
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
        
    elif args.obfus == "tempo":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        model = load_target_model()
        obfus_model, permutations, scaling_factors = ob_tempo(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no", 
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_permutations, restore_scaling_factors = attack_tempo_our(obfus_model, init_model)
        cnt_matched = cnt_total = 0
        for k in permutations.keys():
            if k in restore_permutations:
                m, t = mpa_gpt2_xl_accum_perm_recovery(permutations[k], restore_permutations[k])
                cnt_matched += m
                cnt_total += t
        print(f"[tempo] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = {cnt_matched/cnt_total if cnt_total else 0.0:.4f}")
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                relative_errors.extend(
                    mpa_gpt2_xl_rel_err_scaling(scaling_factors[k], restore_scaling_factors[k])
                )
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[tempo] scaling-factor recovery score: {avg_rel_error}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_permutations": permutations,
                "obfus_scaling_factors": scaling_factors,
                "restore_permutations": restore_permutations,
                "restore_scaling_factors": restore_scaling_factors,
            },
        )
        # Run fine-tuning
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  
            logging_strategy='epoch',
            save_strategy="epoch", 
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        results = trainer.evaluate(eval_dataset=evalset)
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")

    elif args.obfus == "shadownet":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        model = load_target_model()
        obfus_model, permutations, scaling_factors = ob_shadownet(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no', 
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_permutations, restore_scaling_factors = attack_shadownet_our(obfus_model, init_model)
        cnt_matched = cnt_total = 0
        for k in permutations.keys():
            if k in restore_permutations:
                m, t = mpa_gpt2_xl_accum_perm_recovery(permutations[k], restore_permutations[k])
                cnt_matched += m
                cnt_total += t
        print(f"[shadownet] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = {cnt_matched/cnt_total if cnt_total else 0.0:.4f}")
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                relative_errors.extend(
                    mpa_gpt2_xl_rel_err_scaling(scaling_factors[k], restore_scaling_factors[k])
                )
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[shadownet] scaling-factor recovery score: {avg_rel_error}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_permutations": permutations,
                "obfus_scaling_factors": scaling_factors,
                "restore_permutations": restore_permutations,
                "restore_scaling_factors": restore_scaling_factors,
            },
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "LoRO":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, R = ob_LoRO(model, r=args.rank_r)
        eval_obfus_model(obfus_model)
        set_seed()
        restore_model, restore_R = attack_LoRO(obfus_model, init_model, vic_model=get_vic_model(), r=args.rank_r)
        relative_errors = []
        for k in R.keys():
            if k in restore_R:
                relative_errors.extend(mpa_gpt2_xl_rel_err_fro(R[k], restore_R[k]))
        print(f"[LoRO] R relative error: {np.mean(relative_errors) if relative_errors else 0.0}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={"obfus_R": R, "restore_R": restore_R},
        )
        finetune_restore_model(restore_model)

    elif args.obfus == "LRAO":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, R = ob_LRAO(model, init_model, r=args.rank_r)
        eval_obfus_model(obfus_model)
        set_seed()
        restore_model = attack_LRAO(obfus_model, init_model, r=args.rank_r)
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={"obfus_R": R},
        )
        finetune_restore_model(restore_model)

    elif args.obfus == "LRAO+arrowcloak":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, R = ob_LRAO(model, init_model, r=args.rank_r)
        obfus_model, obfus_permutations, obfus_masks, obfus_factors, obfus_weight_factors = ob_arrowcloak(obfus_model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',
            save_strategy="no",
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            do_train=False,
            seed=42,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(
            obfus_model, init_model, vic_model=get_vic_model()
        )
        save_restore_model_pre_finetune(
            restore_model,
            args.restore_dir,
            tokenizer,
            attack_extras={
                "obfus_permutations": obfus_permutations,
                "obfus_masks": obfus_masks,
                "obfus_factors": obfus_factors,
                "obfus_weight_factors": obfus_weight_factors,
                "restore_permutations": restore_permutations,
                "restore_L": restore_L,
                "restore_D": restore_D,
            },
        )
        cnt_matched = cnt_total = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                m, t = mpa_gpt2_xl_accum_perm_recovery(
                    obfus_permutations[k], restore_permutations[k]
                )
                cnt_matched += m
                cnt_total += t
        print(
            f"[arrowcloak] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = "
            f"{cnt_matched/cnt_total if cnt_total else 0.0:.4f}"
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
        

    elif args.obfus == "obfuscatune":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, Q = ob_obfuscatune(model)
        eval_obfus_model(obfus_model)
        set_seed()
        restore_model, restore_Q = attack_obfuscatune(obfus_model, init_model, vic_model=get_vic_model())
        relative_errors = []
        for k in Q.keys():
            if k in restore_Q:
                relative_errors.extend(mpa_gpt2_xl_rel_err_fro(Q[k], restore_Q[k]))
        print(f"[obfuscatune] Q relative error: {np.mean(relative_errors) if relative_errors else 0.0}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={"obfus_Q": Q, "restore_Q": restore_Q},
        )
        finetune_restore_model(restore_model)

    elif args.obfus == "groupcover":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, ob_cluster_index, ob_random_coeff_list, ob_permutation = ob_groupcover(model)
        eval_obfus_model(obfus_model)
        set_seed()
        restore_model, restore_permutation, restore_cluster_index = attack_groupcover(
            obfus_model, init_model, size=4, vic_model=get_vic_model()
        )
        cnt_matched = cnt_total = 0
        for k in ob_permutation.keys():
            if k in restore_permutation:
                m, t = mpa_gpt2_xl_accum_perm_recovery(ob_permutation[k], restore_permutation[k])
                cnt_matched += m
                cnt_total += t
                if m < t:
                    print(f"[groupcover] module {k}  column-permutation leaf does not fully match ({m}/{t})")
        print(
            f"[groupcover] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = "
            f"{cnt_matched/cnt_total if cnt_total else 0.0:.4f}"
        )
        cnt_matched_cl = cnt_total_cl = 0
        for k in ob_cluster_index.keys():
            if k not in restore_cluster_index:
                continue
            m, t = mpa_gpt2_xl_accum_cluster_recovery(
                ob_cluster_index[k], restore_cluster_index[k]
            )
            cnt_matched_cl += m
            cnt_total_cl += t
            if m < t:
                print(f"[groupcover] module {k}  cluster-partition leaf does not fully match ({m}/{t})")
        print(
            f"[groupcover] cluster-partition recovery rate: {cnt_matched_cl}/{cnt_total_cl} = "
            f"{cnt_matched_cl/cnt_total_cl if cnt_total_cl else 0.0:.4f}"
        )
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_cluster_index": ob_cluster_index,
                "obfus_random_coeff_list": ob_random_coeff_list,
                "obfus_permutation": ob_permutation,
                "restore_permutation": restore_permutation,
                "restore_cluster_index": restore_cluster_index,
            },
        )
        finetune_restore_model(restore_model)

    elif args.obfus == "twinshield":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, obfus_permutations, obfus_d = ob_twinshield(model)
        print("TwinShield obfuscated results are stored as packed Wo1/Wo2 tensors; skipping intermediate model evaluation")
        set_seed()
        restore_model, restore_permutations, restore_d = attack_twinshield(
            obfus_model, init_model, vic_model=get_vic_model(), dataset=args.dataset.upper()
        )
        cnt_matched = cnt_total = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                m, t = mpa_gpt2_xl_accum_perm_recovery(obfus_permutations[k], restore_permutations[k])
                cnt_matched += m
                cnt_total += t
        print(
            f"[twinshield] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = "
            f"{cnt_matched/cnt_total if cnt_total else 0.0:.4f}"
        )
        relative_errors = []
        for k in obfus_d.keys():
            if k in restore_d:
                relative_errors.extend(mpa_gpt2_xl_rel_err_vector_pairs(obfus_d[k], restore_d[k]))
        print(f"[twinshield] d mean relative error: {np.mean(relative_errors) if relative_errors else 0.0}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_permutations": obfus_permutations,
                "obfus_d": obfus_d,
                "restore_permutations": restore_permutations,
                "restore_d": restore_d,
            },
        )
        finetune_restore_model(restore_model)

    elif args.obfus == "arrowcloak":
        set_seed()
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        init_model.config.pad_token_id = tokenizer.pad_token_id
        model = load_target_model()
        obfus_model, obfus_permutations, obfus_masks, obfus_factors, obfus_weight_factors = ob_arrowcloak(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',
            save_strategy="no",
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            do_train=False,
            seed=42,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model results: {obfus_result}")
        set_seed()
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(
            obfus_model, init_model, vic_model=get_vic_model()
        )
        save_restore_model_pre_finetune(
            restore_model,
            args.restore_dir,
            tokenizer,
            attack_extras={
                "obfus_permutations": obfus_permutations,
                "obfus_masks": obfus_masks,
                "obfus_factors": obfus_factors,
                "obfus_weight_factors": obfus_weight_factors,
                "restore_permutations": restore_permutations,
                "restore_L": restore_L,
                "restore_D": restore_D,
            },
        )
        cnt_matched = cnt_total = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                m, t = mpa_gpt2_xl_accum_perm_recovery(
                    obfus_permutations[k], restore_permutations[k]
                )
                cnt_matched += m
                cnt_total += t
        print(
            f"[arrowcloak] permutation-structure recovery rate: {cnt_matched}/{cnt_total} = "
            f"{cnt_matched/cnt_total if cnt_total else 0.0:.4f}"
        )
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        restore_model = prepare_restore_model_for_finetune(restore_model)
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    else:
        raise ValueError("Invalid obfuscation method")
