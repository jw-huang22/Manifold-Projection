
import os
import random
import numpy as np
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import evaluate
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import pickle
from utils.utils import *
from pdb import set_trace as st
from utils.methods import *


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="cola", type=str, help="dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="2,3", help="gpu ids")
parser.add_argument("--recover_lr", default=1e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--rank_r", default=32, type=int, help="Rank used by LRAO low-rank obfuscation/recovery")

parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--restore_dir", default="results/arrowmatch_results", type=str, help="restore directory")
parser.add_argument("--obfus_dir", default="tmp/obfus_results", type=str, help="obfus directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--recover_data_ratio", default=0.01, type=float, help="Ratio of recover_data to trainset")
parser.add_argument(
    "--public_non_target_seed",
    default=43,
    type=int,
    help="seed used to rebuild the public model before copying non-target modules",
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if args.model == "bert-base-cased":
    model_name = "bert"
else:
    print("The code about ViT and GPT2 will be published before AE phase.")
    print("Please try bert-base-cased for now.")
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

# number of classes in the dataset
actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"

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
sentence1_key, sentence2_key = task_to_keys[args.dataset]
trainset, evalset, tokenizer = prepare_data(actual_task, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)

# Model & metric
print("Building model..")
set_seed()

model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )

print("Loading metric..")
metric = evaluate.load('glue', actual_task)

path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    prepare_recover_data(model, trainset, args.bs, path, ratio = args.recover_data_ratio)
recover_dataset = load_dataset("json", data_files=path)["train"]
print("recover_data prepared!")

if args.obfus == "tsqp":
    if os.path.exists(args.weight_dir_tsqp):
        model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir_tsqp,  num_labels=num_labels,use_safetensors=True )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels,use_safetensors=True )
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )
    vic_model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def save_restore_model_pre_finetune(restore_model, restore_dir, tokenizer, attack_extras=None):
    """Before recovery fine-tuning, save the current recovered model and tokenizer to pre_finetune_checkpoint.

    attack_extras: attack_* metadata returned by attack_* besides the model; saved as attack_extras.pkl in the same directory.
    """
    pre_dir = os.path.join(restore_dir, "pre_finetune_checkpoint")
    os.makedirs(pre_dir, exist_ok=True)
    restore_model.save_pretrained(pre_dir, safe_serialization=True)
    tokenizer.save_pretrained(pre_dir)
    if attack_extras is not None:
        pkl_path = os.path.join(pre_dir, "attack_extras.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(attack_extras, f)


def is_target_weight(name):
    return (
        "query.weight" in name
        or "key.weight" in name
        or "value.weight" in name
        or "output.dense.weight" in name
        or "intermediate.dense.weight" in name
    )


def build_public_model(seed):
    set_seed(seed)
    public_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )
    if tokenizer.pad_token_id is not None:
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
        name for name in copied if name.startswith("classifier.") or name.startswith("score.")
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


if os.path.exists(f"{args.restore_dir}/pre_finetune_checkpoint"):
    print("Loading pre-finetune checkpoint for finetune..")
    pre_finetune_dir = f"{args.restore_dir}/pre_finetune_checkpoint"
    pre_finetune_snapshot = snapshot_checkpoint_dir(pre_finetune_dir)
    set_seed()
    restore_model = AutoModelForSequenceClassification.from_pretrained(pre_finetune_dir, num_labels=num_labels, use_safetensors=True)
    restore_model.config.pad_token_id = tokenizer.pad_token_id
    restore_model = copy_public_non_target_modules(
        restore_model,
        seed=args.public_non_target_seed,
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
    trainer.train()
    restore_results = trainer.evaluate(eval_dataset=evalset)
    assert_checkpoint_dir_unchanged(pre_finetune_dir, pre_finetune_snapshot)
    print(f"Final recovered model results:{restore_results}")
elif os.path.exists(f"{args.restore_dir}/final_checkpoint") and False:
    print("Loading final model..")
    set_seed()
    final_model = AutoModelForSequenceClassification.from_pretrained(f"{args.restore_dir}/final_checkpoint", num_labels=num_labels, use_safetensors=True)
    final_model.config.pad_token_id = tokenizer.pad_token_id
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
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        restore_model = init_model
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer)
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
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "translinkguard":  
        set_seed() 
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        restore_model, restore_perm = attack_translinkguard_our(obfus_model,init_model, rows) 
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
                _perm_failed.append(f"layer-{_ly} ({_nd}/{_p.size} columnpermutation does not match ground truth)")
        if _perm_failed:
            print("[translinkguard] columnmodules whose permutation was not recovered successfully: " + " | ".join(_perm_failed))
        else:
            print("[translinkguard] columnmodules whose permutation was not recovered successfully: none")
        print(
            f"[translinkguard] permutation-structure recovery rate (column permutation matches ground truth): "
            f"{_n_perm - len(_perm_failed)}/{_n_perm} = {(_n_perm - len(_perm_failed)) / _n_perm if _n_perm else 0.0:.4f}"
        )
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_perm": permutations,
                "restore_perm": restore_perm,
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")

    elif args.obfus == "tsqp":
        set_seed()
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={"restore_scaling_factors": restore_scaling_factors},
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
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")

    elif args.obfus == "soter":
        set_seed()
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        restore_model, restore_scaling_factors = attack_soter_our(obfus_model,init_model)
        # restore_model, restore_scaling_factors = attack_soter(obfus_model,init_model)
        
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                sf = scaling_factors[k]
                rsf = restore_scaling_factors[k]
                
                if isinstance(sf, torch.Tensor):
                    sf = sf.detach().cpu().numpy()
                else:
                    sf = np.array(sf)
                    
                if isinstance(rsf, torch.Tensor):
                    rsf = rsf.detach().cpu().numpy()
                else:
                    rsf = np.array(rsf)
                    
                rel_err = np.abs(rsf - sf) / (np.abs(sf) + 1e-12)
                relative_errors.append(np.mean(rel_err))
            else:
                print(f"module {k} does not have restore_scaling_factors")
                
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[soter] scaling_factors and restore_scaling_factors mean relative error: {avg_rel_error}")

        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_scaling_factors": scaling_factors, "restore_scaling_factors": restore_scaling_factors
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
           
    elif args.obfus == "tempo":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        # restore_model, restore_permutations = attack_tempo(obfus_model,init_model)
        restore_model, restore_permutations, restore_scaling_factors = attack_tempo_our(obfus_model,init_model)
        # check permutation is the same
        cnt = 0
        cnt_failed = 0
        for k in permutations.keys():
            if k in restore_permutations:
                p = permutations[k]
                rp = restore_permutations[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
                cnt += 1
        print(f"[tempo] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")
        # cehck scaling factors relative error
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                sf = scaling_factors[k]
                rsf = restore_scaling_factors[k]
                rel_err = np.abs(rsf - sf) / (np.abs(sf) + 1e-12)
                relative_errors.append(np.mean(rel_err))
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")

    elif args.obfus == "shadownet":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, permutations, scaling_factors = ob_shadownet(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model, restore_permutations, restore_scaling_factors = attack_shadownet_our(obfus_model,init_model)
        # restore_model, restore_permutations = attack_shadownet(obfus_model,init_model)

        # check permutation is the same
        cnt = 0
        cnt_failed = 0
        for k in permutations.keys():
            if k in restore_permutations:
                p = permutations[k]
                rp = restore_permutations[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
                cnt += 1
        print(f"[shadownet] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")
        # cehck scaling factors relative error
        relative_errors = []
        for k in scaling_factors.keys():
            if k in restore_scaling_factors:
                sf = scaling_factors[k]
                rsf = restore_scaling_factors[k]
                rel_err = np.abs(rsf - sf) / (np.abs(sf) + 1e-12)
                relative_errors.append(np.mean(rel_err))
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "LoRO":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        rank_r = args.rank_r
        obfus_model, R = ob_LoRO(model, r=rank_r)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model, restore_R = attack_LoRO(obfus_model,init_model, r=rank_r)
        # restore_model, _ = attack_arrowcloak(obfus_model,init_model)
        # check R relative fro norm
        relative_errors = []
        for k in R.keys():
            if k in restore_R:
                r = R[k]
                rr = restore_R[k]
                rel_err = torch.norm(rr - r, p='fro') / (torch.norm(r, p='fro') + 1e-12)
                relative_errors.append(rel_err.item())
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[LoRO] R relative error: {avg_rel_error}")
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_R": R,
                "restore_R": restore_R,
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "LRAO":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        rank_r = args.rank_r
        obfus_model, R = ob_LRAO(model, init_model, r=rank_r)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model = attack_LRAO(obfus_model,init_model, vic_model=vic_model,r=rank_r)
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer,
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "LRAO+arrowcloak":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        rank_r = args.rank_r
        obfus_model, R = ob_LRAO(model, init_model, r=rank_r)
        obfus_model, obfus_permutations, obfus_masks, obfus_factors, obfus_weight_factors = ob_arrowcloak(obfus_model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(obfus_model,init_model,vic_model)
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
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
        # check permutation is the same
        cnt = 0
        cnt_failed = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                p = obfus_permutations[k].cpu().numpy()
                rp = restore_permutations[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
                cnt += 1
        print(f"[arrowcloak] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")

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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "obfuscatune":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, Q = ob_obfuscatune(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model, restore_Q = attack_obfuscatune(obfus_model,init_model,vic_model=None)
        # restore_model, _ = attack_arrowcloak(obfus_model,init_model)
        # check Q relative fro norm
        relative_errors = []
        for k in Q.keys():
            if k in restore_Q:
                q = Q[k]
                rq = restore_Q[k]
                rel_err = torch.norm(rq - q, p='fro') / (torch.norm(q, p='fro') + 1e-12)
                relative_errors.append(rel_err.item())
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[obfuscatune] Q relative error: {avg_rel_error}")
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_Q": Q,
                "restore_Q": restore_Q,
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "groupcover":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, ob_cluster_index, ob_random_coeff_list, ob_permutation = ob_groupcover(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        groupcover_th = 0.15
        groupcover_max_th = 1.0
        restore_model, restore_permutation, restore_cluster_index = attack_groupcover(
            obfus_model,
            init_model,
            size=4,
            vic_model=vic_model,
            fast=False,
            fast_only=False,
            partial=False,
        )
        save_restore_model_pre_finetune(restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_cluster_index": ob_cluster_index,
                "obfus_random_coeff_list": ob_random_coeff_list,
                "obfus_permutation": ob_permutation,
                "restore_permutation": restore_permutation,
                "restore_cluster_index": restore_cluster_index,
            },
        )
        # check permutation is the same
        cnt = restore_permutation.keys().__len__()
        cnt_failed = 0
        for k in ob_permutation.keys():
            if k in restore_permutation:
                p = ob_permutation[k]
                rp = restore_permutation[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
        print(f"[groupcover] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")
        # check cluster index is the same
        def multiset_partitions_equal(clusters_a, clusters_b):
            """Compare whether two row-cluster sets represent the same partition by multiset equality, ignoring order."""
            if len(clusters_a) != len(clusters_b):
                return False

            def canon(cs):
                return sorted(tuple(sorted(s)) for s in cs)

            return canon(clusters_a) == canon(clusters_b)
        cnt = restore_cluster_index.keys().__len__()
        cnt_failed = 0
        for k in ob_cluster_index.keys():
            if k in restore_cluster_index:
                c = ob_cluster_index[k]
                rc = restore_cluster_index[k]
                if not multiset_partitions_equal(c, rc):
                    print(f"module {k}  of cluster partition mismatch")
                    cnt_failed += 1
        print(f"[groupcover] cluster-partition recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "twinshield":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, obfus_permutations, obfus_d = ob_twinshield(model)
        print("TwinShield obfuscated results are stored as packed Wo1/Wo2 tensors; skipping intermediate model evaluation")
        set_seed()
        restore_model, restore_permutations, restore_d = attack_twinshield(
            obfus_model, init_model, vic_model=vic_model
        )
        cnt = 0
        cnt_failed = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                p = obfus_permutations[k]
                rp = restore_permutations[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
                cnt += 1
        print(f"[twinshield] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")
        relative_errors = []
        for k in obfus_d.keys():
            if k in restore_d:
                d = np.array(obfus_d[k])
                rd = np.array(restore_d[k])
                rel_err = np.abs(rd - d) / (np.abs(d) + 1e-12)
                relative_errors.append(np.mean(rel_err))
        avg_rel_error = np.mean(relative_errors) if relative_errors else 0.0
        print(f"[twinshield] d mean relative error: {avg_rel_error}")
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
            attack_extras={
                "obfus_permutations": obfus_permutations,
                "obfus_d": obfus_d,
                "restore_permutations": restore_permutations,
                "restore_d": restore_d,
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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    elif args.obfus == "arrowcloak":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, obfus_permutations, obfus_masks, obfus_factors, obfus_weight_factors = ob_arrowcloak(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
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
        restore_model, restore_permutations, restore_L, restore_D = attack_arrowcloak_our(obfus_model,init_model,vic_model)
        save_restore_model_pre_finetune(
            restore_model, args.restore_dir, tokenizer,
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
        # check permutation is the same
        cnt = 0
        cnt_failed = 0
        for k in obfus_permutations.keys():
            if k in restore_permutations:
                p = obfus_permutations[k].cpu().numpy()
                rp = restore_permutations[k]
                if not np.array_equal(rp, p):
                    print(f"module {k}  of permutation mismatch")
                    cnt_failed += 1
                cnt += 1
        print(f"[arrowcloak] permutation-structure recovery rate: {cnt - cnt_failed}/{cnt} = {(cnt - cnt_failed)/cnt if cnt else 0.0:.4f}")

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
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"Final recovered model results:{restore_results}")
    else:
        raise ValueError("Invalid obfuscation method")
