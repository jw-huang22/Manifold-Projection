import os
import random
import numpy as np
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    GPT2ForSequenceClassification, 
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
from utils.utils_gpt2 import *
from pdb import set_trace as st
from utils.methods_gpt2 import *



parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="cola", type=str, help="dataset")
parser.add_argument("--model", default="gpt2", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="0", help="gpu ids")
parser.add_argument("--recover_lr", default=1e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--rank_r", default=32, type=int, help="Rank used by LRAO/LoRO obfuscation")

parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--restore_dir", default="results/arrowmatch_results", type=str, help="restore directory")
parser.add_argument("--obfus_dir", default="tmp/obfus_results", type=str, help="obfus directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if args.model == "gpt2":
    model_name = "gpt2_base"
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_obfus_model(obfus_model):
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


def build_obfus_model(obfus_name, target_model, init_model):
    rank_r = args.rank_r
    if obfus_name == "translinkguard":
        obfus_model, _, rows = ob_translinkguard(target_model)
        attack_meta = rows
    elif obfus_name == "tsqp":
        obfus_model, _ = ob_tsqp(target_model)
        attack_meta = None
    elif obfus_name == "soter":
        obfus_model, _, _ = ob_soter(target_model, init_model)
        attack_meta = None
    elif obfus_name == "tempo":
        obfus_model, _, _ = ob_tempo(target_model)
        attack_meta = None
    elif obfus_name == "shadownet":
        obfus_model, _, _ = ob_shadownet(target_model)
        attack_meta = None
    elif obfus_name == "LoRO":
        obfus_model, _ = ob_LoRO(target_model, r=rank_r, noise=1)
        attack_meta = None
    elif obfus_name == "LRAO":
        obfus_model, _ = ob_LRAO(target_model, init_model, r=rank_r)
        attack_meta = None
    elif obfus_name == "LRAO+arrowcloak":
        obfus_model, _ = ob_LRAO(target_model, init_model, r=rank_r)
        obfus_model, _, _, _, _ = ob_arrowcloak(obfus_model)
        attack_meta = None
    elif obfus_name == "obfuscatune":
        obfus_model, _ = ob_obfuscatune(target_model)
        attack_meta = None
    elif obfus_name == "groupcover":
        obfus_model, _, _, _ = ob_groupcover(target_model)
        attack_meta = None
    elif obfus_name == "twinshield":
        obfus_model, _, _ = ob_twinshield(target_model)
        pre_state = init_model.state_dict()
        for name, module in obfus_model.named_parameters():
            if name in pre_state and module.data.ndim == 2:
                pre_cols = pre_state[name].shape[1]
                if module.data.shape[1] != 2 * pre_cols:
                    continue
                if "attn.c_attn.weight" in name and pre_cols % 3 == 0:
                    part_cols = pre_cols // 3
                    parts = module.data.chunk(3, dim=1)
                    module.data = torch.cat(
                        [part[:, :part_cols] for part in parts],
                        dim=1,
                    ).clone().contiguous()
                else:
                    module.data = module.data[:, :pre_cols].clone().contiguous()
        attack_meta = None
    elif obfus_name == "arrowcloak":
        obfus_model, _, _, _, _ = ob_arrowcloak(target_model)
        attack_meta = None
    else:
        raise ValueError("Invalid obfuscation method")
    return obfus_model, attack_meta


def attack_obfus_model(obfus_name, obfus_model, init_model, attack_meta):
    if obfus_name == "translinkguard":
        return attack_translinkguard(obfus_model, init_model, attack_meta)
    if obfus_name == "tsqp":
        restore_model, _ = attack_tsqp(obfus_model, init_model)
        return restore_model
    if obfus_name == "soter":
        restore_model, _ = attack_soter(obfus_model, init_model)
        return restore_model
    if obfus_name == "tempo":
        restore_model, _ = attack_tempo(obfus_model, init_model)
        return restore_model
    if obfus_name == "shadownet":
        restore_model, _ = attack_shadownet(obfus_model, init_model)
        return restore_model
    restore_model, _ = attack_arrowcloak(obfus_model, init_model)
    print(f"[arrowmatch] {obfus_name} was not originally covered; using attack_arrowcloak for recovery")
    return restore_model

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
model = GPT2ForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True)
model.config.pad_token_id = tokenizer.pad_token_id


print("Loading metric..")
metric = evaluate.load('glue', actual_task)

path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    print("Preparing recover_data..")
    prepare_recover_data(model, trainset, args.bs, path, ratio = 0.01)
recover_dataset = load_dataset("json", data_files=path)["train"]
print("Prepare data!")

# Model & metric
print("Building model..")
set_seed()
model = init_obfus_model(args, num_labels, obfus=args.obfus)
model.config.pad_token_id = tokenizer.pad_token_id

if os.path.exists(f"{args.restore_dir}/final_checkpoint") and False:
    set_seed()
    final_model = GPT2ForSequenceClassification.from_pretrained(f"{args.restore_dir}/final_checkpoint", num_labels=num_labels, use_safetensors=True)
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
    set_seed()
    init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    if args.obfus == "black":
        print("black baseline: no obfuscation, fine-tuning public init model directly")
        restore_model = init_model
    else:
        obfus_model, attack_meta = build_obfus_model(args.obfus, model, init_model)
        if args.obfus != "twinshield":
            evaluate_obfus_model(obfus_model)
        else:
            print("TwinShield obfuscated results are stored in column-packed form; skipping intermediate model evaluation")
        set_seed()
        restore_model = attack_obfus_model(args.obfus, obfus_model, init_model, attack_meta)
    restore_model.config.pad_token_id = tokenizer.pad_token_id
    finetune_restore_model(restore_model)
    sys.exit(0)

    # obfuscation with TransLinkGuard
    if args.obfus == "translinkguard":  
        set_seed()
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels) 
        # Obfuscate the model and evaluate obfuscated-model performance
        obfus_model, permutations,rows = ob_translinkguard(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  # model evaluation frequency
            save_strategy="no",  # model save frequency
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  # number of parallel dataloader workers
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
        restore_model = attack_translinkguard(obfus_model,init_model, rows)
        # Run fine-tuning
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  # evaluate every epoch
            logging_strategy='epoch',
            save_strategy="epoch",  # save every epoch
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  # number of parallel dataloader workers
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
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        restore_model, restore_scaling_factors = attack_soter(obfus_model,init_model)
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
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
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
        restore_model, restore_permutations = attack_tempo(obfus_model,init_model)
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
        init_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, permutations, scaling_factors = ob_shadownet(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  # model evaluation frequency
            save_strategy="no",  # model save frequency
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  # number of parallel dataloader workers
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
        restore_model, restore_permutations = attack_shadownet(obfus_model,init_model)
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
