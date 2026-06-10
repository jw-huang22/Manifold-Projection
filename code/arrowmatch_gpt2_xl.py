import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import datasets
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
os.makedirs(args.obfus_dir, exist_ok=True)
set_seed()

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
sentence1_key, sentence2_key = task_to_keys[args.dataset]
trainset, evalset, tokenizer = prepare_data(actual_task, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print("Loading metric..")
metric = evaluate.load('glue', actual_task)

# load model
set_seed()
safetensor_file1, safetensor_file2 = args.weight_dir + "/model-00001-of-00002.safetensors", args.weight_dir + "/model-00002-of-00002.safetensors"
weight1, weight2 = load_file(safetensor_file1), load_file(safetensor_file2)
model = GPT2ForSequenceClassification.from_pretrained("gpt2-xl", num_labels=num_labels)
lora_config = LoraConfig(
        r=8,  
        lora_alpha=16,  
        lora_dropout=0.1,  
        target_modules=["c_fc", "c_attn", "c_proj"],  # target layers for LoRA
    )
lora_model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
lora_model.config.pad_token_id = tokenizer.pad_token_id


path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    print("Preparing recover_data..")
    prepare_recover_data(lora_model, trainset, args.bs, path, ratio = 0.01)

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

if args.obfus == "tsqp":
    safetensor_file1, safetensor_file2 = args.weight_dir_tsqp + "/model-00001-of-00002.safetensors", args.weight_dir_tsqp + "/model-00002-of-00002.safetensors"
else:
    safetensor_file1, safetensor_file2 = args.weight_dir + "/model-00001-of-00002.safetensors", args.weight_dir + "/model-00002-of-00002.safetensors"
weight1, weight2 = load_file(safetensor_file1), load_file(safetensor_file2)
lora_config = LoraConfig(
        r=8,  
        lora_alpha=16,  
        lora_dropout=0.1,  
        target_modules=["c_fc", "c_attn", "c_proj"],  # target layers for LoRA
    )
lora_model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
lora_model.config.pad_token_id = tokenizer.pad_token_id

del lora_model, model


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
    restore_model = get_peft_model(restore_model, lora_config).model
    restore_model.config.pad_token_id = tokenizer.pad_token_id
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


def build_obfus_model(obfus_name, init_model):
    rank_r = args.rank_r
    target_model = adjust_lora_model(
        'gpt2-xl',
        lora_config=lora_config,
        num_labels=num_labels,
        weight1=weight1,
        weight2=weight2,
    )
    target_model.config.pad_token_id = tokenizer.pad_token_id
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
        obfus_model, _ = ob_LoRO(target_model, r=rank_r)
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

if os.path.exists(f"{args.restore_dir}/final_checkpoint"):
    set_seed()
    restore_weight1, restore_weight2 = load_file(f"{args.restore_dir}/final_checkpoint/model-00001-of-00002.safetensors"), load_file(f"{args.restore_dir}/final_checkpoint/model-00002-of-00002.safetensors")
    final_model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=restore_weight1, weight2=restore_weight2)
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
        obfus_model, attack_meta = build_obfus_model(args.obfus, init_model)
        if args.obfus != "twinshield":
            evaluate_obfus_model(obfus_model)
        else:
            print("TwinShield obfuscated results are stored in column-packed form; skipping intermediate model evaluation")
        set_seed()
        restore_model = attack_obfus_model(args.obfus, obfus_model, init_model, attack_meta)
    restore_model.config.pad_token_id = tokenizer.pad_token_id
    finetune_restore_model(restore_model)
    sys.exit(0)

    if args.obfus == "translinkguard":  
        set_seed()
        init_model =  GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels) 
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        restore_model = attack_translinkguard(obfus_model,init_model, rows)
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
        restore_model = get_peft_model(restore_model, lora_config).model
        restore_model.config.pad_token_id = tokenizer.pad_token_id
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
        weight1, weight2 = load_file(f"{args.weight_dir_tsqp}/model-00001-of-00002.safetensors"), load_file(f"{args.weight_dir_tsqp}/model-00002-of-00002.safetensors")
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        restore_model = get_peft_model(restore_model, lora_config).model
        restore_model.config.pad_token_id = tokenizer.pad_token_id
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
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        restore_model = get_peft_model(restore_model, lora_config).model
        restore_model.config.pad_token_id = tokenizer.pad_token_id
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
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        results = trainer.evaluate(eval_dataset=evalset)
        restore_model = get_peft_model(restore_model, lora_config).model
        restore_model.config.pad_token_id = tokenizer.pad_token_id
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
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
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
        restore_model, restore_permutations = attack_shadownet(obfus_model,init_model)
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
        restore_model = get_peft_model(restore_model, lora_config).model
        restore_model.config.pad_token_id = tokenizer.pad_token_id
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
