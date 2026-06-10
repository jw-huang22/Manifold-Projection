import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import torch
import argparse
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

parser.add_argument("--obfus", default="arrowcloak", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
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
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
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
    model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
    model.config.pad_token_id = tokenizer.pad_token_id
    obfus_model, permutations, masks, scaling_factors, norms = ob_arrowcloak(model)
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
    restore_model, restore_permutations = attack_arrowcloak(obfus_model,init_model)
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
    print(f"Results after recovering arrowcloak:{restore_results}")
