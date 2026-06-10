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
import evaluate
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import pickle
from pdb import set_trace as st
from utils.utils import *
from utils.methods import *


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="sst2", type=str, help="dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--recover_lr", default=1e-5, type=float)
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids") # GPUs to use

parser.add_argument("--output_dir", default="evaluate_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
parser.add_argument("--result_of_whitebox_model", default="true", type=str)
parser.add_argument("--result_of_blackbox_model", default="true", type=str)
parser.add_argument("--result_of_obfus_model", default="true", type=str)
parser.add_argument("--result_of_recover_model", default="true", type=str)
parser.add_argument("--result_of_arrowcloak_model", default="true", type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "bert"
args.output_dir = f"{args.output_dir}/{model_name}/{args.dataset}"
os.makedirs(f"{args.output_dir}", exist_ok=True)


args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}/final_checkpoint"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}/final_checkpoint"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

# number of classes in the dataset
actual_task = args.dataset
num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
validation_key =  "validation_matched" if args.dataset == "mnli" else "validation"


# Prepare data
set_seed()
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

print("Loading metric..")
metric = evaluate.load('glue', actual_task)


model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels,use_safetensors=True)

path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    prepare_recover_data(model, trainset, args.bs, path, ratio = 0.01)
recover_dataset = load_dataset("json", data_files=path)["train"]
print("recover_data prepared!")


training_args = TrainingArguments(
    output_dir=f"{args.output_dir}",
    eval_strategy='no',  
    save_strategy="epoch",  
    learning_rate=args.lr,
    per_device_train_batch_size=args.bs,
    per_device_eval_batch_size=args.bs,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    dataloader_num_workers=4,  
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if args.result_of_whitebox_model == "true":
    set_seed()
    whitebox_model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels)
    trainer = Trainer(
        model=whitebox_model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    results = trainer.evaluate(eval_dataset=evalset)
    print(f"White-box(Whitebox model) evaluation results of {model_name} on {args.dataset}: {results}")
    ## Saved under args.output_dir
    with open(f"{args.output_dir}/whitebox_results.txt", "w") as f:
        f.write(f"White-box(Whitebox model) evaluation results of {model_name} on {args.dataset}: {results}")


""""""
if args.result_of_blackbox_model == "true":
    set_seed()
    blackmodel_args = TrainingArguments(
        output_dir=f"{args.output_dir}",
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
    init_model = AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=num_labels,use_safetensors=True)
    trainer = Trainer(
            model=init_model,
            args=blackmodel_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    trainer.train()
    blackbox_results = trainer.evaluate(eval_dataset=evalset)
    print(f"Black-box(Blackbox model) evaluation results of {model_name} on {args.dataset}: {blackbox_results}")
    with open(f"{args.output_dir}/blackbox_results.txt", "w") as f:
        f.write(f"Black-box(Blackbox model) evaluation results of {model_name} on {args.dataset}: {blackbox_results}")

if args.result_of_obfus_model == "true":
    for obfus in ["soter", "tsqp", "translinkguard", "tempo", "shadownet"]:
        set_seed()
        model = init_obfus_model(args, num_labels, obfus)
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_args = TrainingArguments(
            output_dir=f"{args.output_dir}",
            eval_strategy='no',
            save_strategy="no", 
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        if obfus == "translinkguard":
            obfus_model, _, _ = ob_translinkguard(model)
        elif obfus == "tsqp":
            obfus_model, _ = ob_tsqp(model)
        elif obfus == "soter":
            obfus_model, _, _ = ob_soter(model, init_model)
        elif obfus == "shadownet":
            obfus_model, _, _ = ob_shadownet(model)
        elif obfus == "tempo":
            obfus_model, _, _ = ob_tempo(model)
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"Obfuscated model(Obfus_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {obfus_result}")
        ## Avoid overwriting
        with open(f"{args.output_dir}/obfus_results.txt", "a") as f:
            f.write(f"Obfuscated model(Obfus_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {obfus_result}\n")
    
if args.result_of_recover_model == "true":
    for obfus in ["soter", "tsqp", "translinkguard", "tempo", "shadownet"]:
        set_seed()
        recover_dir = f"results/arrowmatch_results/{model_name}/{obfus}/{args.dataset}/final_checkpoint"
        recover_model = AutoModelForSequenceClassification.from_pretrained(recover_dir, num_labels=num_labels,use_safetensors=True)
        trainer = Trainer(
            model=recover_model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        results = trainer.evaluate(eval_dataset=evalset)
        print(f"Recovered model(Recover_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {results}")
        with open(f"{args.output_dir}/recover_results.txt", "a") as f:
            f.write(f"Recovered model(Recover_model) evaluation results of {model_name}-{obfus} on {args.dataset}: {results}\n")

if args.result_of_arrowcloak_model == "true":
    set_seed()
    recover_dir = f"results/arrowcloak_results/{model_name}/arrowcloak/{args.dataset}/final_checkpoint"
    recover_model = AutoModelForSequenceClassification.from_pretrained(recover_dir, num_labels=num_labels,use_safetensors=True)
    trainer = Trainer(
        model=recover_model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    results = trainer.evaluate(eval_dataset=evalset)
    print(f"Recovered model(Recover_model) evaluation results of {model_name}-arrowcloak on {args.dataset}: {results}")
    with open(f"{args.output_dir}/arrowcloak_results.txt", "a") as f:
        f.write(f"Recovered model(Recover_model) evaluation results of {model_name}-arrowcloak on {args.dataset}: {results}\n")

print("=====================================================================")