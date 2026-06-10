import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HOME", ".cache/huggingface")
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
from pdb import set_trace as st
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import *
from utils.utils_gpt2 import *

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a BERT-like model on a GLUE task.')
    parser.add_argument("--dataset", default="sst2", type=str, help="Any dataset in GLUE")
    parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate for training")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--output_dir", default="results/train_results", type=str, help="Directory to save fine-tuned model")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpus", default="2,3", type=str, help="GPU to use")
    parser.add_argument("--tsqp", default="false", type=str, help="Whether to use TSQP")
    return parser.parse_args()


def fine_tune(args):
    """Fine-tune the model with specified arguments."""
    set_seed(args.seed)
    if args.model == "bert-base-cased":
        model_name = "bert"
    elif args.model == "gpt2":
        model_name = "gpt2_base"
    else:
        raise ValueError("Invalid model name")
    args.output_dir = f"{args.output_dir}/{model_name}"
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

    actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
    num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
    validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    dataset = load_dataset("glue", actual_task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=args.max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Load model
    print("Initializing model...")
    set_seed()
    if args.model == "bert-base-cased":
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    elif args.model == "gpt2":
        model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load metric
    print("Loading metric...")
    metric = evaluate.load('glue', actual_task)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.dataset}",
        evaluation_strategy='epoch', 
        save_strategy="epoch", 
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir="./logs", 
        logging_steps=100,  
        dataloader_num_workers=4,   
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if actual_task != "stsb":
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    print("Initializing Trainer...")
    if args.tsqp == "true":
        set_seed()
        if args.model == "bert-base-cased":
            pre_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
            trainer = CustomTrainer(
                model=model,
                pre_model = pre_model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets[validation_key],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )   
        elif args.model == "gpt2":
            pre_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
            pre_model.config.pad_token_id = tokenizer.pad_token_id
            trainer = CustomTrainer_gpt(
                model=model,
                pre_model = pre_model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets[validation_key],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )   
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets[validation_key],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    # Evaluate the pre-trained model
    print("Evaluate the pre-trained model...")
    pretrained_model_perf =  trainer.evaluate(eval_dataset=tokenized_datasets[validation_key])
    print(pretrained_model_perf)
    
    # Train the model
    print("Start fine-tuning...")
    trainer.train()

    # Evaluate the fine-tuned model
    print("Evaluate the fine-tuned model...")
    fine_tuned_model_perf = trainer.evaluate(eval_dataset=tokenized_datasets[validation_key])
    print(fine_tuned_model_perf)
    
    print(f"Pre-trained model performance on {args.dataset}: {pretrained_model_perf}")
    print(f"Fine-tuned model performance on {args.dataset}: {fine_tuned_model_perf}")

if __name__ == "__main__":
    args = parse_args()
    if args.model == "bert-base-cased":
        if args.dataset == "mnli" or args.dataset == "sst2":
            args.lr = 2e-5
        elif args.dataset == "qnli" or args.dataset == "qqp":
            args.lr = 3e-5
        else:
            raise ValueError("Invalid dataset")
    elif args.model == "gpt2":
        if args.dataset == "mnli" or args.dataset == "qnli":
            args.lr = 2e-5
        elif args.dataset == "sst2" or args.dataset == "qqp":
            args.lr = 1e-5
    else:
        raise ValueError("Invalid model")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fine_tune(args)
