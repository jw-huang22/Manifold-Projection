import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from pdb import set_trace as st
from utils.methods_gpt2_xl import *
from peft import get_peft_model, LoraConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on a GLUE task.')
    parser.add_argument("--dataset", default="sst2", type=str, help="Any dataset in GLUE")
    parser.add_argument("--model", default="gpt2-xl", type=str, help="Model you want to fine-tune")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training") 
    parser.add_argument("--max_length", default=512, type=int, help="Max sequence length should be <=1024 for GPT-2")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate for training")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("--output_dir", default="results", type=str, help="Directory to save fine-tuned model")
    parser.add_argument("--gpus", default="0,1", type=str, help="GPU to use")
    parser.add_argument("--tsqp", default="false", type=str, help="Whether to use TSQP")
    parser.add_argument("--full_finetune", default="true", type=str, help="Whether to fine-tune all parameters")
    return parser.parse_args()

def str2bool(value):
    return str(value).lower() in ("true", "1", "yes", "y")

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_ratio = 100 * trainable_params / all_params
    print(
        f"Trainable params: {trainable_params} || "
        f"All params: {all_params} || "
        f"Trainable ratio: {trainable_ratio:.4f}%"
    )

def fine_tune(args):
    """Fine-tune the GPT-2 model for sequence classification."""
    set_seed()
    if args.model == "gpt2-xl":
        model_name = "gpt2_xl"
    else:
        raise ValueError("Invalid model name")
    if args.dataset != "sst2":
        raise ValueError("Invalid dataset")
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
    
    # Add a padding token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    dataset = load_dataset("glue", actual_task)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=args.max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Load GPT-2 model for sequence classification
    print("Loading GPT-2 model for sequence classification...")
    model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    if str2bool(args.full_finetune):
        print("Fine-tuning mode: full-parameter fine-tuning")
    else:
        print("Fine-tuning mode: LoRA fine-tuning")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_fc", "c_attn", "c_proj"],
        )
        peft_model = get_peft_model(model, lora_config)
        model = peft_model.model

    model.config.pad_token_id = tokenizer.pad_token_id
    print_trainable_parameters(model)

    
    # Load evaluation metric
    print("Loading metric...")
    metric = evaluate.load('glue', actual_task)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.dataset}",
        eval_strategy='epoch',
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=100,
        dataloader_num_workers=8,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if actual_task != "stsb":
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    print("Initializing Trainer...")
    if args.tsqp == "true":
        pre_model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        pre_model.config.pad_token_id = tokenizer.pad_token_id
        trainer = CustomTrainer(
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

    print("Start fine-tuning...")
    trainer.train()

    print("Evaluate the fine-tuned model...")
    fine_tuned_model_perf = trainer.evaluate()
    print(fine_tuned_model_perf)

    print(f"Fine-tuned model performance on {args.dataset}: {fine_tuned_model_perf}")

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "sst2":
        args.lr = 2e-5
    else:
        raise RuntimeError("Invalid dataset")
    # Set visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fine_tune(args)
