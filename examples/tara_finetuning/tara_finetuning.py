#!/usr/bin/env python3
"""
Example script demonstrating fine-tuning with training-aware LoRA initialization (TaRA).

TaRA initializes the LoRA factors such that the gradients they induce at the start of training
closely approximate the gradient of the corresponding full-rank weight matrix. It uses one
calibration pass over a few hundred samples to collect activation and output-gradient second
moments per targeted layer, and initializes the adapter from the truncated SVD of the
covariance-weighted base weight. The base weight is replaced by the frozen residual W0 - BA,
so the model output is unchanged at initialization.

This example shows:
1. How to define a calibrate_step callback for covariance collection
2. How to apply initialize_tara_weights to a PEFT model
3. Training with the standard Hugging Face Trainer
4. Saving the fine-tuned model (merged, since the base weight carries the frozen residual)

Reference: "TaRA: Training-Aware Low-Rank Adaptation Initialization" (https://arxiv.org/abs/2609.02639)
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.tara import initialize_tara_weights


def parse_args():
    parser = argparse.ArgumentParser(description="TaRA-initialized LoRA fine-tuning example")

    # Model arguments
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./tara_output", help="Output directory")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    # LoRA configuration; lora_alpha = r (scaling 1) is the setting used by TaRA
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["c_attn"],
        help="Target modules for LoRA (e.g., c_attn for GPT-2)",
    )

    # Calibration arguments
    parser.add_argument(
        "--calibration_iters", type=int, default=16, help="Number of batches for covariance collection"
    )
    parser.add_argument("--calibration_batch_size", type=int, default=4, help="Batch size for calibration")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def prepare_dataset(dataset_name, dataset_config, tokenizer, max_length):
    """Load and prepare the dataset."""
    print(f"\nLoading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result

    print("Tokenizing dataset...")
    return dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["train"].column_names, desc="Tokenizing"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"\nLoading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    tokenized_datasets = prepare_dataset(args.dataset_name, args.dataset_config, tokenizer, args.max_length)

    # Create the PEFT model with the default initialization (lora_B = 0), which TaRA then replaces
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = peft_model.to(device)

    # ===== CALIBRATION PHASE =====
    print("\n" + "=" * 70)
    print("TARA CALIBRATION PHASE")
    print("=" * 70)
    print(f"Collecting activation/gradient covariances over {args.calibration_iters} batches...")

    calibration_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.calibration_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    def calibrate_step():
        """Run forward and backward passes for covariance collection."""
        peft_model.train()
        calibration_iter = iter(calibration_dataloader)
        for _ in range(args.calibration_iters):
            batch = next(calibration_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = peft_model(**batch).loss
            loss.backward()

    initialize_tara_weights(peft_model, calibrate_step)
    print("✓ TaRA initialization complete!")

    # ===== TRAINING PHASE =====
    print("\n" + "=" * 70)
    print("TRAINING PHASE")
    print("=" * 70)
    print("Starting training with TaRA-initialized adapters...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # ===== SAVING PHASE =====
    print("\n" + "=" * 70)
    print("SAVING PHASE")
    print("=" * 70)
    # TaRA replaces the base weight with the frozen residual W0 - BA, so the adapter alone does not
    # describe the fine-tuned model. Merge the adapter into the base and save the full model.
    print("Merging adapter into the base model and saving...")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✓ Training complete! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
