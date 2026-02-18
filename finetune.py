"""
Fine-tune GPT-2 on nanomaterials research text using LoRA.

LoRA (Low-Rank Adaptation) freezes the original 117M GPT-2 parameters and
inserts small trainable matrices (~0.3M params) into the attention layers.
This keeps VRAM usage well under 4GB, fitting comfortably on an RTX A2000.

Usage:
    # 1. Prepare data first (or use built-in samples):
    python prepare_dataset.py --output ./tokenized_data

    # 2. Fine-tune:
    python finetune.py --dataset ./tokenized_data --epochs 5

    # 3. Generate:
    python generate.py --prompt "Graphene oxide nanoparticles"
"""

import argparse
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def print_trainable_summary(model):
    """Print how many parameters are trainable vs frozen."""
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100.0 * trainable / total
    print(f"\n  Total parameters:     {total:>12,}")
    print(f"  Trainable (LoRA):     {trainable:>12,}  ({pct:.2f}%)")
    print(f"  Frozen (GPT-2 base):  {total - trainable:>12,}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on nanomaterials text with LoRA."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./tokenized_data",
        help="Path to tokenized dataset from prepare_dataset.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Base model (default: gpt2 = GPT-2 Small 117M).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./nanomaterials-gpt2-lora",
        help="Where to save the fine-tuned LoRA adapter.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2, safe for 4GB VRAM).",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * this).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4, typical for LoRA).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank — higher = more capacity, more VRAM (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32).",
    )
    args = parser.parse_args()

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")

    # --- Load tokenizer and model ---
    print(f"\nLoading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # Target the attention projection matrices in GPT-2
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_summary(model)

    # --- Load dataset ---
    print(f"\nLoading dataset from: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    print(f"  Training examples: {len(dataset)}")

    # --- Data collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # --- Training arguments ---
    # Tuned for Dell 5760 / RTX A2000 (4GB VRAM)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "cuda"),
        bf16=False,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
        seed=42,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- Train ---
    print(f"\nStarting training:")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Grad accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch:  {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  LoRA rank:        {args.lora_rank}")
    print()

    trainer.train()

    # --- Save ---
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to: {args.output_dir}")
    print("Use generate.py to test the fine-tuned model.")


if __name__ == "__main__":
    main()
