"""
Generate nanomaterials text using the fine-tuned GPT-2 LoRA model.

Usage:
    # Basic generation:
    python generate.py --prompt "Graphene oxide nanosheets were"

    # With custom parameters:
    python generate.py \
        --prompt "Silver nanoparticles synthesized via" \
        --max-tokens 200 \
        --temperature 0.8 \
        --top-p 0.9

    # Compare base vs fine-tuned:
    python generate.py --prompt "The nanoparticles exhibited" --compare
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_finetuned_model(base_model_name: str, adapter_path: str, device: str):
    """Load the base GPT-2 model with the LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # merge LoRA weights for faster inference
    model.to(device)
    model.eval()
    return model, tokenizer


def load_base_model(model_name: str, device: str):
    """Load the vanilla base GPT-2 model (no fine-tuning)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    device: str = "cpu",
) -> str:
    """Generate text continuation from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate nanomaterials text with fine-tuned GPT-2."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The synthesized nanoparticles exhibited",
        help="Text prompt to continue from.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./nanomaterials-gpt2-lora",
        help="Path to the saved LoRA adapter.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gpt2",
        help="Base model name (default: gpt2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum new tokens to generate (default: 150).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7). Lower = more focused.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling cutoff (default: 0.9).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show output from both base GPT-2 and fine-tuned model.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Generate with fine-tuned model ---
    print(f"Prompt: \"{args.prompt}\"\n")
    print("=" * 70)

    print("\n[Fine-tuned model]")
    ft_model, ft_tokenizer = load_finetuned_model(
        args.base_model, args.adapter_path, device
    )
    ft_output = generate_text(
        ft_model, ft_tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(f"{args.prompt}{ft_output}")

    # --- Optionally compare with base model ---
    if args.compare:
        # Free fine-tuned model memory
        del ft_model
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"\n{'=' * 70}")
        print("\n[Base GPT-2 (no fine-tuning)]")
        base_model, base_tokenizer = load_base_model(args.base_model, device)
        base_output = generate_text(
            base_model, base_tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(f"{args.prompt}{base_output}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
