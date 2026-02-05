"""
Merge LoRA adapter weights into base model.

This script runs outside of DeepSpeed context to avoid device_map conflicts.
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_adapter(
    base_model_name: str,
    adapter_dir: str,
    output_dir: str,
    use_gpu: bool = True,
):
    """
    Merge LoRA adapter into base model and save the result.

    Args:
        base_model_name: HuggingFace model name or path to base model
        adapter_dir: Directory containing the LoRA adapter
        output_dir: Directory to save the merged model
        use_gpu: Whether to try GPU merge first (faster)
    """
    print(f"Merging LoRA adapter from {adapter_dir}")
    print(f"Base model: {base_model_name}")
    print(f"Output directory: {output_dir}")

    if use_gpu and torch.cuda.is_available():
        try:
            print("Attempting GPU merge for speed...")
            torch.cuda.empty_cache()

            # Load base model to GPU
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )

            # Load and merge adapter
            print("Loading adapter...")
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

            print("Merging weights...")
            merged_model = peft_model.merge_and_unload()

            # Move to CPU before saving
            print("Moving to CPU for saving...")
            merged_model = merged_model.cpu()
            torch.cuda.empty_cache()

            print("GPU merge successful!")

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"GPU merge failed: {e}")
            print("Falling back to CPU merge...")
            torch.cuda.empty_cache()
            use_gpu = False

    if not use_gpu or not torch.cuda.is_available():
        print("Using CPU merge...")

        # Load base model to CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Load and merge adapter
        print("Loading adapter...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

        print("Merging weights...")
        merged_model = peft_model.merge_and_unload()

        print("CPU merge completed.")

    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Merge complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HuggingFace model name or path to base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing the LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU merge (skip GPU attempt)",
    )

    args = parser.parse_args()

    merge_lora_adapter(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu_only,
    )
