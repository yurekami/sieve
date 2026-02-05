"""
SIEVE: Soft Distillation Trainer

Train models using soft distillation with top-k token distributions from a teacher.

Features:
- KL divergence loss between student and teacher distributions
- Configurable LoRA support for parameter-efficient training
- Full fine-tuning option
- Feedback removal from prompts during training (for internalization)
"""

import re
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from sieve.synthetic_data_gen import FEEDBACK_START_DELIMITER, FEEDBACK_END_DELIMITER
import wandb


def read_soft_distill_parquet(path: str) -> pd.DataFrame:
    """Read parquet files with deeply nested structures safely.

    PyArrow 22+ has issues with nested data when chunked. This function
    uses batch reading to work around the limitation.
    """
    # Fall back to batch reading for deeply nested structures
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    batches = []
    for batch in pf.iter_batches(batch_size=500):
        batches.append(batch.to_pandas())
    return pd.concat(batches, ignore_index=True)


class SoftDistillationDataCollator:
    """Custom data collator for soft distillation training."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 32768,
        remove_feedback: bool = False,
        enable_thinking: bool = False,
        topk: int = 100,
    ):
        """
        Args:
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            remove_feedback: If True, removes the selected_feedback from each prompt.
                            This is used for internalization - the model learns to
                            apply feedback without seeing it in the prompt.
            enable_thinking: If True, enable thinking mode for Qwen3 models (default: False)
            topk: Number of top tokens to use for distillation (default: 100)

        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.remove_feedback = remove_feedback
        self.enable_thinking = enable_thinking
        self.topk = topk

        # Detect if model supports chat_template_kwargs (Qwen3 only)
        self.supports_chat_template_kwargs = "Qwen3" in tokenizer.name_or_path

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Process prompts and tokenize
        prompts = [f["prompt"] for f in features]

        # Remove feedback section from prompts if configured (for internalization)
        if self.remove_feedback:
            # Regex pattern to match feedback section including delimiters and surrounding whitespace
            pattern = re.compile(
                rf"\s*{re.escape(FEEDBACK_START_DELIMITER)}.*?{re.escape(FEEDBACK_END_DELIMITER)}\s*",
                re.DOTALL,
            )
            for i in range(len(prompts)):
                if isinstance(prompts[i], list) and len(prompts[i]) > 0:
                    content = prompts[i][0]["content"]
                    new_content = pattern.sub("\n\n", content)
                    # Clean up any excessive newlines
                    while "\n\n\n" in new_content:
                        new_content = new_content.replace("\n\n\n", "\n\n")
                    prompts[i][0]["content"] = new_content.strip()

        teacher_responses = [f["teacher_response"] for f in features]
        teacher_distributions = [f["teacher_distribution"] for f in features]

        # Create full conversations for tokenization
        conversations = []
        response_start_positions = []

        # Detect if teacher responses have thinking content
        # When enable_thinking=True during data generation, responses start with <think>
        # This affects how we calculate template overhead for Qwen3 models
        responses_have_thinking = any(
            r.strip().startswith("<think>") for r in teacher_responses if r
        )

        # Calculate chat template overhead once (tokens added for assistant role marker)
        # These are tokens like <|im_start|>assistant\n that aren't in VLLM's logprobs
        template_overhead = self._calculate_template_overhead(
            content_has_thinking=responses_have_thinking
        )

        for prompt, response in zip(prompts, teacher_responses):
            # Handle both string and list prompts
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            # Tokenize prompt only to find where response starts
            kwargs = {}
            if self.supports_chat_template_kwargs:
                kwargs["chat_template_kwargs"] = {
                    "enable_thinking": self.enable_thinking
                }

            prompt_only = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
            # Add template overhead to get where actual content starts
            response_start_positions.append(len(prompt_only) + template_overhead)

            # Convert prompt format to conversation
            conversation = self.tokenizer.apply_chat_template(
                prompt + [{"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
                **kwargs,
            )
            conversations.append(conversation)

        # Tokenize conversations
        # Use add_special_tokens=False because apply_chat_template already adds BOS
        # Without this, Llama tokenizer adds an extra BOS causing 1-token offset
        tokenized = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        batch["input_ids"] = tokenized["input_ids"]
        batch["attention_mask"] = tokenized["attention_mask"]

        # Create labels: mask prompt tokens with -100 to exclude from loss
        labels = tokenized["input_ids"].clone()
        for i, response_start in enumerate(response_start_positions):
            labels[i, :response_start] = -100
        labels[tokenized["attention_mask"] == 0] = -100
        batch["labels"] = labels

        # Process teacher distributions into tensor format
        teacher_token_ids, teacher_probs = self._process_teacher_distributions_tensor(
            teacher_distributions,
            tokenized["input_ids"].shape,
            response_start_positions,
            topk=self.topk,
        )
        batch["teacher_token_ids"] = teacher_token_ids
        batch["teacher_probs"] = teacher_probs

        return batch

    def _calculate_template_overhead(self, content_has_thinking: bool = False):
        """Calculate how many tokens the chat template adds for assistant role marker.

        Args:
            content_has_thinking: If True, the content already starts with <think>.
                                  This affects Qwen3 templates where content without
                                  thinking gets wrapped in an empty <think></think> block,
                                  but content with existing thinking is used as-is.
        """
        test_prompt = [{"role": "user", "content": "test"}]

        kwargs = {}
        if self.supports_chat_template_kwargs:
            kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        with_gen_prompt = self.tokenizer.apply_chat_template(
            test_prompt,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )

        # Use appropriate test content based on whether actual responses have thinking
        # When enable_thinking=True and content already has <think>, template uses as-is
        # When enable_thinking=True and content has NO thinking, template adds empty wrapper
        if content_has_thinking:
            # Content starts with <think>, search for that token
            test_content = "<think>\ntest reasoning\n</think>\n\nX"
            search_content = "<think>"
        else:
            test_content = "X"
            search_content = "X"

        full_with_content = self.tokenizer.apply_chat_template(
            test_prompt + [{"role": "assistant", "content": test_content}],
            tokenize=True,
            add_generation_prompt=False,
            **kwargs,
        )

        search_tokens = self.tokenizer.encode(search_content, add_special_tokens=False)

        # Find where the actual content starts after template tokens
        overhead = None
        for i in range(len(with_gen_prompt), len(full_with_content)):
            if full_with_content[i : i + len(search_tokens)] == search_tokens:
                overhead = i - len(with_gen_prompt)
                break

        if overhead is None:
            print(
                f"ERROR: Could not detect template overhead for {self.tokenizer.name_or_path}"
            )
            print(f"  content_has_thinking: {content_has_thinking}")
            print(f"  with_gen_prompt length: {len(with_gen_prompt)}")
            print(f"  full_with_content length: {len(full_with_content)}")
            print(f"  with_gen_prompt tokens: {with_gen_prompt}")
            print(f"  full_with_content tokens: {full_with_content}")
            print(f"  search_tokens: {search_tokens}")
            print(
                f"  Decoded with_gen_prompt: {self.tokenizer.decode(with_gen_prompt)}"
            )
            print(
                f"  Decoded full_with_content: {self.tokenizer.decode(full_with_content)}"
            )
            raise ValueError(
                "Could not detect template overhead - this will cause misalignment"
            )

        return overhead

    def _process_teacher_distributions_tensor(
        self, teacher_distributions, shape, response_start_positions, topk=100
    ):
        """Process teacher distributions into tensor format for GPU efficiency."""
        batch_size, seq_len = shape

        teacher_token_ids = torch.full(
            (batch_size, seq_len, topk), fill_value=-1, dtype=torch.long
        )
        teacher_probs = torch.zeros((batch_size, seq_len, topk), dtype=torch.float32)

        for batch_idx, dist_list in enumerate(teacher_distributions):
            response_start = response_start_positions[batch_idx]

            for teacher_pos in range(len(dist_list)):
                sequence_pos = response_start + teacher_pos

                if sequence_pos >= seq_len:
                    break

                dist_data = dist_list[teacher_pos]
                if dist_data is None:
                    continue

                if "token_ids" in dist_data and "probabilities" in dist_data:
                    token_ids_raw = dist_data["token_ids"]
                    probs_raw = dist_data["probabilities"]

                    token_ids_list = []
                    probs_list = []
                    for tid, prob in zip(token_ids_raw, probs_raw):
                        if tid is not None and prob is not None:
                            token_ids_list.append(tid)
                            probs_list.append(prob)

                    num_valid = min(len(token_ids_list), topk)
                    if num_valid > 0:
                        teacher_token_ids[batch_idx, sequence_pos, :num_valid] = (
                            torch.tensor(token_ids_list[:num_valid], dtype=torch.long)
                        )
                        teacher_probs[batch_idx, sequence_pos, :num_valid] = (
                            torch.tensor(probs_list[:num_valid], dtype=torch.float32)
                        )

        return teacher_token_ids, teacher_probs


class SoftDistillationTrainer(Trainer):
    """Custom trainer for soft distillation with KL divergence loss."""

    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute KL divergence loss between student and teacher distributions."""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        # Compute KL divergence loss (forward cross-entropy)
        kl_loss = self._compute_kl_loss_vectorized(
            outputs.logits,
            inputs["teacher_token_ids"],
            inputs["teacher_probs"],
            inputs["attention_mask"],
            inputs["labels"],
        )

        # Store for logging
        self.current_lm_loss = outputs.loss.detach()
        self.current_kl_loss = kl_loss.detach()

        return (kl_loss, outputs) if return_outputs else kl_loss

    def _compute_kl_loss_vectorized(
        self, student_logits, teacher_token_ids, teacher_probs, attention_mask, labels
    ):
        """Compute forward cross-entropy between student and teacher distributions."""
        teacher_token_ids = teacher_token_ids.to(student_logits.device)
        teacher_probs = teacher_probs.to(student_logits.device)
        labels = labels.to(student_logits.device)
        attention_mask = attention_mask.to(student_logits.device)

        # Shift for autoregressive alignment
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_token_ids = teacher_token_ids[:, 1:, :].contiguous()
        shift_teacher_probs = teacher_probs[:, 1:, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attention_mask = attention_mask[:, 1:].contiguous()

        shift_logits = shift_logits / self.temperature
        student_log_probs = F.log_softmax(shift_logits, dim=-1)

        gathered_log_probs = torch.gather(
            student_log_probs, 2, shift_teacher_token_ids.clamp(min=0)
        )

        valid_token_mask = (shift_teacher_token_ids >= 0) & (shift_teacher_probs > 0)

        loss_per_token = -shift_teacher_probs * gathered_log_probs
        masked_loss = torch.where(
            valid_token_mask, loss_per_token, torch.zeros_like(loss_per_token)
        )

        loss_per_position = masked_loss.sum(dim=-1)

        valid_position_mask = (shift_labels != -100) & (shift_attention_mask == 1)

        kl_loss = (
            loss_per_position * valid_position_mask
        ).sum() / valid_position_mask.sum().clamp(min=1)

        return kl_loss


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: List[str] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """Create a LoRA configuration."""
    if target_modules is None:
        # Default target modules for most transformer architectures
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def load_model_for_training(
    model_path: str,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """
    Load a model for training with optional LoRA and quantization.

    Args:
        model_path: Path to the model
        use_lora: Whether to use LoRA
        lora_config: LoRA configuration (created if None and use_lora=True)
        load_in_4bit: Load model in 4-bit quantization
        load_in_8bit: Load model in 8-bit quantization
    """
    # Setup quantization config if needed
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # Apply LoRA if requested
    if use_lora:
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        if lora_config is None:
            lora_config = create_lora_config()

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="SIEVE: Soft distillation training")
    # Required arguments
    parser.add_argument("--run_identifier", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_to_train", type=str, required=True)

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Distillation parameters
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for softmax scaling"
    )

    # LoRA parameters
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated list of target modules for LoRA",
    )

    # Quantization
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (requires LoRA)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization (requires LoRA)",
    )

    # Feedback removal for internalization
    parser.add_argument(
        "--remove_feedback",
        action="store_true",
        help="Remove selected_feedback from prompts during training for internalization",
    )

    # Other options
    parser.add_argument(
        "--train_size", type=int, default=None, help="Limit training examples"
    )
    parser.add_argument("--wandb_project", type=str, default="sieve-soft-distillation")

    # Thinking mode control
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (default: disabled)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top tokens to use for distillation",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.run_identifier,
            config=vars(args),
        )

    # Load tokenizer
    if local_rank == 0:
        print(f"Loading tokenizer: {args.model_to_train}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_to_train)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup LoRA config if needed
    lora_config = None
    if args.use_lora:
        target_modules = None
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

        lora_config = create_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
        )

    # Load model
    if local_rank == 0:
        print(f"Loading model: {args.model_to_train}")
        if args.use_lora:
            print(f"Using LoRA with r={args.lora_r}, alpha={args.lora_alpha}")

    model = load_model_for_training(
        args.model_to_train,
        use_lora=args.use_lora,
        lora_config=lora_config,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Load dataset - support comma-separated paths for concatenation
    # Use custom reader to handle PyArrow issues with deeply nested structures
    # IMPORTANT: Only rank 0 loads to avoid CPU OOM from 8x memory usage
    from datasets import Dataset

    data_paths = [path.strip() for path in args.data_path.split(",")]

    # Use a fixed shared path so all ranks can find the dataset
    temp_arrow_path = "/tmp/soft_distill_train_shared.arrow"
    ready_flag_path = "/tmp/soft_distill_train_ready.flag"

    # Clean up any stale flag from previous runs (only rank 0)
    if local_rank == 0:
        import shutil

        if os.path.exists(ready_flag_path):
            os.remove(ready_flag_path)
        if os.path.exists(temp_arrow_path):
            shutil.rmtree(temp_arrow_path)

        print(f"Loading dataset from: {args.data_path}")
        # Load all files using our safe reader and concatenate
        all_dfs = []
        for path in data_paths:
            print(f"  Loading {path}...")
            df = read_soft_distill_parquet(path)
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        if len(data_paths) > 1:
            print(f"Concatenated into {len(combined_df)} total examples")

        # Subsample before converting to dataset to save memory
        if args.train_size is not None and args.train_size < len(combined_df):
            import random

            random.seed(42)
            indices = random.sample(range(len(combined_df)), args.train_size)
            combined_df = combined_df.iloc[indices].reset_index(drop=True)
            print(f"Subsampled to {len(combined_df)} examples")

        # Convert to HuggingFace dataset and save to shared path
        dataset = Dataset.from_pandas(combined_df)
        dataset.save_to_disk(temp_arrow_path)
        print(f"Saved dataset to {temp_arrow_path}")

        # Free the dataframe memory
        del combined_df
        del all_dfs
        import gc

        gc.collect()

        # Signal that dataset is ready
        with open(ready_flag_path, "w") as f:
            f.write("ready")
        print("Dataset ready flag set")

    else:
        # Non-rank-0 processes wait for the ready flag
        import time

        print(f"Rank {local_rank} waiting for dataset to be ready...")
        max_wait = 1800  # 30 minutes max wait
        wait_time = 0
        while not os.path.exists(ready_flag_path):
            time.sleep(1)
            wait_time += 1
            if wait_time >= max_wait:
                raise TimeoutError(f"Rank {local_rank} timed out waiting for dataset")
        print(f"Rank {local_rank} found ready flag after {wait_time}s")

    # All ranks load from the arrow file (memory-mapped, much more efficient)
    dataset = Dataset.load_from_disk(temp_arrow_path)

    if local_rank == 0:
        print(f"Training on {len(dataset)} examples")

    if local_rank == 0:
        print(f"Thinking mode: {'enabled' if args.enable_thinking else 'disabled'}")

    # Create data collator with optional feedback removal for internalization
    data_collator = SoftDistillationDataCollator(
        tokenizer,
        max_length=args.max_length,
        remove_feedback=args.remove_feedback,
        enable_thinking=args.enable_thinking,
        topk=args.topk,
    )

    # Training arguments
    # Disable gradient checkpointing when using LoRA (incompatible with DeepSpeed ZeRO-3)
    use_gradient_checkpointing = not args.use_lora

    training_args = TrainingArguments(
        output_dir=args.run_identifier,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_only_model=True,
        bf16=True,
        dataloader_drop_last=True,
        report_to="wandb" if local_rank == 0 else "none",
        run_name=args.run_identifier,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if use_gradient_checkpointing
        else {},
    )

    # Create trainer
    trainer = SoftDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        temperature=args.temperature,
    )

    # Train
    if local_rank == 0:
        print("Starting training...")
    trainer.train()

    # Save final model
    if args.use_lora:
        # For LoRA, just save the adapter (merge will be done separately outside DeepSpeed)
        if local_rank == 0:
            print(f"Saving LoRA adapter to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        # For full fine-tuning, save the full model
        trainer.save_model()

    if local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
