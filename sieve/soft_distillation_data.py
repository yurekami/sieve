"""
SIEVE: Soft Distillation Data Preparation

Prepare datasets for soft distillation training by capturing teacher model
probability distributions (top-k logits) for each token.

This module:
1. Takes an existing dataset with prompts
2. Queries a teacher model to get responses with logprobs
3. Extracts empirical token distributions (top-k)
4. Saves the prepared dataset for soft distillation training

No rejection sampling - we capture all teacher outputs. Filtering for
correctness should be done separately if needed.
"""

import argparse
import os
import re
import concurrent.futures
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
import litellm
from transformers import AutoTokenizer

from utils.vllm import start_vllm_server

# Import delimiters from synthetic_data_gen for consistency
from sieve.synthetic_data_gen import FEEDBACK_START_DELIMITER, FEEDBACK_END_DELIMITER


def replace_feedback_in_prompt(prompt: List[Dict], all_feedback: str) -> List[Dict]:
    """
    Replace the selected feedback in a prompt with all feedback.

    The prompt contains text between FEEDBACK_START_DELIMITER and FEEDBACK_END_DELIMITER
    that represents the selected portion of feedback. This function replaces that
    section with the full feedback content.

    Args:
        prompt: Chat format prompt [{"role": "user", "content": "..."}]
        all_feedback: The full NL feedback to substitute

    Returns:
        Modified prompt with all feedback substituted
    """
    modified_prompt = []
    for msg in prompt:
        if msg["role"] == "user":
            content = msg["content"]
            # Use regex to find and replace the feedback section
            # Pattern: FEEDBACK_START_DELIMITER ... FEEDBACK_END_DELIMITER
            pattern = (
                re.escape(FEEDBACK_START_DELIMITER)
                + r".*?"
                + re.escape(FEEDBACK_END_DELIMITER)
            )
            replacement = (
                f"{FEEDBACK_START_DELIMITER}\n{all_feedback}\n{FEEDBACK_END_DELIMITER}"
            )
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            modified_prompt.append({"role": msg["role"], "content": new_content})
        else:
            modified_prompt.append(msg)
    return modified_prompt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare soft distillation data from an existing dataset"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="Path to input dataset (parquet file with 'prompt' column)",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        required=True,
        help="Teacher model path (local, HF, or GCP format)",
    )
    parser.add_argument(
        "--student_tokenizer_path",
        type=str,
        required=True,
        help="Path to the student's tokenizer for pre-computing token IDs",
    )
    parser.add_argument(
        "--samples_per_input",
        type=int,
        default=1,
        help="Number of rollouts per problem",
    )
    parser.add_argument(
        "--k_tokens",
        type=int,
        default=100,
        help="Number of top-k tokens to sample for empirical distribution",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model length for VLLM",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for VLLM",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Data parallel size for VLLM (native vLLM data parallelism)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (default: disabled)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens for teacher response generation. If not set, generates until EOS or context limit (recommended for thinking models)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Number of parallel workers for generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the prepared dataset",
    )
    parser.add_argument(
        "--skip_server",
        action="store_true",
        help="Skip starting vLLM server (assume it's already running)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base URL for the teacher model",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="If specified, subsample the dataset to this size",
    )
    parser.add_argument(
        "--include_all_feedback",
        action="store_true",
        help="Replace selected feedback in prompts with all feedback from --all_feedback_file",
    )
    parser.add_argument(
        "--all_feedback_file",
        type=str,
        default=None,
        help="Path to file containing all NL feedback (required if --include_all_feedback is set)",
    )

    return parser.parse_args()


def query_teacher_with_logits(
    prompt: List[Dict],
    max_tokens: int = None,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    api_base: str = "http://127.0.0.1:8000/v1",
    top_logprobs: int = 100,
):
    """Query teacher model and get response with logprobs.

    Args:
        max_tokens: Maximum tokens to generate. If None, generates until EOS
                    or context limit (recommended for thinking models).
    """
    try:
        extra_body = {
            # Represent logprobs tokens as 'token_id:{id}' strings to avoid retokenization drift
            # See: https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#extra-parameters
            "return_tokens_as_token_ids": True,
        }
        if not enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        # Build kwargs, only include max_tokens if specified
        completion_kwargs = {
            "model": "openai/model",
            "messages": prompt,
            "temperature": temperature,
            "api_base": api_base,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "extra_body": extra_body,
        }
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens

        response = litellm.completion(**completion_kwargs)

        content = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs
        return content, logprobs_data
    except Exception as e:
        print(f"Error querying teacher model: {e}")
        return "", None


def extract_empirical_distribution(
    logprobs_data,
    k: int = 100,
    tokenizer=None,
    unk_token_id: int = None,
):
    """Extract empirical distribution from logprobs data and compute token IDs."""
    if not logprobs_data or not logprobs_data.content:
        return None

    token_distributions = []
    for token_logprob in logprobs_data.content:
        if token_logprob.top_logprobs:
            tokens = []
            token_ids = []
            probs = []

            for top_logprob in token_logprob.top_logprobs[:k]:
                token_str = top_logprob.token
                tokens.append(token_str)
                probs.append(np.exp(top_logprob.logprob))

                if tokenizer is not None:
                    # Check if token is in 'token_id:{id}' format (from return_tokens_as_token_ids)
                    if token_str.startswith("token_id:"):
                        try:
                            token_id = int(token_str.split(":")[1])
                            token_ids.append(token_id)
                            continue
                        except (ValueError, IndexError):
                            pass

                    # Fallback: use convert_tokens_to_ids for direct vocab lookup
                    token_id = tokenizer.convert_tokens_to_ids(token_str)
                    if token_id != tokenizer.unk_token_id:
                        token_ids.append(token_id)
                    else:
                        # Fallback to encode for tokens not in vocab directly
                        encoded = tokenizer.encode(token_str, add_special_tokens=False)
                        if len(encoded) == 1:
                            token_ids.append(encoded[0])
                        else:
                            # Multi-token or empty encoding - use unk
                            token_ids.append(unk_token_id)

            # Renormalize top-k probabilities to sum to 1
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]

            token_distributions.append(
                {
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "probabilities": probs,
                    "original_mass": prob_sum,
                }
            )

    return token_distributions


def process_teacher_response(args):
    """Process teacher response (CPU-bound post-processing)."""
    (
        example,
        teacher_response,
        logprobs_data,
        k_tokens,
        tokenizer,
        unk_token_id,
    ) = args

    try:
        if logprobs_data is None:
            return None

        # Extract empirical distribution with tokenizer.encode()
        empirical_dist = extract_empirical_distribution(
            logprobs_data,
            k=k_tokens,
            tokenizer=tokenizer,
            unk_token_id=unk_token_id,
        )

        if empirical_dist is None:
            return None

        # Build the soft distillation example
        soft_example = {
            "prompt": example["prompt"],
            "teacher_response": teacher_response,
            "teacher_distribution": empirical_dist,
        }

        # Copy over any additional metadata from the original example
        for key in example:
            if key not in soft_example:
                soft_example[key] = example[key]

        return soft_example

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def process_single_example_optimized(args):
    """Process a single example with tokenizer.encode() for token IDs."""
    (
        example,
        k_tokens,
        enable_thinking,
        temperature,
        max_tokens,
        tokenizer,
        unk_token_id,
        api_base,
    ) = args

    try:
        prompt = example["prompt"]

        # Get teacher response and logits
        teacher_response, logprobs_data = query_teacher_with_logits(
            prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
            api_base=api_base,
            top_logprobs=k_tokens,
        )

        if logprobs_data is None:
            return None

        # Extract empirical distribution with tokenizer.encode()
        empirical_dist = extract_empirical_distribution(
            logprobs_data,
            k=k_tokens,
            tokenizer=tokenizer,
            unk_token_id=unk_token_id,
        )

        if empirical_dist is None:
            return None

        # Build the soft distillation example
        soft_example = {
            "prompt": prompt,
            "teacher_response": teacher_response,
            "teacher_distribution": empirical_dist,
        }

        # Copy over any additional metadata from the original example
        for key in example:
            if key not in soft_example:
                soft_example[key] = example[key]

        return soft_example

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def prepare_soft_distillation_dataset(
    dataset: List[Dict],
    k_tokens: int = 100,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    max_tokens: int = None,
    max_workers: int = 12,
    tokenizer=None,
    api_base: str = "http://127.0.0.1:8000/v1",
) -> List[Dict]:
    """Prepare dataset with teacher's soft labels using parallel processing with streaming pipeline."""
    print(
        f"Preparing soft labels for {len(dataset)} examples with {max_workers} workers"
    )
    print(
        f"Max tokens per response: {max_tokens if max_tokens else 'unlimited (until EOS)'}"
    )

    unk_token_id = tokenizer.unk_token_id

    process_args = [
        (
            ex,
            k_tokens,
            enable_thinking,
            temperature,
            max_tokens,
            tokenizer,
            unk_token_id,
            api_base,
        )
        for ex in dataset
    ]

    examples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_example_optimized, args): idx
            for idx, args in enumerate(process_args)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(process_args),
            desc="Generating soft labels",
        ):
            result = future.result()
            if result is not None:
                examples.append(result)

    print(f"Generated {len(examples)}/{len(dataset)} examples")
    return examples


def main():
    args = parse_arguments()

    server_process = None
    temp_dir = None

    try:
        # Handle input dataset(s) - supports comma-separated list of files
        input_datasets = [p.strip() for p in args.input_dataset.split(",")]
        print(f"Processing {len(input_datasets)} input dataset(s)")

        all_records = []
        for input_ds in input_datasets:
            input_dataset_path = input_ds

            # Load input dataset
            print(f"Loading input dataset from {input_dataset_path}")
            records = pd.read_parquet(input_dataset_path).to_dict("records")
            print(f"Loaded {len(records)} examples from {input_dataset_path}")
            all_records.extend(records)

        dataset = all_records
        print(f"Total loaded: {len(dataset)} examples")

        # Subsample if requested
        if args.subset_size is not None and args.subset_size < len(dataset):
            import random

            random.seed(42)  # For reproducibility
            dataset = random.sample(dataset, args.subset_size)
            print(f"Subsampled to {len(dataset)} examples")

        # Handle --include_all_feedback: replace selected feedback with all feedback
        if args.include_all_feedback:
            if not args.all_feedback_file:
                raise ValueError(
                    "--all_feedback_file is required when --include_all_feedback is set"
                )

            print(f"Loading all feedback from: {args.all_feedback_file}")
            with open(args.all_feedback_file, "r") as f:
                all_feedback = f.read()

            print(
                f"Replacing selected feedback with all feedback in {len(dataset)} examples..."
            )
            for example in dataset:
                example["prompt"] = replace_feedback_in_prompt(
                    example["prompt"], all_feedback
                )
            print("Feedback replacement complete")

        # Handle teacher model setup
        if not args.skip_server:
            teacher_model_path = args.teacher_model

            # Start vLLM server with optional data parallelism
            print(f"Starting vLLM server for teacher model: {teacher_model_path}")
            if args.data_parallel_size > 1:
                print(
                    f"Using vLLM native data parallelism: DP={args.data_parallel_size}, TP={args.tensor_parallel_size}"
                )

            server_process = start_vllm_server(
                model_to_serve_name=teacher_model_path,
                served_model_name="model",
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                data_parallel_size=args.data_parallel_size,
                max_logprobs=100,
            )

        # Load student tokenizer
        print(f"Loading student tokenizer from: {args.student_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer_path)

        # Duplicate examples for multiple rollouts
        if args.samples_per_input > 1:
            print(f"Creating {args.samples_per_input} rollouts per example")
            dataset = dataset * args.samples_per_input

        # Prepare soft distillation dataset
        examples = prepare_soft_distillation_dataset(
            dataset,
            k_tokens=args.k_tokens,
            enable_thinking=args.enable_thinking,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            tokenizer=tokenizer,
            api_base=args.api_base,
        )

        # Save dataset
        pd.DataFrame(examples).to_parquet(args.output_path)
        print(f"\nSaved {len(examples)} examples to {args.output_path}")

        # Preview
        if examples:
            print("\n" + "=" * 60)
            print("EXAMPLE PREVIEW")
            print("=" * 60)
            ex = examples[0]
            print(f"Prompt: {ex['prompt']}")
            print(f"Teacher response: {ex['teacher_response']}")
            print(f"Distribution length: {len(ex['teacher_distribution'])}")

    finally:
        if server_process:
            print("Terminating VLLM server...")
            server_process.terminate()
            server_process.wait()

        if temp_dir and os.path.exists(temp_dir):
            import shutil

            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
