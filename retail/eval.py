import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import litellm
from dotenv import load_dotenv

from utils.vllm import start_vllm_server
from utils.reward_utils import extract_boxed_answer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate models on retail discount task"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to local model directory or HF model name",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Whether the model is a local model or from HF",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192, help="Maximum model length for VLLM"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for VLLM",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=100,
        help="Maximum number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="data/retail_discount_eval_256_all_rules.parquet",
        help="Path to evaluation dataset. Use variant-specific files: *_all_rules.parquet, *_only_applicable_rules.parquet, or *_no_rules.parquet",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default=None,
        help="If specified, evaluate an API model (e.g., gpt-4o). Skips VLLM.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Base URL for the API. Only needed for custom/local models.",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Path to save detailed evaluation results as JSON",
    )
    parser.add_argument(
        "--no_think", action="store_true", help="Turn off thinking mode for Qwen3"
    )

    return parser.parse_args()


def query_model(
    prompt,
    api_model: str = None,
    api_base_url: str = None,
    no_think: bool = False,
    original_model_name: str = None,
) -> str:
    """
    Query a model with a prompt and return the generated text using litellm.
    Can query either a VLLM server or any API supported by litellm.
    """
    model = api_model
    api_base = api_base_url
    temperature = 0  # Default temperature
    top_p = 0.95  # Default values for Qwen3 parameters
    top_k = 20
    min_p = 0

    # Use original_model_name for model-specific logic detection
    model_for_logic = original_model_name if original_model_name else model

    # Set model-specific parameters
    if "gpt-5" in (model or ""):
        temperature = 1.0
    elif "Qwen3" in model_for_logic:
        temperature = 0.6

    # If no api_model is specified, we default to the local VLLM server.
    if not model:
        model = (
            "openai/model"  # litellm convention for custom OpenAI-compatible servers
        )
        api_base = "http://127.0.0.1:8000/v1"

    try:
        print(
            f"[REQUEST] Model: {model}, Original: {original_model_name}, Messages: {prompt}"
        )
        # litellm ignores api_base=None if it's not needed for the specified model
        # For Qwen3, use extra_body to pass chat_template_kwargs
        extra_body = {}
        if no_think:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        response = litellm.completion(
            model=model,
            messages=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            api_base=api_base,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        print(f"[RESPONSE] Content: {content}")
        return content
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""


def process_item(
    item,
    api_model: str = None,
    api_base_url: str = None,
    no_think: bool = False,
    original_model_name: str = None,
):
    """Process a single evaluation item."""
    prompt = item["prompt"]
    ground_truth = item["reward_model"]["ground_truth"]

    try:
        # Generate the completion from the model
        completion = query_model(
            prompt,
            api_model=api_model,
            api_base_url=api_base_url,
            no_think=no_think,
            original_model_name=original_model_name,
        )

        # Extract the boxed answer
        extracted_answer = extract_boxed_answer(completion)

        # Check if the extracted answer matches the ground truth final price
        is_correct = False
        if extracted_answer is not None:
            try:
                # Try to parse the extracted answer as a float (price)
                # Remove dollar signs and other currency symbols
                cleaned_answer = (
                    extracted_answer.replace("$", "")
                    .replace("\\", "")
                    .replace(",", "")
                    .strip()
                )
                extracted_price = float(cleaned_answer)
                ground_truth_price = float(ground_truth)

                # Check if prices match (allow small floating point differences)
                is_correct = abs(extracted_price - ground_truth_price) <= 0.02
            except (ValueError, TypeError):
                # If we can't parse as a number, it's incorrect
                is_correct = False

        print(f"""
Prompt: {prompt[0]["content"]}
Ground Truth: {ground_truth}
Extracted Answer: {extracted_answer}
Correct: {is_correct}
""")

        return {
            "correct": is_correct,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "completion": completion,
        }

    except Exception as e:
        print(f"Error processing item: {e}")
        return {
            "correct": False,
            "extracted_answer": None,
            "ground_truth": ground_truth,
            "completion": "",
            "error": str(e),
        }


def evaluate_model(
    dataset,
    max_workers=12,
    api_model: str = None,
    api_base_url: str = None,
    no_think: bool = False,
    original_model_name: str = None,
):
    """
    Evaluate the model on the retail discount task in parallel.

    Args:
        dataset: The dataset to evaluate on
        max_workers: Maximum number of parallel workers
        api_model: API model name if using API
        api_base_url: API base URL if using custom API
    """
    # Prepare constant iterables for additional arguments
    api_models = [api_model] * len(dataset)
    api_base_urls = [api_base_url] * len(dataset)
    no_thinks = [no_think] * len(dataset)
    original_model_names = [original_model_name] * len(dataset)

    # Process items in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each dataset item with shared flags across workers
        results = list(
            tqdm(
                executor.map(
                    process_item,
                    dataset,
                    api_models,
                    api_base_urls,
                    no_thinks,
                    original_model_names,
                ),
                total=len(dataset),
                desc="Evaluating",
            )
        )

    # Calculate aggregate metrics
    correct_answers = sum(1 for r in results if r["correct"])
    total = len(results)

    accuracy = correct_answers / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct_answers,
        "total": total,
        "detailed_results": results,
    }


def main():
    load_dotenv()
    args = parse_arguments()

    server_process = None
    temp_dir = None
    model_name = ""

    if args.api_model:
        model_name = args.api_model
        print(f"Evaluating API model: {model_name}")
    else:
        # Handle local or HF model
        if args.is_local:
            model_name = os.path.basename(args.model_path)
            model_to_serve = args.model_path
        else:
            model_name = args.model_path
            model_to_serve = args.model_path

        print(f"Starting VLLM server for model: {model_name}")

        # Start VLLM server
        server_process = start_vllm_server(
            model_to_serve_name=model_to_serve,
            served_model_name="model",
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    try:
        # Load dataset
        print(f"Loading dataset from {args.eval_data_path}")
        dataset = pd.read_parquet(args.eval_data_path).to_dict("records")

        # Run evaluation
        print("\nRunning evaluation...")
        results = evaluate_model(
            dataset,
            max_workers=args.max_workers,
            api_model=args.api_model,
            api_base_url=args.api_base_url,
            no_think=args.no_think,
            original_model_name=model_to_serve if not args.api_model else None,
        )

        print("\n=== Evaluation Results ===")
        print(f"Model: {model_name}")
        print(
            f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})"
        )

        # Save detailed results if requested
        if args.save_results:
            print(f"\nSaving detailed results to {args.save_results}")
            with open(args.save_results, "w") as f:
                json.dump(
                    {
                        "model_name": model_name,
                        "eval_data_path": args.eval_data_path,
                        "summary": {
                            "accuracy": results["accuracy"],
                            "correct": results["correct"],
                            "total": results["total"],
                        },
                        "detailed_results": results["detailed_results"],
                    },
                    f,
                    indent=2,
                )

    finally:
        if server_process:
            # Terminate VLLM server
            print("Terminating VLLM server...")
            server_process.terminate()
            server_process.wait()
            print("VLLM server terminated")

        # Clean up temporary directory if used
        if temp_dir and os.path.exists(temp_dir):
            import shutil

            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
