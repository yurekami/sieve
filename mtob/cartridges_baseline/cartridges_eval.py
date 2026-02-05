"""
Cartridges Evaluation for MTOB

Evaluates a trained Cartridge on the MTOB Kalamang->English translation task.
Uses Tokasaurus to serve the model with the cartridge and evaluates
using the chrF metric (same as official benchmark).

Usage:
    python -m mtob.cartridges_baseline.cartridges_eval \
        --cartridge_id your-wandb-run-id \
        --cartridge_source wandb \
        --tokasaurus_url http://localhost:10210

Or for baseline (no cartridge):
    python -m mtob.cartridges_baseline.cartridges_eval \
        --baseline \
        --tokasaurus_url http://localhost:10210
"""

import argparse
import json
import sys
from pathlib import Path
import requests
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
import evaluate

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Load chrF metric
chrf_metric = evaluate.load("chrf")

# MTOB test data paths
MTOB_SPLITS_PATH = Path(__file__).parent.parent / "mtob-official" / "splits"


def load_test_examples(direction: str = "ke"):
    """Load test examples for the specified direction."""
    path = MTOB_SPLITS_PATH / f"test_examples_{direction}.json"
    data = json.loads(path.read_text())
    # Skip the first element (header with big-bench-canary)
    return data[1:]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate Cartridge on MTOB translation task"
    )
    parser.add_argument(
        "--cartridge_id",
        type=str,
        default=None,
        help="WandB run ID or path to the trained cartridge",
    )
    parser.add_argument(
        "--cartridge_source",
        type=str,
        default="wandb",
        choices=["wandb", "local", "hf"],
        help="Source of the cartridge (wandb, local, or hf)",
    )
    parser.add_argument(
        "--tokasaurus_url",
        type=str,
        default="http://localhost:10210",
        help="URL of the Tokasaurus server",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="ke",
        choices=["ke", "ek"],
        help="Translation direction: ke (Kalamang->English) or ek (English->Kalamang)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=50,
        help="Maximum number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Path to save detailed evaluation results as JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline evaluation (no cartridge)",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        default=True,
        help="Disable thinking mode for generation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model name (must match model served by Tokasaurus for correct chat template)",
    )

    return parser.parse_args()


def build_prompt(sentence: str, source_lang: str, target_lang: str) -> str:
    """Build translation prompt matching MTOB format."""
    return f"""You are tasked with translating the following sentence from {source_lang} to {target_lang}: "{sentence}".
I understand that you may not be familiar enough with {source_lang} or {target_lang} to make a confident translation, but please give your best guess.
Make sure you respond with only the translation and absolutely no other text. Do not repeat the input sentence."""


def query_with_cartridge(
    prompt: str,
    cartridge_id: str,
    cartridge_source: str,
    tokasaurus_url: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
    no_think: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
) -> str:
    """Query Tokasaurus with a cartridge."""
    try:
        # Match the official TokasaurusClient request format exactly
        request_body = {
            "model": model_name,  # Must match actual model for correct chat template
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,  # Official uses max_completion_tokens
            "temperature": temperature,
            "apply_chat_template_overrides": {"enable_thinking": not no_think},
            "logprobs_in_fingerprint": True,
        }

        if cartridge_id:
            request_body["cartridges"] = [
                {
                    "id": cartridge_id,
                    "source": cartridge_source,
                    "force_redownload": False,
                }
            ]
            # Use /custom/cartridge/chat/completions for single requests with cartridge
            endpoint = f"{tokasaurus_url}/custom/cartridge/chat/completions"
        else:
            endpoint = f"{tokasaurus_url}/v1/chat/completions"

        response = requests.post(
            endpoint,
            json=request_body,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying: {e}")
        return ""


def clean_prediction(text: str) -> str:
    """Clean up model prediction for evaluation."""
    # Remove thinking tags if present
    if "</think>" in text:
        text = text.split("</think>")[1]
    if "</thinking>" in text:
        text = text.split("</thinking>")[1]

    # Strip whitespace
    text = text.strip()

    # Remove quotes if the entire response is quoted
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text


def process_item(args):
    """Process a single evaluation item."""
    (
        item,
        source_lang,
        target_lang,
        cartridge_id,
        cartridge_source,
        tokasaurus_url,
        temperature,
        max_tokens,
        no_think,
        model_name,
    ) = args

    prompt = build_prompt(item["original"], source_lang, target_lang)
    ground_truth = item["ground_truth"]

    try:
        completion = query_with_cartridge(
            prompt,
            cartridge_id,
            cartridge_source,
            tokasaurus_url,
            temperature,
            max_tokens,
            no_think,
            model_name,
        )

        prediction = clean_prediction(completion)

        return {
            "prediction": prediction,
            "reference": ground_truth,
            "completion": completion,
        }

    except Exception as e:
        print(f"Error processing item: {e}")
        return {
            "prediction": "",
            "reference": ground_truth,
            "completion": "",
            "error": str(e),
        }


def evaluate_model(
    test_data,
    source_lang: str,
    target_lang: str,
    cartridge_id: str,
    cartridge_source: str,
    tokasaurus_url: str,
    max_workers: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 128,
    no_think: bool = True,
    model_name: str = "Qwen/Qwen3-4B",
):
    """Evaluate the model on MTOB test set in parallel."""
    process_args = [
        (
            item,
            source_lang,
            target_lang,
            cartridge_id,
            cartridge_source,
            tokasaurus_url,
            temperature,
            max_tokens,
            no_think,
            model_name,
        )
        for item in test_data
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_item, process_args),
                total=len(test_data),
                desc="Evaluating",
            )
        )

    predictions = [r["prediction"] for r in results]
    references = [r["reference"] for r in results]

    # Compute chrF score
    chrf_score = chrf_metric.compute(predictions=predictions, references=references)[
        "score"
    ]

    # Compute exact match rate
    exact_matches = sum(
        1
        for p, r in zip(predictions, references)
        if p.lower().strip() == r.lower().strip()
    )
    exact_match_rate = exact_matches / len(predictions) * 100

    return {
        "chrf": chrf_score,
        "exact_match": exact_match_rate,
        "num_examples": len(predictions),
        "predictions": predictions,
        "references": references,
        "detailed_results": results,
    }


def main():
    load_dotenv()
    args = parse_arguments()

    # Validate args
    if not args.baseline and not args.cartridge_id:
        print("Error: Must specify --cartridge_id or --baseline")
        return

    # Determine languages
    if args.direction == "ke":
        source_lang, target_lang = "Kalamang", "English"
    else:
        source_lang, target_lang = "English", "Kalamang"

    # Load test data
    print(f"Loading test data for direction: {args.direction}")
    test_data = load_test_examples(args.direction)
    print(f"Loaded {len(test_data)} test examples")

    if args.baseline:
        print("\nRunning baseline evaluation (no cartridge)")
        cartridge_id = None
    else:
        print(f"\nEvaluating with cartridge: {args.cartridge_id}")
        print(f"Cartridge source: {args.cartridge_source}")
        cartridge_id = args.cartridge_id

    print(f"Tokasaurus URL: {args.tokasaurus_url}")
    print(f"Model name: {args.model_name}")

    # Debug: Check what model Tokasaurus is actually serving
    try:
        import requests as req
        models_resp = req.get(f"{args.tokasaurus_url}/v1/models", timeout=10)
        if models_resp.ok:
            models_data = models_resp.json()
            print(f"Tokasaurus serving model: {models_data}")
    except Exception as e:
        print(f"Could not query Tokasaurus models endpoint: {e}")

    print("\nRunning evaluation...")
    results = evaluate_model(
        test_data,
        source_lang,
        target_lang,
        cartridge_id=cartridge_id,
        cartridge_source=args.cartridge_source,
        tokasaurus_url=args.tokasaurus_url,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        no_think=args.no_think,
        model_name=args.model_name,
    )

    print("\n" + "=" * 50)
    print("MTOB Cartridges Evaluation Results")
    print("=" * 50)
    print(f"Direction: {args.direction}")
    print(f"Number of examples: {results['num_examples']}")
    print(f"chrF score: {results['chrf']:.2f}")
    print(f"Exact match: {results['exact_match']:.2f}%")

    if args.baseline:
        print("\n(Baseline - no cartridge)")
    else:
        print(f"\nCartridge: {args.cartridge_id}")

    # Print sample results
    print("\n" + "-" * 50)
    print("Sample Predictions:")
    print("-" * 50)
    detailed = results["detailed_results"]
    for i in range(min(5, len(detailed))):
        item = detailed[i]
        print(f"\nExample {i + 1}:")
        print(f"  Reference:  {item['reference']}")
        print(f"  Prediction: {item['prediction']}")
        is_match = item['prediction'].lower().strip() == item['reference'].lower().strip()
        print(f"  Match: {'✓' if is_match else '✗'}")

    # Save results
    if args.save_results:
        print(f"\nSaving results to {args.save_results}")
        save_results = {
            "direction": args.direction,
            "cartridge_id": args.cartridge_id if not args.baseline else None,
            "baseline": args.baseline,
            "chrf": results["chrf"],
            "exact_match": results["exact_match"],
            "num_examples": results["num_examples"],
        }

        with open(args.save_results, "w") as f:
            json.dump(save_results, f, indent=2)


if __name__ == "__main__":
    main()
