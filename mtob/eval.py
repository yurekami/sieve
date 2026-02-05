"""
MTOB: Evaluation Script for Kalamang-English Translation

Evaluates model outputs using chrF metric (matching the official MTOB benchmark).
Can evaluate both SIEVE-trained models and baselines.

Usage:
    python -m mtob.eval \
        --model_outputs_path outputs.json \
        --direction ke

Or for evaluating a model directly:
    python -m mtob.eval \
        --model_path path/to/model \
        --direction ke \
        --include_context
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import evaluate
import numpy as np

# Load chrF metric (primary metric for MTOB)
chrf_metric = evaluate.load("chrf")

# Paths to MTOB data
MTOB_SPLITS_PATH = Path(__file__).parent / "mtob-official" / "splits"
MTOB_RESOURCES_PATH = Path(__file__).parent / "mtob-official" / "resources"


def load_test_examples(direction: str = "ke") -> List[Dict]:
    """
    Load test examples for the specified direction.

    Args:
        direction: "ke" for Kalamang->English, "ek" for English->Kalamang

    Returns:
        List of dicts with "original" and "ground_truth" keys
    """
    path = MTOB_SPLITS_PATH / f"test_examples_{direction}.json"
    data = json.loads(path.read_text())

    # Skip the first element (header with big-bench-canary)
    examples = data[1:]

    return examples


def load_grammar_book(version: str = "medium") -> str:
    """Load the grammar book for context."""
    if version == "medium":
        path = MTOB_RESOURCES_PATH / "grammar_book_for_claude_medium.txt"
    elif version == "long":
        path = MTOB_RESOURCES_PATH / "grammar_book_for_claude_long.txt"
    else:
        path = MTOB_RESOURCES_PATH / "grammar_book.tex"
    return path.read_text()


def load_parallel_sentences() -> str:
    """Load parallel sentences for context (Cartridges format)."""
    path = MTOB_SPLITS_PATH / "train_examples.json"
    data = json.loads(path.read_text())
    sentences = data[1:]

    lines = []
    for item in sentences:
        kalamang = item.get("original", "")
        english = item.get("translation", "")
        if kalamang and english and english != "--------------------":
            lines.append(
                f"{kalamang}:{english}"
            )  # Cartridges uses colon without spaces

    return "\n".join(lines)


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """
    Compute chrF score (primary MTOB metric).

    Args:
        predictions: Model outputs
        references: Ground truth translations

    Returns:
        chrF score (0-100)
    """
    result = chrf_metric.compute(predictions=predictions, references=references)
    return result["score"]


def clean_prediction(text: str) -> str:
    """Clean up model prediction for evaluation."""
    # Remove common artifacts
    if text.endswith("<|eot_id|>"):
        text = text[: -len("<|eot_id|>")]

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


def build_translation_prompt(
    sentence: str,
    source_lang: str,
    target_lang: str,
    context: Optional[str] = None,
) -> str:
    """
    Build a translation prompt matching the Cartridges MTOB baseline format.

    Args:
        sentence: The sentence to translate
        source_lang: Source language name
        target_lang: Target language name
        context: Optional grammar/vocabulary context (not used in user prompt for Cartridges format)

    Returns:
        Formatted prompt (user message only - context goes in system prompt)
    """
    # Match Cartridges MTOB baseline format
    return f"""You are tasked with translating the following sentence from {source_lang} to {target_lang}: "{sentence}".
I understand that you may not be familiar enough with {source_lang} or {target_lang} to make a confident translation, but please give your best guess.
Make sure you respond with only the translation and absolutely no other text. Do not repeat the input sentence."""


def build_system_prompt(
    context: Optional[str] = None, direction: str = "ke"
) -> Optional[str]:
    """
    Build system prompt with context (Cartridges format).

    Args:
        context: Grammar book and parallel sentences content
        direction: "ke" for Kalamang->English, "ek" for English->Kalamang

    Returns:
        System prompt string or None if no context
    """
    if not context:
        return None

    source_lang = "Kalamang" if direction == "ke" else "English"
    target_lang = "English" if direction == "ke" else "Kalamang"

    return f"""Please reference the material below to help the user translate from {source_lang} to {target_lang}.

{context}"""


def evaluate_from_file(
    outputs_path: str,
    direction: str = "ke",
) -> Dict:
    """
    Evaluate model outputs from a JSON file.

    Expected format:
    [
        {"prediction": "...", "reference": "..."},
        ...
    ]

    Or:
    [
        {"output": "...", "ground_truth": "..."},
        ...
    ]
    """
    with open(outputs_path) as f:
        data = json.load(f)

    predictions = []
    references = []

    for item in data:
        pred = item.get("prediction") or item.get("output") or item.get("answer", "")
        ref = item.get("reference") or item.get("ground_truth", "")

        predictions.append(clean_prediction(pred))
        references.append(ref)

    chrf_score = compute_chrf(predictions, references)

    return {
        "chrf": chrf_score,
        "num_examples": len(predictions),
        "direction": direction,
    }


def evaluate_model(
    model,
    tokenizer,
    direction: str = "ke",
    include_context: bool = True,
    grammar_version: str = "medium",
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate a model on the MTOB test set.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        direction: "ke" or "ek"
        include_context: Whether to include grammar context
        grammar_version: Which grammar book version to use
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation

    Returns:
        Dict with evaluation results
    """
    import torch
    from tqdm import tqdm

    # Load test examples
    examples = load_test_examples(direction)

    # Prepare context if needed
    context = None
    if include_context:
        grammar = load_grammar_book(grammar_version)
        sentences = load_parallel_sentences()
        context = f"{grammar}\n\n---\nParallel sentences:\n{sentences}"

    # Determine source/target languages
    if direction == "ke":
        source_lang, target_lang = "Kalamang", "English"
    else:
        source_lang, target_lang = "English", "Kalamang"

    predictions = []
    references = []

    # Generate predictions
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
            batch = examples[i : i + batch_size]

            prompts = []
            for ex in batch:
                prompt = build_translation_prompt(
                    ex["original"],
                    source_lang,
                    target_lang,
                    context,
                )
                prompts.append(prompt)
                references.append(ex["ground_truth"])

            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode
            for j, output in enumerate(outputs):
                # Get only the generated part
                input_len = inputs["input_ids"][j].shape[0]
                generated = tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True,
                )
                predictions.append(clean_prediction(generated))

    # Compute metrics
    chrf_score = compute_chrf(predictions, references)

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
        "direction": direction,
        "include_context": include_context,
        "predictions": predictions,
        "references": references,
    }


def evaluate_with_vllm(
    model_path: str,
    direction: str = "ke",
    include_context: bool = False,
    grammar_version: str = "medium",
    include_sentences: bool = True,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    rope_scaling: Optional[str] = None,
    no_think: bool = False,
    temperature: float = 0.0,
) -> Dict:
    """
    Evaluate a model on MTOB test set using vLLM for efficient inference.

    Args:
        model_path: Path to HuggingFace model
        direction: "ke" or "ek"
        include_context: Whether to include grammar context (ICL baseline)
        grammar_version: Which grammar book version to use
        include_sentences: Whether to include parallel sentences in context
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model context length
        rope_scaling: Optional RoPE scaling config (JSON string)
        no_think: Disable thinking for Qwen models
        temperature: Sampling temperature (0.0 for greedy, >0 for sampling)

    Returns:
        Dict with evaluation results
    """
    from vllm import LLM, SamplingParams

    # Load test examples
    examples = load_test_examples(direction)

    # Prepare context if needed (Cartridges format)
    context = None
    if include_context:
        grammar = load_grammar_book(grammar_version)
        # Build context in Cartridges format
        book_label = (
            "Here is a subset of the book"
            if grammar_version in ("medium", "long")
            else "Here is the book"
        )
        context_parts = [
            f"""{book_label}, "A grammar of Kalamang":
START OF GRAMMAR BOOK
{grammar}
END OF GRAMMAR BOOK

The grammar book is now over."""
        ]

        if include_sentences:
            sentences = load_parallel_sentences()
            context_parts.append(f"""Here is the collection of parallel sentences:

START OF PARALLEL SENTENCES
{sentences}
END OF PARALLEL SENTENCES

The collection of parallel sentences is now over.""")

        context = "\n\n".join(context_parts)

    # Determine source/target languages
    if direction == "ke":
        source_lang, target_lang = "Kalamang", "English"
    else:
        source_lang, target_lang = "English", "Kalamang"

    # Build system prompt (context goes here in Cartridges format)
    system_prompt = build_system_prompt(context, direction)

    # Build prompts (user messages only)
    prompts = []
    references = []
    for ex in examples:
        prompt = build_translation_prompt(
            ex["original"],
            source_lang,
            target_lang,
        )
        prompts.append(prompt)
        references.append(ex["ground_truth"])

    # Parse rope_scaling if provided - use hf_overrides for vLLM
    # IMPORTANT: Must also set max_position_embeddings to avoid CUDA crashes
    # See: https://github.com/vllm-project/vllm/issues/17924
    engine_kwargs = {}
    if rope_scaling:
        import json as json_module

        rope_scaling_config = json_module.loads(rope_scaling)
        # Must update both rope_scaling AND max_position_embeddings
        # Otherwise torch compilation will crash with out-of-bounds errors
        engine_kwargs["hf_overrides"] = {
            "rope_scaling": rope_scaling_config,
            "max_position_embeddings": max_model_len,
        }

    # Initialize vLLM
    print(f"Loading model with vLLM: {model_path}")
    print(f"  tensor_parallel_size: {tensor_parallel_size}")
    print(f"  max_model_len: {max_model_len}")
    if rope_scaling:
        print(f"  rope_scaling: {rope_scaling}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        **engine_kwargs,
    )

    # Set up sampling params
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Apply chat template if model supports it
    # Context goes in system prompt (Cartridges format)
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            # Build messages with system prompt if context provided
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if no_think and "qwen" in model_path.lower():
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_prompts.append(formatted)
        else:
            # Fallback: prepend system prompt to user prompt
            if system_prompt:
                formatted_prompts.append(f"{system_prompt}\n\n{prompt}")
            else:
                formatted_prompts.append(prompt)

    # Generate translations
    print(f"Generating translations for {len(formatted_prompts)} examples...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract predictions
    predictions = []
    for output in outputs:
        text = output.outputs[0].text
        predictions.append(clean_prediction(text))

    # Compute metrics
    chrf_score = compute_chrf(predictions, references)

    # Compute exact match rate
    exact_matches = sum(
        1
        for p, r in zip(predictions, references)
        if p.lower().strip() == r.lower().strip()
    )
    exact_match_rate = exact_matches / len(predictions) * 100

    result = {
        "num_examples": len(predictions),
        "direction": direction,
        "include_context": include_context,
        "model_path": model_path,
        "predictions": predictions,
        "references": references,
        "chrf": chrf_score,
        "exact_match": exact_match_rate,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTOB translations")

    # Input options
    parser.add_argument(
        "--model_outputs_path",
        type=str,
        help="Path to JSON file with model outputs (mutually exclusive with --model_path)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to HuggingFace model for live evaluation",
    )

    # Data options
    parser.add_argument(
        "--direction",
        type=str,
        default="ke",
        choices=["ke", "ek"],
        help="Translation direction: ke (Kalamang->English) or ek (English->Kalamang)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save evaluation results",
    )

    # Context options (for ICL baseline)
    parser.add_argument(
        "--include_context",
        action="store_true",
        help="Include grammar book in context (ICL baseline)",
    )
    parser.add_argument(
        "--grammar_book_version",
        type=str,
        default="medium",
        choices=["medium", "long", "full_tex"],
        help="Grammar book version to use",
    )
    parser.add_argument(
        "--include_sentences",
        action="store_true",
        help="Include parallel sentences in context",
    )

    # Model loading options
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Model is local (not from HuggingFace Hub)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--rope_scaling",
        type=str,
        default=None,
        help='RoPE scaling config as JSON string, e.g. \'{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}\'',
    )

    # Generation options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="Disable thinking for Qwen models",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy, >0 for sampling)",
    )

    args = parser.parse_args()

    if args.model_outputs_path:
        # Evaluate from file
        results = evaluate_from_file(args.model_outputs_path, args.direction)
    elif args.model_path:
        model_path = args.model_path

        # Evaluate with vLLM
        results = evaluate_with_vllm(
            model_path=model_path,
            direction=args.direction,
            include_context=args.include_context,
            grammar_version=args.grammar_book_version,
            include_sentences=args.include_sentences,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            rope_scaling=args.rope_scaling,
            no_think=args.no_think,
            temperature=args.temperature,
        )
    else:
        print("Please provide --model_outputs_path or --model_path")
        return

    print("\n" + "=" * 50)
    print("MTOB Evaluation Results")
    print("=" * 50)
    print(f"Direction: {results['direction']}")
    print(f"Number of examples: {results['num_examples']}")

    # Print example predictions
    if "predictions" in results and "references" in results:
        print("\n--- Example Predictions ---")
        preds = results["predictions"]
        refs = results["references"]
        for i in range(min(2, len(preds))):
            print(f"Example {i + 1}:")
            print(f"  Reference:  {refs[i]}")
            print(f"  Prediction: {preds[i]}")

    # Print results
    print(f"chrF score: {results['chrf']:.2f}")
    if "exact_match" in results:
        print(f"Exact match: {results['exact_match']:.2f}%")

    if "include_context" in results:
        print(f"Context included: {results['include_context']}")

    if args.output_path:
        # Save results (without predictions/references for smaller file)
        save_results = {
            k: v for k, v in results.items() if k not in ["predictions", "references"]
        }
        with open(args.output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
