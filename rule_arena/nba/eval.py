"""
Evaluation script for the NBA CBA rules compliance task.

Uses the RuleArena benchmark's existing annotated problems to verify whether
NBA team operations comply with Collective Bargaining Agreement rules.

Usage:
    # Evaluate a HuggingFace model
    python -m rule_arena.nba.eval \
        --model_path "Qwen/Qwen3-8B" \
        --eval_variant no_rules \
        --complexity 2 \
        --max_workers 50 \
        --no_think

    # Evaluate an API model
    python -m rule_arena.nba.eval \
        --api_model "gpt-4o" \
        --eval_variant all_rules \
        --complexity 2
"""

import argparse
import json
import os
import sys
import tempfile
import re
from tqdm import tqdm
import concurrent.futures
import litellm
from dotenv import load_dotenv

# Add RuleArena to path for imports
# eval.py is at rule_arena/nba/eval.py
# RuleArena is at rule_arena/RuleArena/nba/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULE_ARENA_DIR = os.path.join(SCRIPT_DIR, "..", "RuleArena", "nba")
sys.path.insert(0, RULE_ARENA_DIR)

from utils.vllm import start_vllm_server


# System prompt for the NBA assistant
SYSTEM_PROMPT = """You are a helpful NBA team consultant."""

# Salary cap constants for 2024-25 season
SALARY_CAP_CONSTANTS = """Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year."""

# Prompt template with rules
PROMPT_TEMPLATE_WITH_RULES = """You are given rules in NBA Collective Bargaining Agreement and the information about some teams and players. Then you will be given a list of operations, each of which describes how some teams conduct some transaction. You should determine whether each operation complies with the given rules.

{salary_cap_constants}

Reference Rules in NBA Collective Bargaining Agreement:

{reference_rules}

Decide whether any operation by any team violates the rules:

{question}

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/..."""

# Prompt template without rules
PROMPT_TEMPLATE_NO_RULES = """You are given the information about some teams and players. Then you will be given a list of operations, each of which describes how some teams conduct some transaction. You should determine whether each operation complies with NBA Collective Bargaining Agreement rules.

{salary_cap_constants}

Decide whether any operation by any team violates the rules:

{question}

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/..."""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate models on NBA CBA rules compliance task"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to local model directory or HF model name",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Whether the model is a local model or from HF",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768, help="Maximum model length for VLLM"
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
        default=50,
        help="Maximum number of parallel workers for evaluation",
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default="0",
        help="Complexity level(s): 0 (simplest), 1 (medium), 2 (hardest). Can specify multiple levels as comma-separated (e.g., '0,1,2').",
    )
    parser.add_argument(
        "--eval_variant",
        type=str,
        choices=["all_rules", "no_rules"],
        default="no_rules",
        help="Evaluation variant: all_rules (with CBA rules) or no_rules (without rules, tests internalization)",
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
        "--no_think",
        action="store_true",
        help="Disable thinking mode for Qwen3 (default: enabled)",
    )

    return parser.parse_args()


def load_reference_rules() -> str:
    """Load the reference rules from the RuleArena benchmark."""
    rule_path = os.path.join(RULE_ARENA_DIR, "reference_rules.txt")
    with open(rule_path, "r") as f:
        return f.read()


def load_problems(complexity: int):
    """Load annotated problems from the RuleArena benchmark."""
    problem_file = os.path.join(
        RULE_ARENA_DIR, "annotated_problems", f"comp_{complexity}.json"
    )
    with open(problem_file, "r") as f:
        problems = json.load(f)
    return problems


def build_query_prompt(problem: dict) -> str:
    """Build the query prompt from the problem dictionary."""
    team_info = "Team Situations:\n" + "\n".join(problem["team_situations"])
    player_info = "Player Situations:\n" + "\n".join(problem["player_situations"])
    operations = "Operations:\n" + "\n".join(problem["operations"])
    return team_info + "\n\n" + player_info + "\n\n" + operations


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
    temperature = 0.0  # Default temperature for evaluation
    top_p = 0.95
    top_k = 20
    min_p = 0

    # Use original_model_name for model-specific logic detection
    model_for_logic = original_model_name if original_model_name else model

    # Set model-specific parameters
    if "gpt-5" in (model or ""):
        temperature = 1.0
    elif "Qwen3" in (model_for_logic or ""):
        temperature = 0.6

    # If no api_model is specified, we default to the local VLLM server.
    if not model:
        model = "openai/model"
        api_base = "http://127.0.0.1:8000/v1"

    try:
        if "Qwen3" in (model_for_logic or ""):
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
        else:
            response = litellm.completion(
                model=model,
                messages=prompt,
                temperature=temperature,
                api_base=api_base,
            )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""


def extract_answer(response: str) -> dict:
    """Extract the answer from the model response.

    Returns a dict with:
    - violation: bool (True if violation found, False otherwise)
    - illegal_operation: str or None (e.g., "A", "B")
    - problematic_team: str or None (e.g., "A", "B")
    """
    response_clean = response.replace("**", "").strip()

    # Look for "Answer: False" pattern
    if "Answer: False" in response_clean:
        return {
            "violation": False,
            "illegal_operation": None,
            "problematic_team": None,
        }

    # Look for "Answer: True. Illegal Operation: X. Problematic Team: Y." pattern
    pattern = r"Answer:\s*True\.\s*Illegal Operation:\s*([A-Z])\.\s*Problematic Team:\s*([A-Z])"
    match = re.search(pattern, response_clean)
    if match:
        return {
            "violation": True,
            "illegal_operation": match.group(1),
            "problematic_team": match.group(2),
        }

    # Fallback: try to find any indication of True/False
    if "Answer: True" in response_clean:
        return {
            "violation": True,
            "illegal_operation": None,
            "problematic_team": None,
        }

    return None


def process_item(
    item,
    reference_rules: str,
    eval_variant: str,
    api_model: str = None,
    api_base_url: str = None,
    no_think: bool = False,
    original_model_name: str = None,
):
    """Process a single evaluation item."""
    question_prompt = build_query_prompt(item)

    # Ground truth
    ground_truth_violation = item["answer"]
    ground_truth_illegal_op = item.get("illegal_operation")
    ground_truth_team = item.get("problematic_team")

    # Build the prompt based on variant
    if eval_variant == "all_rules":
        user_content = PROMPT_TEMPLATE_WITH_RULES.format(
            salary_cap_constants=SALARY_CAP_CONSTANTS,
            reference_rules=reference_rules,
            question=question_prompt,
        )
    else:  # no_rules
        user_content = PROMPT_TEMPLATE_NO_RULES.format(
            salary_cap_constants=SALARY_CAP_CONSTANTS,
            question=question_prompt,
        )

    # Create chat format prompt
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        # Generate the completion from the model
        completion = query_model(
            prompt,
            api_model=api_model,
            api_base_url=api_base_url,
            no_think=no_think,
            original_model_name=original_model_name,
        )

        # Extract the answer
        extracted = extract_answer(completion)

        # Check if correct
        is_correct = False
        if extracted is not None:
            if not ground_truth_violation:
                # Ground truth is False (no violation)
                is_correct = not extracted["violation"]
            else:
                # Ground truth is True (violation exists)
                is_correct = (
                    extracted["violation"]
                    and extracted["illegal_operation"] == ground_truth_illegal_op
                    and extracted["problematic_team"] == ground_truth_team
                )

        return {
            "correct": is_correct,
            "predicted": extracted,
            "ground_truth": {
                "violation": ground_truth_violation,
                "illegal_operation": ground_truth_illegal_op,
                "problematic_team": ground_truth_team,
            },
            "completion": completion,
            "question": question_prompt,
        }

    except Exception as e:
        print(f"Error processing item: {e}")
        return {
            "correct": False,
            "predicted": None,
            "ground_truth": {
                "violation": ground_truth_violation,
                "illegal_operation": ground_truth_illegal_op,
                "problematic_team": ground_truth_team,
            },
            "completion": "",
            "error": str(e),
        }


def evaluate_model(
    problems,
    reference_rules: str,
    eval_variant: str,
    max_workers=50,
    api_model: str = None,
    api_base_url: str = None,
    no_think: bool = False,
    original_model_name: str = None,
):
    """Evaluate the model on the NBA CBA rules compliance task in parallel."""
    # Process items in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                reference_rules,
                eval_variant,
                api_model,
                api_base_url,
                no_think,
                original_model_name,
            )
            for item in problems
        ]

        results = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(problems),
            desc="Evaluating",
        ):
            results.append(future.result())

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
    model_to_serve = None

    if args.api_model:
        model_name = args.api_model
        print(f"Evaluating API model: {model_name}")
    elif args.model_path:
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
    else:
        raise ValueError("Either --model_path or --api_model must be specified")

    try:
        # Parse complexity levels from comma-separated string
        complexity_levels = [int(c.strip()) for c in args.complexity.split(',')]
        
        # Validate complexity levels
        for c in complexity_levels:
            if c not in [0, 1, 2]:
                raise ValueError(f"Invalid complexity level: {c}. Must be 0, 1, or 2.")
        
        # Load problems and rules
        print(f"\nLoading problems for complexity level(s): {complexity_levels}...")
        problems = []
        for complexity in complexity_levels:
            complexity_problems = load_problems(complexity)
            print(f"  Loaded {len(complexity_problems)} problems from complexity {complexity}")
            problems.extend(complexity_problems)
        print(f"Total problems loaded: {len(problems)}")

        reference_rules = ""
        if args.eval_variant == "all_rules":
            print("Loading reference rules...")
            reference_rules = load_reference_rules()

        # Run evaluation
        print(
            f"\nRunning evaluation with variant '{args.eval_variant}'..."
        )
        print(f"Thinking mode: {'disabled' if args.no_think else 'enabled'}")

        results = evaluate_model(
            problems,
            reference_rules,
            args.eval_variant,
            max_workers=args.max_workers,
            api_model=args.api_model,
            api_base_url=args.api_base_url,
            no_think=args.no_think,
            original_model_name=model_to_serve if not args.api_model else None,
        )

        print("\n=== Evaluation Results ===")
        print(f"Model: {model_name}")
        print(f"Complexity: {args.complexity}")
        print(f"Variant: {args.eval_variant}")
        print(f"Thinking: {'disabled' if args.no_think else 'enabled'}")
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
                        "complexity": args.complexity,
                        "eval_variant": args.eval_variant,
                        "no_think": args.no_think,
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

        # Print example question, response, and expected answer
        print("\n" + "=" * 80)
        print("EXAMPLE RESULT")
        print("=" * 80)

        # Find one correct and one incorrect example
        detailed = results["detailed_results"]
        correct_ex = next((r for r in detailed if r["correct"]), None)
        incorrect_ex = next((r for r in detailed if not r["correct"]), None)

        example = incorrect_ex if incorrect_ex else correct_ex
        if example:
            print(f"\n{'INCORRECT' if not example['correct'] else 'CORRECT'} EXAMPLE:")
            print("-" * 40)
            print(f"QUESTION:\n{example.get('question', 'N/A')}")
            print("-" * 40)
            print(f"MODEL RESPONSE:\n{example.get('completion', 'N/A')}")
            print("-" * 40)
            print(f"EXPECTED ANSWER: {example['ground_truth']}")
            print(f"MODEL PREDICTED: {example['predicted']}")
            print(f"CORRECT: {example['correct']}")

    finally:
        if server_process:
            print("Terminating VLLM server...")
            server_process.terminate()
            server_process.wait()
            print("VLLM server terminated")

        if temp_dir and os.path.exists(temp_dir):
            import shutil

            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
