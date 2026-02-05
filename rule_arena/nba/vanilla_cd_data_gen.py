"""
Vanilla Context Distillation Data Generation for NBA CBA Rules.

This script converts a SIEVE synthetic dataset into a vanilla CD dataset by
replacing the selected rules (inside feedback delimiters) with ALL CBA rules.

Unlike SIEVE (which selects only applicable rules), vanilla CD includes ALL
CBA rules in every prompt. This serves as a baseline to demonstrate the
importance of targeted rule selection.

Usage:
    python -m rule_arena.nba.vanilla_cd_data_gen \
        --input_path nba_sieve_qwen_synthetic_4096.parquet \
        --output_path nba_vanilla_cd_4096.parquet
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd

from sieve.synthetic_data_gen import FEEDBACK_START_DELIMITER, FEEDBACK_END_DELIMITER


# Script directory for finding reference files
SCRIPT_DIR = Path(__file__).parent
RULE_ARENA_DIR = SCRIPT_DIR / ".." / "RuleArena" / "nba"


def load_reference_rules() -> str:
    """Load the full reference rules from the RuleArena benchmark."""
    rule_path = RULE_ARENA_DIR / "reference_rules.txt"
    with open(rule_path, "r") as f:
        return f.read()


def replace_feedback_with_all_rules(prompt_content: str, all_rules: str) -> str:
    """
    Replace the content between feedback delimiters with ALL rules.

    Args:
        prompt_content: The original prompt with selected rules
        all_rules: The full CBA rules to substitute

    Returns:
        The prompt with ALL rules replacing the selected rules
    """
    # Pattern to match everything between feedback delimiters (including the delimiters)
    pattern = re.escape(FEEDBACK_START_DELIMITER) + r".*?" + re.escape(FEEDBACK_END_DELIMITER)

    # New feedback section with ALL rules
    new_feedback = f"{FEEDBACK_START_DELIMITER}\nReference Rules in NBA Collective Bargaining Agreement:\n\n{all_rules}\n{FEEDBACK_END_DELIMITER}"

    # Replace the feedback section
    new_prompt = re.sub(pattern, new_feedback, prompt_content, flags=re.DOTALL)

    return new_prompt


def convert_sieve_to_vanilla_cd(
    input_df: pd.DataFrame,
    all_rules: str,
) -> List[Dict]:
    """
    Convert a SIEVE synthetic dataset to vanilla CD format.

    Takes each example from SIEVE and replaces the selected rules
    with ALL CBA rules.

    Args:
        input_df: DataFrame with 'prompt' column from SIEVE
        all_rules: Full text of all CBA rules

    Returns:
        List of examples with ALL rules in each prompt
    """
    examples = []

    for idx, row in input_df.iterrows():
        prompt = row["prompt"]

        # Handle both list format and direct content
        if isinstance(prompt, list):
            # Chat format: [{"role": "user", "content": "..."}]
            new_prompt = []
            for msg in prompt:
                if msg["role"] == "user":
                    new_content = replace_feedback_with_all_rules(msg["content"], all_rules)
                    new_prompt.append({"role": msg["role"], "content": new_content})
                else:
                    new_prompt.append(msg)
        else:
            # Direct string format
            new_prompt = [{"role": "user", "content": replace_feedback_with_all_rules(prompt, all_rules)}]

        examples.append({"prompt": new_prompt})

    return examples


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert SIEVE synthetic data to vanilla CD (all rules) format"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to SIEVE synthetic dataset (parquet)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for vanilla CD dataset (parquet)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load all CBA rules
    print("Loading CBA reference rules...")
    all_rules = load_reference_rules()
    rule_lines = [line for line in all_rules.split("\n") if line.strip()]
    print(f"Loaded {len(rule_lines)} lines of CBA rules")

    # Load SIEVE synthetic dataset
    print(f"Loading SIEVE synthetic dataset: {args.input_path}")
    input_df = pd.read_parquet(args.input_path)
    print(f"Loaded {len(input_df)} examples")

    print("=" * 60)
    print("Vanilla CD Conversion (NBA CBA)")
    print("=" * 60)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Replacing selected rules with ALL {len(rule_lines)} lines of CBA rules")
    print()

    # Convert to vanilla CD format
    print("Converting to vanilla CD format...")
    examples = convert_sieve_to_vanilla_cd(input_df, all_rules)

    # Save dataset
    df = pd.DataFrame(examples)
    df.to_parquet(args.output_path)
    print(f"Saved {len(examples)} examples to {args.output_path}")

    # Preview comparison
    print("\n" + "=" * 60)
    print("SAMPLE COMPARISON")
    print("=" * 60)

    if len(input_df) > 0:
        original = input_df.iloc[0]["prompt"]
        if isinstance(original, list):
            original_content = original[0]["content"]
        else:
            original_content = original

        new_content = examples[0]["prompt"][0]["content"]

        # Find and show the feedback sections
        orig_start = original_content.find(FEEDBACK_START_DELIMITER)
        orig_end = original_content.find(FEEDBACK_END_DELIMITER) + len(FEEDBACK_END_DELIMITER)
        new_start = new_content.find(FEEDBACK_START_DELIMITER)
        new_end = new_content.find(FEEDBACK_END_DELIMITER) + len(FEEDBACK_END_DELIMITER)

        if orig_start >= 0 and orig_end > orig_start:
            orig_feedback = original_content[orig_start:orig_end]
            print(f"ORIGINAL feedback length: {len(orig_feedback)} chars")
            print(f"ORIGINAL feedback preview:\n{orig_feedback[:500]}...")

        if new_start >= 0 and new_end > new_start:
            new_feedback = new_content[new_start:new_end]
            print(f"\nNEW (all rules) feedback length: {len(new_feedback)} chars")
            print(f"NEW feedback preview:\n{new_feedback[:500]}...")


if __name__ == "__main__":
    main()
