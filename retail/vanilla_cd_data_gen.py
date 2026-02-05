"""
Vanilla Context Distillation Data Generation for Retail Simple.

This script generates training data for a vanilla context distillation baseline.
Unlike SIEVE (which selects only applicable rules), this baseline includes ALL
discount rules in every prompt.

The goal is to demonstrate that without targeted rule selection, the model fails
even with more training samples.

Usage:
    python -m retail.vanilla_cd_data_gen \
        --n_examples 4096 \
        --output_path vanilla_cd_retail_4096.parquet
"""

import argparse
import random
from typing import Dict, List

import pandas as pd

from retail.data_gen import RetailRuleEngine
from sieve.synthetic_data_gen import FEEDBACK_START_DELIMITER, FEEDBACK_END_DELIMITER


# Use a different seed to avoid overlap with eval data (which uses seed=42)
VANILLA_CD_SEED = 12345


def create_vanilla_cd_prompt(
    rule_engine: RetailRuleEngine,
    cart,
    customer,
    total_spend: float,
    promo_code: str = None,
) -> str:
    """
    Create a prompt with ALL rules wrapped in feedback delimiters.

    This is the key difference from the standard data generation:
    - Standard: Only includes applicable rules
    - Vanilla CD: Includes ALL rules in every prompt

    The feedback delimiters allow the soft distillation training to strip
    the rules during internalization.
    """
    # Format cart and customer info
    cart_description = rule_engine.format_cart_description(
        cart, customer, total_spend, promo_code
    )

    # Format ALL rules (not just applicable ones)
    all_rules_text = "\n".join(
        [f"- {rule.natural_language}" for rule in rule_engine.rules]
    )

    # Wrap rules in delimiters for soft distillation
    feedback_section = f"{FEEDBACK_START_DELIMITER}\nDiscount Rules:\n{all_rules_text}\n{FEEDBACK_END_DELIMITER}"

    # Build the full prompt
    prompt = f"""Calculate the final price for the following customer purchase after applying all applicable discount rules.

{cart_description}

IMPORTANT: Apply discounts in this exact order to the running total:
1. Category-specific percentage discounts (apply only the highest discount per category to each category's subtotal)
2. Total purchase percentage discounts (apply only the highest total discount to the remaining amount after step 1)
3. Fixed amount discounts (subtract from the remaining amount after step 2, sum all applicable fixed discounts)

Note: Each discount applies to the current running total, not the original price.

{feedback_section}

Calculate the final price after applying all applicable discount rules. End your answer with \\boxed{{final price}}."""

    return prompt


def generate_vanilla_cd_dataset(
    n_examples: int,
    rule_engine: RetailRuleEngine,
    seed: int = VANILLA_CD_SEED,
) -> List[Dict]:
    """
    Generate vanilla context distillation dataset.

    Each example includes ALL rules in the prompt (wrapped in delimiters),
    regardless of which rules actually apply to the scenario.

    Output format matches what sieve/soft_distillation_data.py expects:
    just a 'prompt' column with chat-format messages.
    """
    random.seed(seed)
    examples = []

    # Promo code distribution (same as original data_gen)
    promo_codes = [
        "SAVE20",
        "WELCOME10",
        "NEWBIE5",
        "STUDENT15",
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    for _ in range(n_examples):
        # Generate random purchase scenario
        cart, total_spend = rule_engine.generate_random_cart()
        customer = rule_engine.generate_random_customer()
        promo_code = random.choice(promo_codes)

        # Create prompt with ALL rules (vanilla CD approach)
        prompt_content = create_vanilla_cd_prompt(
            rule_engine, cart, customer, total_spend, promo_code
        )

        # Only need prompt for soft distillation (no ground truth needed)
        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt_content}],
            }
        )

    return examples


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate vanilla context distillation data for retail simple"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=4096,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=VANILLA_CD_SEED,
        help="Random seed (different from eval seed to avoid overlap)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for generated dataset (parquet)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize rule engine
    rule_engine = RetailRuleEngine()

    print("=" * 60)
    print("Vanilla Context Distillation Data Generation")
    print("=" * 60)
    print(f"Number of examples: {args.n_examples}")
    print(f"Random seed: {args.seed}")
    print(f"Total rules (ALL included in every prompt): {len(rule_engine.rules)}")
    print(f"Output path: {args.output_path}")
    print()

    # Generate dataset
    print("Generating vanilla CD dataset...")
    examples = generate_vanilla_cd_dataset(
        n_examples=args.n_examples,
        rule_engine=rule_engine,
        seed=args.seed,
    )

    # Save dataset
    df = pd.DataFrame(examples)
    df.to_parquet(args.output_path)
    print(f"Saved {len(examples)} examples to {args.output_path}")

    # Preview
    print("\n" + "=" * 60)
    print("SAMPLE GENERATED EXAMPLE")
    print("=" * 60)
    print(f"Prompt:\n{examples[0]['prompt'][0]['content']}")


if __name__ == "__main__":
    main()
