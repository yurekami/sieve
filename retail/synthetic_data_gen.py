"""
Synthetic data generation for the retail discount domain.

This script generates synthetic retail queries where:
1. A base model selects which portion of the discount rules should apply
2. An instruction model generates a realistic shopping scenario for those rules

No programmatic verification or ground truth computation - we trust the model's
generation and let soft distillation handle the learning.

Usage:
    python -m retail.synthetic_data_gen \
        --single_server \
        --n_examples 512 \
        --output_path retail_synthetic_512.parquet
"""

import argparse

from sieve.synthetic_data_gen import (
    SyntheticDataGenerator,
    add_common_args,
    FEEDBACK_START_DELIMITER,
    FEEDBACK_END_DELIMITER,
)
from retail.data_gen import RetailRuleEngine


class RetailSyntheticDataGenerator(SyntheticDataGenerator):
    """
    Synthetic data generator for the retail discount domain.

    Only implements domain-specific methods:
    - get_feedback(): Returns discount rules as NL
    - get_feedback_selection_examples(): Few-shot examples for rule selection
    - get_query_generation_examples(): Few-shot examples for query generation
    - build_final_prompt(): Formats the final training prompt

    Uses the core SIEVE prompts for feedback selection and query generation.
    """

    def __init__(self, rule_engine: RetailRuleEngine = None, **kwargs):
        super().__init__(**kwargs)
        self.rule_engine = rule_engine or RetailRuleEngine()
        self._feedback_cache = None

    def get_feedback(self) -> str:
        """Return all discount rules as one block of natural language."""
        if self._feedback_cache is None:
            lines = ["Discount Rules:"]
            for r in self.rule_engine.rules:
                lines.append(f"- {r.natural_language}")
            self._feedback_cache = "\n".join(lines)
        return self._feedback_cache

    def get_feedback_selection_examples(self) -> str:
        """
        Few-shot examples showing how to select discount rules.

        For base models, these examples are critical for guiding the format.
        Each example shows a valid combination of rules that could apply together.
        """
        if self.use_base_model:
            # Base model: more structured examples with clear input/output format
            # 3 examples with varying bullet counts (2, 3, 4) and diverse rule combinations
            return """Example 1:
Selected guidelines:
- If promo code is 'WELCOME10', apply $10 fixed discount
- If customer is a student AND cart contains electronics, apply 15% discount on electronics items only
###

Example 2:
Selected guidelines:
- If customer is a teacher AND total spend is at least $50, apply 10% discount to total purchase
- If cart contains books items AND total books spend is $45 or greater, apply 10% discount on books items only
- If customer has been a member for 1 or more years, apply 5% discount to total purchase
###

Example 3:
Selected guidelines:
- If customer is a employee AND total spend is at least $50, apply 20% discount to total purchase
- If promo code is 'HOLIDAY30' AND total spend is at least $150, apply 30% discount to total purchase
- If cart contains clothing items AND total clothing spend is $75 or greater, apply 10% discount on clothing items only
- If total spend is at least $200, apply 10% discount to total purchase
###"""
        else:
            # Instruction model: simpler examples
            return """Example 1 - Student with electronics purchase:
- If customer is a student AND total spend is at least $50, apply 10% discount to total purchase
- If customer is a student AND cart contains electronics, apply 15% discount on electronics items only

Example 2 - Senior with promo code:
- If customer is a senior citizen AND total spend is at least $50, apply 15% discount to total purchase
- If promo code is 'SAVE20' AND total spend is at least $100, apply 20% discount to total purchase

Example 3 - Long-time member with category spend:
- If customer has been a member for 5 or more years, apply 15% discount to total purchase
- If cart contains clothing items AND total clothing spend is $75 or greater, apply 10% discount on clothing items only
- If total spend is at least $150, apply 5% discount to total purchase"""

    def get_query_generation_examples(self) -> str:
        """Few-shot examples showing the expected query format."""
        return """Example 1 (for student with electronics rules):
Customer Profile:
- Type: student
- Membership years: 0

Shopping Cart:
- Laptop (electronics): $150.00 x 1
- Headphones (electronics): $45.00 x 1
- Notebook (office): $12.00 x 2

Promo code: None

Example 2 (for senior with promo code rules):
Customer Profile:
- Type: senior
- Membership years: 2

Shopping Cart:
- Vitamins (health): $35.00 x 2
- First Aid Kit (health): $28.00 x 1
- Towel (home): $18.00 x 3

Promo code: SAVE20

IMPORTANT: Output ONLY the Customer Profile, Shopping Cart, and Promo code sections. Do NOT include any headers like 'Guidelines', 'Scenario', or 'Query'. Do NOT include any explanation or question."""

    def build_final_prompt(
        self,
        generated_query: str,
        selected_feedback: str,
        all_feedback: str,
        include_feedback: bool = True,
    ) -> str:
        """
        Build the final training prompt from the generated scenario.

        The selected_feedback is wrapped in delimiters so it can be easily
        stripped during soft distillation training for internalization.
        """
        instructions = """IMPORTANT: Apply discounts in this exact order to the running total:
1. Category-specific percentage discounts (apply only the highest discount per category to each category's subtotal)
2. Total purchase percentage discounts (apply only the highest total discount to the remaining amount after step 1)
3. Fixed amount discounts (subtract from the remaining amount after step 2, sum all applicable fixed discounts)

Note: Each discount applies to the current running total, not the original price."""

        # Wrap feedback in delimiters for easy removal during training
        feedback_section = (
            f"{FEEDBACK_START_DELIMITER}\n{selected_feedback}\n{FEEDBACK_END_DELIMITER}"
        )

        prompt = f"""Calculate the final price for the following customer purchase after applying all applicable discount rules.

{generated_query}

{instructions}

{feedback_section}

Calculate the final price after applying all applicable discount rules. End your answer with \\boxed{{final price}}."""

        return prompt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic retail discount data"
    )
    add_common_args(parser)

    parser.add_argument(
        "--single_server",
        action="store_true",
        help="Use the same server for both base and instruction models",
    )
    parser.add_argument(
        "--start_servers",
        action="store_true",
        help="Start vLLM servers automatically (two servers on different GPU sets)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-8B-Base",
        help="Model path for base model (feedback selection)",
    )
    parser.add_argument(
        "--instruction_model_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path for instruction model (query generation)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for each vLLM server",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Maximum model length for vLLM",
    )
    parser.add_argument(
        "--base_model_gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU indices for base model server",
    )
    parser.add_argument(
        "--instruction_model_gpus",
        type=str,
        default="4,5,6,7",
        help="Comma-separated GPU indices for instruction model server",
    )
    parser.add_argument(
        "--base_model_port",
        type=int,
        default=8000,
        help="Port for base model server",
    )
    parser.add_argument(
        "--instruction_model_port",
        type=int,
        default=8001,
        help="Port for instruction model server",
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use base model (text completion) for feedback selection instead of instruction model (chat completion). Enables BARE-style diverse generation.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    base_server_process = None
    instruction_server_process = None

    try:
        # Start two vLLM servers on different GPU sets
        if args.start_servers:
            from utils.vllm import start_vllm_server

            # Parse GPU indices
            base_gpus = [int(g) for g in args.base_model_gpus.split(",")]
            instruction_gpus = [int(g) for g in args.instruction_model_gpus.split(",")]

            # Start base model server (GPUs 0-3 by default)
            print(
                f"Starting base model server: {args.base_model_path} on GPUs {base_gpus}"
            )
            base_server_process = start_vllm_server(
                model_to_serve_name=args.base_model_path,
                served_model_name="base_model",
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                port=args.base_model_port,
                cuda_devices=base_gpus,
            )

            # Start instruction model server (GPUs 4-7 by default)
            print(
                f"Starting instruction model server: {args.instruction_model_path} on GPUs {instruction_gpus}"
            )
            instruction_server_process = start_vllm_server(
                model_to_serve_name=args.instruction_model_path,
                served_model_name="instruction_model",
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                port=args.instruction_model_port,
                cuda_devices=instruction_gpus,
            )

            # Update model names and API bases
            args.base_model = "base_model"
            args.instruction_model = "instruction_model"
            args.base_model_api_base = f"http://127.0.0.1:{args.base_model_port}/v1"
            args.instruction_model_api_base = (
                f"http://127.0.0.1:{args.instruction_model_port}/v1"
            )

        # Handle single server mode (both models on same server)
        if args.single_server:
            args.instruction_model_api_base = args.base_model_api_base

        # Initialize generator
        generator = RetailSyntheticDataGenerator(
            base_model=args.base_model,
            instruction_model=args.instruction_model,
            base_model_api_base=args.base_model_api_base,
            instruction_model_api_base=args.instruction_model_api_base,
            temperature_base=args.temperature_base,
            temperature_instruction=args.temperature_instruction,
            enable_thinking=args.enable_thinking,
            max_workers=args.max_workers,
            use_base_model=args.use_base_model,
            verify_feedback=args.verify_feedback,
            temperature_verification=args.temperature_verification,
        )

        # Generate dataset
        examples = generator.generate_dataset(
            n_examples=args.n_examples,
            include_feedback_in_prompt=args.include_feedback,
        )

        # Save dataset
        generator.save_dataset(examples, args.output_path)

        # Preview
        if examples:
            print("\n" + "=" * 60)
            print("SAMPLE GENERATED EXAMPLES")
            print("=" * 60)
            print(f"Prompt 1: {examples[0].prompt[0]['content']}")
            print(f"Prompt 2: {examples[1].prompt[0]['content']}")
            print(f"Prompt 3: {examples[2].prompt[0]['content']}")

            print("DECOMPOSED FEEDBACK ITEMS")
            print(generator._decomposed_feedback_items)

    finally:
        if base_server_process:
            print("Terminating base model server...")
            base_server_process.terminate()
            base_server_process.wait()
        if instruction_server_process:
            print("Terminating instruction model server...")
            instruction_server_process.terminate()
            instruction_server_process.wait()


if __name__ == "__main__":
    main()
