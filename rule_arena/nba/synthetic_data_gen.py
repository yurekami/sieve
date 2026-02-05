"""
Synthetic data generation for the NBA CBA rules compliance domain.

This script generates synthetic NBA CBA rules compliance queries where:
1. A base model selects which portion of the CBA rules should apply
2. An instruction model generates a realistic team/player scenario for those rules

Uses the RuleArena NBA benchmark's reference rules as the feedback source.

Usage:
    python -m rule_arena.nba.synthetic_data_gen \
        --single_server \
        --n_examples 512 \
        --output_path nba_synthetic_512.parquet
"""

import argparse
import os

from sieve.synthetic_data_gen import (
    SyntheticDataGenerator,
    add_common_args,
    FEEDBACK_START_DELIMITER,
    FEEDBACK_END_DELIMITER,
)


# Salary cap constants for prompts
SALARY_CAP_CONSTANTS = """Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year."""


class NBASyntheticDataGenerator(SyntheticDataGenerator):
    """
    Synthetic data generator for the NBA CBA rules compliance domain.

    Uses RuleArena's NBA CBA rules as the natural language feedback.
    The rules cover:
    - Maximum salary based on years of service
    - Contract length limitations
    - Salary cap exceptions (Bird rights, Mid-Level, Bi-annual, etc.)
    - Trade rules and Traded Player Exceptions
    - Apron level restrictions
    - Sign-and-trade rules
    - Restricted free agent rules
    """

    def __init__(self, rules_file: str = None, **kwargs):
        super().__init__(**kwargs)

        # Default to the reference rules file
        if rules_file is None:
            # Find the rules file relative to this module
            # synthetic_data_gen.py is at rule_arena/nba/synthetic_data_gen.py
            # RuleArena is at rule_arena/RuleArena/nba/
            module_dir = os.path.dirname(os.path.abspath(__file__))
            rules_file = os.path.join(
                module_dir, "..", "RuleArena", "nba", "reference_rules.txt"
            )

        self.rules_file = rules_file
        self._feedback_cache = None

    def get_feedback(self) -> str:
        """Return all NBA CBA rules as one block of natural language."""
        if self._feedback_cache is None:
            with open(self.rules_file, "r") as f:
                self._feedback_cache = f.read()
        return self._feedback_cache

    def get_feedback_selection_examples(self) -> str:
        """
        Few-shot examples showing how to select NBA CBA rules.

        For base models, these examples guide the model to select coherent
        combinations of rules that could apply to a single transaction scenario.
        """
        if self.use_base_model:
            # Base model: more structured examples with clear input/output format
            return """Example 1:
Selected guidelines:
- For any player who has completed fewer than seven (7) Years of Service, his Player Contract may not provide for a Salary in the first Season that exceeds twenty-five percent (25%) of the Salary Cap.
- A Team's Team Salary may not exceed the Salary Cap at any time unless the Team is using one of the Exceptions.
- A Team may use the Non-Taxpayer Mid-Level Salary Exception to sign and/or acquire by assignment one (1) or more Player Contracts during each Salary Cap Year.
- A Team may not sign or acquire a player using the Non-Taxpayer Mid-Level Salary Exception if, immediately following such transaction, the Team's Team Salary would exceed the First Apron Level.
###

Example 2:
Selected guidelines:
- For any player who has completed ten (10) or more Years of Service, his Player Contract may not provide for a Salary in the first Season that exceeds thirty-five percent (35%) of the Salary Cap.
- A Player Contract may cover up to but no more than four (4) Seasons; a Player Contract between a Qualifying Veteran Free Agent and his Prior Team may cover up to five (5) Seasons.
- A Qualifying Veteran Free Agent may have a Salary Cap Exception, according to which the player may enter into a new Player Contract with his Prior Team.
- For all Player Contracts between Qualifying Veteran Free Agents and their Prior Team, the player's Salary may increase or decrease by no more than eight percent (8%) of the Salary for the first Salary Cap Year.
###

Example 3:
Selected guidelines:
- A Team may use the Standard Traded Player Exception to replace one Traded Player with Replacement Players whose post-assignment Salaries are no more than 100% of the pre-trade Salary plus $250,000.
- If a Team's post-assignment Team Salary would exceed the First Apron Level, then the $250,000 allowance shall be reduced to $0.
- A Team may use the Aggregated Standard Traded Player Exception to replace two or more Traded Players with Replacement Players.
- A Team may not acquire a player using an Aggregated Standard Traded Player Exception if the Team's Team Salary would exceed the Second Apron Level.
###"""
        else:
            # Instruction model: simpler examples
            return """Example 1 - Team signing a young player with Mid-Level Exception:
- Maximum salary for players with fewer than 7 Years of Service is 25% of Salary Cap
- Team Salary may not exceed Salary Cap unless using an Exception
- Non-Taxpayer Mid-Level Exception allows signing players
- Non-Taxpayer Mid-Level Exception has First Apron Level hard cap

Example 2 - Veteran re-signing with his team:
- Maximum salary for 10+ Years of Service is 35% of Salary Cap
- Qualifying Veteran Free Agent can sign up to 5-year contract with Prior Team
- Qualifying Veteran Free Agent Exception allows re-signing
- 8% annual salary increase/decrease allowed for Qualifying Veteran Free Agent

Example 3 - Trade scenario with multiple players:
- Standard Traded Player Exception: 100% of traded player salary + $250,000
- $250,000 allowance reduced to $0 if exceeding First Apron Level
- Aggregated Standard Traded Player Exception for multiple players
- Second Apron Level hard cap for Aggregated Traded Player Exception"""

    def get_query_generation_examples(self) -> str:
        """
        Few-shot examples showing the expected team/player scenario format.

        NOTE: These examples are intentionally different from the eval set
        to avoid data leakage.
        """
        return """Example 1 (for mid-level exception rules):
Team Situations:
Team A has a team salary of $145,000,000.

Player Situations:
Player A was the 12th first-round pick of Team B in 2020 NBA draft when he was 19 years old.
Player A signed a 2-year contract (annual salary $2,000,000, 5% increase per year) with Team B during 2022 Moratorium Period.

Operations:
A. Team A signs a 3-year contract with Player A providing annual salary $10,000,000 in the first Salary Cap Year (2024-2025) and 5% increase per year.

Example 2 (for veteran re-signing rules):
Team Situations:
Team A has a team salary of $130,000,000.

Player Situations:
Player A was the 5th first-round pick of Team A in 2010 NBA draft when he was 19 years old.
Player A signed a 4-year contract (annual salary $30,000,000, 8% increase per year) with Team A during 2020 Moratorium Period.

Operations:
A. Team A signs a 5-year contract with Player A providing annual salary $45,000,000 in the first Salary Cap Year (2024-2025) and 8% increase per year.

Example 3 (for trade rules):
Team Situations:
Team A has a team salary of $170,000,000.
Team B has a team salary of $140,000,000.

Player Situations:
Player A was the 8th first-round pick of Team C in 2018 NBA draft when he was 20 years old.
Player A signed a 4-year contract (annual salary $25,000,000, 5% increase per year) with Team A during 2022 Moratorium Period.
Player B was the 20th first-round pick of Team D in 2019 NBA draft when he was 19 years old.
Player B signed a 3-year contract (annual salary $12,000,000, 5% increase per year) with Team B during 2021 Moratorium Period.

Operations:
A. Team A trades Player A to Team B for Player B.

IMPORTANT: Output ONLY the scenario in this exact format. Include:
- Team Situations with team salaries
- Player Situations with draft info, contract details
- Operations describing the transaction

Do NOT include any headers, explanations, or the answer."""

    def build_final_prompt(
        self,
        generated_query: str,
        selected_feedback: str,
        all_feedback: str,
        include_feedback: bool = True,
    ) -> str:
        """
        Build the final training prompt from the generated team/player scenario.

        The selected_feedback is wrapped in delimiters so it can be easily
        stripped during soft distillation training for internalization.
        """
        instructions = """Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/..."""

        # Wrap feedback in delimiters for easy removal during training
        feedback_section = (
            f"{FEEDBACK_START_DELIMITER}\nReference Rules in NBA Collective Bargaining Agreement:\n{selected_feedback}\n{FEEDBACK_END_DELIMITER}"
        )

        prompt = f"""You are a helpful NBA team consultant. You are given rules in NBA Collective Bargaining Agreement and the information about some teams and players. Then you will be given a list of operations, each of which describes how some teams conduct some transaction. You should determine whether each operation complies with the given rules.

{SALARY_CAP_CONSTANTS}

Decide whether any operation by any team violates the rules:

{generated_query}

{instructions}

{feedback_section}"""

        return prompt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic NBA CBA rules compliance data"
    )
    add_common_args(parser)

    parser.add_argument(
        "--rules_file",
        type=str,
        default=None,
        help="Path to the NBA CBA rules file (default: reference_rules.txt)",
    )
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

        print(f"Thinking mode: {'enabled' if args.enable_thinking else 'disabled'}")

        # Initialize generator
        generator = NBASyntheticDataGenerator(
            rules_file=args.rules_file,
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
            debug_logs=args.debug_logs,
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
            for i in range(min(3, len(examples))):
                print(f"\nPrompt {i + 1}: {examples[i].prompt[0]['content']}")
                print(f"Selected feedback: {examples[i].selected_feedback}")

        if generator._decomposed_feedback_items:
            print("\n" + "=" * 60)
            print(
                f"DECOMPOSED FEEDBACK: {len(generator._decomposed_feedback_items)} items"
            )
            print("=" * 60)

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
