"""
MTOB: Synthetic Data Generation for Kalamang-English Translation

This script generates synthetic translation queries where:
1. A base model selects relevant grammar rules, vocabulary, and example sentences
2. An instruction model generates a Kalamang sentence using those patterns
3. The result is a translation query: "Translate: [Kalamang sentence]"

Uses chunking to handle the medium-sized grammar book (~50k tokens for medium version).

Usage:
    python -m mtob.synthetic_data_gen \
        --grammar_book_version medium \
        --enable_chunking \
        --n_examples 4096 \
        --output_path mtob_synthetic_4096.parquet
"""

import argparse
import json
from pathlib import Path

from sieve.synthetic_data_gen import (
    SyntheticDataGenerator,
    add_common_args,
    FEEDBACK_START_DELIMITER,
    FEEDBACK_END_DELIMITER,
)


# Path to MTOB resources
MTOB_RESOURCES_PATH = Path(__file__).parent / "mtob-official" / "resources"
MTOB_SPLITS_PATH = Path(__file__).parent / "mtob-official" / "splits"


def load_grammar_book(version: str = "medium") -> str:
    """Load the Kalamang grammar book.

    Args:
        version: One of "medium" (recommended), "long", or "full_tex"
    """
    if version == "medium":
        path = MTOB_RESOURCES_PATH / "grammar_book_for_claude_medium.txt"
    elif version == "long":
        path = MTOB_RESOURCES_PATH / "grammar_book_for_claude_long.txt"
    elif version == "full_tex":
        path = MTOB_RESOURCES_PATH / "grammar_book.tex"
    else:
        raise ValueError(f"Unknown grammar book version: {version}")

    return path.read_text()


def load_parallel_sentences() -> str:
    """Load the parallel Kalamang-English sentences from train_examples.json."""
    path = MTOB_SPLITS_PATH / "train_examples.json"
    data = json.loads(path.read_text())

    # Skip the first element (header with big-bench-canary)
    sentences = data[1:]

    # Format as "Kalamang: English" pairs
    lines = []
    for item in sentences:
        kalamang = item.get("original", "")
        english = item.get("translation", "")
        if kalamang and english and english != "--------------------":
            lines.append(f"{kalamang} : {english}")

    return "\n".join(lines)


class MTOBSyntheticDataGenerator(SyntheticDataGenerator):
    """
    Synthetic data generator for Kalamang-English translation.

    Generates translation queries by:
    1. Selecting relevant grammar rules + vocabulary + example sentences
    2. Generating a new Kalamang sentence using those patterns
    3. Wrapping as a translation prompt
    """

    def __init__(
        self,
        grammar_book_version: str = "medium",
        include_sentences: bool = True,
        **kwargs,
    ):
        # Enable chunking by default for grammar book
        kwargs.setdefault("enable_chunking", True)
        kwargs.setdefault("chunk_size", 8192)
        kwargs.setdefault("max_items_for_selection", 50)
        kwargs.setdefault(
            "max_tokens_for_selection", 4000
        )  # ~4k tokens for feedback items

        super().__init__(**kwargs)

        self.grammar_book_version = grammar_book_version
        self.include_sentences = include_sentences
        self._feedback_cache = None

    def get_feedback(self) -> str:
        """Return grammar book + parallel sentences as NLF."""
        if self._feedback_cache is None:
            grammar = load_grammar_book(self.grammar_book_version)

            if self.include_sentences:
                sentences = load_parallel_sentences()
                self._feedback_cache = f"""{grammar}

---
Parallel Kalamang-English Sentences (for reference):
{sentences}"""
            else:
                self._feedback_cache = grammar

        return self._feedback_cache

    def get_feedback_selection_examples(self) -> str:
        """
        Few-shot examples for selecting relevant grammar/vocabulary for translation.

        Each example shows 6-10 items selected together - enough to construct
        an interesting Kalamang sentence for translation.

        NOTE: Examples removed in this code release to prevent leakage of the Kalamang grammar book.
        """
        return ""

    def get_feedback_selection_prompt(self, feedback: str) -> str:
        """
        Core SIEVE prompt: Ask base model to select a subset of feedback.

        This is a key part of the SIEVE method - using a base model with high
        temperature to diversely sample which portions of feedback should apply
        to a synthetic example.

        When use_base_model=True, uses a completion-style prompt with stronger
        formatting cues suitable for base models (no chat template).
        """
        examples = self.get_feedback_selection_examples()

        if self.use_base_model:
            # Base model prompt: completion-style with strong formatting cues
            examples_section = f"\n{examples}\n" if examples else ""
            return f"""Task: Select 6-8 pieces of information from the corpus of knowledge that could be used to construct a single question about translating from Kalamang to English. Your job is to only select pieces of knowledge, not construct a question yourself.

Information:
{feedback}
{examples_section}
Selected information:
-"""

    def get_query_generation_examples(self) -> str:
        """Few-shot examples showing simple Kalamang questions.

        Focus on vocabulary lookups and grammar explanations that can be
        directly answered from the selected feedback.

        NOTE: Examples removed in this code release to prevent leakage of the Kalamang grammar book.
        """
        return ""

    def build_final_prompt(
        self,
        generated_query: str,
        selected_feedback: str,
        all_feedback: str,
        include_feedback: bool = True,
    ) -> str:
        """
        Build the final training prompt for translation.

        The query should already be in format: "Translate this Kalamang sentence to English: [sentence]"
        We add the grammar context wrapped in delimiters.
        """
        # Clean up the query if it has extra formatting
        query = generated_query.strip()

        if include_feedback:
            feedback_section = f"""
{FEEDBACK_START_DELIMITER}
Relevant Kalamang Grammar and Vocabulary:
{selected_feedback}
{FEEDBACK_END_DELIMITER}
"""
        else:
            feedback_section = ""

        return f"""{query}

{feedback_section}"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Kalamang translation data"
    )
    add_common_args(parser)

    parser.add_argument(
        "--grammar_book_version",
        type=str,
        default="medium",
        choices=["medium", "long", "full_tex"],
        help="Version of grammar book to use (default: medium)",
    )
    parser.add_argument(
        "--include_sentences",
        action="store_true",
        default=True,
        help="Include parallel sentences in NLF (default: True)",
    )
    parser.add_argument(
        "--no_sentences",
        action="store_true",
        help="Exclude parallel sentences from NLF",
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

        # Determine whether to include sentences
        include_sentences = args.include_sentences and not args.no_sentences

        # Initialize generator
        generator = MTOBSyntheticDataGenerator(
            grammar_book_version=args.grammar_book_version,
            include_sentences=include_sentences,
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
            # Chunking parameters
            enable_chunking=args.enable_chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tokenizer_path=args.tokenizer_path,
            max_items_for_selection=args.max_items_for_selection,
            max_tokens_for_selection=args.max_tokens_for_selection,
            verification_batch_size=args.verification_batch_size,
        )

        print(f"Grammar book version: {args.grammar_book_version}")
        print(f"Include parallel sentences: {include_sentences}")
        print(f"Chunking enabled: {args.enable_chunking}")

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
            for i, ex in enumerate(examples[:3]):
                print(f"\n--- Example {i + 1} ---")
                print(f"Prompt: {ex.prompt[0]['content']}")
                print(f"Selected feedback: {ex.selected_feedback}")

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
