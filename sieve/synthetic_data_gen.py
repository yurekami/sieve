"""
SIEVE: Synthetic Data Generation (SIEVE-GEN)

Generate diverse synthetic queries from natural language feedback (NLF).

This module provides a framework for:
1. Using a base model to select which portion of NLF should apply to a query
2. Using an instruction model to generate synthetic queries for that NLF subset

The approach treats feedback as pure natural language - one block of text,
no assumptions about structure or how it's organized.

Supports chunking for large NLF that exceeds context limits:
- Chunks NLF into overlapping segments before decomposition
- Samples subsets of decomposed items for feedback selection
- Uses batched verification for efficiency with large item lists
"""

import argparse
import concurrent.futures
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import litellm

# Delimiters for feedback section - used to wrap feedback in prompts so it can
# be easily stripped during soft distillation training for internalization
FEEDBACK_START_DELIMITER = "<|feedback_start|>"
FEEDBACK_END_DELIMITER = "<|feedback_end|>"


@dataclass
class SyntheticExample:
    """Represents a generated synthetic example."""

    prompt: List[Dict[str, str]]  # Chat format: [{"role": "user", "content": "..."}]
    selected_feedback: (
        str  # The portion of NLF that applies (as selected by base model)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticDataGenerator(ABC):
    """
    Abstract base class for synthetic data generation.

    Input: One block of natural language feedback (NLF) - no assumptions about structure.

    Subclasses must implement:
    - get_feedback(): Return the natural language feedback as a single string
    - build_final_prompt(): Build the final training prompt from generated query

    The feedback selection and query generation prompts are part of the core SIEVE
    method and should NOT be overridden unless absolutely necessary.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-8B-Base",
        instruction_model: str = "Qwen/Qwen3-8B",
        base_model_api_base: str = "http://127.0.0.1:8000/v1",
        instruction_model_api_base: str = "http://127.0.0.1:8001/v1",
        temperature_base: float = 1.0,
        temperature_instruction: float = 0.7,
        enable_thinking: bool = False,
        max_workers: int = 12,
        use_base_model: bool = False,
        verify_feedback: bool = False,
        temperature_verification: float = 0.3,
        debug_logs: bool = False,
        # Chunking parameters for large NLF
        enable_chunking: bool = False,
        chunk_size: int = 8192,
        chunk_overlap: int = 512,
        tokenizer_path: str = "Qwen/Qwen3-8B",
        max_items_for_selection: int = 100,
        max_tokens_for_selection: int = 4000,  # Token budget for feedback in selection prompt
        verification_batch_size: int = 50,
    ):
        self.base_model = base_model
        self.instruction_model = instruction_model
        self.base_model_api_base = base_model_api_base
        self.instruction_model_api_base = instruction_model_api_base
        self.temperature_base = temperature_base
        self.temperature_instruction = temperature_instruction
        self.enable_thinking = enable_thinking
        self.max_workers = max_workers
        self.use_base_model = use_base_model
        self.verify_feedback = verify_feedback
        self.temperature_verification = temperature_verification
        self.debug_logs = debug_logs

        # Chunking parameters
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_path = tokenizer_path
        self.max_items_for_selection = max_items_for_selection
        self.max_tokens_for_selection = max_tokens_for_selection
        self.verification_batch_size = verification_batch_size
        self._tokenizer = None  # Lazy loaded

        # Cache for decomposed feedback items (computed once)
        self._decomposed_feedback_items: Optional[List[str]] = None

    @abstractmethod
    def get_feedback(self) -> str:
        """Return the natural language feedback as a single string."""
        pass

    @abstractmethod
    def build_final_prompt(
        self,
        generated_query: str,
        selected_feedback: str,
        all_feedback: str,
        include_feedback: bool = True,
    ) -> str:
        """Build the final training prompt from the generated query."""
        pass

    def get_feedback_selection_examples(self) -> str:
        """
        Return few-shot examples for feedback selection.

        Subclasses should override this to provide domain-specific examples
        that guide the base model to select feedback in the correct format.
        """
        return ""

    def get_query_generation_examples(self) -> str:
        """
        Return few-shot examples for query generation.

        Subclasses should override this to provide domain-specific examples
        that guide the instruction model to generate queries in the correct format.
        """
        return ""

    def get_decomposition_prompt(self, chunk: str, is_chunk: bool = False) -> str:
        """
        Return the prompt for decomposing feedback into items.

        Args:
            chunk: The text to decompose
            is_chunk: If True, this is one chunk of a larger document (allow grouping).
                      If False, this is the full feedback (keep items atomic).

        Subclasses can override this to provide domain-specific decomposition.
        """
        if is_chunk:
            # Chunked mode: allow grouping related concepts since we have many chunks
            return f"""Break down the following content into 30-40 modular items. DO NOT create an excessive amount of items.

Since this is part of a larger document, you may group related concepts together (e.g., a rule with its examples, or multiple pieces of knowledge).

Each item should:
1. Be understandable on its own
2. Group related information together
3. Preserve the original meaning

Content:
{chunk}

Output each item separated by "###". Do NOT number or label items.
You MUST produce between 30-40 items total. For example:
###
Words for animals: pep = pig, kerar = turtle, wang = dugong, war = shark
###"""
        else:
            # Non-chunked mode: keep items atomic for fine-grained selection
            return f"""Break down the following feedback/guidelines/knowledge into atomic, independent items.

Each atomic item should:
1. Express a single, self-contained rule, fact, definition, or example
2. Be evaluable independently (can determine if it applies without needing other items)
3. Preserve the exact meaning and wording from the original

Content:
{chunk}

Output each atomic item separated by "###" on its own line.
For items with sub-bullets or multiple lines, include all lines as part of that item.
Do NOT number or label items. Do not add explanations or commentary.

Example format:
First item content here
###
Second item content here
###
Third item content here

Do not group multiple concepts together. Each item should be atomic."""

    # ========== Chunking Support for Large NLF ==========

    @property
    def tokenizer(self):
        """Lazy load tokenizer only when chunking is enabled."""
        if self._tokenizer is None and self.enable_chunking:
            from transformers import AutoTokenizer

            print(f"Loading tokenizer: {self.tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def chunk_feedback(self, feedback: str) -> List[str]:
        """
        Chunk large feedback into overlapping segments.

        Uses tokenizer to ensure accurate token counts. Chunks overlap
        to avoid splitting concepts at boundaries.

        Args:
            feedback: The full NLF text

        Returns:
            List of text chunks, each approximately chunk_size tokens
        """
        if not self.enable_chunking:
            return [feedback]

        tokens = self.tokenizer.encode(feedback)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            print(f"Feedback fits in single chunk ({total_tokens} tokens)")
            return [feedback]

        chunks = []
        start = 0
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move start forward, accounting for overlap
            start += self.chunk_size - self.chunk_overlap

            # Avoid infinite loop if overlap >= chunk_size
            if self.chunk_overlap >= self.chunk_size:
                start = end

        print(
            f"Split feedback ({total_tokens} tokens) into {len(chunks)} chunks of ~{self.chunk_size} tokens each"
        )
        return chunks

    def _deduplicate_items(self, items: List[str]) -> List[str]:
        """
        Remove duplicate items from the list.

        Uses normalized string comparison (lowercase, stripped whitespace).
        Preserves original casing of the first occurrence.
        """
        seen = set()
        unique_items = []
        for item in items:
            normalized = item.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_items.append(item)
        return unique_items

    def _decompose_single_chunk(self, chunk: str, is_chunk: bool = False) -> List[str]:
        """
        Decompose a single chunk of feedback into items.

        Args:
            chunk: The text to decompose
            is_chunk: If True, this is one chunk of a larger document (allow grouping).
                      If False, this is the full feedback (keep items atomic).
        """
        prompt = self.get_decomposition_prompt(chunk, is_chunk=is_chunk)
        response = self._query_model(
            prompt,
            self.instruction_model,
            self.instruction_model_api_base,
            0.3,  # Low temperature for consistent decomposition
            is_base_model=False,
        )

        if "</think>" in response:
            response = response.split("</think>")[1]

        # Parse response into list of items, splitting by ### delimiter
        items = []
        for item in response.strip().split("###"):
            item = item.strip()
            if item and not item.startswith("#"):  # Skip empty items and comments
                # Clean up any <item X> or similar labels the model might add
                import re

                item = re.sub(r"^<item\s*\d*>\s*", "", item, flags=re.IGNORECASE)
                item = re.sub(r"^item\s*\d+[:\s]*", "", item, flags=re.IGNORECASE)
                item = item.strip()
                if item:
                    items.append(item)

        return items

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
            return f"""Task: Select 3-5 guidelines from the natural language feedback that could apply together to a single scenario/question.

Guidelines:
{feedback}
{examples_section}
Selected guidelines:
-"""  # Start with "-" to guide the model to output a list
        else:
            # Instruction model prompt: chat-style
            examples_section = f"\n\nExamples:\n{examples}\n" if examples else ""
            return f"""You are helping generate diverse training data. Given the following guidelines/feedback, select a random subset that could realistically apply together to a single scenario.

{feedback}

Instructions:
1. Select a subset of guidelines that could logically apply together
2. Be diverse - don't always pick the same combinations
3. Consider which guidelines can realistically co-occur
4. Output ONLY the selected guidelines, copied exactly as they appear above{examples_section}

Selected guidelines:"""

    def get_query_generation_prompt(self, selected_feedback: str) -> str:
        """
        Core SIEVE prompt: Generate a realistic query for the selected feedback.

        This is a key part of the SIEVE method - using an instruction model to
        generate diverse, realistic queries that would trigger the selected feedback.
        """
        examples = self.get_query_generation_examples()
        examples_section = f"\n\nExamples:\n{examples}\n" if examples else ""

        return f"""Generate a realistic question where the following feedback/guidelines/knowledge would apply:

{selected_feedback}

Instructions:
1. Create a specific question where the information applies, similar to the format of the examples below
2. Make it realistic
3. Include all necessary details
4. Output ONLY the question, nothing else{examples_section}

Question:"""

    def _query_model(
        self,
        prompt: str,
        model: str,
        api_base: str,
        temperature: float,
        is_base_model: bool = False,
    ) -> str:
        """
        Query a model and return the response.

        For base models (is_base_model=True), uses text-completion-openai/ prefix
        which routes to the /v1/completions endpoint instead of /v1/chat/completions.
        This avoids applying chat templates to base models.
        """
        try:
            if is_base_model:
                # Use text completion for base models (no chat template)
                response = litellm.completion(
                    model=f"text-completion-openai/{model}",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    api_base=api_base,
                    max_tokens=1024,
                    stop=[
                        "###",
                        "\n\n\n",
                        "---",
                    ],  # Stop sequences to prevent runaway (### matches examples)
                )
            else:
                # Use chat completion for instruction-tuned models
                kwargs = {
                    "model": f"openai/{model}",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "api_base": api_base,
                }
                if not self.enable_thinking:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }

                response = litellm.completion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""

    def select_feedback(self, feedback: str) -> str:
        """
        Use base model to select which portion of feedback should apply.
        Returns the selected portion as a string.

        If use_base_model=True, uses text completion (no chat template).
        Otherwise uses chat completion with instruction model.

        When chunking is enabled and there are many decomposed items,
        samples a subset to fit in context for feedback selection.
        """
        # If chunking enabled and we have decomposed items, use subset sampling
        if self.enable_chunking and self._decomposed_feedback_items is not None:
            items = self._decomposed_feedback_items

            # Shuffle items for random sampling
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)

            # Select items up to token budget
            selected_items = []
            total_tokens = 0
            for item in shuffled_items:
                # Estimate tokens (rough: 1 token ≈ 4 chars)
                item_tokens = len(item) // 4 + 10  # +10 for "- " prefix and newline
                if total_tokens + item_tokens <= self.max_tokens_for_selection:
                    selected_items.append(item)
                    total_tokens += item_tokens
                    if len(selected_items) >= self.max_items_for_selection:
                        break
                elif len(selected_items) == 0:
                    # Always include at least one item even if it exceeds budget
                    selected_items.append(item)
                    total_tokens += item_tokens
                    break

            feedback_for_selection = "\n".join([f"- {item}" for item in selected_items])
            if self.debug_logs:
                print(
                    f"Selected {len(selected_items)}/{len(items)} items (~{total_tokens} tokens) for feedback selection"
                )
        else:
            feedback_for_selection = feedback

        prompt = self.get_feedback_selection_prompt(feedback_for_selection)
        if self.debug_logs:
            print(
                f"DEBUG: feedback_for_selection length: {len(feedback_for_selection)} chars"
            )
            print(
                f"DEBUG: final prompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens)"
            )
        response = self._query_model(
            prompt,
            self.base_model,
            self.base_model_api_base,
            self.temperature_base,
            is_base_model=self.use_base_model,
        )
        return response

    def generate_query(self, selected_feedback: str) -> str:
        """Use instruction model to generate a query for the selected feedback."""
        prompt = self.get_query_generation_prompt(selected_feedback)
        response = self._query_model(
            prompt,
            self.instruction_model,
            self.instruction_model_api_base,
            self.temperature_instruction,
            is_base_model=False,  # Always use chat completion for query generation
        )
        if "</think>" in response:
            response = response.split("</think>")[1]
        return response

    def decompose_feedback_into_items(self, feedback: str) -> List[str]:
        """
        Decompose natural language feedback into atomic, independent items.

        This is called ONCE per dataset generation and cached. It converts
        potentially unstructured feedback into a list of atomic items that
        can be individually verified.

        When chunking is enabled:
        1. Splits feedback into overlapping chunks
        2. Decomposes each chunk separately
        3. Deduplicates items from overlapping regions

        Args:
            feedback: The full natural language feedback (may be structured or unstructured)

        Returns:
            List of atomic feedback items
        """
        if self.enable_chunking:
            chunks = self.chunk_feedback(feedback)
            all_items = []

            # Decompose chunks in parallel for efficiency
            print(f"Decomposing {len(chunks)} chunks in parallel...")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all chunk decomposition tasks (is_chunk=True for chunked mode)
                future_to_idx = {
                    executor.submit(self._decompose_single_chunk, chunk, True): i
                    for i, chunk in enumerate(chunks)
                }

                # Collect results as they complete
                chunk_results = [None] * len(chunks)
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(chunks),
                    desc="Decomposing chunks",
                ):
                    idx = future_to_idx[future]
                    try:
                        items = future.result()
                        chunk_results[idx] = items
                        if self.debug_logs:
                            print(
                                f"Chunk {idx + 1}/{len(chunks)}: extracted {len(items)} items"
                            )
                    except Exception as e:
                        print(f"Error decomposing chunk {idx + 1}: {e}")
                        chunk_results[idx] = []

            # Flatten results in order
            for items in chunk_results:
                if items:
                    all_items.extend(items)

            # Deduplicate items from overlapping regions
            unique_items = self._deduplicate_items(all_items)
            print(
                f"Total items: {len(all_items)}, after deduplication: {len(unique_items)}"
            )
            # Debug: show item size distribution
            if unique_items:
                item_lengths = [len(item) for item in unique_items]
                print(
                    f"Item size stats (chars): min={min(item_lengths)}, max={max(item_lengths)}, avg={sum(item_lengths) // len(item_lengths)}, total={sum(item_lengths)}"
                )
            return unique_items
        else:
            return self._decompose_single_chunk(feedback)

    def get_decomposed_feedback_items(self, feedback: str) -> List[str]:
        """
        Get decomposed feedback items, using cache if available.

        This ensures decomposition only happens once per dataset generation.
        """
        if self._decomposed_feedback_items is None:
            print("Decomposing feedback into atomic items (one-time operation)...")
            self._decomposed_feedback_items = self.decompose_feedback_into_items(
                feedback
            )
            print(
                f"Decomposed into {len(self._decomposed_feedback_items)} atomic items"
            )
        return self._decomposed_feedback_items

    def verify_single_feedback_item(self, query: str, feedback_item: str) -> bool:
        """
        Verify if a single feedback item applies to a query.

        Uses binary yes/no classification which is more reliable than
        open-ended recall.

        Args:
            query: The generated query/scenario
            feedback_item: A single atomic feedback item to check

        Returns:
            True if the feedback item applies, False otherwise
        """
        prompt = f"""Is the following feedback item relevant to answering or handling this query?

Query:
{query}

Feedback item:
{feedback_item}

Answer with ONLY "Yes" or "No". Consider:
- Would this feedback item help someone answer or respond to this query?
- Does any part of the feedback (rules, vocabulary, examples) apply to this query?
- Be strict: if the feedback is not clearly needed, answer "No"

Answer:"""

        response = self._query_model(
            prompt,
            self.instruction_model,
            self.instruction_model_api_base,
            self.temperature_verification,
            is_base_model=False,
        )

        if "</think>" in response:
            response = response.split("</think>")[1]

        # Parse yes/no response
        response_lower = response.strip().lower()
        return response_lower.startswith("yes")

    def verify_feedback_items_batched(
        self, query: str, feedback_items: List[str]
    ) -> List[str]:
        """
        Verify feedback items using batched selection instead of per-item binary.

        More efficient for large lists - asks model to select relevant items
        from a batch rather than checking each individually.

        Args:
            query: The generated query/scenario
            feedback_items: List of atomic feedback items to check

        Returns:
            List of feedback items that apply to the query
        """
        relevant_items = []
        batch_size = self.verification_batch_size

        for i in range(0, len(feedback_items), batch_size):
            batch = feedback_items[i : i + batch_size]
            batch_str = "\n".join([f"{j + 1}. {item}" for j, item in enumerate(batch)])

            prompt = f"""Given this query:
{query}

From the following items, select ONLY the ones that would help answer or respond to this query.
An item is relevant if any part of it (rules, vocabulary, examples, feedback) applies to the query.
Output the numbers of relevant items, comma-separated (e.g., "1, 5, 12").
If none are relevant, output "none".

Items:
{batch_str}

Relevant item numbers:"""

            response = self._query_model(
                prompt,
                self.instruction_model,
                self.instruction_model_api_base,
                self.temperature_verification,
                is_base_model=False,
            )

            if "</think>" in response:
                response = response.split("</think>")[1]

            # Parse response for numbers
            response = response.strip().lower()
            if response == "none" or not response:
                continue

            try:
                # Extract numbers from response
                import re

                numbers = [int(n) for n in re.findall(r"\d+", response)]
                for n in numbers:
                    if 0 < n <= len(batch):
                        relevant_items.append(batch[n - 1])
            except (ValueError, IndexError):
                if self.debug_logs:
                    print(f"Failed to parse verification response: {response}")
                continue

        # Deduplicate verified items (preserving order)
        seen = set()
        unique_items = []
        for item in relevant_items:
            normalized = item.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_items.append(item)
        return unique_items

    def verify_feedback_items(self, query: str, feedback_items: List[str]) -> List[str]:
        """
        Verify which feedback items apply to a query.

        For small lists, uses per-item binary classification (more accurate).
        For large lists (when chunking enabled), uses batched selection (more efficient).

        Args:
            query: The generated query/scenario
            feedback_items: List of atomic feedback items to check

        Returns:
            List of feedback items that apply to the query
        """
        # Use batched verification for large lists when chunking is enabled
        if self.enable_chunking and len(feedback_items) > self.verification_batch_size:
            if self.debug_logs:
                print(f"Using batched verification for {len(feedback_items)} items")
            return self.verify_feedback_items_batched(query, feedback_items)

        # Original per-item verification for smaller lists
        def verify_item(item: str) -> tuple[str, bool]:
            """Verify a single item and return (item, applies)."""
            applies = self.verify_single_feedback_item(query, item)
            return (item, applies)

        # Parallelize verification across all feedback items
        applicable_items = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(feedback_items), self.max_workers)
        ) as executor:
            results = executor.map(verify_item, feedback_items)
            for item, applies in results:
                if applies:
                    applicable_items.append(item)

        return applicable_items

    def generate_single_example(
        self,
        all_feedback: str,
        include_feedback_in_prompt: bool = True,
    ) -> Optional[SyntheticExample]:
        """Generate a single synthetic example."""
        try:
            # Phase 1: Select feedback using base model
            selected_feedback = self.select_feedback(all_feedback)
            if self.debug_logs:
                print(f"\n{'=' * 60}")
                print("DEBUG (selected feedback):")
                print(selected_feedback)
                print(f"{'=' * 60}\n")
            if not selected_feedback:
                return None

            # Phase 2: Generate query using instruction model
            generated_query = self.generate_query(selected_feedback)
            if self.debug_logs:
                print(f"\n{'=' * 60}")
                print("DEBUG (generated query):")
                print(generated_query)
                print(f"{'=' * 60}\n")
            if not generated_query:
                return None

            # Phase 3 (optional): Verify which feedback items actually apply
            # This uses per-item binary classification which is more reliable than
            # asking the LLM to recall all applicable items
            final_feedback = selected_feedback
            if self.verify_feedback:
                # Get decomposed feedback items (cached after first call)
                feedback_items = self.get_decomposed_feedback_items(all_feedback)
                if self.debug_logs:
                    print(f"\n{'=' * 60}")
                    print(
                        f"DEBUG (total decomposed feedback items): {len(feedback_items)}"
                    )
                    print(f"{'=' * 60}\n")

                # Verify each item against the generated query
                verified_items = self.verify_feedback_items(
                    generated_query, feedback_items
                )

                if verified_items:
                    final_feedback = "\n".join([f"- {item}" for item in verified_items])
                else:
                    # No items verified - skip this example
                    return None

            # Build final prompt
            final_prompt = self.build_final_prompt(
                generated_query,
                final_feedback,
                all_feedback,
                include_feedback=include_feedback_in_prompt,
            )

            if not final_prompt:
                return None

            metadata = {
                "generated_query": generated_query,
                "seed_feedback": selected_feedback,
            }
            if self.verify_feedback:
                metadata["verified_feedback"] = final_feedback

            return SyntheticExample(
                prompt=[{"role": "user", "content": final_prompt}],
                selected_feedback=final_feedback,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error generating example: {e}")
            return None

    def generate_dataset(
        self,
        n_examples: int,
        include_feedback_in_prompt: bool = True,
    ) -> List[SyntheticExample]:
        """Generate a dataset of synthetic examples with parallel processing."""
        all_feedback = self.get_feedback()
        print(f"Generating {n_examples} synthetic examples")

        # Pre-compute decomposition if verification is enabled OR chunking is enabled
        # (chunking uses decomposed items for subset sampling in feedback selection)
        if self.verify_feedback or self.enable_chunking:
            self.get_decomposed_feedback_items(all_feedback)

        examples = []

        def generate_one(_):
            return self.generate_single_example(
                all_feedback, include_feedback_in_prompt
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [executor.submit(generate_one, i) for i in range(n_examples)]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=n_examples,
                desc="Generating synthetic data",
            ):
                result = future.result()
                if result is not None:
                    examples.append(result)

        print(f"Successfully generated {len(examples)}/{n_examples} examples")
        return examples

    def to_dataframe(self, examples: List[SyntheticExample]) -> pd.DataFrame:
        """Convert examples to a pandas DataFrame."""
        records = []
        for ex in examples:
            records.append(
                {
                    "prompt": ex.prompt,
                    "selected_feedback": ex.selected_feedback,
                    **ex.metadata,
                }
            )
        return pd.DataFrame(records)

    def save_dataset(
        self,
        examples: List[SyntheticExample],
        output_path: str,
    ):
        """Save dataset to parquet file."""
        df = self.to_dataframe(examples)
        df.to_parquet(output_path)
        print(f"Saved {len(examples)} examples to {output_path}")


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments for synthetic data generation scripts."""
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model for feedback selection (more diverse)",
    )
    parser.add_argument(
        "--instruction_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Instruction model for query generation",
    )
    parser.add_argument(
        "--base_model_api_base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base URL for base model",
    )
    parser.add_argument(
        "--instruction_model_api_base",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="API base URL for instruction model",
    )
    parser.add_argument(
        "--temperature_base",
        type=float,
        default=1.0,
        help="Temperature for base model (higher = more diverse)",
    )
    parser.add_argument(
        "--temperature_instruction",
        type=float,
        default=0.7,
        help="Temperature for instruction model",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (default: disabled)",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=512,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for generated dataset (parquet)",
    )
    parser.add_argument(
        "--include_feedback",
        action="store_true",
        default=True,
        help="Include feedback in the final prompt",
    )
    parser.add_argument(
        "--verify_feedback",
        action="store_true",
        help="Use per-item verification to determine which feedback applies to generated queries",
    )
    parser.add_argument(
        "--temperature_verification",
        type=float,
        default=0.3,
        help="Temperature for feedback verification (lower = more conservative)",
    )
    parser.add_argument(
        "--debug_logs",
        action="store_true",
        help="Enable debug logging",
    )
    # Chunking arguments for large NLF
    parser.add_argument(
        "--enable_chunking",
        action="store_true",
        help="Enable chunking for large NLF that exceeds context limits",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Size of each chunk in tokens (default: 8192)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=512,
        help="Overlap between chunks in tokens (default: 512)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to tokenizer for chunking (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max_items_for_selection",
        type=int,
        default=100,
        help="Maximum items to show for feedback selection when chunking (default: 100)",
    )
    parser.add_argument(
        "--max_tokens_for_selection",
        type=int,
        default=4000,
        help="Maximum tokens budget for feedback in selection prompt (default: 4000)",
    )
    parser.add_argument(
        "--verification_batch_size",
        type=int,
        default=50,
        help="Batch size for batched verification when chunking (default: 50)",
    )
    return parser
