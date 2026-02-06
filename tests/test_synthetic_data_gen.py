"""
Tests for sieve/synthetic_data_gen.py

Tests cover:
- SyntheticExample dataclass
- Feedback delimiters
- Deduplication logic
- Prompt generation (instruction vs base model modes)
- Full generation pipeline with mocked LLM calls
- DataFrame conversion
- CLI argument parsing
"""

import argparse
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd

from sieve.synthetic_data_gen import (
    SyntheticDataGenerator,
    SyntheticExample,
    FEEDBACK_START_DELIMITER,
    FEEDBACK_END_DELIMITER,
    add_common_args,
)


# ============= Test Subclass =============


class ConcreteTestDataGenerator(SyntheticDataGenerator):
    """Concrete test subclass with minimal implementation."""

    def __init__(self, feedback_text="Test feedback line 1\nTest feedback line 2", **kwargs):
        super().__init__(**kwargs)
        self.feedback_text = feedback_text

    def get_feedback(self):
        return self.feedback_text

    def build_final_prompt(self, generated_query, selected_feedback, all_feedback, include_feedback=True):
        if include_feedback:
            return f"Query: {generated_query}\nFeedback: {selected_feedback}"
        return f"Query: {generated_query}"


# ============= Tests =============


class TestSyntheticExample:
    """Tests for SyntheticExample dataclass."""

    def test_creation(self):
        """Test creating a SyntheticExample with all fields."""
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        selected_feedback = "Use step-by-step reasoning"
        metadata = {"source": "test", "iteration": 1}

        example = SyntheticExample(
            prompt=prompt,
            selected_feedback=selected_feedback,
            metadata=metadata,
        )

        assert example.prompt == prompt
        assert example.selected_feedback == selected_feedback
        assert example.metadata == metadata

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        example = SyntheticExample(
            prompt=[{"role": "user", "content": "test"}],
            selected_feedback="feedback",
        )

        assert example.metadata == {}


class TestFeedbackDelimiters:
    """Tests for feedback delimiter constants."""

    def test_delimiter_values(self):
        """Test that delimiters are correct strings."""
        assert FEEDBACK_START_DELIMITER == "<|feedback_start|>"
        assert FEEDBACK_END_DELIMITER == "<|feedback_end|>"


class TestDeduplication:
    """Tests for _deduplicate_items method."""

    def test_removes_duplicates(self):
        """Test that case-insensitive duplicates are removed."""
        gen = ConcreteTestDataGenerator()
        items = [
            "First item",
            "Second item",
            "FIRST ITEM",  # Duplicate
            "third item",
            "Second Item",  # Duplicate
        ]

        result = gen._deduplicate_items(items)

        assert len(result) == 3
        assert "First item" in result
        assert "Second item" in result
        assert "third item" in result

    def test_preserves_first_occurrence(self):
        """Test that first occurrence casing is preserved."""
        gen = ConcreteTestDataGenerator()
        items = [
            "First Item",  # Original casing
            "FIRST ITEM",  # Duplicate
            "first item",  # Duplicate
        ]

        result = gen._deduplicate_items(items)

        assert len(result) == 1
        assert result[0] == "First Item"  # Original casing preserved

    def test_strips_whitespace(self):
        """Test that whitespace differences don't prevent deduplication."""
        gen = ConcreteTestDataGenerator()
        items = [
            "  Test item  ",
            "Test item",
            "TEST ITEM   ",
        ]

        result = gen._deduplicate_items(items)

        assert len(result) == 1

    def test_empty_list(self):
        """Test deduplication of empty list."""
        gen = ConcreteTestDataGenerator()
        result = gen._deduplicate_items([])
        assert result == []


class TestFeedbackSelectionPrompt:
    """Tests for get_feedback_selection_prompt method."""

    def test_instruction_model_mode(self):
        """Test prompt generation for instruction model (use_base_model=False)."""
        gen = ConcreteTestDataGenerator(use_base_model=False)
        feedback = "Rule 1: Do X\nRule 2: Do Y"

        prompt = gen.get_feedback_selection_prompt(feedback)

        assert "You are helping generate diverse training data" in prompt
        assert feedback in prompt
        assert "Select a subset of guidelines" in prompt
        assert "Selected guidelines:" in prompt
        # Instruction mode should NOT end with "-"
        assert not prompt.strip().endswith("-")

    def test_base_model_mode(self):
        """Test prompt generation for base model (use_base_model=True)."""
        gen = ConcreteTestDataGenerator(use_base_model=True)
        feedback = "Rule 1: Do X\nRule 2: Do Y"

        prompt = gen.get_feedback_selection_prompt(feedback)

        assert "Task: Select 3-5 guidelines" in prompt
        assert feedback in prompt
        assert "Guidelines:" in prompt
        assert "Selected guidelines:\n-" in prompt.strip()
        # Base model mode should end with "-" to guide list output
        assert prompt.strip().endswith("-")

    def test_includes_feedback_text(self):
        """Test that provided feedback is included in prompt."""
        gen = ConcreteTestDataGenerator()
        feedback = "Unique feedback text 12345"

        prompt = gen.get_feedback_selection_prompt(feedback)

        assert feedback in prompt


class TestQueryGenerationPrompt:
    """Tests for get_query_generation_prompt method."""

    def test_includes_selected_feedback(self):
        """Test that selected feedback is included in the prompt."""
        gen = ConcreteTestDataGenerator()
        selected_feedback = "Rule: Always verify inputs\nExample: Check email format"

        prompt = gen.get_query_generation_prompt(selected_feedback)

        assert selected_feedback in prompt
        assert "Generate a realistic question" in prompt
        assert "Question:" in prompt

    def test_prompt_structure(self):
        """Test the overall prompt structure."""
        gen = ConcreteTestDataGenerator()
        selected_feedback = "Test feedback"

        prompt = gen.get_query_generation_prompt(selected_feedback)

        assert "feedback/guidelines/knowledge would apply" in prompt
        assert "Instructions:" in prompt
        assert "Output ONLY the question" in prompt


class TestGenerateSingleExample:
    """Tests for generate_single_example method with mocked LLM calls."""

    @patch("sieve.synthetic_data_gen.litellm.completion")
    def test_happy_path(self, mock_completion):
        """Test successful example generation."""
        # Mock LLM responses
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.content = "- Selected feedback item"

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.content = "What is the discount for item X?"

        mock_completion.side_effect = [mock_response_1, mock_response_2]

        gen = ConcreteTestDataGenerator()
        result = gen.generate_single_example("All feedback here", include_feedback_in_prompt=True)

        assert result is not None
        assert isinstance(result, SyntheticExample)
        assert result.prompt[0]["role"] == "user"
        assert "What is the discount for item X?" in result.prompt[0]["content"]
        assert "Selected feedback item" in result.prompt[0]["content"]
        assert result.selected_feedback == "- Selected feedback item"

    @patch("sieve.synthetic_data_gen.litellm.completion")
    def test_empty_feedback_selection(self, mock_completion):
        """Test that empty feedback selection returns None."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""

        mock_completion.return_value = mock_response

        gen = ConcreteTestDataGenerator()
        result = gen.generate_single_example("All feedback here")

        assert result is None

    @patch("sieve.synthetic_data_gen.litellm.completion")
    def test_empty_query_generation(self, mock_completion):
        """Test that empty query generation returns None."""
        # First call returns feedback, second call returns empty query
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.content = "- Valid feedback"

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.content = ""

        mock_completion.side_effect = [mock_response_1, mock_response_2]

        gen = ConcreteTestDataGenerator()
        result = gen.generate_single_example("All feedback here")

        assert result is None

    @patch("sieve.synthetic_data_gen.litellm.completion")
    def test_without_feedback_in_prompt(self, mock_completion):
        """Test generation without feedback in final prompt."""
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.content = "- Selected feedback"

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.content = "Generated query"

        mock_completion.side_effect = [mock_response_1, mock_response_2]

        gen = ConcreteTestDataGenerator()
        result = gen.generate_single_example("All feedback", include_feedback_in_prompt=False)

        assert result is not None
        # build_final_prompt should exclude feedback
        assert "Feedback:" not in result.prompt[0]["content"]
        assert "Query: Generated query" in result.prompt[0]["content"]

    @patch("sieve.synthetic_data_gen.litellm.completion")
    def test_metadata_fields(self, mock_completion):
        """Test that metadata contains expected fields."""
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock()]
        mock_response_1.choices[0].message.content = "- Feedback text"

        mock_response_2 = Mock()
        mock_response_2.choices = [Mock()]
        mock_response_2.choices[0].message.content = "Query text"

        mock_completion.side_effect = [mock_response_1, mock_response_2]

        gen = ConcreteTestDataGenerator()
        result = gen.generate_single_example("All feedback")

        assert result is not None
        assert "generated_query" in result.metadata
        assert "seed_feedback" in result.metadata
        assert result.metadata["generated_query"] == "Query text"
        assert result.metadata["seed_feedback"] == "- Feedback text"


class TestToDataFrame:
    """Tests for to_dataframe method."""

    def test_converts_to_dataframe(self):
        """Test conversion of examples to DataFrame."""
        examples = [
            SyntheticExample(
                prompt=[{"role": "user", "content": "Query 1"}],
                selected_feedback="Feedback 1",
                metadata={"iteration": 1},
            ),
            SyntheticExample(
                prompt=[{"role": "user", "content": "Query 2"}],
                selected_feedback="Feedback 2",
                metadata={"iteration": 2},
            ),
        ]

        gen = ConcreteTestDataGenerator()
        df = gen.to_dataframe(examples)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "prompt" in df.columns
        assert "selected_feedback" in df.columns
        assert "iteration" in df.columns

    def test_empty_list(self):
        """Test conversion of empty list."""
        gen = ConcreteTestDataGenerator()
        df = gen.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_metadata_expansion(self):
        """Test that metadata is expanded into columns."""
        examples = [
            SyntheticExample(
                prompt=[{"role": "user", "content": "Query"}],
                selected_feedback="Feedback",
                metadata={"key1": "value1", "key2": "value2"},
            ),
        ]

        gen = ConcreteTestDataGenerator()
        df = gen.to_dataframe(examples)

        assert "key1" in df.columns
        assert "key2" in df.columns
        assert df.iloc[0]["key1"] == "value1"
        assert df.iloc[0]["key2"] == "value2"


class TestAddCommonArgs:
    """Tests for add_common_args CLI argument parser."""

    def test_parser_has_expected_arguments(self):
        """Test that parser includes all expected arguments."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args([
            "--output_path", "test.parquet",
        ])

        # Check core arguments exist
        assert hasattr(args, "base_model")
        assert hasattr(args, "instruction_model")
        assert hasattr(args, "base_model_api_base")
        assert hasattr(args, "instruction_model_api_base")
        assert hasattr(args, "temperature_base")
        assert hasattr(args, "temperature_instruction")
        assert hasattr(args, "enable_thinking")
        assert hasattr(args, "n_examples")
        assert hasattr(args, "max_workers")
        assert hasattr(args, "output_path")
        assert hasattr(args, "include_feedback")
        assert hasattr(args, "verify_feedback")
        assert hasattr(args, "temperature_verification")
        assert hasattr(args, "debug_logs")

    def test_parser_has_chunking_arguments(self):
        """Test that parser includes chunking-related arguments."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args([
            "--output_path", "test.parquet",
            "--enable_chunking",
        ])

        assert hasattr(args, "enable_chunking")
        assert hasattr(args, "chunk_size")
        assert hasattr(args, "chunk_overlap")
        assert hasattr(args, "tokenizer_path")
        assert hasattr(args, "max_items_for_selection")
        assert hasattr(args, "max_tokens_for_selection")
        assert hasattr(args, "verification_batch_size")
        assert args.enable_chunking is True

    def test_default_values(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        args = parser.parse_args([
            "--output_path", "test.parquet",
        ])

        assert args.base_model == "Qwen/Qwen3-8B"
        assert args.instruction_model == "Qwen/Qwen3-8B"
        assert args.temperature_base == 1.0
        assert args.temperature_instruction == 0.7
        assert args.n_examples == 512
        assert args.max_workers == 12
        assert args.chunk_size == 8192
        assert args.chunk_overlap == 512
        assert args.verification_batch_size == 50

    def test_output_path_required(self):
        """Test that output_path is a required argument."""
        parser = argparse.ArgumentParser()
        add_common_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestChunkFeedback:
    """Tests for chunk_feedback method."""

    def test_chunking_disabled_returns_single_chunk(self):
        """Test that chunking disabled returns entire feedback as single chunk."""
        gen = ConcreteTestDataGenerator(enable_chunking=False)
        feedback = "This is test feedback\n" * 100

        result = gen.chunk_feedback(feedback)

        assert len(result) == 1
        assert result[0] == feedback

    @patch("transformers.AutoTokenizer")
    def test_small_feedback_returns_single_chunk(self, mock_tokenizer_cls):
        """Test that small feedback (< chunk_size) returns single chunk."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tokenizer.decode.return_value = "Small feedback"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        gen = ConcreteTestDataGenerator(enable_chunking=True, chunk_size=100)
        result = gen.chunk_feedback("Small feedback")

        assert len(result) == 1

    @patch("transformers.AutoTokenizer")
    def test_large_feedback_splits_into_chunks(self, mock_tokenizer_cls):
        """Test that large feedback is split into multiple chunks."""
        # Simulate 1000 tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(1000))
        mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: f"chunk_{len(tokens)}"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        gen = ConcreteTestDataGenerator(
            enable_chunking=True,
            chunk_size=300,
            chunk_overlap=50,
        )
        result = gen.chunk_feedback("Large feedback text")

        # 1000 tokens / (300 - 50 overlap) = ~4 chunks
        assert len(result) > 1


class TestInitialization:
    """Tests for SyntheticDataGenerator initialization."""

    def test_stores_all_config_params(self):
        """Test that __init__ stores all configuration parameters."""
        gen = ConcreteTestDataGenerator(
            base_model="custom-base",
            instruction_model="custom-instruct",
            base_model_api_base="http://localhost:9000",
            instruction_model_api_base="http://localhost:9001",
            temperature_base=0.9,
            temperature_instruction=0.5,
            enable_thinking=True,
            max_workers=8,
            use_base_model=True,
            verify_feedback=True,
            temperature_verification=0.2,
            debug_logs=True,
            enable_chunking=True,
            chunk_size=4096,
            chunk_overlap=256,
            tokenizer_path="custom-tokenizer",
            max_items_for_selection=50,
            max_tokens_for_selection=2000,
            verification_batch_size=25,
        )

        assert gen.base_model == "custom-base"
        assert gen.instruction_model == "custom-instruct"
        assert gen.base_model_api_base == "http://localhost:9000"
        assert gen.instruction_model_api_base == "http://localhost:9001"
        assert gen.temperature_base == 0.9
        assert gen.temperature_instruction == 0.5
        assert gen.enable_thinking is True
        assert gen.max_workers == 8
        assert gen.use_base_model is True
        assert gen.verify_feedback is True
        assert gen.temperature_verification == 0.2
        assert gen.debug_logs is True
        assert gen.enable_chunking is True
        assert gen.chunk_size == 4096
        assert gen.chunk_overlap == 256
        assert gen.tokenizer_path == "custom-tokenizer"
        assert gen.max_items_for_selection == 50
        assert gen.max_tokens_for_selection == 2000
        assert gen.verification_batch_size == 25

    def test_default_values(self):
        """Test that default values are set correctly."""
        gen = ConcreteTestDataGenerator()

        assert gen.base_model == "Qwen/Qwen3-8B-Base"
        assert gen.instruction_model == "Qwen/Qwen3-8B"
        assert gen.temperature_base == 1.0
        assert gen.temperature_instruction == 0.7
        assert gen.enable_thinking is False
        assert gen.max_workers == 12
        assert gen.use_base_model is False
        assert gen.verify_feedback is False
        assert gen.enable_chunking is False
        assert gen._tokenizer is None
        assert gen._decomposed_feedback_items is None
