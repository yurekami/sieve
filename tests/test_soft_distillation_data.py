"""
Tests for sieve/soft_distillation_data.py

Tests cover:
1. replace_feedback_in_prompt: feedback replacement in prompts
2. extract_empirical_distribution: logprobs extraction and normalization
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from sieve.soft_distillation_data import (
    replace_feedback_in_prompt,
    extract_empirical_distribution,
)
from sieve.synthetic_data_gen import FEEDBACK_START_DELIMITER, FEEDBACK_END_DELIMITER


class TestReplaceFeedbackInPrompt:
    """Tests for replace_feedback_in_prompt function."""

    def test_basic_replacement(self):
        """Test basic feedback replacement in a single user message."""
        prompt = [
            {
                "role": "user",
                "content": f"Here is the feedback:\n{FEEDBACK_START_DELIMITER}\nOld feedback here\n{FEEDBACK_END_DELIMITER}\nNow answer the question.",
            }
        ]
        new_feedback = "New feedback content"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        expected_content = f"Here is the feedback:\n{FEEDBACK_START_DELIMITER}\n{new_feedback}\n{FEEDBACK_END_DELIMITER}\nNow answer the question."
        assert result[0]["content"] == expected_content

    def test_only_user_messages_modified(self):
        """Test that only user messages are modified, assistant messages remain unchanged."""
        prompt = [
            {
                "role": "user",
                "content": f"Question: {FEEDBACK_START_DELIMITER}\nOld feedback\n{FEEDBACK_END_DELIMITER}",
            },
            {
                "role": "assistant",
                "content": f"Answer with {FEEDBACK_START_DELIMITER}\nShould not change\n{FEEDBACK_END_DELIMITER}",
            },
            {
                "role": "user",
                "content": f"Follow-up: {FEEDBACK_START_DELIMITER}\nOld feedback 2\n{FEEDBACK_END_DELIMITER}",
            },
        ]
        new_feedback = "New feedback"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        assert len(result) == 3
        # User messages should have new feedback
        assert new_feedback in result[0]["content"]
        assert new_feedback in result[2]["content"]
        # Assistant message should be unchanged
        assert "Should not change" in result[1]["content"]
        assert "New feedback" not in result[1]["content"]

    def test_multiple_messages_in_prompt(self):
        """Test replacement across multiple user messages."""
        prompt = [
            {
                "role": "user",
                "content": f"First: {FEEDBACK_START_DELIMITER}\nFeedback 1\n{FEEDBACK_END_DELIMITER}",
            },
            {"role": "assistant", "content": "Response 1"},
            {
                "role": "user",
                "content": f"Second: {FEEDBACK_START_DELIMITER}\nFeedback 2\n{FEEDBACK_END_DELIMITER}",
            },
        ]
        new_feedback = "Unified feedback"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        assert len(result) == 3
        assert new_feedback in result[0]["content"]
        assert new_feedback in result[2]["content"]
        assert "Feedback 1" not in result[0]["content"]
        assert "Feedback 2" not in result[2]["content"]

    def test_no_delimiters_no_change(self):
        """Test that prompts without delimiters are unchanged."""
        prompt = [
            {"role": "user", "content": "This is a simple prompt without delimiters."},
            {"role": "assistant", "content": "This is a response."},
        ]
        new_feedback = "New feedback"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        assert len(result) == 2
        assert result[0]["content"] == prompt[0]["content"]
        assert result[1]["content"] == prompt[1]["content"]

    def test_immutability(self):
        """Test that input prompt is not mutated (immutability check)."""
        original_prompt = [
            {
                "role": "user",
                "content": f"Test: {FEEDBACK_START_DELIMITER}\nOld feedback\n{FEEDBACK_END_DELIMITER}",
            }
        ]
        original_content = original_prompt[0]["content"]
        new_feedback = "New feedback"

        result = replace_feedback_in_prompt(original_prompt, new_feedback)

        # Original prompt should be unchanged
        assert original_prompt[0]["content"] == original_content
        assert "New feedback" not in original_prompt[0]["content"]
        # Result should have new feedback
        assert "New feedback" in result[0]["content"]
        # They should be different objects
        assert result is not original_prompt
        assert result[0] is not original_prompt[0]

    def test_multiline_feedback_replacement(self):
        """Test replacement with multiline feedback content."""
        prompt = [
            {
                "role": "user",
                "content": f"Instructions:\n{FEEDBACK_START_DELIMITER}\nLine 1\nLine 2\nLine 3\n{FEEDBACK_END_DELIMITER}\nEnd.",
            }
        ]
        new_feedback = "New line 1\nNew line 2\nNew line 3"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        assert "New line 1" in result[0]["content"]
        assert "New line 2" in result[0]["content"]
        assert "New line 3" in result[0]["content"]
        assert "Line 1" not in result[0]["content"]

    def test_empty_feedback_section(self):
        """Test replacement when feedback section is empty."""
        prompt = [
            {
                "role": "user",
                "content": f"Text before {FEEDBACK_START_DELIMITER}{FEEDBACK_END_DELIMITER} text after",
            }
        ]
        new_feedback = "New feedback"

        result = replace_feedback_in_prompt(prompt, new_feedback)

        expected_content = f"Text before {FEEDBACK_START_DELIMITER}\n{new_feedback}\n{FEEDBACK_END_DELIMITER} text after"
        assert result[0]["content"] == expected_content


class TestExtractEmpiricalDistribution:
    """Tests for extract_empirical_distribution function."""

    def test_returns_none_when_logprobs_none(self):
        """Test that None is returned when logprobs_data is None."""
        result = extract_empirical_distribution(None, k=100)
        assert result is None

    def test_returns_none_when_content_empty(self):
        """Test that None is returned when logprobs_data.content is empty."""
        mock_logprobs = MagicMock()
        mock_logprobs.content = None
        result = extract_empirical_distribution(mock_logprobs, k=100)
        assert result is None

        mock_logprobs.content = []
        result = extract_empirical_distribution(mock_logprobs, k=100)
        assert result is None

    def test_extracts_tokens_and_probabilities(self):
        """Test correct extraction of tokens, token_ids, and probabilities from mock logprobs."""
        # Create mock logprobs structure
        mock_logprobs = MagicMock()

        # Create mock top_logprobs for 2 tokens
        mock_top_logprob_1 = MagicMock()
        mock_top_logprob_1.token = "token_id:100"
        mock_top_logprob_1.logprob = np.log(0.8)

        mock_top_logprob_2 = MagicMock()
        mock_top_logprob_2.token = "token_id:200"
        mock_top_logprob_2.logprob = np.log(0.2)

        # Create token_logprob with top_logprobs
        mock_token_logprob_1 = MagicMock()
        mock_token_logprob_1.top_logprobs = [mock_top_logprob_1, mock_top_logprob_2]

        mock_token_logprob_2 = MagicMock()
        mock_top_logprob_3 = MagicMock()
        mock_top_logprob_3.token = "token_id:300"
        mock_top_logprob_3.logprob = np.log(0.6)

        mock_top_logprob_4 = MagicMock()
        mock_top_logprob_4.token = "token_id:400"
        mock_top_logprob_4.logprob = np.log(0.4)

        mock_token_logprob_2.top_logprobs = [mock_top_logprob_3, mock_top_logprob_4]

        mock_logprobs.content = [mock_token_logprob_1, mock_token_logprob_2]

        # Create mock tokenizer to enable token_ids extraction
        mock_tokenizer = MagicMock()
        mock_tokenizer.unk_token_id = 0

        result = extract_empirical_distribution(mock_logprobs, k=100, tokenizer=mock_tokenizer)

        assert result is not None
        assert len(result) == 2

        # Check first token distribution
        assert result[0]["tokens"] == ["token_id:100", "token_id:200"]
        assert result[0]["token_ids"] == [100, 200]
        # Probabilities should be renormalized to sum to 1
        assert abs(sum(result[0]["probabilities"]) - 1.0) < 1e-6
        # Original probabilities were 0.8 and 0.2, so they should stay the same after renorm
        assert abs(result[0]["probabilities"][0] - 0.8) < 1e-6
        assert abs(result[0]["probabilities"][1] - 0.2) < 1e-6

        # Check second token distribution
        assert result[1]["tokens"] == ["token_id:300", "token_id:400"]
        assert result[1]["token_ids"] == [300, 400]
        assert abs(sum(result[1]["probabilities"]) - 1.0) < 1e-6
        assert abs(result[1]["probabilities"][0] - 0.6) < 1e-6
        assert abs(result[1]["probabilities"][1] - 0.4) < 1e-6

    def test_handles_token_id_format(self):
        """Test parsing of 'token_id:{id}' format (from vLLM's return_tokens_as_token_ids)."""
        mock_logprobs = MagicMock()

        mock_top_logprob = MagicMock()
        mock_top_logprob.token = "token_id:12345"
        mock_top_logprob.logprob = np.log(1.0)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob]

        mock_logprobs.content = [mock_token_logprob]

        # Create mock tokenizer to enable token_ids extraction
        mock_tokenizer = MagicMock()
        mock_tokenizer.unk_token_id = 0

        result = extract_empirical_distribution(mock_logprobs, k=100, tokenizer=mock_tokenizer)

        assert result is not None
        assert len(result) == 1
        assert result[0]["token_ids"] == [12345]
        assert result[0]["tokens"] == ["token_id:12345"]

    def test_renormalizes_probabilities(self):
        """Test that top-k probabilities are renormalized to sum to 1.0."""
        mock_logprobs = MagicMock()

        # Create probabilities that don't sum to 1 (simulating top-k truncation)
        mock_top_logprob_1 = MagicMock()
        mock_top_logprob_1.token = "token_id:1"
        mock_top_logprob_1.logprob = np.log(0.3)  # Original: 0.3

        mock_top_logprob_2 = MagicMock()
        mock_top_logprob_2.token = "token_id:2"
        mock_top_logprob_2.logprob = np.log(0.2)  # Original: 0.2

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob_1, mock_top_logprob_2]

        mock_logprobs.content = [mock_token_logprob]

        result = extract_empirical_distribution(mock_logprobs, k=100)

        assert result is not None
        # Probabilities should sum to 1 after renormalization
        assert abs(sum(result[0]["probabilities"]) - 1.0) < 1e-6
        # Original sum was 0.5, so renormalized should be 0.3/0.5=0.6 and 0.2/0.5=0.4
        assert abs(result[0]["probabilities"][0] - 0.6) < 1e-6
        assert abs(result[0]["probabilities"][1] - 0.4) < 1e-6
        # original_mass should store the original sum
        assert abs(result[0]["original_mass"] - 0.5) < 1e-6

    def test_respects_k_parameter(self):
        """Test that k parameter limits the number of top tokens extracted."""
        mock_logprobs = MagicMock()

        # Create 10 top_logprobs
        top_logprobs = []
        for i in range(10):
            mock_top_logprob = MagicMock()
            mock_top_logprob.token = f"token_id:{i}"
            mock_top_logprob.logprob = np.log(0.1)
            top_logprobs.append(mock_top_logprob)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = top_logprobs

        mock_logprobs.content = [mock_token_logprob]

        # Create mock tokenizer to enable token_ids extraction
        mock_tokenizer = MagicMock()
        mock_tokenizer.unk_token_id = 0

        # Extract with k=5
        result = extract_empirical_distribution(mock_logprobs, k=5, tokenizer=mock_tokenizer)

        assert result is not None
        assert len(result) == 1
        # Should only have 5 tokens
        assert len(result[0]["tokens"]) == 5
        assert len(result[0]["token_ids"]) == 5
        assert len(result[0]["probabilities"]) == 5

    def test_with_mock_tokenizer(self):
        """Test extraction with a mock tokenizer for fallback token ID lookup."""
        mock_logprobs = MagicMock()

        # Token without token_id: format (fallback to tokenizer)
        mock_top_logprob_1 = MagicMock()
        mock_top_logprob_1.token = "hello"
        mock_top_logprob_1.logprob = np.log(0.7)

        # Token with token_id: format (direct parsing)
        mock_top_logprob_2 = MagicMock()
        mock_top_logprob_2.token = "token_id:999"
        mock_top_logprob_2.logprob = np.log(0.3)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob_1, mock_top_logprob_2]

        mock_logprobs.content = [mock_token_logprob]

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids = MagicMock(return_value=42)
        mock_tokenizer.unk_token_id = 0

        result = extract_empirical_distribution(
            mock_logprobs, k=100, tokenizer=mock_tokenizer, unk_token_id=0
        )

        assert result is not None
        assert len(result) == 1
        # First token should use convert_tokens_to_ids (returns 42)
        # Second token should parse token_id: directly (999)
        assert result[0]["token_ids"] == [42, 999]
        assert result[0]["tokens"] == ["hello", "token_id:999"]

    def test_tokenizer_encode_fallback(self):
        """Test fallback to tokenizer.encode when convert_tokens_to_ids returns unk."""
        mock_logprobs = MagicMock()

        mock_top_logprob = MagicMock()
        mock_top_logprob.token = "rare_token"
        mock_top_logprob.logprob = np.log(1.0)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob]

        mock_logprobs.content = [mock_token_logprob]

        # Mock tokenizer that returns unk for convert_tokens_to_ids
        mock_tokenizer = MagicMock()
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = MagicMock(return_value=0)  # Returns unk
        mock_tokenizer.encode = MagicMock(return_value=[777])  # Fallback encode returns 777

        result = extract_empirical_distribution(
            mock_logprobs, k=100, tokenizer=mock_tokenizer, unk_token_id=0
        )

        assert result is not None
        assert len(result) == 1
        # Should use encode fallback result
        assert result[0]["token_ids"] == [777]
        mock_tokenizer.encode.assert_called_once_with(
            "rare_token", add_special_tokens=False
        )

    def test_multi_token_encoding_fallback(self):
        """Test that multi-token encoding falls back to unk_token_id."""
        mock_logprobs = MagicMock()

        mock_top_logprob = MagicMock()
        mock_top_logprob.token = "multitoken"
        mock_top_logprob.logprob = np.log(1.0)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob]

        mock_logprobs.content = [mock_token_logprob]

        # Mock tokenizer that returns multi-token encoding
        mock_tokenizer = MagicMock()
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = MagicMock(return_value=0)  # Returns unk
        mock_tokenizer.encode = MagicMock(return_value=[111, 222, 333])  # Multi-token

        unk_token_id = 999

        result = extract_empirical_distribution(
            mock_logprobs, k=100, tokenizer=mock_tokenizer, unk_token_id=unk_token_id
        )

        assert result is not None
        assert len(result) == 1
        # Should use unk_token_id for multi-token encoding
        assert result[0]["token_ids"] == [unk_token_id]

    def test_empty_top_logprobs(self):
        """Test handling of empty top_logprobs lists."""
        mock_logprobs = MagicMock()

        # Token with no top_logprobs
        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = []

        mock_logprobs.content = [mock_token_logprob]

        result = extract_empirical_distribution(mock_logprobs, k=100)

        # When top_logprobs is empty (falsy), the token is skipped entirely
        # So the result should be an empty list, not a list with one empty entry
        assert result is not None
        assert len(result) == 0

    def test_original_mass_stored(self):
        """Test that original probability mass is stored before renormalization."""
        mock_logprobs = MagicMock()

        mock_top_logprob_1 = MagicMock()
        mock_top_logprob_1.token = "token_id:1"
        mock_top_logprob_1.logprob = np.log(0.25)

        mock_top_logprob_2 = MagicMock()
        mock_top_logprob_2.token = "token_id:2"
        mock_top_logprob_2.logprob = np.log(0.15)

        mock_token_logprob = MagicMock()
        mock_token_logprob.top_logprobs = [mock_top_logprob_1, mock_top_logprob_2]

        mock_logprobs.content = [mock_token_logprob]

        result = extract_empirical_distribution(mock_logprobs, k=100)

        assert result is not None
        # Original mass should be 0.25 + 0.15 = 0.4
        assert abs(result[0]["original_mass"] - 0.4) < 1e-6
        # But probabilities should be renormalized to sum to 1
        assert abs(sum(result[0]["probabilities"]) - 1.0) < 1e-6
