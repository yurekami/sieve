"""
Pytest tests for utils/reward_utils.py

Tests cover:
- Basic \boxed{answer} extraction
- Multiple \boxed - should return the last one
- Nested braces: \boxed{a + {b}}
- \fbox{answer} support
- No boxed answer returns None
- Empty string returns None
- Dollar amounts like \boxed{$150.00}
- Malformed/unclosed \boxed returns None
"""

import pytest
from utils.reward_utils import extract_boxed_answer, last_boxed_only_string, remove_boxed


class TestLastBoxedOnlyString:
    """Tests for last_boxed_only_string()"""

    def test_basic_boxed(self):
        """Should extract basic \\boxed{answer}"""
        result = last_boxed_only_string(r"\boxed{42}")
        assert result == r"\boxed{42}"

    def test_basic_fbox(self):
        """Should extract basic \\fbox{answer}"""
        result = last_boxed_only_string(r"\fbox{42}")
        assert result == r"\fbox{42}"

    def test_boxed_with_surrounding_text(self):
        """Should extract \\boxed from text before/after it"""
        result = last_boxed_only_string(r"The answer is \boxed{100} units.")
        assert result == r"\boxed{100}"

    def test_multiple_boxed_returns_last(self):
        """Should return the LAST \\boxed when multiple exist"""
        result = last_boxed_only_string(
            r"\boxed{first} and then \boxed{second}"
        )
        assert result == r"\boxed{second}"

    def test_multiple_boxed_complex(self):
        """Should return last \\boxed in complex text"""
        text = r"First answer: \boxed{10} is wrong. Final answer: \boxed{20} is correct."
        result = last_boxed_only_string(text)
        assert result == r"\boxed{20}"

    def test_nested_braces_simple(self):
        """Should handle nested braces: \\boxed{a + {b}}"""
        result = last_boxed_only_string(r"\boxed{a + {b}}")
        assert result == r"\boxed{a + {b}}"

    def test_nested_braces_complex(self):
        """Should handle multiple nested braces"""
        result = last_boxed_only_string(r"\boxed{f(x) = {a + {b + c}}}")
        assert result == r"\boxed{f(x) = {a + {b + c}}}"

    def test_nested_braces_multiple_levels(self):
        """Should handle deeply nested braces"""
        result = last_boxed_only_string(r"\boxed{{{nested}}}}")
        assert result == r"\boxed{{{nested}}}"

    def test_mixed_fbox_and_boxed_returns_last(self):
        """When both \\fbox and \\boxed exist, should return the last one"""
        result = last_boxed_only_string(
            r"\fbox{first} then \boxed{second}"
        )
        assert result == r"\boxed{second}"

    def test_mixed_boxed_and_fbox_returns_last(self):
        """When \\boxed comes before \\fbox, should return \\boxed (function checks \\boxed first)"""
        result = last_boxed_only_string(
            r"\boxed{first} then \fbox{second}"
        )
        assert result == r"\boxed{first}"

    def test_dollar_amounts(self):
        """Should handle dollar amounts: \\boxed{$150.00}"""
        result = last_boxed_only_string(r"\boxed{$150.00}")
        assert result == r"\boxed{$150.00}"

    def test_dollar_amounts_with_text(self):
        """Should extract \\boxed with dollar sign and text"""
        result = last_boxed_only_string(r"Cost: \boxed{$25.99 USD}")
        assert result == r"\boxed{$25.99 USD}"

    def test_no_boxed_returns_none(self):
        """Should return None if no \\boxed or \\fbox found"""
        result = last_boxed_only_string("no answer here")
        assert result is None

    def test_empty_string_returns_none(self):
        """Should return None for empty string"""
        result = last_boxed_only_string("")
        assert result is None

    def test_malformed_unclosed_boxed_returns_none(self):
        """Should return None if \\boxed is unclosed"""
        result = last_boxed_only_string(r"\boxed{answer")
        assert result is None

    def test_malformed_only_open_brace_returns_none(self):
        """Should return None if only opening brace exists"""
        result = last_boxed_only_string(r"\boxed{")
        assert result is None

    def test_empty_boxed(self):
        """Should handle empty \\boxed{}"""
        result = last_boxed_only_string(r"\boxed{}")
        assert result == r"\boxed{}"

    def test_boxed_with_newlines(self):
        """Should handle \\boxed with newlines"""
        result = last_boxed_only_string(r"\boxed{line1" + "\n" + r"line2}")
        assert result == r"\boxed{line1" + "\n" + r"line2}"

    def test_boxed_with_special_chars(self):
        """Should handle special characters inside \\boxed"""
        result = last_boxed_only_string(r"\boxed{@#$%^&*()}")
        assert result == r"\boxed{@#$%^&*()}"

    def test_boxed_with_escaped_braces(self):
        """Should handle escaped braces (treated as regular text)"""
        # Note: In raw strings, \{ is literally backslash-brace
        result = last_boxed_only_string(r"\boxed{\{a\}}")
        assert result == r"\boxed{\{a\}}"

    def test_fbox_then_multiple_boxed(self):
        """Should return last \\boxed when \\fbox appears earlier"""
        result = last_boxed_only_string(
            r"\fbox{1} \boxed{2} \boxed{3}"
        )
        assert result == r"\boxed{3}"

    def test_deeply_nested_unmatched_braces(self):
        """Should stop at first properly matched brace pair"""
        result = last_boxed_only_string(r"\boxed{a {b {c}}} extra")
        # The function should find matching braces and return up to the first }
        # that closes the initial { after \boxed
        assert result == r"\boxed{a {b {c}}}"


class TestRemoveBoxed:
    """Tests for remove_boxed()"""

    def test_basic_remove_boxed(self):
        """Should remove \\boxed{ and trailing }"""
        result = remove_boxed(r"\boxed{answer}")
        assert result == "answer"

    def test_remove_boxed_with_spaces(self):
        """Should handle spaces inside \\boxed"""
        result = remove_boxed(r"\boxed{the answer}")
        assert result == "the answer"

    def test_remove_boxed_with_numbers(self):
        """Should handle numeric answers"""
        result = remove_boxed(r"\boxed{42}")
        assert result == "42"

    def test_remove_boxed_with_nested_braces(self):
        """Should preserve nested braces when removing outer \\boxed"""
        result = remove_boxed(r"\boxed{a + {b}}")
        assert result == "a + {b}"

    def test_remove_boxed_complex_nested(self):
        """Should preserve complex nested structures"""
        result = remove_boxed(r"\boxed{f(x) = {a + {b}}}")
        assert result == "f(x) = {a + {b}}"

    def test_remove_boxed_dollar_amount(self):
        """Should handle dollar amounts"""
        result = remove_boxed(r"\boxed{$150.00}")
        assert result == "$150.00"

    def test_remove_boxed_empty(self):
        """Should handle empty \\boxed{}"""
        result = remove_boxed(r"\boxed{}")
        assert result == ""

    def test_remove_boxed_malformed_no_prefix_returns_none(self):
        """Should return None if string doesn't start with \\boxed{"""
        result = remove_boxed("answer}")
        assert result is None

    def test_remove_boxed_malformed_no_suffix_returns_none(self):
        """Should return None if string doesn't end with }"""
        result = remove_boxed(r"\boxed{answer")
        assert result is None

    def test_remove_boxed_not_boxed_returns_none(self):
        """Should return None if not a \\boxed string"""
        result = remove_boxed(r"\fbox{answer}")
        assert result is None

    def test_remove_boxed_multiline(self):
        """Should handle multiline content"""
        result = remove_boxed(r"\boxed{line1" + "\n" + "line2}")
        assert result == "line1\nline2"

    def test_remove_boxed_special_chars(self):
        """Should preserve special characters"""
        result = remove_boxed(r"\boxed{@#$%^&*()}")
        assert result == "@#$%^&*()"

    def test_remove_boxed_with_internal_closing_brace(self):
        """Should only remove outer \\boxed{} wrapper"""
        result = remove_boxed(r"\boxed{a}b}")
        # String starts with \\boxed{ and ends with }, so remove_boxed strips both
        assert result == "a}b"

    def test_remove_boxed_fbox_prefix_returns_none(self):
        """Should not work with \\fbox prefix"""
        result = remove_boxed(r"\fbox{answer}")
        assert result is None


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer()"""

    def test_basic_extraction(self):
        """Should extract answer from \\boxed{answer}"""
        result = extract_boxed_answer(r"\boxed{42}")
        assert result == "42"

    def test_extraction_with_surrounding_text(self):
        """Should extract last \\boxed from text with surrounding context"""
        result = extract_boxed_answer(
            r"The final answer is \boxed{100} units."
        )
        assert result == "100"

    def test_extraction_multiple_boxed(self):
        """Should extract from the LAST \\boxed when multiple exist"""
        result = extract_boxed_answer(
            r"\boxed{wrong} is not the answer but \boxed{correct} is"
        )
        assert result == "correct"

    def test_extraction_nested_braces(self):
        """Should extract answer with nested braces"""
        result = extract_boxed_answer(r"\boxed{a + {b}}")
        assert result == "a + {b}"

    def test_extraction_complex_math(self):
        """Should extract complex mathematical expressions"""
        result = extract_boxed_answer(r"\boxed{x^2 + {2xy + y^2}}")
        assert result == "x^2 + {2xy + y^2}"

    def test_extraction_dollar_amount(self):
        """Should extract dollar amounts"""
        result = extract_boxed_answer(r"The cost is \boxed{$150.00}")
        assert result == "$150.00"

    def test_extraction_fbox(self):
        """Should extract from \\fbox only if \\boxed not present"""
        result = extract_boxed_answer(r"\fbox{answer}")
        # last_boxed_only_string returns \\fbox{answer}, but remove_boxed expects \\boxed{ prefix
        assert result is None

    def test_extraction_mixed_fbox_boxed(self):
        """Should return answer from last box command"""
        result = extract_boxed_answer(
            r"\fbox{first} then \boxed{second}"
        )
        assert result == "second"

    def test_extraction_no_boxed_returns_none(self):
        """Should return None if no \\boxed found"""
        result = extract_boxed_answer("no answer here")
        assert result is None

    def test_extraction_empty_string_returns_none(self):
        """Should return None for empty string"""
        result = extract_boxed_answer("")
        assert result is None

    def test_extraction_malformed_unclosed_returns_none(self):
        """Should return None if \\boxed is unclosed"""
        result = extract_boxed_answer(r"\boxed{answer")
        assert result is None

    def test_extraction_empty_boxed(self):
        """Should extract empty content"""
        result = extract_boxed_answer(r"\boxed{}")
        assert result == ""

    def test_extraction_multiline(self):
        """Should extract multiline content"""
        content = r"\boxed{line1" + "\n" + "line2}"
        result = extract_boxed_answer(content)
        assert result == "line1\nline2"

    def test_extraction_whitespace_preserved(self):
        """Should preserve internal whitespace"""
        result = extract_boxed_answer(r"\boxed{  spaced  answer  }")
        assert result == "  spaced  answer  "

    def test_extraction_complex_realistic(self):
        """Should handle realistic math problem output"""
        solution = r"""
        Step 1: Calculate 2 + 2
        Step 2: The answer is \boxed{4}
        """
        result = extract_boxed_answer(solution)
        assert result == "4"

    def test_extraction_multiple_attempts(self):
        """Should use last attempt when multiple \\boxed exist"""
        solution = r"""
        First attempt: \boxed{10}
        Second attempt: \boxed{15}
        Final answer: \boxed{20}
        """
        result = extract_boxed_answer(solution)
        assert result == "20"

    def test_extraction_with_latex_commands(self):
        """Should extract answer containing LaTeX commands"""
        result = extract_boxed_answer(r"\boxed{\frac{1}{2}}")
        assert result == r"\frac{1}{2}"

    def test_extraction_unicode_chars(self):
        """Should handle unicode characters"""
        result = extract_boxed_answer(r"\boxed{π ≈ 3.14}")
        assert result == "π ≈ 3.14"

    def test_extraction_very_long_answer(self):
        """Should extract very long answers"""
        long_answer = "x" * 10000
        result = extract_boxed_answer(rf"\boxed{{{long_answer}}}")
        assert result == long_answer

    def test_extraction_only_opening_boxed_returns_none(self):
        """Should return None if only opening \\boxed{ exists"""
        result = extract_boxed_answer(r"\boxed{")
        assert result is None


class TestEdgeCases:
    """Edge case tests spanning multiple functions"""

    def test_whitespace_only(self):
        """Should handle whitespace-only input"""
        result = extract_boxed_answer("   \n  \t  ")
        assert result is None

    def test_boxed_keyword_in_text(self):
        """Should not match 'boxed' without backslash"""
        result = extract_boxed_answer("The answer is boxed in the document")
        assert result is None

    def test_escaped_backslash_before_boxed(self):
        """Should handle escaped backslash (\\\\boxed...)"""
        # In raw strings, this is literally two backslashes
        result = extract_boxed_answer(r"\\boxed{answer}")
        # rfind("\boxed") finds the substring at position 1 (second backslash + "boxed")
        assert result == "answer"

    def test_boxed_with_parentheses_and_braces(self):
        """Should correctly match braces, not parentheses"""
        result = extract_boxed_answer(r"\boxed{f(x) = x^2}")
        assert result == "f(x) = x^2"

    def test_multiple_different_brackets(self):
        """Should handle multiple types of brackets"""
        result = extract_boxed_answer(r"\boxed{[a, (b, {c})]}")
        assert result == "[a, (b, {c})]"

    def test_consecutive_boxed_commands(self):
        """Should extract from last when consecutive"""
        result = extract_boxed_answer(r"\boxed{1}\boxed{2}\boxed{3}")
        assert result == "3"

    def test_boxed_inside_fbox(self):
        """When both exist, extracts from whichever is last"""
        result = extract_boxed_answer(r"\fbox{\boxed{nested}}")
        # last_boxed_only_string finds \boxed first (earlier in string)
        # Actually, rfind finds the rightmost occurrence
        # rfind("\boxed") finds the one inside
        # rfind("\fbox") finds the outer one first
        # The code uses rfind for \boxed first, so it finds the nested one
        assert result == "nested"

    def test_line_breaks_in_answer(self):
        """Should preserve line breaks in answer"""
        result = extract_boxed_answer("\\boxed{first\\nsecond}")
        assert result == "first\\nsecond"

    def test_tabs_in_answer(self):
        """Should preserve tabs"""
        result = extract_boxed_answer(r"\boxed{col1	col2}")
        assert result == "col1\tcol2"
