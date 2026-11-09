"""Tests for improved coding prompt construction."""

from __future__ import annotations

import pytest

from benchmarks.prompt_builder import (
    build_coding_prompt,
    extract_function_name_from_prompt,
    _infer_signature,
    _has_explicit_signature,
)


def test_build_prompt_adds_function_name_and_signature() -> None:
    """Test that function name and signature are added to generic prompts."""
    question = {
        "prompt": "Write a function to add two numbers.",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [
                {"args": [2, 3], "expected": 5},
                {"args": [5, 7], "expected": 12},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    assert "add_numbers" in enhanced
    assert "def add_numbers(n, m)" in enhanced
    assert "Write a function to add two numbers." in enhanced
    assert "exact signature" in enhanced.lower()


def test_build_prompt_preserves_explicit_signature() -> None:
    """Test that prompts with explicit signature are preserved."""
    question = {
        "prompt": "Write a Python function named is_even(number) that returns True when an integer is even.",
        "test_spec": {
            "function_name": "is_even",
            "cases": [{"args": [4], "expected": True}]
        }
    }

    enhanced = build_coding_prompt(question)
    # Should preserve because signature pattern is present
    assert enhanced == "Write a Python function named is_even(number) that returns True when an integer is even."


def test_build_prompt_handles_missing_test_spec() -> None:
    """Test that questions without test_spec return original prompt."""
    question = {
        "prompt": "Explain the concept of recursion.",
    }

    enhanced = build_coding_prompt(question)
    assert enhanced == "Explain the concept of recursion."


def test_build_prompt_handles_list_parameter() -> None:
    """Test signature inference for list parameters."""
    question = {
        "prompt": "Write a function to reverse a list.",
        "test_spec": {
            "function_name": "reverse_list",
            "cases": [
                {"args": [[1, 2, 3]], "expected": [3, 2, 1]},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    assert "reverse_list" in enhanced
    assert "items" in enhanced  # Should infer "items" for list parameter


def test_build_prompt_handles_string_parameter() -> None:
    """Test signature inference for string parameters."""
    question = {
        "prompt": "Write a function to count vowels.",
        "test_spec": {
            "function_name": "count_vowels",
            "cases": [
                {"args": ["hello"], "expected": 2},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    assert "count_vowels" in enhanced
    assert "text" in enhanced  # Should infer "text" for string parameter


def test_build_prompt_handles_multiple_parameters() -> None:
    """Test signature inference for multiple parameters."""
    question = {
        "prompt": "Write a function to check divisibility.",
        "test_spec": {
            "function_name": "is_divisible",
            "cases": [
                {"args": [10, 5], "expected": True},
                {"args": [7, 3], "expected": False},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    assert "is_divisible" in enhanced
    assert "is_divisible(n, m)" in enhanced


def test_infer_signature_no_args() -> None:
    """Test signature inference for zero-argument function."""
    signature = _infer_signature("get_value", [])
    assert signature == "get_value()"


def test_infer_signature_single_arg() -> None:
    """Test signature inference for single argument."""
    signature = _infer_signature("double", [{"args": [5], "expected": 10}])
    assert signature == "double(n)"


def test_infer_signature_multiple_args() -> None:
    """Test signature inference for multiple arguments."""
    signature = _infer_signature("add", [{"args": [2, 3], "expected": 5}])
    assert signature == "add(n, m)"


def test_infer_signature_list_arg() -> None:
    """Test signature inference recognizes lists."""
    signature = _infer_signature("process", [{"args": [[1, 2, 3]], "expected": 6}])
    assert signature == "process(items)"


def test_infer_signature_string_arg() -> None:
    """Test signature inference recognizes strings."""
    signature = _infer_signature("reverse", [{"args": ["hello"], "expected": "olleh"}])
    assert signature == "reverse(text)"


def test_extract_function_name_from_named_pattern() -> None:
    """Test extracting function name from 'function named X' pattern."""
    prompt = "Write a Python function named add_numbers(a, b) that returns the sum."
    name = extract_function_name_from_prompt(prompt)
    assert name == "add_numbers"


def test_extract_function_name_from_def_pattern() -> None:
    """Test extracting function name from 'def X(' pattern."""
    prompt = "Implement: def calculate_sum(a, b):"
    name = extract_function_name_from_prompt(prompt)
    assert name == "calculate_sum"


def test_extract_function_name_not_found() -> None:
    """Test that None is returned when no function name found."""
    prompt = "Write a function to add two numbers."
    name = extract_function_name_from_prompt(prompt)
    assert name is None


def test_build_prompt_realistic_mbpp() -> None:
    """Test with realistic MBPP-style question."""
    question = {
        "prompt": "Write a function to find the largest subset where each pair is divisible.\nReturn Python code defining the required function.",
        "test_spec": {
            "function_name": "largest_subset",
            "description": "Find largest divisible subset",
            "cases": [
                {"args": [[1, 3, 6, 13, 17, 18], 6], "expected": 4},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    assert "largest_subset" in enhanced
    assert "exact signature" in enhanced.lower()
    assert "must be named 'largest_subset' exactly" in enhanced.lower()


def test_build_prompt_mentions_name_but_no_signature() -> None:
    """Test that prompts mentioning function name without signature get enhanced."""
    question = {
        "prompt": "Write a function called add_numbers that adds two integers.",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [
                {"args": [2, 3], "expected": 5},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    # Should enhance because signature is not explicitly provided
    assert "add_numbers" in enhanced
    assert "def add_numbers(n, m)" in enhanced
    assert "exact signature" in enhanced.lower()


def test_build_prompt_has_exact_def_signature() -> None:
    """Test that prompts with exact 'def func(params):' are preserved."""
    question = {
        "prompt": "Implement the function:\ndef add_numbers(a, b):\n    pass",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [
                {"args": [2, 3], "expected": 5},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    # Should preserve because exact signature is present
    assert enhanced == "Implement the function:\ndef add_numbers(a, b):\n    pass"


def test_build_prompt_has_function_named_with_parens() -> None:
    """Test that 'function named X(' pattern is recognized."""
    question = {
        "prompt": "Write a function named calculate_sum(a, b) that returns the sum.",
        "test_spec": {
            "function_name": "calculate_sum",
            "cases": [
                {"args": [2, 3], "expected": 5},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    # Should preserve because signature pattern is present
    assert enhanced == "Write a function named calculate_sum(a, b) that returns the sum."


def test_build_prompt_wrong_name_in_prompt() -> None:
    """Test that prompt with different function name gets enhanced with correct one."""
    question = {
        "prompt": "Write a function called my_add that adds two numbers.",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [
                {"args": [2, 3], "expected": 5},
            ]
        }
    }

    enhanced = build_coding_prompt(question)
    # Should enhance because the signature doesn't match expected function
    assert "add_numbers" in enhanced
    assert "def add_numbers(n, m)" in enhanced


def test_has_explicit_signature_detects_def_pattern() -> None:
    """Test that _has_explicit_signature detects def pattern."""
    assert _has_explicit_signature("def func_name(a, b):", "func_name") is True
    assert _has_explicit_signature("def other_name(a, b):", "func_name") is False


def test_has_explicit_signature_detects_function_named_pattern() -> None:
    """Test that _has_explicit_signature detects 'function named' pattern."""
    assert _has_explicit_signature("function named func_name(a, b)", "func_name") is True
    assert _has_explicit_signature("function called func_name that does X", "func_name") is False


def test_build_prompt_preserves_original_task_wording() -> None:
    """Test that original task wording is preserved when enhancing."""
    question = {
        "prompt": "Write a Python function to calculate factorial of n.",
        "test_spec": {
            "function_name": "factorial",
            "cases": [
                {"args": [5], "expected": 120},
            ]
        }
    }

    enhanced = build_coding_prompt(question)

    # Original wording preserved
    assert "Write a Python function to calculate factorial of n." in enhanced
    # Enhancement added
    assert "Implement a Python function with this exact signature:" in enhanced
    assert "def factorial(n)" in enhanced
    assert "must be named 'factorial' exactly" in enhanced
