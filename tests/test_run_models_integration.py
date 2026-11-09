"""Tests for run_models.py integration with prompt builder."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from benchmarks.run_models import build_result_row, run_benchmark
from benchmarks.prompt_builder import build_coding_prompt
from benchmarks.dataset import canonical_question_id
from app.inference import InferenceResult


def test_build_result_row_stores_original_prompt() -> None:
    """Test that the original prompt is stored separately from inference prompt."""
    question = {
        "question_id": "test_01",
        "category": "python_coding",
        "evaluation_type": "python_tests",
        "prompt": "Write a function to add two numbers.",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [{"args": [2, 3], "expected": 5}]
        }
    }

    # The enhanced prompt that was sent to Ollama
    inference_prompt = build_coding_prompt(question)

    result = Mock()
    result.status = "success"
    result.response = "def add_numbers(a, b):\n    return a + b"
    result.to_dict.return_value = {
        "model": "test-model",
        "response": "def add_numbers(a, b):\n    return a + b",
        "status": "success",
        "prompt_tokens": 10,
        "generated_tokens": 20,
        "total_duration_ns": 1000,
        "load_duration_ns": 100,
        "prompt_eval_duration_ns": 200,
        "eval_duration_ns": 300,
    }

    row = build_result_row("test-model", question, inference_prompt, result, 2.0)

    # The stored prompt should be the ORIGINAL question prompt
    assert row["prompt"] == "Write a function to add two numbers."
    # The inference_prompt should be the enhanced version
    assert row["inference_prompt"] == inference_prompt
    assert "exact signature" in row["inference_prompt"].lower()
    assert "def add_numbers" in row["inference_prompt"]


def test_coding_prompt_enhancement() -> None:
    """Test that coding tasks get enhanced prompts."""
    coding_question = {
        "prompt": "Write a function to add two numbers.",
        "evaluation_type": "python_tests",
        "test_spec": {
            "function_name": "add_numbers",
            "cases": [{"args": [2, 3], "expected": 5}]
        }
    }

    enhanced_coding = build_coding_prompt(coding_question)

    # Coding should be enhanced
    assert "exact signature" in enhanced_coding.lower()
    assert "def add_numbers" in enhanced_coding


def test_math_prompt_unchanged() -> None:
    """Test that math tasks do NOT get enhanced."""
    math_question = {
        "prompt": "What is 2 + 2?",
        "evaluation_type": "exact_numeric",
        "expected_answer": 4
    }

    enhanced_math = build_coding_prompt(math_question)

    # Math should NOT be enhanced
    assert enhanced_math == "What is 2 + 2?"


def test_prompt_builder_preserves_original_prompt() -> None:
    """Test that the original task wording is preserved in enhanced prompt."""
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


def test_question_ids_filter() -> None:
    """Test that question_ids parameter filters dataset correctly."""
    questions = [
        {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Add two numbers.", "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
        {"question_id": "code_02", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Reverse string.", "test_spec": {"function_name": "reverse_string", "cases": [{"args": ["abc"], "expected": "cba"}]}},
        {"question_id": "math_01", "category": "mathematical_reasoning", "evaluation_type": "exact_numeric",
         "prompt": "What is 2+2?", "expected_answer": 4},
    ]

    # Filter by IDs
    question_ids = ["code_01", "math_01"]
    filtered = [q for q in questions if canonical_question_id(q) in set(question_ids)]

    assert len(filtered) == 2
    assert {canonical_question_id(q) for q in filtered} == {"code_01", "math_01"}


def test_question_ids_empty_raises() -> None:
    """Test that empty question_ids list raises error."""
    mock_client = Mock()

    with pytest.raises(ValueError, match="question_ids list must not be empty"):
        run_benchmark(
            client=mock_client,
            models=["test-model"],
            question_ids=[],
            overwrite=True,
        )


def test_question_ids_no_match_raises() -> None:
    """Test that non-matching question_ids raises error with specific IDs."""
    mock_client = Mock()

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = [
            {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
             "prompt": "Add two numbers.", "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
        ]

        with pytest.raises(ValueError, match="Question IDs not found in dataset: nonexistent_id"):
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                question_ids=["nonexistent_id"],
                overwrite=True,
            )


def test_question_ids_mixed_valid_invalid_raises() -> None:
    """Test that mixed valid and invalid IDs raises error listing unmatched IDs."""
    mock_client = Mock()

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = [
            {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
             "prompt": "Add two numbers.", "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
            {"question_id": "code_02", "category": "python_coding", "evaluation_type": "python_tests",
             "prompt": "Reverse string.", "test_spec": {"function_name": "reverse_string", "cases": [{"args": ["abc"], "expected": "cba"}]}},
        ]

        with pytest.raises(ValueError, match="Question IDs not found in dataset:.*invalid_id.*nonexistent"):
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                question_ids=["code_01", "invalid_id", "nonexistent"],
                overwrite=True,
            )


def test_run_benchmark_filters_by_question_ids() -> None:
    """Test that run_benchmark uses question_ids filter correctly."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="def add_numbers(a, b):\n    return a + b",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Add two numbers.", "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
        {"question_id": "code_02", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Reverse string.", "test_spec": {"function_name": "reverse_string", "cases": [{"args": ["abc"], "expected": "cba"}]}},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                question_ids=["code_01"],
                overwrite=True,
                code_timeout=2.0,
            )

            # Verify output
            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should only have 1 row (1 model × 1 filtered question)
            assert len(rows) == 1
            assert rows[0]["question_id"] == "code_01"

        finally:
            if output_path.exists():
                output_path.unlink()


def test_run_benchmark_preserves_original_question_ids() -> None:
    """Test that retry preserves original question IDs in output."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="def largest_subset(a, n):\n    return 4",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "mbpp_0620", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Test prompt.", "test_spec": {"function_name": "largest_subset", "cases": [{"args": [[1, 2], 2], "expected": 2}]}},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                question_ids=["mbpp_0620"],
                overwrite=True,
                code_timeout=2.0,
            )

            # Verify output has correct question_id
            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["question_id"] == "mbpp_0620"
            assert rows[0]["model"] == "test-model"

        finally:
            if output_path.exists():
                output_path.unlink()


def test_run_benchmark_default_behavior_unchanged() -> None:
    """Test that without question_ids, all questions are processed."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="42",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "math_01", "category": "mathematical_reasoning", "evaluation_type": "exact_numeric",
         "prompt": "What is 2+2?", "expected_answer": 4},
        {"question_id": "math_02", "category": "mathematical_reasoning", "evaluation_type": "exact_numeric",
         "prompt": "What is 3+3?", "expected_answer": 6},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                overwrite=True,
                code_timeout=2.0,
            )

            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have 2 rows (1 model × 2 questions)
            assert len(rows) == 2
            question_ids = {row["question_id"] for row in rows}
            assert question_ids == {"math_01", "math_02"}

        finally:
            if output_path.exists():
                output_path.unlink()


def test_enhanced_prompt_sent_to_model() -> None:
    """Test that enhanced prompt is actually sent to the model for coding tasks."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="def add_numbers(a, b):\n    return a + b",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Write a function to add two numbers.",
         "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                overwrite=True,
                code_timeout=2.0,
            )

            # Verify that generate was called with enhanced prompt
            assert mock_client.generate.call_count == 1
            call_args = mock_client.generate.call_args
            sent_prompt = call_args[0][1]  # Second argument to generate()

            # Enhanced prompt should include signature
            assert "def add_numbers" in sent_prompt
            assert "exact signature" in sent_prompt.lower()

        finally:
            if output_path.exists():
                output_path.unlink()


def test_math_prompt_sent_unchanged() -> None:
    """Test that math prompts are sent unchanged to the model."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="4",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "math_01", "category": "mathematical_reasoning", "evaluation_type": "exact_numeric",
         "prompt": "What is 2+2?", "expected_answer": 4},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                overwrite=True,
                code_timeout=2.0,
            )

            # Verify that generate was called with original prompt
            assert mock_client.generate.call_count == 1
            call_args = mock_client.generate.call_args
            sent_prompt = call_args[0][1]

            # Should be original prompt, not enhanced
            assert sent_prompt == "What is 2+2?"

            # Verify CSV has both columns with same value for math
            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["prompt"] == "What is 2+2?"
            assert rows[0]["inference_prompt"] == "What is 2+2?"

        finally:
            if output_path.exists():
                output_path.unlink()


def test_coding_prompt_inference_prompt_different() -> None:
    """Test that coding tasks have different prompt and inference_prompt columns."""
    mock_client = Mock()
    mock_client.generate.return_value = InferenceResult(
        model="test-model",
        response="def add_numbers(a, b):\n    return a + b",
        wall_clock_seconds=1.5,
        status="success",
        prompt_tokens=10,
        generated_tokens=20,
        total_duration_ns=1000,
        load_duration_ns=100,
        prompt_eval_duration_ns=200,
        eval_duration_ns=300,
    )

    questions = [
        {"question_id": "code_01", "category": "python_coding", "evaluation_type": "python_tests",
         "prompt": "Write a function to add two numbers.",
         "test_spec": {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]}},
    ]

    with patch("benchmarks.run_models.load_dataset") as mock_load:
        mock_load.return_value = questions

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            run_benchmark(
                client=mock_client,
                models=["test-model"],
                output_path=output_path,
                overwrite=True,
                code_timeout=2.0,
            )

            # Verify CSV has different values for coding task
            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            # Original prompt
            assert rows[0]["prompt"] == "Write a function to add two numbers."
            # Enhanced prompt
            assert "Write a function to add two numbers." in rows[0]["inference_prompt"]
            assert "exact signature" in rows[0]["inference_prompt"].lower()
            assert "def add_numbers" in rows[0]["inference_prompt"]
            # They should be different
            assert rows[0]["prompt"] != rows[0]["inference_prompt"]

        finally:
            if output_path.exists():
                output_path.unlink()
