"""Unit tests for Ollama parsing, dataset validation, and aggregate metrics."""

from __future__ import annotations

from fractions import Fraction
from unittest.mock import Mock, patch

import pytest
import requests

from app.inference import OllamaClient
from benchmarks.code_eval import evaluate_python_response
from benchmarks.dataset import load_dataset, validate_dataset
from benchmarks.evaluate import _extract_fraction, summarize_rows
from benchmarks.run_models import run_benchmark


def test_successful_response_parsing() -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "response": "42", "prompt_eval_count": 8, "eval_count": 2,
        "total_duration": 100, "load_duration": 20,
        "prompt_eval_duration": 30, "eval_duration": 40,
    }
    with patch("app.inference.requests.post", return_value=response) as post:
        result = OllamaClient().generate("qwen2.5:1.5b", "What is 6 times 7?")
    assert result.status == "success"
    assert result.response == "42"
    assert result.prompt_tokens == 8
    assert result.generated_tokens == 2
    assert result.eval_duration_ns == 40
    assert post.call_args.kwargs["json"]["stream"] is False
    assert post.call_args.kwargs["json"]["options"]["temperature"] == 0.0


def test_connection_failure_is_retained() -> None:
    with patch("app.inference.requests.post", side_effect=requests.ConnectionError("offline")):
        result = OllamaClient().generate("qwen2.5:1.5b", "prompt")
    assert result.status == "failed"
    assert result.error_type == "connection_error"
    assert "offline" in (result.error_message or "")


def test_timeout_is_retained() -> None:
    with patch("app.inference.requests.post", side_effect=requests.Timeout("slow")):
        result = OllamaClient(timeout=3).generate("qwen2.5:3b", "prompt")
    assert result.status == "failed"
    assert result.error_type == "timeout"


def test_missing_model_error_is_classified() -> None:
    response = Mock()
    response.status_code = 404
    response.reason = "Not Found"
    response.json.return_value = {"error": "model not found"}
    with patch("app.inference.requests.post", return_value=response):
        result = OllamaClient().generate("missing:model", "prompt")
    assert result.status == "failed"
    assert result.error_type == "model_not_found"
    assert result.error_message == "model not found"


def test_dataset_has_thirty_valid_questions() -> None:
    questions = load_dataset()
    assert len(questions) == 30
    assert len({question["id"] for question in questions}) == 30
    assert sum(question["category"] == "mathematical_reasoning" for question in questions) == 15
    assert sum(question["category"] == "python_coding" for question in questions) == 15


def test_dataset_rejects_duplicate_ids() -> None:
    question = {
        "id": "same", "category": "mathematical_reasoning", "prompt": "1 + 1?",
        "evaluation_type": "exact_numeric", "expected_answer": 2,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset([question, question.copy()])


def test_evaluation_metrics_skip_infrastructure_failures() -> None:
    rows = [
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": "true", "response": "4", "wall_clock_seconds": "1", "generated_tokens": "4"},
        {"model": "small", "question_id": "q2", "category": "math", "status": "success", "correct": "false", "response": "3", "wall_clock_seconds": "3", "generated_tokens": "6"},
        {"model": "small", "question_id": "q3", "category": "math", "status": "failed", "correct": "", "wall_clock_seconds": "2", "generated_tokens": ""},
        {"model": "large", "question_id": "q1", "category": "math", "status": "success", "correct": "false", "response": "5", "wall_clock_seconds": "4", "generated_tokens": "8"},
    ]
    summary = summarize_rows(rows)
    small = next(row for row in summary if row["model"] == "small" and row["category"] == "all")
    assert small["accuracy"] == 0.5
    assert small["failure_rate"] == pytest.approx(1 / 3)
    assert small["average_latency_seconds"] == 2
    assert small["median_latency_seconds"] == 2
    assert small["p95_latency_seconds"] == 3
    assert small["average_generated_tokens"] == 5
    assert small["paired_disagreement_rate"] == 1


@pytest.mark.parametrize(
    ("first_correct", "second_correct", "expected_disagreement"),
    [("true", "true", 0), ("false", "false", 0), ("true", "false", 1)],
)
def test_paired_disagreement_compares_correctness_not_response_text(
    first_correct: str, second_correct: str, expected_disagreement: float
) -> None:
    rows = [
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": first_correct, "response": "first model wording"},
        {"model": "large", "question_id": "q1", "category": "math", "status": "success", "correct": second_correct, "response": "completely different wording"},
    ]
    summary = summarize_rows(rows)
    assert summary[0]["paired_disagreement_rate"] == expected_disagreement


def test_paired_disagreement_excludes_infrastructure_failures() -> None:
    rows = [
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": "true", "response": "answer"},
        {"model": "large", "question_id": "q1", "category": "math", "status": "failed", "correct": "false", "response": ""},
    ]
    summary = summarize_rows(rows)
    assert summary[0]["paired_disagreement_rate"] == ""


def test_paired_disagreement_excludes_unevaluated_responses() -> None:
    rows = [
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": "true", "response": "answer"},
        {"model": "large", "question_id": "q1", "category": "math", "status": "success", "correct": "", "response": "not evaluated"},
    ]
    summary = summarize_rows(rows)
    assert summary[0]["paired_disagreement_rate"] == ""


def test_paired_disagreement_requires_two_distinct_models() -> None:
    rows = [
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": "true", "response": "first attempt"},
        {"model": "small", "question_id": "q1", "category": "math", "status": "success", "correct": "false", "response": "duplicate row"},
    ]
    summary = summarize_rows(rows)
    assert summary[0]["paired_disagreement_rate"] == ""


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Final answer: -12", Fraction(-12)),
        ("Result: 3.75", Fraction(15, 4)),
        ("Probability: 2/5", Fraction(2, 5)),
        (
            "There are 12 items, and 3 are selected.\n"
            "The intermediate result is 9.\nFinal answer: 2/5",
            Fraction(2, 5),
        ),
    ],
)
def test_extract_fraction_supported_formats_and_explanation_numbers(
    response: str, expected: Fraction
) -> None:
    assert _extract_fraction(response) == expected


def test_python_answer_runs_in_restricted_child_process() -> None:
    passed, error = evaluate_python_response(
        "def add_numbers(a, b):\n    return a + b",
        {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]},
    )
    assert passed is True
    assert error is None


def test_python_answer_allows_safe_string_normalization() -> None:
    passed, error = evaluate_python_response(
        "def is_palindrome(text):\n    normalized = text.casefold()\n    return normalized == normalized[::-1]",
        {"function_name": "is_palindrome", "cases": [{"args": ["Level"], "expected": True}]},
    )
    assert passed is True
    assert error is None


def test_python_answer_rejects_imports() -> None:
    passed, error = evaluate_python_response(
        "import os\ndef add_numbers(a, b):\n    return a + b",
        {"function_name": "add_numbers", "cases": [{"args": [2, 3], "expected": 5}]},
    )
    assert passed is False
    assert error is not None and "top-level" in error


def test_python_answer_execution_timeout() -> None:
    passed, error = evaluate_python_response(
        "def never_finishes(value):\n    while True:\n        pass",
        {"function_name": "never_finishes", "cases": [{"args": [0], "expected": 0}]},
        timeout_seconds=0.2,
    )
    assert passed is False
    assert error is not None and "timeout" in error.lower()


def test_runner_refuses_to_overwrite_results(tmp_path) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("keep this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_benchmark(Mock(), ["model"], output_path=output)
    assert output.read_text(encoding="utf-8") == "keep this"