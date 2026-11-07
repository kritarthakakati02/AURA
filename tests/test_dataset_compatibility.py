"""Compatibility tests for legacy and expanded benchmark question IDs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.inference import InferenceResult
from benchmarks.dataset import load_dataset, validate_dataset
from benchmarks.run_models import run_benchmark


def _legacy_math_question() -> dict:
    return {
        "id": "legacy_math_01",
        "category": "mathematical_reasoning",
        "prompt": "What is 6 times 7? Give the final numeric answer on the last line.",
        "expected_answer": 42,
        "evaluation_type": "exact_numeric",
    }


def _expanded_python_question() -> dict:
    return {
        "question_id": "mbpp_0901",
        "category": "python_coding",
        "prompt": "Write add_one(value) to return its argument plus one.",
        "test_spec": {
            "function_name": "add_one",
            "description": "Return the input value plus one.",
            "cases": [{"args": [1], "expected": 2}],
        },
        "evaluation_type": "python_tests",
        "source": {"name": "MBPP"},
        "source_question_id": "901",
        "split": "train",
    }


def _successful_result(model: str, response: str) -> InferenceResult:
    return InferenceResult(
        model=model,
        response=response,
        wall_clock_seconds=0.1,
        prompt_tokens=4,
        generated_tokens=3,
        total_duration_ns=10,
        load_duration_ns=2,
        prompt_eval_duration_ns=3,
        eval_duration_ns=4,
        status="success",
    )


def test_loads_original_dataset_and_preserves_legacy_id() -> None:
    questions = load_dataset()
    assert len(questions) == 30
    assert questions[0]["question_id"] == questions[0]["id"]
    assert questions[0]["id"] == "math_01"


def test_loads_expanded_question_id_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "expanded.json"
    expanded = _expanded_python_question()
    dataset_path.write_text(json.dumps([expanded]), encoding="utf-8")

    questions = load_dataset(dataset_path)

    assert questions[0]["question_id"] == "mbpp_0901"
    assert "id" not in questions[0]
    assert questions[0]["test_spec"] == expanded["test_spec"]


def test_rejects_duplicate_ids_across_both_schemas() -> None:
    legacy = _legacy_math_question()
    expanded = {
        **_expanded_python_question(),
        "question_id": legacy["id"],
    }
    with pytest.raises(ValueError, match="duplicate question id"):
        validate_dataset([legacy, expanded])


def test_mocked_run_writes_canonical_ids_for_both_schemas(tmp_path: Path) -> None:
    dataset_path = tmp_path / "mixed.json"
    questions = [_legacy_math_question(), _expanded_python_question()]
    dataset_path.write_text(json.dumps(questions), encoding="utf-8")
    output_path = tmp_path / "results.csv"
    client = Mock()
    client.generate.side_effect = [
        _successful_result("qwen2.5:1.5b", "42"),
        _successful_result(
            "qwen2.5:1.5b",
            "def add_one(value):\n    return value + 1",
        ),
    ]

    run_benchmark(
        client,
        ["qwen2.5:1.5b"],
        dataset_path=dataset_path,
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as result_file:
        rows = list(csv.DictReader(result_file))
    assert [row["question_id"] for row in rows] == ["legacy_math_01", "mbpp_0901"]
    assert [row["correct"] for row in rows] == ["True", "True"]
    assert [row["evaluation_type"] for row in rows] == ["exact_numeric", "python_tests"]