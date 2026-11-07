"""Loading and structural validation for AURA benchmark questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmark_questions.json"
VALID_EVALUATION_TYPES = {"exact_numeric", "python_tests"}


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load a JSON dataset and reject malformed or duplicate questions."""
    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            questions = json.load(dataset_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load benchmark dataset at {path}: {exc}") from exc
    validate_dataset(questions)
    return questions


def validate_dataset(questions: Any) -> None:
    """Validate required fields and evaluation-specific test specifications."""
    if not isinstance(questions, list) or not questions:
        raise ValueError("dataset must be a non-empty list")

    seen_ids: set[str] = set()
    for index, question in enumerate(questions):
        location = f"question at index {index}"
        if not isinstance(question, dict):
            raise ValueError(f"{location} must be an object")
        for field in ("id", "category", "prompt", "evaluation_type"):
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"{location} requires a non-empty '{field}'")
        if question["id"] in seen_ids:
            raise ValueError(f"duplicate question id: {question['id']}")
        seen_ids.add(question["id"])

        evaluation_type = question["evaluation_type"]
        if evaluation_type not in VALID_EVALUATION_TYPES:
            raise ValueError(f"unsupported evaluation type: {evaluation_type}")
        if evaluation_type == "exact_numeric":
            if not isinstance(question.get("expected_answer"), (str, int, float)):
                raise ValueError(f"{location} requires an expected_answer")
        else:
            _validate_test_spec(question.get("test_spec"), location)


def _validate_test_spec(spec: Any, location: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{location} requires a test_spec object")
    if not isinstance(spec.get("function_name"), str) or not spec["function_name"].isidentifier():
        raise ValueError(f"{location} test_spec requires a valid function_name")
    if not isinstance(spec.get("description"), str) or not spec["description"].strip():
        raise ValueError(f"{location} test_spec requires a description")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{location} test_spec requires at least one case")
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("args"), list)
            or "expected" not in case
        ):
            raise ValueError(f"{location} contains a malformed test case")