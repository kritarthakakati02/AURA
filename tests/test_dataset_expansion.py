"""Tests for expansion dataset validation and deterministic split preparation."""

from __future__ import annotations

import copy
import pytest

from benchmarks.prepare_dataset import (
    CODE_CATEGORY,
    MATH_CATEGORY,
    SPLIT_CATEGORIES,
    SPLITS,
    _split_records,
    is_near_duplicate,
)
from benchmarks.validate_dataset import validate_expanded_dataset


def _alpha_label(value: int) -> str:
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(ord("a") + remainder) + output
    return output


def _source(category: str) -> dict[str, str]:
    if category == MATH_CATEGORY:
        return {
            "name": "GSM8K", "url": "https://github.com/openai/grade-school-math",
            "license": "MIT", "license_url": "https://github.com/openai/grade-school-math/blob/master/LICENSE",
            "source_commit": "abc123", "transformation": "Extract final answer.",
        }
    return {
        "name": "MBPP", "url": "https://github.com/google-research/google-research/tree/master/mbpp",
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_commit": "def456", "transformation": "Convert literal tests.",
        "upstream_repository_license": "Apache-2.0",
    }


def _valid_dataset() -> tuple[list[dict], dict[str, dict]]:
    questions = []
    manifests = {}
    for split, category_counts in SPLIT_CATEGORIES.items():
        split_ids = []
        for category, count in category_counts.items():
            for index in range(count):
                question_id = f"{category}_{split}_{_alpha_label(index + len(questions) + 1)}"
                prompt = f"Solve this distinct prompt alpha {_alpha_label(len(questions) + 1)}."
                question = {
                    "question_id": question_id,
                    "category": category,
                    "prompt": prompt,
                    "evaluation_type": "exact_numeric" if category == MATH_CATEGORY else "python_tests",
                    "source": _source(category),
                    "source_question_id": str(index + 1),
                    "split": split,
                }
                if category == MATH_CATEGORY:
                    question["expected_answer"] = "42"
                else:
                    question["test_spec"] = {
                        "function_name": "answer", "description": "Return a value.",
                        "cases": [{"args": [1], "expected": 1}],
                    }
                questions.append(question)
                split_ids.append(question_id)
        manifests[split] = {
            "split": split,
            "count": len(split_ids),
            "category_counts": category_counts.copy(),
            "question_ids": split_ids,
        }
    return questions, manifests


def test_valid_400_question_dataset_and_balanced_splits() -> None:
    questions, manifests = _valid_dataset()
    report = validate_expanded_dataset(questions, manifests)
    assert report["total"] == 400
    assert report["category_counts"] == {CODE_CATEGORY: 200, MATH_CATEGORY: 200}
    assert report["split_counts"]["train"]["total"] == 240
    assert report["split_counts"]["validation"]["total"] == 80
    assert report["split_counts"]["test"]["total"] == 80


def test_split_generation_is_reproducible_and_disjoint() -> None:
    records = [
        {"question_id": f"{category}_{index}", "category": category, "prompt": f"unique {category} task {_alpha_label(index + 1)}"}
        for category in (MATH_CATEGORY, CODE_CATEGORY)
        for index in range(200)
    ]
    first, manifest = _split_records(records, seed=91)
    second, second_manifest = _split_records(records, seed=91)
    assert first == second
    assert manifest == second_manifest
    all_ids = [record["question_id"] for rows in first.values() for record in rows]
    assert len(all_ids) == len(set(all_ids)) == 400
    assert {name: len(rows) for name, rows in first.items()} == SPLITS


def test_duplicate_ids_are_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[1]["question_id"] = questions[0]["question_id"]
    with pytest.raises(ValueError, match="duplicate question_id"):
        validate_expanded_dataset(questions, manifests)


def test_invalid_math_answer_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[0]["expected_answer"] = "approximately forty two"
    with pytest.raises(ValueError, match="invalid exact numeric answer"):
        validate_expanded_dataset(questions, manifests)


def test_missing_required_field_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[0].pop("prompt")
    with pytest.raises(ValueError, match="requires non-empty 'prompt'"):
        validate_expanded_dataset(questions, manifests)


def test_invalid_evaluation_type_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[0]["evaluation_type"] = "llm_judge"
    with pytest.raises(ValueError, match="must use exact_numeric"):
        validate_expanded_dataset(questions, manifests)


def test_malformed_coding_test_spec_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    coding = next(item for item in questions if item["category"] == CODE_CATEGORY)
    coding["test_spec"]["cases"][0].pop("expected")
    with pytest.raises(ValueError, match="malformed case"):
        validate_expanded_dataset(questions, manifests)


def test_missing_license_provenance_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[0]["source"].pop("license_url")
    with pytest.raises(ValueError, match="provenance missing"):
        validate_expanded_dataset(questions, manifests)


def test_split_leakage_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    leaked_id = manifests["train"]["question_ids"][0]
    manifests["test"]["question_ids"][0] = leaked_id
    with pytest.raises(ValueError, match="question leakage"):
        validate_expanded_dataset(questions, manifests)


def test_unbalanced_split_size_is_rejected() -> None:
    questions, manifests = _valid_dataset()
    questions[0]["split"] = "validation"
    with pytest.raises(ValueError, match="split 'train' expected"):
        validate_expanded_dataset(questions, manifests)


def test_near_duplicate_detector_normalizes_numbers() -> None:
    assert is_near_duplicate("Add 3 items to the box", "Add 8 items to the box")
    assert not is_near_duplicate("Return the sorted values", "Compute the matrix determinant")


def test_near_duplicate_dataset_prompts_are_rejected() -> None:
    questions, manifests = _valid_dataset()
    first_math = next(item for item in questions if item["category"] == MATH_CATEGORY)
    second_math = next(item for item in questions if item["category"] == MATH_CATEGORY and item is not first_math)
    first_math["prompt"] = "A farmer sold 12 apples at the market and earned 24 dollars. What was the price per apple?"
    second_math["prompt"] = "The farmer sold 18 apples at the market and earned 36 dollars. What was the price per apple?"
    with pytest.raises(ValueError, match="near-duplicate prompts"):
        validate_expanded_dataset(questions, manifests)


def test_near_duplicates_are_checked_across_categories() -> None:
    questions, manifests = _valid_dataset()
    math_question = next(item for item in questions if item["category"] == MATH_CATEGORY)
    code_question = next(item for item in questions if item["category"] == CODE_CATEGORY)
    math_question["prompt"] = "A farmer sold 12 apples at the market and earned 24 dollars. What was the price per apple?"
    code_question["prompt"] = "The farmer sold 18 apples at the market and earned 36 dollars. What was the price per apple?"
    with pytest.raises(ValueError, match="near-duplicate prompts"):
        validate_expanded_dataset(questions, manifests)


def test_test_split_manifest_is_required_to_match_question_level_ids() -> None:
    questions, manifests = _valid_dataset()
    altered = copy.deepcopy(manifests)
    altered["validation"]["question_ids"].pop()
    with pytest.raises(ValueError, match="manifest IDs do not match"):
        validate_expanded_dataset(questions, altered)