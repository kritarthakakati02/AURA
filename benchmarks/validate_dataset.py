"""Validate expanded benchmark provenance, records, deduplication, and splits."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.prepare_dataset import (
    CODE_CATEGORY,
    MATH_CATEGORY,
    SPLIT_CATEGORIES,
    SPLITS,
    is_near_duplicate,
    normalize_for_deduplication,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "benchmark_expansion.json"
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits"
_MATH_ANSWER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?$")
logger = logging.getLogger(__name__)


def validate_expanded_dataset(
    questions: Any,
    split_manifests: dict[str, Any],
    expected_split_sizes: dict[str, int] | None = None,
    expected_category_splits: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Raise ValueError on schema, provenance, duplicate, or split violations."""
    expected_split_sizes = expected_split_sizes or SPLITS
    expected_category_splits = expected_category_splits or SPLIT_CATEGORIES
    if not isinstance(questions, list):
        raise ValueError("dataset must be a JSON list")
    expected_total = sum(expected_split_sizes.values())
    if len(questions) != expected_total:
        raise ValueError(f"expected {expected_total} questions, found {len(questions)}")

    seen_ids: set[str] = set()
    categories = Counter()
    split_questions: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_split_sizes}
    for index, question in enumerate(questions):
        where = f"question at index {index}"
        if not isinstance(question, dict):
            raise ValueError(f"{where} must be an object")
        required = ("question_id", "category", "prompt", "evaluation_type", "source", "source_question_id", "split")
        for field in required:
            value = question.get(field)
            if not isinstance(value, str) or not value.strip():
                if field != "source":
                    raise ValueError(f"{where} requires non-empty '{field}'")
        question_id = question["question_id"]
        if question_id in seen_ids:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen_ids.add(question_id)
        category = question["category"]
        evaluation_type = question["evaluation_type"]
        if category == MATH_CATEGORY:
            if evaluation_type != "exact_numeric":
                raise ValueError(f"{where} math question must use exact_numeric")
            answer = question.get("expected_answer")
            if not isinstance(answer, (str, int, float)) or not _MATH_ANSWER.fullmatch(str(answer)):
                raise ValueError(f"{where} has invalid exact numeric answer")
            if "test_spec" in question:
                raise ValueError(f"{where} math question must not include test_spec")
        elif category == CODE_CATEGORY:
            if evaluation_type != "python_tests":
                raise ValueError(f"{where} coding question must use python_tests")
            if "expected_answer" in question:
                raise ValueError(f"{where} coding question must use test_spec, not expected_answer")
            _validate_test_spec(question.get("test_spec"), where)
        else:
            raise ValueError(f"{where} has unsupported category '{category}'")
        _validate_source(question.get("source"), where)
        split = question["split"]
        if split not in split_questions:
            raise ValueError(f"{where} has invalid split '{split}'")
        split_questions[split].append(question)
        categories[category] += 1

    expected_category_totals = {MATH_CATEGORY: 200, CODE_CATEGORY: 200}
    if dict(categories) != expected_category_totals:
        raise ValueError(f"category counts must be {expected_category_totals}; found {dict(categories)}")

    _validate_no_near_duplicates(questions)
    _validate_splits(split_questions, split_manifests, expected_split_sizes, expected_category_splits)
    return {
        "total": len(questions),
        "category_counts": dict(sorted(categories.items())),
        "split_counts": {
            split: {
                "total": len(records),
                "categories": dict(sorted(Counter(row["category"] for row in records).items())),
            }
            for split, records in split_questions.items()
        },
    }


def _validate_test_spec(spec: Any, where: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{where} requires test_spec")
    function_name = spec.get("function_name")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError(f"{where} test_spec requires a valid function_name")
    if not isinstance(spec.get("description"), str) or not spec["description"].strip():
        raise ValueError(f"{where} test_spec requires a description")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{where} test_spec requires non-empty cases")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("args"), list) or "expected" not in case:
            raise ValueError(f"{where} test_spec has malformed case")
        try:
            json.dumps([case["args"], case["expected"]], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} test case is not JSON-compatible: {exc}") from exc


def _validate_source(source: Any, where: str) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"{where} requires source provenance object")
    required = ("name", "url", "license", "license_url", "source_commit", "transformation")
    missing = [field for field in required if not isinstance(source.get(field), str) or not source[field].strip()]
    if missing:
        raise ValueError(f"{where} source provenance missing: {', '.join(missing)}")
    if not source["url"].startswith("https://") or not source["license_url"].startswith("https://"):
        raise ValueError(f"{where} source and license URLs must use HTTPS")
    if source["name"] == "MBPP":
        if source["license"] != "CC-BY-4.0" or source.get("upstream_repository_license") != "Apache-2.0":
            raise ValueError(f"{where} MBPP license metadata is incomplete")


def _validate_no_near_duplicates(questions: list[dict[str, Any]]) -> None:
    for index, current in enumerate(questions):
        current_normalized = normalize_for_deduplication(current["prompt"])
        for previous in questions[:index]:
            if current_normalized == normalize_for_deduplication(previous["prompt"]):
                raise ValueError(f"exact duplicate prompts: {previous['question_id']} and {current['question_id']}")
            if is_near_duplicate(current["prompt"], previous["prompt"]):
                raise ValueError(f"near-duplicate prompts: {previous['question_id']} and {current['question_id']}")


def _validate_splits(
    split_questions: dict[str, list[dict[str, Any]]],
    manifests: dict[str, Any],
    expected_sizes: dict[str, int],
    expected_categories: dict[str, dict[str, int]],
) -> None:
    ids_by_split: dict[str, set[str]] = {}
    for split in expected_sizes:
        entry = manifests.get(split)
        question_ids = entry.get("question_ids") if isinstance(entry, dict) else None
        if not isinstance(question_ids, list):
            raise ValueError(f"missing question IDs in manifest for split '{split}'")
        current_ids = set(question_ids)
        if len(current_ids) != len(question_ids):
            raise ValueError(f"duplicate question IDs in '{split}' manifest")
        for earlier_split, earlier_ids in ids_by_split.items():
            overlap = earlier_ids & current_ids
            if overlap:
                raise ValueError(f"question leakage between '{earlier_split}' and '{split}': {sorted(overlap)[:3]}")
        ids_by_split[split] = current_ids

    ids_by_split.clear()
    for split, expected_size in expected_sizes.items():
        records = split_questions[split]
        if len(records) != expected_size:
            raise ValueError(f"split '{split}' expected {expected_size} questions, found {len(records)}")
        counts = Counter(row["category"] for row in records)
        if dict(counts) != expected_categories[split]:
            raise ValueError(f"split '{split}' category counts must be {expected_categories[split]}, found {dict(counts)}")
        entry = manifests.get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"missing manifest for split '{split}'")
        question_ids = entry.get("question_ids")
        actual_ids = {row["question_id"] for row in records}
        if not isinstance(question_ids, list) or len(question_ids) != len(records) or set(question_ids) != actual_ids:
            raise ValueError(f"split '{split}' manifest IDs do not match dataset records")
        if entry.get("count") != expected_size:
            raise ValueError(f"split '{split}' manifest count mismatch")
        if entry.get("category_counts") != dict(counts):
            raise ValueError(f"split '{split}' manifest category counts mismatch")
        ids_by_split[split] = actual_ids
    split_names = list(ids_by_split)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1:]:
            overlap = ids_by_split[split] & ids_by_split[other]
            if overlap:
                raise ValueError(f"question leakage between '{split}' and '{other}': {sorted(overlap)[:3]}")


def validate_files(dataset_path: Path = DEFAULT_DATASET, split_dir: Path = DEFAULT_SPLIT_DIR) -> dict[str, Any]:
    try:
        questions = json.loads(dataset_path.read_text(encoding="utf-8"))
        manifests = {
            split: json.loads((split_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))
            for split in SPLITS
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset or split manifests: {exc}") from exc
    return validate_expanded_dataset(questions, manifests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    args = parser.parse_args()
    report = validate_files(args.dataset, args.split_dir)
    print(json.dumps(report, indent=2))
    logger.info("Dataset validation passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()