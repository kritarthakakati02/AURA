"""Re-evaluate coding responses using the improved evaluator.

This script takes existing raw results and re-evaluates coding responses
using the improved code_eval_v2 module without re-running model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from benchmarks.evaluate_v2 import evaluate_response

logger = logging.getLogger(__name__)


def load_question_specs(splits_dir: Path) -> dict[str, dict[str, Any]]:
    """Load test_spec for each question_id from split files."""
    specs = {}

    for split_file in splits_dir.glob("*.json"):
        if "manifest" in split_file.name:
            continue

        try:
            with split_file.open("r", encoding="utf-8") as f:
                questions = json.load(f)

            for q in questions:
                qid = q.get("question_id")
                if qid and q.get("evaluation_type") == "python_tests":
                    specs[qid] = {
                        "evaluation_type": "python_tests",
                        "test_spec": q.get("test_spec"),
                        "category": q.get("category"),
                    }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s: %s", split_file, exc)

    return specs


def re_evaluate_results(
    input_path: Path,
    output_path: Path,
    question_specs: dict[str, dict[str, Any]],
    filter_coding_only: bool = True,
    filter_failed_only: bool = False,
) -> dict[str, int]:
    """Re-evaluate responses from raw results CSV.

    Returns:
        Statistics dict with counts of re-evaluated results
    """
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    stats = {
        "total_rows": len(rows),
        "coding_tasks": 0,
        "re_evaluated": 0,
        "newly_correct": 0,
        "still_incorrect": 0,
        "infrastructure_failures": 0,
        "missing_test_specs": 0,
        "empty_responses": 0,
        "exceptions": 0,
        "skipped_correct": 0,
        "skipped_non_coding": 0,
    }

    output_rows = []

    for row in rows:
        evaluation_type = row.get("evaluation_type")
        question_id = row.get("question_id")
        status = row.get("status")
        original_correct = row.get("correct", "").lower()

        # Skip non-coding tasks if requested
        if filter_coding_only and evaluation_type != "python_tests":
            stats["skipped_non_coding"] += 1
            output_rows.append(row)
            continue

        stats["coding_tasks"] += 1

        # Track infrastructure failures separately
        if status != "success":
            stats["infrastructure_failures"] += 1
            output_rows.append(row)
            continue

        # Skip already correct responses if requested
        if filter_failed_only and original_correct == "true":
            stats["skipped_correct"] += 1
            output_rows.append(row)
            continue

        # Get question spec
        spec = question_specs.get(question_id)
        if not spec:
            stats["missing_test_specs"] += 1
            logger.warning("No test spec found for %s, skipping", question_id)
            output_rows.append(row)
            continue

        # Check for empty response
        response = row.get("response", "")
        if not response:
            stats["empty_responses"] += 1
            output_rows.append(row)
            continue

        # Re-evaluate
        try:
            correct, error = evaluate_response(spec, response)

            # Update row
            new_row = row.copy()
            new_row["correct"] = str(correct)
            new_row["evaluation_error"] = error or ""

            stats["re_evaluated"] += 1

            # Track changes
            if correct and original_correct != "true":
                stats["newly_correct"] += 1
                logger.info("Newly correct: %s", question_id)
            elif not correct and original_correct == "true":
                logger.warning("Regression: %s was correct, now incorrect", question_id)
            elif not correct:
                stats["still_incorrect"] += 1

            output_rows.append(new_row)

        except Exception as exc:
            stats["exceptions"] += 1
            logger.exception("Re-evaluation failed for %s: %s", question_id, exc)
            output_rows.append(row)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info("Wrote re-evaluated results to %s", output_path)
    return stats


def print_stats(stats: dict[str, int], title: str) -> None:
    """Print re-evaluation statistics."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Coding tasks: {stats['coding_tasks']}")
    print()
    print(f"Skipped (non-coding): {stats['skipped_non_coding']}")
    print(f"Skipped (already correct): {stats['skipped_correct']}")
    print(f"Infrastructure failures (skipped): {stats['infrastructure_failures']}")
    print(f"Missing test specs (skipped): {stats['missing_test_specs']}")
    print(f"Empty responses (skipped): {stats['empty_responses']}")
    print(f"Exceptions during re-evaluation: {stats['exceptions']}")
    print()
    print(f"Successfully re-evaluated: {stats['re_evaluated']}")
    print(f"  Newly correct: {stats['newly_correct']}")
    print(f"  Still incorrect: {stats['still_incorrect']}")

    if stats['re_evaluated'] > 0:
        recovery_rate = stats['newly_correct'] / stats['re_evaluated'] * 100
        print(f"  Recovery rate: {recovery_rate:.1f}%")
    print("=" * 80)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input raw results CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output re-evaluated CSV")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"), help="Directory with question specs")
    parser.add_argument("--failed-only", action="store_true", help="Re-evaluate only failed responses")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if args.output.exists():
        response = input(f"{args.output} exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    logger.info("Loading question specs from %s", args.splits_dir)
    specs = load_question_specs(args.splits_dir)
    logger.info("Loaded %d coding task specs", len(specs))

    logger.info("Re-evaluating %s", args.input)
    stats = re_evaluate_results(args.input, args.output, specs, filter_failed_only=args.failed_only)

    print_stats(stats, f"RE-EVALUATION RESULTS: {args.input.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
