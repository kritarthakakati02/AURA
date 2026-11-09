"""Analyze coding failures from existing raw results to understand error patterns.

This script reads the raw benchmark results and categorizes coding task failures
to understand what would be fixed by the improved evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def analyze_coding_failures(result_path: Path) -> dict:
    """Analyze coding task failures and categorize them."""

    with result_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Filter to coding tasks
    coding_tasks = [r for r in rows if r.get("evaluation_type") == "python_tests"]

    # Categorize by status and correctness
    total = len(coding_tasks)
    infrastructure_failures = [r for r in coding_tasks if r.get("status") != "success"]
    evaluated = [r for r in coding_tasks if r.get("status") == "success"]
    correct = [r for r in evaluated if r.get("correct", "").lower() == "true"]
    incorrect = [r for r in evaluated if r.get("correct", "").lower() == "false"]
    unevaluated = [r for r in evaluated if r.get("correct", "") == ""]

    # Categorize error types
    error_categories = Counter()
    error_samples = {}

    for row in incorrect:
        error = row.get("evaluation_error", "")

        if "top-level function definitions" in error:
            category = "top_level_statement_rejection"
        elif "Expected function" in error and "was not defined" in error:
            category = "function_name_mismatch"
        elif "syntax" in error.lower():
            category = "syntax_error"
        elif "Only direct calls" in error:
            category = "disallowed_call"
        elif "One or more" in error and "test cases failed" in error:
            category = "wrong_answer"
        elif "timeout" in error.lower():
            category = "timeout"
        elif error == "":
            category = "unevaluated"
        else:
            category = "other"

        error_categories[category] += 1

        # Keep sample for each category
        if category not in error_samples and len(row.get("response", "")) < 1000:
            error_samples[category] = {
                "question_id": row.get("question_id"),
                "prompt": row.get("prompt", "")[:200],
                "response": row.get("response", "")[:500],
                "error": error[:200],
            }

    # Infrastructure failures
    infra_types = Counter()
    for row in infrastructure_failures:
        error_type = row.get("error_type", "unknown")
        infra_types[error_type] += 1

    report = {
        "total_coding_tasks": total,
        "infrastructure_failures": len(infrastructure_failures),
        "infrastructure_by_type": dict(infra_types),
        "evaluated": len(evaluated),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "unevaluated": len(unevaluated),
        "accuracy_of_evaluated": f"{len(correct)}/{len(evaluated)} = {len(correct)/len(evaluated)*100:.1f}%" if evaluated else "n/a",
        "error_categories": dict(error_categories.most_common()),
        "error_samples": error_samples,
    }

    return report


def print_report(report: dict) -> None:
    """Print formatted analysis report."""
    print("=" * 80)
    print("CODING BENCHMARK FAILURE ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total coding tasks: {report['total_coding_tasks']}")
    print(f"Infrastructure failures: {report['infrastructure_failures']}")
    if report['infrastructure_by_type']:
        for error_type, count in report['infrastructure_by_type'].items():
            print(f"  - {error_type}: {count}")
    print()

    print(f"Successfully evaluated: {report['evaluated']}")
    print(f"  Correct: {report['correct']}")
    print(f"  Incorrect: {report['incorrect']}")
    print(f"  Unevaluated: {report['unevaluated']}")
    print(f"  Measured accuracy: {report['accuracy_of_evaluated']}")
    print()

    print("Error category breakdown:")
    print("-" * 80)
    for category, count in report['error_categories'].items():
        print(f"{category:40s}: {count:4d}")
        if category in report['error_samples']:
            sample = report['error_samples'][category]
            print(f"  Sample: {sample['question_id']}")
            print(f"  Prompt: {sample['prompt']}...")
            if sample['error']:
                print(f"  Error: {sample['error']}...")
            print()

    print("=" * 80)
    print()

    # Report candidates for re-evaluation (NOT recovered answers)
    top_level_rejected = report['error_categories'].get('top_level_statement_rejection', 0)
    function_mismatch = report['error_categories'].get('function_name_mismatch', 0)
    wrong_answers = report['error_categories'].get('wrong_answer', 0)
    syntax_errors = report['error_categories'].get('syntax_error', 0)
    disallowed_calls = report['error_categories'].get('disallowed_call', 0)
    timeouts = report['error_categories'].get('timeout', 0)

    print("CANDIDATES FOR RE-EVALUATION:")
    print(f"  Top-level statement rejection: {top_level_rejected} (would need new evaluator)")
    print(f"  Function name mismatch: {function_mismatch} (would need enhanced prompts)")
    print(f"  Wrong algorithmic answer: {wrong_answers} (genuine failures)")
    print(f"  Syntax errors: {syntax_errors} (genuine failures)")
    print(f"  Disallowed calls: {disallowed_calls} (genuine failures)")
    print(f"  Timeouts: {timeouts} (genuine failures)")
    print()

    current_correct = report['correct']
    evaluated = report['evaluated']

    if evaluated > 0:
        current_acc = current_correct / evaluated * 100
        candidates_total = top_level_rejected + function_mismatch
        print(f"  Current measured accuracy: {current_acc:.1f}% ({current_correct}/{evaluated})")
        print(f"  Candidates for re-evaluation: {candidates_total} ({top_level_rejected} + {function_mismatch})")
        print()
        print("  NOTE: Candidates are NOT recovered answers. Actual recovery rate must be")
        print("  measured by re-running evaluation with the improved pipeline.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, default=Path("results/train_results.csv"))
    parser.add_argument("--validation-results", type=Path, default=Path("results/validation_results.csv"))
    parser.add_argument("--output", type=Path, help="Save JSON report to file")
    args = parser.parse_args()

    if args.train_results.exists():
        print("\nTRAINING SET ANALYSIS:")
        print("=" * 80)
        train_report = analyze_coding_failures(args.train_results)
        print_report(train_report)

        if args.output:
            output_train = args.output.parent / f"{args.output.stem}_train.json"
            output_train.write_text(json.dumps(train_report, indent=2), encoding="utf-8")
            print(f"Saved training analysis to {output_train}")

    if args.validation_results.exists():
        print("\nVALIDATION SET ANALYSIS:")
        print("=" * 80)
        val_report = analyze_coding_failures(args.validation_results)
        print_report(val_report)

        if args.output:
            output_val = args.output.parent / f"{args.output.stem}_validation.json"
            output_val.write_text(json.dumps(val_report, indent=2), encoding="utf-8")
            print(f"Saved validation analysis to {output_val}")


if __name__ == "__main__":
    main()
