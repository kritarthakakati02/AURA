"""Improved deterministic answer checks with granular error categories.

Changes from evaluate.py:
1. Uses improved code_eval_v2 with better extraction and error categories
2. Preserves detailed error information instead of collapsing to generic messages
3. Returns structured error categories for better failure analysis
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks.code_eval_v2 import evaluate_python_response

logger = logging.getLogger(__name__)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?")


def evaluate_response(
    question: dict[str, Any], response: str, code_timeout: float = 2.0
) -> tuple[bool, str | None]:
    """Evaluate one successful model response without using an LLM judge.

    Returns:
        (correct: bool, error_detail: str | None)

        For coding tasks, error_detail is categorized:
        - "syntax_error: ..."
        - "missing_function: ..."
        - "function_name_mismatch: ..."
        - "unsupported_syntax: ..."
        - "execution_error: ..."
        - "timeout: ..."
        - "wrong_answer: ..."
        - "infrastructure_error: ..."
    """
    if question["evaluation_type"] == "exact_numeric":
        actual = _extract_fraction(response)
        expected = _as_fraction(question["expected_answer"])
        return actual is not None and actual == expected, None if actual is not None else "No numeric answer found"
    if question["evaluation_type"] == "python_tests":
        return evaluate_python_response(response, question["test_spec"], code_timeout)
    return False, f"Unsupported evaluation type: {question['evaluation_type']}"


def _extract_fraction(response: str) -> Fraction | None:
    """Read the last numeric token on the final non-empty line.

    Benchmark math prompts request the final answer on its own final line, so
    numbers in preceding explanation lines are intentionally ignored.
    """
    lines = [line for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    matches = _NUMBER_PATTERN.findall(lines[-1])
    if not matches:
        return None
    try:
        return Fraction(re.sub(r"\s+", "", matches[-1]))
    except (ValueError, ZeroDivisionError):
        return None


def _as_fraction(value: Any) -> Fraction:
    return Fraction(str(value).strip())


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Calculate per-model/category metrics and paired correctness disagreement."""
    models = sorted({row.get("model", "") for row in rows if row.get("model")})
    categories = sorted({row.get("category", "") for row in rows if row.get("category")})
    group_keys = [(model, category) for model in models for category in ["all", *categories]]
    output: list[dict[str, Any]] = []

    for model, category in group_keys:
        selected = [
            row for row in rows
            if row.get("model") == model
            and (category == "all" or row.get("category") == category)
        ]
        attempts = len(selected)
        valid = [row for row in selected if row.get("status") == "success"]
        evaluated = [row for row in valid if row.get("correct", "").lower() in {"true", "false"}]
        correct_count = sum(row["correct"].lower() == "true" for row in evaluated)
        latencies = _numbers(row.get("wall_clock_seconds") for row in selected)
        generated_tokens = _numbers(row.get("generated_tokens") for row in valid)
        output.append({
            "model": model,
            "category": category,
            "attempts": attempts,
            "valid_responses": len(valid),
            "evaluated_responses": len(evaluated),
            "correct": correct_count,
            "accuracy": correct_count / len(evaluated) if evaluated else "",
            "average_latency_seconds": statistics.mean(latencies) if latencies else "",
            "median_latency_seconds": statistics.median(latencies) if latencies else "",
            "p95_latency_seconds": _percentile_95(latencies),
            "average_generated_tokens": statistics.mean(generated_tokens) if generated_tokens else "",
            "failure_rate": (attempts - len(valid)) / attempts if attempts else "",
            "paired_disagreement_rate": _paired_disagreement(rows, category),
        })
    return output


def _numbers(values: Any) -> list[float]:
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return parsed


def _percentile_95(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _paired_disagreement(rows: list[dict[str, str]], category: str) -> float | str:
    by_question: dict[str, dict[str, bool]] = {}
    for row in rows:
        if category != "all" and row.get("category") != category:
            continue
        question_id = row.get("question_id", "")
        model = row.get("model", "")
        correctness = row.get("correct", "").lower()
        if (
            row.get("status") == "success"
            and correctness in {"true", "false"}
            and question_id
            and model
        ):
            by_question.setdefault(question_id, {})[model] = correctness == "true"
    pairs = [outcomes for outcomes in by_question.values() if len(outcomes) >= 2]
    if not pairs:
        return ""
    disagreements = sum(len(set(outcomes.values())) > 1 for outcomes in pairs)
    return disagreements / len(pairs)


def evaluate_file(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Read raw benchmark CSV and write machine-readable and readable summaries."""
    try:
        with input_path.open("r", encoding="utf-8", newline="") as result_file:
            rows = list(csv.DictReader(result_file))
    except OSError as exc:
        raise ValueError(f"Could not read benchmark results at {input_path}: {exc}") from exc

    summaries = summarize_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    text_path = output_dir / "summary.txt"
    columns = [
        "model", "category", "attempts", "valid_responses", "evaluated_responses", "correct",
        "accuracy", "average_latency_seconds", "median_latency_seconds", "p95_latency_seconds",
        "average_generated_tokens", "failure_rate", "paired_disagreement_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    text_path.write_text(_render_text_summary(summaries), encoding="utf-8")
    logger.info("Wrote evaluation summaries to %s and %s", csv_path, text_path)
    return csv_path, text_path


def _render_text_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "AURA benchmark summary",
        "Accuracy excludes failed infrastructure requests and includes evaluated responses only.",
        "Latency includes all attempts; token averages include successful responses with token counts.",
        "Paired correctness disagreement is the fraction of questions with at least two distinct models having valid evaluated outcomes where models disagree.",
        "",
        "model | category | accuracy | avg latency (s) | median (s) | p95 (s) | avg generated tokens | failure rate | paired disagreement",
    ]
    for row in rows:
        accuracy = _display_ratio(row["correct"], row["evaluated_responses"])
        lines.append(
            f"{row['model']} | {row['category']} | {accuracy} | "
            f"{_display(row['average_latency_seconds'])} | {_display(row['median_latency_seconds'])} | "
            f"{_display(row['p95_latency_seconds'])} | {_display(row['average_generated_tokens'])} | "
            f"{_display(row['failure_rate'])} | {_display(row['paired_disagreement_rate'])}"
        )
    return "\n".join(lines) + "\n"


def _display(value: Any) -> str:
    return "n/a" if value == "" else f"{value:.4f}" if isinstance(value, float) else str(value)


def _display_ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/benchmark_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    evaluate_file(args.input, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
