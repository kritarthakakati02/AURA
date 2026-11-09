"""Run the fixed AURA benchmark against one or more local Ollama models."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.inference import InferenceResult, OllamaClient
from benchmarks.dataset import DEFAULT_DATASET_PATH, canonical_question_id, load_dataset
from benchmarks.evaluate import evaluate_response
from benchmarks.prompt_builder import build_coding_prompt

logger = logging.getLogger(__name__)
DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "benchmark_results.csv"
RESULT_COLUMNS = [
    "timestamp_utc", "model", "question_id", "category", "evaluation_type", "prompt",
    "inference_prompt", "response", "status", "correct", "evaluation_error", "wall_clock_seconds",
    "prompt_tokens", "generated_tokens", "total_duration_ns", "load_duration_ns",
    "prompt_eval_duration_ns", "eval_duration_ns", "error_type", "error_message",
]


def build_result_row(
    model: str, question: dict[str, Any], inference_prompt: str, result: InferenceResult, code_timeout: float
) -> dict[str, Any]:
    """Build result row for CSV output.

    Args:
        model: Model name
        question: Question dict with original prompt
        inference_prompt: The actual prompt sent to Ollama (may be enhanced for coding tasks)
        result: Inference result from Ollama
        code_timeout: Timeout for code evaluation

    Note:
    - 'prompt' field stores the original question prompt for traceability
    - 'inference_prompt' field stores what was actually sent to Ollama
    """
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question_id": canonical_question_id(question),
        "category": question["category"],
        "evaluation_type": question["evaluation_type"],
        "prompt": question["prompt"],  # Original question prompt
        "inference_prompt": inference_prompt,  # What was actually sent to Ollama
        "response": result.response,
        "status": result.status,
        "correct": "",
        "evaluation_error": "",
        **{key: value for key, value in result.to_dict().items() if key != "model" and key != "response" and key != "status"},
    }
    if result.status == "success" and result.response is not None:
        try:
            correct, error = evaluate_response(question, result.response, code_timeout)
            row["correct"] = correct
            row["evaluation_error"] = error or ""
        except Exception as exc:
            logger.exception("Evaluation failed for %s / %s", model, canonical_question_id(question))
            row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_benchmark(
    client: OllamaClient,
    models: list[str],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_RESULT_PATH,
    sample: int | None = None,
    overwrite: bool = False,
    code_timeout: float = 2.0,
    question_ids: list[str] | None = None,
) -> Path:
    """Run every selected question for every model and preserve every attempt.

    Args:
        client: Ollama client
        models: List of model names
        dataset_path: Path to dataset JSON
        output_path: Path to output CSV
        sample: Optional limit on number of questions
        overwrite: Whether to overwrite existing output
        code_timeout: Timeout for code execution
        question_ids: Optional list of question IDs to run (filters dataset)

    Returns:
        Path to output CSV
    """
    if not models or any(not model.strip() for model in models):
        raise ValueError("provide at least one non-empty model name")
    if sample is not None and sample < 1:
        raise ValueError("sample must be a positive integer")
    if code_timeout <= 0:
        raise ValueError("code timeout must be greater than zero")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    if question_ids is not None and not question_ids:
        raise ValueError("question_ids list must not be empty if provided")

    questions = load_dataset(dataset_path)

    # Filter by question_ids if provided
    if question_ids is not None:
        question_id_set = set(question_ids)
        available_ids = {canonical_question_id(q) for q in questions}
        unmatched_ids = question_id_set - available_ids

        if unmatched_ids:
            raise ValueError(f"Question IDs not found in dataset: {', '.join(sorted(unmatched_ids))}")

        questions = [q for q in questions if canonical_question_id(q) in question_id_set]
        logger.info("Filtered to %d questions by ID", len(questions))

    if sample is not None:
        questions = questions[:sample]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running %d questions against %d models", len(questions), len(models))

    with output_path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for model in models:
            for question in questions:
                # Use enhanced prompt for coding tasks, original prompt for math
                inference_prompt = question["prompt"]
                if question.get("evaluation_type") == "python_tests":
                    inference_prompt = build_coding_prompt(question)

                # Generate response with the (possibly enhanced) prompt
                result = client.generate(model, inference_prompt)

                # Build row (stores both original and inference prompts)
                writer.writerow(build_result_row(model, question, inference_prompt, result, code_timeout))
                result_file.flush()
                logger.info("%s %s: %s", model, canonical_question_id(question), result.status)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "qwen2.5:3b"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--code-timeout", type=float, default=2.0)
    parser.add_argument("--sample", type=int, help="Run only the first N questions for smoke testing")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing raw result CSV")
    parser.add_argument("--question-ids", nargs="+", help="Run only specific question IDs (space-separated)")
    args = parser.parse_args()
    run_benchmark(
        OllamaClient(args.base_url, args.request_timeout), args.models, args.dataset,
        args.output, args.sample, args.overwrite, args.code_timeout, args.question_ids,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
