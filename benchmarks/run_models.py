"""Run the fixed AURA benchmark against one or more local Ollama models."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.inference import InferenceResult, OllamaClient
from benchmarks.dataset import DEFAULT_DATASET_PATH, load_dataset
from benchmarks.evaluate import evaluate_response

logger = logging.getLogger(__name__)
DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "benchmark_results.csv"
RESULT_COLUMNS = [
    "timestamp_utc", "model", "question_id", "category", "evaluation_type", "prompt",
    "response", "status", "correct", "evaluation_error", "wall_clock_seconds",
    "prompt_tokens", "generated_tokens", "total_duration_ns", "load_duration_ns",
    "prompt_eval_duration_ns", "eval_duration_ns", "error_type", "error_message",
]


def build_result_row(
    model: str, question: dict[str, Any], result: InferenceResult, code_timeout: float
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question_id": question["id"],
        "category": question["category"],
        "evaluation_type": question["evaluation_type"],
        "prompt": question["prompt"],
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
            logger.exception("Evaluation failed for %s / %s", model, question["id"])
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
) -> Path:
    """Run every selected question for every model and preserve every attempt."""
    if not models or any(not model.strip() for model in models):
        raise ValueError("provide at least one non-empty model name")
    if sample is not None and sample < 1:
        raise ValueError("sample must be a positive integer")
    if code_timeout <= 0:
        raise ValueError("code timeout must be greater than zero")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    questions = load_dataset(dataset_path)
    if sample is not None:
        questions = questions[:sample]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running %d questions against %d models", len(questions), len(models))

    with output_path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for model in models:
            for question in questions:
                result = client.generate(model, question["prompt"])
                writer.writerow(build_result_row(model, question, result, code_timeout))
                result_file.flush()
                logger.info("%s %s: %s", model, question["id"], result.status)
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
    args = parser.parse_args()
    run_benchmark(
        OllamaClient(args.base_url, args.request_timeout), args.models, args.dataset,
        args.output, args.sample, args.overwrite, args.code_timeout,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()