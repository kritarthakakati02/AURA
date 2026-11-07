"""Prepare a sourced, reproducible 400-question AURA benchmark expansion."""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import logging
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.dataset import DEFAULT_DATASET_PATH

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "benchmark_expansion.json"
SPLIT_DIR = ROOT / "data" / "splits"
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GOOGLE_RESEARCH_COMMIT = "0eebfd403ba438bfd14cb70b9b24a7745c997083"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_COMMIT}/grade_school_math/data/train.jsonl"
)
GSM8K_LICENSE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_COMMIT}/LICENSE"
)
MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{GOOGLE_RESEARCH_COMMIT}/mbpp/mbpp.jsonl"
)
MATH_CATEGORY = "mathematical_reasoning"
CODE_CATEGORY = "python_coding"
SPLITS = {"train": 240, "validation": 80, "test": 80}
SPLIT_CATEGORIES = {
    "train": {MATH_CATEGORY: 120, CODE_CATEGORY: 120},
    "validation": {MATH_CATEGORY: 40, CODE_CATEGORY: 40},
    "test": {MATH_CATEGORY: 40, CODE_CATEGORY: 40},
}
_FINAL_ANSWER = re.compile(r"^####\s*([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*$")
_WORD_OR_NUMBER = re.compile(r"\d+(?:\.\d+)?|[a-z]+", flags=re.IGNORECASE)
_FUNCTION_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
logger = logging.getLogger(__name__)


def normalize_for_deduplication(text: str) -> str:
    """Normalize wording and numbers for deterministic near-duplicate checks."""
    tokens = ["<number>" if token[0].isdigit() else token.lower() for token in _WORD_OR_NUMBER.findall(text)]
    return " ".join(tokens)


def is_near_duplicate(first: str, second: str, threshold: float = 0.92) -> bool:
    """Compare normalized token sequences, treating changed numbers as equivalent."""
    tokens_first = normalize_for_deduplication(first).split()
    tokens_second = normalize_for_deduplication(second).split()
    if tokens_first == tokens_second:
        return True
    matcher = difflib.SequenceMatcher(None, tokens_first, tokens_second, autojunk=False)
    return matcher.real_quick_ratio() >= threshold and matcher.ratio() >= threshold


def _download(url: str, destination: Path, refresh: bool) -> bytes:
    if destination.exists() and not refresh:
        return destination.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "AURA-benchmark-preparation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload


def _read_jsonl(payload: bytes, source_name: str) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSONL from {source_name}: {exc}") from exc


def _math_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, record in enumerate(raw_records, start=1):
        question = record.get("question")
        solution = record.get("answer")
        if not isinstance(question, str) or not isinstance(solution, str):
            continue
        final_lines = [line.strip() for line in solution.splitlines() if line.strip()]
        match = _FINAL_ANSWER.fullmatch(final_lines[-1]) if final_lines else None
        if not match:
            continue
        answer = match.group(1).replace(",", "")
        records.append({
            "question_id": f"gsm8k_train_{row_index:05d}",
            "category": MATH_CATEGORY,
            "prompt": f"{question.strip()}\nGive your final numeric answer on the last line.",
            "expected_answer": answer,
            "evaluation_type": "exact_numeric",
            "source": {
                "name": "GSM8K",
                "url": "https://github.com/openai/grade-school-math",
                "license": "MIT",
                "license_url": "https://github.com/openai/grade-school-math/blob/master/LICENSE",
                "source_commit": GSM8K_COMMIT,
                "transformation": (
                    "Used the official train.jsonl question; appended a final-answer formatting "
                    "instruction; extracted the numeric value after the final #### marker and "
                    "removed thousands-separator commas."
                ),
            },
            "source_question_id": f"train-line-{row_index:05d}",
        })
    return records


def _json_value(value: Any) -> Any:
    """Accept only values whose JSON conversion preserves Python test semantics."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite float in MBPP test")
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError(f"test value is not JSON-compatible without semantic change: {type(value).__name__}")


def _convert_mbpp_test(test: str) -> tuple[str, list[Any], Any]:
    parsed = ast.parse(test)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise ValueError("expected one assert statement")
    expression = parsed.body[0].test
    if (
        not isinstance(expression, ast.Compare)
        or len(expression.ops) != 1
        or not isinstance(expression.ops[0], ast.Eq)
        or len(expression.comparators) != 1
        or not isinstance(expression.left, ast.Call)
        or not isinstance(expression.left.func, ast.Name)
        or expression.left.keywords
    ):
        raise ValueError("assertion is not a direct function-call equality")
    function_name = expression.left.func.id
    args = _json_value([ast.literal_eval(argument) for argument in expression.left.args])
    expected = _json_value(ast.literal_eval(expression.comparators[0]))
    return function_name, args, expected


def _code_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in raw_records:
        task_id = record.get("task_id")
        if not isinstance(task_id, int) or not 601 <= task_id <= 974:
            continue
        if record.get("test_setup_code", "").strip():
            continue
        prompt = record.get("text")
        tests = record.get("test_list")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(tests, list) or not tests:
            continue
        try:
            converted = [_convert_mbpp_test(test) for test in tests]
            function_names = {function_name for function_name, _, _ in converted}
            if len(function_names) != 1:
                continue
        except (SyntaxError, TypeError, ValueError):
            continue

        function_name = next(iter(function_names))
        records.append({
            "question_id": f"mbpp_{task_id:04d}",
            "category": CODE_CATEGORY,
            "prompt": f"{prompt.strip()}\nReturn Python code defining the required function.",
            "test_spec": {
                "function_name": function_name,
                "description": prompt.strip(),
                "cases": [
                    {"args": args, "expected": expected}
                    for _, args, expected in converted
                ],
            },
            "evaluation_type": "python_tests",
            "source": {
                "name": "MBPP",
                "url": "https://github.com/google-research/google-research/tree/master/mbpp",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "license_metadata_url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
                "source_commit": GOOGLE_RESEARCH_COMMIT,
                "upstream_repository_license": "Apache-2.0",
                "license_note": (
                    "The Google Research repository has an Apache-2.0 repository license; "
                    "the Google Research Datasets Hugging Face card explicitly labels MBPP "
                    "data CC-BY-4.0. The dataset-card declaration is recorded for the data, "
                    "and both upstream notices must be reviewed when redistributing."
                ),
                "transformation": (
                    "Used task text and test_list from an upstream documented training task "
                    "(task_id 601-974); converted only direct assert function-call equality "
                    "tests with literal JSON-compatible arguments and expected values. "
                    "Reference code was not loaded or executed."
                ),
            },
            "source_question_id": str(task_id),
        })
    return records


def _legacy_prompts(path: Path) -> dict[str, list[str]]:
    try:
        legacy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load legacy dataset at {path}: {exc}") from exc
    return {
        category: [
            item["prompt"] for item in legacy
            if item.get("category") == category and isinstance(item.get("prompt"), str)
        ]
        for category in (MATH_CATEGORY, CODE_CATEGORY)
    }


def _select_unique(
    records: list[dict[str, Any]], legacy_prompts: list[str], count: int, rng: random.Random
) -> tuple[list[dict[str, Any]], Counter[str]]:
    shuffled = records.copy()
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    previous_prompts = list(legacy_prompts)
    removed: Counter[str] = Counter()
    for record in shuffled:
        prompt = record["prompt"]
        if any(normalize_for_deduplication(prompt) == normalize_for_deduplication(previous) for previous in previous_prompts):
            removed["exact"] += 1
            continue
        if any(is_near_duplicate(prompt, previous) for previous in previous_prompts):
            removed["near"] += 1
            continue
        selected.append(record)
        previous_prompts.append(prompt)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} unique records available for requested {count}; "
            f"duplicates removed: {dict(removed)}"
        )
    return selected, removed


def _split_records(
    selected: list[dict[str, Any]], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    by_category = {
        category: [record.copy() for record in selected if record["category"] == category]
        for category in (MATH_CATEGORY, CODE_CATEGORY)
    }
    split_records = {name: [] for name in SPLITS}
    cursor = {name: 0 for name in SPLITS}
    for category, records in by_category.items():
        rng.shuffle(records)
        category_cursor = 0
        for split_name in SPLITS:
            size = SPLIT_CATEGORIES[split_name][category]
            subset = records[category_cursor: category_cursor + size]
            category_cursor += size
            cursor[split_name] += size
            for record in subset:
                record["split"] = split_name
            split_records[split_name].extend(subset)
    for split_name in SPLITS:
        rng.shuffle(split_records[split_name])
    manifest = {
        "random_seed": seed,
        "method": (
            "Use only source training partitions; deduplicate exact normalized text and "
            "near duplicates by SequenceMatcher ratio >= 0.92 after lowercasing and "
            "replacing numeric tokens with <number>; select 200 per category; seeded "
            "shuffle independently by category; allocate 120/40/40 per category to "
            "train/validation/test."
        ),
        "split_purposes": {
            "train": "Router training only after a later phase authorizes it.",
            "validation": "Model and threshold selection; do not use test results.",
            "test": "Held-out final evaluation; never use for training or threshold selection.",
        },
        "splits": {},
    }
    for split_name, records in split_records.items():
        counts = Counter(record["category"] for record in records)
        manifest["splits"][split_name] = {
            "count": len(records),
            "category_counts": dict(sorted(counts.items())),
            "question_ids": [record["question_id"] for record in records],
        }
    return split_records, manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(
    seed: int = 2026,
    output_path: Path = OUTPUT_PATH,
    split_dir: Path = SPLIT_DIR,
    legacy_path: Path = DEFAULT_DATASET_PATH,
    overwrite: bool = False,
    refresh_raw: bool = False,
) -> dict[str, Any]:
    """Download pinned source training data and build the processed benchmark."""
    destinations = [output_path, split_dir / "manifest.json"] + [split_dir / f"{name}.json" for name in SPLITS]
    destinations += [split_dir / f"{name}_manifest.json" for name in SPLITS]
    if not overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {', '.join(existing)}")

    gsm_bytes = _download(GSM8K_URL, RAW_DIR / "gsm8k_train.jsonl", refresh_raw)
    _download(GSM8K_LICENSE_URL, RAW_DIR / "GSM8K_LICENSE.txt", refresh_raw)
    mbpp_request = urllib.request.Request(MBPP_URL, headers={"User-Agent": "AURA-benchmark-preparation"})
    with urllib.request.urlopen(mbpp_request, timeout=60) as response:
        mbpp_bytes = response.read()
    gsm_records = _math_records(_read_jsonl(gsm_bytes, "GSM8K"))
    mbpp_raw_records = _read_jsonl(mbpp_bytes, "MBPP")
    mbpp_training_records = [
        row for row in mbpp_raw_records
        if isinstance(row.get("task_id"), int) and 601 <= row["task_id"] <= 974
    ]
    if refresh_raw or not (RAW_DIR / "mbpp_training.jsonl").exists():
        training_jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mbpp_training_records)
        (RAW_DIR / "mbpp_training.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "mbpp_training.jsonl").write_text(training_jsonl, encoding="utf-8")
    mbpp_records = _code_records(mbpp_training_records)
    legacy = _legacy_prompts(legacy_path)
    rng = random.Random(seed)
    math_selected, math_removed = _select_unique(
        gsm_records, legacy[MATH_CATEGORY], 200, rng
    )
    code_selected, code_removed = _select_unique(
        mbpp_records,
        legacy[CODE_CATEGORY] + [record["prompt"] for record in math_selected],
        200,
        rng,
    )
    selected = math_selected + code_selected
    split_records, manifest = _split_records(selected, seed)
    selected_with_splits = [
        record for split_name in SPLITS for record in split_records[split_name]
    ]
    source_checksums = {
        "gsm8k_train_sha256": hashlib.sha256(gsm_bytes).hexdigest(),
        "mbpp_sha256": hashlib.sha256(mbpp_bytes).hexdigest(),
    }
    manifest["source_checksums"] = source_checksums
    manifest["source_candidate_counts"] = {
        "gsm8k": {
            "training_rows": len(_read_jsonl(gsm_bytes, "GSM8K")),
            "rows_with_extractable_final_numeric_answer": len(gsm_records),
            "selected": len(math_selected),
        },
        "mbpp": {
            "documented_training_task_ids": len(mbpp_training_records),
            "tasks_with_convertible_deterministic_tests": len(mbpp_records),
            "tasks_excluded_for_setup_or_unsupported_tests": len(mbpp_training_records) - len(mbpp_records),
            "selected": len(code_selected),
        },
    }
    manifest["duplicates_removed"] = {
        MATH_CATEGORY: dict(math_removed),
        CODE_CATEGORY: dict(code_removed),
        "total": sum(math_removed.values()) + sum(code_removed.values()),
    }

    _write_json(output_path, selected_with_splits)
    for split_name, records in split_records.items():
        category_counts = dict(sorted(Counter(record["category"] for record in records).items()))
        _write_json(split_dir / f"{split_name}.json", records)
        _write_json(split_dir / f"{split_name}_manifest.json", {
            "split": split_name,
            "random_seed": seed,
            "count": len(records),
            "category_counts": category_counts,
            "question_ids": [record["question_id"] for record in records],
        })
    _write_json(split_dir / "manifest.json", manifest)
    logger.info(
        "Prepared %d questions (math=%d, coding=%d); duplicates removed=%d",
        len(selected), len(math_selected), len(code_selected), manifest["duplicates_removed"]["total"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--legacy-dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace processed dataset and split manifests")
    parser.add_argument("--refresh-raw", action="store_true", help="Redownload pinned source files into data/raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_dataset(args.seed, args.output, args.split_dir, args.legacy_dataset, args.overwrite, args.refresh_raw)


if __name__ == "__main__":
    main()