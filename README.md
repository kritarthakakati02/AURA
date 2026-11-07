# AURA Benchmarking Infrastructure

Phase 2 provides repeatable local inference runs and deterministic evaluation. It does not choose models or implement adaptive routing.

## Assumptions

- Ollama is reachable at `http://localhost:11434` and exposes `/api/generate`.
- The requested model tags are `qwen2.5:1.5b` and `qwen2.5:3b`.
- Math prompts require the final numeric answer on the last non-empty line. Exact rational answers such as `2/5` are supported.
- Coding answers define the named function and return it in a Python fenced block or as source text.
- A new benchmark run creates a new raw CSV. Existing raw results are protected unless `--overwrite` is explicitly supplied.

## Setup and execution (PowerShell)

From the project root, activate the existing environment and install the small declared dependency set:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run a small smoke benchmark first:

```powershell
python -m benchmarks.run_models --sample 2 --output results\smoke_results.csv
python -m benchmarks.evaluate --input results\smoke_results.csv
```

Run all 30 questions against both default models:

```powershell
python -m benchmarks.run_models
python -m benchmarks.evaluate
```

To deliberately replace the default raw results:

```powershell
python -m benchmarks.run_models --overwrite
```

Useful options include `--models qwen2.5:1.5b qwen2.5:3b`, `--sample 5`, `--request-timeout 180`, `--code-timeout 2`, `--base-url http://localhost:11434`, `--dataset data\benchmark_questions.json`, and `--output results\trial.csv`. The evaluator accepts `--input` and `--output-dir`.

Run unit tests without Ollama:

```powershell
python -m pytest -q
```

## Files

- `app/__init__.py` marks the inference package.
- `app/inference.py` implements the reusable Ollama HTTP client and structured success/failure records.
- `benchmarks/__init__.py` marks the benchmark utilities package.
- `benchmarks/dataset.py` loads and validates unique question IDs and evaluation specifications.
- `benchmarks/code_eval.py` extracts candidate code, validates a restricted AST, and evaluates deterministic cases in an isolated Python subprocess with a timeout.
- `benchmarks/run_models.py` runs each selected prompt against every selected model and flushes results to disk after each attempt.
- `benchmarks/evaluate.py` checks exact numeric answers and aggregates metrics into CSV and text reports.
- `data/benchmark_questions.json` contains 15 mathematical reasoning and 15 Python coding questions.
- `tests/test_inference.py` mocks HTTP and tests dataset/evaluation behavior without a running Ollama service.
- `requirements.txt` lists only runtime HTTP and test dependencies.
- `results/` is the output directory for raw and summarized benchmark records.

## CSV fields

`benchmark_results.csv` stores `timestamp_utc`, `model`, `question_id`, `category`, `evaluation_type`, `prompt`, raw `response`, request `status`, `correct`, `evaluation_error`, wall-clock seconds, prompt/generated token counts, Ollama total/load/prompt-evaluation/generation durations in nanoseconds, and `error_type`/`error_message`. Unavailable measurements and correctness for failed requests are blank. A generation may succeed while answer evaluation fails; that is recorded in `evaluation_error`, not as an inference failure.

`summary.csv` contains one row per model and category (including `all`), with attempt/valid/evaluated/correct counts, accuracy, average and median latency, nearest-rank p95 latency, average generated tokens, failure rate, and paired disagreement rate. Failed inference attempts are excluded from accuracy, but included in latency and the failure-rate denominator. Paired disagreement is the fraction of questions with at least two distinct models that have successful, evaluated correctness values and differ on correct versus incorrect. `summary.txt` renders the same metrics for quick reading. Exact numeric evaluation reads the last numeric token on the final non-empty response line; this matches the dataset's instruction to put the final numeric answer last, while ignoring numbers in preceding explanation lines.

## Security and interpretation limits

Generated Python is never executed in the main application process. Candidate source is restricted to top-level functions, a limited set of AST operations and built-ins, bounded source/AST size, and a two-second child-process timeout. This is risk reduction, not a hardened operating-system sandbox: Python subprocesses run with the current user's OS permissions, and this implementation does not enforce memory/CPU quotas or filesystem/network isolation. Only run benchmarks with models and prompts you trust; for hostile code, use a separately configured OS sandbox/container with resource limits.

The exact numeric evaluator reads the last numeric expression on the final non-empty response line; ambiguous formatting can produce false negatives. The coding suite covers only its listed deterministic cases. Latency and token counts are affected by hardware, model warmup, Ollama version, and concurrent load. Reports are measurements only and make no performance claim.

## Next phase handoff

Send back `results/benchmark_results.csv`, `results/summary.csv`, and `results/summary.txt`, plus the machine/GPU and memory details, Ollama version, exact model tags, any failed-request rows, and whether the runs were cold or warm. Keep the raw CSV so later analysis can inspect exact outputs and timing metadata. Do not infer a routing policy from summary accuracy alone.

## Benchmark Expansion

The separate 400-question research expansion lives under `data/processed/` and `data/splits/`. It does not replace or alter `data/benchmark_questions.json`, and it is not wired into the inference runner by default.

Sources and provenance:

- GSM8K: the [OpenAI grade-school-math repository](https://github.com/openai/grade-school-math) describes 8.5K human-written grade-school word problems and provides official `train.jsonl`/`test.jsonl` files. Its repository `LICENSE` is MIT. The expansion draws only from the pinned training file at commit `3101c7d5072418e28b9008a6636bde82a006892c`; it does not use GSM8K test rows. GSM8K does not publish an original per-question ID, so `source_question_id` is the stable one-based line number in that pinned training file, not an author-provided ID. The answer is extracted from the source solution's final `####` line; commas are removed from numeric answers and the prompt gets an answer-format instruction.
- MBPP: the [Google Research MBPP source](https://github.com/google-research/google-research/tree/master/mbpp) documents task IDs 601-974 as training, 511-600 as validation, 11-510 as test, and 1-10 as few-shot prompting. The [Google Research Datasets MBPP card](https://huggingface.co/datasets/google-research-datasets/mbpp) labels the data CC-BY-4.0. The containing [Google Research repository LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE) is Apache-2.0. These are distinct declarations; AURA records the data-card CC-BY-4.0 label and the repository Apache-2.0 license on each MBPP item rather than conflating them. Confirm applicable licensing with the dataset maintainers/legal reviewer before external redistribution. Only pinned task IDs 601-974 are considered, from commit `0eebfd403ba438bfd14cb70b9b24a7745c997083`.

The preparation script selects 200 GSM8K training questions and 200 eligible MBPP training questions. MBPP reference solutions are never imported or executed. Only standalone `assert function(literal_args) == literal_expected` cases with JSON-compatible literals are converted; tests requiring setup code, unsupported assertion forms, sets/tuple-key dictionaries or non-literal expressions are skipped. The MBPP prompt is retained, a short return-code instruction is appended, and the published test assertions become deterministic `test_spec` cases. No difficulty field is added because neither source supplies a compatible per-item difficulty rating and AURA does not create one.

Reproducibly prepare the expanded dataset and balanced splits (PowerShell, from the project root):

```powershell
python -m benchmarks.prepare_dataset --seed 2026
```

The script pins both upstream commits, caches GSM8K training data and only MBPP training-ID rows under `data/raw/`, and refuses to replace existing processed output unless asked. To deliberately regenerate the processed files with the same seed:

```powershell
python -m benchmarks.prepare_dataset --seed 2026 --overwrite
```

Validate the produced dataset and manifests:

```powershell
python -m benchmarks.validate_dataset
```

Run all tests, including the expansion validation tests:

```powershell
pytest -v
```

Splits use seed `2026` by default, after source-training-only selection and deduplication against the original 30 questions and earlier selected records across both categories. Deduplication lowercases text, replaces numeric tokens with `<number>`, and checks token-sequence similarity at a 0.92 threshold. Each category is shuffled reproducibly and allocated 120 training, 40 validation, and 40 test questions. Thus each split is balanced 50/50 by math/coding: train 240 (120/120), validation 80 (40/40), test 80 (40/40). Each `data/splits/*_manifest.json` stores its question IDs, size, category counts, and seed; the aggregate `manifest.json` also records source checksums, source-candidate exclusions, and duplicate counts. The held-out test split must not inform router training or threshold/model selection; validation is the only selection split.

Run inference on each expanded split separately (each command writes to a new result path and does not overwrite existing output):

```powershell
python -m benchmarks.run_models --dataset data\splits\train.json --output results\train_benchmark_results.csv
python -m benchmarks.evaluate --input results\train_benchmark_results.csv --output-dir results\train_summary

python -m benchmarks.run_models --dataset data\splits\validation.json --output results\validation_benchmark_results.csv
python -m benchmarks.evaluate --input results\validation_benchmark_results.csv --output-dir results\validation_summary

python -m benchmarks.run_models --dataset data\splits\test.json --output results\test_benchmark_results.csv
python -m benchmarks.evaluate --input results\test_benchmark_results.csv --output-dir results\test_summary
```

Do not use the test split during router training, feature iteration, prompt tuning, or threshold selection. Run test inference only after those choices are frozen.

See [data/processed/DATASET_CARD.md](data/processed/DATASET_CARD.md) for intended use, attribution, source license notes, and known limitations.