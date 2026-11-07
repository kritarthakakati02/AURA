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