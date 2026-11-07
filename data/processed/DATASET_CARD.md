# AURA Benchmark Expansion Dataset Card

## Summary

This is a fixed 400-question evaluation dataset for comparing local language-model responses on grade-school mathematical reasoning and entry-level Python function generation. It is intended for research measurement and future routing-policy development, not as a general measure of intelligence or software engineering ability. The original 30-question benchmark remains at `data/benchmark_questions.json` and is not changed by expansion preparation.

## Composition

- 200 mathematical reasoning questions sourced from GSM8K's training file.
- 200 small Python coding questions sourced from eligible MBPP training tasks.
- Train: 240 questions, 120 per category.
- Validation: 80 questions, 40 per category.
- Test: 80 questions, 40 per category.
- No `difficulty` value is emitted: neither selected source provides a supported common difficulty label.

`question_id` is a stable AURA ID. For MBPP, `source_question_id` preserves the published integer task ID. GSM8K has no original question ID in its JSONL; AURA records `train-line-NNNNN`, the one-based source line within the pinned training file, as a documented surrogate. It must not be presented as a native GSM8K identifier.

## Sources, License, and Transformations

**GSM8K**

- Source: OpenAI, [grade-school-math repository](https://github.com/openai/grade-school-math), [paper](https://arxiv.org/abs/2110.14168).
- License verified: MIT in the upstream repository's `LICENSE` file: [license text](https://github.com/openai/grade-school-math/blob/master/LICENSE).
- Pinned data source: `grade_school_math/data/train.jsonl` at commit `3101c7d5072418e28b9008a6636bde82a006892c`.
- Provenance: per-question `source` records link the repository and license and record the commit and transformation; `source_question_id` gives the stable train-file line number.
- Transformation: preserve the source question, append a final-answer formatting instruction, read the number after the last `####` marker from the published worked solution, remove thousands separators, and use exact numeric evaluation. The published solution body is not included in the processed question record.
- AURA excludes the official GSM8K test partition from the expanded dataset.

**MBPP**

- Source: Google Research, [MBPP repository directory](https://github.com/google-research/google-research/tree/master/mbpp), [paper](https://arxiv.org/abs/2108.07732), and [Google Research Datasets card](https://huggingface.co/datasets/google-research-datasets/mbpp).
- The Google Research Datasets card explicitly lists **CC-BY-4.0** for MBPP data. The containing [Google Research repository LICENSE](https://github.com/google-research/google-research/blob/master/LICENSE) is **Apache-2.0**. These declarations are not silently treated as interchangeable. Per-record metadata retains both labels, with CC-BY-4.0 as the data-card license declaration. Legal review/maintainer confirmation is still recommended before redistributing this derived dataset outside this research repository.
- Pinned source: `mbpp/mbpp.jsonl` at commit `0eebfd403ba438bfd14cb70b9b24a7745c997083`. The source README defines task IDs 601-974 as training, 511-600 as validation, 11-510 as test, and 1-10 as prompting examples. Only training IDs are considered. No MBPP validation, test, or prompting task is used.
- Transformation: preserve task text and source ID; append a concise code-return instruction; structurally parse, but do not execute, the published tests; retain only simple direct function-call equality assertions whose arguments and expected values are literal and JSON-compatible. Setup-code tests and unsupported Python expressions are skipped. Reference solutions are not placed in processed records or run by preparation.
- Attribution: cite “Mostly Basic Python Problems (MBPP), Google Research, Austin et al. (2021), https://arxiv.org/abs/2108.07732; source https://github.com/google-research/google-research/tree/master/mbpp; data-card license CC-BY-4.0.” AURA modifications include test-case conversion, prompt-format addition, selection, and splitting.

## Preparation and Splitting

Run `python -m benchmarks.prepare_dataset --seed 2026`, then `python -m benchmarks.validate_dataset`. Source commit hashes and SHA-256 checksums are recorded by the aggregate split manifest. The source selection uses only GSM8K train rows and eligible MBPP training task IDs. Selection removes exact normalized duplicates and near duplicates (token SequenceMatcher ratio at least 0.92), including comparisons across both categories and against the existing 30-question dataset. It uses a fixed random seed and category-specific shuffles to allocate 120/40/40 math and 120/40/40 coding questions to train/validation/test. The split manifests hold per-question IDs; validation rejects overlap or any manifest/data mismatch.

The test set is for final held-out evaluation only. Do not use its examples, answers, or measured outcomes for model/router training, feature development, threshold selection, or iterative prompt tuning. Use training for training and validation for selection; report test performance only after choices are frozen.

## Evaluation

Math uses exact numeric comparison against extracted source answers. Formatting errors, ambiguous unit conventions, or flawed source solutions can lead to incorrect labels. Coding uses deterministic published MBPP tests, not an LLM judge. Tests are translated from AST-parsed assertions and candidate-generated code remains subject to AURA's restricted AST validation and timed child-process execution. This is not a hardened operating-system sandbox.

## Limitations and Biases

- GSM8K represents English-language grade-school arithmetic word problems; it is not representative of advanced mathematics or all cultures and contexts.
- MBPP is a small, crowdsourced, entry-level function-generation collection. Some prompts/tests are underspecified, test coverage is sparse, and source examples can contain errors or rely on assumptions not fully stated in prompts.
- The MBPP test conversion deliberately excludes setup-dependent and non-literal tests; the resulting subset may overrepresent simple, self-contained tasks. The exact excluded count is reported by preparation.
- The near-duplicate filter is heuristic. It can miss semantic duplicates or exclude legitimate related questions. Similarity is reported as an algorithm, not a guarantee of semantic independence.
- The category-balanced fixed-seed split does not establish a universal distribution and does not guarantee source-author difficulty balance.
- Public benchmark contamination may affect model results. The source's historical split ranges are not the AURA split manifests; MBPP tasks included here come only from the source's designated training partition.
- Latency depends on hardware, warmup, model/runtime versions, and system load. Dataset correctness and benchmark metrics do not establish adaptive-routing improvements.

## Intended Use and Out-of-Scope Use

Use for reproducible offline measurement of local model accuracy, latency, resource/timing metadata, and correctness disagreement. Do not use as a sole deployment safety evaluation, a broad capability ranking, a claim of performance improvement, or evidence that a routing algorithm is effective. No router or classifier is implemented by this dataset infrastructure.
