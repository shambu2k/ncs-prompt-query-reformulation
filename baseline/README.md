# CodeXGLUE AdvTest Baseline

This folder contains a lightweight baseline for the official CodeXGLUE
`Text-Code/NL-code-search-Adv` task.

Why this baseline exists:

- The official task repo is cloned in `CodeXGLUE/`.
- The official evaluator is used unchanged.
- The official CodeBERT training script targets very old PyTorch/Transformers
  versions and is not a good first runnable baseline on Apple Silicon CPU.
- Your project email thread explicitly lists BM25 as an acceptable baseline, so
  this gives you something reproducible now while you decide how far to push
  CodeBERT or PromptBreeder later.

## Files

- `prepare_codexglue_adv.py`: rebuilds `valid.jsonl` and `test.jsonl` from the
  bundled `dataset.zip` without downloading the full training corpus.
- `run_bm25_baseline.py`: runs a pure-Python BM25 retriever and writes
  CodeXGLUE-format predictions.

## Commands

Create the small local environment used for the official evaluator:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy
```

Prepare the split files:

```bash
.venv/bin/python baseline/prepare_codexglue_adv.py
```

Run BM25 on validation:

```bash
.venv/bin/python baseline/run_bm25_baseline.py --split valid
.venv/bin/python CodeXGLUE/Text-Code/NL-code-search-Adv/evaluator/evaluator.py \
  -a baseline/data/valid.jsonl \
  -p baseline/out/valid.predictions.jsonl
```

Run BM25 on test:

```bash
.venv/bin/python baseline/run_bm25_baseline.py --split test
.venv/bin/python CodeXGLUE/Text-Code/NL-code-search-Adv/evaluator/evaluator.py \
  -a baseline/data/test.jsonl \
  -p baseline/out/test.predictions.jsonl
```

## Notes

- This baseline uses `docstring_tokens` as the natural-language query and
  `function_tokens` as the code document.
- It is retrieval-only and training-free.
- PromptBreeder is cloned in `PromptBreeder/`, but that repo currently targets
  Cohere and is not wired into CodeXGLUE yet.
- Results from this setup on this machine:
  - `valid` MRR: `0.0518`
  - `test` MRR: `0.0421`
