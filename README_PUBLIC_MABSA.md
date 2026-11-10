# SignalMira: public multilingual labeled evaluation (no manual annotation)

M-ABSA is a publicly released dataset with **aspect term, category and sentiment polarity** labels for 21 languages across 7 domains. This script uses the **phone / test** split for **English, French, Spanish, Arabic and Chinese**, sampling 30 reviews from each (150 total). License listed as Apache-2.0. Paper: https://aclanthology.org/2025.emnlp-main.128/ ; dataset: https://huggingface.co/datasets/Multilingual-NLP/M-ABSA.

The dataset uses machine translation plus human review. Its labels are NOT exactly the same as SignalMira's opinion schema, so the scorer compares **coarse aspect category family** and **coarse category + polarity** using the visible alias dictionary in the script. This is a **proxy quality test**, not exact span-level triplet F1. Unknown source categories cause examples to be excluded, while unknown model aspects count as false positives. Source classes are broad. Edit the mapping only on a development split before final test, never on the results you report as held out.

**This dataset does not have gold truth for** overall sentiment, reported problems, customer suggestions, suggestion attachment, or original Unicode offsets. We intentionally do NOT manufacture labels or claim those metrics. To evaluate those particular fields, obtain a different labeled dataset (SemEval-2019 Task 9 has suggestion-present labels, not exact suggestion-to-aspect linking), or annotate a modest sample from your real business domain.

## Install

Extract this ZIP at the project root, preserving the `scripts/` and `tests/` folders.

```bash
cd ~/projects/signalmira
uv run python -m pytest -q tests/test_eval_public_mabsa.py
```

## Download and prepare a fixed published test sample (no DATABASE_URL)

```bash
uv run python -m scripts.eval_public_mabsa --prepare-only --per-language 30
```

Outputs:
- `tests/data/opinion_eval/mabsa_phone_150.jsonl` original real reviews and the public category/polarity labels mapped to broad aspect families
- `tests/data/opinion_eval/mabsa_phone_150.metadata.json` resolved immutable dataset SHA and exclusion counts

## Evaluate local 4B

```bash
uv run python -m scripts.eval_public_mabsa \
  --model "$LLM_MODEL" --revision "$LLM_REVISION" \
  --predictions tests/data/opinion_eval/mabsa_qwen4b.jsonl
```

This invokes **your existing** LangChain extractor and prompt, one model call per review. Results saved to `mabsa_qwen4b.jsonl` and `mabsa_qwen4b.summary.json`. No PostgreSQL writes. Keep the model hot; p95 is per-review warmed extraction, not throughput under concurrent load.

## Evaluate 9B on the same reviews (sequentially, NOT both resident on the 12GB GPU)

```bash
ollama pull qwen3.5:9b
export QWEN9_SHA="$(curl -fsS http://127.0.0.1:11434/api/tags | uv run python -c 'import json,sys; print(next(m["digest"] for m in json.load(sys.stdin)["models"] if m["name"]=="qwen3.5:9b"))')"
ollama stop qwen3.5:4b
uv run python -m scripts.eval_public_mabsa \
  --model qwen3.5:9b --revision "$QWEN9_SHA" \
  --predictions tests/data/opinion_eval/mabsa_qwen9b.jsonl
```

To re-score predictions without GPU:

```bash
uv run python -m scripts.eval_public_mabsa --rescore --predictions tests/data/opinion_eval/mabsa_qwen4b.jsonl
```

**Do not treat a high coarse-category score as proof** the model gets precise aspects, reported issues, or customer suggestions right. Your previous real-world scenarios still require those additional gold annotations.
