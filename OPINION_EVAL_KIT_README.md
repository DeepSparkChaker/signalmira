# SignalMira multilingual opinion evaluation kit

**Purpose:** A real, human-labeled 150-review evaluation for your existing
`enrichment.langchain_opinions.LangChainOpinionExtractor`. No architecture changes,
no database writes and no paid API requests. Supports Qwen3.5 4B / 9B and the
existing `OpinionConfig`/`OpinionService`.

## Included

- `scripts/build_opinion_gold.py`: sample up to 150 **real reviews** from tenant-scoped
  PostgreSQL or an existing CSV; aim for 30 per `en,fr,es,ar,zh` when available.
  Uses streaming and reservoir sampling, O(sample size) RAM.
- `scripts/label_opinion_gold.py`: **human-only** labeling, crash-safe and resumable.
  Records review overall polarity, one or more English canonical aspects, each
  polarity, and whether a real problem or explicit customer suggestion exists.
  Optional original-language suggestion quote is prompted when present.
- `scripts/eval_opinion_gold.py`: run your actual opinion extractor; calculate
  accuracy, macro/micro F1, presence precision/recall and warmed p50/p95 latency.
  Per-language and dev/test results included.
- `scripts/compare_opinion_models.py`: compare 4B and 9B on exactly the same labels.
- `tests/test_opinion_eval_kit.py`: tests for matching and metric correctness.
- `tests/data/opinion_eval/demo_synthetic_gold.jsonl`: tiny synthetic **smoke-test**
  fixture; **NOT real data or evidence of model quality**.

**Important:** The zip does NOT contain 150 real labeled reviews, because we cannot
access your local PostgreSQL/CSV or truthfully create human judgments. The sampler
creates them from your own real corpus; a human must then annotate them.

## Setup

Unzip in `~/projects/signalmira` from WSL. It only adds new files. It does **not**
overwrite `opinions.py`, `opinion_schema.py`, the LLM factory, migrations or jobs.

```bash
cd ~/projects/signalmira
uv run python -m pytest -q tests/test_opinion_eval_kit.py
```

## 1. Sample real reviews (NO inference)

```bash
uv run python -m scripts.build_opinion_gold \
  --tenant demo --count 150 --seed 42 \
  --out tests/data/opinion_eval/real_reviews_150.jsonl
```

Alternative (real multilingual CSV; columns: `review_id,text,language`, optional `source`):

```bash
uv run python -m scripts.build_opinion_gold \
  --csv data/multilingual_real_reviews.csv --count 150 \
  --out tests/data/opinion_eval/real_reviews_150.jsonl
```

The report shows how many reviews from each language were actually available.
If your `demo` tenant has only English UCI-Amazon reviews, the file **cannot**
magically contain French/Spanish/Arabic/Chinese. Source real multilingual reviews
from an appropriately licensed corpus or additional client/test data.

The sample is stratified where possible, split ~20% `dev` / ~80% `test`
**before** labeling. `dev` is for prompt tuning; `test` is a held-out final check.
A low `--max-scanned` can make the sample biased; the default scans all source rows
with a PostgreSQL server-side cursor and bounded client memory.

**Privacy:** The JSONL contains original raw reviews; keep it private and outside
version control if real user information is present. The scripts never upload it.

## 2. Human label the dev sample, then hold-out test sample

```bash
uv run python -m scripts.label_opinion_gold \
  --file tests/data/opinion_eval/real_reviews_150.jsonl --split dev
```

A reviewer reads each original review and enters labels. Example:

```json
"gold": {
  "overall_sentiment": "mixed",
  "opinions": [
    {"aspect": "camera", "sentiment": "positive",
     "reported_problem": false, "customer_suggestion": false},
    {"aspect": "battery", "sentiment": "negative",
     "reported_problem": true, "customer_suggestion": true,
     "suggestion_quote": "Please fix it."}
  ]
}
```

Use canonical **English `snake_case`** aspects (e.g. `battery`, `payment`,
`customer_support`, `bluetooth_connection`), but keep suggestion quotes in the
original language. Use the same taxonomy for human gold across all languages;
otherwise exact aspect matching becomes meaningless.

Use `--split test` to label the held-out part **before** your final test run.
Don't inspect held-out results to change the prompt and then report it as an
unbiased test. On a commercial project, have two reviewers independently annotate
20-30% of examples and adjudicate disagreements.

## 3. Benchmark Qwen3.5-4B (existing Ollama installation)

```bash
uv run python -m scripts.eval_opinion_gold \
  --gold tests/data/opinion_eval/real_reviews_150.jsonl \
  --split dev --model "$LLM_MODEL" --revision "$LLM_REVISION" \
  --out tests/data/opinion_eval/qwen4b_dev.jsonl
```

The script reuses your **existing LangChain extractor and prompt**, warms it once,
then evaluates only human-labeled rows. It writes one result per row immediately.
If interrupted, rerun with `--resume`; the implementation refuses to resume with
another model, revision or prompt/config hash.

## 4. Benchmark Qwen3.5-9B separately

**Don't load the two models at the same time on a 12 GB GPU.**

```bash
ollama pull qwen3.5:9b
export QWEN9_SHA="$(curl -fsS http://127.0.0.1:11434/api/tags | \
  uv run python -c 'import json,sys; print(next(m["digest"] for m in json.load(sys.stdin)["models"] if m["name"]=="qwen3.5:9b"))')"

uv run python -m scripts.eval_opinion_gold \
  --gold tests/data/opinion_eval/real_reviews_150.jsonl \
  --split dev --model qwen3.5:9b --revision "$QWEN9_SHA" \
  --out tests/data/opinion_eval/qwen9b_dev.jsonl
```

Run separately. Quantization, context length and model serving settings impact
VRAM and measured latency. Ollama settings may keep a model resident for 5 minutes,
so use `ollama stop qwen3.5:4b` when switching if needed.

## 5. Compare models

```bash
uv run python -m scripts.compare_opinion_models \
  --gold tests/data/opinion_eval/real_reviews_150.jsonl \
  --split dev \
  --a tests/data/opinion_eval/qwen4b_dev.jsonl \
  --b tests/data/opinion_eval/qwen9b_dev.jsonl
```

The test report includes per-language scores and failure details:
`tests/data/opinion_eval/qwen4b_dev.jsonl.dev.summary.json`.

To re-score outputs without another LLM call:

```bash
uv run python -m scripts.eval_opinion_gold \
  --gold tests/data/opinion_eval/real_reviews_150.jsonl --split dev \
  --predictions tests/data/opinion_eval/qwen4b_dev.jsonl
```

## Metrics (definitions)

- **Overall sentiment:** review-level accuracy and macro F1 across gold-present
  classes (`negative`, `neutral`, `positive`, `mixed`).
- **Aspect extraction:** one-to-one **exact canonical English aspect** matching;
  micro precision, recall and F1.
- **Aspect+polarity:** both aspect and sentiment must match to count TP.
- **Aspect polarity on matched aspects:** accuracy and macro F1 **conditional on
  the aspect being found**. NOT end-to-end polarity quality.
- **Reported-problem presence:** precision/recall/F1 for whether the correct
  aspect has a reported problem. NOT semantic correctness of its English summary.
- **Customer-suggestion presence:** same for an explicit suggestion; false
  invented suggestions count as false positives.
- **Suggestion quote exact recall:** strict quote comparison (ignoring only outer
  punctuation/whitespace) when a human supplied an exact original-language quote.
- **Success rate:** extraction failures/invalid outputs reduce coverage and
  retrieval recall; they are not omitted from the denominator.
- **Latency p50/p95:** wall time of each warmed extraction including local
  Ollama transport and validation; excludes model warmup, DB and annotation time.
  Compare on the same machine, same concurrency and comparable load.

For missing metric support (no gold positive examples), precision/recall/F1 may
be `null` instead of falsely reporting 100%. With only 30 dev examples, the
reported macro F1 can be noisy; see the by-language counts.

## Preliminary release targets (proposed, NOT measured)

| Metric | Initial target |
|---|---:|
| Valid schema + verifiable quotes | >=99% |
| Aspect micro F1 | >=0.85 |
| Aspect+polarity F1 | >=0.80 |
| Overall sentiment macro F1 | >=0.80 |
| Problem presence recall | >=0.85 |
| Suggestion presence precision | >=0.95 |
| Suggestion presence recall | >=0.85 |
| p95 extraction latency | Set after warm 4B measurement on your GPU |

Treat these as **engineering objectives**. Quality thresholds must be checked
by language; good global scores can hide poor Arabic/Chinese performance.
For semantic correctness of problem descriptions and suggested fixes, manually
review model outputs, or add adjudicated fine-grained human labels later.

## Public data for an independent benchmark

The [M-ABSA dataset](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA)
contains aspect term/category/polarity triplets across 21 languages, including
our five, and is marked Apache-2.0 on the dataset card. It is a useful **separate
public ABSA evaluation**. It does **not** annotate our custom `reported_problem`
and `customer_suggestion` fields, so **don't use it as if those were gold
negative labels**. Its parallel samples are translated and human-reviewed, not
independent native customer reviews in all languages.

The [SemEval-2016 Task 5 benchmark](https://alt.qcri.org/semeval2016/task5/)
also contains multilingual manually labeled aspect/sentiment datasets; its
annotation ontology differs from SignalMira's. Check individual data license
terms before redistributing or commercial use.
