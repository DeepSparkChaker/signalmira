# SignalMira — Opinion extraction LangChain v2 (incremental patch)

This is an **add-on to your current project** and to the previous `signalmira_langchain_upgrade` package, not a replacement for your sentiment or retrieval engines.

## Files
- `enrichment/langchain_opinions.py` replaces the first-generation LangChain adapter (NOT `enrichment/opinions.py`). It keeps `OpinionConfig`, legacy Ollama digest checks, `extract`, `aextract`, `warmup`, and `config_hash` compatible with existing `sentiment_jobs.py`.
- `enrichment/opinion_service.py` provides a reusable `analyze_review` and streaming `analyze_many` for API, CSV, and future analyst tools. It has **no database write side effects**.
- `scripts/opinion_preview.py` shows a review or CSV output, without saving any data.
- `tests/test_opinion_v2.py` tests the bounded adapter with a fake model.

## Installation (from ~/projects/signalmira)

**First** install the previously supplied `signalmira_langchain_upgrade` package, including its `llm_providers/llm_factory.py`, or ensure your factory supports `provider_override`, `model_override`, `max_output_tokens`, `context_tokens`, `timeout_s`, and `ollama_base_url`.

Then from your Windows Downloads folder:

```bash
cd ~/projects/signalmira
unzip -o /mnt/c/Users/chaker/Downloads/signalmira_opinion_v2.zip -d .
uv add 'langchain>=1,<2' langchain-ollama langchain-openai langchain-anthropic
uv run python -m unittest discover -s tests -p 'test_opinion_v2.py' -v
```

**Important:** do not overwrite `opinion_schema.py`, `opinions.py`, `sentiment.py`, `retrieval/hybrid.py`, or the existing database migration. Keep your versioned data.

### Smoke test without writing to PostgreSQL

```bash
ollama list
export LLM_MODEL='qwen3.5:4b'
export LLM_REVISION="$(uv run python - <<'PY'
import json, os, urllib.request
with urllib.request.urlopen('http://127.0.0.1:11434/api/tags') as r:
    models=json.load(r)['models']
found=[m for m in models if m.get('name') == os.environ['LLM_MODEL']]
if len(found)!=1: raise RuntimeError('Model tag not found')
print(found[0]['digest'])
PY
)"

uv run python -m scripts.opinion_preview \
  --model "$LLM_MODEL" --revision "$LLM_REVISION" \
  --text 'The camera is excellent but the battery drains overnight.'
```

### Stream up to 10 CSV rows (does not load CSV into memory)

```bash
uv run python -m scripts.opinion_preview \
  --model "$LLM_MODEL" --revision "$LLM_REVISION" \
  --csv /path/to/reviews.csv --limit 10
```

CSV columns: `review_id,text` (UTF-8). Each output line has `tenant_id`, `review_id`, source-text SHA-256, selected model, model revision, extraction contract hash, and validated opinions.

### Enable in existing persisted job

In `enrichment/sentiment_jobs.py`, change **one import**:

```python
# Was:
from enrichment.opinions import OpinionExtractor, OpinionConfig
# Now:
from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor as OpinionExtractor
```

Keep the remaining job unchanged. A new `config_hash` version avoids mixing old and new extraction results.

```bash
export DATABASE_URL='postgresql://signalmira:local_dev_only@127.0.0.1:5432/signalmira'
uv run python -m enrichment.sentiment_jobs opinions \
  --tenant demo --llm-model "$LLM_MODEL" --llm-revision "$LLM_REVISION" --limit 10
```

Verify that the real installed model returns valid schema and correct quote offsets before processing bulk records.

## Architecture and important limitations

- Sentiment, language, NER, NED and topics remain independent. The LLM extracts **opinion-level** sentiment, customer-reported problems, and *explicit* customer suggestions; it **does not** invent root causes or business recommendations.
- The reusable service can be called from a future authenticated API or exposed as a narrowly scoped LangChain tool. The **Analyst** should generate recommendations later, with its own provenance and evidence records.
- `OpinionConfig` intentionally represents a **local Ollama model pinned by an installed 64-character digest**. A future provider-neutral extraction config must track OpenAI/Anthropic model IDs and server versions differently; don't falsely label their revision as an Ollama digest.
- One inference slot per **Python process**, not across containers. A real global backpressure queue and GPU-service admission control are required for Cloud Run scaling. `aextract()` does not block the event loop; cancelling it **does not** cancel a running GPU inference.
- LLM inference speed depends on the *serving engine*; this adapter alone does not add batched GPU inference. For real throughput use Ollama/vLLM/SGLang serving with measured batching, pinned quantizations and separate CPU sentiment workers.
- This is a **tested integration patch**, not a full production certification: integration and load tests still required for actual PostgreSQL RLS, real Ollama response formats, memory limits, multilingual accuracy, worker crashes and API/CSV authentication.
