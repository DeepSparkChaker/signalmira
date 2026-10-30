# SignalMira — Optimized NER (drop-in replacement)

**Replace only `enrichment/ner.py` in your current project.** This retains the tables `feedback_ner_state`, `feedback_entities` and reads your existing `feedback_reviews`. No changes to `ingestion/`, `storage/`, `ned.py`, or `topics.py`.

## What it does

`feedback_reviews` → skip unchanged records → GLiNER2.5 (or spaCy) in bounded batches → **one PostgreSQL transaction per page** → `feedback_entities` and `feedback_ner_state`.

- Exact model *snapshot commit SHA*, including in `feedback_ner_state.model_revision` and the config fingerprint; offline supported.
- GLiNER2.5 Base default; Small, Multi or spaCy Fast with optional `EntityRuler` phrase rules.
- CUDA FP16, `torch.inference_mode()`, length-sorted bounded batches, optional warm-up and benchmark.
- Long inputs use official `extract_entities_long()`; preserve exact spans, confidences and nested mentions.
- Psycopg `COPY` into a temporary per-session staging table; set-based entity replacement; one transaction per page; source rows locked and rechecked to avoid stale results.
- Existing outputs are overwritten if you switch profile/configuration for the same tenant/review, as in the original schema. If customers need simultaneous multiple configurations, version the entity tables in a separate migration.

## Install once (inside WSL SignalMira)

```bash
uv add --optional ner 'gliner2[local]' spacy 'psycopg[binary]' huggingface-hub
uv run --extra ner python -m spacy download en_core_web_sm
```

Optional advanced attention (after benchmarking the default):

```bash
uv add --optional ner 'gliner2[disentangled-flash]'
```

Keep the model cache and `.venv` on the Linux filesystem rather than a mounted `/mnt/c` drive when possible.

## Obtain the actual model revision (connected to internet, once)

```bash
uv run --extra ner python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
path = snapshot_download('fastino/gliner2.5-base-v1')
print('Commit SHA:', Path(path).name)
PY
```

Copy this 40-character SHA into the `<COMMIT_SHA>` placeholder below. Using `--offline` requires a complete cached snapshot for that revision. For Cloud Run, preload the snapshot into the image or a mounted persistent model directory instead of downloading on every job.

## Commands

```bash
# Apply local bootstrap / compatible new model_revision field + lookup index
uv run --extra ner python -m enrichment.ner --init

# One pasted review (online first call; then offline with pinned SHA)
uv run --extra ner python -m enrichment.ner \
  --text 'My Google Pixel 9 battery is excellent.' \
  --profile accurate --device cuda

# Begin with 10 source reviews; keep everything else untouched
uv run --extra ner python -m enrichment.ner \
  --tenant demo --source uci-amazon --limit 10 \
  --profile accurate --device cuda --batch-size 8 --page-size 128 \
  --revision '<COMMIT_SHA>' --offline

# Benchmark actual batch throughput (results may vary)
uv run --extra ner python -m scripts.benchmark_ner \
  --device cuda --batch-size 8 --warmup 5 --runs 30 \
  --revision '<COMMIT_SHA>' --offline
```

For long-lived workers, instantiate `NER(Config(...))` once per process, warm it up once, and reuse it. Do **not** create a new model per client request. Batching across authenticated tenants requires additional queueing, per-job permissions and rate controls; this CLI provides tenant-scoped offline jobs, not a complete multi-tenant service.

## Check entities in PostgreSQL

```bash
docker compose exec -T db psql -U signalmira -d signalmira -c \
"SELECT e.source, e.review_id, e.label, e.text, e.confidence, s.model_revision
 FROM feedback_entities e
 JOIN feedback_ner_state s USING (tenant_id, source, review_id)
 JOIN feedback_reviews r USING (tenant_id, source, review_id)
 WHERE e.tenant_id='demo' AND s.content_hash=r.content_hash
 ORDER BY e.source, e.review_id, e.mention_index LIMIT 20;"
```

**RAG filter:** use `(tenant_id, label, lower(text), source, review_id)` with *verified* tenant ID, and join state/originals on matching `content_hash` to exclude stale entities after a review update. An exact entity filter can reduce recall if the NER missed an entity; use a vector-only fallback for low-confidence queries. Never treat NER confidence as a calibrated probability.

## Test

```bash
PYTHONPATH=. uv run --extra ner python -m unittest discover -s tests -p 'test_ner.py' -v
# Run only when you have a live local Postgres database, and verify this is a disposable test DB.
TEST_DATABASE_URL="$DATABASE_URL" PYTHONPATH=. uv run --extra ner \
  python -m unittest discover -s tests -p 'test_postgres_ner.py' -v
```

The local DDL bootstrap is not a replacement for migrations, PostgreSQL Row-Level Security, queue backpressure, end-to-end labeled NER accuracy tests, or deployment hardening. Pin dependency versions through `uv.lock` and enforce tenant authorization server-side.
