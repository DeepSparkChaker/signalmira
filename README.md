# SignalMira — Step 2: streaming ingestion → PostgreSQL

**Only raw review ingestion and storage. No AI, embeddings, or RAG.**

## Quick start (WSL with Docker)

```bash
uv sync
docker compose up -d --wait db
export DATABASE_URL='postgresql://signalmira:local_dev_only@127.0.0.1:5432/signalmira'
uv run python -m storage init
uv run python -m storage import data/sample_reviews.csv --tenant demo --source-key demo-csv
uv run python -m storage get --tenant demo --source-key demo-csv --review-id r-001
```

Read-back includes all columns: raw text, company, product, date, URL, JSONB attributes, hash and timestamps.

Repeat the import and verify **written: 0** and **skipped/unchanged: 2**. JSONL scraper example:

```bash
uv run python -m storage import data/sample_scraped.jsonl --tenant demo --source-key scraper
uv run python -m storage get --tenant demo --source-key scraper --review-id web-001
```

## How it scales

```text
CSV / JSONL generator ──► batches of 500 ──► PostgreSQL TEMP staging (COPY)
                                                │
                                                └──► one set-based UPSERT
                                                      into feedback_reviews
```

- Memory is bounded by **one CSV/JSONL record and the configured batch**, not the full file.
- `COPY` reduces client/server round-trips; `ON CONFLICT` inserts new rows, updates changed rows, and skips unchanged data.
- Rows are unique by `(tenant_id, source, review_id)`. Inside a batch, last duplicate wins; skipped count includes duplicates.
- One transaction per batch; if a later batch fails, already committed batches remain. Safe to rerun.
- Additional metadata is stored in `attributes JSONB`. Small `.json` arrays are supported (10 MB limit); for big exports use streamed `.jsonl`.

Tune after measuring with your actual review sizes and Cloud SQL network latency:

```bash
uv run python -m storage import data/reviews.csv --tenant demo --source-key partner-a --batch-size 1000
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
TEST_DATABASE_URL="$DATABASE_URL" uv run python -m unittest discover -s tests -v
```

`TEST_DATABASE_URL` enables live PostgreSQL tests for insert/read-back, idempotency, updates, tenant/source IDs, JSONB metadata, duplicate IDs, and multi-batch replay. They require a running database and were **not executed in the packaged build environment**.

## Cloud SQL later

The Python import functions work with a Cloud SQL PostgreSQL DSN; configure secrets and networking when deploying a Cloud Run Job. Do not use the local development password in cloud. For multi-customer production deployments, still implement **authenticated tenant resolution, PostgreSQL RLS, schema migrations, monitoring/alerts, retries/dead-letter handling, backups, and ingestion version tracking**. Don't expose this trusted CLI as a public API.
