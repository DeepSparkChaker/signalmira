# SignalMira — full BEIR NFCorpus PostgreSQL benchmark

This kit tests **your actual** `indexing.chunk_store`, `retrieval.semantic`,
`retrieval.hybrid.search_vector`, and `retrieval.reranker.Reranker`.
Unlike the previous TF-IDF benchmark, all 3,633 NFCorpus documents are indexed
in real PostgreSQL with **pg_textsearch BM25 + Qwen3 embeddings + pgvector**.
Your original SQL fuses BM25 top 40 and dense top 40 by RRF (k=60) to top 25.
Qwen or BGE reranks these top 25 chunks; the benchmark evaluates **8 unique
BEIR document IDs**, not eight possibly-duplicated chunks.

**Required:** existing SignalMira checkout, local `signalmira-db-1` container,
`pg_textsearch` and `vector`, your earlier `scripts.prepare_beir_eval` and
`scripts.eval_retrieval_rankings` scripts, `uv`, `DATABASE_URL` configured.

## 1. Copy kit into your Linux checkout

From `~/projects/signalmira`:

```bash
unzip -o signalmira_full_beir_pg_benchmark.zip
uv run python -m unittest discover -s tests -p test_beir_pg_full_eval.py -v
```

This package **adds only** `scripts/beir_pg_full_eval.py`,
`tests/test_beir_pg_full_eval.py`, and `BEIR_FULL_BENCHMARK_README.md`. It does not overwrite existing retrieval files.

## 2. Download and prepare the genuine BEIR data

```bash
uv run python -m scripts.prepare_beir_eval --dataset nfcorpus --limit 100
```

This creates `tests/data/beir_eval/nfcorpus/{corpus.jsonl,queries.jsonl,qrels.tsv}`.
It saves the full 3,633-document corpus and 100 real test queries. Do not
subsample the corpus just because you evaluate 100 questions.

## 3. Create a separate, EMPTY evaluation database

**DO NOT use `CREATE DATABASE ... TEMPLATE signalmira`:** it would copy all
of your `demo` data. Instead, use the schema-only clone below. Run it while
the main local database is in a consistent state and make sure the existing
local backup is retained. This does **not** stop or modify `signalmira`.

```bash
mkdir -p ~/signalmira_backups
# Create an empty database; do this only once.
docker exec signalmira-db-1 \
  psql -v ON_ERROR_STOP=1 -U signalmira -d postgres \
  -c 'CREATE DATABASE signalmira_beir OWNER signalmira;'

# Export schema *only*: NO customer review data included.
docker exec signalmira-db-1 \
  pg_dump -U signalmira -d signalmira --schema-only -Fc \
  > ~/signalmira_backups/signalmira_schema_only.dump

# Restore schema and existing index definitions into isolated DB.
docker exec -i signalmira-db-1 \
  pg_restore -U signalmira -d signalmira_beir \
  --no-owner --no-acl --exit-on-error \
  < ~/signalmira_backups/signalmira_schema_only.dump
```

If `signalmira_beir` already exists, **do not recreate it or restore twice**.
Schema-only cloning preserves the same table constraints/migrations and BM25/
HNSW index definitions while keeping data empty. It may take time because it
recreates schema objects and index definitions. Any restore failure must be
resolved before continuing.

## 4. Switch DATABASE_URL in this terminal only

From your WSL shell:

```bash
# Use your existing local Docker development credentials.
# Do not put a production/Cloud SQL URL here.
export DATABASE_URL='postgresql://signalmira:local_dev_only@127.0.0.1:5432/signalmira_beir'

uv run python -m scripts.beir_pg_full_eval check
```

The script refuses every database **except** `signalmira_beir`. Be careful:
subsequent commands in this shell (including your existing Streamlit app) will
use the evaluation database until you restore the original DATABASE_URL.

## 5. Import all real documents as reviews, then chunk and embed

```bash
uv run python -m scripts.beir_pg_full_eval ingest

EMBED_SHA=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
uv run python -m scripts.beir_pg_full_eval index --revision "$EMBED_SHA" --device cuda

uv run python -m scripts.beir_pg_full_eval check --revision "$EMBED_SHA"
```

`ingest` writes source documents to `feedback_reviews` under tenant `beir_eval`
with source `beir-nfcorpus` and original BEIR doc IDs as `review_id`.
`index` calls your existing incremental chunking and embedding functions. It
reuses one pinned, warmed Qwen3 embedding model. BM25 indexes new chunks
on insertion automatically. Both `ingest` and `index` can be resumed after a
failure. **Run only one index job at a time**.

Expected check: 3,633 feedback reviews, at least 3,633 chunks, and a
matching embedding count equal to chunk count. The actual chunk count can
be higher than document count because the longer documents are chunked.

If ingestion fails due to extra non-nullable columns in your particular
`feedback_reviews` migration, inspect that schema. This kit assumes the
columns `tenant_id, source, review_id, text, content_hash, attributes` and
server-side defaults for all other required columns. Do **not** bypass
constraints or RLS to make a test pass.

## 6. Evaluate the actual SQL hybrid baseline and Qwen

Qwen reranker SHA from your successful prior run:
`e61197ed45024b0ed8a2d74b80b4d909f1255473`.

```bash
export EMBED_SHA=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
export QWEN_SHA=e61197ed45024b0ed8a2d74b80b4d909f1255473

# Actual hybrid SQL alone: BM25 + exact vectors + RRF
uv run python -m scripts.beir_pg_full_eval eval \
  --revision "$EMBED_SHA" --device cuda \
  --mode exact --reranker off \
  --output tests/data/beir_eval/pg_rrf.jsonl

# The same hybrid SQL, then real Qwen cross-encoder reranking
uv run python -m scripts.beir_pg_full_eval eval \
  --revision "$EMBED_SHA" --device cuda \
  --mode exact --reranker qwen --reranker-revision "$QWEN_SHA" \
  --output tests/data/beir_eval/pg_qwen.jsonl
```

Results include baseline RRF @8, reranked @8, candidate Recall@25,
embedding/PostgreSQL/reranking p50/p95, and all ranked document IDs in JSONL.
These queries execute against the **entire indexed NFCorpus**, not just 100
preselected docs. Benchmark script refuses incomplete indexes.

## 7. Compare BGE (optional, only if cached)

```bash
export BGE_SHA="$(uv run python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("BAAI/bge-reranker-v2-m3").sha)')"
uv run python -m scripts.beir_pg_full_eval eval \
  --revision "$EMBED_SHA" --device cuda --mode exact \
  --reranker bge --reranker-revision "$BGE_SHA" \
  --output tests/data/beir_eval/pg_bge.jsonl
```

Only one selected reranker is loaded for each invocation. The experiment
is sequential; don't launch Qwen and BGE simultaneously on a 12GB GPU.

## 8. Compare HNSW against exact

```bash
uv run python -m scripts.beir_pg_full_eval eval \
  --revision "$EMBED_SHA" --device cuda --mode hnsw \
  --reranker qwen --reranker-revision "$QWEN_SHA" \
  --output tests/data/beir_eval/pg_hnsw_qwen.jsonl
```

`--mode exact` is the accuracy reference. PostgreSQL may scan vectors exactly
for small datasets even with the HNSW mode selected: check `EXPLAIN` before
claiming that an ANN index was used.

## Safety and interpretation

- This is **an offline local benchmark**, not a production RLS penetration test:
  the `signalmira` database role may bypass row-level security.
- `--allow-shared-idf` applies to this isolated benchmark only. It does not
  establish tenant-safe BM25 IDF for the live SaaS product.
- Do not share customer data between benchmark and real databases.
- BEIR assesses **documents**. Your chunking policy affects candidate quality
  and rerank opportunity. This adapter deduplicates by BEIR doc ID before
  scoring to avoid treating multiple chunks from the same doc as separate hits.
- These published judgments can be incomplete: unjudged documents are treated
  as irrelevant by the evaluation implementation.
- No accuracy is claimed until the above commands run on your own database.

Sources: https://github.com/beir-cellar/beir ;
https://github.com/timescale/pg_textsearch ;
https://github.com/pgvector/pgvector
