# SignalMira — Semantic RAG, Step 5 (no BM25 yet)

Adds **two small production-oriented Python modules and one SQL migration** to the existing SignalMira root:
`retrieval/embedders.py`, `retrieval/semantic.py`, and
`migrations/004_rag_embeddings.sql`, plus two tests. The original review,
NER, NED, topics, language and chunking modules are unchanged.

## Architecture

`feedback_reviews -> rag_chunks (existing 1024 char/64 overlap) -> Qwen3
Embedding-0.6B -> rag_embeddings vector(1024) -> exact cosine search`.
Optional HNSW is a separate admin command. BM25 + RRF + reranker will come next.

- **Versioning**: every vector stores exact Hugging Face model commit SHA,
  backend, model contract hash, and original review hash. Different models and
  revisions have different `model_key` values; searches never mix them.
- **Secure ownership**: `rag_embeddings` has `(tenant_id, chunk_id)` composite FK
  with ON DELETE CASCADE; RLS `USING` and `WITH CHECK` fail closed without tenant
  context. App SQL still requires authenticated tenant authorization and a
  restricted PostgreSQL role without SUPERUSER/BYPASSRLS.
- **Bounded memory**: keyset pages (64), inference batches (8), no collection-wide
  Python list, lazy model/GPU, no chunk-text cache and no per-batch GPU cache flush.
  Embedding runs outside DB transactions; writes verify current source hashes.
- **Multilingual**: one Qwen3 model for every language, document text unmodified,
  English task instruction only on query. Search reads language from
  `feedback_language` if current, falling back to the chunk metadata. Topic
  embeddings remain separate.
- **Local performance**: PyTorch CUDA FP16 if available; CPU FP32 fallback. vLLM
  optional for a dedicated trusted local inference server; its checkpoint must
  be pinned and its weights manually verified.

## 1. Install dependencies (inside SignalMira WSL root)

```bash
uv add "sentence-transformers>=3.0" "transformers>=4.51.0" numpy
# psycopg already exists in SignalMira; no pgvector Python driver is needed:
# vector values are validated and passed as parameters, cast by Postgres.
```

**Database requirement:** existing `feedback_language` table from Step 4 and
server-side `pgvector` extension installed (PostgreSQL 17+
preferred). Check from WSL:

```bash
docker compose exec db psql -U signalmira -d signalmira -c \
  "SELECT name, default_version FROM pg_available_extensions WHERE name='vector';"
```

If no row: your current PostgreSQL Docker image lacks pgvector. Install the
extension for the **same PostgreSQL major version**, or use a compatible
pgvector image (for example `pgvector/pgvector:pg17` for PG17). **Back up DB
and check major version before changing an existing Docker image.**

## 2. Initialize existing chunks, then vectors

```bash
# Only if you haven't already applied Step 2:
uv run python -m indexing.chunk_store init
uv run python -m indexing.chunk_store sync --tenant demo --batch-size 100
# New additive Step 5 schema (requires DB schema owner/admin account):
uv run python -m retrieval.semantic init
# Embeds all missing chunks; first run downloads model weights.
uv run python -m retrieval.semantic sync --tenant demo --batch-size 8 --page-size 64
# Second run should show written: 0, assuming no source updates.
uv run python -m retrieval.semantic sync --tenant demo --batch-size 8 --page-size 64
```

On the first model download, the Hugging Face snapshot is resolved to an
immutable commit SHA and stored in the vector metadata. Future runs using
`--revision <sha>` keep the exact checkpoint. Set `--device cpu` for CPU only,
`--device cuda` to **require** a working CUDA install, or omit for auto/CUDA.
For previously cached model use `--offline`. On a 12 GB GPU, start with
`--batch-size 8`; decrease to `--batch-size 2` if any OOM occurs. Do not
change the embedding model/revision between indexing and search.

## 3. Search and add HNSW only if beneficial

```bash
uv run python -m retrieval.semantic search --tenant demo \
  --query 'Battery drains quickly and phone overheats' --k 10
# Exact search is the default, even after HNSW is installed.
# Separate schema-owner/admin command; pgvector >=0.8:
uv run python -m retrieval.semantic hnsw
uv run python -m retrieval.semantic search --tenant demo \
  --query 'Battery drains quickly and phone overheats' --mode hnsw --k 10
uv run python -m retrieval.semantic benchmark --tenant demo \
  --query 'Battery drains quickly and phone overheats' --repeats 5
```

Benchmark reports **agreement with exact top-k** and SQL search latency,
NOT human relevance/accuracy. PostgreSQL may still choose an exact scan in
`--mode hnsw` for very small tables; inspect `EXPLAIN (ANALYZE, BUFFERS)`
when comparing ANN engine performance. With a few thousand chunks, **exact** is often
faster and more reliable; HNSW can lose filtered recall with many tenants
sharing its graph. Benchmark using different tenants, products and languages.

## 4. Optional dedicated vLLM server (not required locally)

Qwen and vLLM support this model in embedding/pooling mode. Serve one pinned
checkpoint on a **trusted local server** (model weights shared, review data not).
Choose a vLLM version compatible with your CUDA/PyTorch GPU setup:

```bash
# Example; use the same 40-hex revision in the server and client.
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --runner pooling --revision "$MODEL_SHA" \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --max-model-len 2048 --gpu-memory-utilization 0.65
uv add openai
uv run python -m retrieval.semantic sync --tenant demo \
  --backend vllm --revision "$MODEL_SHA" --batch-size 8
```

vLLM embeddings are saved under a **different model_key** from local
SentenceTransformers embeddings, intentionally, until you verify numeric
compatibility; don't combine the two search contracts. Never transmit private
review text to an untrusted or public vLLM endpoint. Only use an approved local
or authenticated internal HTTPS service.

## 5. Tests

```bash
uv run python -m pytest tests/test_semantic.py -q
# Optional *separate disposable database*; never use production database:
# The DB name MUST end with `_test`. Requires its own pgvector installation.
export TEST_DATABASE_URL='postgresql://.../signalmira_test'
uv run python -m pytest tests/test_semantic_postgres.py -q -rs
```

## Production boundary (not optional)

Your earlier audit showed **11 existing tenant-owned tables without RLS**.
The new embeddings table's policy alone does **not** secure the rest. Before
public login, finish restricted-role/RLS migration for originals, NER, NED,
Topics, chunks, language, and all future tables; run real two-tenant tests
using non-bypassing runtime credentials. `--tenant` is a trusted local operator
flag, not authorization.

Never count vector hits as number of customer complaints. Aggregate issues,
sentiment, product changes, periods and verified counts from SQL after
NER/NED/opinion enrichment. Retrieval supplies cited evidence only.
