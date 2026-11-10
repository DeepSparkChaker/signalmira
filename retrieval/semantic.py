"""Qwen3 -> pgvector incremental indexing and tenant-scoped semantic retrieval.

CLI (from SignalMira root):
  python -m retrieval.semantic init
  python -m indexing.chunk_store sync --tenant demo
  python -m retrieval.semantic sync --tenant demo
  python -m retrieval.semantic search --tenant demo --query 'battery life problems'
  python -m retrieval.semantic hnsw
  python -m retrieval.semantic benchmark --tenant demo --query 'battery life problems'

Local CLI --tenant is a trusted operator/job input, NOT authentication. In a
multi-user API, authorize tenant membership BEFORE calling these functions.
All tenant SQL runs inside security.tenant_context.tenant_transaction().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import numpy as np
import re
import time
from itertools import batched
from pathlib import Path
from uuid import UUID

from security.tenant_context import tenant_transaction, validate_tenant
from retrieval.embedders import (DIMENSIONS, Embedder, LocalQwenEmbedder,
                                 VllmQwenEmbedder, checked_vector)

MIGRATION = Path(__file__).resolve().parents[1] / "migrations/004_rag_embeddings.sql"
START_UUID = UUID(int=0)

PENDING = """
SELECT c.chunk_id, c.chunk_text, c.source, c.review_id, c.chunk_index,
       c.input_hash, c.chunker_version
FROM public.rag_chunks AS c
LEFT JOIN public.rag_embeddings AS e
  ON e.tenant_id=c.tenant_id AND e.chunk_id=c.chunk_id AND e.model_key=%s
WHERE c.tenant_id=%s AND c.chunk_id > %s::uuid
  AND (e.chunk_id IS NULL OR e.chunk_input_hash IS DISTINCT FROM c.input_hash
       OR e.chunker_version IS DISTINCT FROM c.chunker_version)
ORDER BY c.chunk_id
LIMIT %s
"""

STAGE_DDL = """
CREATE TEMP TABLE IF NOT EXISTS rag_embeddings_stage (
    chunk_id UUID NOT NULL,
    input_hash TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedding vector(1024) NOT NULL
) ON COMMIT DELETE ROWS
"""

COPY_STAGE = """
COPY pg_temp.rag_embeddings_stage
  (chunk_id, input_hash, chunker_version, embedding) FROM STDIN
"""

UPSERT = """
INSERT INTO public.rag_embeddings AS existing
  (tenant_id, chunk_id, model_key, model_name, model_revision, backend,
   config_hash, chunk_input_hash, chunker_version, embedding)
SELECT c.tenant_id, c.chunk_id, %s, %s, %s, %s, %s,
       p.input_hash, p.chunker_version, p.embedding
FROM pg_temp.rag_embeddings_stage AS p
JOIN public.rag_chunks AS c
  ON c.tenant_id=%s AND c.chunk_id=p.chunk_id
 AND c.input_hash=p.input_hash AND c.chunker_version=p.chunker_version
ON CONFLICT (tenant_id, chunk_id, model_key) DO UPDATE SET
    embedding=EXCLUDED.embedding,
    chunk_input_hash=EXCLUDED.chunk_input_hash,
    chunker_version=EXCLUDED.chunker_version,
    model_name=EXCLUDED.model_name,
    model_revision=EXCLUDED.model_revision,
    backend=EXCLUDED.backend,
    config_hash=EXCLUDED.config_hash,
    updated_at=now()
WHERE existing.chunk_input_hash IS DISTINCT FROM EXCLUDED.chunk_input_hash
   OR existing.chunker_version IS DISTINCT FROM EXCLUDED.chunker_version
RETURNING chunk_id
"""

# Find nearest chunk IDs first; hydrate full review metadata only for final k.
# The EXISTS guard also excludes vectors whose source chunk changed since indexing.
# Optional source/product filters MUST be inside this candidate selection, before LIMIT.
NEAREST_BASE = """
WITH nearest AS MATERIALIZED (
    SELECT e.chunk_id, (e.embedding <=> %s::vector(1024)) AS distance
    FROM public.rag_embeddings AS e
    WHERE e.tenant_id=%s AND e.model_key=%s
      AND EXISTS (
          SELECT 1 FROM public.rag_chunks AS eligible
          WHERE eligible.tenant_id=e.tenant_id AND eligible.chunk_id=e.chunk_id
            AND eligible.input_hash=e.chunk_input_hash
            AND eligible.chunker_version=e.chunker_version
"""

HYDRATE = """
)
SELECT c.chunk_id, c.source, c.review_id, c.chunk_index, c.chunk_text,
       c.start_index, c.end_index, c.company, c.product_id, c.review_date,
       COALESCE(l.language, c.language) AS language,
       1.0 - nearest.distance AS similarity
FROM nearest
JOIN public.rag_chunks AS c ON c.tenant_id=%s AND c.chunk_id=nearest.chunk_id
LEFT JOIN public.feedback_language AS l
  ON l.tenant_id=c.tenant_id AND l.source=c.source AND l.review_id=c.review_id
 AND l.input_hash=c.review_content_hash
ORDER BY nearest.distance, c.chunk_id
"""


def _search_sql(mode: str, *, source: str | None, product_id: str | None) -> str:
    """Build fixed SQL with value placeholders; never interpolate user input."""
    query = NEAREST_BASE
    if source is not None:
        query += " AND eligible.source=%s\n"
    if product_id is not None:
        query += " AND eligible.product_id=%s\n"
    # pgvector ANN indexes require ORDER BY the bare distance operator.
    # + 0 disables the ANN index for the *exact* baseline while keeping the
    # B-tree chunk/tenant indexes available (unlike enable_indexscan=off).
    distance = "e.embedding <=> %s::vector(1024)"
    if mode == "exact":
        distance = f"({distance}) + 0"
    query += f"      )\n    ORDER BY {distance}\n    LIMIT %s\n"
    return query + HYDRATE


def initialize(conn) -> None:
    """Schema migration: run with admin/owner connection, separately from worker."""
    if conn.autocommit is not True:
        raise RuntimeError("Initialize using an autocommit=True migration connection")
    if conn.execute("SELECT to_regclass('public.rag_chunks') AS table_name").fetchone()["table_name"] is None:
        raise RuntimeError("Missing rag_chunks; run `python -m indexing.chunk_store init` first")
    with conn.transaction():
        conn.execute(MIGRATION.read_text(encoding="utf-8"))


def create_hnsw(conn) -> None:
    """Optional ANN index, concurrent for less blocking; run after initial backfill."""
    if conn.autocommit is not True:
        raise RuntimeError("CREATE INDEX CONCURRENTLY requires autocommit=True")
    ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    if ext is None:
        raise RuntimeError("pgvector extension not installed; run `init` first")
    version = tuple(int(part) for part in ext["extversion"].split(".")[:2])
    if version < (0, 8):
        raise RuntimeError("Upgrade pgvector to >=0.8 for iterative HNSW scans")
    conn.execute("""CREATE INDEX CONCURRENTLY IF NOT EXISTS rag_embeddings_hnsw_cosine_idx
                    ON public.rag_embeddings USING hnsw
                    (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)""")


def sync(conn, tenant: str, embedder: Embedder, *, page_size: int = 64,
         batch_size: int = 8, limit: int = 0) -> dict:
    """Keyset pages; small GPU batches; hash-guarded atomic upserts.

    Embedding runs OUTSIDE DB transactions. Each DB write re-checks current
    chunk hash/version so edits/deletes while a model runs cannot be indexed.
    FK ON DELETE CASCADE cleans vectors when a review/chunk is deleted.
    """
    tenant = validate_tenant(tenant)
    if not (1 <= batch_size <= page_size <= 1000) or limit < 0:
        raise ValueError("Require 1 <= batch_size <= page_size <= 1000 and limit >= 0")
    spec = embedder.spec  # resolves checkpoint SHA without loading GPU model
    if spec.dimensions != DIMENSIONS:
        raise ValueError("Unsupported embedding dimension")
    # Register the real PostgreSQL vector type so COPY adapts float32 NumPy
    # arrays directly. Only the trusted temporary staging table receives COPY.
    from pgvector.psycopg import register_vector
    register_vector(conn)
    conn.execute(STAGE_DDL)
    after = START_UUID
    stats = {"scanned": 0, "written": 0, "stale_or_concurrent": 0,
             "model_key": spec.model_key, "model_revision": spec.revision}

    while True:
        amount = min(page_size, limit - stats["scanned"]) if limit else page_size
        if not amount:
            break
        with tenant_transaction(conn, tenant):
            page = conn.execute(PENDING, (spec.model_key, tenant, after, amount)).fetchall()
        if not page:
            break
        after = page[-1]["chunk_id"]
        stats["scanned"] += len(page)
        for group in batched(page, batch_size):
            vectors = np.asarray(
                embedder.encode_documents([r["chunk_text"] for r in group]),
                dtype=np.float32,
            )
            if vectors.shape != (len(group), DIMENSIONS) or not np.isfinite(vectors).all():
                raise ValueError("Embedder returned invalid vector shape or values")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
                raise ValueError("Embedder returned non-normalized vectors")
            # COPY and hash-guarded merge MUST be one tenant-scoped transaction.
            # ON COMMIT DELETE ROWS removes staging contents before conn reuse.
            with tenant_transaction(conn, tenant):
                with conn.cursor() as cur:
                    with cur.copy(COPY_STAGE) as cp:
                        for row, vector in zip(group, vectors, strict=True):
                            cp.write_row((row["chunk_id"], row["input_hash"],
                                          row["chunker_version"], vector))
                    inserted = cur.execute(UPSERT, (
                        spec.model_key, spec.name, spec.revision,
                        spec.backend, spec.model_key, tenant,
                    )).fetchall()
            stats["written"] += len(inserted)
            stats["stale_or_concurrent"] += len(group) - len(inserted)

    return stats


def search_vector(conn, tenant: str, query_vector, *, model_key: str,
                  k: int = 10, mode: str = "exact", source: str | None = None,
                  product_id: str | None = None) -> list[dict]:
    """Tenant-scoped search; exact baseline or opt-in HNSW.

    Tenant/model/source/product and source-hash constraints are applied
    BEFORE top-k selection; only the final k rows join language metadata.
    HNSW may return fewer than k with selective filters; compare to exact.
    """
    tenant = validate_tenant(tenant)
    if not 1 <= k <= 100 or mode not in ("exact", "hnsw"):
        raise ValueError("k=1..100; mode exact or hnsw")
    if not isinstance(model_key, str) or not re.fullmatch(r"[0-9a-f]{64}", model_key):
        raise ValueError("model_key must be the registered embedding contract key")
    vector = checked_vector(query_vector)   # validated, contiguous float32 ndarray
    query = _search_sql(mode, source=source, product_id=product_id)
    params = [vector, tenant, model_key]
    if source is not None:
        params.append(source)
    if product_id is not None:
        params.append(product_id)
    params.extend((vector, k, tenant))

    # Register the vector adapter on THIS connection. In production pools,
    # register once in the pool's configure callback instead.
    from pgvector.psycopg import register_vector
    register_vector(conn)
    with tenant_transaction(conn, tenant):
        if mode == "hnsw":
            present = conn.execute(
                "SELECT to_regclass('public.rag_embeddings_hnsw_cosine_idx') AS idx"
            ).fetchone()["idx"]
            if present is None:
                raise RuntimeError("HNSW index is missing; run `python -m retrieval.semantic hnsw`")
            conn.execute("SET LOCAL hnsw.ef_search=100")
            conn.execute("SET LOCAL hnsw.iterative_scan=strict_order")
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def search(conn, tenant: str, query: str, embedder: Embedder, **filters) -> list[dict]:
    return search_vector(conn, tenant, embedder.encode_query(query),
                         model_key=embedder.spec.model_key, **filters)


def benchmark(conn, tenant: str, query: str, embedder: Embedder, *, k: int = 10,
              repeats: int = 5, source: str | None = None,
              product_id: str | None = None) -> dict:
    """Compare ANN agreement with one exact reference (not semantic relevance)."""
    if not 1 <= repeats <= 30:
        raise ValueError("repeats must be between 1 and 30")
    vector = embedder.encode_query(query)
    kwargs = dict(model_key=embedder.spec.model_key, k=k,
                  source=source, product_id=product_id)
    start = time.perf_counter()
    exact = search_vector(conn, tenant, vector, mode="exact", **kwargs)
    exact_ms = (time.perf_counter() - start) * 1000
    durations = []
    last = []
    for _ in range(repeats):
        start = time.perf_counter()
        last = search_vector(conn, tenant, vector, mode="hnsw", **kwargs)
        durations.append((time.perf_counter() - start) * 1000)
    expected = {str(r["chunk_id"]) for r in exact}
    actual = {str(r["chunk_id"]) for r in last}
    return {
        "exact": {"count": len(exact), "latency_ms": round(exact_ms, 2),
                  "chunk_ids": [str(r["chunk_id"]) for r in exact]},
        "hnsw": {"count": len(last),
                 "median_ms": round(statistics.median(durations), 2),
                 "p95_ms": round(sorted(durations)[math.ceil(.95 * repeats) - 1], 2),
                 "chunk_ids": [str(r["chunk_id"]) for r in last],
                 "agreement_with_exact_at_k": (
                     round(len(expected & actual) / len(expected), 4)
                     if expected else None)},
    }


def _embedder(args) -> Embedder:
    if args.backend == "vllm":
        if not args.revision:
            raise ValueError("--revision (pinned 40-hex model checkpoint) required for vLLM")
        return VllmQwenEmbedder(revision=args.revision, base_url=args.vllm_url)
    return LocalQwenEmbedder(revision=args.revision, device=args.device,
                             offline=args.offline,
                             batch_size=getattr(args, "batch_size", 8),
                             attention=args.attention)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create extension + embeddings table (admin only)")
    sub.add_parser("hnsw", help="Build optional HNSW index concurrently (admin only)")
    for command in ("sync", "search", "benchmark"):
        p = sub.add_parser(command)
        p.add_argument("--tenant", required=True, help="Authorized operator/job tenant")
        p.add_argument("--backend", choices=("sentence-transformers", "vllm"),
                       default="sentence-transformers")
        p.add_argument("--revision", help="Pinned 40-hex model SHA; required for vLLM")
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--offline", action="store_true", help="Use already cached model files")
        p.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
        p.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
        if command == "sync":
            p.add_argument("--page-size", type=int, default=64)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--limit", type=int, default=0)
        else:
            p.add_argument("--query", required=True)
            p.add_argument("--k", type=int, default=10)
            p.add_argument("--source")
            p.add_argument("--product-id")
            if command == "search":
                p.add_argument("--mode", choices=("exact", "hnsw"), default="exact")
            else:
                p.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    from storage.postgres import connect
    with connect(os.environ["DATABASE_URL"]) as conn:
        if args.command == "init":
            initialize(conn)
            print("rag_embeddings schema ready (no HNSW index yet)")
        elif args.command == "hnsw":
            create_hnsw(conn)
            print("HNSW index ready")
        else:
            embedder = _embedder(args)
            if args.command == "sync":
                data = sync(conn, args.tenant, embedder,
                            page_size=args.page_size, batch_size=args.batch_size,
                            limit=args.limit)
            elif args.command == "search":
                data = search(conn, args.tenant, args.query, embedder,
                              k=args.k, source=args.source,
                              product_id=args.product_id, mode=args.mode)
            else:
                data = benchmark(conn, args.tenant, args.query, embedder, k=args.k,
                                 repeats=args.repeats, source=args.source,
                                 product_id=args.product_id)
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
