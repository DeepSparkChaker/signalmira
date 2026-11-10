"""One PostgreSQL statement: tenant-filtered BM25/FTS and pgvector, SQL RRF.

Query embedding is computed by the caller BEFORE the DB statement. This is one
network round trip, but PostgreSQL is not guaranteed to execute both CTE legs
concurrently. Only ranking IDs travel through RRF; hydrate 25 final chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from security.tenant_context import tenant_transaction, validate_tenant
from retrieval.embedders import (DIMENSIONS, LocalQwenEmbedder,
                                 checked_vector)
from retrieval.lexical import _check_query, _where, check_backend, BM25_INDEX

MODEL_KEY_RE = re.compile(r'[0-9a-f]{64}\Z')


def build_sql(*, backend: str, mode: str, source: str | None = None,
              product_id: str | None = None) -> str:
    """SQL contains no interpolated user input; only allowlisted fragments."""
    check_backend(backend, allow_shared_idf=True)  # actual permission in search_vector
    if mode not in ('exact', 'hnsw'):
        raise ValueError('mode must be exact or hnsw')
    lex_filter = _where(source, product_id, alias='c')
    dense_filter = _where(source, product_id, alias='eligible')
    if backend == 'bm25':
        lexical = f"""
        SELECT c.chunk_id,
               c.chunk_text <@> to_bm25query(%s, '{BM25_INDEX}') AS score
        FROM public.rag_chunks AS c
        WHERE c.tenant_id=%s {lex_filter}
        ORDER BY c.chunk_text <@> to_bm25query(%s, '{BM25_INDEX}')
        LIMIT %s
        """
        lex_order = 'score ASC, chunk_id'
        lex_valid = 'WHERE score < 0'  # BM25 non-matching scores are zero
    else:
        lexical = f"""
        SELECT c.chunk_id,
               ts_rank_cd(to_tsvector('simple'::regconfig, c.chunk_text),
                          websearch_to_tsquery('simple'::regconfig, %s)) AS score
        FROM public.rag_chunks AS c
        WHERE c.tenant_id=%s {lex_filter}
          AND to_tsvector('simple'::regconfig, c.chunk_text)
              @@ websearch_to_tsquery('simple'::regconfig, %s)
        ORDER BY score DESC, c.chunk_id
        LIMIT %s
        """
        lex_order = 'score DESC, chunk_id'
        lex_valid = ''
    # For exact baseline, only suppress ANN ordering, not B-tree indexes.
    dense_order = 'e.embedding <=> %s::vector(1024)'
    if mode == 'exact':
        dense_order = f'({dense_order}) + 0'
    return f"""
    WITH lexical AS MATERIALIZED (
        {lexical}
    ), lr AS (
        SELECT chunk_id, row_number() OVER (ORDER BY {lex_order}) AS sparse_rank
        FROM lexical
        {lex_valid}
    ), dense AS MATERIALIZED (
        SELECT e.chunk_id,
               e.embedding <=> %s::vector(1024) AS distance
        FROM public.rag_embeddings AS e
        WHERE e.tenant_id=%s AND e.model_key=%s
          AND EXISTS (
              SELECT 1 FROM public.rag_chunks AS eligible
              WHERE eligible.tenant_id=e.tenant_id AND eligible.chunk_id=e.chunk_id
                AND eligible.input_hash=e.chunk_input_hash
                AND eligible.chunker_version=e.chunker_version
                {dense_filter}
          )
        ORDER BY {dense_order}
        LIMIT %s
    ), dr AS (
        SELECT chunk_id, row_number() OVER (ORDER BY distance, chunk_id) AS dense_rank
        FROM dense
    ), merged AS (
        SELECT COALESCE(lr.chunk_id, dr.chunk_id) AS chunk_id,
               lr.sparse_rank, dr.dense_rank,
               COALESCE(1.0 / (%s + lr.sparse_rank), 0.0)
               + COALESCE(1.0 / (%s + dr.dense_rank), 0.0) AS rrf_score
        FROM lr FULL OUTER JOIN dr USING (chunk_id)
    ), top AS MATERIALIZED (
        SELECT * FROM merged
        ORDER BY rrf_score DESC, chunk_id
        LIMIT %s
    )
    SELECT c.chunk_id, c.source, c.review_id, c.chunk_index, c.chunk_text,
           c.start_index, c.end_index, c.company, c.product_id, c.review_date,
           COALESCE(l.language, c.language) AS language,
           top.sparse_rank, top.dense_rank, top.rrf_score
    FROM top
    JOIN public.rag_chunks AS c ON c.tenant_id=%s AND c.chunk_id=top.chunk_id
    LEFT JOIN public.feedback_language AS l
      ON l.tenant_id=c.tenant_id AND l.source=c.source AND l.review_id=c.review_id
     AND l.input_hash=c.review_content_hash
    ORDER BY top.rrf_score DESC, c.chunk_id
    """


def build_params(tenant: str, query: str, vector, model_key: str, *,
                 backend: str, mode: str = 'hnsw', source: str | None = None,
                 product_id: str | None = None, sparse_k: int = 40,
                 dense_k: int = 40, fused_k: int = 25, rrf_k: int = 60) -> list:
    tenant = validate_tenant(tenant)
    query = _check_query(query, sparse_k)
    if not all(isinstance(x, int) and not isinstance(x, bool)
               for x in (dense_k, fused_k, rrf_k)):
        raise ValueError('K values must be integers')
    if not (1 <= dense_k <= 100 and 1 <= fused_k <= 100 and 1 <= rrf_k <= 1000):
        raise ValueError('dense_k/fused_k must be 1..100 and rrf_k 1..1000')
    if not isinstance(model_key, str) or not MODEL_KEY_RE.fullmatch(model_key):
        raise ValueError('model_key must be SHA256 of the document embedding contract')
    vector = checked_vector(vector)
    p = [query, tenant]
    if source is not None:
        p.append(source)
    if product_id is not None:
        p.append(product_id)
    p += [query, sparse_k, vector, tenant, model_key]
    if source is not None:
        p.append(source)
    if product_id is not None:
        p.append(product_id)
    p += [vector, dense_k, rrf_k, rrf_k, fused_k, tenant]
    return p


def search_vector(conn, tenant: str, query: str, query_vector, *, model_key: str,
                  lexical_backend: str = 'fts', allow_shared_idf: bool = False,
                  mode: str = 'hnsw', sparse_k: int = 40, dense_k: int = 40,
                  fused_k: int = 25, rrf_k: int = 60,
                  source: str | None = None,
                  product_id: str | None = None) -> list[dict]:
    """Single SQL statement: fetch max fused_k hydrated chunks under RLS."""
    check_backend(lexical_backend, allow_shared_idf)
    sql = build_sql(backend=lexical_backend, mode=mode,
                    source=source, product_id=product_id)
    p = build_params(tenant, query, query_vector, model_key,
                     backend=lexical_backend, mode=mode,
                     source=source, product_id=product_id,
                     sparse_k=sparse_k, dense_k=dense_k,
                     fused_k=fused_k, rrf_k=rrf_k)
    from pgvector.psycopg import register_vector
    register_vector(conn)  # for a pool: register in configure callback once
    with tenant_transaction(conn, tenant):
        if mode == 'hnsw':
            row = conn.execute(
                "SELECT to_regclass('public.rag_embeddings_hnsw_cosine_idx') AS idx"
            ).fetchone()
            if row['idx'] is None:
                raise RuntimeError('Missing HNSW index; run python -m retrieval.semantic hnsw')
            conn.execute('SET LOCAL hnsw.ef_search=100')
            conn.execute('SET LOCAL hnsw.iterative_scan=strict_order')
        conn.execute("SET LOCAL statement_timeout='5000ms'")  # tune from p95 load tests
        conn.execute("SET LOCAL work_mem='16MB'")
        return [dict(row) for row in conn.execute(sql, p).fetchall()]


def search(conn, tenant: str, query: str, embedder, **kw) -> list[dict]:
    """Synchronous entry point. Reuse one warm embedder per worker."""
    return search_vector(conn, tenant, query, embedder.encode_query(query),
                         model_key=embedder.spec.model_key, **kw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tenant', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--lexical-backend', choices=('bm25', 'fts'), default='fts')
    p.add_argument('--allow-shared-idf', action='store_true')
    p.add_argument('--mode', choices=('hnsw', 'exact'), default='exact')
    p.add_argument('--source')
    p.add_argument('--product-id')
    p.add_argument('--revision', help='40-char pinned Qwen checkpoint SHA')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    embedder = LocalQwenEmbedder(revision=args.revision, device=args.device,
                                 offline=args.offline)
    from storage.postgres import connect
    with connect(os.environ['DATABASE_URL']) as conn:
        hits = search(conn, args.tenant, args.query, embedder,
                      lexical_backend=args.lexical_backend,
                      allow_shared_idf=args.allow_shared_idf,
                      mode=args.mode, source=args.source,
                      product_id=args.product_id)
        print(json.dumps(hits, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
