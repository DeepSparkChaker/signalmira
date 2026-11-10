"""SignalMira tenant-scoped lexical retrieval: pg_textsearch BM25 or native FTS.

Requires an authorized tenant (CLI tenant is a trusted operator input, not auth).
Uses the existing `rag_chunks` table; indexes update with its inserts/updates/deletes.

Security: shared pg_textsearch BM25 corpus statistics include RLS-hidden rows;
using backend='bm25' therefore REQUIRES an explicit administrator-approved opt-in.
The native 'fts' fallback does not use cross-tenant BM25 corpus statistics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from security.tenant_context import tenant_transaction, validate_tenant

ROOT = Path(__file__).resolve().parents[1]
BM25_INDEX = 'rag_chunks_bm25_simple_idx'  # deployment-controlled, never user input
BM25_MIGRATION = ROOT / 'migrations/005_rag_bm25.sql'
FTS_MIGRATION = ROOT / 'migrations/005_rag_fts_fallback.sql'


def check_backend(backend: str, allow_shared_idf: bool = False) -> None:
    if backend not in ('bm25', 'fts'):
        raise ValueError('backend must be bm25 or fts')
    if backend == 'bm25' and not allow_shared_idf:
        raise PermissionError(
            'BM25 on a multi-tenant shared index can leak aggregate word frequencies '
            'across RLS tenants; require explicit --allow-shared-idf approval, or use fts'
        )


def initialize(conn, *, backend: str, allow_shared_idf: bool = False) -> None:
    """Administrator-only index creation; run on autocommit migration connection."""
    check_backend(backend, allow_shared_idf)
    if conn.autocommit is not True:
        raise RuntimeError('Index creation requires autocommit=True')
    row = conn.execute("SELECT to_regclass('public.rag_chunks') AS rel").fetchone()
    if row['rel'] is None:
        raise RuntimeError('Missing rag_chunks: initialize chunk store first')
    if backend == 'bm25':
        pg = int(conn.execute('SHOW server_version_num').fetchone()['server_version_num'])
        if pg < 170000:
            raise RuntimeError('pg_textsearch needs PostgreSQL 17 or newer')
        conn.execute('CREATE EXTENSION IF NOT EXISTS pg_textsearch')
        conn.execute(BM25_MIGRATION.read_text(encoding='utf-8'))
    else:
        conn.execute(FTS_MIGRATION.read_text(encoding='utf-8'))


def _check_query(query: str, k: int) -> str:
    if not isinstance(query, str) or not query.strip() or '\x00' in query or len(query) > 2048:
        raise ValueError('query must be nonempty, without NUL, <=2048 chars')
    if not 1 <= k <= 100:
        raise ValueError('k must be 1..100')
    return query.strip()


def _where(source: str | None, product_id: str | None, alias: str = 'c') -> str:
    # Only static SQL fragments. Filter VALUES always use placeholders.
    return ((' AND ' + alias + '.source=%s') if source is not None else '') + (
        (' AND ' + alias + '.product_id=%s') if product_id is not None else ''
    )


def _sql(backend: str, *, source: str | None, product_id: str | None) -> str:
    filters = _where(source, product_id)
    if backend == 'bm25':
        return f"""
            SELECT c.chunk_id, c.source, c.review_id, c.chunk_index, c.chunk_text,
                   c.start_index, c.end_index, c.company, c.product_id, c.review_date,
                   -(c.chunk_text <@> to_bm25query(%s, '{BM25_INDEX}')) AS lexical_score
            FROM public.rag_chunks AS c
            WHERE c.tenant_id=%s {filters}
            ORDER BY c.chunk_text <@> to_bm25query(%s, '{BM25_INDEX}')
            LIMIT %s
        """
    return f"""
        SELECT c.chunk_id, c.source, c.review_id, c.chunk_index, c.chunk_text,
               c.start_index, c.end_index, c.company, c.product_id, c.review_date,
               ts_rank_cd(to_tsvector('simple'::regconfig, c.chunk_text),
                          websearch_to_tsquery('simple'::regconfig, %s)) AS lexical_score
        FROM public.rag_chunks AS c
        WHERE c.tenant_id=%s {filters}
          AND to_tsvector('simple'::regconfig, c.chunk_text)
              @@ websearch_to_tsquery('simple'::regconfig, %s)
        ORDER BY lexical_score DESC, c.chunk_id
        LIMIT %s
    """


def search(conn, tenant: str, query: str, *, k: int = 40,
           backend: str = 'fts', allow_shared_idf: bool = False,
           source: str | None = None, product_id: str | None = None) -> list[dict]:
    """Return current chunks, filtered by tenant/source/product BEFORE LIMIT.

    pg_textsearch uses negative BM25 for ASC order; exposed lexical_score is
    positive relevance. FTS uses positive ts_rank_cd. Never compare the raw
    numeric scores across backends or combine them with cosine scores.
    """
    tenant = validate_tenant(tenant)
    query = _check_query(query, k)
    check_backend(backend, allow_shared_idf)
    sql = _sql(backend, source=source, product_id=product_id)
    params: list[object] = [query, tenant]
    if source is not None:
        params.append(source)
    if product_id is not None:
        params.append(product_id)
    params.extend([query, k])
    with tenant_transaction(conn, tenant):
        return [dict(row) for row in conn.execute(sql, params).fetchall()]



def explain(conn, tenant: str, query: str, *, k: int = 40,
            backend: str = 'fts', allow_shared_idf: bool = False,
            source: str | None = None, product_id: str | None = None) -> str:
    """DB integration check. Runs actual search under tenant RLS; operator-only."""
    tenant = validate_tenant(tenant)
    query = _check_query(query, k)
    check_backend(backend, allow_shared_idf)
    params: list[object] = [query, tenant]
    if source is not None:
        params.append(source)
    if product_id is not None:
        params.append(product_id)
    params.extend([query, k])
    with tenant_transaction(conn, tenant):
        rows = conn.execute(
            'EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) ' +
            _sql(backend, source=source, product_id=product_id), params
        ).fetchall()
        return '\n'.join(str(next(iter(row.values()))) for row in rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('init', help='Admin-only create index concurrently')
    i.add_argument('--backend', choices=('bm25', 'fts'), default='fts')
    i.add_argument('--allow-shared-idf', action='store_true')
    s = sub.add_parser('search', help='Search chunks by tenant')
    s.add_argument('--tenant', required=True)
    s.add_argument('--query', required=True)
    s.add_argument('--k', type=int, default=40)
    s.add_argument('--backend', choices=('bm25', 'fts'), default='fts')
    s.add_argument('--allow-shared-idf', action='store_true')
    s.add_argument('--source')
    s.add_argument('--product-id')
    s.add_argument('--explain', action='store_true', help='Show actual query plan')
    args = p.parse_args()
    from storage.postgres import connect
    with connect(os.environ['DATABASE_URL']) as conn:
        if args.command == 'init':
            initialize(conn, backend=args.backend,
                       allow_shared_idf=args.allow_shared_idf)
            print(f'{args.backend} index ready')
        elif args.explain:
            print(explain(conn, args.tenant, args.query, k=args.k,
                          backend=args.backend, allow_shared_idf=args.allow_shared_idf,
                          source=args.source, product_id=args.product_id))
        else:
            results = search(conn, args.tenant, args.query, k=args.k,
                             backend=args.backend,
                             allow_shared_idf=args.allow_shared_idf,
                             source=args.source, product_id=args.product_id)
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
