-- RLS-compatible fallback for installations where BM25 is disallowed/unavailable.
-- Native PostgreSQL GIN full-text search; ranking is ts_rank_cd, NOT BM25.
-- Run with autocommit=True due to CREATE INDEX CONCURRENTLY.
CREATE INDEX CONCURRENTLY IF NOT EXISTS rag_chunks_fts_simple_idx
    ON public.rag_chunks USING gin (to_tsvector('simple'::regconfig, chunk_text));
