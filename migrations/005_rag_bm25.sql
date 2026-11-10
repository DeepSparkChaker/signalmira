-- SignalMira: real BM25 on existing current chunk text, no duplicated corpus.
-- Prerequisites: PostgreSQL 17+; pg_textsearch installed + preloaded; authorized
-- administrator explicitly accepts cross-tenant IDF leakage documented upstream.
-- Run once with a migration connection in autocommit mode (not a transaction).
CREATE INDEX CONCURRENTLY IF NOT EXISTS rag_chunks_bm25_simple_idx
    ON public.rag_chunks USING bm25 (chunk_text)
    WITH (text_config = 'simple', k1 = 1.2, b = 0.75);
