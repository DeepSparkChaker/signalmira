-- SignalMira Step 2: one current chunk set per review, derived from feedback_reviews.
-- Requires the existing feedback_reviews(tenant_id, source, review_id) PK.
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    review_id TEXT NOT NULL,
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    chunk_text TEXT NOT NULL,
    start_index INT NOT NULL CHECK (start_index >= 0),
    end_index INT NOT NULL,
    input_hash TEXT NOT NULL,
    review_content_hash TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    company TEXT,
    product_id TEXT,
    review_date DATE,
    language TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (end_index > start_index),
    CHECK (length(chunk_text) = end_index - start_index),
    UNIQUE (tenant_id, source, review_id, chunk_index),
    FOREIGN KEY (tenant_id, source, review_id)
        REFERENCES feedback_reviews (tenant_id, source, review_id)
        ON DELETE CASCADE
);

-- For incremental sync, lookups, and later retrieval filtering.
CREATE INDEX IF NOT EXISTS rag_chunks_filter_idx
    ON rag_chunks (tenant_id, product_id, review_date DESC);

-- RLS is defense-in-depth: use a non-superuser/no-BYPASSRLS API role in prod.
-- Each DB transaction must set: SELECT set_config('app.tenant_id', $1, true).
ALTER TABLE rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rag_chunks_tenant_policy ON rag_chunks;
CREATE POLICY rag_chunks_tenant_policy ON rag_chunks
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
