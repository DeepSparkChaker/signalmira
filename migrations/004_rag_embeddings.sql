-- Step 5, apply AFTER migrations/002_rag_chunks.sql; pgvector must be installed
-- in the PostgreSQL server. Run as schema owner/migration role, not API role.
CREATE EXTENSION IF NOT EXISTS vector;

-- A composite FK makes it impossible to attach a tenant-A embedding to a
-- tenant-B chunk, including on the SQL write path.
CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_tenant_chunk_uidx
    ON public.rag_chunks (tenant_id, chunk_id);

CREATE TABLE IF NOT EXISTS public.rag_embeddings (
    tenant_id TEXT NOT NULL,
    chunk_id UUID NOT NULL,
    model_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    backend TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    chunk_input_hash TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, chunk_id, model_key),
    FOREIGN KEY (tenant_id, chunk_id)
        REFERENCES public.rag_chunks (tenant_id, chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS rag_embeddings_tenant_model_idx
    ON public.rag_embeddings (tenant_id, model_key);

-- Fail closed without transaction-local tenant context. This is one barrier,
-- NOT a substitute for authenticated APIs and non-BYPASSRLS DB credentials.
ALTER TABLE public.rag_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rag_embeddings FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rag_embeddings_tenant_guard ON public.rag_embeddings;
CREATE POLICY rag_embeddings_tenant_guard ON public.rag_embeddings
    AS RESTRICTIVE FOR ALL TO PUBLIC
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
DROP POLICY IF EXISTS rag_embeddings_tenant_access ON public.rag_embeddings;
CREATE POLICY rag_embeddings_tenant_access ON public.rag_embeddings
    AS PERMISSIVE FOR ALL TO PUBLIC
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
REVOKE ALL ON public.rag_embeddings FROM PUBLIC;
-- Optional existing restricted roles from security.tenant_rls.apply.
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='sm_tenant_reader') THEN
    GRANT SELECT ON public.rag_embeddings TO sm_tenant_reader;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='sm_tenant_worker') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.rag_embeddings TO sm_tenant_worker;
  END IF;
END $$;
-- HNSW is NOT built by this migration: use `python -m retrieval.semantic hnsw`.
