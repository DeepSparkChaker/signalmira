-- Run after 020_sentiment_opinions.sql. Additive, no existing tables modified.
-- Scope: saved aspect membership and immutable membership history only.
CREATE TABLE IF NOT EXISTS public.sm_aspect_groups (
    tenant_id       text NOT NULL,
    group_id        uuid NOT NULL DEFAULT gen_random_uuid(),
    extractor_key   text NOT NULL CHECK (extractor_key ~ '^[0-9a-f]{64}$'),
    name            text NOT NULL CHECK (length(btrim(name)) BETWEEN 1 AND 120),
    aspects         text[] NOT NULL CHECK (cardinality(aspects) BETWEEN 1 AND 50),
    version         integer NOT NULL DEFAULT 1 CHECK (version >= 1),
    created_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, group_id),
    UNIQUE (tenant_id, extractor_key, name)
);
CREATE INDEX IF NOT EXISTS sm_aspect_groups_listing_idx
    ON public.sm_aspect_groups (tenant_id, extractor_key, name);

CREATE TABLE IF NOT EXISTS public.sm_aspect_group_versions (
    tenant_id       text NOT NULL,
    group_id        uuid NOT NULL,
    version         integer NOT NULL CHECK (version >= 1),
    name            text NOT NULL,
    aspects         text[] NOT NULL CHECK (cardinality(aspects) BETWEEN 1 AND 50),
    created_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, group_id, version),
    FOREIGN KEY (tenant_id, group_id)
        REFERENCES public.sm_aspect_groups (tenant_id, group_id) ON DELETE RESTRICT
);

ALTER TABLE public.sm_aspect_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sm_aspect_groups FORCE ROW LEVEL SECURITY;
ALTER TABLE public.sm_aspect_group_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sm_aspect_group_versions FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public'
                 AND tablename='sm_aspect_groups' AND policyname='sm_aspect_groups_tenant') THEN
    CREATE POLICY sm_aspect_groups_tenant ON public.sm_aspect_groups
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public'
                 AND tablename='sm_aspect_group_versions'
                 AND policyname='sm_aspect_group_versions_tenant') THEN
    CREATE POLICY sm_aspect_group_versions_tenant ON public.sm_aspect_group_versions
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
END $$;
