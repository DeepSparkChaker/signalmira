-- SignalMira language metadata, separate from immutable original feedback.
-- Applied by: python -m enrichment.language init
-- Tenant-specific table. In production, align privileges/policies with
-- security.tenant_rls and test using an UNPRIVILEGED application role.
CREATE TABLE IF NOT EXISTS public.feedback_language (
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    review_id TEXT NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('detected','unknown','mixed')),
    confidence DOUBLE PRECISION,
    margin DOUBLE PRECISION,
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    detector TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, source, review_id),
    CONSTRAINT feedback_language_review_fk
        FOREIGN KEY (tenant_id, source, review_id)
        REFERENCES public.feedback_reviews(tenant_id, source, review_id)
        ON DELETE CASCADE,
    CONSTRAINT feedback_language_lang_check
        CHECK (language ~ '^[a-z]{2,3}$' OR language IN ('und','mul')),
    CONSTRAINT feedback_language_confidence_check
        CHECK (confidence IS NULL OR (confidence BETWEEN 0 AND 1)),
    CONSTRAINT feedback_language_margin_check
        CHECK (margin IS NULL OR (margin BETWEEN 0 AND 1))
);
CREATE INDEX IF NOT EXISTS feedback_language_tenant_lang_idx
    ON public.feedback_language (tenant_id, language);

-- Apply policies at table creation. Never grant this table to an untrusted SQL client.
-- The release-wide security.tenant_rls migration replaces the permissive policy
-- below with one explicitly assigned to restricted runtime roles.
ALTER TABLE public.feedback_language ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_language FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sm_tenant_guard ON public.feedback_language;
CREATE POLICY sm_tenant_guard ON public.feedback_language
    AS RESTRICTIVE FOR ALL TO PUBLIC
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
DROP POLICY IF EXISTS sm_tenant_access ON public.feedback_language;
CREATE POLICY sm_tenant_access ON public.feedback_language
    AS PERMISSIVE FOR ALL TO PUBLIC
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
REVOKE ALL ON TABLE public.feedback_language FROM PUBLIC;
