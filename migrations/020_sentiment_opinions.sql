-- SignalMira v1: additive, versioned sentiment + opinion enrichment.
-- Run only after confirming this migration number is unused in your repository.
-- Stores ALL model/input versions. Current facts always join source content_hash.

CREATE TABLE IF NOT EXISTS public.feedback_sentiment (
    tenant_id       text NOT NULL,
    source          text NOT NULL,
    review_id       text NOT NULL,
    input_hash      text NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    config_hash     text NOT NULL CHECK (config_hash ~ '^[0-9a-f]{64}$'),
    model_name      text NOT NULL,
    model_revision  text NOT NULL,
    sentiment       text NOT NULL CHECK (sentiment IN ('negative','neutral','positive')),
    negative_score  double precision NOT NULL CHECK (negative_score BETWEEN 0 AND 1),
    neutral_score   double precision NOT NULL CHECK (neutral_score BETWEEN 0 AND 1),
    positive_score  double precision NOT NULL CHECK (positive_score BETWEEN 0 AND 1),
    is_truncated    boolean NOT NULL DEFAULT false,
    processed_at    timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, source, review_id, input_hash, config_hash)
);
CREATE INDEX IF NOT EXISTS sm_sentiment_version_idx
    ON public.feedback_sentiment (tenant_id, config_hash, source, review_id);

CREATE TABLE IF NOT EXISTS public.feedback_opinion_runs (
    tenant_id       text NOT NULL,
    source          text NOT NULL,
    review_id       text NOT NULL,
    input_hash      text NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    extractor_key   text NOT NULL CHECK (extractor_key ~ '^[0-9a-f]{64}$'),
    model_name      text NOT NULL,
    model_revision  text NOT NULL,
    schema_version  text NOT NULL,
    llm_overall     text NOT NULL CHECK (llm_overall IN ('positive','negative','neutral','mixed')),
    processed_at    timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, source, review_id, input_hash, extractor_key)
);
CREATE INDEX IF NOT EXISTS sm_opinion_runs_version_idx
    ON public.feedback_opinion_runs (tenant_id, extractor_key, source, review_id);

CREATE TABLE IF NOT EXISTS public.feedback_opinions (
    tenant_id         text NOT NULL,
    source            text NOT NULL,
    review_id         text NOT NULL,
    input_hash        text NOT NULL,
    extractor_key     text NOT NULL,
    opinion_index     smallint NOT NULL CHECK (opinion_index BETWEEN 0 AND 15),
    aspect            text NOT NULL,
    target            text,
    sentiment         text NOT NULL CHECK (sentiment IN ('positive','negative','neutral','mixed')),
    evidence_quote    text NOT NULL,
    start_index       integer NOT NULL CHECK (start_index >= 0),
    end_index         integer NOT NULL CHECK (end_index > start_index),
    reported_problem  text,
    suggestion_quote  text,
    suggestion_start  integer,
    suggestion_end    integer,
    PRIMARY KEY (tenant_id, source, review_id, input_hash, extractor_key, opinion_index),
    FOREIGN KEY (tenant_id, source, review_id, input_hash, extractor_key)
      REFERENCES public.feedback_opinion_runs
        (tenant_id, source, review_id, input_hash, extractor_key)
      ON DELETE CASCADE,
    CHECK ((suggestion_quote IS NULL AND suggestion_start IS NULL AND suggestion_end IS NULL)
       OR (suggestion_quote IS NOT NULL AND suggestion_start >= 0
           AND suggestion_end > suggestion_start))
);
CREATE INDEX IF NOT EXISTS sm_opinions_aspect_idx
    ON public.feedback_opinions (tenant_id, extractor_key, aspect, sentiment);

-- Retry bookkeeping: bounded exponential-ish backoff and operator-visible dead letters.
CREATE TABLE IF NOT EXISTS public.feedback_opinion_failures (
    tenant_id       text NOT NULL,
    source          text NOT NULL,
    review_id       text NOT NULL,
    input_hash      text NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    extractor_key   text NOT NULL CHECK (extractor_key ~ '^[0-9a-f]{64}$'),
    attempts        integer NOT NULL DEFAULT 1 CHECK (attempts BETWEEN 1 AND 5),
    error_type      text NOT NULL,
    last_failed_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
    next_retry_at   timestamptz NOT NULL DEFAULT (clock_timestamp() + interval '5 minutes'),
    PRIMARY KEY (tenant_id, source, review_id, input_hash, extractor_key)
);
CREATE INDEX IF NOT EXISTS sm_opinion_failures_due_idx
    ON public.feedback_opinion_failures (tenant_id, extractor_key, next_retry_at);

-- No tenant_id may be inferred from frontend input; authentication/authorization
-- must be performed upstream, then security.tenant_context sets app.tenant_id.
ALTER TABLE public.feedback_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_sentiment FORCE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinion_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinion_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinions FORCE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinion_failures ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_opinion_failures FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies
                 WHERE schemaname='public' AND tablename='feedback_sentiment'
                   AND policyname='sm_sentiment_tenant') THEN
    CREATE POLICY sm_sentiment_tenant ON public.feedback_sentiment
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies
                 WHERE schemaname='public' AND tablename='feedback_opinion_runs'
                   AND policyname='sm_opinion_runs_tenant') THEN
    CREATE POLICY sm_opinion_runs_tenant ON public.feedback_opinion_runs
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies
                 WHERE schemaname='public' AND tablename='feedback_opinions'
                   AND policyname='sm_opinions_tenant') THEN
    CREATE POLICY sm_opinions_tenant ON public.feedback_opinions
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies
                 WHERE schemaname='public' AND tablename='feedback_opinion_failures'
                   AND policyname='sm_opinion_failures_tenant') THEN
    CREATE POLICY sm_opinion_failures_tenant ON public.feedback_opinion_failures
      USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''))
      WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), ''));
  END IF;
END $$;
