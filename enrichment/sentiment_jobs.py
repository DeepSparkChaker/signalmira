"""Incremental tenant-scoped background workers for sentiment + opinions.

Dedicated CLI connection; short database transactions; no GPU work in SQL tx.
Every output is tied to original content_hash and immutable model config hash.
Usage: python -m enrichment.sentiment_jobs sentiment --tenant demo --revision SHA
       python -m enrichment.sentiment_jobs opinions --tenant demo --llm-model MODEL --llm-revision DIGEST
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from security.tenant_context import tenant_transaction, validate_tenant
from enrichment.sentiment import LocalSentimentClassifier, SentimentConfig, MODEL_ID
from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import (
    LangChainOpinionExtractor as OpinionExtractor
)
from enrichment.opinion_schema import SCHEMA_VERSION

LOG = logging.getLogger(__name__)

# The source data is the source of truth. No work is performed twice for the
# same (tenant, source, review, content_hash, model/prompt version).
PENDING_SENTIMENT = """
SELECT f.source, f.review_id, f.text, f.content_hash
FROM public.feedback_reviews f
WHERE f.tenant_id = %s
  AND (f.source, f.review_id) > (%s, %s)
  AND NOT EXISTS (
      SELECT 1 FROM public.feedback_sentiment s
      WHERE s.tenant_id=f.tenant_id AND s.source=f.source
        AND s.review_id=f.review_id AND s.input_hash=f.content_hash
        AND s.config_hash=%s
  )
ORDER BY f.source, f.review_id
LIMIT %s
"""
PENDING_OPINIONS = """
SELECT f.source, f.review_id, f.text, f.content_hash
FROM public.feedback_reviews f
WHERE f.tenant_id = %s
  AND (f.source, f.review_id) > (%s, %s)
  AND NOT EXISTS (
      SELECT 1 FROM public.feedback_opinion_runs r
      WHERE r.tenant_id=f.tenant_id AND r.source=f.source
        AND r.review_id=f.review_id AND r.input_hash=f.content_hash
        AND r.extractor_key=%s
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.feedback_opinion_failures e
      WHERE e.tenant_id=f.tenant_id AND e.source=f.source
        AND e.review_id=f.review_id AND e.input_hash=f.content_hash
        AND e.extractor_key=%s
        AND (e.attempts >= 5 OR e.next_retry_at > clock_timestamp())
  )
ORDER BY f.source, f.review_id
LIMIT %s
"""

# Guard against stale enrichment if content changed while model was running.
SAVE_SENTIMENT = """
INSERT INTO public.feedback_sentiment (
 tenant_id,source,review_id,input_hash,config_hash,model_name,model_revision,
 sentiment,negative_score,neutral_score,positive_score,is_truncated)
SELECT f.tenant_id, f.source, f.review_id, f.content_hash,
       %s,%s,%s,%s,%s,%s,%s,%s
FROM public.feedback_reviews f
WHERE f.tenant_id=%s AND f.source=%s AND f.review_id=%s AND f.content_hash=%s
ON CONFLICT DO NOTHING
RETURNING review_id
"""
SAVE_RUN = """
INSERT INTO public.feedback_opinion_runs (
 tenant_id,source,review_id,input_hash,extractor_key,model_name,model_revision,
 schema_version,llm_overall)
SELECT f.tenant_id,f.source,f.review_id,f.content_hash,%s,%s,%s,%s,%s
FROM public.feedback_reviews f
WHERE f.tenant_id=%s AND f.source=%s AND f.review_id=%s AND f.content_hash=%s
ON CONFLICT DO NOTHING
RETURNING review_id
"""
SAVE_FAILURE = """
INSERT INTO public.feedback_opinion_failures (
 tenant_id,source,review_id,input_hash,extractor_key,attempts,error_type)
SELECT f.tenant_id,f.source,f.review_id,f.content_hash,%s,1,%s
FROM public.feedback_reviews f
WHERE f.tenant_id=%s AND f.source=%s AND f.review_id=%s AND f.content_hash=%s
ON CONFLICT (tenant_id,source,review_id,input_hash,extractor_key)
DO UPDATE SET
 attempts=LEAST(5,feedback_opinion_failures.attempts+1),
 last_failed_at=clock_timestamp(),
 next_retry_at=clock_timestamp() + interval '5 minutes'
              * LEAST(5,feedback_opinion_failures.attempts+1),
 error_type=EXCLUDED.error_type
"""
DELETE_FAILURE = """
DELETE FROM public.feedback_opinion_failures
WHERE tenant_id=%s AND source=%s AND review_id=%s AND input_hash=%s
  AND extractor_key=%s
"""
SAVE_OPINION = """
INSERT INTO public.feedback_opinions (
 tenant_id,source,review_id,input_hash,extractor_key,opinion_index,aspect,target,
 sentiment,evidence_quote,start_index,end_index,reported_problem,
 suggestion_quote,suggestion_start,suggestion_end)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""


def _acquire(conn, tenant: str, job: str) -> None:
    result = conn.execute(
        'SELECT pg_try_advisory_lock(hashtext(%s),hashtext(%s)) AS ok',
        (job, tenant)).fetchone()['ok']
    # Session-level lock survives commit; end implicit transaction before worker runs.
    conn.commit()
    if not result:
        raise RuntimeError(f'Another {job} worker is processing tenant {tenant}')


def _release(conn, tenant: str, job: str) -> None:
    conn.rollback()  # Ensure failed statements do not prevent unlocking.
    conn.execute('SELECT pg_advisory_unlock(hashtext(%s),hashtext(%s))',
                 (job, tenant))
    conn.commit()


def _pending(conn, tenant: str, sql: str, last: tuple[str, str],
             version: str, page_size: int) -> list[dict]:
    versions = (version, version) if sql == PENDING_OPINIONS else (version,)
    with tenant_transaction(conn, tenant):
        return [dict(row) for row in conn.execute(
            sql, (tenant, last[0], last[1], *versions, page_size)).fetchall()]


def sync_sentiment(conn, tenant: str, classifier, *, limit: int = 1000,
                   page_size: int = 16) -> dict:
    tenant = validate_tenant(tenant)
    if not 1 <= page_size <= min(32, classifier.config.max_batch) or not 1 <= limit <= 100_000:
        raise ValueError('Invalid page or limit')
    stats = {'examined': 0, 'saved': 0, 'stale_or_existing': 0}
    last = ('','')
    _acquire(conn, tenant, 'sm_sentiment')
    try:
        while stats['examined'] < limit:
            rows = _pending(conn, tenant, PENDING_SENTIMENT, last,
                            classifier.config.config_hash, min(page_size, limit-stats['examined']))
            if not rows:
                break
            predictions = classifier.predict_many([r['text'] for r in rows])
            if len(predictions) != len(rows):
                raise RuntimeError('Classifier returned incorrect number of predictions')
            with tenant_transaction(conn, tenant):
                for row, prediction in zip(rows, predictions):
                    params = (classifier.config.config_hash, prediction.model_id,
                              prediction.model_revision, prediction.label,
                              prediction.scores['negative'], prediction.scores['neutral'],
                              prediction.scores['positive'], prediction.truncated, tenant, row['source'],
                              row['review_id'], row['content_hash'])
                    saved = conn.execute(SAVE_SENTIMENT, params).fetchone()
                    stats['saved' if saved else 'stale_or_existing'] += 1
            stats['examined'] += len(rows)
            last = (rows[-1]['source'], rows[-1]['review_id'])
        return stats
    finally:
        _release(conn, tenant, 'sm_sentiment')


def sync_opinions(conn, tenant: str, extractor: OpinionExtractor, *,
                  limit: int = 50, page_size: int = 8) -> dict:
    tenant = validate_tenant(tenant)
    if not 1 <= page_size <= 32 or not 1 <= limit <= 100_000:
        raise ValueError('Invalid page or limit')
    stats = {'examined': 0, 'saved': 0, 'opinions': 0, 'failed': 0, 'stale_or_existing': 0}
    last = ('','')
    _acquire(conn, tenant, 'sm_opinions')
    try:
        while stats['examined'] < limit:
            rows = _pending(conn, tenant, PENDING_OPINIONS, last,
                            extractor.config.config_hash, min(page_size, limit-stats['examined']))
            if not rows:
                break
            for row in rows:
                stats['examined'] += 1
                try:
                    result = extractor.extract(row['text'])  # LLM call outside transaction
                except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                    LOG.warning('Opinion extraction failed for %s/%s: %s', row['source'],
                                row['review_id'], type(exc).__name__)
                    stats['failed'] += 1
                    with tenant_transaction(conn, tenant):
                        conn.execute(SAVE_FAILURE, (
                            extractor.config.config_hash, type(exc).__name__, tenant,
                            row['source'], row['review_id'], row['content_hash']))
                    continue  # Retry after backoff, up to five attempts.
                with tenant_transaction(conn, tenant):
                    saved = conn.execute(SAVE_RUN, (
                        extractor.config.config_hash, extractor.config.model,
                        extractor.config.model_revision, SCHEMA_VERSION,
                        result.overall_sentiment, tenant, row['source'],
                        row['review_id'], row['content_hash'])).fetchone()
                    if not saved:
                        stats['stale_or_existing'] += 1
                        continue
                    for index, op in enumerate(result.opinions):
                        sug = op.customer_suggestion
                        conn.execute(SAVE_OPINION, (
                            tenant,row['source'],row['review_id'],row['content_hash'],
                            extractor.config.config_hash,index,op.aspect,op.target,
                            op.sentiment,op.evidence.quote,op.evidence.start,op.evidence.end,
                            op.reported_problem,sug.quote if sug else None,
                            sug.start if sug else None,sug.end if sug else None))
                    conn.execute(DELETE_FAILURE, (
                        tenant, row['source'], row['review_id'], row['content_hash'],
                        extractor.config.config_hash))
                    stats['saved'] += 1
                    stats['opinions'] += len(result.opinions)
            last = (rows[-1]['source'], rows[-1]['review_id'])
        return stats
    finally:
        _release(conn, tenant, 'sm_opinions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init','sentiment','opinions','both'))
    parser.add_argument('--tenant', default='demo')
    parser.add_argument('--revision', help='Pinned classifier HF SHA')
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    parser.add_argument('--llm-model')
    parser.add_argument('--llm-revision')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    from storage.postgres import connect as pg_connect
    with pg_connect(os.environ['DATABASE_URL']) as conn:
        if args.command == 'init':
            from pathlib import Path
            path = Path(__file__).resolve().parents[1] / 'migrations/020_sentiment_opinions.sql'
            with conn.transaction():
                conn.execute(path.read_text(encoding='utf-8'))
            print('Sentiment/opinion tables ready')
            return
        if args.command in ('sentiment','both') and not args.revision:
            parser.error('--revision required for sentiment')
        if args.command in ('opinions','both') and (not args.llm_model or not args.llm_revision):
            parser.error('--llm-model and --llm-revision required for opinions')
        result = {}
        if args.command in ('sentiment','both'):
            classifier = LocalSentimentClassifier(SentimentConfig(
                revision=args.revision, model_id=args.model, device=args.device))
            try:
                result['sentiment'] = sync_sentiment(conn, args.tenant, classifier, limit=args.limit)
            finally:
                classifier.close()
        if args.command in ("opinions", "both"):
            extractor = OpinionExtractor(
                OpinionConfig(
                    model=args.llm_model,
                    model_revision=args.llm_revision
                )
            )
            try:
                extractor.verify_model_digest()

                result["opinions"] = sync_opinions(
                    conn, args.tenant, extractor,
                    limit=args.limit
                )
            finally:
                extractor.close()
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
