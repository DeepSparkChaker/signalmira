"""Inspect or enrich ONE already-saved SignalMira review, then validate aspect search.

init: additive 021 migration. inspect: READ-ONLY. sync-one: opt-in inference + DB write.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from enrichment import aspect_groups as ag
from security.tenant_context import tenant_transaction
from storage.postgres import connect

MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/021_aspect_groups.sql'


def sync_one(conn, tenant: str, source: str, review_id: str,
             model: str, revision: str) -> dict:
    """Compute + save a single current opinion run. Model inference OUTSIDE SQL tx."""
    from enrichment.langchain_opinions import LangChainOpinionExtractor
    from enrichment.opinions import OpinionConfig
    from enrichment.opinion_schema import SCHEMA_VERSION

    tenant = ag.validate_tenant(tenant)
    config = OpinionConfig(model=model, model_revision=revision, provider='ollama')
    extractor = LangChainOpinionExtractor(config)
    try:
        with tenant_transaction(conn, tenant):
            row = conn.execute("""
                SELECT source, review_id, text, content_hash
                FROM public.feedback_reviews
                WHERE tenant_id=%s AND source=%s AND review_id=%s
            """, (tenant, source, review_id)).fetchone()
            if row is None:
                raise ValueError('Saved review not found')
            exists = conn.execute("""
                SELECT 1 FROM public.feedback_opinion_runs
                WHERE tenant_id=%s AND source=%s AND review_id=%s
                  AND input_hash=%s AND extractor_key=%s
            """, (tenant, source, review_id, row['content_hash'],
                  extractor.config.config_hash)).fetchone()
        if exists:
            return {'status': 'already_saved', 'extractor_key': extractor.config.config_hash}

        extractor.verify_model_digest()
        result = extractor.extract(row['text'])

        with tenant_transaction(conn, tenant):
            saved = conn.execute("""
                INSERT INTO public.feedback_opinion_runs
                  (tenant_id,source,review_id,input_hash,extractor_key,
                   model_name,model_revision,schema_version,llm_overall)
                SELECT f.tenant_id,f.source,f.review_id,f.content_hash,
                       %s,%s,%s,%s,%s
                FROM public.feedback_reviews f
                WHERE f.tenant_id=%s AND f.source=%s AND f.review_id=%s
                  AND f.content_hash=%s
                ON CONFLICT DO NOTHING RETURNING review_id
            """, (extractor.config.config_hash, config.model, config.model_revision,
                  SCHEMA_VERSION, result.overall_sentiment, tenant, source, review_id,
                  row['content_hash'])).fetchone()
            if saved is None:
                return {'status': 'stale_or_existing', 'extractor_key': extractor.config.config_hash}
            for index, op in enumerate(result.opinions):
                sug = op.customer_suggestion
                conn.execute("""
                    INSERT INTO public.feedback_opinions
                       (tenant_id, source, review_id, input_hash, extractor_key,
                        opinion_index, aspect, target, sentiment, evidence_quote,
                        start_index, end_index, reported_problem,
                        suggestion_quote, suggestion_start, suggestion_end)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (tenant, source, review_id, row['content_hash'],
                      extractor.config.config_hash, index, op.aspect, op.target,
                      op.sentiment, op.evidence.quote, op.evidence.start,
                      op.evidence.end, op.reported_problem,
                      sug.quote if sug else None, sug.start if sug else None,
                      sug.end if sug else None))
        return {'status': 'saved', 'opinions': len(result.opinions),
                'extractor_key': extractor.config.config_hash}
    finally:
        extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'inspect', 'sync-one', 'smoke'))
    parser.add_argument('--tenant', default='demo')
    parser.add_argument('--source', default='uci-amazon')
    parser.add_argument('--review-id', default='uci-amazon-0694')
    parser.add_argument('--model', default='qwen3.5:4b')
    parser.add_argument('--revision', help='64-character Ollama digest; required for sync-one')
    args = parser.parse_args()
    if args.command == 'sync-one' and not args.revision:
        parser.error('--revision is required for sync-one')
    with connect(os.environ['DATABASE_URL']) as conn:
        if args.command == 'init':
            with conn.transaction():
                conn.execute(MIGRATION.read_text(encoding='utf-8'))
            print('Aspect group tables ready')
            return
        if args.command == 'sync-one':
            print(json.dumps(sync_one(conn, args.tenant, args.source,
                                      args.review_id, args.model, args.revision), indent=2))
        review = ag.saved_review(conn, args.tenant, args.source, args.review_id)
        print(json.dumps(review, indent=2, ensure_ascii=False, default=str))
        if args.command == 'smoke':
            if review is None or not review['opinions']:
                raise RuntimeError('No CURRENT saved opinions: run sync-one for this review first')
            first = review['opinions'][0]
            key, aspect = first['extractor_key'], first['aspect']
            hits = ag.find_reviews(conn, args.tenant, key, [aspect], limit=10)
            trend = ag.weekly_trend(conn, args.tenant, key, [aspect])
            found = any(h['source'] == args.source and h['review_id'] == args.review_id
                        for h in hits)
            print(json.dumps({'aspect': aspect, 'selected_model_key': key,
                              'saved_review_is_in_first_10_matches': found,
                              'matching_opinion_rows': len(hits),
                              'weekly_points': len(trend),
                              'weekly': trend}, ensure_ascii=False, default=str, indent=2))
            return
        runs = ag.list_extractor_runs(conn, args.tenant)
        print('Saved CURRENT model versions:',
              json.dumps(runs, indent=2, ensure_ascii=False, default=str))
        if runs:
            print('Available aspects:', json.dumps(
                ag.list_aspects(conn, args.tenant, runs[0]['extractor_key']),
                indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
