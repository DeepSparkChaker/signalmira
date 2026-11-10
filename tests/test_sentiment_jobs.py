"""Offline fake-DB regression tests for idempotent/stale incremental writes."""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_patch_root = str(Path(__file__).resolve().parents[1])
if _patch_root not in sys.path:
    sys.path.insert(0, _patch_root)
try:
    import security.tenant_context
except ModuleNotFoundError:
    security = types.ModuleType('security')
    tenant_context = types.ModuleType('security.tenant_context')
    tenant_context.tenant_transaction = lambda conn, tenant: contextlib.nullcontext()
    tenant_context.validate_tenant = lambda tenant: tenant if tenant else (_ for _ in ()).throw(ValueError())
    security.tenant_context = tenant_context
    sys.modules['security'] = security
    sys.modules['security.tenant_context'] = tenant_context

from enrichment import sentiment_jobs as jobs
from enrichment.sentiment import SentimentConfig, SentimentPrediction
from enrichment.opinions import OpinionConfig
from enrichment.opinion_schema import ReviewOpinions

HASH = 'a'*64
TEXT = 'Battery fails.'


class Cursor:
    def __init__(self, one=None, all_=None):
        self.one, self.all_ = one, all_ or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.all_


class Connection:
    def __init__(self, *, stale=False):
        self.calls = []
        self.pages = 0
        self.stale = stale

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if 'pg_try_advisory_lock' in sql:
            return Cursor(one={'ok': True})
        if 'pg_advisory_unlock' in sql:
            return Cursor(one={'pg_advisory_unlock': True})
        if sql == jobs.PENDING_SENTIMENT or sql == jobs.PENDING_OPINIONS:
            self.pages += 1
            return Cursor(all_=[{'source':'uci','review_id':'r1','text':TEXT,'content_hash':HASH}]
                          if self.pages == 1 else [])
        if sql == jobs.SAVE_SENTIMENT or sql == jobs.SAVE_RUN:
            return Cursor(one=None if self.stale else {'review_id': 'r1'})
        if sql == jobs.SAVE_OPINION or sql == jobs.DELETE_FAILURE or sql == jobs.SAVE_FAILURE:
            return Cursor()
        raise AssertionError('Unexpected SQL')

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeClassifier:
    config = SentimentConfig(revision='a'*40)

    def predict_many(self, texts):
        return [SentimentPrediction('negative',
                {'negative': 0.9, 'neutral': 0.05, 'positive': 0.05},
                self.config.model_id, self.config.revision, self.config.config_hash)
                for _ in texts]


class FakeExtractor:
    config = OpinionConfig(model='test', model_revision='a'*64)

    def extract(self, text):
        return ReviewOpinions.model_validate({
            'overall_sentiment':'negative', 'opinions':[{
                'aspect':'battery', 'sentiment':'negative',
                'evidence':{'quote':'Battery fails','start':0,'end':13}}
            ]}).verify(text)


class JobTests(unittest.TestCase):
    def test_sentiment_write_is_hash_guarded(self):
        conn = Connection()
        with patch.object(jobs, 'tenant_transaction', lambda *_: contextlib.nullcontext()):
            stats = jobs.sync_sentiment(conn, 'demo', FakeClassifier())
        self.assertEqual(stats, {'examined':1, 'saved':1, 'stale_or_existing':0})
        sql, params = next((sql,p) for sql,p in conn.calls if sql == jobs.SAVE_SENTIMENT)
        self.assertIn('f.content_hash=%s', sql)
        self.assertEqual(params[-1], HASH)

    def test_stale_source_is_not_saved(self):
        conn = Connection(stale=True)
        with patch.object(jobs, 'tenant_transaction', lambda *_: contextlib.nullcontext()):
            stats = jobs.sync_sentiment(conn, 'demo', FakeClassifier())
        self.assertEqual(stats['stale_or_existing'], 1)

    def test_opinions_are_atomic_with_run(self):
        conn = Connection()
        with patch.object(jobs, 'tenant_transaction', lambda *_: contextlib.nullcontext()):
            stats = jobs.sync_opinions(conn, 'demo', FakeExtractor())
        self.assertEqual(stats['saved'], 1)
        self.assertEqual(stats['opinions'], 1)
        self.assertEqual(sum(1 for sql,_ in conn.calls if sql == jobs.SAVE_OPINION),1)

    def test_failed_extraction_is_retryable_and_tracked(self):
        conn = Connection()
        class Broken(FakeExtractor):
            def extract(self, text):
                raise ValueError('invalid evidence')
        with patch.object(jobs, 'tenant_transaction', lambda *_: contextlib.nullcontext()):
            stats = jobs.sync_opinions(conn, 'demo', Broken())
        self.assertEqual(stats['failed'], 1)
        self.assertTrue(any(sql == jobs.SAVE_FAILURE for sql,_ in conn.calls))
        self.assertTrue('e.attempts >= 5' in jobs.PENDING_OPINIONS)

    def test_stale_opinion_no_child_inserts(self):
        conn = Connection(stale=True)
        with patch.object(jobs, 'tenant_transaction', lambda *_: contextlib.nullcontext()):
            stats = jobs.sync_opinions(conn, 'demo', FakeExtractor())
        self.assertEqual(stats['stale_or_existing'], 1)
        self.assertFalse(any(sql == jobs.SAVE_OPINION for sql,_ in conn.calls))


if __name__ == '__main__':
    unittest.main()
