"""Explicitly opt-in integration test against local PostgreSQL.

Requires TEST_DATABASE_URL and RUN_SIGNALMIRA_LANGUAGE_DB_TEST=1.
Creates isolated random tenant rows and deletes them afterward.
Does not enable any RLS on the rest of SignalMira.
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from enrichment import language as lg
from security.tenant_context import tenant_transaction

pytestmark = pytest.mark.skipif(
    not os.getenv('TEST_DATABASE_URL') or
    os.getenv('RUN_SIGNALMIRA_LANGUAGE_DB_TEST') != '1',
    reason='Requires TEST_DATABASE_URL and RUN_SIGNALMIRA_LANGUAGE_DB_TEST=1',
)


def test_incremental_postgres_tenant_scope(monkeypatch):
    import psycopg
    from psycopg.rows import dict_row

    monkeypatch.setattr(lg, '_detector_version', lambda: 'integration-test-v1')
    t_a='sm-langa-'+uuid4().hex[:15]
    t_b='sm-langb-'+uuid4().hex[:15]
    text_a='The payment processing failed again.'
    text_b='El pago falla otra vez.'
    with psycopg.connect(os.environ['TEST_DATABASE_URL'],
                         autocommit=True,row_factory=dict_row) as conn:
        table_exists=conn.execute("SELECT to_regclass('public.feedback_language') AS r").fetchone()['r']
        assert table_exists is not None, 'Run: python -m enrichment.language init'
        try:
            for tenant, text, hashed in ((t_a,text_a,'h-a'),(t_b,text_b,'h-b')):
                with tenant_transaction(conn,tenant):
                    conn.execute("""INSERT INTO feedback_reviews
                        (tenant_id,source,review_id,text,content_hash)
                        VALUES (%s,'langtest','r1',%s,%s)""",(tenant,text,hashed))
            fn=lambda text: lg.LanguageResult(
                'en' if text==text_a else 'es', 'detected', .97, .94,
                lg.DETECTOR, 'integration-test-v1', lg.config_hash('integration-test-v1'))
            assert lg.sync(conn,t_a,detect_fn=fn)['written']==1
            assert lg.sync(conn,t_b,detect_fn=fn)['written']==1
            assert lg.sync(conn,t_a,detect_fn=fn)['written']==0
            with tenant_transaction(conn,t_a):
                rows=conn.execute('SELECT tenant_id,language FROM feedback_language').fetchall()
                assert rows==[{'tenant_id':t_a,'language':'en'}]
            with tenant_transaction(conn,t_b):
                rows=conn.execute('SELECT tenant_id,language FROM feedback_language').fetchall()
                assert rows==[{'tenant_id':t_b,'language':'es'}]
        finally:
            for tenant in (t_a,t_b):
                with tenant_transaction(conn,tenant):
                    conn.execute("""DELETE FROM feedback_reviews
                         WHERE tenant_id=%s AND source='langtest' AND review_id='r1'""",(tenant,))
