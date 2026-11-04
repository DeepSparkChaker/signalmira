from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from enrichment import ner_auto


class Cursor:
    def __init__(self, rows): self.rows=rows
    def fetchall(self): return self.rows


class FakeDb:
    autocommit=True
    def __init__(self):
        self.context=None
        self.info=SimpleNamespace(transaction_status=SimpleNamespace(name='IDLE'))
        self.rows=[
            dict(source='s',review_id='a',text='The payment failed',content_hash='h1',
                 language='en',language_hash='h1',ner_hash=None,ner_config=None),
            dict(source='s',review_id='b',text='El pago falló muchas veces',content_hash='h2',
                 language='es',language_hash='h2',ner_hash=None,ner_config=None),
            dict(source='s',review_id='c',text='支付一直失败',content_hash='h3',
                 language=None,language_hash=None,ner_hash=None,ner_config=None),
            dict(source='s',review_id='x',text='Other company',content_hash='h4',
                 language='en',language_hash='h4',ner_hash=None,ner_config=None),
        ]
        self.created=False
        self.saved=[]

    @contextmanager
    def transaction(self):
        assert self.info.transaction_status.name == 'IDLE'
        self.info.transaction_status.name='INTRANS'
        try: yield self
        finally:
            self.context=None
            self.info.transaction_status.name='IDLE'

    def execute(self, sql, params=()):
        text=str(sql)
        if 'set_config' in text:
            self.context=params[0]
            return Cursor([])
        if 'CREATE TEMP TABLE' in text:
            self.created=True
            return Cursor([])
        if 'FROM public.feedback_reviews f' in text:
            assert self.context == params[0]
            after=(params[1],params[2])
            rows=[r for r in self.rows if r['review_id'] != 'x' and
                  (r['source'],r['review_id'])>after]
            return Cursor(rows[:params[-1]])
        raise AssertionError('unexpected SQL '+text[:160])


class FakeNer:
    def __init__(self, profile): self.profile=profile
    @property
    def config_hash(self): return 'test:'+self.profile
    @property
    def revision(self): return 'revision:'+self.profile
    def extract_many(self,texts):
        for txt in texts:
            yield [{'text': txt, 'label':'feedback', 'start':0, 'end':len(txt)}]


def test_auto_ner_routes_each_review_and_avoids_mixing(monkeypatch):
    db=FakeDb()
    fake_module=ModuleType('enrichment.ner')
    def write_batch(conn, tenant, engine, group):
        assert conn.context==tenant
        conn.saved.append((engine.profile,[r['review_id'] for r,_ in group]))
        return (len(group),sum(len(mentions) for _,mentions in group))
    fake_module._write_batch=write_batch
    monkeypatch.setitem(sys.modules,'enrichment.ner',fake_module)
    result=ner_auto.sync_auto(db,'A',page_size=2,
         ner_factory=lambda profile:FakeNer(profile))
    assert result['reviews_processed']==3
    assert result['profiles_loaded']==['multilingual']
    assert db.saved == [('multilingual',['a','b']),('multilingual',['c'])]
    assert db.context is None


def test_ner_auto_fails_on_untrusted_tenant():
    db=FakeDb()
    with pytest.raises(ValueError):
        ner_auto.sync_auto(db,' A ')
    assert not db.created
