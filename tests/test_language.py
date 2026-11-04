"""Offline language/storage tests, using a fake Lingua scoring engine.

Run separately from live-model quality evaluation. Does NOT claim that the
real Lingua classifier is accurate on these examples without installing it.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from enrichment import language as lg
from enrichment.auto_nlp import analyze_one, ner_profile
from enrichment.topic_auto import discovery_profile


class FakeLanguage:
    def __init__(self, short: str):
        self.iso_code_639_1 = SimpleNamespace(name=short.upper())


class FakeChoice:
    def __init__(self, code: str, value: float):
        self.language = FakeLanguage(code)
        self.value = value


class FakeEngine:
    def __init__(self, choices):
        self.choices = [FakeChoice(*choice) for choice in choices]
        self.calls = 0

    def compute_language_confidence_values(self, text):
        self.calls += 1
        return self.choices


@pytest.fixture(autouse=True)
def detector_version(monkeypatch):
    monkeypatch.setattr(lg, '_detector_version', lambda: 'test-1')


def fake_result(lang, *, version='test-1'):
    status = 'mixed' if lang == 'mul' else 'unknown' if lang == 'und' else 'detected'
    return lg.LanguageResult(lang, status, 0.95 if status == 'detected' else None,
                             0.80 if status == 'detected' else None,
                             lg.DETECTOR, version, lg.config_hash(version))


@pytest.mark.parametrize('code,text', [
    ('en', 'The payment processing never works and the battery is hot.'),
    ('es', 'El pago falla constantemente y necesito ayuda urgente.'),
    ('fr', 'Le paiement échoue sans cesse, je souhaite une solution.'),
    ('ar', 'عملية الدفع تفشل باستمرار ولم يقدم الدعم أي حل مناسب.'),
    ('zh', '支付一直失败而且无法联系到客户支持人员。'),
])
def test_language_from_fake_classifier(code, text):
    engine = FakeEngine([(code, 0.97), ('en' if code != 'en' else 'fr', 0.03)])
    out = lg.detect_text(text, engine=engine)
    assert out.language == code
    assert out.status == 'detected'
    assert out.margin > 0.8
    assert engine.calls == 1
    assert out.config_hash == lg.config_hash()


@pytest.mark.parametrize('text', ['', '   ', 'ok', '123456789', '👍👍👍'])
def test_unknown_without_model(text):
    assert lg.detect_text(text, engine=None).language == 'und'


def test_ambiguous_returns_unknown():
    x = lg.detect_text('Poor quality of customer service.', engine=FakeEngine([('en', .52), ('fr', .48)]))
    assert x.language == 'und'
    assert x.confidence == .52


def test_mixed_script_uses_multilingual_without_model():
    text = 'English support is continuously failing.' + ' عملية الدفع تفشل باستمرار في كل مرة'
    out = lg.detect_text(text)
    assert out.language == 'mul'
    assert ner_profile(out.language) == 'multilingual'


def test_original_text_not_modified():
    text = '  L’interface  💥 不好!  '
    original = text[:]
    lg.detect_text(text, engine=FakeEngine([('fr', .96), ('en', .04)]))
    assert text == original


def test_profiles():
    assert ner_profile('en') == 'multilingual'
    for code in ('es', 'fr', 'ar', 'zh', 'und', 'mul'):
        assert ner_profile(code) == 'multilingual'


def test_realtime_does_not_write_or_load_NER():
    result = analyze_one('El pago falla constantemente.', tenant='tenant-A',
                         ner_enabled=False, detector=lambda _: fake_result('es'))
    assert result['ner_profile'] == 'multilingual'
    assert result['entities'] == [] and result['opinions'] is None
    assert result['language']['language'] == 'es'


def test_realtime_calls_optional_pipeline():
    class Worker:
        def extract(self, text):
            return [{'text': 'NovaPay', 'label': 'company', 'start': 0, 'end': 7}]
    x = analyze_one('NovaPay fails each week.', tenant='tenant-A',
                    detector=lambda _: fake_result('en'),
                    ner_engine=Worker(), opinion_extractor=lambda txt: {'issue':'payment failures'},
                    topic_predictor=lambda txt: 'payments')
    assert x['ner_profile'] == 'multilingual'
    assert x['entities'][0]['text'] == 'NovaPay'
    assert x['opinions']['issue'] == 'payment failures'
    assert x['topic'] == 'payments'


@dataclass
class FakeReview:
    tenant: str
    source: str
    review_id: str
    text: str
    content_hash: str


class ResultCursor:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return self.rows


class FakeConn:
    """Psycopg-like transaction behavior plus a minimal in-memory SQL model."""
    autocommit = True

    def __init__(self, reviews):
        self.reviews = list(reviews)
        self.languages = {}
        self.context = None
        self.info = SimpleNamespace(transaction_status=SimpleNamespace(name='IDLE'))
        self.queries = []

    @contextmanager
    def transaction(self):
        assert self.info.transaction_status.name == 'IDLE'
        self.info.transaction_status.name = 'INTRANS'
        try:
            yield self
        finally:
            self.info.transaction_status.name = 'IDLE'
            self.context = None

    def execute(self, sql, params=()):
        sql = str(sql)
        self.queries.append(sql)
        if "set_config('app.tenant_id'" in sql:
            self.context = params[0]
            return ResultCursor([])
        assert self.context is not None, 'tenant query outside RLS transaction'
        if 'FROM public.feedback_reviews AS f' in sql and 'ORDER BY f.source,f.review_id' in sql:
            tenant, after_source, after_id, key, limit = params
            assert tenant == self.context
            rows = []
            for f in sorted(self.reviews, key=lambda f: (f.source,f.review_id)):
                if f.tenant != tenant or (f.source,f.review_id) <= (after_source,after_id):
                    continue
                old = self.languages.get((f.tenant,f.source,f.review_id))
                if old and old['input_hash']==f.content_hash and old['config_hash']==key:
                    continue
                rows.append({'source':f.source,'review_id':f.review_id,
                             'text':f.text,'content_hash':f.content_hash})
                if len(rows) >= limit:
                    break
            return ResultCursor(rows)
        if 'jsonb_to_recordset' in sql:
            data, tenant = params
            assert tenant == self.context
            result = []
            for item in json.loads(data):
                source_row = next((r for r in self.reviews if (r.tenant,r.source,r.review_id)==
                                   (tenant,item['source'],item['review_id'])),None)
                if source_row is None or source_row.content_hash != item['input_hash']:
                    continue
                identity=(tenant,item['source'],item['review_id'])
                old=self.languages.get(identity)
                if old and old['input_hash']==item['input_hash'] and old['config_hash']==item['config_hash']:
                    continue
                self.languages[identity]=item
                result.append({'1':1})
            return ResultCursor(result)
        if 'GROUP BY COALESCE(l.language' in sql:
            key, tenant = params
            assert tenant == self.context
            counts = {}
            for r in self.reviews:
                if r.tenant != tenant: continue
                v=self.languages.get((tenant,r.source,r.review_id))
                code=v['language'] if v and v['input_hash']==r.content_hash and v['config_hash']==key else 'pending'
                counts[code]=counts.get(code,0)+1
            return ResultCursor([{'language':code,'n':n} for code,n in counts.items()])
        raise AssertionError('Unexpected SQL: '+sql[:200])


def test_incremental_sync_idempotency_and_tenant_isolation():
    reviews = [FakeReview('A', 's',f'r{i}', 'English text about failing payments',f'h{i}') for i in range(40)]
    reviews += [FakeReview('B', 's', 'r1', 'Pago falló en España', 'other-hash')]
    conn = FakeConn(reviews)
    fn=lambda text: fake_result('en' if 'English' in text else 'es')
    first=lg.sync(conn, 'A', batch_size=7, detect_fn=fn)
    assert first == {'examined': 40, 'written': 40, 'changed_during_detection': 0}
    assert len(conn.languages)==40 and not any(key[0]=='B' for key in conn.languages)
    assert lg.sync(conn, 'A', detect_fn=fn)['written'] == 0
    stats=lg.tenant_stats(conn, 'A')
    assert stats['topic_profile_recommended']=='multilingual_m3'
    assert stats['english']==40 and stats['pending']==0
    assert lg.tenant_stats(conn, 'B')['pending'] == 1
    reviews[0].text='El pago falla otra vez'
    reviews[0].content_hash='new-hash'
    changed=lg.sync(conn,'A',detect_fn=fn)
    assert changed['written']==1
    new_stats=lg.tenant_stats(conn,'A')
    assert new_stats['non_english'] == 1
    assert new_stats['topic_profile_recommended'] == 'multilingual_m3'
    assert conn.context is None and conn.info.transaction_status.name=='IDLE'


def test_language_change_during_detection_is_rejected():
    row=FakeReview('A','s','r1','Any complaints about payment','h1')
    conn=FakeConn([row])
    def change_during_run(text):
        row.content_hash = 'h2'
        return fake_result('en')
    result=lg.sync(conn,'A',detect_fn=change_during_run)
    assert result['changed_during_detection']==1
    assert not conn.languages


def test_explicit_topic_profile_overrides_stats():
    conn=FakeConn([FakeReview('A','s','1','bonjour','h1')])
    assert discovery_profile(conn,'A')=='multilingual_m3'
    assert discovery_profile(conn,'A',explicit='fast')=='fast'


def test_invalid_tenant_does_not_query():
    conn=FakeConn([])
    with pytest.raises(ValueError):
        lg.sync(conn,' A ')
    assert conn.queries == []
