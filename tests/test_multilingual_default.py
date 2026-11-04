"""Regression tests for fixed multilingual defaults and tenant-scoped NER."""
from contextlib import contextmanager
from types import SimpleNamespace, ModuleType
import sys

import pytest

from enrichment.auto_nlp import analyze_one, ner_profile
from enrichment import language, ner_auto
from enrichment.ner import Config
from enrichment.topics import Settings
from enrichment.topic_policy import Policy
from enrichment.topic_auto import discovery_profile


@pytest.mark.parametrize("lang", ["en", "es", "fr", "ar", "zh", "und", "mul", "de"])
def test_ner_always_multilingual_by_default(lang):
    assert ner_profile(lang) == "multilingual"


def test_override_must_be_explicit():
    assert ner_profile("und", override="accurate") == "accurate"
    with pytest.raises(ValueError):
        ner_profile("en", override="invented")


def test_one_review_does_not_route_by_detected_language():
    detection = language.LanguageResult(
        "en", "detected", 0.95, 0.5, "lingua", "test", "hash")
    result = analyze_one("The payment failed.", tenant="A", ner_enabled=False,
                         detector=lambda _: detection)
    assert result["ner_profile"] == "multilingual"
    assert result["language"]["language"] == "en"
    override = analyze_one("The payment failed.", tenant="A", ner_enabled=False,
                           detector=lambda _: detection,
                           ner_profile_override="accurate")
    assert override["ner_profile"] == "accurate"


def test_new_pipeline_and_topic_defaults_are_multilingual():
    assert Config().profile == "multilingual"
    assert Settings().profile == "multilingual_m3"
    assert Policy().profile == "multilingual_m3"
    assert discovery_profile(None, "demo") == "multilingual_m3"
    assert discovery_profile(None, "demo", "accurate") == "accurate"


class Db:
    def __init__(self, rows):
        self.rows = rows
        self.context = None
        self.saved = []
        self.requested = []

    def execute(self, sql, params=()):
        text = str(sql)
        if "CREATE TEMP TABLE" in text:
            return SimpleNamespace()
        if "FROM public.feedback_reviews f" in text:
            assert self.context == params[0]
            after = (params[1], params[2])
            result = [r for r in self.rows if (r["source"],r["review_id"]) > after]
            return SimpleNamespace(fetchall=lambda: result[:params[-1]])
        raise AssertionError(f"Unexpected SQL: {text[:80]}")


@contextmanager
def tenant_context(conn, tenant):
    assert tenant == "A"
    assert conn.context is None
    conn.context = tenant
    try:
        yield conn
    finally:
        conn.context = None


class Model:
    config_hash = "model-v1"
    revision = "sha"
    def extract_many(self, texts):
        for text in texts:
            yield [{"text": text, "label": "product"}]


def test_batch_multilingual_single_model_and_reprocessing(monkeypatch):
    rows = [dict(source="review",review_id=str(i),text=text,content_hash=f"h{i}",
                 ner_hash=None, ner_config=None) for i,text in enumerate([
                     "The payment failed", "El pago falla", "عملية الدفع فشلت", "支付失败"
                 ])]
    db=Db(rows)
    monkeypatch.setattr(ner_auto, "tenant_transaction", tenant_context)
    fake=ModuleType("enrichment.ner")
    def write(conn, tenant, model, items):
        assert conn.context==tenant
        conn.saved.extend((item["review_id"], model.config_hash) for item,_ in items)
        return len(items), len(items)
    fake._write_batch=write
    monkeypatch.setitem(sys.modules, "enrichment.ner", fake)
    calls=[]
    result=ner_auto.sync_auto(db,"A",page_size=2,
             ner_factory=lambda profile:(calls.append(profile) or Model()))
    assert calls==["multilingual"]
    assert result["reviews_processed"]==4
    assert result["entities"]==4
    assert len(db.saved)==4
    assert db.context is None


def test_bad_tenant_and_profile_fail_before_queries():
    db=Db([])
    with pytest.raises(ValueError):
        ner_auto.sync_auto(db," A ")
    with pytest.raises(ValueError):
        ner_auto.sync_auto(db,"A",profile="wrong")


@contextmanager
def fake_scope(conn, tenant):
    assert tenant == 'demo'
    yield conn


class CountsDb:
    def __init__(self, counts):
        self.counts = counts
    def execute(self, sql, params=()):
        assert params[-1] == 'demo'
        return SimpleNamespace(fetchall=lambda: [
            {'language': lang, 'n': n} for lang, n in self.counts.items()])


def test_language_statistics_never_decide_model(monkeypatch):
    monkeypatch.setattr(language, "tenant_transaction", fake_scope)
    monkeypatch.setattr(language, "config_hash", lambda *_:"test")
    for counts in ({"en": 516, "und": 486}, {"en": 998, "es": 2},
                   {"en":1000}, {"und": 200}, {}):
        result=language.tenant_stats(CountsDb(counts),"demo")
        assert result["topic_profile_recommended"]=="multilingual_m3"
    assert language.tenant_stats(CountsDb({"en":516,"und":486}),"demo")['unknown']==486
