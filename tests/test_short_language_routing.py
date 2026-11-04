"""Regression: short/unknown reviews don't change model selection."""
from contextlib import contextmanager
from types import SimpleNamespace

from enrichment import language as lg
from enrichment.auto_nlp import ner_profile, analyze_one


@contextmanager
def scope(conn, tenant):
    assert tenant == conn.allowed
    yield conn


class Db:
    allowed='demo'
    def __init__(self, counts): self.counts=counts
    def execute(self, query, params=()):
        assert params[-1]==self.allowed
        return SimpleNamespace(fetchall=lambda: [
            {'language':lang, 'n':n} for lang,n in self.counts.items()])


def test_unknowns_do_not_force_any_model(monkeypatch):
    monkeypatch.setattr(lg,'tenant_transaction',scope)
    monkeypatch.setattr(lg,'config_hash',lambda *_:'test')
    for counts in ({'en':516,'und':486}, {'en':516,'und':485,'es':1}):
        stats=lg.tenant_stats(Db(counts),'demo')
        assert stats['topic_profile_recommended']=='multilingual_m3'
        assert stats['unknown']==counts['und']


def test_all_languages_share_multilingual_ner_by_default():
    for lang in ('en','und','es','fr','zh','mul'):
        assert ner_profile(lang)=='multilingual'
    assert ner_profile('und',override='accurate')=='accurate'


def test_preview_retains_unknown_as_metadata():
    detection=lg.LanguageResult('und','unknown',None,None,lg.DETECTOR,'test','hash')
    output=analyze_one('Great phone!',tenant='demo',ner_enabled=False,
                       detector=lambda _:detection)
    assert output['ner_profile']=='multilingual'
    assert output['language']['language']=='und'
