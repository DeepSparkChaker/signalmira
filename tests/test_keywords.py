"""Fast topic-label tests: no model download or database required."""
import sys
import types

from enrichment.topics import _english_tokens, _keyword_text, _language_counts, _multilingual_tokens


def test_english_preserves_relevant_negation_phrase():
    text = "The battery doesn't work and the camera is great."
    keywords = _keyword_text(text, profile="accurate")
    assert "battery" in keywords
    assert "not_work" in keywords
    assert "the" not in keywords.split()
    assert "and" not in keywords.split()


def test_old_representative_review_no_longer_only_stopwords():
    docs = ["Poor product.", "Doesn't do the job.", "Not good enough for the price."]
    keys = [_keyword_text(d, profile="accurate") for d in docs]
    assert "poor" in keys[0]
    assert "job" in keys[1]
    assert "price" in keys[2]
    assert all("the" not in k.split() for k in keys)


def test_model_text_remains_intact():
    text = "The battery is not working."
    _ = _keyword_text(text, profile="accurate")
    assert text == "The battery is not working."


def test_contractions():
    assert _english_tokens("Won't charge. Can't connect. Doesn't work.") == [
        'will', 'not', 'charge', 'can', 'not', 'connect', 'does', 'not', 'work'
    ]


def test_multilingual_language_specific_stopwords(monkeypatch):
    import enrichment.topics as mod
    mod._stopwords.cache_clear()
    monkeypatch.setitem(sys.modules, "stopwordsiso", types.SimpleNamespace(
        stopwords=lambda lang: {"en": {"the", "and"}, "fr": {"la", "et"},
                                 "es": {"la", "el"}, "zh": {"的"},
                                 "ar": {"في", "ال"}}.get(lang, set())))
    monkeypatch.setitem(sys.modules, "jieba", types.SimpleNamespace(
        cut=lambda text, HMM=False: ["电池", "耗电"] if text == "电池耗电" else [text]))
    try:
        assert 'batería' in _keyword_text('La batería no funciona', profile='multilingual_m3', language='es')
        assert 'no_funciona' in _keyword_text('La batería no funciona', profile='multilingual_m3', language='es')
        assert '电池' in _keyword_text('电池耗电', profile='multilingual_m3', language='zh')
        assert 'في' not in _keyword_text('البطارية في الهاتف', profile='multilingual_m3', language='ar').split()
    finally:
        mod._stopwords.cache_clear()


def test_language_counts():
    assert _language_counts([{"lang": "en"}, {"lang": "fr"}, {"lang": "en"}]) == {'en':2,'fr':1}
