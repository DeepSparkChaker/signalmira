"""Fast tests: SQL parameterization, bounded reading and safe output."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend import service

ALL = {k: True for k in ("feedback_ner_state", "feedback_entities", "entity_links",
    "sm_topic_active", "sm_topic_discovery_state", "sm_topic_taxonomies",
    "sm_topic_assignments", "sm_topic_definitions")}


def test_filters_use_parameters_and_limit_before_enrichment():
    query, params = service.build_review_query(source="uci-amazon", keyword="battery' OR true --",
        entity="product", entity_name="Pixel", topic_id=5, has_ner=True,
        tables=ALL, taxonomy_id="tenant-taxonomy-id")
    assert query.count("%s") == len(params) + 1  # tenant is prepended by caller
    assert "battery' OR true --" not in query
    assert "LIMIT 10" in query and query.index("LIMIT 10") < query.index("jsonb_agg")
    assert "content_hash=f.content_hash" in query
    assert "sm_topic_assignments" in query
    assert params == ["uci-amazon", "battery' OR true --", "product", "Pixel",
                      "tenant-taxonomy-id", 5, "tenant-taxonomy-id"]


def test_empty_optional_tables():
    empty = dict.fromkeys(ALL,False)
    query, params = service.build_review_query(tables=empty)
    assert "feedback_entities" not in query
    assert "sm_topic_assignments" not in query
    assert "LIMIT 10" in query and len(params) == 0
    assert query.count("%s") == 1


def test_highlights_escape_review_and_labels():
    text = '<script>alert(1)</script> Pixel 9'
    markup = service.highlights(text,[{"text":"Pixel 9","label":"product",
                                       "start":26, "end":33,"confidence":0.85}])
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "Pixel 9" in markup
    assert "85%" in markup


def test_overlapping_mentions_avoid_duplicate_text():
    text = "Google Pixel"
    markup = service.highlights(text,[
        {"text":"Google","label":"brand","start":0,"end":6,"confidence":0.9},
        {"text":"Google Pixel","label":"product","start":0,"end":12,"confidence":0.7}])
    assert markup.count("Google Pixel") == 1


def test_tenant_allowlist(monkeypatch):
    monkeypatch.setenv("SIGNALMIRA_DEMO_TENANTS","demo, client-a, demo")
    assert service.tenants() == ("demo","client-a")
    try:
        service.guard("evil")
    except PermissionError:
        pass
    else:
        assert False, "Unallowed tenant accepted"


def test_check_model_offline_never_downloads(monkeypatch):
    from frontend import prepare_models
    import types
    import sys
    calls = []
    def fake_snapshot_download(*,repo_id,local_files_only):
        calls.append((repo_id,local_files_only))
        return '/cached/model'
    fake_hub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    fake_ner = types.SimpleNamespace(MODELS={"balanced":"fastino/gliner2.5-small-v1"})
    monkeypatch.setitem(sys.modules,'huggingface_hub',fake_hub)
    monkeypatch.setitem(sys.modules,'enrichment.ner',fake_ner)
    ok, _ = prepare_models.ner_ready('balanced')
    assert ok and calls == [('fastino/gliner2.5-small-v1', True)]
