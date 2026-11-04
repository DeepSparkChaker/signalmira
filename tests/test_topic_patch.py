"""Check fixed model-selection policy; the old patcher is no longer needed."""
from enrichment.topic_policy import Policy
from enrichment.topics import Settings
from enrichment.topic_auto import discovery_profile
import inspect
from enrichment import topic_policy


def test_default_multilingual_for_new_taxonomies():
    assert Policy().profile=='multilingual_m3'
    assert Settings().profile=='multilingual_m3'
    assert discovery_profile(None,'demo')=='multilingual_m3'


def test_language_snapshot_still_has_fresh_join():
    src=inspect.getsource(topic_policy.create_source_snapshot)
    assert 'LEFT JOIN feedback_language l' in src
    assert 'l.input_hash=f.content_hash' in src


def test_explicit_english_candidate_is_allowed():
    assert discovery_profile(None,'demo','accurate')=='accurate'
