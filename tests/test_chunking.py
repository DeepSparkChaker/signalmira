"""Offline tests for exact offsets, streaming, stability, and multilingual text."""

from types import SimpleNamespace
from uuid import UUID

import pytest

from indexing.chunking import ChunkSettings, chunk_review, chunk_reviews


def review(text, tenant_id="demo", source="csv", review_id="r-001"):
    # These are exactly the fields provided by ingestion.schemas.FeedbackRecord.
    return SimpleNamespace(text=text, tenant_id=tenant_id, source=source, review_id=review_id)


@pytest.mark.parametrize("text", ["Small review", " é 😀\n" + "a" * 500, "a" * 512])
def test_512_or_less_is_unchanged(text):
    chunks = list(chunk_review(review(text)))
    assert len(chunks) == 1
    assert (chunks[0].text, chunks[0].start_index, chunks[0].end_index) == (text, 0, len(text))


def test_513_splits_and_overlaps():
    chunks = list(chunk_review(review("x" * 513)))
    assert len(chunks) == 2
    assert chunks[0].end_index == 512
    assert chunks[1].start_index == 448
    assert chunks[1].end_index == 513


@pytest.mark.parametrize("text", [
    "This is a sentence. Another sentence! " * 100,
    "repeat\nrepeat\nrepeat\n" * 100,
    "电池耗电很快。" * 200,
    "البطارية تنفد بسرعة، وهذا مزعج. " * 60,
    "A" * 8000,
    "line1\n\nline2\n" * 100,
])
def test_long_text_has_exact_offsets_and_full_coverage(text):
    chunks = list(chunk_review(review(text)))
    assert len(chunks) >= 2
    assert chunks[0].start_index == 0 and chunks[-1].end_index == len(text)
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.text == text[chunk.start_index:chunk.end_index]
        assert 0 < len(chunk.text) <= 512
        assert chunk.start_index < chunk.end_index
        if idx:
            assert chunks[idx - 1].start_index < chunk.start_index <= chunks[idx - 1].end_index
            assert chunk.start_index < chunks[idx - 1].end_index  # Positive overlap


def test_ids_stable_distinct_and_versioned():
    r = review("Test. " * 200)
    first = list(chunk_review(r))
    again = list(chunk_review(r))
    assert first == again
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(UUID(c.chunk_id).version == 5 for c in first)
    assert first[0].chunk_id != list(chunk_review(review(r.text, tenant_id="other")))[0].chunk_id
    changed_policy = list(chunk_review(r, settings=ChunkSettings(overlap_chars=96)))
    assert first[0].chunker_version != changed_policy[0].chunker_version
    assert first[0].chunk_id != changed_policy[0].chunk_id


def test_edit_changes_input_hash_for_reindex():
    r = review("A" * 800)
    original = list(chunk_review(r))
    edited = list(chunk_review(review("B" * 800)))
    assert original[0].input_hash != edited[0].input_hash
    # Stable source-position IDs allow an UPDATE; sync uses input_hash to re-embed.
    assert original[0].chunk_id == edited[0].chunk_id


def test_postgres_dict_row_supported():
    r = {"tenant_id": "demo", "source": "csv", "review_id": "x", "text": "Very good!"}
    c = list(chunk_review(r))[0]
    assert c.text == r["text"] and c.review_id == "x"


def test_streaming_only_reads_on_demand():
    count = 0

    def incoming():
        nonlocal count
        for i in range(10000):
            count += 1
            yield review("ok", review_id=str(i))

    chunks = chunk_reviews(incoming())
    assert count == 0
    assert next(chunks).review_id == "0"
    assert count == 1
    assert next(chunks).review_id == "1"
    assert count == 2


@pytest.mark.parametrize("settings", [
    ChunkSettings(max_chars=4, overlap_chars=2),
    ChunkSettings(max_chars=32, overlap_chars=31),
    ChunkSettings(max_chars=512, overlap_chars=0),
])
def test_custom_settings_terminate(settings):
    text = "hello world! " * 100
    chunks = list(chunk_review(review(text), settings=settings))
    assert chunks[-1].end_index == len(text)
    assert max(map(lambda c: len(c.text), chunks)) <= settings.max_chars
    assert all(c.text == text[c.start_index:c.end_index] for c in chunks)


@pytest.mark.parametrize("settings", [
    {"max_chars": 1}, {"overlap_chars": 512}, {"overlap_chars": -1},
    {"min_cut_fraction": 0}, {"min_cut_fraction": 1.5},
])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        ChunkSettings(**settings)


@pytest.mark.parametrize("record", [
    review(""), review("  \n"), review("good", tenant_id=""),
    review("good", source=" "), review("good", review_id=""),
])
def test_invalid_review_rejected(record):
    with pytest.raises(ValueError):
        list(chunk_review(record))
