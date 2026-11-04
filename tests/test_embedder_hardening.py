"""Production hardening checks; no actual model weights, network or GPU required."""
from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from retrieval import embedders as e

SHA = "b" * 40


def test_index_key_separate_from_query_policy(monkeypatch):
    a = e.EmbeddingSpec(revision=SHA)
    before_index, before_query = a.model_key, a.query_key
    monkeypatch.setattr(e, "QUERY_POLICY", e.QUERY_POLICY + "-new")
    assert a.model_key == before_index
    assert a.query_key != before_query


def test_validation_shape_nan_zero_inf_and_batch_normalization():
    m = np.zeros((2, 1024), dtype=np.float64)
    m[:, 4] = [3, 5]
    result = e.validate_matrix(m, 2)
    assert result.shape == (2, 1024) and result.dtype == np.float32
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    for bad in (np.ones((2, 9)), np.full((2, 1024), np.nan), np.zeros((2, 1024))):
        with pytest.raises(e.EmbeddingValidationError):
            e.validate_matrix(bad, 2)


def test_reject_unpinned_revision_and_unsafe_url():
    with pytest.raises(ValueError):
        e.LocalQwenEmbedder(revision="main")
    with pytest.raises(ValueError):
        e.VllmQwenEmbedder(revision=SHA, base_url="http://localhost:8000.evil.com/v1")
    with pytest.raises(ValueError):
        e.VllmQwenEmbedder(revision=SHA, base_url="http://127.0.0.1.evil.com/v1")
    assert e.VllmQwenEmbedder(revision=SHA, base_url="http://localhost:8000/v1").spec.revision == SHA


def test_production_requires_pin(monkeypatch):
    monkeypatch.setenv("SIGNALMIRA_REQUIRE_PINNED", "1")
    with pytest.raises(ValueError, match="revision"):
        e.LocalQwenEmbedder(revision=None)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        assert not kwargs["truncation"] and not kwargs["padding"]
        return {"input_ids": [list(range(len(t))) for t in texts]}


class FakeModel:
    loads = 0
    active = 0
    peak = 0
    max_forward = 0
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        with self.lock:
            type(self).loads += 1
        time.sleep(.005)
        self.tokenizer = FakeTokenizer()

    def eval(self): return self
    def get_sentence_embedding_dimension(self): return 1024

    def encode(self, texts, *, batch_size, **kwargs):
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            type(self).max_forward = max(type(self).max_forward, batch_size)
        assert kwargs["prompt"] == ""  # prevent implicit model default instruction
        time.sleep(.005)
        v = np.zeros((len(texts), 1024), dtype=np.float32)
        v[:, 1] = 9
        with self.lock:
            type(self).active -= 1
        return v


def test_concurrent_model_load_only_once_and_forward_is_bounded(monkeypatch):
    FakeModel.loads = FakeModel.active = FakeModel.peak = FakeModel.max_forward = 0
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))
    # No network: pin a pretend resolved snapshot and fake local model loader.
    obj = e.LocalQwenEmbedder(revision=SHA, device="cpu", batch_size=3)
    obj._spec = e.EmbeddingSpec(revision=SHA)
    obj._path = "/dev/null/fake_model"
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(obj.encode_documents, ["hello world"] * 7) for _ in range(6)]
        results = [f.result() for f in futures]
    assert FakeModel.loads == 1
    assert FakeModel.peak == 1
    assert FakeModel.max_forward <= 3
    assert all(r.shape == (7, 1024) for r in results)


def test_local_token_limit_rejects_oversize(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))
    obj = e.LocalQwenEmbedder(revision=SHA, device="cpu")
    obj._spec = e.EmbeddingSpec(revision=SHA)
    obj._path = "/dev/null/fake_model"
    with pytest.raises(e.EmbeddingValidationError, match="2048"):
        obj.encode_documents(["a" * 2050])


def test_remote_tokenizer_bound_batch_and_indices(monkeypatch):
    obj = e.VllmQwenEmbedder(revision=SHA, batch_size=2)
    monkeypatch.setattr(obj, "_load_tokenizer", lambda: FakeTokenizer())
    calls = []
    class FakeAPI:
        def create(self, *, model, input):
            calls.append(tuple(input))
            data = []
            for i in reversed(range(len(input))):
                a = [0.0] * 1024
                a[i] = 3.0
                data.append(types.SimpleNamespace(index=i, embedding=a))
            return types.SimpleNamespace(data=data)
    fake_client = types.SimpleNamespace(embeddings=FakeAPI())
    monkeypatch.setattr(obj, "_client_instance", lambda: fake_client)
    rows = obj.encode_documents(["hi", "one", "two", "three", "four"])
    assert rows.shape == (5, 1024)
    assert list(map(len, calls)) == [2, 2, 1]
    assert rows[0, 0] == pytest.approx(1.0) and rows[1, 1] == pytest.approx(1.0)
    with pytest.raises(e.EmbeddingValidationError):
        obj.encode_documents(["x" * 2100])


def test_async_to_thread_safe(monkeypatch):
    import asyncio
    obj = e.LocalQwenEmbedder(revision=SHA)
    monkeypatch.setattr(obj, "encode_query", lambda s: np.ones(1024, dtype=np.float32))
    answer = asyncio.run(obj.encode_query_async("fast?"))
    assert answer.shape == (1024,)
