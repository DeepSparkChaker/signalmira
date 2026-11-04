"""Offline tests do not download weights or require PostgreSQL/pgvector."""
from __future__ import annotations

import json
from contextlib import contextmanager
from uuid import UUID

import numpy as np
import pytest

from retrieval import semantic as db
from retrieval.embedders import (
    EmbeddingSpec, LocalQwenEmbedder, VllmQwenEmbedder,
    checked_vector, query_text,
)

SHA = "a" * 40

@pytest.fixture(autouse=True)
def fake_pgvector_for_mock_connections(monkeypatch):
    """FakeConnections cannot be registered with the real psycopg adapter."""
    import sys
    import types
    package = types.ModuleType("pgvector")
    package.__path__ = []
    module = types.ModuleType("pgvector.psycopg")
    module.register_vector = lambda conn: None
    monkeypatch.setitem(sys.modules, "pgvector", package)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", module)



def test_contract_is_deterministic_and_versioned():
    a = EmbeddingSpec(revision=SHA)
    assert a.model_key == EmbeddingSpec(revision=SHA).model_key
    assert len(a.model_key) == 64
    assert a.model_key != EmbeddingSpec(revision="b" * 40).model_key
    assert a.model_key != EmbeddingSpec(revision=SHA, backend="vllm").model_key
    original_query_key = a.query_key
    assert len(original_query_key) == 64
    from retrieval import embedders
    old = embedders.QUERY_POLICY
    embedders.QUERY_POLICY = old + ":new-task"
    try:
        assert EmbeddingSpec(revision=SHA).model_key == a.model_key
        assert EmbeddingSpec(revision=SHA).query_key != original_query_key
    finally:
        embedders.QUERY_POLICY = old
    assert a.dimensions == 1024


@pytest.mark.parametrize("revision", ["main", "", "z"*40, "a"*39])
def test_unpinned_or_bad_version_rejected(revision):
    with pytest.raises(ValueError):
        EmbeddingSpec(revision=revision)


def test_prompt_only_for_query():
    q = query_text("El pago falla.")
    assert q.startswith("Instruct: ")
    assert q.endswith("\nQuery: El pago falla.")
    assert "customer feedback" in q  # task instruction stays in English


def test_vector_validation_normalizes_and_catches_invalid():
    a = np.zeros(1024, dtype="float32")
    a[0] = 3.0
    result = checked_vector(a)
    assert result.shape == (1024,) and result.dtype == np.float32
    assert result[0] == pytest.approx(1)
    for bad in (np.zeros(1024), [0.4] * 2, [float("nan")] * 1024):
        with pytest.raises(ValueError):
            checked_vector(bad)


def test_models_lazy_without_network():
    local = LocalQwenEmbedder(revision=SHA, offline=True)
    assert local._model is None
    server = VllmQwenEmbedder(revision=SHA)
    assert server._client is None
    with pytest.raises(ValueError):
        VllmQwenEmbedder(revision=SHA, base_url="http://external.example/v1")


def test_schema_has_tenant_fk_cascade_rls_and_index_not_auto():
    sql = db.MIGRATION.read_text()
    assert "FOREIGN KEY (tenant_id, chunk_id)" in sql
    assert "REFERENCES public.rag_chunks (tenant_id, chunk_id) ON DELETE CASCADE" in sql
    assert "FORCE ROW LEVEL SECURITY" in sql
    assert "current_setting('app.tenant_id', true)" in sql
    assert "CREATE INDEX IF NOT EXISTS rag_embeddings_tenant_model_idx" in sql
    assert "CREATE INDEX CONCURRENTLY" not in sql


@contextmanager
def _tenant_context(conn, tenant):
    assert tenant not in ("", None)
    conn.scopes.append(tenant)
    try:
        yield conn
    finally:
        conn.scopes.pop()


class DummyCursor:
    def __init__(self, rows=()):
        self.rows = rows
    def fetchall(self):
        return list(self.rows)
    def fetchone(self):
        return next(iter(self.rows)) if self.rows else None


class FakeCopy:
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def write_row(self, row):
        assert self.conn.scopes
        self.conn.stage.append(row)

class FakeWriteCursor:
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def copy(self, sql):
        assert "pg_temp.rag_embeddings_stage" in sql
        self.conn.stage = []
        return FakeCopy(self.conn)
    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

class FakeConn:
    def cursor(self):
        return FakeWriteCursor(self)
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.stage = []
        self.scopes = []
        self.queries = []
        self.saved = {}
        self.autocommit = True
        self.hnsw = False
    def execute(self, sql, params=()):
        self.queries.append((sql, params, tuple(self.scopes)))
        if "FROM public.rag_chunks AS c" in sql:
            model_key, tenant, after, limit = params
            assert self.scopes == [tenant]
            found = []
            for row in sorted(self.chunks, key=lambda x: x["chunk_id"]):
                identity = (tenant, row["chunk_id"], model_key)
                current = self.saved.get(identity)
                if (row["tenant"] == tenant and row["chunk_id"] > after and
                    (current is None or current != row["input_hash"])):
                    found.append(row)
            return DummyCursor([dict(r) for r in found[:limit]])
        if "INSERT INTO public.rag_embeddings AS existing" in sql:
            model_key, _, _, _, _, tenant = params
            assert self.scopes == [tenant]
            result = []
            for chunk_id, input_hash, chunker_version, vector in self.stage:
                key = (tenant, chunk_id, model_key)
                chunk = next((r for r in self.chunks if r["chunk_id"] == chunk_id
                              and r["tenant"] == tenant
                              and r["input_hash"] == input_hash
                              and r["chunker_version"] == chunker_version), None)
                if chunk and self.saved.get(key) != input_hash:
                    self.saved[key] = input_hash
                    result.append({"chunk_id": str(chunk_id)})
                assert isinstance(vector, np.ndarray) and vector.shape == (1024,)
            return DummyCursor(result)
        if "to_regclass('public.rag_embeddings_hnsw_cosine_idx')" in sql:
            return DummyCursor([{"idx": "public.rag_embeddings_hnsw_cosine_idx" if self.hnsw else None}])
        if "FROM public.rag_embeddings e" in sql:
            assert self.scopes == [params[1]]
            assert "WHERE e.tenant_id=%s AND e.model_key=%s" in sql
            assert "ORDER BY e.embedding <=> %s::vector(1024) LIMIT %s" in sql
            return DummyCursor([])
        if "SELECT extversion FROM pg_extension" in sql:
            return DummyCursor([{"extversion": "0.8.1"}])
        return DummyCursor()


class FakeEmbedder:
    spec = EmbeddingSpec(revision=SHA)
    def __init__(self):
        self.calls = []
    def encode_documents(self, texts):
        self.calls.append(tuple(texts))
        vec = np.zeros(1024, dtype="float32")
        vec[0] = 1.0
        return [vec for _ in texts]
    def encode_query(self, text):
        vec = np.zeros(1024)
        vec[1] = 1.0
        return vec


def _chunk(n, tenant):
    return dict(chunk_id=UUID(int=n), chunk_text=f"Review {n}",
                source="example", review_id=f"r{n}", chunk_index=0,
                input_hash=f"content-{n}", chunker_version="char1024",
                tenant=tenant)


def test_incremental_tenant_scoped_paged_sync(monkeypatch):
    monkeypatch.setattr(db, "tenant_transaction", _tenant_context)
    conn = FakeConn([_chunk(1, "tenant-a"), _chunk(2, "tenant-b"),
                     _chunk(3, "tenant-a")])
    embedder = FakeEmbedder()
    first = db.sync(conn, "tenant-a", embedder, page_size=1, batch_size=1)
    assert first["scanned"] == 2 and first["written"] == 2
    assert len(embedder.calls) == 2
    assert all(tenant == "tenant-a" for tenant, _, _ in conn.saved)
    second = db.sync(conn, "tenant-a", embedder, page_size=2, batch_size=2)
    assert second["scanned"] == 0 and second["written"] == 0
    assert len(embedder.calls) == 2
    assert conn.scopes == []
    assert any("rag_embeddings_stage" in sql for sql, _, _ in conn.queries)


def test_changed_chunks_reembed_only_affected(monkeypatch):
    monkeypatch.setattr(db, "tenant_transaction", _tenant_context)
    chunks = [_chunk(1, "a"), _chunk(2, "a")]
    conn = FakeConn(chunks)
    embedder = FakeEmbedder()
    db.sync(conn, "a", embedder)
    chunks[0]["input_hash"] = "new-version"
    again = db.sync(conn, "a", embedder)
    assert again["scanned"] == 1 and again["written"] == 1


def test_stale_chunk_not_written_if_changed_mid_inference(monkeypatch):
    monkeypatch.setattr(db, "tenant_transaction", _tenant_context)
    chunks = [_chunk(1, "a")]
    conn = FakeConn(chunks)
    class Race(FakeEmbedder):
        def encode_documents(self, texts):
            chunks[0]["input_hash"] = "new-content"
            return super().encode_documents(texts)
    result = db.sync(conn, "a", Race(), page_size=1, batch_size=1)
    assert result["written"] == 0
    assert result["stale_or_concurrent"] == 1


def test_search_scoped_and_modes(monkeypatch):
    monkeypatch.setattr(db, "tenant_transaction", _tenant_context)
    conn = FakeConn()
    emb = FakeEmbedder()
    assert db.search(conn, "a", "payment failed", emb, source="one", product_id="p", k=3) == []
    sql, params, scope = next((sql, pars, scope) for sql, pars, scope in conn.queries
                              if "FROM public.rag_embeddings e" in sql)
    assert scope == ("a",) and params[1:3] == ["a", emb.spec.model_key]
    assert params[3:5] == ["one", "p"]
    assert "AND c.source=%s" in sql and "AND c.product_id=%s" in sql
    assert any("enable_indexscan=off" in s for s, _, _ in conn.queries)
    with pytest.raises(RuntimeError, match="HNSW index is missing"):
        db.search(conn, "a", "payment failed", emb, mode="hnsw")
    conn.hnsw = True
    assert db.search(conn, "a", "payment failed", emb, mode="hnsw") == []
    assert any("hnsw.iterative_scan" in s for s, _, _ in conn.queries)
    assert conn.scopes == []


def test_reject_bad_inputs_before_sql(monkeypatch):
    monkeypatch.setattr(db, "tenant_transaction", _tenant_context)
    conn, emb = FakeConn(), FakeEmbedder()
    with pytest.raises(ValueError):
        db.sync(conn, "a", emb, batch_size=16, page_size=8)
    with pytest.raises(ValueError):
        db.search_vector(conn, "a", np.ones(1024), model_key="bad")
    with pytest.raises(ValueError):
        db.search_vector(conn, "a", np.ones(1024), model_key=emb.spec.model_key, k=0)


def test_hnsw_ddl_is_explicit_and_concurrent():
    conn = FakeConn()
    db.create_hnsw(conn)
    assert any("CREATE INDEX CONCURRENTLY" in sql for sql, _, _ in conn.queries)
