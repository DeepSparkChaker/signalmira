"""Regression tests for filtered exact/HNSW SQL without a live database."""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import pytest

from retrieval import semantic as s
from retrieval.embedders import EmbeddingSpec

KEY = EmbeddingSpec(revision='a' * 40).model_key


def test_exact_candidate_filters_are_before_top_k_and_no_ann_index():
    sql = s._search_sql('exact', source='uci-amazon', product_id='p1')
    eligible = sql.index('AND EXISTS')
    source = sql.index('eligible.source=%s')
    product = sql.index('eligible.product_id=%s')
    limit = sql.index('LIMIT %s')
    hydration = sql.index('FROM nearest\nJOIN public.rag_chunks')
    assert eligible < source < product < limit < hydration
    assert 'ORDER BY (e.embedding <=> %s::vector(1024)) + 0' in sql
    assert 'eligible.input_hash=e.chunk_input_hash' in sql
    assert 'eligible.chunker_version=e.chunker_version' in sql
    assert 'c.tenant_id=%s AND c.chunk_id=nearest.chunk_id' in sql
    assert 'l.tenant_id=c.tenant_id' in sql
    assert '1.0 - nearest.distance' in sql


def test_hnsw_direct_distance_operator_and_filters_not_after_limit():
    sql = s._search_sql('hnsw', source='partner', product_id=None)
    assert 'ORDER BY e.embedding <=> %s::vector(1024)' in sql
    assert 'ORDER BY (e.embedding <=>' not in sql
    assert sql.index('eligible.source=%s') < sql.index('LIMIT %s')
    assert 'eligible.product_id=%s' not in sql
    assert sql.count('%s') == 7  # 2 vector + tenant/model + source + limit + hydrate tenant


class FakeDB:
    def __init__(self):
        self.queries = []
        self.in_scope = None
        self.index_present = True

    def execute(self, sql, params=()):
        self.queries.append((sql, params, self.in_scope))
        if "to_regclass('public.rag_embeddings_hnsw_cosine_idx')" in sql:
            return Result([{'idx': 'public.rag_embeddings_hnsw_cosine_idx' if self.index_present else None}])
        return Result([])


class Result:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return self.rows
    def fetchone(self): return self.rows[0]


@contextmanager
def fake_tenant(conn, tenant):
    assert conn.in_scope is None
    conn.in_scope = tenant
    try:
        yield conn
    finally:
        conn.in_scope = None


def test_native_vector_adapter_and_hnsw_settings_scoped(monkeypatch):
    # The original test module provides its own pgvector mock. Here install a
    # lightweight substitute: real pgvector is tested separately against a DB.
    import sys, types
    pgvector = types.ModuleType('pgvector')
    pgvector.__path__ = []
    adapter = types.ModuleType('pgvector.psycopg')
    adapter.register_vector = lambda conn: None
    monkeypatch.setitem(sys.modules, 'pgvector', pgvector)
    monkeypatch.setitem(sys.modules, 'pgvector.psycopg', adapter)
    monkeypatch.setattr(s, 'tenant_transaction', fake_tenant)
    db = FakeDB()
    v = np.zeros(1024, dtype=np.float32)
    v[12] = 3
    s.search_vector(db, 'tenant-a', v, model_key=KEY, source='partner', product_id='sku', k=3)
    queries = [(sql, pars, scope) for sql, pars, scope in db.queries if 'WITH nearest AS MATERIALIZED' in sql]
    assert len(queries) == 1
    sql, params, scope = queries[0]
    assert scope == 'tenant-a'
    assert isinstance(params[0], np.ndarray) and params[0].dtype == np.float32
    assert isinstance(params[-3], np.ndarray)
    assert np.linalg.norm(params[0]) == pytest.approx(1)
    assert params[1:5] == ['tenant-a', KEY, 'partner', 'sku']
    assert params[-2:] == [3, 'tenant-a']
    assert not any('enable_indexscan' in sql for sql, _, _ in db.queries)
    assert db.in_scope is None

    s.search_vector(db, 'tenant-b', v, model_key=KEY, mode='hnsw', k=5)
    assert any('hnsw.iterative_scan=strict_order' in sql for sql, _, _ in db.queries)
    assert db.in_scope is None


def test_benchmark_uses_exact_once_and_hnsw_repeats(monkeypatch):
    calls = []
    def search_vector(conn, tenant, vector, **kwargs):
        calls.append(kwargs['mode'])
        ids = [1, 2, 3] if kwargs['mode'] == 'exact' else [2, 3, 4]
        return [{'chunk_id': item} for item in ids]
    monkeypatch.setattr(s, 'search_vector', search_vector)
    class FakeEmb:
        spec = EmbeddingSpec(revision='a' * 40)
        def encode_query(self, query):
            v = np.zeros(1024, dtype=np.float32)
            v[0] = 1
            return v
    result = s.benchmark(object(), 'tenant-a', 'battery', FakeEmb(), repeats=3)
    assert calls == ['exact', 'hnsw', 'hnsw', 'hnsw']
    assert result['hnsw']['agreement_with_exact_at_k'] == pytest.approx(2 / 3, abs=1e-4)
    assert result['exact']['count'] == 3
