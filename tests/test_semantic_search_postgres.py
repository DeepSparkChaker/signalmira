"""Optional real SQL correctness test. Requires a disposable PostgreSQL *_test DB.

TEST_DATABASE_URL=postgresql://.../signalmira_test \\
  uv run python -m pytest tests/test_semantic_search_postgres.py -q -rs

This uses a temporary unique model_key and removes all inserted review rows in
finally. Never point TEST_DATABASE_URL at a customer's database.
"""
from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

from retrieval import semantic
from security.tenant_context import tenant_transaction


@pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'), reason='Needs a disposable PostgreSQL *_test database')
def test_exact_filters_before_top_k_and_rejects_stale_vectors():
    from storage.postgres import connect, create_tables
    from indexing.chunk_store import initialize as init_chunks
    from enrichment.language import initialize as init_language
    from pgvector.psycopg import register_vector

    with connect(os.environ['TEST_DATABASE_URL']) as conn:
        if not conn.info.dbname.endswith('_test'):
            pytest.fail('Refuse to modify database unless its name ends in _test')
        create_tables(conn)
        init_chunks(conn)
        init_language(conn)
        semantic.initialize(conn)
        register_vector(conn)

        tag = uuid4().hex[:12]
        a, b = 'smfix-a-' + tag, 'smfix-b-' + tag
        source_x, source_y = 'smfix-x-' + tag, 'smfix-y-' + tag
        model_key = uuid4().hex + uuid4().hex
        vectors = {
            'a_x': np.array([0.8, 0.6] + [0.0] * 1022, dtype=np.float32),
            'a_y': np.array([1.0, 0.0] + [0.0] * 1022, dtype=np.float32),
            'b_x': np.array([1.0, 0.0] + [0.0] * 1022, dtype=np.float32),
        }
        items = [
            (a, source_x, 'product-x', 'a_x'),
            (a, source_y, 'product-y', 'a_y'),
            (b, source_x, 'product-x', 'b_x'),
        ]
        chunk_ids = {}
        try:
            for tenant, source, product_id, name in items:
                chunk_ids[name] = uuid4()
                text = 'Feedback for ' + name
                content_hash = 'testhash-' + name
                with tenant_transaction(conn, tenant):
                    conn.execute('''
                        INSERT INTO public.feedback_reviews
                        (tenant_id,source,review_id,text,product_id,content_hash)
                        VALUES (%s,%s,%s,%s,%s,%s)
                    ''', (tenant, source, name, text, product_id, content_hash))
                    conn.execute('''
                        INSERT INTO public.rag_chunks
                        (chunk_id,tenant_id,source,review_id,chunk_index,chunk_text,
                         start_index,end_index,input_hash,review_content_hash,chunker_version,product_id)
                        VALUES (%s,%s,%s,%s,0,%s,0,%s,%s,%s,'test-v1',%s)
                    ''', (chunk_ids[name], tenant, source, name, text, len(text),
                          content_hash, content_hash, product_id))
                    conn.execute('''
                        INSERT INTO public.rag_embeddings
                        (tenant_id,chunk_id,model_key,model_name,model_revision,
                         backend,config_hash,chunk_input_hash,chunker_version,embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'test-v1',%s)
                    ''', (tenant, chunk_ids[name], model_key, 'test-embedder', 'a'*40,
                          'test', 'test-policy', content_hash, vectors[name]))

            query = vectors['a_y']
            unfiltered = semantic.search_vector(conn, a, query, model_key=model_key, k=1)
            assert [r['chunk_id'] for r in unfiltered] == [chunk_ids['a_y']]
            filtered = semantic.search_vector(conn, a, query, model_key=model_key,
                                              k=1, product_id='product-x', source=source_x)
            assert [r['chunk_id'] for r in filtered] == [chunk_ids['a_x']]
            # A's search cannot see B's closer vector, regardless of filters.
            assert chunk_ids['b_x'] not in [r['chunk_id'] for r in filtered]

            with tenant_transaction(conn, a):
                conn.execute('''UPDATE public.rag_chunks SET input_hash='changed-after-inference'
                                WHERE tenant_id=%s AND chunk_id=%s''', (a, chunk_ids['a_x']))
            assert semantic.search_vector(conn, a, query, model_key=model_key,
                                          k=1, product_id='product-x') == []
        finally:
            for tenant in (a, b):
                with tenant_transaction(conn, tenant):
                    conn.execute('''DELETE FROM public.feedback_reviews
                                    WHERE tenant_id=%s AND source IN (%s,%s)''',
                                 (tenant, source_x, source_y))
