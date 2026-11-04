"""Opt-in full SQL/RLS test. Use a disposable DB ending in `_test`.

TEST_DATABASE_URL='postgresql://.../signalmira_test' uv run pytest tests/test_semantic_postgres.py -q -rs
Never point TEST_DATABASE_URL at your real SignalMira database.
"""
from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

from retrieval import semantic
from retrieval.embedders import EmbeddingSpec
from security.tenant_context import tenant_transaction


class StubEmbedder:
    spec = EmbeddingSpec(revision="a" * 40)

    def encode_documents(self, texts):
        result = []
        for text in texts:
            vector = np.zeros(1024, dtype="float32")
            vector[0 if "battery" in text else 1] = 1.0
            result.append(vector)
        return result

    def encode_query(self, query):
        vector = np.zeros(1024, dtype="float32")
        vector[0 if "battery" in query else 1] = 1.0
        return vector


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Set a disposable PostgreSQL *_test database URL")
def test_real_pgvector_sync_search_rls_and_fk():
    from psycopg import sql as pgsql
    from storage.postgres import connect, create_tables
    from indexing.chunk_store import initialize as init_chunks

    with connect(os.environ["TEST_DATABASE_URL"]) as conn:
        if not conn.info.dbname.endswith("_test"):
            pytest.fail("TEST_DATABASE_URL database name must end in _test")
        create_tables(conn)
        init_chunks(conn)
        semantic.initialize(conn)
        role = "sm_sem_probe_" + uuid4().hex[:12]
        a = "tenant-sem-" + uuid4().hex[:8]
        b = "tenant-sem-" + uuid4().hex[:8]
        cid_a, cid_b = uuid4(), uuid4()
        test_items = ((a, cid_a, "battery fails"), (b, cid_b, "battery fails"))
        try:
            for tenant, chunk_id, text in test_items:
                with tenant_transaction(conn, tenant):
                    conn.execute("""INSERT INTO public.feedback_reviews
                      (tenant_id, source, review_id, text, content_hash)
                      VALUES (%s, 'semantic-test', 'one', %s, 'test-input-hash')""",
                      (tenant, text))
                    conn.execute("""INSERT INTO public.rag_chunks
                      (chunk_id, tenant_id, source, review_id, chunk_index,
                       chunk_text, start_index, end_index, input_hash,
                       review_content_hash, chunker_version)
                      VALUES (%s, %s, 'semantic-test', 'one', 0,
                       %s, 0, %s, 'test-input-hash', 'test-input-hash', 'test-1024')""",
                      (chunk_id, tenant, text, len(text)))
                out = semantic.sync(conn, tenant, StubEmbedder(), batch_size=1, page_size=1)
                assert out["written"] == 1

            # Confirm the search source is tenant-scoped before testing restricted role.
            found_a = semantic.search(conn, a, "battery", StubEmbedder(), k=10)
            found_b = semantic.search(conn, b, "battery", StubEmbedder(), k=10)
            assert {r["chunk_id"] for r in found_a} == {cid_a}
            assert {r["chunk_id"] for r in found_b} == {cid_b}

            conn.execute(pgsql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(
                pgsql.Identifier(role)))
            conn.execute(pgsql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(pgsql.Identifier(role)))
            conn.execute(pgsql.SQL("GRANT SELECT ON public.rag_chunks TO {}").format(
                pgsql.Identifier(role)))
            conn.execute(pgsql.SQL("GRANT SELECT, INSERT ON public.rag_embeddings TO {}").format(
                pgsql.Identifier(role)))

            # RLS is active even with an unrestricted SELECT by an unprivileged role.
            for tenant, allowed in ((a, cid_a), (b, cid_b)):
                with tenant_transaction(conn, tenant):
                    conn.execute(pgsql.SQL("SET LOCAL ROLE {}").format(pgsql.Identifier(role)))
                    active = conn.execute(
                        "SELECT row_security_active('public.rag_embeddings'::regclass) AS active"
                    ).fetchone()["active"]
                    assert active
                    items = conn.execute("SELECT chunk_id FROM public.rag_embeddings").fetchall()
                    assert {r["chunk_id"] for r in items} == {allowed}

            with conn.transaction():
                conn.execute(pgsql.SQL("SET LOCAL ROLE {}").format(pgsql.Identifier(role)))
                assert conn.execute("SELECT COUNT(*) AS n FROM public.rag_embeddings").fetchone()["n"] == 0

            # A cannot insert an embedding pretending to belong to B.
            import psycopg
            with tenant_transaction(conn, a):
                conn.execute(pgsql.SQL("SET LOCAL ROLE {}").format(pgsql.Identifier(role)))
                with pytest.raises(psycopg.Error):
                    with conn.transaction():  # savepoint to contain expected SQL error
                        conn.execute("""INSERT INTO public.rag_embeddings
                          (tenant_id,chunk_id,model_key,model_name,model_revision,
                           backend,config_hash,chunk_input_hash,chunker_version,embedding)
                          VALUES (%s,%s,%s,'test','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                           'test','test','test-input-hash','test-1024',%s::vector(1024))""",
                           (b, cid_b, 'test-attack', '[1' + ',0'*1023 + ']'))

            # A cannot attach its embedding to a chunk that belongs to B.
            # The composite (tenant_id,chunk_id) FK must reject the mismatch.
            with tenant_transaction(conn, a):
                with pytest.raises(psycopg.errors.ForeignKeyViolation):
                    with conn.transaction():
                        conn.execute("""INSERT INTO public.rag_embeddings
                          (tenant_id,chunk_id,model_key,model_name,model_revision,
                           backend,config_hash,chunk_input_hash,chunker_version,embedding)
                          VALUES (%s,%s,%s,'test','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                           'test','test','test-input-hash','test-1024',%s::vector(1024))""",
                           (a, cid_b, 'test-attack', '[1' + ',0'*1023 + ']'))

        finally:
            for tenant, _, _ in test_items:
                # Cleanup admin-owned fixtures; deletions cascade to chunks/vectors.
                with tenant_transaction(conn, tenant):
                    conn.execute("DELETE FROM public.feedback_reviews WHERE tenant_id=%s "
                                 "AND source='semantic-test' AND review_id='one'", (tenant,))
            conn.execute(pgsql.SQL("DROP ROLE IF EXISTS {}").format(pgsql.Identifier(role)))
