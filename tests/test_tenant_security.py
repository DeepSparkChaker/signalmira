"""Offline tests; optional PostgreSQL smoke uses a disposable ROLLBACK-only fixture."""
from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from security.tenant_context import tenant_transaction, validate_tenant
from security import tenant_rls as rls


class Conn:
    def __init__(self, rows=None, autocommit=True):
        from types import SimpleNamespace
        self.rows = rows or []
        self.queries = []
        self.open_transactions = 0
        self.autocommit = autocommit
        self.info = SimpleNamespace(transaction_status=SimpleNamespace(name="IDLE"))

    @contextmanager
    def transaction(self):
        self.open_transactions += 1
        self.info.transaction_status.name = "INTRANS"
        try:
            yield self
        finally:
            self.open_transactions -= 1
            self.info.transaction_status.name = (
                "IDLE" if self.open_transactions == 0 else "INTRANS"
            )

    def execute(self, sql, params=()):
        self.queries.append((str(sql), params, self.open_transactions))
        return self

    def fetchall(self):
        return self.rows


@pytest.mark.parametrize('bad', ['', ' ', ' demo', 'demo ', 'x' * 81, 'a\x00b', None, 7])
def test_bad_tenant_rejected(bad):
    with pytest.raises(ValueError):
        validate_tenant(bad)


def test_tenant_id_kept_exactly():
    assert validate_tenant('Demo-客户') == 'Demo-客户'


def test_scope_sets_local_guc_only_inside_transaction():
    conn = Conn()
    with tenant_transaction(conn, 't-A'):
        assert conn.open_transactions == 1
        assert conn.queries == [(
            "SELECT set_config('app.tenant_id', %s, true)", ('t-A',), 1
        )]
        conn.execute('SELECT 1')
    assert conn.open_transactions == 0


def test_scope_closes_after_error():
    conn = Conn()
    with pytest.raises(RuntimeError):
        with tenant_transaction(conn, 't-A'):
            raise RuntimeError('boom')
    assert conn.open_transactions == 0


def test_nested_tenant_context_is_rejected_without_switching_tenant():
    conn = Conn()
    with tenant_transaction(conn, "tenant-A"):
        before = list(conn.queries)
        with pytest.raises(RuntimeError, match="inside another transaction"):
            with tenant_transaction(conn, "tenant-B"):
                pass
        assert conn.queries == before


def test_already_open_transaction_is_rejected():
    conn = Conn()
    with conn.transaction():
        with pytest.raises(RuntimeError, match="inside another transaction"):
            with tenant_transaction(conn, "tenant-A"):
                pass
    assert not conn.queries


def test_non_autocommit_connection_is_rejected():
    conn = Conn(autocommit=False)
    with pytest.raises(RuntimeError, match="autocommit"):
        with tenant_transaction(conn, "tenant-A"):
            pass
    assert not conn.queries


def test_tenant_context_can_switch_only_after_previous_transaction_exits():
    conn = Conn()
    with tenant_transaction(conn, "tenant-A"):
        assert conn.info.transaction_status.name == "INTRANS"
    with tenant_transaction(conn, "tenant-B"):
        assert conn.info.transaction_status.name == "INTRANS"
    assert conn.info.transaction_status.name == "IDLE"
    assert [item[1] for item in conn.queries] == [("tenant-A",), ("tenant-B",)]


def test_invalid_tenant_does_not_touch_database():
    conn = Conn()
    with pytest.raises(ValueError):
        with tenant_transaction(conn, ' '):
            pass
    assert not conn.queries


def test_rls_policy_must_fail_closed_without_setting():
    assert 'NULLIF' in rls.PRED
    assert "current_setting('app.tenant_id', true)" in rls.PRED


def test_audit_flags_incomplete_table(capsys):
    conn = Conn([dict(table_name='feedback_reviews', rls_enabled=False,
                      force_rls=False, guard_present=False, access_present=False)])
    assert not rls.audit(conn)
    assert 'NOT READY' in capsys.readouterr().out


def test_audit_accepts_complete_table(capsys):
    conn = Conn([dict(table_name='feedback_reviews', rls_enabled=True,
                      force_rls=True, guard_present=True, access_present=True)])
    assert rls.audit(conn)
    assert 'PASS' in capsys.readouterr().out


def test_missing_tables_are_not_mistaken_for_security():
    assert not rls.audit(Conn([]))


def test_role_with_bypass_is_rejected():
    with patch.object(rls, '_role', return_value=dict(
        rolname='sm_tenant_reader', rolsuper=False, rolbypassrls=True, rolcanlogin=False)):
        with pytest.raises(RuntimeError):
            rls._check_role(object(), 'sm_tenant_reader')


def test_runtime_group_cannot_be_login_role():
    with patch.object(rls, '_role', return_value=dict(
        rolname='sm_tenant_reader', rolsuper=False, rolbypassrls=False, rolcanlogin=True)):
        with pytest.raises(RuntimeError):
            rls._check_role(object(), 'sm_tenant_reader')


@pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'), reason='needs local PostgreSQL admin test DB')
def test_real_postgresql_two_tenants():
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(os.environ['TEST_DATABASE_URL'], autocommit=True, row_factory=dict_row) as conn:
        rls.smoke(conn)  # all fixture DDL/data rolled back

@pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'), reason='needs local PostgreSQL test DB')
def test_real_postgresql_tenant_scope_does_not_leak():
    """Check the helper, separately from the rollback-only RLS smoke fixture."""
    import psycopg
    with psycopg.connect(os.environ['TEST_DATABASE_URL'], autocommit=True) as conn:
        with tenant_transaction(conn, 'tenant-A'):
            assert conn.execute(
                "SELECT current_setting('app.tenant_id', true)"
            ).fetchone()[0] == 'tenant-A'
            with pytest.raises(RuntimeError, match='inside another transaction'):
                with tenant_transaction(conn, 'tenant-B'):
                    pass
            assert conn.execute(
                "SELECT current_setting('app.tenant_id', true)"
            ).fetchone()[0] == 'tenant-A'
        assert conn.execute(
            "SELECT nullif(current_setting('app.tenant_id', true), '')"
        ).fetchone()[0] is None
        with tenant_transaction(conn, 'tenant-B'):
            assert conn.execute(
                "SELECT current_setting('app.tenant_id', true)"
            ).fetchone()[0] == 'tenant-B'
        assert conn.execute(
            "SELECT nullif(current_setting('app.tenant_id', true), '')"
        ).fetchone()[0] is None
