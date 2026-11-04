"""Trusted-backend tenant boundary for PostgreSQL transaction pooling.

The caller MUST authorize a tenant before calling this helper. It must only
run on an autocommit Psycopg connection with no open transaction, otherwise
SET LOCAL might survive a nested SAVEPOINT and leak into an outer transaction.
Never give customers arbitrary SQL access to the application DB credentials.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def validate_tenant(tenant_id: str) -> str:
    """Require an already-authorized, unchanged 1..80-character identifier."""
    if (not isinstance(tenant_id, str) or not tenant_id or len(tenant_id) > 80
            or tenant_id != tenant_id.strip() or "\x00" in tenant_id):
        raise ValueError("Expected an already-authorized tenant ID (1..80 chars)")
    return tenant_id


@contextmanager
def tenant_transaction(conn: Any, authorized_tenant_id: str) -> Iterator[Any]:
    """Make one *top-level* tenant transaction, never a nested savepoint.

    Requires psycopg.connect(..., autocommit=True). The transaction-local
    setting is cleared on COMMIT/ROLLBACK. All tenant SQL must be issued here.
    """
    tenant = validate_tenant(authorized_tenant_id)
    if getattr(conn, "autocommit", None) is not True:
        raise RuntimeError("tenant_transaction requires an autocommit connection")
    status = getattr(getattr(conn, "info", None), "transaction_status", None)
    if getattr(status, "name", None) != "IDLE":
        raise RuntimeError("Cannot enter tenant_transaction inside another transaction")

    with conn.transaction():
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
        yield conn
