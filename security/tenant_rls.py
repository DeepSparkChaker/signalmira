"""Audit, stage, and test SignalMira PostgreSQL row-level security.

    python -m security.tenant_rls audit
    python -m security.tenant_rls smoke
    python -m security.tenant_rls apply --confirm ENABLE_RLS
    python -m security.tenant_rls verify --tenant demo --other second-tenant

`apply` changes existing table access. Run ONLY after ALL ingestion, NER, NED,
topic, chunk and API queries use security.tenant_context.tenant_transaction
with separate unprivileged login roles. Migration account must own all tables.
"""
from __future__ import annotations

import argparse
import os
import secrets

from .tenant_context import validate_tenant

READER = "sm_tenant_reader"
WORKER = "sm_tenant_worker"
PRED = "tenant_id = NULLIF(current_setting('app.tenant_id', true), '')"

# Discover *every* tenant_id-owning ordinary/partitioned table, including
# sm_topic_* and future RAG/vector tables. No hard-coded incomplete table list.
TABLES_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name, c.oid AS oid,
       c.relrowsecurity AS rls_enabled, c.relforcerowsecurity AS force_rls,
       pg_get_userbyid(c.relowner) AS owner,
       EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid=c.oid
               AND p.polname='sm_tenant_guard') AS guard_present,
       EXISTS (SELECT 1 FROM pg_policy p WHERE p.polrelid=c.oid
               AND p.polname='sm_tenant_access') AS access_present
FROM pg_class c
JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relkind IN ('r','p')
  AND EXISTS (
     SELECT 1 FROM pg_attribute a
     WHERE a.attrelid=c.oid AND a.attname='tenant_id'
       AND a.attnum>0 AND NOT a.attisdropped
  )
ORDER BY c.relname
"""


def _sql():
    from psycopg import sql
    return sql


def _role(conn, name: str):
    return conn.execute(
        "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname=%s",
        (name,),
    ).fetchone()


def _check_role(conn, name: str):
    item = _role(conn, name)
    if not item or item["rolsuper"] or item["rolbypassrls"] or item["rolcanlogin"]:
        raise RuntimeError(f"{name}: missing or privileged; cannot verify RLS safely")
    return item


def inspect(conn) -> list[dict]:
    return [dict(row) for row in conn.execute(TABLES_SQL).fetchall()]


def audit(conn) -> bool:
    items = inspect(conn)
    if not items:
        print("FAIL: No public tables with tenant_id. Initialize SignalMira first.")
        return False
    print("TABLE                         RLS  FORCE  GUARD  ACCESS")
    for row in items:
        print(f"{row['table_name'][:28]:28s}  "
              f"{str(row['rls_enabled']):5s}"
              f"{str(row['force_rls']):7s}"
              f"{str(row['guard_present']):7s}"
              f"{str(row['access_present']):7s}")
    good = all(r["rls_enabled"] and r["force_rls"]
               and r["guard_present"] and r["access_present"] for r in items)
    print(f"AUDIT: {'PASS' if good else 'NOT READY'}; {len(items)} tenant tables")
    print("NOTE: Covers public tenant_id tables only. Review views, functions, grants,"
          " backups, shared caches, and any private-data tables lacking tenant_id.")
    return good


def apply(conn) -> int:
    """Atomic DDL; restrictive guard prevents older permissive policies widening scope."""
    sql = _sql()
    tables = inspect(conn)
    if not tables or not any(r["table_name"] == "feedback_reviews" for r in tables):
        raise RuntimeError("feedback_reviews not found; initialize original storage first")
    with conn.transaction():
        for name in (READER, WORKER):
            if _role(conn, name) is None:
                conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(
                    sql.Identifier(name)))
            _check_role(conn, name)
        for row in tables:
            ref = sql.Identifier(row["schema_name"], row["table_name"])
            conn.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(ref))
            conn.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(ref))
            # Existing permissive policies (e.g. rag_chunks_tenant_policy) cannot
            # widen access, because every authorized row must pass this AND guard.
            conn.execute(sql.SQL("DROP POLICY IF EXISTS sm_tenant_guard ON {}").format(ref))
            conn.execute(sql.SQL(
                "CREATE POLICY sm_tenant_guard ON {} AS RESTRICTIVE FOR ALL TO PUBLIC "
                "USING (" + PRED + ") WITH CHECK (" + PRED + ")").format(ref))
            conn.execute(sql.SQL("DROP POLICY IF EXISTS sm_tenant_access ON {}").format(ref))
            conn.execute(sql.SQL(
                "CREATE POLICY sm_tenant_access ON {} AS PERMISSIVE FOR ALL "
                "TO sm_tenant_reader, sm_tenant_worker "
                "USING (" + PRED + ") WITH CHECK (" + PRED + ")").format(ref))
            conn.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM PUBLIC").format(ref))
            conn.execute(sql.SQL("GRANT SELECT ON TABLE {} TO sm_tenant_reader").format(ref))
            conn.execute(sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {} TO sm_tenant_worker"
            ).format(ref))
    return len(tables)


class _RollbackProbe(Exception):
    pass


def smoke(conn) -> None:
    """Two-tenant RLS test using disposable roles/table, rolled back on success.

    Requires an admin account allowed to CREATE ROLE and SET ROLE. Unlike
    `verify`, this does not change existing SignalMira tables or persist data.
    """
    sql = _sql()
    suffix = secrets.token_hex(5)
    reader, worker, table = (f"sm_pr_{suffix}", f"sm_pw_{suffix}", f"sm_probe_{suffix}")
    ident = sql.Identifier("public", table)
    try:
        with conn.transaction():
            conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(
                sql.Identifier(reader)))
            conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(
                sql.Identifier(worker)))
            conn.execute(sql.SQL("CREATE TABLE {} (tenant_id TEXT NOT NULL, secret TEXT NOT NULL)").format(ident))
            # Seed before enabling FORCE RLS, so no bypass is used during testing.
            conn.execute(sql.SQL("INSERT INTO {} VALUES ('tenant-A', 'A-only'), ('tenant-B', 'B-only')").format(ident))
            conn.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(ident))
            conn.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(ident))
            conn.execute(sql.SQL("CREATE POLICY probe_guard ON {} AS RESTRICTIVE FOR ALL "
                                 "USING (" + PRED + ") WITH CHECK (" + PRED + ")").format(ident))
            conn.execute(sql.SQL("CREATE POLICY probe_access ON {} FOR ALL TO {}, {} "
                                 "USING (" + PRED + ") WITH CHECK (" + PRED + ")").format(
                                     ident, sql.Identifier(reader), sql.Identifier(worker)))
            conn.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(ident, sql.Identifier(reader)))
            conn.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO {}").format(
                ident, sql.Identifier(worker)))
            conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(reader)))
            assert conn.execute(sql.SQL("SELECT row_security_active({}::regclass) AS active").format(
                sql.Literal(f"public.{table}"))).fetchone()["active"]
            # No tenant in transaction => cannot see ANY tenant's data.
            assert conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(ident)).fetchone()["n"] == 0
            # The fixture owns an outer ROLLBACK-only transaction to keep the
            # temporary roles/table. Test the policy here, not the top-level
            # tenant_transaction helper (which intentionally rejects nesting).
            for tenant, expected in (("tenant-A", "A-only"), ("tenant-B", "B-only")):
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
                result = conn.execute(sql.SQL("SELECT secret FROM {}").format(ident)).fetchall()
                assert [r["secret"] for r in result] == [expected], result
                conn.execute("SELECT set_config('app.tenant_id', '', true)")
                assert conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(
                    ident)).fetchone()["n"] == 0
            conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(worker)))
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", ("tenant-A",))
            assert conn.execute(sql.SQL("DELETE FROM {} WHERE tenant_id='tenant-B'").format(
                ident)).rowcount == 0
            # Deliberately catch the RLS error inside a nested SQL savepoint;
            # no tenant_transaction is opened within the outer transaction.
            from psycopg import Error
            try:
                with conn.transaction():
                    conn.execute(sql.SQL("INSERT INTO {} VALUES ('tenant-B','bad')").format(ident))
            except Error:
                pass
            else:
                raise AssertionError("WITH CHECK failed to block cross-tenant insert")
            conn.execute("SELECT set_config('app.tenant_id', '', true)")
            assert conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(
                ident)).fetchone()["n"] == 0
            raise _RollbackProbe
    except _RollbackProbe:
        pass
    print("SMOKE PASS: missing context denied, A/B isolated, forbidden insert/delete blocked;")
    print("             probe roles/table/data all rolled back")


def verify(conn, tenant: str, other: str) -> None:
    """Check active RLS and tenant-only visibility on all currently existing tables."""
    tenant, other = validate_tenant(tenant), validate_tenant(other)
    if tenant == other:
        raise ValueError("Choose two different tenant IDs")
    if not audit(conn):
        raise RuntimeError("RLS schema audit failed. Do not launch customer access")
    _check_role(conn, READER)
    sql = _sql()
    tables = inspect(conn)
    with conn.transaction():
        conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(READER)))
        for row in tables:
            ref = sql.Identifier(row["schema_name"], row["table_name"])
            literal = sql.Literal(f"{row['schema_name']}.{row['table_name']}")
            if not conn.execute(sql.SQL("SELECT row_security_active({}::regclass) AS active")
                                .format(literal)).fetchone()["active"]:
                raise AssertionError(f"RLS bypassed on {row['table_name']}")
            for t in (tenant, other):
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", (t,))
                outside = conn.execute(sql.SQL(
                    "SELECT count(*) AS n FROM {} WHERE tenant_id <> %s"
                ).format(ref), (t,)).fetchone()["n"]
                if outside:
                    raise AssertionError(f"Cross-tenant rows in {row['table_name']}")
                conn.execute("SELECT set_config('app.tenant_id', '', true)")
        print(f"VERIFY PASS: {len(tables)} tenant tables active under reader role;"
              " no cross-tenant rows returned")
    print("NOTE: This verifies table policies under SET ROLE, not login authentication,"
          " arbitrary SQL exposure, views, or every background worker path.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("smoke")
    migration = sub.add_parser("apply")
    migration.add_argument("--confirm", required=True, help="Must equal ENABLE_RLS")
    check = sub.add_parser("verify")
    check.add_argument("--tenant", required=True)
    check.add_argument("--other", required=True)
    args = parser.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("DATABASE_URL is not set")
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(dsn.replace("postgresql+psycopg://", "postgresql://", 1),
                         autocommit=True, row_factory=dict_row) as conn:
        if args.cmd == "audit":
            if not audit(conn):
                raise SystemExit(2)
        elif args.cmd == "smoke":
            smoke(conn)
        elif args.cmd == "apply":
            if args.confirm != "ENABLE_RLS":
                parser.error("Deliberate opt-in required: --confirm ENABLE_RLS")
            print("WARNING: existing workers need tenant_transaction and unprivileged roles.")
            print(f"RLS enabled for {apply(conn)} tenant-owned tables")
        else:
            verify(conn, args.tenant, args.other)


if __name__ == "__main__":
    main()
