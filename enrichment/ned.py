"""SignalMira entity linking (NED): tenant-scoped catalog + pg_trgm fuzzy aliases.

Usage: python enrichment/ned.py --init
       python enrichment/ned.py --tenant demo --add --id pixel9 --name 'Pixel 9' \
           --label product --alias 'Google Pixel 9'
       python enrichment/ned.py --tenant demo --mention 'Pixel 9' --label product
       python enrichment/ned.py --tenant demo --sync

Do not confuse name similarity with entity certainty: ambiguous matches stay unresolved.
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from itertools import batched

DDL = (
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    """CREATE TABLE IF NOT EXISTS entity_catalog (
        tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL,
        label TEXT NOT NULL, canonical_name TEXT NOT NULL,
        PRIMARY KEY (tenant_id, entity_id)
    )""",
    """CREATE TABLE IF NOT EXISTS entity_aliases (
        tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL,
        alias TEXT NOT NULL, alias_key TEXT NOT NULL,
        PRIMARY KEY (tenant_id,entity_id,alias_key),
        FOREIGN KEY (tenant_id,entity_id) REFERENCES entity_catalog ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS entity_aliases_tenant_idx ON entity_aliases (tenant_id, alias_key)",
    "CREATE INDEX IF NOT EXISTS entity_aliases_trgm_idx ON entity_aliases USING gin (alias_key gin_trgm_ops)",
    """CREATE TABLE IF NOT EXISTS entity_links (
        tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
        mention_index INT NOT NULL,
        entity_id TEXT, canonical_name TEXT,
        method TEXT NOT NULL, score DOUBLE PRECISION,
        linked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id, source, review_id, mention_index),
        FOREIGN KEY (tenant_id,source,review_id,mention_index)
          REFERENCES feedback_entities ON DELETE CASCADE,
        FOREIGN KEY (tenant_id,entity_id) REFERENCES entity_catalog ON DELETE CASCADE
    )""",
    "CREATE INDEX IF NOT EXISTS entity_links_entity_idx ON entity_links (tenant_id,entity_id)",
)


def normalize(text: str) -> str:
    """Normalize spelling for lookup without modifying the original mention."""
    if not isinstance(text, str):
        raise TypeError("Expected a string")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def connect():
    import psycopg
    from psycopg.rows import dict_row
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def initialize(conn):
    """Create entity catalog / alias / linking tables (use migrations in production)."""
    with conn.transaction():
        for statement in DDL:
            conn.execute(statement)


def add_entity(conn, tenant: str, entity_id: str, label: str, name: str,
               aliases: tuple[str, ...] = ()): 
    """Register canonical entity plus admin-curated aliases, scoped to tenant."""
    if not all((tenant, entity_id, label, name)):
        raise ValueError("Provide tenant, entity ID, label and canonical name")
    values = [(tenant, entity_id, a, normalize(a)) for a in (name, *aliases)]
    if any(not v[-1] for v in values):
        raise ValueError("Empty alias")
    with conn.transaction():
        conn.execute("""INSERT INTO entity_catalog (tenant_id,entity_id,label,canonical_name)
            VALUES (%s,%s,%s,%s) ON CONFLICT (tenant_id,entity_id) DO UPDATE
            SET label=EXCLUDED.label, canonical_name=EXCLUDED.canonical_name""",
            (tenant, entity_id, label, name))
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO entity_aliases (tenant_id,entity_id,alias,alias_key)
               VALUES (%s,%s,%s,%s) ON CONFLICT (tenant_id,entity_id,alias_key) DO NOTHING""",
               values)


def resolve(conn, *, tenant: str, mention: str, label: str,
            threshold: float = 0.80, margin: float = 0.08) -> dict | None:
    """Exact first, then fuzzy. Abstain if multiple catalog IDs are plausible."""
    key = normalize(mention)
    if not tenant or not label or not key:
        return None
    if not 0.35 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Threshold must be >=0.35; margin between 0 and 1")
    exact = conn.execute("""SELECT DISTINCT c.entity_id,c.canonical_name
        FROM entity_aliases a JOIN entity_catalog c USING (tenant_id,entity_id)
        WHERE a.tenant_id=%s AND c.label=%s AND a.alias_key=%s LIMIT 2""",
        (tenant, label, key)).fetchall()
    if len(exact) == 1:
        return {**exact[0], "method": "exact", "score": 1.0}
    if exact:  # two IDs share this alias; do not guess
        return None
    candidates = conn.execute("""SELECT c.entity_id,c.canonical_name,
            MAX(similarity(a.alias_key,%s)) AS score
        FROM entity_aliases a JOIN entity_catalog c USING (tenant_id,entity_id)
        WHERE a.tenant_id=%s AND c.label=%s AND a.alias_key %% %s
        GROUP BY c.entity_id,c.canonical_name ORDER BY score DESC LIMIT 2""",
        (key, tenant, label, key)).fetchall()
    if not candidates:
        return None
    winner = candidates[0]
    if (winner["score"] < threshold or
        (len(candidates) > 1 and winner["score"] - candidates[1]["score"] < margin)):
        return None
    return {"entity_id": winner["entity_id"], "canonical_name": winner["canonical_name"],
            "method": "fuzzy", "score": float(winner["score"])}


def sync(conn, tenant: str, *, source: str | None = None, force: bool = False,
         page_size: int = 200, threshold: float = 0.80, limit: int = 0) -> tuple[int, int]:
    """Link NER mentions incrementally; unresolved links are recorded too."""
    if not tenant or not 1 <= page_size <= 2000 or limit < 0:
        raise ValueError("Provide tenant; page_size 1..2000")
    acquired = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s),hashtext(%s)) AS yes",
                            ("signalmira_ned", tenant)).fetchone()["yes"]
    if not acquired:
        raise RuntimeError("Entity linking already running for this tenant")
    n = unresolved = 0
    after = ("", "", -1)
    cache = {}
    try:
        while True:
            query = """SELECT e.source,e.review_id,e.mention_index,e.text,e.label
                FROM feedback_entities e WHERE e.tenant_id=%s
                AND (e.source,e.review_id,e.mention_index) > (%s,%s,%s)"""
            params: list = [tenant, *after]
            if source is not None:
                query += " AND e.source=%s"
                params.append(source)
            if not force:
                query += """ AND NOT EXISTS (SELECT 1 FROM entity_links l
                    WHERE l.tenant_id=e.tenant_id AND l.source=e.source
                    AND l.review_id=e.review_id AND l.mention_index=e.mention_index)"""
            query += " ORDER BY e.source,e.review_id,e.mention_index LIMIT %s"
            params.append(page_size)
            rows = conn.execute(query, params).fetchall()
            if not rows:
                return n, unresolved
            last = rows[-1]
            after = (last["source"], last["review_id"], last["mention_index"])
            for group in batched(rows, 100):
                if limit:
                    group = group[:max(0, limit - n)]
                    if not group:
                        return n, unresolved
                links = []
                for row in group:
                    lookup = (row["label"], normalize(row["text"]))
                    if lookup not in cache:
                        cache[lookup] = resolve(conn, tenant=tenant, mention=row["text"],
                                                label=row["label"], threshold=threshold)
                        if len(cache) > 20000:
                            cache.clear()
                    result = cache[lookup]
                    links.append((tenant, row["source"], row["review_id"], row["mention_index"],
                                  result["entity_id"] if result else None,
                                  result["canonical_name"] if result else None,
                                  result["method"] if result else "unresolved",
                                  result["score"] if result else None))
                    unresolved += result is None
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany("""INSERT INTO entity_links
                            (tenant_id,source,review_id,mention_index,entity_id,canonical_name,method,score)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (tenant_id,source,review_id,mention_index) DO UPDATE SET
                                entity_id=EXCLUDED.entity_id,
                                canonical_name=EXCLUDED.canonical_name,
                                method=EXCLUDED.method,score=EXCLUDED.score,linked_at=now()""",
                            links)
                n += len(group)
                if limit and n >= limit:
                    return n, unresolved
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s),hashtext(%s))",
                     ("signalmira_ned", tenant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--tenant")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--add", action="store_true")
    modes.add_argument("--mention")
    modes.add_argument("--sync", action="store_true")
    parser.add_argument("--id", help="Canonical catalog entity ID")
    parser.add_argument("--name", help="Canonical display name")
    parser.add_argument("--alias", action="append", default=[], help="Repeat for more aliases")
    parser.add_argument("--label", help="Entity type, e.g. product")
    parser.add_argument("--source", help="Optional review source filter")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--force", action="store_true", help="Re-link after catalog changes")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.init and not args.tenant:
        parser.error("--tenant is required")
    with connect() as conn:
        if args.init:
            initialize(conn)
            print("Catalog and link tables ready")
        elif args.add:
            add_entity(conn, args.tenant, args.id, args.label, args.name, tuple(args.alias))
            print("Entity registered")
        elif args.mention:
            import json
            print(json.dumps(resolve(conn, tenant=args.tenant, mention=args.mention,
                    label=args.label or "product", threshold=args.threshold), indent=2))
        elif args.sync:
            print("(linked, unresolved):", sync(conn, args.tenant, source=args.source,
                  force=args.force, threshold=args.threshold, limit=args.limit))
        else:
            parser.error("Choose --init, --add, --mention or --sync")


if __name__ == "__main__":
    main()
