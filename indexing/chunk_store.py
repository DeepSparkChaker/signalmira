"""Sync deterministic review chunks into PostgreSQL (no embedding model needed).

Run: python -m indexing.chunk_store init
     python -m indexing.chunk_store sync --tenant demo
     python -m indexing.chunk_store get --tenant demo --source uci-amazon --review-id uci-amazon-0001
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import UUID

from indexing.chunking import ChunkSettings, chunk_review
from storage.postgres import connect

MIGRATION = Path(__file__).resolve().parents[1] / "migrations/002_rag_chunks.sql"

# Read one bounded page, locking only source rows while replacing changed chunks.
PAGE = """
SELECT f.tenant_id, f.source, f.review_id, f.text, f.content_hash,
       f.company, f.product_id, f.review_date,
       f.attributes->>'language' AS language,
       c.review_content_hash AS indexed_hash, c.chunker_version AS indexed_version
FROM feedback_reviews AS f
LEFT JOIN rag_chunks AS c
  ON c.tenant_id = f.tenant_id AND c.source = f.source
 AND c.review_id = f.review_id AND c.chunk_index = 0
WHERE f.tenant_id = %s AND (f.source, f.review_id) > (%s, %s)
ORDER BY f.source, f.review_id
LIMIT %s
FOR SHARE OF f
"""

INSERT = """
INSERT INTO rag_chunks (
    chunk_id, tenant_id, source, review_id, chunk_index, chunk_text,
    start_index, end_index, input_hash, review_content_hash,
    chunker_version, company, product_id, review_date, language
)
SELECT chunk_id, tenant_id, source, review_id, chunk_index, chunk_text,
       start_index, end_index, input_hash, review_content_hash,
       chunker_version, company, product_id, review_date, language
FROM rag_chunks_stage
"""


def initialize(conn) -> None:
    """Apply the additive migration. Use a managed migration tool before SaaS rollout."""
    with conn.transaction():
        conn.execute(MIGRATION.read_text(encoding="utf-8"))


def sync(conn, tenant: str, *, batch_size: int = 100,
         settings: ChunkSettings | None = None) -> dict[str, int]:
    """Incremental, bounded-memory sync; each page is atomic and retry-safe."""
    if not tenant or not 1 <= batch_size <= 1000:
        raise ValueError("Provide a tenant and batch_size in 1..1000")
    settings = settings or ChunkSettings(max_chars=512, overlap_chars=64)
    locked = conn.execute(
        "SELECT pg_try_advisory_lock(hashtext(%s), hashtext(%s)) AS locked",
        ("sm_chunk_sync", tenant),
    ).fetchone()["locked"]
    if not locked:
        raise RuntimeError("Another chunk sync is running for this tenant")

    stats = {"scanned_reviews": 0, "updated_reviews": 0,
             "unchanged_reviews": 0, "written_chunks": 0}
    last_source = last_id = ""
    try:
        # Session-local staging permits fast COPY without bypassing target-table RLS.
        conn.execute("""CREATE TEMP TABLE IF NOT EXISTS rag_chunks_stage
                        (LIKE rag_chunks INCLUDING DEFAULTS) ON COMMIT DELETE ROWS""")
        while True:
            with conn.transaction():
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
                rows = conn.execute(PAGE, (tenant, last_source, last_id, batch_size)).fetchall()
                if not rows:
                    break
                stats["scanned_reviews"] += len(rows)
                changed = [r for r in rows if r["indexed_hash"] != r["content_hash"]
                           or r["indexed_version"] != settings.version]
                stats["unchanged_reviews"] += len(rows) - len(changed)

                if changed:
                    # Delete the old current set and insert its replacement atomically.
                    conn.execute("""
                        DELETE FROM rag_chunks AS c
                        USING unnest(%s::text[], %s::text[]) AS k(source, review_id)
                        WHERE c.tenant_id = %s AND c.source = k.source
                          AND c.review_id = k.review_id
                    """, ([r["source"] for r in changed],
                          [r["review_id"] for r in changed], tenant))
                    count = 0
                    with conn.cursor() as cur:
                        with cur.copy("""COPY rag_chunks_stage (
                            chunk_id, tenant_id, source, review_id, chunk_index,
                            chunk_text, start_index, end_index, input_hash,
                            review_content_hash, chunker_version, company,
                            product_id, review_date, language
                        ) FROM STDIN""") as copy:
                            for row in changed:
                                for chunk in chunk_review(row, settings=settings):
                                    copy.write_row((
                                        UUID(chunk.chunk_id), chunk.tenant_id,
                                        chunk.source, chunk.review_id, chunk.chunk_index,
                                        chunk.text, chunk.start_index, chunk.end_index,
                                        chunk.input_hash, row["content_hash"],
                                        chunk.chunker_version, row["company"],
                                        row["product_id"], row["review_date"],
                                        row["language"],
                                    ))
                                    count += 1
                        cur.execute(INSERT)
                    stats["updated_reviews"] += len(changed)
                    stats["written_chunks"] += count
                last_source, last_id = rows[-1]["source"], rows[-1]["review_id"]
        return stats
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s), hashtext(%s))",
                     ("sm_chunk_sync", tenant))


def get_chunks(conn, tenant: str, source: str, review_id: str) -> list[dict]:
    """Read only one tenant's chunks, in original text order."""
    with conn.transaction():
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
        return list(conn.execute("""
            SELECT chunk_id, chunk_index, chunk_text, start_index, end_index,
                   input_hash, review_content_hash, chunker_version, company,
                   product_id, review_date, language
            FROM rag_chunks
            WHERE tenant_id = %s AND source = %s AND review_id = %s
            ORDER BY chunk_index
        """, (tenant, source, review_id)).fetchall())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    s = sub.add_parser("sync")
    s.add_argument("--tenant", required=True, help="Trusted tenant from job context")
    s.add_argument("--batch-size", type=int, default=100)
    s.add_argument("--max-chars", type=int, default=1024)
    s.add_argument("--overlap", type=int, default=64)
    g = sub.add_parser("get")
    g.add_argument("--tenant", required=True)
    g.add_argument("--source", required=True)
    g.add_argument("--review-id", required=True)
    args = parser.parse_args()
    with connect(os.environ["DATABASE_URL"]) as conn:
        if args.command == "init":
            initialize(conn)
            print("rag_chunks table ready")
        elif args.command == "sync":
            print(json.dumps(sync(conn, args.tenant, batch_size=args.batch_size,
                settings=ChunkSettings(max_chars=args.max_chars,
                                       overlap_chars=args.overlap)), indent=2))
        else:
            print(json.dumps(get_chunks(conn, args.tenant, args.source,
                                        args.review_id), indent=2, default=str))


if __name__ == "__main__":
    main()
